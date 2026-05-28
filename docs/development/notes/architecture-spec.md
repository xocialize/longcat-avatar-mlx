# LongCat-Video-Avatar 1.5 — Architecture Spec (port-locked)

**Sources:**
- `refs/longcat-video/longcat_video/modules/avatar/*` — PyTorch source (authoritative for module shapes + forward order)
- `notes/tech-report.txt` — extracted from `assets/LongCat-Video-Avatar-1.5-Tech-Report.pdf` (authoritative for training-time strategy + audio encoder pipeline)
- `notes/audio-injection-wiring.md` — companion file with per-block wiring

This document is the **single oracle** for the MLX port. Where the two sources disagree (e.g. `audio_block` count), the **shipped `config.json`** (S0.4 deliverable) is the tiebreaker. Constructor defaults in the PyTorch source are NOT authoritative — they may reflect v1.0 (Wav2Vec2) values overridden by config for v1.5.

## Generative process — Flow Matching

[Tech report §4, eqs. 3-4]:
```
x_t = (1 - t) · x_0 + t · ε,    t ∈ [0,1],   ε ~ N(0,I)
v_t = x_0 - ε                                          (target velocity)
L   = || v_pred(x_t, c, t; θ) - v_t ||²
```
MLX port uses `mlx_arsenal.diffusion.FlowMatchEulerDiscreteScheduler`. Match `t` schedule to PT reference. The scheduler config from HF (S0.4) overrides arsenal defaults.

## Component overview

```
input image (ref)  --VAE encode-->  ref_latent  (16ch, 4× temporal compressed, 8× spatial)
input audio (.wav) --vocal sep & Whisper-large encode + group-mean-pool--> audio_embs  (T, ?, ?, 1280?)
text prompt        --umT5-XXL encoder--> text_emb (N_token, 4096)

ref_latent + noise_latents -> DiT (48 blocks) -> velocity prediction
                                              ↑ audio_embs, text_emb conditioning

velocity -> Euler step (8 NFE w/ DMD LoRA) -> denoised latent
denoised latent -> Wan VAE decode -> output video (30 fps, default 93 frames ≈ 3.1s)
```

## DiT (`LongCatVideoAvatarTransformer3DModel`)

Constructor defaults (TO BE OVERRIDDEN BY `config.json`):

```python
in_channels=16, out_channels=16          # Wan VAE latent
hidden_size=4096, depth=48, num_heads=32  # head_dim = 128
caption_channels=4096                      # umT5-XXL hidden
mlp_ratio=4                                # FFN inner = hidden * 4 then SwiGLU 2/3-reduce + round to multiple_of=256 → 11008 (NOT 16384, see int8-format.md)
adaln_tembed_dim=512
frequency_embedding_size=256
patch_size=(1, 2, 2)                       # T=1, H=W=2; temporal NOT patchified (assert)
# audio config — values may differ in shipped v1.5 config.json
audio_window=5      # video frames per audio window
audio_block=12      # Whisper layer-group count   ⚠️ tech-report says 5 groups for v1.5
audio_channel=768   # Whisper hidden dim          ⚠️ Whisper-large is 1280
intermediate_dim=512
output_dim=768
context_tokens=32   # audio tokens per video latent that the cross-attn sees
vae_scale=4         # VAE temporal compression
audio_prenorm=False # whether to LN audio K-source before cross-attn
class_range=24, class_interval=4   # MultiTalk L-RoPE positions
```

### Module list (in declaration order; mirror this in MLX)

| Field | Class | Output shape |
|---|---|---|
| `x_embedder` | `PatchEmbed3D(patch_size=(1,2,2), 16, 4096)` | Conv3d; flatten to `(B, N=T·H·W/4, 4096)` |
| `t_embedder` | `TimestepEmbedder(t_embed_dim=512, freq=256)` | sin/cos(256) → Linear(256,512) → SiLU → Linear(512,512), out `(B*T, 512)` |
| `y_embedder` | `CaptionEmbedder(4096, 4096)` | Linear(4096,4096) → GELU(tanh) → Linear(4096,4096) |
| `blocks` | `ModuleList[48 × LongCatAvatarSingleStreamBlock]` | see per-block below |
| `audio_proj` | `AudioProjModel(...)` | runs once; outputs `(B, T_latent, 32, 768)` |
| `final_layer` | `FinalLayer_FP32(4096, 4, 16, 512)` | AdaLN(no-affine LN + 2-param shift/scale) + Linear → unpatchify |

### Per-block forward (`LongCatAvatarSingleStreamBlock`)

(see [notes/audio-injection-wiring.md](audio-injection-wiring.md) for full pseudo-code)

1. **AdaLN params** (fp32): `adaLN_modulation(t)` → 6 params for self-attn + FFN; `audio_adaLN_modulation(t[:, num_cond_latents:])` → 3 params for audio cross-attn output gating.
2. **Self-attn** (with Reference Skip Q-slicing): `x = x + gate_msa * attn(modulate_fp32(LN_noaffine(x), shift_msa, scale_msa))`. Returns `x_ref_attn_map` for multitalk routing.
3. **Text cross-attn** (no AdaLN modulation, just LN-pre): `x = x + cross_attn(LN_affine(x), text)`.
4. **Audio cross-attn** (LN-pre on both sides + AdaLN-gated output): `x = x + gate_audio * modulate_fp32(LN_noaffine(audio_cross_attn(LN_affine(x), LN_or_identity(audio))), audio_shift, audio_scale)`. Cond region of the output is zeros.
5. **FFN** (SwiGLU, AdaLN-modulated): `x = x + gate_mlp * ffn(modulate_fp32(LN_noaffine(x), shift_mlp, scale_mlp))`.

### Normalization conventions (must replicate exactly)

- **`modulate_fp32`** casts to fp32, applies LN (no-affine), computes `(scale + 1) * norm + shift`, casts back. Modulation params are ALWAYS fp32.
- **`RMSNorm_FP32`** (qk_norm, eps=1e-6): `out = (x / sqrt(mean(x²) + eps)).type_as(x) * weight`, but the norm computation runs in fp32 internally regardless of input dtype.
- **`LayerNorm_FP32`** (eps=1e-6): cast inputs → fp32, `F.layer_norm` in fp32, cast back. Both elementwise_affine variants are used.

## Audio path (verified against tech report §3.2)

**Tech-report-level flow (now verified against [pipeline:557-612](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)):**
1. Vocal separation: MelBand RoFormer (cleans BGM out of speech). Sample rate 16 kHz, loudness-normalized to -23 LUFS.
2. **Whisper feature extractor** in chunks of `MEL_CHUNK = 750*640 = 480000` samples: wav → mel spectrogram.
3. **Whisper-large encoder** in `ENC_CHUNK = 3000` mel-frame slices, with `output_hidden_states=True` returning the embedding + 32 transformer layers. Stack across layers: `(1, T_enc, n_layers=33, D=1280)`.
4. **Group-mean-pool**, exactly:
   ```python
   feat0 = mean(layers[0:8])     # 5 groups, indices [0:8], [8:16], [16:24], [24:32], [32]
   feat1 = mean(layers[8:16])
   feat2 = mean(layers[16:24])
   feat3 = mean(layers[24:32])
   feat4 = layers[32]            # singleton (the final layer)
   audio_emb = stack([feat0..feat4], dim=2)   # (1, T_enc, 5, 1280)
   ```
5. **Linear interpolation** in time: 50 Hz (`ENC_FPS`) → `fps` (default 25). Output: `(T_video, 5, 1280)`.
6. **AudioProjModel** (the audio adapter inside the DiT, run once at the top of `dit.forward`) aggregates a temporal context window per frame and downsamples to match VAE's 4× temporal compression. Output: **32 context tokens per video latent at dim 768**.

**v1.0 vs v1.5 parameter mismatch (resolved):** The `AudioProjModel` constructor defaults (`seq_len=5, blocks=12, channels=768`) are **v1.0 (Wav2Vec2) values that the v1.5 `config.json` must override**. For v1.5 with Whisper-large: `audio_block = 5` (groups, not Whisper layer count) and `audio_channel = 1280` (Whisper-large hidden dim). Confirm via S0.4. The 5-D shape `(B, T, W, S, C)` that AudioProjModel.forward expects internally is a separate windowing step done inside the DiT forward (lines 425-435) — that part doesn't change between v1.0 and v1.5.

## Text path

- Text encoder: **umT5-XXL** (bilingual zh/en).
- Output: `(B, 1, N_token, 4096)` (the leading dim of 1 is squeezed inside the DiT).
- `caption_channels = 4096` matches umT5-XXL hidden.
- The DiT packs variable-length text across batch into `(1, sum(seqlens), C)` using `encoder_attention_mask`. MLX port can simplify to padded `(B, max_len, C)` + bool mask for cleaner shapes — verify it doesn't break attention by comparing against PT on a 2-batch input.

## Sampler / CFG (locked from pipeline source)

- **8 NFE** with DMD2 Generator LoRA loaded.
- **Default CFG: `text_guidance_scale=4.0, audio_guidance_scale=4.0`** ([pipeline:649-650](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)).
- Distilled scheduler keeps the same `t` schedule as the teacher (50-step FlowMatchEuler), but samples 8 of those steps.
- **CFG combiner** ([pipeline:838](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)) — exact formula:

  ```python
  # Per step:
  # Pass 1 (batch=2): full-cond [text + audio] AND text-uncond [no_text + audio] (latent doubled)
  #   noise_pred_uncond_text, noise_pred_cond = dit(input × 2, [neg_text, pos_text], audio).chunk(2)
  # Pass 2 (batch=1): both-uncond [no_text + no_audio]
  #   noise_pred_uncond = dit(input, neg_text, zeros_like(audio))
  # Combine:
  noise_pred = noise_pred_uncond
             + text_guidance_scale  * (noise_pred_cond        - noise_pred_uncond_text)
             + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
  ```

  Translation: `text_scale` weights the *text contribution given audio is present*; `audio_scale` weights the *audio contribution against the fully-unconditional baseline*. This **IS** what v2 called "Disentangled Unconditional Guidance" — but it is a combiner formula, not a separate architectural piece. **Total: 3 forward passes per step (2-batch + 1-batch), or 24 forward passes per inference at 8 NFE.** Some optimization possible by skipping branches when one scale = 1.0.

- `noise_pred = -noise_pred` ([pipeline:841](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)) — the model predicts `−v` (i.e. `ε − x_0`), so flip the sign for scheduler compatibility. Don't miss this.

## Long-video generation

Tech report §3.3 mentions **multi-clip rollout** for RLHF training: earlier clips provide temporal context, later clips are generated conditioned on them. At inference time, the DiT supports this directly via:

- `num_cond_latents`, `num_ref_latents`, `ref_img_index`, `mask_frame_range` — control which temporal regions are conditioning vs. noise
- `forward_with_kv_cache` — reuses K,V from prior chunk for cheap continuation
- `kv_cache_dict` — per-block KV cache

There is **no separate "Cross-Chunk Latent Stitching" primitive named in the tech report.** The pipeline file likely chains these inference modes back-to-back. **For the MLX port v1, single-chunk inference is sufficient for the Stage 1 decision gate** (5s @ 480P real-portrait). Multi-chunk continuation can be added once the per-chunk path is parity-locked.

## What's NOT in the model (lives at pipeline level)

1. **CFG combiner** — RESOLVED above; 3-pass with text+audio decomposition. [pipeline:826-844](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py).
2. **Audio encoder + pooling** — RESOLVED above; Whisper-large, 33 layers → 5 group-pooled features, linear-interp 50Hz→25fps. [pipeline:557-612](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py).
3. **Sampling loop** — `FlowMatchEulerDiscreteScheduler.set_timesteps + .step` invoked 8 times when `use_distill=True`. The `get_timesteps_sigmas` method ([pipeline:380](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)) computes the distilled subset of sigmas — to be ported as a small helper.
4. **Chunked inference orchestration** — uses `forward_with_kv_cache` and `_update_kv_cache_dict` / `_cache_clean_latents` / `_get_kv_cache_dict` ([pipeline:400-428](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)). For Stage 1 v1, single-chunk inference is sufficient.
5. **Vocal separator** — MelBand RoFormer entry point in `audio_process/`. Likely deferred to post-port (assume user pre-cleans audio for v1).
6. **Latent normalization** — `normalize_latents` / `denormalize_latents` ([pipeline:456-477](refs/longcat-video/longcat_video/pipeline_longcat_video_avatar.py)) — Wan VAE has per-channel mean/std scaling. Port verbatim from the pipeline constants.

## Things the MLX port can drop / replace

| Reference uses | MLX replacement | Reason |
|---|---|---|
| `flash_attn_func`, `flash_attn_bsa_3d`, `xformers.memory_efficient_attention` | `mx.fast.scaled_dot_product_attention` (dense) | No FlashAttention on Metal; block-sparse deferred per v2 |
| `ulysses_wrapper` (multi-GPU sharding) | drop entirely | Single-device, unified memory |
| `context_parallel_util.split_cp_2d` / `gather_cp_2d` | drop | Same |
| `torch.amp.autocast('cuda', dtype=torch.float32)` | explicit `.astype(mx.float32)` cast at the same boundary | No autocast in MLX |
| `nn.Conv3d` with PT (O,I,T,H,W) weights | `mlx.nn.Conv3d` with MLX (O,T,H,W,I) | transpose in conversion |
| `RMSNorm_FP32` | hand-write or wrap `mx.fast.rms_norm` w/ fp32 cast | See norm conventions above |
| `LayerNorm_FP32` (no-affine variant) | hand-write 4-line fp32 LN | `mx.fast.layer_norm` can't easily be no-affine |
| `flash_attn_interface.flash_attn_func` | same `mx.fast.sdpa` | Inside `_process_attn` wrapper |

## Decision: keep block forward order verbatim

Per skill rule: **no refactoring during port.** Even though some operations look like they could be combined (e.g. the `audio_adaLN_modulation` reuses `mod_norm_attn` which is unintuitive), KEEP THEM. The MLX `LongCatAvatarSingleStreamBlock` should be a near-textual translation of `longcat_video_dit_avatar.py:98-191`. Reorganize ONLY after end-to-end parity locks.
