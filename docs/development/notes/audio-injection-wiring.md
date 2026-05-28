# Audio-injection wiring in `LongCatVideoAvatarTransformer3DModel`

**Resolves: v2 Open Question #1** (the #1 port blocker).
**Source-of-truth files:**
- `refs/longcat-video/longcat_video/modules/avatar/longcat_video_dit_avatar.py` (model + block forward)
- `refs/longcat-video/longcat_video/modules/avatar/attention.py` (`Attention` + `SingleStreamAttention`)
- `refs/longcat-video/longcat_video/modules/avatar/blocks.py` (`AudioProjModel`)
- `refs/longcat-video/longcat_video/modules/blocks.py` (base `FeedForwardSwiGLU`, `RMSNorm_FP32`, `LayerNorm_FP32`, `PatchEmbed3D`, `TimestepEmbedder`, `CaptionEmbedder`, `FinalLayer_FP32`, `modulate_fp32`)

## TL;DR

Audio is wired as a **dedicated cross-attention layer per block**, residual-added like text cross-attention but additionally gated by a per-block, per-timestep AdaLN modulation (`audio_adaLN_modulation` → 3 params: shift/scale/gate). The audio K/V comes from a small standalone MLP (`AudioProjModel`) that runs **once** at the top of the DiT forward and outputs 32 context tokens per video latent at dim 768. No FiLM, no per-token gating, no audio-driven AdaLN of the visual path. The wrapper is `mx.fast.scaled_dot_product_attention`-compatible (dense — `flash_attn_func` / `flash_attn_bsa_3d` / `xformers` are interchangeable inside the `_process_attn` wrapper).

## Per-block forward pass

From `LongCatAvatarSingleStreamBlock.forward` ([dit_avatar.py:98-191](refs/longcat-video/longcat_video/modules/avatar/longcat_video_dit_avatar.py)):

```text
inputs: x [B,N,C], y_text [1, N_valid, C], audio_hidden_states [B*T, 32, 768], t [B,T,C_t]

# 1. AdaLN params (fp32)
shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
    = adaLN_modulation(t).chunk(6)                                    # SiLU + Linear(512, 6*4096)
audio_shift_mca, audio_scale_mca, audio_gate_mca
    = audio_adaLN_modulation(t[:, num_cond_latents:]).chunk(3)        # SiLU + Linear(512, 3*4096)

# 2. Self-attn (with Reference Skip in self.attn)
x_m = modulate_fp32(mod_norm_attn, x, shift_msa, scale_msa)            # LN no-affine + (1+scale)*x + shift
x_s, x_ref_attn_map = self.attn(x_m, shape, num_cond_latents, ...,
                                ref_img_index, mask_frame_range, ref_target_masks)
x = x + gate_msa * x_s

# 3. Text cross-attn (NO modulation; just LN-pre + residual)
x = x + self.cross_attn(pre_crs_attn_norm(x), y_text, y_seqlens,
                        num_cond_latents, shape)

# 4. Audio cross-attn (LN-pre on both Q-source and K-source; residual + AdaLN-gated)
audio_out_cond, audio_out_noise = self.audio_cross_attn(
    pre_video_crs_attn_norm(x),                       # visual Q
    pre_audio_crs_attn_norm(audio_hidden_states),     # audio K/V
    shape, num_cond_latents, x_ref_attn_map, human_num)
audio_out_noise = modulate_fp32(mod_norm_attn, audio_out_noise,        # NOTE: reuses mod_norm_attn (no-affine LN)
                                audio_shift_mca, audio_scale_mca)
audio_add = gate_audio * audio_out_noise
if audio_out_cond is not None:
    audio_add = concat([audio_out_cond, audio_add], dim=1)             # cond region is zeros
x = x + audio_add

# 5. FFN (modulated)
x_m = modulate_fp32(mod_norm_ffn, x, shift_mlp, scale_mlp)
x = x + gate_mlp * ffn(x_m)
```

## Audio path (DiT-level, runs once)

From `LongCatVideoAvatarTransformer3DModel.forward` ([dit_avatar.py:420-451](refs/longcat-video/longcat_video/modules/avatar/longcat_video_dit_avatar.py)):

Input: `audio_embs` shape `[B, T_frames, W=5, S=12, C_a=768]` — these are **Whisper-Large-v3 encoder hidden states**, windowed (5 tokens per frame for "first" path, vae_scale-aware for "latter" path) across 12 Whisper layers, dim 768.

```text
first_frame_audio = audio_embs[:, :1]                              # [B, 1, 5, 12, 768]
latter_frame_audio = audio_embs[:, 1:]                             # [B, T-1, 5, 12, 768]

# Realign latter frames to the VAE temporal grouping (vae_scale=4 frames -> 1 latent):
latter = rearrange(latter, "b (n_t n) w s c -> b n_t n w s c", n=4)
# Pull "first/middle/last" slices around the middle audio token (middle=2 for W=5):
latter_first  = latter[:,:,:1,  :middle+1, ...]       # the 5-element window for the boundary frame
latter_middle = latter[:,:,1:-1, middle:middle+1, ...] # single-token for interior frames
latter_last   = latter[:,:,-1:, middle:, ...]
latter_audio  = concat([latter_first, latter_middle, latter_last], dim=2)
                                                        # [B, (T-1)//vae_scale, W'=W-1+vae_scale, 12, 768]

audio_hidden_states = self.audio_proj(first_frame_audio, latter_audio)   # -> [B, T_latent, 32, 768]
```

Then `audio_hidden_states` is reshaped to `(B*T_latent, 32, 768)` so each latent frame attends to its own 32 audio context tokens.

## `AudioProjModel` (3-layer MLP, [blocks.py:8-87](refs/longcat-video/longcat_video/modules/avatar/blocks.py))

```text
input_dim    = audio_window * audio_blocks * audio_channels = 5 * 12 * 768 = 46080
input_dim_vf = (audio_window + vae_scale - 1) * audio_blocks * audio_channels = 8 * 12 * 768 = 73728
intermediate = 512
context_tokens = 32
output_dim = 768

proj1     : Linear(46080  -> 512)   ReLU
proj1_vf  : Linear(73728  -> 512)   ReLU                   # parallel path for vae-scale-aligned frames
concat first + latter along T-axis -> Linear(512 -> 512) ReLU
              -> Linear(512 -> 32 * 768 = 24576)
              -> reshape to (B, T_latent, 32, 768)
              -> LayerNorm(768)
```

## `SingleStreamAttention` = audio cross-attention ([attention.py:281-468](refs/longcat-video/longcat_video/modules/avatar/attention.py))

- `q_linear: Linear(4096, 4096, bias=True)` — visual → Q
- `kv_linear: Linear(768, 4096*2, bias=True)` — audio → K, V
- `q_norm`, `k_norm`: `RMSNorm_FP32(head_dim=128, eps=1e-6)`
- `proj: Linear(4096, 4096)`
- Per-frame layout: visual tokens reshape to `(B*N_t, S, C)`; cross-attn runs per-frame against that frame's 32 audio tokens
- **Multitalk extras** (only active when `ref_target_masks` is provided): 1D-RoPE positions on Q/K derived from `x_ref_attn_map` to spatially route human1 / human2 / background. `class_range=24, class_interval=4` default — human1 gets [0,4], human2 gets [20,24], background gets ~12.
- Returns `(audio_output_cond=None or zeros, audio_output_noise)` — cond region gets zeros, only noise tokens get the cross-attended update.

## Reference Skip Attention (inside `Attention`, not a separate layer)

[`attention.py:118-214`](refs/longcat-video/longcat_video/modules/avatar/attention.py) and `:216-278` (kv-cache variant). The mechanism is **Q-slicing in self-attention**:

- When `mask_frame_range > 0` and `num_cond_latents > 1` (video continuation), compute a temporal window `[ref_img_index - mask_frame_range, ref_img_index + mask_frame_range]`.
- Split Q over noise tokens into `front | maskref | back`.
- `front` and `back` attend to **all** K/V (ref + cond + noise).
- `maskref` attends only to **non-reference** K/V — prevents the reference image from inducing repeating motion in that temporal window.
- Also disables Block-Sparse Attention for that path (`enable_bsa = False`) because the temporal dimension wouldn't be divisible by BSA chunks.

For the MLX port, this is pure slicing + extra calls into `mx.fast.scaled_dot_product_attention` — no new primitives needed.

## What's NOT here (lives at pipeline level)

These need to be read from `pipeline_longcat_video_avatar.py` (~72 KB), not the DiT:

- **Disentangled Unconditional Guidance** combiner (CFG with separate audio_cfg + text_cfg, plus "silence ≠ frozen body" disentanglement)
- **Cross-Chunk Latent Stitching** (skip VAE re-encode between chunks)
- **DMD 8-step sampler** (FlowMatchEuler + distillation LoRA path)

## Numerical conventions to preserve

These appear in every `modulate_fp32` and `*_FP32` class — replicate exactly in MLX:

1. **AdaLN modulation math runs in fp32**, not bf16. `(scale + 1) * norm(x.float()) + shift`, all in fp32, cast back to input dtype at the end.
2. **RMSNorm** keeps the norm computation in fp32 even if `x` is bf16: `norm(x.float()).type_as(x) * weight`.
3. **LayerNorm** same — cast inputs to fp32, run F.layer_norm in fp32, cast back.
4. **Final layer** asserts `t.dtype == torch.float32` — the timestep embedding stays fp32 all the way through.

MLX equivalents:
- For RMSNorm: hand-wrap `mx.fast.rms_norm` with explicit fp32 cast in/out, OR write the four-line fp32 RMSNorm inline. `mx.fast.rms_norm` internally accumulates in fp32 already, but the *weight* multiply may not — verify with a parity test.
- For LayerNorm with elementwise_affine=False (used inside `modulate_fp32`): the simplest port is to write a tiny fp32 LayerNorm by hand. Don't try to coerce `mx.fast.layer_norm` to no-affine; the hand-written version is 4 lines.
- For `modulate_fp32`: hand-port directly. Eight lines.

## Config defaults (the oracle — verify against shipped config.json in S0.4)

From `LongCatVideoAvatarTransformer3DModel.__init__` ([dit_avatar.py:199-232](refs/longcat-video/longcat_video/modules/avatar/longcat_video_dit_avatar.py)):

| Field | Default | Notes |
|---|---|---|
| `in_channels` | 16 | Wan VAE latent channels |
| `out_channels` | 16 | same |
| `hidden_size` | 4096 | |
| `depth` | 48 | block count |
| `num_heads` | 32 | → head_dim = 128 |
| `caption_channels` | 4096 | umT5-XXL hidden |
| `mlp_ratio` | 4 | FFN inner = 16384 (before SwiGLU 2/3 reduction) |
| `adaln_tembed_dim` | 512 | |
| `frequency_embedding_size` | 256 | |
| `patch_size` | (1, 2, 2) | (T, H, W); temporal patchify = 1 (`assert patch_size[0]==1`) |
| `audio_window` | 5 | Whisper window per frame |
| `audio_block` | 12 | Whisper layer count consumed |
| `audio_channel` | 768 | Whisper hidden dim |
| `intermediate_dim` | 512 | AudioProjModel inner |
| `output_dim` | 768 | audio context-token dim |
| `context_tokens` | 32 | per-frame audio tokens to cross-attend |
| `vae_scale` | 4 | VAE temporal compression |
| `audio_prenorm` | False | LayerNorm on audio K-source before cross-attn (Identity if False) |
| `class_range` | 24 | multitalk RoPE class range |
| `class_interval` | 4 | multitalk RoPE per-human interval |
| `text_tokens_zero_pad` | False | text padding handling |

These are constructor defaults. The real shipped values come from `config.json` in the HF repo — to be verified in S0.4. Per skill rule: defaults are not the oracle, the config is.
