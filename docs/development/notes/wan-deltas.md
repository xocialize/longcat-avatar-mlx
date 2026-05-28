# LongCat-Video-Avatar 1.5 vs Wan-2.1 architectural deltas

**Sources:**
- `refs/longcat-video/longcat_video/modules/` (PyTorch reference)
- `refs/mlx-video/mlx_video/models/wan_2/` (existing MLX precedent for Wan 2.1)
- `refs/longcat-video/longcat_video/modules/avatar/` (Avatar-specific overlay)

## TL;DR

**Roughly 70 % of the port can reuse mlx-video's `wan_2/` module nearly verbatim.** The novel work is concentrated in the audio path (~30 % of the port surface): AudioProjModel, SingleStreamAttention (audio cross-attn), and the per-block wiring that inserts audio cross-attn between text cross-attn and FFN.

## Reusable from `mlx-video/mlx_video/models/wan_2/` (verified file by file)

| File | Lines | Reuse strategy |
|---|---|---|
| `vae.py` (629) | AutoencoderKLWan | Direct reuse — same `z_dim=16, base_dim=96, dim_mult=[1,2,4,4], temperal_downsample=[F,T,T]` per [vae/config.json](config-snapshot/longcat-video--vae-config.json) |
| `scheduler.py` (447) | FlowMatchEulerDiscreteScheduler | Reuse with `shift=7.0` override for Avatar; `shift=12.0` for base |
| `rope.py` (176) | 3D RoPE | Reuse for visual self-attn; the Avatar's `RotaryPositionalEmbedding` (visual) and `RotaryPositionalEmbedding1D` (audio multitalk) both share head_dim=128 |
| `attention.py` (221) | Attention helpers | Reuse the SDPA wrapper, mask conventions, QKNorm patterns |
| `text_encoder.py` (239) | T5LayerNorm, T5Attention, T5RelativeEmbedding, T5FeedForward, T5SelfAttentionBlock | **VERIFY:** mlx-video uses generic `T5*` names. umT5-XXL is bilingual (vocab 256384 vs T5's 32128) and has **per-layer** relative-position bias vs vanilla T5's shared. Either mlx-video already handles this (umT5 ≈ T5 architecturally aside from those two points) or we extend. Confirm at Stage 1.2. |
| `tiling.py` (338) | VAE tiling for memory-bound decode | Reuse if needed for 720P; bf16 480P should fit without tiling on 128 GB |
| `convert.py` (808) | Weight-conversion recipe template | Pattern only — adapt for LongCat's tensor names |
| `wan_2.py` (388) | Top-level pipeline entry | Pattern only — our pipeline_mlx.py mirrors Meituan's pipeline_longcat_video_avatar.py |
| `i2v_utils.py` (60) | I2V helpers (ref image insertion) | Reuse |
| `utils.py` (191) | shared utilities | Reuse |

**Total reusable lines from mlx-video: ~2700+** (everything except `transformer.py`, `generate.py`, `postprocess.py`, `wan_2.py` which are model-specific).

## What's TRULY novel (must write from scratch for the MLX port)

### From [`longcat_video/modules/avatar/`](../refs/longcat-video/longcat_video/modules/avatar/) — Avatar-1.5 specific
| File | PT lines | Component | Notes |
|---|---|---|---|
| `blocks.py` | 88 | `AudioProjModel` (3-layer MLP, group-pool windowing) | Simple — straightforward port |
| `attention.py` | 275 | `Attention` (visual self-attn w/ Reference Skip Q-slicing) | The Reference Skip Q-splitting logic adds branching but no new primitives |
| `attention.py` | (same) | `SingleStreamAttention` (audio cross-attn + multitalk L-RoPE) | Most complex novel piece |
| `longcat_video_dit_avatar.py` | 539 | `LongCatAvatarSingleStreamBlock`, `LongCatVideoAvatarTransformer3DModel` | The DiT block forward order + audio path orchestration |

### From [`longcat_video/modules/`](../refs/longcat-video/longcat_video/modules/) — shared with base (port once)
| File | PT lines | Component | Notes |
|---|---|---|---|
| `blocks.py` | 227 | PatchEmbed3D, TimestepEmbedder, CaptionEmbedder, FeedForwardSwiGLU, RMSNorm_FP32, LayerNorm_FP32, FinalLayer_FP32, modulate_fp32 | All standard, ~one-to-one translations |
| `lora_utils.py` | ? | LoRA loader / multi-LoRA forward patching | For DMD LoRA; mlx-video has Wan Lightning LoRA precedent — adapt that |

### Pipeline-level novel work
- Audio encoder: Whisper-large-v3 + group-mean-pool over 33 hidden states (port encoder forward; mlx-audio has the Whisper encoder code we can lift)
- Disentangled CFG combiner (3-pass per step, text + audio scales) — small, ~30 lines
- DMD 8-step sigma schedule selection via `get_timesteps_sigmas(use_distill=True)` — ~15 lines
- Latent normalization with per-channel `latents_mean`/`latents_std` — verbatim from VAE config

## Block Sparse Attention — explicitly dropped per v2

Reference uses `flash_attn_bsa_3d` with `sparsity=0.9375, chunk_3d_shape_q=[4,4,4], chunk_3d_shape_k=[4,4,4]` (from [dit/config.json](config-snapshot/longcat-video--dit-config.json)). MLX port replaces ALL attention paths (`flash_attn_func`, `flash_attn_bsa_3d`, `xformers.memory_efficient_attention`) with **dense `mx.fast.scaled_dot_product_attention`**. The wrapper's outer Q-slicing logic for Reference Skip Attention stays — only the inner `_process_attn` is replaced.

## What the base LongCat-Video has that Wan 2.1 doesn't (relevant to the diff)

From [longcat-video/dit/config.json](config-snapshot/longcat-video--dit-config.json):
- Same `hidden_size=4096, depth=48, num_heads=32, head_dim=128` as Wan 2.1 14B
- `text_tokens_zero_pad=True` (Avatar also)
- `caption_channels=4096` (same as Wan)
- `bsa_params` for block-sparse attention (we drop this; Wan 2.1 doesn't ship with BSA)

## Bottom-line estimate for Stage 1

- **Reusable from mlx-video**: ~2700 lines (VAE, RoPE, scheduler, text encoder helpers, attention helpers, tiling, utils)
- **Novel MLX code to write**: ~2000 lines (Audio adapter, audio cross-attn, Avatar DiT block, Avatar DiT model, pipeline with CFG)
- **Weight conversion recipe**: ~600 lines (cribbed from wan_2/convert.py + Meituan tensor name remapping)

That's tractable in the v2 estimate of ~3-4 weeks for a focused port.
