# longcat-avatar-mlx

Apple MLX port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven video diffusion model — for inference on Apple
Silicon (M-series).

End-to-end inference produces visually coherent, audio-synced video on a
single 128 GB M5 Max:

| Variant | Sampler | First frame @ 256×432×29 |
|---|---|---|
| `bf16` | 50-step Flow Matching | ~6 min |
| `bf16-dmd-merged` | 8-step DMD distilled | **~105 s** |

Published on HuggingFace as a collection:
[🤗 mlx-community/longcat-video-avatar-15-mlx](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375)

- 🤗 [mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged) — DMD pre-merged into DiT, **recommended** (46 GB, 25 files)
- 🤗 [mlx-community/LongCat-Video-Avatar-1.5-bf16](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) — base bf16 + separate DMD LoRA (49 GB, 26 files)

## Quick start

```bash
# 1. Clone + venv
git clone https://github.com/xocialize/longcat-avatar-mlx
cd longcat-avatar-mlx
python3.12 -m venv .venv
.venv/bin/pip install -e ".[parity]"      # parity extras pull torch/diffusers/transformers
.venv/bin/pip install librosa Pillow imageio imageio-ffmpeg

# 2. Pull MLX weights (~43 GB, one-time, resumable)
hf download mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged \
    --local-dir ./weights

# 3. Run inference against Meituan's shipped demo inputs
.venv/bin/python scripts/run_inference.py \
    --weights ./weights/.. \
    --variant merged \
    --num-frames 93 \
    --height 480 --width 832 \
    --out output.mp4
```

Bring your own portrait + audio + scene prompt by editing
`scripts/run_inference.py:load_demo_inputs()` (or pass through your own
preprocessor — see `longcat_video_avatar/pipeline_mlx.py:LongCatAvatarPipeline.__call__`).

## What's inside

```
longcat_video_avatar/
├── models/
│   ├── autoencoder_kl_wan.py    # Wan 2.1 VAE
│   ├── umt5.py                  # umT5-XXL text encoder
│   ├── whisper.py               # Whisper-Large-v3 encoder
│   ├── longcat_video_dit.py     # base 48-block DiT
│   ├── attention.py             # 3D self-attn + text cross-attn primitives
│   ├── blocks.py                # PatchEmbed3D, TimestepEmbedder, SwiGLU FFN, …
│   ├── rope_3d.py               # 3D + 1D RoPE
│   └── avatar/                  # Avatar 1.5 overlay
│       ├── attention.py         # SingleStreamAttention + Reference Skip
│       ├── blocks.py            # AudioProjModel
│       └── longcat_video_dit_avatar.py  # full Avatar DiT
├── pipeline_mlx.py              # LongCatAvatarPipeline (umT5 + Whisper + VAE + DiT + LoRA + CFG)
├── guidance.py                  # 3-pass disentangled CFG + DMD sigmas
├── audio_process.py             # Whisper post-processing (33→5 group pool, windowing)
└── lora.py                      # DMD LoRA loader (split-fused QKV/KV merge math)
```

Inference flow:

```
ref_image  ──▶ Wan VAE encode    ──┐
                                   ▼
audio_wav  ──▶ Whisper → group-pool ──▶ AudioProjModel ──▶ audio cross-attn
                                   │
prompt     ──▶ umT5 encode ──┐    │
                             ▼    ▼
                           48-block Avatar DiT (with DMD LoRA pre-merged)
                                   │
                                   ▼
                             FlowMatchEuler step × 8
                                   │
                                   ▼
                             Wan VAE decode ──▶ video
```

## Hardware requirements

- **Apple Silicon M-series** (tested on M5 Max 128 GB unified memory).
- **bf16 inference** at 480p: peak ~50 GB RAM. 64 GB Mac viable; 32 GB Mac
  will OOM without int4 quantization (not yet shipped).
- **First-time setup**: ~60 GB downloaded source weights cached at
  `~/.cache/huggingface/hub/` plus ~43 GB published MLX weights.
  `hf download --include='dit/*'` etc. can partial-pull if disk is tight.

## Components + provenance

Every component is loaded from the corresponding HF subdir:

| Component | Source | Schema | Notes |
|---|---|---|---|
| Wan 2.1 VAE | `meituan-longcat/LongCat-Video/vae` | diffusers 0.38 canonical | encode parity 7e-6, decode parity 1.2e-2 (Metal-GPU fp32 precision per CLAUDE.md L10) |
| umT5-XXL | `meituan-longcat/LongCat-Video/text_encoder` | HF transformers | per-block relative position bias, gated GeLU FFN |
| Whisper-large-v3 (encoder only) | `meituan-longcat/LongCat-Video-Avatar-1.5/whisper-large-v3` | HF transformers | 33 hidden states → 5-group mean-pool → 25 Hz linear-interp |
| Avatar DiT | `meituan-longcat/LongCat-Video-Avatar-1.5/base_model` | Meituan custom | 48 blocks × {self-attn, text-cross, audio-cross, SwiGLU FFN}, per-block AdaLN gating, Reference Skip Q-slicing |
| DMD LoRA | `meituan-longcat/LongCat-Video-Avatar-1.5/lora/dmd_lora.safetensors` | Kohya-style with split-fused QKV/KV | 7 LoRA targets/block × 48 blocks = 336 merges |
| Scheduler | `…/scheduler/scheduler_config.json` (shift=7.0) | diffusers `FlowMatchEulerDiscreteScheduler` | DMD distilled sigma schedule supplied by `guidance.get_dmd_distilled_sigmas` |

## Tests

70 smoke + keymap tests run in < 5 seconds with no weights:

```bash
.venv/bin/python -m pytest tests/smoke -v   # 70 pass
```

PT-vs-MLX numerical parity tests are opt-in:

```bash
.venv/bin/pip install -e ".[parity]"
LONGCAT_VAE_AUTO_DOWNLOAD=1 \
    .venv/bin/python -m pytest tests/parity/test_vae_parity.py -v
# Each component has its own LONGCAT_*_AUTO_DOWNLOAD env var.
```

## License

MIT. Adapted from work by [Meituan LongCat Team](https://github.com/meituan-longcat/LongCat-Video)
(MIT) and the Wan VAE module pattern from [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video)
(MIT). See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{longcat-avatar-mlx,
  title  = {longcat-avatar-mlx: Apple MLX port of LongCat-Video-Avatar-1.5},
  author = {xocialize},
  year   = {2026},
  url    = {https://github.com/xocialize/longcat-avatar-mlx},
}

@techreport{meituan2026longcat,
  title       = {LongCat-Video-Avatar 1.5 Technical Report},
  author      = {Meituan LongCat Team},
  institution = {Meituan},
  year        = {2026},
  url         = {https://github.com/meituan-longcat/LongCat-Video},
}
```
