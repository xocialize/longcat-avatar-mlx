# longcat-avatar-mlx

Apple MLX port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven video diffusion model — for inference on Apple
Silicon (M-series).

End-to-end inference produces visually coherent, audio-synced video on a
single 128 GB M5 Max:

| Variant | Sampler | Disk | 29-frame wall clock @ 256×432 |
|---|---|---|---|
| `bf16-dmd-merged` | 8-step DMD distilled | 43 GB | **~105 s** |
| `bf16` | 50-step Flow Matching (+ runtime LoRA) | 46 GB | ~6 min |
| `q4-dmd-merged` | 8-step DMD distilled | 24 GB | ~102 s |
| `q8-dmd-merged` | 8-step DMD distilled | 31 GB | ~151 s |

Published on HuggingFace as a collection:
[🤗 mlx-community/longcat-video-avatar-15-mlx](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375)

- 🤗 [`bf16-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged) — DMD pre-merged into DiT, **recommended for 64 GB+ Macs** (43 GB) — CLI `--variant merged`
- 🤗 [`bf16`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) — base bf16 + separate DMD LoRA, runtime-merge / multi-strength (46 GB) — CLI `--variant base`
- 🤗 [`q4-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged) — 4-bit DiT, **recommended for 32–48 GB Macs** (24 GB) — CLI `--variant q4-merged`
- 🤗 [`q8-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged) — 8-bit DiT, middle-ground (31 GB) — CLI `--variant q8-merged`

> **Note on variant names:** the HF repos use `…-dmd-merged` suffixes; the
> `run_inference.py` CLI maps them to the shorter `{merged, base, q4-merged, q8-merged}`
> (see `VARIANT_DIRNAMES`).

## Quick start

```bash
# 1. Clone + venv
git clone https://github.com/xocialize/longcat-avatar-mlx
cd longcat-avatar-mlx
python3.12 -m venv .venv
# runtime deps (mlx/safetensors/hf_hub/numpy) come from the package; the inference CLI
# additionally needs Pillow, imageio(+ffmpeg), librosa, and transformers — install them:
.venv/bin/pip install -e ".[parity]"      # parity extras pull torch/diffusers/transformers
.venv/bin/pip install librosa Pillow imageio imageio-ffmpeg

# 2. Pull MLX weights (one-time, resumable). Each variant lives in its own HF repo.
hf download mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged \
    --local-dir ./weights/LongCat-Video-Avatar-1.5-bf16-dmd-merged

# 3. Run inference against Meituan's shipped demo inputs.
#    --weights points at the directory that CONTAINS the per-variant subdir(s).
.venv/bin/python scripts/run_inference.py \
    --weights ./weights \
    --variant merged \
    --num-frames 93 \
    --height 480 --width 832 \
    --out output.mp4
```

CLI flags (`scripts/run_inference.py`): `--weights` (required), `--variant
{base,merged,q4-merged,q8-merged}` (default `merged`), `--height` (480), `--width` (832),
`--num-frames` (93), `--seed` (42), `--out`.

Bring your own portrait + audio + scene prompt by editing
`scripts/run_inference.py::load_demo_inputs()`, or call
`longcat_video_avatar/pipeline_mlx.py::LongCatAvatarPipeline.__call__` directly.

`scripts/download_weights.py` is a helper that pulls the upstream source weights (for
re-conversion / parity); end users only need the published MLX variants above.

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
├── lora.py                      # DMD LoRA loader (split-fused QKV/KV merge math)
└── utils/                       # shared helpers
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
                                   │  FlowMatchEuler step × 8
                                   ▼
                             Wan VAE decode ──▶ video
```

## Hardware requirements

- **Apple Silicon M-series** (tested on M5 Max 128 GB unified memory).
- **bf16 inference** at 480p: peak ~50 GB RAM. 64 GB Mac viable; 32 GB Mac
  should use `q4-merged`.
- **First-time setup**: published MLX variant (24–46 GB) plus, for re-conversion,
  ~60 GB of source weights cached at `~/.cache/huggingface/hub/`.

## Components + provenance

Every component is loaded from the corresponding HF subdir of the published variant:

| Component | Loaded from | Notes |
|---|---|---|
| Wan 2.1 VAE | `<variant>/vae/` | encode parity 7e-6, decode parity 1.2e-2 (Metal-GPU fp32 precision) |
| umT5-XXL | `<variant>/text_encoder/` (sharded) | per-block relative position bias, gated GeLU FFN |
| Whisper-large-v3 (encoder only) | `<variant>/audio_encoder/` | 33 hidden states → 5-group mean-pool → 25 Hz linear-interp |
| Avatar DiT | `<variant>/dit/` (sharded) | 48 blocks × {self-attn, text-cross, audio-cross, SwiGLU FFN}, per-block AdaLN, Reference Skip Q-slicing |
| DMD LoRA | `<variant>/lora/dmd_lora.safetensors` (base variant only) | Kohya-style split-fused QKV/KV; 7 targets/block × 48 = 336 merges |

## Tests

70 smoke + keymap test functions (13 smoke files) run in seconds with no weights:

```bash
.venv/bin/python -m pytest tests/smoke -v
```

PT-vs-MLX numerical parity tests are opt-in (each component has its own
`LONGCAT_*_AUTO_DOWNLOAD` env var):

```bash
.venv/bin/pip install -e ".[parity]"
LONGCAT_VAE_AUTO_DOWNLOAD=1 \
    .venv/bin/python -m pytest tests/parity/test_vae_parity.py -v
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
```
