# Scripts — manual operation reference

## Standalone weight download (recovery from a reaped conversion)

If the background conversion in `recipes/convert_longcat_avatar.py` gets
killed by the harness (timeout, OOM, manual cancel), you can pull the
sources yourself from any terminal:

```bash
cd /Users/dustinnielson/DEV_INT/longcat-avatar-mlx
.venv/bin/python scripts/download_weights.py
```

The script uses `huggingface_hub.snapshot_download`, which is fully
resumable — re-running it after Ctrl-C picks up where it left off (cached
files are reused, partials are continued).

Options:
- `--only base` — just the LongCat-Video repo (VAE, umT5, tokenizer; ~25 GB)
- `--only avatar` — just the Avatar repo (DiT, LoRA, Whisper, Avatar scheduler; ~37 GB)
- `--only both` — everything (~60 GB) **[default]**
- `--dry-run` — print the patterns and exit, no network

When downloads complete, finish with the conversion:

```bash
.venv/bin/python -m recipes.convert_longcat_avatar --variant both \
    --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights
```

The conversion script will hit the same HF cache (`~/.cache/huggingface/hub/`)
and skip every file that's already downloaded — only the conversion math
runs (which is the fast part, CPU-bound).

## End-to-end inference

Once both conversion variants land:

```bash
.venv/bin/python scripts/run_inference.py \
    --weights /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights \
    --variant merged \
    --out scripts/output.mp4
```

Reads Meituan's shipped demo (`refs/longcat-video/assets/avatar/single/man.{png,mp3}`)
and the example prompt; runs the full denoising loop; saves an MP4 plus
a `.npy` fallback.

## Disk usage

| Path | Use |
|---|---|
| `~/.cache/huggingface/hub/` | Source PT weights (~60 GB, shared with any other HF-using project) |
| `/Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights/` | Converted MLX weights (~76 GB for both variants) |
| Total | ~136 GB |

You can delete the HF cache after conversion if disk pressure is high —
the converted MLX weights are self-contained.
