---
license: mit
library_name: mlx
pipeline_tag: text-to-video
tags:
  - mlx
  - apple-silicon
  - video-generation
  - audio-driven-video
  - longcat
  - distilled
  - quantized
  - 8-bit
base_model:
  - mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged
language:
  - en
  - zh
---

Part of the [LongCat-Video-Avatar 1.5 — MLX](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375) collection.

# LongCat-Video-Avatar-1.5-q8-dmd-merged (MLX)

8-bit quantized variant of [mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged).
Same model, same DMD pre-merge, same 8-step inference path — just with the
DiT Linears quantized to 8-bit via `mlx.nn.quantize` for smaller-RAM Macs.

| | |
|---|---|
| **DiT** | 8-bit quantized (`group_size=64`, skip `final_layer.linear` + embedders + AdaLN) |
| **DiT shards** | ~18 GB (4 shards) |
| **umT5 / Whisper / VAE** | bf16 (unchanged from the bf16-dmd-merged variant) |
| **Total disk** | ~31 GB |
| **Min unified memory** | ~32 GB |
| **Inference** | 8-step DMD distilled (unchanged) |
| **License** | MIT |

## Performance

Measured on Apple M5 Max (128 GB unified memory), 256 × 432 × 29 frames,
8-step DMD sampling:

| Variant | Wall clock | ms/frame |
|---|---|---|
| bf16-dmd-merged | ~105 s | ~3.6 s |
| **q4-dmd-merged** | ~102 s | ~3.5 s |
| **q8-dmd-merged** | ~151 s | ~5.2 s |

q4 is bandwidth-bound (matches bf16 throughput); q8 currently runs slower
on M5's quantized matmul kernels but uses ~half the DiT disk vs bf16. Pick
the variant by RAM budget, not speed.

## Loading

The runtime pipeline (`longcat_video_avatar.pipeline_mlx.LongCatAvatarPipeline`)
auto-detects the `quantization` block in `dit/config.json` and applies
`mlx.nn.quantize` before loading the quantized weights. No user-facing API
change vs. the bf16 variant.

```bash
hf download mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged \
    --local-dir ./weights
.venv/bin/python scripts/run_inference.py \
    --weights ./weights/.. \
    --variant q8-merged \
    --num-frames 93 \
    --out output.mp4
```

## Source

Quantized from the bf16-dmd-merged variant via
[`recipes/convert_longcat_avatar.py`](https://github.com/xocialize/longcat-avatar-mlx/blob/main/recipes/convert_longcat_avatar.py).
Run with `--variant q8-merged --out <dir>` to reproduce from Meituan's
PT sources.

See the [bf16-dmd-merged card](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged)
for full architecture details, citation, and the non-quantized variant.
