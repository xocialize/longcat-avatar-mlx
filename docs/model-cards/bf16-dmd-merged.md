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
  - flow-matching
base_model:
  - meituan-longcat/LongCat-Video-Avatar-1.5
language:
  - en
  - zh
---

Part of the [LongCat-Video-Avatar 1.5 — MLX](https://huggingface.co/collections/mlx-community/longcat-video-avatar-15-mlx-6a185d1af4a43074d882e375) collection.


# LongCat-Video-Avatar-1.5-bf16-dmd-merged (MLX)

Apple MLX bf16 weights for [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven video diffusion model — with the DMD step-distillation LoRA
**pre-merged** into the DiT weights. Recommended variant for inference: produces
the same outputs as the base + LoRA combination but in **8 sampling steps** with
no LoRA loading at runtime.

## TL;DR

| | |
|---|---|
| **Architecture** | Wan 2.1 VAE + umT5-XXL + Whisper-Large-v3 + 48-block Avatar DiT (with DMD LoRA pre-merged) |
| **Params** | ~13.6 B DiT + ~11 B umT5 + ~0.6 B Whisper encoder + 0.5 B VAE |
| **Format** | bf16, sharded safetensors (HF-style per-component subdirs) |
| **Disk** | ~43 GB |
| **Hardware** | Apple Silicon M-series, 64 GB+ unified memory recommended for 480p |
| **Inference** | 8-step DMD distilled, FlowMatchEulerDiscreteScheduler with shift=7.0 |
| **License** | MIT (matches upstream Meituan) |

## Quick start

```bash
# 1. Pull weights (~43 GB)
hf download mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged \
    --local-dir ./weights

# 2. Set up inference (Python 3.12)
git clone https://github.com/xocialize/longcat-avatar-mlx
cd longcat-avatar-mlx
python3.12 -m venv .venv
.venv/bin/pip install -e ".[parity]"
.venv/bin/pip install librosa Pillow imageio imageio-ffmpeg

# 3. Run end-to-end
.venv/bin/python scripts/run_inference.py \
    --weights ./weights/.. \
    --variant merged \
    --num-frames 93 \
    --out output.mp4
```

Programmatic usage:

```python
import json
import pathlib
import mlx.core as mx

from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig
from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
    LongCatVideoAvatarTransformer3DModel,
)
from longcat_video_avatar.models.umt5 import UMT5EncoderModel
from longcat_video_avatar.models.whisper import WhisperEncoder

W = pathlib.Path("./weights")

vae = AutoencoderKLWan.from_config(json.loads((W/"vae/config.json").read_text()))
vae.load_weights(str(W/"vae/diffusion_pytorch_model.safetensors"), strict=False)

# (Load umT5/Whisper/DiT similarly via from_config + load_weights; see
# scripts/run_inference.py:build_pipeline() for the sharded-load helper.)

pipeline = LongCatAvatarPipeline(vae=vae, text_encoder=umt5,
                                 audio_encoder=whisper, dit=dit,
                                 config=PipelineConfig())
video = pipeline(image=ref_image_chw, audio_mel=mel_features,
                 text_embeds=text_embeds, text_mask=text_mask,
                 uncond_embeds=neg_embeds, uncond_mask=neg_mask)
```

## Variants

| Variant | DiT dtype | Disk | Sampling | Best for |
|---|---|---|---|---|
| **bf16-dmd-merged** *(this card)* | bf16 | 43 GB | 8-step | 64 GB+ Macs, recommended baseline |
| [bf16](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) | bf16 (+ separate LoRA) | 46 GB | 8-step or 50-step | runtime-merge / multi-strength experiments |
| [q4-dmd-merged](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged) | 4-bit quantized | 24 GB | 8-step | 32–48 GB Macs, comparable speed to bf16 |
| [q8-dmd-merged](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged) | 8-bit quantized | 31 GB | 8-step | middle ground RAM / quality |

## Performance

Tested on Apple M5 Max (128 GB unified memory):

| Resolution | Frames | Wall clock | ms/frame |
|---|---|---|---|
| 256 × 432 | 29 | ~105 s | ~3.6 s |
| 480 × 832 | 93 | TBD (estimate ~20–30 min) | — |

Peak memory at 480 × 832: ~50 GB unified. The VAE attention layers are
routed to a CPU stream by default (see [CLAUDE.md L10](https://github.com/xocialize/longcat-avatar-mlx/blob/main/CLAUDE.md))
to recover strict fp32 precision; net perf hit is negligible.

## Layout

```
LongCat-Video-Avatar-1.5-bf16-dmd-merged/
├── README.md                           # this file
├── pipeline_config.json
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors        # ~254 MB (bf16)
├── text_encoder/
│   ├── config.json
│   ├── model.safetensors.index.json
│   └── model-{00001,00002,00003}-of-00003.safetensors   # ~11 GB total (bf16)
├── audio_encoder/                                  # Whisper-large-v3 ENCODER only
│   ├── config.json
│   └── model.safetensors                           # ~1.3 GB (bf16)
├── dit/                                            # DMD LoRA pre-merged
│   ├── config.json
│   ├── diffusion_pytorch_model.safetensors.index.json
│   └── diffusion_pytorch_model-{00001..00007}-of-00007.safetensors   # ~33 GB total
├── scheduler/
│   └── scheduler_config.json                       # FlowMatchEuler, shift=7.0
└── tokenizer/                                       # umT5 tokenizer files
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## Source weights

Provenance, in case you want to verify or re-derive these weights:

| Subdir | Source | Conversion |
|---|---|---|
| `vae/` | `meituan-longcat/LongCat-Video/vae/` | Conv3d weight transpose `(O,I,T,H,W)→(O,T,H,W,I)`; dtype passthrough |
| `text_encoder/` | `meituan-longcat/LongCat-Video/text_encoder/` | HF verbose → mlx-compact key rename (`shared`→`token_embedding`, `encoder.block.{B}.layer.{0,1}.…`→`blocks.{B}.{attn,ffn,norm{1,2}}.…`); dtype cast to bf16 |
| `audio_encoder/` | `meituan-longcat/LongCat-Video-Avatar-1.5/whisper-large-v3/model.safetensors` | `model.encoder.` prefix strip; Conv1d weight transpose; encoder-only |
| `dit/` | `meituan-longcat/LongCat-Video-Avatar-1.5/base_model/` + `…/lora/dmd_lora.safetensors` | passthrough names; **DMD LoRA merged into 336 modules** (7 LoRA targets × 48 blocks); adaLN_modulation weights kept at fp32 |
| `scheduler/` | `meituan-longcat/LongCat-Video-Avatar-1.5/scheduler/` | verbatim copy |
| `tokenizer/` | `meituan-longcat/LongCat-Video/tokenizer/` | verbatim copy |

Conversion recipe: [`recipes/convert_longcat_avatar.py`](https://github.com/xocialize/longcat-avatar-mlx/blob/main/recipes/convert_longcat_avatar.py)
in the companion GitHub repo. Run with `--variant merged --out <dir>` to
reproduce these weights from Meituan's PT sources.

## Numerical conventions preserved from upstream

- **fp32 internal compute** for RMSNorm / LayerNorm / AdaLN modulation
  (Meituan's `_FP32` suffix convention) — `adaLN_modulation` and `gamma`
  weights are stored fp32 to defend this.
- **Negative velocity flip** before scheduler step (`noise_pred = -noise_pred`)
  — Meituan's DiT outputs `-v`; we flip to `+v` for `FlowMatchEulerDiscreteScheduler.step`.
- **3-pass disentangled CFG** combiner:
  `uncond + s_t·(cond − uncond_text) + s_a·(uncond_text − uncond)`.
  Defaults: `s_t = s_a = 4.0` (matches PT DMD distillation).
- **DMD distilled sigma schedule** — 8 sigmas spanning `[1.0, 0.124]`
  computed by `guidance.get_dmd_distilled_sigmas`. The pipeline overwrites
  the trailing sentinel sigma (mlx-arsenal appends `1.0`; we replace with
  `0.0` to actually denoise to clean at the final step — see
  [CLAUDE.md L17](https://github.com/xocialize/longcat-avatar-mlx/blob/main/docs/development/skill-lessons.md)).

## License

MIT. Matches upstream Meituan LongCat-Video license. Adapted code from
Blaizzy/mlx-video (MIT) for some MLX op primitives; full attribution in
[LICENSE](https://github.com/xocialize/longcat-avatar-mlx/blob/main/LICENSE).

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
