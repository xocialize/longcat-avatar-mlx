# longcat-avatar-mlx

Apple MLX port of [LongCat-Video-Avatar](https://github.com/meituan-longcat/LongCat-Video) —
Meituan's audio-driven talking-avatar video diffusion model — for inference on Apple
Silicon (M-series). A reference image, speech and a scene prompt make a 25 fps clip in
which the person in the image speaks.

**Current: [LongCat-Video-Avatar 1.6](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.6).**
The 1.5 variants still load and run.

## What 1.6 changes

1.6 keeps 1.5's architecture (13.6 B-parameter DiT, Wan 2.1 VAE, umT5-XXL, Whisper-large-v3)
and ships a new base DiT with **two** distillation LoRAs:

- `loraA` is the DMD LoRA, `loraB` the OPSD LoRA.
- They are switched **during** the 8-step denoise. With upstream's documented default
  (`--mix_steps=4 --mix_alpha=1.0`), steps 1–4 run DMD and steps 5–8 run OPSD. In general
  the first `mix_steps` steps run `mix_alpha * DMD + (1 - mix_alpha) * OPSD`.
- So the LoRAs cannot be merged into the weights ahead of time. This port runs them the way
  upstream does, as side-paths: `base(x) + up(down(x)) * multiplier * alpha_scale`
  (`longcat_video_avatar/lora.py`: `LoRALinear`, `LoRAMix`). That also works over a
  quantized base.
- Audio is no longer loudness-normalized, and is cut or zero-padded to the clip's length plus
  one second.

Long clips are a chain of segments, as upstream makes them: 93 frames (3.72 s) from the
image, then 80 more (3.2 s) per continuation, each conditioned on the last 13 frames of the
one before plus the reference image's latent (`generate_avc`, no KV cache).

### Corrections to the 1.5 pipeline, made with this update

Reading upstream's pipeline again for 1.6 turned up places where the 1.5 port's pipeline
(not its model code) had drifted from it. All are fixed, for 1.5 variants too:

| Was | Upstream, and now |
|---|---|
| 3-pass CFG at 4.0 / 4.0 on the distilled path | The demo forces both scales to 1.0 under `--use_distill`: no CFG, **one DiT pass per step**, about 3× faster. 4.0 / 4.0 is the 50-step default. |
| The image's latent was prepended as an extra latent and dropped before decoding | The image's latent **replaces the first** latent frame and is decoded: the image is frame 0, and audio frame *n* drives video frame *n*. |
| The conditioning latent ran at the step's timestep and was stepped by the scheduler | Conditioning latents run at **timestep 0** and are never stepped. |
| Continuation split reference / conditioning attention only when `mask_frame_range > 0` | Always split for `num_cond_latents > 1`; 1.6's default is `mask_frame_range = 0`. |
| One dense attention call over all tokens | Query-sliced: 32 heads × 37,000² scores is ~87 GB at 480p × 93 frames. Upstream uses flash attention. |
| Audio features interpolated over the 30 s padded mel | Only the audio's real extent is interpolated, as `get_audio_embedding_whisper` does. |

## Variants

Converted weights, per-component layout (`dit/`, `lora/`, `text_encoder/`, `tokenizer/`,
`audio_encoder/`, `vae/`):

| Variant | DiT | LoRAs | CLI `--variant` |
|---|---|---|---|
| [`LongCat-Video-Avatar-1.6-bf16`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.6-bf16) | bf16 | DMD + OPSD, runtime | `v1.6-bf16` |
| [`LongCat-Video-Avatar-1.6-q8`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.6-q8) | 8-bit | DMD + OPSD, runtime | `v1.6-q8` |
| [`LongCat-Video-Avatar-1.6-q4`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.6-q4) | 4-bit | DMD + OPSD, runtime | `v1.6-q4` |
| [`LongCat-Video-Avatar-1.5-bf16-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged) | bf16, DMD merged | — | `merged` |
| [`LongCat-Video-Avatar-1.5-bf16`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-bf16) | bf16 | DMD, merged at load | `base` |
| [`LongCat-Video-Avatar-1.5-q4-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q4-dmd-merged) | 4-bit, DMD merged | — | `q4-merged` |
| [`LongCat-Video-Avatar-1.5-q8-dmd-merged`](https://huggingface.co/mlx-community/LongCat-Video-Avatar-1.5-q8-dmd-merged) | 8-bit, DMD merged | — | `q8-merged` |

All seven are on the Hub under `mlx-community`. The recipe below rebuilds the 1.6 ones from Meituan's sources.

## Speed and memory (1.6 bf16, M5 Max 128 GB)

MLX 0.31.2, 8 distilled steps, no CFG. Step time grows with the token count
(`frames/4 × height/16 × width/16`), a little faster than linearly.

| Clip | Size | Per step | Whole run | MLX peak |
|---|---|---|---|---|
| 29 frames (1.16 s) | 432×256 | 3.0 s | 30 s | 58 GB |
| 2 segments, 173 frames (6.92 s) | 768×512 | 53 s | 15.4 min | 98 GB |

## Quick start

```bash
# 1. Clone + venv
git clone https://github.com/xocialize/longcat-avatar-mlx
cd longcat-avatar-mlx
python3.12 -m venv .venv
.venv/bin/pip install -e ".[parity]"      # parity extras pull torch/diffusers/transformers
.venv/bin/pip install librosa Pillow mlx-arsenal

# 2. Pull the converted weights (46 GB; `-q8` is 33 GB, `-q4` 26 GB)
hf download mlx-community/LongCat-Video-Avatar-1.6-bf16 \
    --local-dir ./weights/LongCat-Video-Avatar-1.6-bf16
#    or convert them yourself from Meituan's sources:
#    .venv/bin/python -m recipes.convert_longcat_avatar --variant v1.6-bf16 --out ./weights

# 3. Run against Meituan's demo inputs, or your own (ffmpeg muxes the speech into the MP4)
.venv/bin/python scripts/run_inference.py \
    --weights ./weights --variant v1.6-bf16 \
    --image portrait.png --audio speech.wav --prompt "A woman speaks to the camera in a sunlit kitchen" \
    --num-segments 2 --out output.mp4
```

CLI flags (`scripts/run_inference.py`): `--weights` (the directory that contains the
variant's directory), `--variant`, `--image` / `--audio` / `--prompt` (default: Meituan's
demo), `--num-segments` (1 = 93 frames; each more adds 80), `--num-frames` (frames per
segment, 4k + 1), `--height` / `--width` (default: upstream's 480p bucket closest to the
image's shape), `--seed`, `--mix-steps` / `--mix-alpha` (1.6), `--text-guidance-scale` /
`--audio-guidance-scale` (default 1.0: no CFG), `--out`.

Give the model clean speech: upstream's vocal separator (Kim_Vocal_2) is not ported.

From Python, `longcat_video_avatar/pipeline_mlx.py::LongCatAvatarPipeline` has
`generate_ai2v`, `generate_avc` and `__call__` (the whole clip).

`scripts/download_weights.py` is a helper that pulls the upstream 1.5 source weights.

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
├── pipeline_mlx.py              # LongCatAvatarPipeline: generate_ai2v, generate_avc, the whole clip
├── guidance.py                  # 3-pass disentangled CFG (undistilled) + distilled sigmas
├── audio_process.py             # Whisper post-processing (33→5 group pool, per-frame windows)
├── lora.py                      # LoRA merge math (1.5) and runtime side-paths + DMD/OPSD schedule (1.6)
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
                           48-block Avatar DiT (1.6: DMD LoRA steps 1–4, OPSD 5–8)
                                   │  FlowMatchEuler step × 8, no CFG
                                   ▼
                             Wan VAE decode ──▶ video
```

## Hardware requirements

- **Apple Silicon M-series** (tested on M5 Max 128 GB unified memory).
- **bf16 inference** at 480p × 93 frames: MLX peak about 98 GB with umT5 kept loaded (the
  Swift engine package evicts it per request). Smaller sizes and the quantized variants
  need much less.
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
| LoRAs | `<variant>/lora/dmd_lora.safetensors`, and for 1.6 `opsd_lora.safetensors` | Kohya-style split-fused QKV/KV; 7 targets/block × 48 = 336 modules each. 1.6: runtime side-paths. 1.5 `base`: merged at load |

## Tests

82 smoke + keymap test functions (13 smoke files) run in seconds with no weights:

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
  title  = {longcat-avatar-mlx: Apple MLX port of LongCat-Video-Avatar},
  author = {xocialize},
  year   = {2026},
  url    = {https://github.com/xocialize/longcat-avatar-mlx},
}
```
