# CLAUDE.md — longcat-avatar-mlx

Operational orientation for Claude when working in this repo.

## What this is

MLX port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video).
Meituan's audio-driven video diffusion model on Apple Silicon (13.6B-param
DiT + Wan VAE + umT5-XXL + Whisper-Large-v3 + DMD step-distillation LoRA).

## Where things live

- **[README.md](README.md)** — public-facing quick start, variants, perf numbers.
- **[longcat_video_avatar/](longcat_video_avatar/)** — the MLX package. File
  names mirror Meituan's `longcat_video/` layout 1:1 per the mlx-porting
  skill's isomorphic-structure rule.
- **[recipes/](recipes/)** — weight conversion recipe (PT → MLX, with both
  variants and the silent-zero-trap defense).
- **[scripts/](scripts/)** — user-facing CLIs: `download_weights.py`,
  `run_inference.py`, plus README walkthroughs.
- **[tests/](tests/)** — smoke (no deps, no weights) and parity (PT comparison,
  needs `[parity]` extras and weights via opt-in env vars).
- **[docs/development/](docs/development/)** — Stage 0 architectural recon
  notes, port plan, skill-lessons captured during the port.
- **[docs/model-cards/](docs/model-cards/)** — reference copies of the
  mlx-community model cards (published to HF separately).
- **`refs/`** — reference checkouts of `meituan-longcat/LongCat-Video`,
  `Blaizzy/mlx-video`, `Blaizzy/mlx-audio`. Gitignored. Re-clone via:
  ```bash
  git clone --depth 1 https://github.com/meituan-longcat/LongCat-Video.git refs/longcat-video
  ```

## Running tests

```bash
.venv/bin/python -m pytest tests/smoke -v          # always green, no weights
LONGCAT_VAE_AUTO_DOWNLOAD=1 \
  .venv/bin/python -m pytest tests/parity -v        # PT parity, needs [parity] extras
```

Each per-component parity test has its own `LONGCAT_{NAME}_AUTO_DOWNLOAD=1`
env var so you can opt in component-by-component instead of pulling all
~60 GB of source weights at once.

## Skill in use

`/mlx-porting` is the operational skill for this port. Load it before
making structural changes. Per-lesson rationale lives in
[docs/development/skill-lessons.md](docs/development/skill-lessons.md) —
18 `[toolkit candidate]`-tagged entries with file:line citations.
