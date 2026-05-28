"""Standalone download of all LongCat-Video-Avatar-1.5 source weights from HuggingFace.

Run this if the conversion script's background download keeps getting reaped
(harness timeout, manual cancel, network blip). Once this completes, the
conversion script will reuse the HF cache and skip the download phase.

Behavior:
- Uses `huggingface_hub.snapshot_download` which is fully resumable on
  interruption (caches by content hash; re-run is idempotent).
- Total disk footprint in `~/.cache/huggingface/hub/`: ~60 GB.
- Progress bars are emitted by the library; safe to Ctrl-C anytime.
- Safe to run from any terminal session; the cache is shared with the
  conversion script.

Usage:
    python scripts/download_weights.py
    # or, after `pip install -e .` to make `huggingface_hub` available:
    python -m scripts.download_weights

When all downloads complete, run:
    python -m recipes.convert_longcat_avatar --variant both \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights
"""

from __future__ import annotations

import argparse
import sys

# Components we need, organized by HF repo. `allow_patterns` is the
# `snapshot_download` parameter — only matching files are downloaded.
DOWNLOADS = {
    "meituan-longcat/LongCat-Video": [
        # Wan VAE
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        # umT5-XXL text encoder (sharded, 22 GB total)
        "text_encoder/model.safetensors.index.json",
        "text_encoder/model-*-of-*.safetensors",
        "text_encoder/config.json",
        # Tokenizer
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
        "tokenizer/spiece.model",  # SentencePiece model — may or may not be present
        # Scheduler (base model's; the Avatar's takes precedence in the pipeline)
        "scheduler/scheduler_config.json",
        # Top-level
        "config.json",
        "model_index.json",
        "README.md",
        "LICENSE",
    ],
    "meituan-longcat/LongCat-Video-Avatar-1.5": [
        # Avatar DiT (sharded, ~32 GB total bf16)
        "base_model/diffusion_pytorch_model.safetensors.index.json",
        "base_model/diffusion_pytorch_model-*-of-*.safetensors",
        "base_model/config.json",
        # DMD LoRA (~2.5 GB)
        "lora/dmd_lora.safetensors",
        # Whisper-large-v3 encoder (we only use model.safetensors at bf16 — 3 GB)
        "whisper-large-v3/model.safetensors",
        "whisper-large-v3/config.json",
        "whisper-large-v3/generation_config.json",
        "whisper-large-v3/preprocessor_config.json",
        # Avatar scheduler (shift=7.0 — overrides the base scheduler in pipeline)
        "scheduler/scheduler_config.json",
        # Top-level
        "config.json",
        "README.md",
    ],
}


def download_one(repo_id: str, patterns: list[str]) -> None:
    """Download all files from `repo_id` matching `patterns`. Resumable."""
    from huggingface_hub import snapshot_download

    print(f"\n=== Downloading from {repo_id} ===")
    print(f"  Patterns ({len(patterns)}):")
    for p in patterns:
        print(f"    {p}")
    path = snapshot_download(
        repo_id=repo_id,
        allow_patterns=patterns,
        # Default cache dir: ~/.cache/huggingface/hub/
        # tqdm progress bars are emitted automatically.
    )
    print(f"  Cache root for this repo: {path}")


def estimate_total_size() -> str:
    """Rough estimate for user planning. Actual usage varies."""
    return (
        "  - Wan VAE:           ~0.5 GB (single file)\n"
        "  - umT5-XXL:          ~22 GB (5 shards)\n"
        "  - Whisper-large-v3:  ~3 GB (bf16 single file)\n"
        "  - Avatar DiT:        ~32 GB (6 shards bf16)\n"
        "  - DMD LoRA:          ~2.5 GB (single file)\n"
        "  - configs/tokenizer: <100 MB\n"
        "  Total:               ~60 GB"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--only",
        choices=("base", "avatar", "both"),
        default="both",
        help=(
            "Which repo(s) to download. 'base' = LongCat-Video (VAE + umT5 + "
            "tokenizer); 'avatar' = LongCat-Video-Avatar-1.5 (DiT + LoRA + Whisper). "
            "Default is 'both'."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be downloaded and exit. Does not hit the network.",
    )
    args = parser.parse_args()

    print("LongCat-Video-Avatar-1.5 weight download")
    print("Expected disk footprint:")
    print(estimate_total_size())
    print(f"\nDownload mode: {args.only}")
    if args.dry_run:
        print("\n(dry-run — no actual downloads)")
        for repo_id, patterns in DOWNLOADS.items():
            if args.only == "base" and repo_id != "meituan-longcat/LongCat-Video":
                continue
            if args.only == "avatar" and repo_id == "meituan-longcat/LongCat-Video":
                continue
            print(f"\nWould fetch from {repo_id}:")
            for p in patterns:
                print(f"  {p}")
        return

    try:
        if args.only in ("base", "both"):
            download_one("meituan-longcat/LongCat-Video", DOWNLOADS["meituan-longcat/LongCat-Video"])
        if args.only in ("avatar", "both"):
            download_one(
                "meituan-longcat/LongCat-Video-Avatar-1.5",
                DOWNLOADS["meituan-longcat/LongCat-Video-Avatar-1.5"],
            )
    except KeyboardInterrupt:
        print("\nInterrupted. Re-run this script to resume (cached files are reused).")
        sys.exit(130)
    except Exception as e:
        print(f"\nDownload failed: {e!r}")
        print("Re-run this script to retry. Cached partial files will be reused.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All downloads complete. Next step:")
    print(
        "  python -m recipes.convert_longcat_avatar --variant both \\\n"
        "      --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-weights"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
