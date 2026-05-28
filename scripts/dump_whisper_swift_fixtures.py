"""Dump Python-MLX Whisper-encoder reference outputs for the Swift parity test.

Same pattern as dump_{vae,umt5}_swift_fixtures.py. Generates a seeded
synthetic mel input (since we don't need actual audio for parity testing
— the goal is matching the network's outputs given a fixed input) plus
the model's final post-LayerNorm output. ~500 KB total.

Usage:
    .venv/bin/python scripts/dump_whisper_swift_fixtures.py \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/whisper-parity/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import mlx.core as mx
import numpy as np


# Small but representative shape:
#   B=1, num_mel_bins=128 (Whisper-large-v3 spec), T_mel=64
#   → T_enc=32 after stride-2 conv2
INPUT_SHAPE = (1, 128, 64)


def _seeded_mel(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Synthetic log-mel-ish input in the [-1, 1] range Whisper expects."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32).clip(-1.0, 1.0)


def _load_whisper_from_hf(repo_id: str):
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.whisper import WhisperEncoder

    cfg_path = hf_hub_download(repo_id=repo_id, filename="audio_encoder/config.json")
    config = json.loads(pathlib.Path(cfg_path).read_text())
    model = WhisperEncoder.from_config(config)
    weights_path = hf_hub_download(
        repo_id=repo_id, filename="audio_encoder/model.safetensors"
    )
    model.load_weights(weights_path, strict=False)
    mx.eval(model.parameters())
    return model, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading Python-MLX Whisper encoder from {args.weights}...")
    model, _ = _load_whisper_from_hf(args.weights)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building synthetic mel input {INPUT_SHAPE} with seed={args.seed}")
    mel_np = _seeded_mel(INPUT_SHAPE, seed=args.seed)

    print("Running Python-MLX Whisper encoder forward...")
    # Weights are bf16; output is bf16 — cast to fp32 for numpy interop
    out_mx = model(mx.array(mel_np))
    out_mx = out_mx.astype(mx.float32)
    mx.eval(out_mx)
    out = np.asarray(out_mx)
    print(f"  output: shape={out.shape}, dtype={out.dtype}")

    paths = {
        "input_mel.npy": mel_np,
        "output.npy": out,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_whisper_swift_fixtures.py",
        "source_weights": args.weights,
        "seed": args.seed,
        "shapes": {
            "input_mel": list(INPUT_SHAPE),
            "output": list(out.shape),
        },
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Like umT5 (L22), expect Python-MLX vs Swift-MLX bf16 matmul drift.": None,
            "# Whisper is smaller (d_model=1280 vs umT5's 4096) but 32 layers": None,
            "# vs 24 — net compounded drift is likely in the same ballpark.": None,
            "output_max_abs": 0.15,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
