"""Dump Python-MLX UMT5 reference outputs for the Swift port's parity test.

Same pattern as scripts/dump_vae_swift_fixtures.py — but for the text encoder.
Loads the bf16 umT5 weights from HF, runs encode on a seeded input, dumps
`.npy` fixtures (~130 KB total) for Swift-side comparison.

Comparison strategy: the Python-MLX umT5 is already PT-parity-tested via
tests/parity/test_umt5_parity.py at max_abs < 1e-3. Targeting Swift-MLX
vs Python-MLX at 1e-4 transitively gives PT parity.

Usage:
    .venv/bin/python scripts/dump_umt5_swift_fixtures.py \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/umt5-parity/
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
#   B=1 (matches inference), L=16 (enough for the relative-position bias
#   to exercise multiple buckets without bloating the fixture).
INPUT_SHAPE = (1, 16)


def _seeded_ids(shape: tuple[int, ...], vocab_size: int, seed: int) -> np.ndarray:
    """Generate a deterministic input id sequence. Avoid id=0 (padding) so
    every position carries real signal."""
    rng = np.random.default_rng(seed)
    return rng.integers(low=1, high=min(vocab_size, 50_000), size=shape, dtype=np.int32)


def _load_umt5_from_hf(repo_id: str):
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.umt5 import UMT5EncoderModel

    config_path = hf_hub_download(repo_id=repo_id, filename="text_encoder/config.json")
    config = json.loads(pathlib.Path(config_path).read_text())
    model = UMT5EncoderModel.from_config(config)

    # Sharded: index.json + 3 shards
    idx_path = hf_hub_download(repo_id=repo_id, filename="text_encoder/model.safetensors.index.json")
    idx = json.loads(pathlib.Path(idx_path).read_text())
    shards = sorted(set(idx["weight_map"].values()))
    for shard in shards:
        shard_path = hf_hub_download(repo_id=repo_id, filename=f"text_encoder/{shard}")
        model.load_weights(shard_path, strict=False)
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

    print(f"Loading Python-MLX UMT5 from {args.weights}...")
    model, config = _load_umt5_from_hf(args.weights)
    vocab = config.get("vocab_size", 256_384)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building input {INPUT_SHAPE} with seed={args.seed}, vocab={vocab}")
    ids_np = _seeded_ids(INPUT_SHAPE, vocab_size=vocab, seed=args.seed)
    mask_np = np.ones(INPUT_SHAPE, dtype=np.int32)

    print("Running Python-MLX umT5 forward...")
    # umT5 weights are bf16; the forward returns bf16. numpy doesn't speak
    # bf16, so cast to fp32 before crossing the boundary. The Swift port's
    # parity test loads the .npy as fp32 too — fair apples-to-apples.
    out_mx = model(mx.array(ids_np), mask=mx.array(mask_np))
    out_mx = out_mx.astype(mx.float32)
    mx.eval(out_mx)
    out = np.asarray(out_mx)
    print(f"  output: shape={out.shape}, dtype={out.dtype}")

    paths = {
        "input_ids.npy": ids_np,
        "input_mask.npy": mask_np,
        "output.npy": out,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_umt5_swift_fixtures.py",
        "source_weights": args.weights,
        "seed": args.seed,
        "shapes": {
            "input_ids": list(INPUT_SHAPE),
            "input_mask": list(INPUT_SHAPE),
            "output": list(out.shape),
        },
        "vocab_size": vocab,
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Swift-MLX vs Python-MLX": None,
            "# Python-MLX vs PT is < 1e-3 per tests/parity/test_umt5_parity.py": None,
            "# We target 1e-4 here (fp32-rounding-only divergence between runtimes).": None,
            "output_max_abs": 1e-4,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
