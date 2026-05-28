"""Dump Python-MLX VAE reference outputs for the Swift port's parity test.

Generates small (~600 KB total) .npy fixtures that the Swift parity test
loads, then compares its own VAE outputs against. Uses the already-validated
Python-MLX VAE (which the project's tests/parity/test_vae_parity.py confirms
against PT to encode 1e-3 / decode 2e-2) — so passing the Swift parity test
at a tight threshold (~1e-5) means Swift-MLX matches Python-MLX, which by
transitivity matches PT.

Why fixture-based instead of running Python at Swift test time:
- Removes Python from the Swift test toolchain
- Keeps the Swift test deterministic (no MLX seed-state divergence)
- Lets us version the fixtures alongside the Swift parity test

Usage:
    .venv/bin/python scripts/dump_vae_swift_fixtures.py \\
        --weights mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/vae-parity/

The --weights argument is an HF repo id; weights are fetched via
huggingface_hub if not already cached locally.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import mlx.core as mx
import numpy as np


# Tiny shapes so the bundled fixtures stay small. Both shapes still exercise
# the chunked-encode/decode paths:
# - encode: 5 frames → 2 chunks (1 + (5-1)/4 = 2)
# - decode: 3 latent frames → 3 chunks (one per frame)
ENCODE_INPUT_SHAPE = (1, 3, 5, 16, 16)   # ~15 KB fp32
DECODE_INPUT_SHAPE = (1, 16, 3, 8, 8)    # ~24 KB fp32


def _seeded_input(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Same seeding convention as tests/parity/_helpers.make_seeded_input."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _load_vae_from_hf(repo_id: str):
    """Pull the bf16 VAE weights from HF and construct our MLX
    AutoencoderKLWan against them.
    """
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan

    config_path = hf_hub_download(repo_id=repo_id, filename="vae/config.json")
    weights_path = hf_hub_download(
        repo_id=repo_id, filename="vae/diffusion_pytorch_model.safetensors"
    )
    config = json.loads(pathlib.Path(config_path).read_text())
    vae = AutoencoderKLWan.from_config(config, encoder=True)
    vae.load_weights(weights_path, strict=False)
    mx.eval(vae.parameters())
    return vae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
        help="HF repo id for the published bf16 VAE",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=pathlib.Path,
        help="Output directory for the .npy fixtures + manifest.json",
    )
    parser.add_argument(
        "--encode-seed", type=int, default=42, help="Seed for encode input"
    )
    parser.add_argument(
        "--decode-seed", type=int, default=43, help="Seed for decode input"
    )
    args = parser.parse_args()

    print(f"Loading Python-MLX VAE from {args.weights}...")
    vae = _load_vae_from_hf(args.weights)

    args.out.mkdir(parents=True, exist_ok=True)

    # ----- encode -----
    print(f"Building encode input {ENCODE_INPUT_SHAPE} with seed={args.encode_seed}")
    enc_in = _seeded_input(ENCODE_INPUT_SHAPE, seed=args.encode_seed)
    enc_in = np.clip(enc_in, -1.0, 1.0)
    print("Running Python-MLX encode...")
    enc_out = np.asarray(vae.encode(mx.array(enc_in)))
    print(f"  encode output: shape={enc_out.shape}, dtype={enc_out.dtype}")

    # ----- decode -----
    print(f"Building decode input {DECODE_INPUT_SHAPE} with seed={args.decode_seed}")
    dec_in = _seeded_input(DECODE_INPUT_SHAPE, seed=args.decode_seed)
    print("Running Python-MLX decode...")
    dec_out = np.asarray(vae.decode(mx.array(dec_in)))
    print(f"  decode output: shape={dec_out.shape}, dtype={dec_out.dtype}")

    # ----- save .npy fixtures -----
    paths = {
        "encode_input.npy": enc_in,
        "encode_output.npy": enc_out,
        "decode_input.npy": dec_in,
        "decode_output.npy": dec_out,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    # ----- manifest -----
    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_vae_swift_fixtures.py",
        "source_weights": args.weights,
        "seeding": {"encode": args.encode_seed, "decode": args.decode_seed},
        "shapes": {
            "encode_input": list(ENCODE_INPUT_SHAPE),
            "encode_output": list(enc_out.shape),
            "decode_input": list(DECODE_INPUT_SHAPE),
            "decode_output": list(dec_out.shape),
        },
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Compare Swift-MLX against these Python-MLX outputs": None,
            "# Both runtimes call into the same mlx-c — divergence should be": None,
            "# tiny (fp32 rounding in different op orderings).": None,
            "encode_max_abs": 1e-4,
            "decode_max_abs": 1e-4,
        },
    }
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {manifest_path}")

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
