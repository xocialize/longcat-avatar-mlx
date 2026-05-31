"""Dump Python-MLX base-DiT reference outputs for the Swift parity test.

Same pattern as dump_{vae,umt5,whisper}_swift_fixtures.py. Loads the BASE
LongCatVideoTransformer3DModel (text-only, no avatar audio overlay) from
the published bf16-dmd-merged variant — the audio-only safetensors keys
are silently dropped at load time via strict=False, since the base class
has no audio attributes to receive them.

Inputs are synthetic at small spatial sizes (T=1, H=8, W=8 latent) so the
parity test runs in seconds even at depth=48 production layers. Output
size ~13 MB.

Usage:
    .venv/bin/python scripts/dump_dit_swift_fixtures.py \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/dit-parity/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import mlx.core as mx
import numpy as np


# Tiny shapes so the bundled fixtures stay small + parity runs quick.
# Latent: B=1, C=16, T=1, H=8, W=8 → 8x8 latent at one frame.
# Text:   B=1, N=4 tokens (we use a short seq for speed).
LATENT_SHAPE = (1, 16, 1, 8, 8)
TEXT_TOKENS = 4
CAPTION_CHANNELS = 4096


def _seeded_normal(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _load_dit_from_hf(repo_id: str):
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.longcat_video_dit import (
        LongCatVideoTransformer3DModel,
    )

    cfg_path = hf_hub_download(repo_id=repo_id, filename="dit/config.json")
    config = json.loads(pathlib.Path(cfg_path).read_text())
    model = LongCatVideoTransformer3DModel.from_config(config)

    # Sharded weights — load via the index
    idx_path = hf_hub_download(repo_id=repo_id, filename="dit/diffusion_pytorch_model.safetensors.index.json")
    idx = json.loads(pathlib.Path(idx_path).read_text())
    shards = sorted(set(idx["weight_map"].values()))
    print(f"  Loading {len(shards)} shards...")
    for shard in shards:
        sp = hf_hub_download(repo_id=repo_id, filename=f"dit/{shard}")
        # strict=False so avatar-overlay keys (audio_proj, audio_adaLN, etc.)
        # are silently dropped (base class has no slots for them).
        model.load_weights(sp, strict=False)
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

    print(f"Loading Python-MLX base DiT from {args.weights}...")
    model, config = _load_dit_from_hf(args.weights)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building synthetic input: latent {LATENT_SHAPE}, text [1, 1, {TEXT_TOKENS}, {CAPTION_CHANNELS}]")
    hidden_states_np = _seeded_normal(LATENT_SHAPE, seed=args.seed)
    timestep_np = np.array([500.0], dtype=np.float32)
    text_np = _seeded_normal(
        (1, 1, TEXT_TOKENS, CAPTION_CHANNELS), seed=args.seed + 1
    )
    text_mask_np = np.ones((1, TEXT_TOKENS), dtype=np.int32)

    print("Running Python-MLX DiT forward (this can take 30-90s at depth=48)...")
    out_mx = model(
        hidden_states=mx.array(hidden_states_np),
        timestep=mx.array(timestep_np),
        encoder_hidden_states=mx.array(text_np),
        encoder_attention_mask=mx.array(text_mask_np),
    )
    out_mx = out_mx.astype(mx.float32)
    mx.eval(out_mx)
    out = np.asarray(out_mx)
    print(f"  output: shape={out.shape}, dtype={out.dtype}, abs.max={np.abs(out).max():.4g}")

    paths = {
        "hidden_states.npy": hidden_states_np,
        "timestep.npy": timestep_np,
        "encoder_hidden_states.npy": text_np,
        "encoder_attention_mask.npy": text_mask_np,
        "output.npy": out,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_dit_swift_fixtures.py",
        "source_weights": args.weights,
        "seed": args.seed,
        "shapes": {
            "hidden_states": list(LATENT_SHAPE),
            "timestep": [1],
            "encoder_hidden_states": [1, 1, TEXT_TOKENS, CAPTION_CHANNELS],
            "encoder_attention_mask": [1, TEXT_TOKENS],
            "output": list(out.shape),
        },
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Per L22 + S3.4/S3.5 findings: the base DiT uses attn.qkv + sdpa,": None,
            "# FFN swiglu via plain Linears. 48 layers x bf16 noise compounds.": None,
            "# Target generous threshold while we measure actual drift.": None,
            "output_max_abs": 0.5,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
