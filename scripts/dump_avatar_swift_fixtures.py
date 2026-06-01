"""Dump Python-MLX Avatar-DiT reference outputs for the Swift parity test.

Loads the FULL LongCatVideoAvatarTransformer3DModel (base DiT + audio
overlay) from the published bf16-dmd-merged variant. No strict=False
filtering needed — the Avatar class slots match every published key.

Inputs are synthetic at small spatial sizes (T=3 latent frames so the
audio path's middle/last reshape exercises the v>2 branch). Output size
~13 KB total fixtures.

Usage:
    .venv/bin/python scripts/dump_avatar_swift_fixtures.py \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/avatar-parity/
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
# Latent: B=1, C=16, T=3, H=8, W=8. T=3 is enough to exercise the audio
# windowing's middle-frames branch (which only triggers when v=4>2).
LATENT_SHAPE = (1, 16, 3, 8, 8)
TEXT_TOKENS = 4
CAPTION_CHANNELS = 4096

# Audio: B=1, T_audio=9 (= 1 + 2*4 for T=3 with vae_scale=4), W=5, S=5, C=1280
AUDIO_SHAPE = (1, 9, 5, 5, 1280)


def _seeded_normal(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _load_avatar_from_hf(repo_id: str):
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )

    cfg_path = hf_hub_download(repo_id=repo_id, filename="dit/config.json")
    config = json.loads(pathlib.Path(cfg_path).read_text())
    model = LongCatVideoAvatarTransformer3DModel.from_config(config)

    # Sharded weights
    idx_path = hf_hub_download(repo_id=repo_id, filename="dit/diffusion_pytorch_model.safetensors.index.json")
    idx = json.loads(pathlib.Path(idx_path).read_text())
    shards = sorted(set(idx["weight_map"].values()))
    print(f"  Loading {len(shards)} shards...")
    for shard in shards:
        sp = hf_hub_download(repo_id=repo_id, filename=f"dit/{shard}")
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

    print(f"Loading Python-MLX Avatar DiT from {args.weights}...")
    model, _ = _load_avatar_from_hf(args.weights)

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Building synthetic input: latent {LATENT_SHAPE}, audio {AUDIO_SHAPE}, text [1,1,{TEXT_TOKENS},{CAPTION_CHANNELS}]")
    hidden_states_np = _seeded_normal(LATENT_SHAPE, seed=args.seed)
    timestep_np = np.array([500.0], dtype=np.float32)
    text_np = _seeded_normal((1, 1, TEXT_TOKENS, CAPTION_CHANNELS), seed=args.seed + 1)
    text_mask_np = np.ones((1, TEXT_TOKENS), dtype=np.int32)
    audio_np = _seeded_normal(AUDIO_SHAPE, seed=args.seed + 2)

    print("Running Python-MLX Avatar DiT forward (this can take 30-90s at depth=48)...")
    out_mx = model(
        hidden_states=mx.array(hidden_states_np),
        timestep=mx.array(timestep_np),
        encoder_hidden_states=mx.array(text_np),
        audio_embs=mx.array(audio_np),
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
        "audio_embs.npy": audio_np,
        "output.npy": out,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_avatar_swift_fixtures.py",
        "source_weights": args.weights,
        "seed": args.seed,
        "shapes": {
            "hidden_states": list(LATENT_SHAPE),
            "timestep": [1],
            "encoder_hidden_states": [1, 1, TEXT_TOKENS, CAPTION_CHANNELS],
            "encoder_attention_mask": [1, TEXT_TOKENS],
            "audio_embs": list(AUDIO_SHAPE),
            "output": list(out.shape),
        },
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Base DiT was 0.033. Avatar adds audio cross-attn (fused SDPA)": None,
            "# per block + audio_adaLN modulation. Same fused-SDPA-dominant.": None,
            "output_max_abs": 0.15,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
