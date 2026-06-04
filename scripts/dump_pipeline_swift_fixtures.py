"""Dump Python-MLX end-to-end pipeline reference for the Swift parity test.

Runs the full LongCatAvatarPipeline (VAE + umT5 + Whisper + Avatar DiT +
3-pass CFG + 8-step DMD denoising loop) on synthetic small inputs with
controlled initial noise, then dumps every input + the output.

Inputs are intentionally small (H=W=64, 5 frames, 4 text tokens, mel
T=200) so the parity test runs in ~30s on M5 Max even with the full
48-layer DiT at depth=48. Initial noise is generated externally (numpy
seeded) so Python and Swift see identical noise, since MLX random state
isn't compatible across language bindings.

Usage:
    .venv/bin/python scripts/dump_pipeline_swift_fixtures.py \\
        --out /Users/dustinnielson/DEV_INT/longcat-avatar-mlx-swift/Tests/LongCatVideoAvatarTests/Resources/pipeline-parity/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

import mlx.core as mx
import numpy as np


# Small but exercises every code path in the pipeline.
HEIGHT = 64
WIDTH = 64
NUM_FRAMES = 5   # T_lat_full = 1 + (5-1)/4 + 1 (ref) = 3 latent frames total
T_MEL = 200      # 2s @ 100Hz mel rate
TEXT_TOKENS = 4
CAPTION_CHANNELS = 4096


def _seeded_normal(shape: tuple[int, ...], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape).astype(np.float32)


def _load_pipeline_components(repo_id: str):
    """Load all 4 components from HF and assemble the pipeline."""
    import json as _json
    import pathlib as _pl
    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )
    from longcat_video_avatar.models.umt5 import UMT5EncoderModel
    from longcat_video_avatar.models.whisper import WhisperEncoder

    # VAE
    print("  Loading VAE...")
    vae_cfg = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="vae/config.json")).read_text())
    vae = AutoencoderKLWan.from_config(vae_cfg, encoder=True)
    vae_w = hf_hub_download(repo_id=repo_id, filename="vae/diffusion_pytorch_model.safetensors")
    vae.load_weights(vae_w, strict=False)

    # umT5
    print("  Loading umT5...")
    umt5_cfg = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="text_encoder/config.json")).read_text())
    umt5 = UMT5EncoderModel.from_config(umt5_cfg)
    umt5_idx = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="text_encoder/model.safetensors.index.json")).read_text())
    for shard in sorted(set(umt5_idx["weight_map"].values())):
        umt5.load_weights(hf_hub_download(repo_id=repo_id, filename=f"text_encoder/{shard}"), strict=False)

    # Whisper
    print("  Loading Whisper...")
    whisp_cfg = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="audio_encoder/config.json")).read_text())
    whisper = WhisperEncoder.from_config(whisp_cfg)
    whisp_w = hf_hub_download(repo_id=repo_id, filename="audio_encoder/model.safetensors")
    whisper.load_weights(whisp_w, strict=False)

    # DiT
    print("  Loading Avatar DiT (this is the big one)...")
    dit_cfg = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="dit/config.json")).read_text())
    dit = LongCatVideoAvatarTransformer3DModel.from_config(dit_cfg)
    dit_idx = _json.loads(_pl.Path(hf_hub_download(repo_id=repo_id, filename="dit/diffusion_pytorch_model.safetensors.index.json")).read_text())
    for shard in sorted(set(dit_idx["weight_map"].values())):
        dit.load_weights(hf_hub_download(repo_id=repo_id, filename=f"dit/{shard}"), strict=False)

    mx.eval(vae.parameters(), umt5.parameters(), whisper.parameters(), dit.parameters())
    return vae, umt5, whisper, dit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
    )
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"Loading pipeline from {args.weights}...")
    vae, umt5, whisper, dit = _load_pipeline_components(args.weights)

    from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig
    cfg = PipelineConfig(num_frames=NUM_FRAMES)
    pipeline = LongCatAvatarPipeline(
        vae=vae, text_encoder=umt5, audio_encoder=whisper, dit=dit, config=cfg
    )

    # ---- Generate inputs ----
    print(f"Building inputs (H=W={HEIGHT}, T={NUM_FRAMES}, T_mel={T_MEL})")
    # Reference image in [-1, 1]. Shape [B=1, 3, T=1, H, W]
    image_np = np.clip(_seeded_normal((1, 3, 1, HEIGHT, WIDTH), seed=args.seed), -1, 1)
    # Mel features: Whisper-large-v3 has 128 mel bins
    mel_np = _seeded_normal((1, 128, T_MEL), seed=args.seed + 1)
    # Text embeddings + masks (would normally come from umT5 forward on a tokenized prompt;
    # we synthesize them so we don't drag tokenization into this fixture)
    text_np = _seeded_normal((1, 1, TEXT_TOKENS, CAPTION_CHANNELS), seed=args.seed + 2)
    text_mask_np = np.ones((1, TEXT_TOKENS), dtype=np.int32)
    uncond_np = _seeded_normal((1, 1, TEXT_TOKENS, CAPTION_CHANNELS), seed=args.seed + 3)
    uncond_mask_np = np.ones((1, TEXT_TOKENS), dtype=np.int32)

    # Initial noise — generated externally so Python and Swift see identical noise.
    # Shape: [B=1, 16, T_lat=2 noise frames, H/8=8, W/8=8]
    v = cfg.vae_scale_temporal
    T_lat_noise = 1 + (NUM_FRAMES - 1) // v   # = 2 for NUM_FRAMES=5
    noise_np = _seeded_normal((1, 16, T_lat_noise, HEIGHT // 8, WIDTH // 8), seed=args.seed + 4)
    print(f"  Initial noise shape: {noise_np.shape}")

    # ---- Run pipeline (monkey-patch _make_initial_noise to use our fixture) ----
    print("Running Python-MLX pipeline (this will take ~30-60s)...")
    pipeline._make_initial_noise = lambda *_args, **_kwargs: mx.array(noise_np)

    output = pipeline(
        image=mx.array(image_np),
        audio_mel=mx.array(mel_np),
        text_embeds=mx.array(text_np),
        text_mask=mx.array(text_mask_np),
        uncond_embeds=mx.array(uncond_np),
        uncond_mask=mx.array(uncond_mask_np),
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        seed=args.seed,
    )
    output = output.astype(mx.float32)
    mx.eval(output)
    out_np = np.asarray(output)
    print(f"  output: shape={out_np.shape}, dtype={out_np.dtype}, abs.max={np.abs(out_np).max():.4g}")

    # ---- Save fixtures ----
    paths = {
        "image.npy": image_np,
        "audio_mel.npy": mel_np,
        "text_embeds.npy": text_np,
        "text_mask.npy": text_mask_np,
        "uncond_embeds.npy": uncond_np,
        "uncond_mask.npy": uncond_mask_np,
        "initial_noise.npy": noise_np,
        "output.npy": out_np,
    }
    for name, arr in paths.items():
        p = args.out / name
        np.save(p, arr)
        print(f"  wrote {p}  ({p.stat().st_size:,} bytes)")

    def _sha256(p: pathlib.Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    manifest: dict[str, Any] = {
        "generator": "longcat-avatar-mlx/scripts/dump_pipeline_swift_fixtures.py",
        "source_weights": args.weights,
        "seed": args.seed,
        "config": {
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "T_mel": T_MEL,
            "text_tokens": TEXT_TOKENS,
            "caption_channels": CAPTION_CHANNELS,
        },
        "shapes": {name: list(arr.shape) for name, arr in paths.items()},
        "files": {name: {"sha256_16": _sha256(args.out / name)} for name in paths},
        "thresholds": {
            "# Avatar DiT one-pass measured 0.32. Full pipeline runs the DiT 3x": None,
            "# per step for 8 steps = 24 DiT passes, then VAE encode + decode.": None,
            "# Expect end-to-end divergence to compound to ~1-3 abs given 8 steps": None,
            "# of Avatar DiT drift + a VAE decode pass. Threshold conservative.": None,
            "output_max_abs": 5.0,
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    total = sum((args.out / name).stat().st_size for name in paths)
    print(f"\nTotal fixture size: {total:,} bytes ({total / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
