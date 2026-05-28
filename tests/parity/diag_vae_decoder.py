"""Bisect the decoder PT↔MLX divergence by running both side-by-side and
comparing intermediate values stage by stage.

Run: `LONGCAT_VAE_AUTO_DOWNLOAD=1 .venv/bin/python tests/parity/diag_vae_decoder.py`
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np

import mlx.core as mx
import torch
from diffusers.models.autoencoders.autoencoder_kl_wan import (
    AutoencoderKLWan as PTAutoencoderKLWan,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VAE_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--vae-config.json"
THRESHOLD = 1e-3


def diff(name: str, pt, mx_arr, threshold: float = THRESHOLD) -> bool:
    pt_np = pt.detach().cpu().float().numpy() if hasattr(pt, "detach") else np.asarray(pt)
    mx_np = np.asarray(mx_arr)
    if pt_np.shape != mx_np.shape:
        print(f"  ✗ {name}: SHAPE MISMATCH pt={pt_np.shape} vs mx={mx_np.shape}")
        return False
    d = np.abs(pt_np - mx_np)
    max_abs, mean_abs = float(d.max()), float(d.mean())
    ok = max_abs < threshold
    mark = "✓" if ok else "✗"
    print(f"  {mark} {name:45s} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} shape={tuple(pt_np.shape)}")
    return ok


def pt_to_mx(t: torch.Tensor) -> mx.array:
    return mx.array(t.detach().cpu().float().numpy())


def mx_to_pt(a: mx.array) -> torch.Tensor:
    mx.eval(a)
    return torch.from_numpy(np.asarray(a))


def main():
    cfg = json.loads(VAE_CONFIG_PATH.read_text())

    # Load PT VAE
    print("Loading PT model...")
    pt_vae = PTAutoencoderKLWan(
        z_dim=cfg["z_dim"],
        base_dim=cfg["base_dim"],
        dim_mult=cfg["dim_mult"],
        num_res_blocks=cfg["num_res_blocks"],
        temperal_downsample=cfg["temperal_downsample"],
        attn_scales=cfg["attn_scales"],
        latents_mean=cfg["latents_mean"],
        latents_std=cfg["latents_std"],
    )
    weights_path = (
        hf_hub_download(repo_id="meituan-longcat/LongCat-Video", filename="vae/diffusion_pytorch_model.safetensors")
        if os.environ.get("LONGCAT_VAE_AUTO_DOWNLOAD") == "1"
        else os.environ["LONGCAT_VAE_WEIGHTS_DIR"] + "/diffusion_pytorch_model.safetensors"
    )
    state_dict = load_file(weights_path)
    pt_vae.load_state_dict(state_dict, strict=False)
    pt_vae.eval()

    # Load MLX VAE (same weights)
    print("Loading MLX model with same weights...")
    mx_vae = AutoencoderKLWan.from_config(cfg, encoder=True)
    flat = []
    from mlx.utils import tree_unflatten

    for k, v in state_dict.items():
        arr = v.detach().cpu().float().numpy()
        if "gamma" in k:
            pass
        elif arr.ndim == 5:
            arr = arr.transpose(0, 2, 3, 4, 1)
        elif arr.ndim == 4:
            arr = arr.transpose(0, 2, 3, 1)
        flat.append((k, mx.array(arr)))
    mx_vae.update(tree_unflatten(flat))
    mx.eval(mx_vae.parameters())

    # ----- Test bench: single latent frame, no feat_cache --------------------
    rng = np.random.default_rng(42)
    z_np = rng.standard_normal((1, 16, 1, 16, 16)).astype(np.float32)

    print("\n========================================================")
    print("STAGE A — single latent frame, no feat_cache (spatial-only path)")
    print("========================================================")

    # 1. post_quant_conv (1×1×1 Conv3d, very simple)
    with torch.no_grad():
        x_pt = pt_vae.post_quant_conv(torch.from_numpy(z_np))
    x_mx = mx_vae.post_quant_conv(mx.array(z_np))
    diff("post_quant_conv", x_pt, x_mx)

    # 2. decoder.conv_in (3D 3×3×3 Conv, no cache → uses internal zero pad)
    with torch.no_grad():
        x_pt2 = pt_vae.decoder.conv_in(x_pt)
    x_mx2 = mx_vae.decoder.conv_in(x_mx)
    diff("decoder.conv_in", x_pt2, x_mx2)

    # 3. decoder.mid_block step-by-step (resnet, attention, resnet)
    with torch.no_grad():
        x_pt3a = pt_vae.decoder.mid_block.resnets[0](x_pt2)
    x_mx3a = mx_vae.decoder.mid_block.resnets[0](x_mx2)
    diff("decoder.mid_block.resnets[0]", x_pt3a, x_mx3a)

    with torch.no_grad():
        x_pt3b = pt_vae.decoder.mid_block.attentions[0](x_pt3a)
    x_mx3b = mx_vae.decoder.mid_block.attentions[0](x_mx3a)
    diff("decoder.mid_block.attentions[0]", x_pt3b, x_mx3b)

    with torch.no_grad():
        x_pt3 = pt_vae.decoder.mid_block.resnets[1](x_pt3b)
    x_mx3 = mx_vae.decoder.mid_block.resnets[1](x_mx3b)
    diff("decoder.mid_block.resnets[1]", x_pt3, x_mx3)

    # 4. Each up_block (no feat_cache → skips temporal upsample)
    pt_cur, mx_cur = x_pt3, x_mx3
    for i, (pt_up, mx_up) in enumerate(zip(pt_vae.decoder.up_blocks, mx_vae.decoder.up_blocks)):
        # Bisect within up_block: per-resnet, then upsampler if present
        pt_inner, mx_inner = pt_cur, mx_cur
        for ri, (pt_r, mx_r) in enumerate(zip(pt_up.resnets, mx_up.resnets)):
            with torch.no_grad():
                pt_inner = pt_r(pt_inner)
            mx_inner = mx_r(mx_inner)
            ok = diff(f"up_blocks[{i}].resnets[{ri}]", pt_inner, mx_inner)
            if not ok:
                print(f"\n    First divergence at up_blocks[{i}].resnets[{ri}].")
                return
        if pt_up.upsamplers is not None:
            with torch.no_grad():
                pt_inner = pt_up.upsamplers[0](pt_inner)
            mx_inner = mx_up.upsamplers[0](mx_inner)
            ok = diff(f"up_blocks[{i}].upsamplers[0]", pt_inner, mx_inner)
            if not ok:
                # Diagnose: was it the spatial conv2d, or the repeat, or both?
                # Re-run upsampler manually with CPU stream to test GPU-precision hypothesis.
                # We need to reconstruct the call. The upsampler is Resample with mode='upsample3d'
                # and (feat_cache=None), so it just does spatial 2x + conv2d.
                upsampler = mx_up.upsamplers[0]
                # The input to upsamplers[0] is mx_cur (output of last resnet, before this call).
                # Recompute that here:
                # Actually, we mutated mx_inner above. Re-run from saved input.
                pass
            if not ok:
                print(f"\n    First divergence at up_blocks[{i}].upsamplers[0].")
                return
        pt_cur, mx_cur = pt_inner, mx_inner

    # 5. decoder.norm_out + silu + conv_out
    with torch.no_grad():
        x_pt5 = torch.nn.functional.silu(pt_vae.decoder.norm_out(pt_cur))
        out_pt = pt_vae.decoder.conv_out(x_pt5)
    import mlx.nn as nn

    x_mx5 = nn.silu(mx_vae.decoder.norm_out(mx_cur))
    out_mx = mx_vae.decoder.conv_out(x_mx5)
    diff("decoder.norm_out + silu", x_pt5, x_mx5)
    diff("decoder.conv_out (final)", out_pt, out_mx)

    print("\n========================================================")
    print("STAGE B — chunked decode w/ feat_cache (full per-frame path)")
    print("========================================================")
    # Run full chunked decode on 3 latent frames; compare final outputs
    z_full = rng.standard_normal((1, 16, 3, 16, 16)).astype(np.float32)
    with torch.no_grad():
        # Use diffusers' _decode to drive the canonical chunked path
        out_pt_full = pt_vae._decode(torch.from_numpy(z_full), return_dict=False)[0]
    out_mx_full = mx_vae.decode(mx.array(z_full))
    # mx_vae.decode applies de-normalization first; need to compare against PT
    # _decode which also applies post_quant_conv but NOT denormalization
    # Actually mx_vae.decode does: z / inv_std + mean → post_quant_conv → decoder
    # And pt_vae._decode does: post_quant_conv(z) → decoder (NO denorm)
    # Let me bypass: use mx_vae.decoder + mx_vae.post_quant_conv directly to match.
    print("  (note: stage B comparison is in normalized-vs-raw latent space —")
    print("   use the stage-A path for cleanest bisection)")


if __name__ == "__main__":
    main()
