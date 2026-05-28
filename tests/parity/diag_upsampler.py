"""Isolate up_blocks[1].upsamplers[0] divergence. Drive both with the SAME
fp32 input (from PT after resnets[2]) and compare:
  - Our existing MLX path on GPU
  - The same path on CPU stream
  - Spatial-only Conv2d on a hand-replicated input
"""

from __future__ import annotations
import json, os, pathlib, numpy as np
import torch, mlx.core as mx, mlx.nn as nn
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as PTAutoencoderKLWan
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VAE_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--vae-config.json"

def diff(name, pt, mx_arr):
    pt_np = pt.detach().cpu().float().numpy() if hasattr(pt, "detach") else np.asarray(pt)
    mx_np = np.asarray(mx_arr)
    if pt_np.shape != mx_np.shape:
        print(f"  ✗ {name}: SHAPE pt={pt_np.shape} mx={mx_np.shape}")
        return
    d = np.abs(pt_np - mx_np)
    print(f"  {'✓' if d.max() < 1e-3 else '✗'} {name:50s} max={d.max():.3e} mean={d.mean():.3e}")

cfg = json.loads(VAE_CONFIG_PATH.read_text())
pt_vae = PTAutoencoderKLWan(
    z_dim=cfg["z_dim"], base_dim=cfg["base_dim"], dim_mult=cfg["dim_mult"],
    num_res_blocks=cfg["num_res_blocks"], temperal_downsample=cfg["temperal_downsample"],
    attn_scales=cfg["attn_scales"], latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"],
)
weights = hf_hub_download(repo_id="meituan-longcat/LongCat-Video", filename="vae/diffusion_pytorch_model.safetensors")
sd = load_file(weights)
pt_vae.load_state_dict(sd, strict=False); pt_vae.eval()

mx_vae = AutoencoderKLWan.from_config(cfg, encoder=True)
from mlx.utils import tree_unflatten
flat = []
for k, v in sd.items():
    arr = v.detach().cpu().float().numpy()
    if "gamma" in k: pass
    elif arr.ndim == 5: arr = arr.transpose(0, 2, 3, 4, 1)
    elif arr.ndim == 4: arr = arr.transpose(0, 2, 3, 1)
    flat.append((k, mx.array(arr)))
mx_vae.update(tree_unflatten(flat)); mx.eval(mx_vae.parameters())

# Drive both decoders up to the input of up_blocks[1].upsamplers[0]
rng = np.random.default_rng(42)
z_np = rng.standard_normal((1, 16, 1, 16, 16)).astype(np.float32)

with torch.no_grad():
    x_pt = pt_vae.post_quant_conv(torch.from_numpy(z_np))
    x_pt = pt_vae.decoder.conv_in(x_pt)
    x_pt = pt_vae.decoder.mid_block(x_pt)
    x_pt = pt_vae.decoder.up_blocks[0](x_pt)
    x_pt = pt_vae.decoder.up_blocks[1].resnets[0](x_pt)
    x_pt = pt_vae.decoder.up_blocks[1].resnets[1](x_pt)
    x_pt = pt_vae.decoder.up_blocks[1].resnets[2](x_pt)  # shape (1, 384, 1, 32, 32)

# Use the SAME numpy array on both sides to eliminate any drift before the upsampler
x_pt_np = x_pt.detach().cpu().float().numpy()
x_mx = mx.array(x_pt_np)

# Reference: PT upsampler
with torch.no_grad():
    out_pt = pt_vae.decoder.up_blocks[1].upsamplers[0](x_pt)
print(f"PT upsampler output shape: {tuple(out_pt.shape)}")

# Our MLX upsampler on GPU (default)
upsampler = mx_vae.decoder.up_blocks[1].upsamplers[0]
out_mx_gpu = upsampler(x_mx)
mx.eval(out_mx_gpu)
diff("upsampler GPU (default)", out_pt, out_mx_gpu)

# Same on CPU stream
with mx.stream(mx.cpu):
    out_mx_cpu = upsampler(x_mx)
    mx.eval(out_mx_cpu)
diff("upsampler CPU stream", out_pt, out_mx_cpu)

# Manual spatial-only path
b, c, t, h, w = x_mx.shape
x = x_mx.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)  # (1, 32, 32, 384)
x_rep = mx.repeat(x, 2, axis=1)
x_rep = mx.repeat(x_rep, 2, axis=2)  # (1, 64, 64, 384)
out_conv = upsampler.resample[1](x_rep)
out_manual = out_conv.reshape(b, t, h * 2, w * 2, upsampler.dim // 2).transpose(0, 4, 1, 2, 3)
mx.eval(out_manual)
diff("upsampler manual GPU", out_pt, out_manual)

# Manual CPU
with mx.stream(mx.cpu):
    x_cpu = x_mx.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
    x_rep_cpu = mx.repeat(x_cpu, 2, axis=1)
    x_rep_cpu = mx.repeat(x_rep_cpu, 2, axis=2)
    out_conv_cpu = upsampler.resample[1](x_rep_cpu)
    out_manual_cpu = out_conv_cpu.reshape(b, t, h * 2, w * 2, upsampler.dim // 2).transpose(0, 4, 1, 2, 3)
    mx.eval(out_manual_cpu)
diff("upsampler manual CPU", out_pt, out_manual_cpu)

# Just the conv2d on PT-replicated upsample input (to isolate repeat vs conv2d)
with torch.no_grad():
    # Mimic the spatial pre-conv input that PT would produce internally
    x_pt_perm = x_pt.permute(0, 2, 1, 3, 4).reshape(1 * 1, 384, 32, 32)
    x_pt_ups = torch.nn.functional.interpolate(x_pt_perm, scale_factor=(2.0, 2.0), mode="nearest-exact")  # (1, 384, 64, 64)
    x_pt_conv_in = x_pt_ups.detach().cpu().float().numpy()

# Compare PT's "nearest-exact" output to our mx.repeat output
x_mx_repeat_only = mx.repeat(mx.repeat(x_mx.transpose(0, 2, 3, 4, 1).reshape(1, 32, 32, 384), 2, axis=1), 2, axis=2)
# x_mx_repeat_only is (1, 64, 64, 384); convert to PT layout (1, 384, 64, 64) for comparison
x_mx_repeat_pt_layout = x_mx_repeat_only.transpose(0, 3, 1, 2)
diff("nearest-exact (PT) vs mx.repeat (MLX)", torch.from_numpy(x_pt_conv_in), x_mx_repeat_pt_layout)

# Conv2d alone, with identical input on both sides
x_conv_in_np = x_pt_conv_in.transpose(0, 2, 3, 1)  # to channels-last (1, 64, 64, 384)
x_conv_in_mx = mx.array(x_conv_in_np)
out_conv_mx = upsampler.resample[1](x_conv_in_mx)
mx.eval(out_conv_mx)
# PT conv2d on the same input
with torch.no_grad():
    out_conv_pt = upsampler.resample[1](x_pt_ups)  # huh — PT has nn.Sequential, [1] is the Conv2d
# But wait, upsampler.resample is OUR list, not PT's. PT's is `pt_vae.decoder.up_blocks[1].upsamplers[0].resample[1]`
pt_upsampler = pt_vae.decoder.up_blocks[1].upsamplers[0]
with torch.no_grad():
    out_conv_pt = pt_upsampler.resample[1](x_pt_ups)
out_conv_mx_pt_layout = out_conv_mx.transpose(0, 3, 1, 2)
diff("Conv2d alone (identical input)", out_conv_pt, out_conv_mx_pt_layout)
