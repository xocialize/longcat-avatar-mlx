"""Drill into decoder.mid_block.attentions[0]: feed identical input to both
PT and MLX, check each sub-step (norm, to_qkv, SDP, proj, residual).
"""

from __future__ import annotations
import json
import os
import pathlib
import numpy as np
import torch
import mlx.core as mx
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
    print(f"  {'✓' if d.max() < 1e-3 else '✗'} {name:40s} max={d.max():.3e}  mean={d.mean():.3e}  shape={tuple(pt_np.shape)}")


cfg = json.loads(VAE_CONFIG_PATH.read_text())
pt_vae = PTAutoencoderKLWan(
    z_dim=cfg["z_dim"], base_dim=cfg["base_dim"], dim_mult=cfg["dim_mult"],
    num_res_blocks=cfg["num_res_blocks"], temperal_downsample=cfg["temperal_downsample"],
    attn_scales=cfg["attn_scales"], latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"],
)
weights = hf_hub_download(repo_id="meituan-longcat/LongCat-Video", filename="vae/diffusion_pytorch_model.safetensors")
sd = load_file(weights)
pt_vae.load_state_dict(sd, strict=False)
pt_vae.eval()

mx_vae = AutoencoderKLWan.from_config(cfg, encoder=True)
from mlx.utils import tree_unflatten
flat = []
for k, v in sd.items():
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

# Pick an input that the attention sees in real inference.
rng = np.random.default_rng(42)
z_np = rng.standard_normal((1, 16, 1, 16, 16)).astype(np.float32)
x_pt = pt_vae.post_quant_conv(torch.from_numpy(z_np))
x_pt = pt_vae.decoder.conv_in(x_pt)
x_pt = pt_vae.decoder.mid_block.resnets[0](x_pt)  # [1, 384, 1, 16, 16]
x_mx = mx_vae.post_quant_conv(mx.array(z_np))
x_mx = mx_vae.decoder.conv_in(x_mx)
x_mx = mx_vae.decoder.mid_block.resnets[0](x_mx)

print("\n--- input to attention block ---")
diff("attention input", x_pt, x_mx)

pt_attn = pt_vae.decoder.mid_block.attentions[0]
mx_attn = mx_vae.decoder.mid_block.attentions[0]

# Step 1: permute + reshape (B,C,T,H,W) -> (B*T, C, H, W)
b_pt, c_pt, t_pt, h_pt, w_pt = x_pt.shape
x_pt_r = x_pt.permute(0, 2, 1, 3, 4).reshape(b_pt * t_pt, c_pt, h_pt, w_pt)
# Our MLX does the same internally.
# Step 2: norm
x_pt_n = pt_attn.norm(x_pt_r)
# MLX equivalent: extract intermediate manually
b, c, t, h, w = x_mx.shape
x_mx_r = x_mx.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
x_mx_n = mx_attn.norm(x_mx_r)
diff("after norm (CF order)", x_pt_n, x_mx_n)

# Step 3: to_qkv
# PT: x_pt_n is (B*T, C, H, W) - channels first.
# Our MLX needs (B*T, H, W, C) for Conv2d.
x_mx_n_cl = x_mx_n.transpose(0, 2, 3, 1)
qkv_pt = pt_attn.to_qkv(x_pt_n)  # (B*T, 3C, H, W)
qkv_mx = mx_attn.to_qkv(x_mx_n_cl)  # (B*T, H, W, 3C)
# Transpose MLX to PT layout for comparison
qkv_mx_pt_layout = qkv_mx.transpose(0, 3, 1, 2)
diff("after to_qkv", qkv_pt, qkv_mx_pt_layout)

# Step 4: SDP call. Use PT's exact reshape pattern.
batch_size_pt = qkv_pt.shape[0]
channels_pt = c_pt
qkv_pt_flat = qkv_pt.reshape(batch_size_pt, 1, channels_pt * 3, -1).permute(0, 1, 3, 2).contiguous()
q_pt, k_pt, v_pt = qkv_pt_flat.chunk(3, dim=-1)
with torch.no_grad():
    sdp_pt = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt)

# Our MLX path
qkv_mx_flat = qkv_mx.reshape(b * t, h * w, 3, c).transpose(2, 0, 1, 3)
q_mx, k_mx, v_mx = qkv_mx_flat[0], qkv_mx_flat[1], qkv_mx_flat[2]
diff("q", q_pt.squeeze(1), q_mx)
diff("k", k_pt.squeeze(1), k_mx)
diff("v", v_pt.squeeze(1), v_mx)

q_mx_h = q_mx[:, None, :, :]
k_mx_h = k_mx[:, None, :, :]
v_mx_h = v_mx[:, None, :, :]
sdp_mx = mx.fast.scaled_dot_product_attention(q_mx_h, k_mx_h, v_mx_h, scale=c ** -0.5)
diff("SDP (mx.fast)", sdp_pt, sdp_mx)

# Manual SDP for comparison
import mlx.nn as nn
scale = c ** -0.5
qk_mx = (q_mx_h @ k_mx_h.transpose(0, 1, 3, 2)) * scale
attn_mx = mx.softmax(qk_mx, axis=-1)
sdp_manual = attn_mx @ v_mx_h
diff("SDP (mlx manual)", sdp_pt, sdp_manual)

# Same SDP in pure numpy fp32 (gold reference)
q_np = np.asarray(q_mx_h).astype(np.float32)
k_np = np.asarray(k_mx_h).astype(np.float32)
v_np = np.asarray(v_mx_h).astype(np.float32)
qk_np = (q_np @ k_np.transpose(0, 1, 3, 2)) * scale
qk_np_max = qk_np.max(axis=-1, keepdims=True)
attn_np = np.exp(qk_np - qk_np_max)
attn_np = attn_np / attn_np.sum(axis=-1, keepdims=True)
sdp_np = attn_np @ v_np
diff("SDP (numpy fp32)", sdp_pt, sdp_np)
diff("SDP (numpy vs mlx)", sdp_np, sdp_manual)

# Force PT to use the math backend (no flash, no mem-efficient)
import torch.nn.attention as ta
with ta.sdpa_kernel(ta.SDPBackend.MATH):
    sdp_pt_math = torch.nn.functional.scaled_dot_product_attention(q_pt, k_pt, v_pt)
diff("SDP (pt MATH backend)", sdp_pt_math, sdp_np)
sdp_pt_math_d = sdp_pt_math.detach()
diff("SDP (pt MATH vs default)", sdp_pt_math_d, sdp_pt)

# Try MLX on CPU stream — does fp32 then match PT?
with mx.stream(mx.cpu):
    qk_cpu = (q_mx_h @ k_mx_h.transpose(0, 1, 3, 2)) * scale
    attn_cpu = mx.softmax(qk_cpu, axis=-1)
    sdp_cpu = attn_cpu @ v_mx_h
    mx.eval(sdp_cpu)
diff("SDP (mlx CPU)", sdp_pt, sdp_cpu)
diff("SDP (mlx CPU vs GPU)", sdp_manual, sdp_cpu)

# Cast to fp64 on GPU as a precision test
q_64 = q_mx_h.astype(mx.float64)
k_64 = k_mx_h.astype(mx.float64)
v_64 = v_mx_h.astype(mx.float64)
qk_64 = (q_64 @ k_64.transpose(0, 1, 3, 2)) * scale
attn_64 = mx.softmax(qk_64, axis=-1)
sdp_64 = (attn_64 @ v_64).astype(mx.float32)
diff("SDP (mlx fp64)", sdp_pt, sdp_64)

# Step 5: full attention forward
with torch.no_grad():
    out_pt = pt_attn(x_pt)
out_mx = mx_attn(x_mx)
diff("attention(x) full", out_pt, out_mx)
