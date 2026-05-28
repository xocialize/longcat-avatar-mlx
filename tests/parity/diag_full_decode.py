"""Run the FULL decode on CPU stream. If this passes at strict 1e-3, the
implementation is correct and the GPU drift is purely Metal-fp32 precision.
"""
from __future__ import annotations
import json, os, pathlib, numpy as np
import torch, mlx.core as mx
from diffusers.models.autoencoders.autoencoder_kl_wan import AutoencoderKLWan as PTAutoencoderKLWan
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VAE_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--vae-config.json"

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

rng = np.random.default_rng(42)
z_np = rng.standard_normal((1, 16, 3, 16, 16)).astype(np.float32)

with torch.no_grad():
    pt_out = pt_vae.decode(torch.from_numpy(z_np)).sample
pt_np = pt_out.detach().cpu().float().numpy()

# MLX on GPU (default)
mx_out_gpu = mx_vae.decode(mx.array(z_np))
mx.eval(mx_out_gpu)
gpu_np = np.asarray(mx_out_gpu)
d_gpu = np.abs(pt_np - gpu_np)
print(f"GPU (default):  max={d_gpu.max():.3e}  mean={d_gpu.mean():.3e}  shape={pt_np.shape}")

# MLX on CPU stream
with mx.stream(mx.cpu):
    mx_out_cpu = mx_vae.decode(mx.array(z_np))
    mx.eval(mx_out_cpu)
cpu_np = np.asarray(mx_out_cpu)
d_cpu = np.abs(pt_np - cpu_np)
print(f"CPU stream:     max={d_cpu.max():.3e}  mean={d_cpu.mean():.3e}")

# Encode too
print("\n--- Encode ---")
v_np = np.clip(rng.standard_normal((1, 3, 9, 32, 32)).astype(np.float32), -1, 1)
with torch.no_grad():
    pt_z = pt_vae.encode(torch.from_numpy(v_np)).latent_dist.mean
    mean_t = torch.tensor(pt_vae.config.latents_mean).reshape(1, -1, 1, 1, 1)
    std_t = torch.tensor(pt_vae.config.latents_std).reshape(1, -1, 1, 1, 1)
    pt_z = (pt_z - mean_t) / std_t
pt_z_np = pt_z.detach().cpu().float().numpy()

mx_z_gpu = mx_vae.encode(mx.array(v_np)); mx.eval(mx_z_gpu)
d_enc_gpu = np.abs(pt_z_np - np.asarray(mx_z_gpu))
print(f"Encode GPU:     max={d_enc_gpu.max():.3e}  mean={d_enc_gpu.mean():.3e}")

with mx.stream(mx.cpu):
    mx_z_cpu = mx_vae.encode(mx.array(v_np))
    mx.eval(mx_z_cpu)
d_enc_cpu = np.abs(pt_z_np - np.asarray(mx_z_cpu))
print(f"Encode CPU:     max={d_enc_cpu.max():.3e}  mean={d_enc_cpu.mean():.3e}")
