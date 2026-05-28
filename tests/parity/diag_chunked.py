"""Per-iteration bisection of the chunked decode loop."""
from __future__ import annotations
import json, os, pathlib, numpy as np
import torch, mlx.core as mx
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
    print(f"  {'✓' if d.max() < 5e-3 else '✗'} {name:40s} max={d.max():.3e} mean={d.mean():.3e} shape={tuple(pt_np.shape)}")

cfg = json.loads(VAE_CONFIG_PATH.read_text())
pt_vae = PTAutoencoderKLWan(
    z_dim=cfg["z_dim"], base_dim=cfg["base_dim"], dim_mult=cfg["dim_mult"],
    num_res_blocks=cfg["num_res_blocks"], temperal_downsample=cfg["temperal_downsample"],
    attn_scales=cfg["attn_scales"], latents_mean=cfg["latents_mean"], latents_std=cfg["latents_std"],
)
weights = hf_hub_download(repo_id="meituan-longcat/LongCat-Video", filename="vae/diffusion_pytorch_model.safetensors")
sd = load_file(weights); pt_vae.load_state_dict(sd, strict=False); pt_vae.eval()

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

# Drive PT via its _decode chunked loop, capture per-iteration outputs
rng = np.random.default_rng(42)
z_np = rng.standard_normal((1, 16, 3, 16, 16)).astype(np.float32)

# PT side: replicate _decode loop, capturing each chunk's output
pt_vae.clear_cache()
x_pt = pt_vae.post_quant_conv(torch.from_numpy(z_np))
pt_chunks = []
for i in range(3):
    pt_vae._conv_idx = [0]
    with torch.no_grad():
        if i == 0:
            out = pt_vae.decoder(x_pt[:, :, i:i+1], feat_cache=pt_vae._feat_map, feat_idx=pt_vae._conv_idx, first_chunk=True)
        else:
            out = pt_vae.decoder(x_pt[:, :, i:i+1], feat_cache=pt_vae._feat_map, feat_idx=pt_vae._conv_idx)
    pt_chunks.append(out.clone())

# MLX side: same loop, our model
mx_z = mx.array(z_np)
mean = mx_vae.mean.reshape(1, -1, 1, 1, 1); inv_std = mx_vae.inv_std.reshape(1, -1, 1, 1, 1)
# Note: our decode does z = z / inv_std + mean first. But PT _decode operates on raw z.
# To match PT exactly, skip the de-norm step and call post_quant_conv directly:
x_mx = mx_vae.post_quant_conv(mx_z)

num_slots = mx_vae._count_decoder_cache_slots()
feat_cache = [None] * num_slots
mx_chunks = []
for i in range(3):
    feat_idx = [0]
    chunk = x_mx[:, :, i:i+1]
    out_mx = mx_vae.decoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
    mx.eval(out_mx)
    mx_chunks.append(out_mx)
    print(f"--- After chunk {i} ---")
    print(f"  PT chunk shape: {tuple(pt_chunks[i].shape)}, MX chunk shape: {tuple(out_mx.shape)}")
    diff(f"chunk {i} output", pt_chunks[i], out_mx)
    # Show slot count and feat_cache state
    rep_slots = sum(1 for s in feat_cache if isinstance(s, str) and s == "Rep")
    arr_slots = sum(1 for s in feat_cache if not isinstance(s, (str, type(None))))
    none_slots = sum(1 for s in feat_cache if s is None)
    print(f"  feat_cache: {rep_slots} Rep, {arr_slots} arrays, {none_slots} None  (total {num_slots} slots)")
    pt_rep_slots = sum(1 for s in pt_vae._feat_map if isinstance(s, str) and s == "Rep")
    pt_arr_slots = sum(1 for s in pt_vae._feat_map if not isinstance(s, (str, type(None))))
    pt_none_slots = sum(1 for s in pt_vae._feat_map if s is None)
    print(f"  PT feat_cache: {pt_rep_slots} Rep, {pt_arr_slots} arrays, {pt_none_slots} None (total {pt_vae._conv_num} slots)")
