"""Smoke tests for video continuation: the ref/cond attention split, the conditioning KV cache and
runtime LoRAs. Tiny random weights, no downloads, a few seconds on CPU."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from longcat_video_avatar.kv_cache import build_cache, forward_noise
from longcat_video_avatar.lora import compute_merged_delta, group_lora_tensors
from longcat_video_avatar.models.avatar.attention import Attention as AvatarAttention
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video_avatar.runtime_lora import LoRALinear, attach_lora, set_lora_multipliers

TINY = {
    "in_channels": 4, "out_channels": 4, "hidden_size": 64, "depth": 2, "num_heads": 4, "caption_channels": 32,
    "mlp_ratio": 4, "adaln_tembed_dim": 32, "frequency_embedding_size": 32, "patch_size": [1, 2, 2],
    "text_tokens_zero_pad": True, "audio_window": 5, "audio_block": 5, "audio_channel": 32,
    "intermediate_dim": 32, "output_dim": 32, "context_tokens": 8, "vae_scale": 4, "audio_prenorm": False,
    "class_range": 24, "class_interval": 4,
}


def _rand(shape, seed):
    return mx.array(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def test_reference_and_conditioning_attend_separately_without_mask():
    """Continuation with mask_frame_range=0 (Meituan's default): reference tokens must not see the
    conditioning tokens and vice versa (Meituan attention.py:149-160)."""
    attn = AvatarAttention(dim=64, num_heads=4)
    shape, tpf = (6, 2, 2), 4  # 1 ref + 2 cond + 3 noisy latents, 4 tokens each
    x = _rand((1, 24, 64), 0)
    kw = dict(shape=shape, num_cond_latents=3, num_ref_latents=1, ref_img_index=4, mask_frame_range=0)
    out, _ = attn(x, **kw)
    x2 = x.at[:, tpf:3 * tpf].add(_rand((1, 2 * tpf, 64), 1))  # change only the conditioning tokens
    out2, _ = attn(x2, **kw)
    assert float(mx.abs(out[:, :tpf] - out2[:, :tpf]).max()) < 1e-6  # reference output unchanged
    assert float(mx.abs(out[:, 3 * tpf:] - out2[:, 3 * tpf:]).max()) > 1e-4  # noisy tokens see them


def _tiny_dit():
    mx.random.seed(0)
    dit = LongCatVideoAvatarTransformer3DModel.from_config(TINY)
    mx.eval(dit.parameters())
    return dit


def test_kv_cache_matches_full_forward():
    dit = _tiny_dit()
    T, H, W, nc = 6, 4, 4, 3  # ref + 2 cond + 3 noisy latents
    lat = _rand((1, 4, T, H, W), 2)
    audio = _rand((1, 1 + 4 * (T - 2), 5, 5, 32), 3)  # audio frames for the 5 non-ref latents
    text, mask = _rand((1, 1, 6, 32), 4), mx.array([[1, 1, 1, 1, 0, 0]])
    t = 700.0
    ts = mx.full((1, T), t)
    ts[:, :nc] = 0
    full = dit(lat, ts, text, audio, encoder_attention_mask=mask, num_cond_latents=nc, num_ref_latents=1,
               ref_img_index=4, mask_frame_range=0)[:, :, nc:]
    cache = build_cache(dit, lat[:, :, :nc], H, W, T, 4, 1)
    part = forward_noise(dit, lat[:, :, nc:], t, text, mask, audio, cache, T, 4, 1)
    mx.eval(full, part)
    assert part.shape == full.shape
    assert float(mx.abs(full - part).max()) < 1e-4


def test_runtime_lora_matches_merged_delta():
    dit = _tiny_dit()
    rank, d = 2, 64
    name = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv"
    sd = {f"{name}.lora_down.weight": _rand((3 * rank, d), 5),
          **{f"{name}.lora_up.blocks.{i}.weight": _rand((d, rank), 6 + i) for i in range(3)},
          f"{name}.alpha_scale": mx.array(0.5)}
    path = "/tmp/longcat_test_lora.safetensors"
    mx.save_safetensors(path, sd)
    x = _rand((1, 5, d), 9)
    assert attach_lora(dit, path, "user", dtype=mx.float32) == 1
    lin = dit.blocks[0].attn.qkv
    assert isinstance(lin, LoRALinear)
    off = lin(x)
    set_lora_multipliers(dit, {"user": 0.8})
    on = lin(x)
    delta = compute_merged_delta(group_lora_tensors(sd)["blocks.0.attn.qkv"], multiplier=0.8)
    expected = x @ delta.T  # the LoRA's contribution, as if merged into the weights
    got = (on - off).astype(mx.float32)
    rel = float(mx.abs(got - expected).max()) / float(mx.abs(expected).max())
    assert rel < 2e-2, rel  # base weights are bf16 in the model, the sum is rounded once
    set_lora_multipliers(dit, {"user": 0.0})
    assert float(mx.abs(lin(x) - off).max()) == 0.0
