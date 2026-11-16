"""KV cache for the conditioning latents in video continuation (Meituan's `use_kv_cache`).

In continuation the DiT input is [reference latent, 4 latents of the last 13 frames, 20 noisy latents].
The first 5 have timestep 0, get no text/audio cross-attention (both return zeros on that span) and in
self-attention only see themselves (reference and conditioning separately). So their per-block keys and
values don't change across denoising steps as long as the active LoRA doesn't change: compute them once
(`build_cache`) and run each step on the noisy latents only (`forward_noise`). The prediction matches the
full forward (tests/smoke/test_continuation_smoke.py) with 1/5 fewer tokens.

Unlike Meituan's cache path, cross-attention is not skipped on the noisy tokens, and the cache can be
rebuilt when 1.6's DMD -> OPSD LoRA switch happens, so it stays exact with LoRA mixing too. Block functions
go through `fast.compiled`, so they are compiled when the DiT was mounted with `fast.apply(compile=True)`.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.fast import compiled, rope_tables, rotate
from longcat_video_avatar.models.blocks import modulate_fp32


def _rope(attn, q, k, grid, frame_index, num_ref, start):
    cos, sin = (f[start:start + q.shape[2]] for f in rope_tables(attn.rope_3d, grid, frame_index, num_ref))
    return rotate(q, cos, sin), rotate(k, cos, sin)


def _qkv(attn, x):
    B, N, _ = x.shape
    qkv = attn.qkv(x).reshape(B, N, 3, attn.num_heads, attn.head_dim).transpose(2, 0, 3, 1, 4)
    return attn.q_norm(qkv[0]), attn.k_norm(qkv[1]), qkv[2]


def _ada(block, t):
    ada = block.adaLN_modulation[1](nn.silu(t)).astype(mx.float32)[:, :, None, :]
    return mx.split(ada, 6, axis=-1)


def _ffn(block, x, T, shift_mlp, scale_mlp, gate_mlp):
    B, N, C = x.shape
    x_m = modulate_fp32(block.mod_norm_ffn, x.reshape(B, T, -1, C), shift_mlp, scale_mlp).reshape(B, N, C)
    x_s = block.ffn(x_m)
    return (x.astype(mx.float32) + (gate_mlp.astype(mx.float32) * x_s.reshape(B, T, -1, C).astype(mx.float32))
            .reshape(B, N, C)).astype(x.dtype)


def _cond_block(block, x, t, grid, num_ref, ref_img_index):
    """One block on the conditioning latents only: returns their new state and the keys/values to keep."""
    B, N, C = x.shape
    T_c = t.shape[1]
    tpf = grid[1] * grid[2]
    sh, sc, g, sh2, sc2, g2 = _ada(block, t)
    x_m = modulate_fp32(block.mod_norm_attn, x.reshape(B, T_c, -1, C), sh, sc).reshape(B, N, C)
    attn = block.attn
    q, k, v = _qkv(attn, x_m)
    q, k = _rope(attn, q, k, grid, ref_img_index, num_ref, 0)
    x_ref = attn._process_attn(q[:, :, :tpf * num_ref], k[:, :, :tpf * num_ref], v[:, :, :tpf * num_ref])
    x_cond = attn._process_attn(q[:, :, tpf * num_ref:], k[:, :, tpf * num_ref:], v[:, :, tpf * num_ref:])
    out = attn.proj(mx.concatenate([x_ref, x_cond], axis=2).transpose(0, 2, 1, 3).reshape(B, N, C))
    x = (x.astype(mx.float32) + (g.astype(mx.float32) * out.reshape(B, T_c, -1, C).astype(mx.float32))
         .reshape(B, N, C)).astype(x.dtype)
    return _ffn(block, x, T_c, sh2, sc2, g2), k, v


def build_cache(dit, cond_latents, H_lat, W_lat, T_total, ref_img_index, num_ref=1):
    """cond_latents: [1, 16, T_c, H, W] (reference + conditioning). Returns [(k, v)] per block."""
    B, _, T_c, _, _ = cond_latents.shape
    dtype = dit.x_embedder.proj.weight.dtype
    x = dit.x_embedder(cond_latents.astype(dtype))
    t = dit.t_embedder(mx.zeros((B * T_c,), mx.float32), dtype=mx.float32).reshape(B, T_c, -1)
    grid = (T_total, H_lat // 2, W_lat // 2)
    f = compiled(dit, "cond", _cond_block)
    cache = []
    for block in dit.blocks:
        x, k, v = f(block, x, t, consts=(grid, num_ref, ref_img_index))
        cache.append((k, v))
        mx.eval(x, k, v)
    return cache


def _noise_block(block, x, y, t, audio_hs, k_c, v_c, y_seqlens, grid, num_ref, ref_img_index):
    """One block on the noisy latents only, attending to the cached conditioning keys/values."""
    B, N, C = x.shape
    T_n = t.shape[1]
    shape = (T_n, grid[1], grid[2])
    start = (grid[0] - T_n) * grid[1] * grid[2]
    x_dtype = x.dtype
    sh, sc, g, sh2, sc2, g2 = _ada(block, t)
    x_m = modulate_fp32(block.mod_norm_attn, x.reshape(B, T_n, -1, C), sh, sc).reshape(B, N, C)
    attn = block.attn
    q, k, v = _qkv(attn, x_m)
    q, k = _rope(attn, q, k, grid, ref_img_index, num_ref, start)
    o = attn._process_attn(q, mx.concatenate([k_c, k], axis=2), mx.concatenate([v_c, v], axis=2))
    o = attn.proj(o.transpose(0, 2, 1, 3).reshape(B, N, C))
    x = (x.astype(mx.float32) + (g.astype(mx.float32) * o.reshape(B, T_n, -1, C).astype(mx.float32))
         .reshape(B, N, C)).astype(x_dtype)
    x = x + block.cross_attn(block.pre_crs_attn_norm(x), y, list(y_seqlens), num_cond_latents=None, shape=shape)
    a_ada = block.audio_adaLN_modulation[1](nn.silu(t)).astype(mx.float32)[:, :, None, :]
    a_sh, a_sc, a_g = mx.split(a_ada, 3, axis=-1)
    _, a_out = block.audio_cross_attn(block.pre_video_crs_attn_norm(x), block._maybe_pre_audio_norm(audio_hs),
                                      shape=shape, num_cond_latents=0, x_ref_attn_map=None, human_num=None)
    a_out = modulate_fp32(block.mod_norm_attn, a_out.reshape(B, T_n, -1, C), a_sh, a_sc).reshape(B, -1, C)
    add = (a_g.astype(mx.float32) * a_out.reshape(B, T_n, -1, C).astype(mx.float32)).reshape(B, -1, C)
    x = (x.astype(mx.float32) + add).astype(x_dtype)
    return _ffn(block, x, T_n, sh2, sc2, g2)


def forward_noise(dit, latents, t_value, text, text_mask, audio_embs, cache, T_total, ref_img_index, num_ref=1):
    """DiT prediction for the noisy latents only, attending to the cached conditioning keys/values."""
    B, _, T_n, H, W = latents.shape
    N_h, N_w = H // 2, W // 2
    dtype = dit.x_embedder.proj.weight.dtype
    x = dit.x_embedder(latents.astype(dtype))
    t = dit.t_embedder(mx.full((B * T_n,), t_value, mx.float32).astype(dtype).astype(mx.float32),
                       dtype=mx.float32).reshape(B, T_n, -1)
    y = dit.y_embedder(text.astype(dtype))
    # text packing as in the DiT forward (text_tokens_zero_pad)
    mask = text_mask.reshape(B, -1)
    if dit.text_tokens_zero_pad:
        y = y * mask[:, None, :, None]
        mask = mx.ones_like(mask)
    y_seqlens = [int(mx.sum(mask[b]).item()) for b in range(B)]
    ehs = y.squeeze(1)
    y = mx.concatenate([ehs[b, :y_seqlens[b]] for b in range(B)], axis=0)[None]
    audio_hs = dit._prepare_audio_hidden_states(audio_embs.astype(dtype), num_ref, T_n)
    audio_hs = audio_hs.reshape(B * T_n, audio_hs.shape[2], audio_hs.shape[3])
    f = compiled(dit, "noise", _noise_block)
    consts = (tuple(y_seqlens), (T_total, N_h, N_w), num_ref, ref_img_index)
    for block, (k_c, v_c) in zip(dit.blocks, cache):
        x = f(block, x, y, t, audio_hs, k_c, v_c, consts=consts)
    x = dit.final_layer(x, t, (T_n, N_h, N_w))
    return dit._unpatchify(x, T_n, N_h, N_w).astype(mx.float32)
