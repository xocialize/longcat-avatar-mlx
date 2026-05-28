"""MLX port of LongCat-Video DiT base attention layers.

PyTorch reference: `refs/longcat-video/longcat_video/modules/attention.py`.

Two classes:
- `Attention` — visual self-attention with QKNorm + 3D RoPE. Supports the
  `num_cond_latents` branching for image-to-video / video-continuation modes.
  Returns optional KV cache for long-video continuation.
- `MultiHeadCrossAttention` — text cross-attention. Handles the
  variable-length-text packed format: text tokens for the whole batch are
  concatenated into a single sequence `[1, N_valid_total, C]` with
  `kv_seqlen=[N_per_batch_i]`. We build a block-diagonal mask for SDPA.

All CUDA-only attention backends (FlashAttention v2/v3, xformers, BSA, ulysses)
are replaced with `mx.fast.scaled_dot_product_attention` (dense). Per
CLAUDE.md L10 expect ~1e-3 max_abs drift vs PT-CPU on Metal-GPU fp32.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.models.blocks import RMSNorm_FP32
from longcat_video_avatar.models.rope_3d import RotaryPositionalEmbedding


class Attention(nn.Module):
    """Visual self-attention with QKNorm + 3D RoPE.

    Matches base `modules/attention.py:Attention`. Avatar overlay adds
    Reference Skip Q-slicing in `models/avatar/attention.py:Attention`
    (Stage 1.6).
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        # Fused QKV projection (matches PT key name).
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.proj = nn.Linear(dim, dim)

        self.rope_3d = RotaryPositionalEmbedding(self.head_dim)

    def _process_attn(self, q: mx.array, k: mx.array, v: mx.array) -> mx.array:
        """Dense SDPA wrapper. PT version branches over flash/bsa/xformers — we
        only have dense on Metal. q/k/v: [B, H, S, D]. Returns [B, H, S_q, D].
        """
        return mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)

    def __call__(
        self,
        x: mx.array,
        shape: tuple[int, int, int],
        num_cond_latents: Optional[int] = None,
        return_kv: bool = False,
    ):
        """x: [B, N, C], shape: (T, H, W). Returns x of same shape (+ optional KV cache)."""
        B, N, C = x.shape

        qkv = self.qkv(x)
        # [B, N, 3*C] -> [B, N, 3, H, D] -> [3, B, H, N, D]
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)

        if return_kv:
            k_cache, v_cache = k, v  # MLX arrays are immutable; no need to clone

        q, k = self.rope_3d(q, k, shape)

        if num_cond_latents is not None and num_cond_latents > 0:
            # Image-to-video / video-continuation: process conditioning tokens
            # separately from noise tokens (matches PT line 124-135).
            tokens_per_frame = N // shape[0]
            ncl_thw = num_cond_latents * tokens_per_frame

            q_cond = q[:, :, :ncl_thw]
            k_cond = k[:, :, :ncl_thw]
            v_cond = v[:, :, :ncl_thw]
            x_cond = self._process_attn(q_cond, k_cond, v_cond)

            q_noise = q[:, :, ncl_thw:]
            x_noise = self._process_attn(q_noise, k, v)

            out = mx.concatenate([x_cond, x_noise], axis=2)
        else:
            out = self._process_attn(q, k, v)

        # [B, H, N, D] -> [B, N, H, D] -> [B, N, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)

        if return_kv:
            return out, (k_cache, v_cache)
        return out

    def forward_with_kv_cache(
        self,
        x: mx.array,
        shape: tuple[int, int, int],
        num_cond_latents: int,
        kv_cache: tuple[mx.array, mx.array],
    ) -> mx.array:
        """Chunked continuation path. Mirrors PT `forward_with_kv_cache`.

        Concatenates fresh K/V onto cached K/V, applies RoPE over the extended
        temporal grid, processes attention with the noise queries only.
        """
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)

        T, H, W = shape
        k_cache, v_cache = kv_cache
        # Broadcast cache if it was stored at B=1 but inference is B>1
        if k_cache.shape[0] == 1 and B > 1:
            k_cache = mx.broadcast_to(k_cache, (B, *k_cache.shape[1:]))
            v_cache = mx.broadcast_to(v_cache, (B, *v_cache.shape[1:]))

        if num_cond_latents is not None and num_cond_latents > 0:
            k_full = mx.concatenate([k_cache, k], axis=2)
            v_full = mx.concatenate([v_cache, v], axis=2)
            # Pad q with zeros for the cache positions so RoPE indices align.
            q_padding = mx.concatenate([mx.zeros_like(k_cache), q], axis=2)
            q_padding, k_full = self.rope_3d(q_padding, k_full, (T + num_cond_latents, H, W))
            q = q_padding[:, :, -N:]
        else:
            k_full = mx.concatenate([k_cache, k], axis=2)
            v_full = mx.concatenate([v_cache, v], axis=2)

        out = self._process_attn(q, k_full, v_full)
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)


class MultiHeadCrossAttention(nn.Module):
    """Text cross-attention with variable-length text packing.

    PT uses `flash_attn_varlen_func` with cu_seqlens to pack variable-length
    text across the batch into a single sequence `[1, sum(N_valid_i), C]`.
    The MLX equivalent: build a block-diagonal additive mask of shape
    `[1, 1, B*N_visual, sum(N_valid_i)]` so each batch item's visual queries
    only attend to its own text K/V slice.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(dim, dim)
        self.kv_linear = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=1e-6)

    def _process_cross_attn(
        self, x: mx.array, cond: mx.array, kv_seqlen: list[int]
    ) -> mx.array:
        """x: [B, N, C] (visual). cond: [1, sum(kv_seqlen), C] (packed text).
        kv_seqlen: [B] list, number of valid text tokens per batch item.
        """
        B, N, C = x.shape
        assert C == self.dim and cond.shape[2] == self.dim

        # Pack x across batch the same way: [1, B*N, C]
        x_packed = x.reshape(1, B * N, C)
        q = self.q_linear(x_packed).reshape(1, B * N, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).reshape(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0], kv[:, :, 1]

        q = self.q_norm(q)
        k = self.k_norm(k)

        # [1, S, H, D] -> [1, H, S, D] for SDPA
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # Build block-diagonal mask: allow queries from batch i to attend
        # only to K/V tokens from batch i.
        sum_kv = int(sum(kv_seqlen))
        # mask shape: [1, 1, B*N, sum_kv]. Additive: 0 = allow, -inf = block.
        # We build in fp32 for sentinel-value precision, then downcast to
        # q.dtype so mx.fast.scaled_dot_product_attention's promotion rule
        # (`mask dtype must promote to output dtype`) is satisfied. For bf16
        # Q/K/V, fp32 → bf16 is a downcast and MLX rejects it.
        mask = mx.full((B * N, sum_kv), -3.389e38, dtype=mx.float32)
        kv_offset = 0
        q_offset = 0
        for b_i, ki in enumerate(kv_seqlen):
            block = mx.zeros((N, int(ki)), dtype=mx.float32)
            mask[q_offset : q_offset + N, kv_offset : kv_offset + int(ki)] = block
            q_offset += N
            kv_offset += int(ki)
        mask = mask[None, None, :, :].astype(q.dtype)  # promote to Q dtype (e.g. bf16)

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        # [1, H, B*N, D] -> [1, B*N, H, D] -> [B, N, C]
        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        return self.proj(out)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        kv_seqlen: list[int],
        num_cond_latents: Optional[int] = None,
        shape: Optional[tuple[int, int, int]] = None,
    ) -> mx.array:
        """x: [B, N, C] (visual). cond: [1, sum(kv_seqlen), C] (packed text).

        When `num_cond_latents > 0` (image-to-video / video-continuation), only
        the noise portion of visual tokens participates; the cond region is
        zero-padded back in.
        """
        if num_cond_latents is None or num_cond_latents == 0:
            return self._process_cross_attn(x, cond, kv_seqlen)

        assert shape is not None
        B, N, C = x.shape
        tokens_per_frame = N // shape[0]
        ncl_thw = num_cond_latents * tokens_per_frame
        x_noise = x[:, ncl_thw:]
        out_noise = self._process_cross_attn(x_noise, cond, kv_seqlen)
        zeros_cond = mx.zeros((B, ncl_thw, C), dtype=out_noise.dtype)
        return mx.concatenate([zeros_cond, out_noise], axis=1)
