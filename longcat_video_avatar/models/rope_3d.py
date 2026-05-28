"""MLX port of LongCat-Video 3D RoPE.

PyTorch reference: `refs/longcat-video/longcat_video/modules/avatar/rope_3d.py`
(the Avatar variant — supports `frame_index`/`num_ref_latents` for reference
image insertion; the base variant from `modules/rope_3d.py` is a subset).

Head dim is split into three sub-dimensions:
- `dim_t = head_dim - 4 * (head_dim // 6)`
- `dim_h = 2 * (head_dim // 6)`
- `dim_w = 2 * (head_dim // 6)`

For LongCat's `head_dim=128`: `dim_t=44, dim_h=42, dim_w=42` (sum=128 ✓).

cp_split_hw multi-GPU sharding is dropped (single-device, unified memory).
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def rotate_half(x: mx.array) -> mx.array:
    """Pair-wise rotation: [..., d*r] is reshaped to [..., d, r=2] and
    we return `[-x2, x1]` per pair, then flatten back. Equivalent to PT's
    `rotate_half` used in standard RoPE.
    """
    *prefix, last = x.shape
    assert last % 2 == 0
    x = x.reshape(*prefix, last // 2, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    out = mx.stack([-x2, x1], axis=-1)
    return out.reshape(*prefix, last)


class RotaryPositionalEmbedding(nn.Module):
    """3D RoPE for video tokens. Per-grid-size freqs are memoized.

    `forward(q, k, grid_size, frame_index=None, num_ref_latents=None)` returns
    rotated `(q, k)` with the same shape as input. The optional `frame_index`
    and `num_ref_latents` adjust the temporal grid for reference-image
    insertion (Avatar/video-continuation tasks).
    """

    def __init__(self, head_dim: int):
        super().__init__()
        assert head_dim % 8 == 0, "head_dim must be a multiple of 8 for 3D RoPE"
        self.head_dim = head_dim
        self.base = 10000
        self._freqs_cache: dict[str, mx.array] = {}

    def _precompute_freqs(
        self,
        grid_size: tuple[int, int, int],
        frame_index: int | None,
        num_ref_latents: int | None,
    ) -> mx.array:
        num_frames, height, width = grid_size
        dim_t = self.head_dim - 4 * (self.head_dim // 6)
        dim_h = 2 * (self.head_dim // 6)
        dim_w = 2 * (self.head_dim // 6)

        # Inverse freqs per sub-axis (half-dim each, repeated below)
        idx_t = mx.arange(0, dim_t, 2, dtype=mx.float32)[: dim_t // 2]
        idx_h = mx.arange(0, dim_h, 2, dtype=mx.float32)[: dim_h // 2]
        idx_w = mx.arange(0, dim_w, 2, dtype=mx.float32)[: dim_w // 2]
        freqs_t = 1.0 / (self.base ** (idx_t / dim_t))
        freqs_h = 1.0 / (self.base ** (idx_h / dim_h))
        freqs_w = 1.0 / (self.base ** (idx_w / dim_w))

        # Grid points
        if frame_index is not None and num_ref_latents is not None:
            # Reference image at position `frame_index`, rest is a contiguous
            # range over [0, num_frames - num_ref_latents).
            ref = mx.array([float(frame_index)], dtype=mx.float32)
            cont = mx.arange(0, num_frames - num_ref_latents, dtype=mx.float32)
            grid_t = mx.concatenate([ref, cont], axis=0)
        else:
            grid_t = mx.arange(0, num_frames, dtype=mx.float32)
        grid_h = mx.arange(0, height, dtype=mx.float32)
        grid_w = mx.arange(0, width, dtype=mx.float32)

        # Outer products to get per-position freqs along each axis
        freqs_t = grid_t[:, None] * freqs_t[None, :]  # (T, dim_t/2)
        freqs_h = grid_h[:, None] * freqs_h[None, :]  # (H, dim_h/2)
        freqs_w = grid_w[:, None] * freqs_w[None, :]  # (W, dim_w/2)

        # Repeat each pair (... n -> ... n*2, i.e. interleave each freq twice
        # so that rotate_half pairs adjacent rotations correctly)
        freqs_t = mx.repeat(freqs_t, 2, axis=-1)  # (T, dim_t)
        freqs_h = mx.repeat(freqs_h, 2, axis=-1)  # (H, dim_h)
        freqs_w = mx.repeat(freqs_w, 2, axis=-1)  # (W, dim_w)

        # Broadcast to (T, H, W, dim_t+dim_h+dim_w=head_dim) via outer-axis
        # concatenation. Each (T, H, W) position has [freqs_t, freqs_h, freqs_w].
        T, _, _ = num_frames, height, width
        H, W = height, width
        freqs_t_b = mx.broadcast_to(freqs_t[:, None, None, :], (T, H, W, dim_t))
        freqs_h_b = mx.broadcast_to(freqs_h[None, :, None, :], (T, H, W, dim_h))
        freqs_w_b = mx.broadcast_to(freqs_w[None, None, :, :], (T, H, W, dim_w))
        freqs = mx.concatenate([freqs_t_b, freqs_h_b, freqs_w_b], axis=-1)

        # Flatten (T, H, W, D) -> (T*H*W, D)
        return freqs.reshape(T * H * W, self.head_dim)

    def __call__(
        self,
        q: mx.array,
        k: mx.array,
        grid_size: tuple[int, int, int],
        frame_index: int | None = None,
        num_ref_latents: int | None = None,
    ) -> tuple[mx.array, mx.array]:
        """q, k: [B, head, seq, head_dim]. Returns rotated q, k."""
        key_name = f"{grid_size[0]}.{grid_size[1]}.{grid_size[2]}-{frame_index}-{num_ref_latents}"
        if key_name not in self._freqs_cache:
            self._freqs_cache[key_name] = self._precompute_freqs(grid_size, frame_index, num_ref_latents)
        freqs = self._freqs_cache[key_name]
        cos = mx.cos(freqs)[None, None, :, :]  # [1, 1, seq, head_dim]
        sin = mx.sin(freqs)[None, None, :, :]

        out_dtype = q.dtype
        q_f = q.astype(mx.float32)
        k_f = k.astype(mx.float32)
        q_r = q_f * cos + rotate_half(q_f) * sin
        k_r = k_f * cos + rotate_half(k_f) * sin
        return q_r.astype(out_dtype), k_r.astype(out_dtype)


class RotaryPositionalEmbedding1D(nn.Module):
    """1D RoPE applied at arbitrary positions. Used for MultiTalk human routing.

    `forward(x, pos_indices)` rotates `x` per the given positions.
    `pos_indices` is a 1D array of length seq.
    """

    def __init__(self, head_dim: int):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000

    def _precompute_freqs(self, pos_indices: mx.array) -> mx.array:
        idx = mx.arange(0, self.head_dim, 2, dtype=mx.float32)[: self.head_dim // 2]
        freqs = 1.0 / (self.base ** (idx / self.head_dim))
        freqs = pos_indices.astype(mx.float32)[:, None] * freqs[None, :]
        return mx.repeat(freqs, 2, axis=-1)  # (seq, head_dim)

    def __call__(self, x: mx.array, pos_indices: mx.array) -> mx.array:
        """x: [B, head, seq, head_dim]. Returns rotated x."""
        freqs = self._precompute_freqs(pos_indices)
        cos = mx.cos(freqs)[None, None, :, :]
        sin = mx.sin(freqs)[None, None, :, :]
        out_dtype = x.dtype
        x_f = x.astype(mx.float32)
        x_r = x_f * cos + rotate_half(x_f) * sin
        return x_r.astype(out_dtype)
