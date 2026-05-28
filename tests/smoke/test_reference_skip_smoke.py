"""Smoke tests for Reference Skip Attention (S1.6).

The Reference Skip path activates only when `num_cond_latents > 1` AND
`mask_frame_range > 0`. With those args unset, the Avatar Attention output
must match the standard cond/noise branching of base Attention (regression
guard required by v3 plan Stage 1.6 gate).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from longcat_video_avatar.models.attention import Attention as BaseAttention
from longcat_video_avatar.models.avatar.attention import Attention as AvatarAttention


def _seeded_input(shape, seed=0):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal(shape).astype(np.float32))


def test_avatar_attn_matches_base_when_reference_skip_inactive():
    """When mask_frame_range is unused, AvatarAttention's output should be
    numerically identical to BaseAttention's (Reference Skip is a no-op).
    """
    # Build base and avatar with the same parameter shapes
    base = BaseAttention(dim=64, num_heads=4)
    avatar = AvatarAttention(dim=64, num_heads=4)

    # Copy params verbatim from base to avatar via tree_flatten
    from mlx.utils import tree_flatten, tree_unflatten

    base_params = dict(tree_flatten(base.parameters()))
    avatar.update(tree_unflatten(list(base_params.items())))
    mx.eval(base.parameters(), avatar.parameters())

    x = _seeded_input((1, 12, 64), seed=42)
    shape = (3, 2, 2)  # T=3, H=2, W=2 → N=12

    base_out = base(x, shape=shape, num_cond_latents=0)
    av_out, x_ref = avatar(x, shape=shape, num_cond_latents=0)
    mx.eval(base_out, av_out)

    diff = mx.abs(base_out - av_out).max()
    assert float(diff) < 1e-6, f"avatar attn diverges from base: max_abs={float(diff)}"
    assert x_ref is None


def test_avatar_attn_reference_skip_active_path():
    """With ref_img_index, num_ref_latents, mask_frame_range, and num_cond_latents
    set, the Q-slicing branch activates. Output shape must stay the same.
    """
    avatar = AvatarAttention(dim=64, num_heads=4)

    # T=8 frames, H=2, W=2 → N=32
    # num_cond_latents=2 (1 ref + 1 cond), num_ref_latents=1, ref_img_index=4
    # → 6 noise frames; mask_frame_range=1 → start_noise=2, end_noise=4 (window 2..3)
    x = _seeded_input((1, 32, 64), seed=42)
    out, x_ref = avatar(
        x,
        shape=(8, 2, 2),
        num_cond_latents=2,
        num_ref_latents=1,
        ref_img_index=4,
        mask_frame_range=1,
    )
    mx.eval(out)
    assert out.shape == (1, 32, 64)
    assert x_ref is None  # single talker — no MultiTalk routing


def test_avatar_attn_reference_skip_out_of_range_falls_back():
    """If the mask window is out of bounds, the branch falls back to standard
    cond/noise attention (no Q-slicing)."""
    avatar = AvatarAttention(dim=64, num_heads=4)
    x = _seeded_input((1, 32, 64), seed=42)
    # mask_frame_range=100 makes end_noise > num_noisy_frames → fallback
    out, x_ref = avatar(
        x,
        shape=(8, 2, 2),
        num_cond_latents=2,
        num_ref_latents=1,
        ref_img_index=4,
        mask_frame_range=100,
    )
    mx.eval(out)
    assert out.shape == (1, 32, 64)
    assert x_ref is None
