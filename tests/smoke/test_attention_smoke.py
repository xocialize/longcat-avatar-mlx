"""Smoke tests for the base DiT attention layers."""

from __future__ import annotations

import mlx.core as mx

from longcat_video_avatar.models.attention import Attention, MultiHeadCrossAttention


def test_self_attention_shape_text_to_video_path():
    """Standard self-attn: no cond latents, just visual tokens."""
    attn = Attention(dim=512, num_heads=8)
    # Latent grid: T=3, H=2, W=2 → N=12
    x = mx.random.normal((1, 12, 512))
    out = attn(x, shape=(3, 2, 2))
    mx.eval(out)
    assert out.shape == (1, 12, 512)


def test_self_attention_returns_kv():
    attn = Attention(dim=256, num_heads=4)
    x = mx.random.normal((1, 12, 256))
    out, kv = attn(x, shape=(3, 2, 2), return_kv=True)
    mx.eval(out, kv[0], kv[1])
    assert out.shape == (1, 12, 256)
    # KV: [B, H, N, D]
    assert kv[0].shape == (1, 4, 12, 64)
    assert kv[1].shape == (1, 4, 12, 64)


def test_self_attention_cond_branch():
    """Image-to-video: first num_cond_latents temporal frames are conditioning."""
    attn = Attention(dim=256, num_heads=4)
    # T=4, H=2, W=2, N=16. num_cond_latents=1 → 1 cond frame, 3 noise frames.
    x = mx.random.normal((1, 16, 256))
    out = attn(x, shape=(4, 2, 2), num_cond_latents=1)
    mx.eval(out)
    assert out.shape == (1, 16, 256)


def test_cross_attention_b1():
    """Cross-attn with B=1: kv_seqlen = [N_valid]."""
    ca = MultiHeadCrossAttention(dim=512, num_heads=8)
    visual = mx.random.normal((1, 12, 512))  # B=1, N_visual=12
    text = mx.random.normal((1, 16, 512))  # packed: 1 batch * 16 tokens
    out = ca(visual, text, kv_seqlen=[16])
    mx.eval(out)
    assert out.shape == (1, 12, 512)


def test_cross_attention_b2_blockdiag_mask():
    """B=2 with different per-batch text lengths — block-diagonal mask."""
    ca = MultiHeadCrossAttention(dim=256, num_heads=4)
    visual = mx.random.normal((2, 8, 256))  # B=2, N_visual=8 each
    text = mx.random.normal((1, 20, 256))  # packed: 10 + 10 tokens
    out = ca(visual, text, kv_seqlen=[10, 10])
    mx.eval(out)
    assert out.shape == (2, 8, 256)


def test_cross_attention_with_num_cond_latents():
    """num_cond_latents > 0: cond region zero-padded in output."""
    ca = MultiHeadCrossAttention(dim=256, num_heads=4)
    visual = mx.random.normal((1, 12, 256))  # T=3, N=12 (4 tokens/frame)
    text = mx.random.normal((1, 8, 256))
    out = ca(visual, text, kv_seqlen=[8], num_cond_latents=1, shape=(3, 2, 2))
    mx.eval(out)
    assert out.shape == (1, 12, 256)
    # First 4 tokens (1 cond frame * 4 tokens/frame) should be exactly zero
    cond_region = out[:, :4]
    mx.eval(cond_region)
    import numpy as np
    arr = np.asarray(cond_region)
    assert np.all(arr == 0.0), f"cond region should be zero, max={arr.max()}"


if __name__ == "__main__":
    test_self_attention_shape_text_to_video_path()
    test_self_attention_returns_kv()
    test_self_attention_cond_branch()
    test_cross_attention_b1()
    test_cross_attention_b2_blockdiag_mask()
    test_cross_attention_with_num_cond_latents()
    print("all attention smoke tests passed")
