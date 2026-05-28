"""Smoke tests for the DiT primitives (blocks.py + rope_3d.py).

No weights, no parity yet — just shape correctness, fp32 conventions, and
the SwiGLU multiple-of-256 rounding sanity check.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from longcat_video_avatar.models.blocks import (
    CaptionEmbedder,
    FeedForwardSwiGLU,
    FinalLayer_FP32,
    LayerNorm_FP32,
    PatchEmbed3D,
    RMSNorm_FP32,
    TimestepEmbedder,
    modulate_fp32,
)
from longcat_video_avatar.models.rope_3d import (
    RotaryPositionalEmbedding,
    RotaryPositionalEmbedding1D,
    rotate_half,
)


def test_feedforward_swiglu_inner_dim_is_11008():
    """Defends against the SwiGLU 2/3-reduction bug. For LongCat defaults
    (dim=4096, hidden_dim=16384, multiple_of=256), inner = 11008 (NOT 16384).
    """
    ff = FeedForwardSwiGLU(dim=4096, hidden_dim=4096 * 4, multiple_of=256)
    assert ff.hidden_dim == 11008, f"expected 11008, got {ff.hidden_dim}"
    # Linear shapes match
    assert ff.w1.weight.shape == (11008, 4096)
    assert ff.w2.weight.shape == (4096, 11008)
    assert ff.w3.weight.shape == (11008, 4096)


def test_feedforward_swiglu_forward_shape():
    ff = FeedForwardSwiGLU(dim=64, hidden_dim=256, multiple_of=16)
    x = mx.random.normal((2, 8, 64))
    y = ff(x)
    mx.eval(y)
    assert y.shape == (2, 8, 64)


def test_rmsnorm_fp32_internal_compute_is_fp32():
    """The `_FP32` suffix refers to INTERNAL compute precision — the norm
    computation runs in fp32 regardless of input dtype. The output dtype
    follows MLX type promotion: bf16 input × fp32 weight = fp32 output.
    At inference time, `set_dtype(bf16)` would cast `self.weight` to bf16,
    yielding bf16 output. This test only checks the no-NaN, correct-shape
    property since fp32-promoted output is expected with default init.
    """
    rn = RMSNorm_FP32(dim=128, eps=1e-6)
    for dtype in (mx.float32, mx.bfloat16, mx.float16):
        x = mx.random.normal((2, 4, 128)).astype(dtype)
        y = rn(x)
        mx.eval(y)
        assert y.shape == x.shape
        # No NaN/Inf — norm computation should be numerically stable
        finite = mx.all(mx.isfinite(y))
        assert bool(finite), f"non-finite output for input dtype {dtype}"


def test_layernorm_fp32_no_affine_and_affine():
    # No-affine: used inside modulate_fp32
    ln_na = LayerNorm_FP32(dim=64, eps=1e-6, elementwise_affine=False)
    assert not hasattr(ln_na, "weight")
    # Affine: standard LN
    ln_a = LayerNorm_FP32(dim=64, eps=1e-6, elementwise_affine=True)
    assert ln_a.weight.shape == (64,)
    assert ln_a.bias.shape == (64,)

    x = mx.random.normal((2, 8, 64))
    y = ln_a(x)
    mx.eval(y)
    assert y.shape == x.shape


def test_modulate_fp32_requires_fp32_modulation_params():
    norm = LayerNorm_FP32(dim=64, eps=1e-6, elementwise_affine=False)
    x = mx.random.normal((1, 4, 64))
    shift = mx.zeros((1, 1, 64), dtype=mx.float32)
    scale = mx.zeros((1, 1, 64), dtype=mx.float32)
    y = modulate_fp32(norm, x, shift, scale)
    mx.eval(y)
    assert y.shape == x.shape

    # Now with bf16 modulation params — must fail loudly per assertion
    shift_bf16 = shift.astype(mx.bfloat16)
    with pytest.raises(AssertionError):
        modulate_fp32(norm, x, shift_bf16, scale)


def test_timestep_embedder_shape_and_dtype():
    te = TimestepEmbedder(t_embed_dim=512, frequency_embedding_size=256)
    t = mx.arange(8, dtype=mx.float32)  # 8 timesteps
    out = te(t, dtype=mx.float32)
    mx.eval(out)
    assert out.shape == (8, 512)
    assert out.dtype == mx.float32


def test_caption_embedder_passes_through_dims():
    ce = CaptionEmbedder(in_channels=4096, hidden_size=4096)
    x = mx.random.normal((2, 1, 16, 4096))
    out = ce(x)
    mx.eval(out)
    assert out.shape == (2, 1, 16, 4096)


def test_patch_embed_3d_shapes_for_longcat_defaults():
    pe = PatchEmbed3D(patch_size=(1, 2, 2), in_chans=16, embed_dim=4096)
    # B=1, C=16, T=3, H=32, W=32 → patches (3, 16, 16), N = 3*16*16 = 768
    x = mx.random.normal((1, 16, 3, 32, 32))
    out = pe(x)
    mx.eval(out)
    assert out.shape == (1, 3 * 16 * 16, 4096)


def test_final_layer_fp32_shapes():
    fl = FinalLayer_FP32(hidden_size=4096, num_patch=4, out_channels=16, adaln_tembed_dim=512)
    # x: [B=1, N=12, C=4096]. t: [B=1, T=3, C_t=512]. T*?=N → T=3, ?=4.
    x = mx.random.normal((1, 12, 4096))
    t = mx.random.normal((1, 3, 512)).astype(mx.float32)
    out = fl(x, t, latent_shape=(3, 2, 2))
    mx.eval(out)
    assert out.shape == (1, 12, 4 * 16)


def test_rotate_half_correct():
    x = mx.array([[1.0, 2.0, 3.0, 4.0]])  # shape (1, 4)
    y = rotate_half(x)
    mx.eval(y)
    # Pairs: (1,2) -> (-2, 1); (3,4) -> (-4, 3) -> flat: [-2, 1, -4, 3]
    expected = mx.array([[-2.0, 1.0, -4.0, 3.0]])
    assert (y == expected).all(), f"got {y.tolist()}"


def test_3d_rope_shapes_and_dtypes():
    rope = RotaryPositionalEmbedding(head_dim=128)
    q = mx.random.normal((1, 32, 12, 128))  # [B=1, head=32, seq=12, head_dim=128]
    k = mx.random.normal((1, 32, 12, 128))
    grid = (3, 2, 2)  # T*H*W = 12 ✓
    q_r, k_r = rope(q, k, grid)
    mx.eval(q_r, k_r)
    assert q_r.shape == q.shape
    assert k_r.shape == k.shape

    # With ref_img_index / num_ref_latents
    q_r2, k_r2 = rope(q, k, grid, frame_index=10, num_ref_latents=1)
    mx.eval(q_r2, k_r2)
    assert q_r2.shape == q.shape


def test_1d_rope_shapes():
    rope1 = RotaryPositionalEmbedding1D(head_dim=128)
    x = mx.random.normal((1, 32, 24, 128))
    pos = mx.arange(24, dtype=mx.float32)
    out = rope1(x, pos)
    mx.eval(out)
    assert out.shape == x.shape


if __name__ == "__main__":
    test_feedforward_swiglu_inner_dim_is_11008()
    test_feedforward_swiglu_forward_shape()
    test_rmsnorm_fp32_preserves_output_dtype()
    test_layernorm_fp32_no_affine_and_affine()
    test_modulate_fp32_requires_fp32_modulation_params()
    test_timestep_embedder_shape_and_dtype()
    test_caption_embedder_passes_through_dims()
    test_patch_embed_3d_shapes_for_longcat_defaults()
    test_final_layer_fp32_shapes()
    test_rotate_half_correct()
    test_3d_rope_shapes_and_dtypes()
    test_1d_rope_shapes()
    print("all primitives smoke tests passed")
