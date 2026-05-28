"""MLX port of LongCat-Video DiT primitives.

PyTorch reference: `refs/longcat-video/longcat_video/modules/blocks.py`.
Each class name and forward order mirrors the PT source per the
mlx-porting skill's isomorphic-structure rule.

The `*_FP32` suffix on RMSNorm / LayerNorm / FinalLayer is a Meituan
convention: their forward casts the input to fp32 for the norm + scale +
bias computation, then casts back. We replicate this exactly — silent
bf16 accumulation in AdaLN modulation has been shown to break parity in
their training runs.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


class FeedForwardSwiGLU(nn.Module):
    """SwiGLU-gated FFN.

    Internal `hidden_dim` follows the SwiGLU 2/3 rule: `int(2 * hidden_dim / 3)`,
    optionally scaled by `ffn_dim_multiplier`, rounded up to `multiple_of`.
    For `dim=4096, hidden_dim=16384, multiple_of=256` (LongCat defaults):
    `int(2 * 16384 / 3) = 10922`, rounded up to `256 * 43 = 11008`.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class RMSNorm_FP32(nn.Module):
    """RMS norm with fp32 internal compute (matches PT `RMSNorm_FP32`).

    Forward does: `((x.float() * rsqrt(mean(x.float()**2) + eps)).type_as(x)) * weight`.
    Norm computation runs in fp32 regardless of input dtype; the learned `weight`
    is applied in the input dtype.
    """

    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        x_f32 = x.astype(mx.float32)
        norm = x_f32 * mx.rsqrt(mx.mean(x_f32 * x_f32, axis=-1, keepdims=True) + self.eps)
        return norm.astype(out_dtype) * self.weight


class LayerNorm_FP32(nn.Module):
    """LayerNorm with fp32 internal compute.

    Wraps a standard LayerNorm but forces fp32 inputs to `F.layer_norm` and
    casts back. PT's `LayerNorm_FP32` subclasses `nn.LayerNorm` and overrides
    `forward`; we inline the equivalent.

    `elementwise_affine=False` means no weight/bias — used inside `modulate_fp32`.
    `elementwise_affine=True` adds learned `weight` and `bias`.
    """

    def __init__(self, dim: int, eps: float, elementwise_affine: bool):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
            self.bias = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        out_dtype = x.dtype
        x_f32 = x.astype(mx.float32)
        mean = mx.mean(x_f32, axis=-1, keepdims=True)
        var = mx.mean((x_f32 - mean) ** 2, axis=-1, keepdims=True)
        x_f32 = (x_f32 - mean) * mx.rsqrt(var + self.eps)
        if self.elementwise_affine:
            x_f32 = x_f32 * self.weight.astype(mx.float32) + self.bias.astype(mx.float32)
        return x_f32.astype(out_dtype)


def modulate_fp32(norm_func, x, shift, scale):
    """AdaLN-Zero style modulation in fp32, then cast back.

    Asserts `shift.dtype == scale.dtype == fp32`. The result is
    `norm(x.float()) * (scale + 1) + shift`, returned in `x.dtype`.
    """
    assert shift.dtype == mx.float32 and scale.dtype == mx.float32, (
        "Modulation params must be fp32; AdaLN math diverges in bf16."
    )
    dtype = x.dtype
    x_f32 = norm_func(x.astype(mx.float32))
    x_f32 = x_f32 * (scale + 1) + shift
    return x_f32.astype(dtype)


class _Conv3dPlaceholder(nn.Module):
    """Holder for Conv3d weight + bias so checkpoint keys become
    `x_embedder.proj.weight` (matching PT `self.proj = nn.Conv3d(...)`).

    Weight layout: `(O, kT, kH, kW, I)` (MLX channels-last). PT
    `(O, I, kT, kH, kW)` is transposed at load time.
    """

    def __init__(self, out_channels: int, in_channels: int, kernel_size: tuple):
        super().__init__()
        kt, kh, kw = kernel_size
        self.weight = mx.zeros((out_channels, kt, kh, kw, in_channels))
        self.bias = mx.zeros((out_channels,))


class PatchEmbed3D(nn.Module):
    """3D patchify via Conv3d.

    MLX has no native Conv3d, so we emulate via sliding Conv2d over the time
    axis (same approach as `autoencoder_kl_wan.CausalConv3d`). For LongCat
    `patch_size=(1, 2, 2)` the kernel is non-causal and identical to applying
    Conv2d on each frame independently — but we keep the general path for
    forward compatibility with future patch_size values.
    """

    def __init__(
        self,
        patch_size: tuple = (2, 4, 4),
        in_chans: int = 3,
        embed_dim: int = 96,
        flatten: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.flatten = flatten
        # PT key path: `proj.weight`, `proj.bias`
        self.proj = _Conv3dPlaceholder(embed_dim, in_chans, patch_size)

    def __call__(self, x: mx.array) -> mx.array:
        """x: [B, C, T, H, W] (channel-second, PT convention). Output: [B, N, C]
        when `flatten=True`, otherwise [B, C, T_p, H_p, W_p]."""
        b, c, t, h, w = x.shape
        kt, kh, kw = self.patch_size

        # Pad to multiples of patch_size if needed
        if w % kw != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, 0), (0, kw - w % kw)])
        if h % kh != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (0, kh - h % kh), (0, 0)])
        if t % kt != 0:
            x = mx.pad(x, [(0, 0), (0, 0), (0, kt - t % kt), (0, 0), (0, 0)])

        # Channels-last for MLX conv: [B, T, H, W, C]
        x = x.transpose(0, 2, 3, 4, 1)
        b, t, h, w, c = x.shape

        # 3D conv = stack of per-time-step 2D convs (one per kt window).
        t_out = t // kt
        h_out = h // kh
        w_out = w // kw

        # Reshape weight: (O, kT, kH, kW, I) -> (O, kH, kW, kT*I)
        w_2d = self.proj.weight.transpose(0, 2, 3, 1, 4).reshape(
            self.embed_dim, kh, kw, kt * c
        )

        outputs = []
        for ti in range(t_out):
            t_start = ti * kt
            window = x[:, t_start : t_start + kt]
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c)
            out_2d = mx.conv2d(window, w_2d, stride=(kh, kw)) + self.proj.bias
            outputs.append(out_2d)
        out = mx.stack(outputs, axis=1)  # (B, t_out, h_out, w_out, embed_dim)

        if self.flatten:
            out = out.reshape(b, t_out * h_out * w_out, self.embed_dim)
        else:
            out = out.transpose(0, 4, 1, 2, 3)
        return out


def _timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
    """Sinusoidal timestep embedding. `t` is 1D of shape (N,)."""
    half = dim // 2
    freqs = mx.exp(-math.log(max_period) * mx.arange(0, half, dtype=mx.float32) / half)
    args = t[:, None].astype(mx.float32) * freqs[None]
    embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
    if dim % 2:
        embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """Sinusoidal embedding + 2-layer MLP. Output is fp32 by convention.

    PT key names: `mlp.0.{weight,bias}`, `mlp.2.{weight,bias}` (the middle
    SiLU has no params). We mirror that with a list.
    """

    def __init__(self, t_embed_dim: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.t_embed_dim = t_embed_dim
        self.frequency_embedding_size = frequency_embedding_size
        # Match PT's nn.Sequential indices: [0] Linear, [1] SiLU (no params),
        # [2] Linear. We use a list with None placeholder for [1].
        self.mlp = [
            nn.Linear(frequency_embedding_size, t_embed_dim, bias=True),
            None,  # SiLU
            nn.Linear(t_embed_dim, t_embed_dim, bias=True),
        ]

    def __call__(self, t: mx.array, dtype: mx.Dtype = mx.float32) -> mx.array:
        t_freq = _timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.astype(dtype)
        x = self.mlp[0](t_freq)
        x = nn.silu(x)
        x = self.mlp[2](x)
        return x


class CaptionEmbedder(nn.Module):
    """2-layer MLP with GELU(tanh approx). PT name: `y_proj`."""

    def __init__(self, in_channels: int, hidden_size: int):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        # PT: y_proj.0 = Linear, y_proj.1 = GELU(tanh), y_proj.2 = Linear
        self.y_proj = [
            nn.Linear(in_channels, hidden_size, bias=True),
            None,  # GELU(tanh)
            nn.Linear(hidden_size, hidden_size, bias=True),
        ]

    def __call__(self, caption: mx.array) -> mx.array:
        # Input: [B, _, N, C_in]
        x = self.y_proj[0](caption)
        x = nn.gelu_approx(x)  # tanh-approximate GELU (matches PT `approximate="tanh"`)
        x = self.y_proj[2](x)
        return x


class FinalLayer_FP32(nn.Module):
    """Final AdaLN-Zero head: LN(no-affine) -> Linear, with shift/scale from t.

    PT: `norm_final` (LayerNorm_FP32 no-affine), `linear` (Linear),
    `adaLN_modulation` (nn.Sequential of SiLU + Linear(t_embed_dim, 2*hidden_size)).
    """

    def __init__(self, hidden_size: int, num_patch: int, out_channels: int, adaln_tembed_dim: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_patch = num_patch
        self.out_channels = out_channels
        self.adaln_tembed_dim = adaln_tembed_dim

        self.norm_final = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        # adaLN_modulation: SiLU + Linear(adaln_tembed_dim, 2 * hidden_size)
        self.adaLN_modulation = [
            None,  # SiLU
            nn.Linear(adaln_tembed_dim, 2 * hidden_size, bias=True),
        ]

    def __call__(self, x: mx.array, t: mx.array, latent_shape: tuple) -> mx.array:
        # t shape: [B, T, C_t]. Must be fp32.
        assert t.dtype == mx.float32, "FinalLayer expects fp32 timestep embedding"
        b, n, c = x.shape
        t_frames = latent_shape[0]

        # SiLU + Linear(t) -> chunk(2)
        t_in = nn.silu(t)
        ada = self.adaLN_modulation[1](t_in)  # [B, T, 2*hidden]
        ada = ada[:, :, None, :]  # [B, T, 1, C]
        shift, scale = mx.split(ada, 2, axis=-1)
        x = modulate_fp32(self.norm_final, x.reshape(b, t_frames, -1, c), shift, scale).reshape(b, n, c)
        x = self.linear(x)
        return x
