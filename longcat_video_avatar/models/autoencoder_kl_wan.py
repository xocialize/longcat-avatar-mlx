"""MLX port of the Wan 2.1 VAE (`AutoencoderKLWan`) used by LongCat-Video.

PyTorch reference: diffusers 0.38's
`diffusers/models/autoencoders/autoencoder_kl_wan.py` (`is_residual=False`
variant — Wan 2.1, as shipped by `meituan-longcat/LongCat-Video`).

Module hierarchy mirrors the PT class names exactly so weights load with only
a Conv3d/Conv2d transpose, no key remapping. The earlier draft of this file
adapted from `Blaizzy/mlx-video`'s `wan_2/vae.py`, but that port targets a
different Wan VAE checkpoint variant with a different channel pattern
(see `notes/vae-schema-mismatch.md`) and could not load Meituan's weights.

Op primitives (`CausalConv3d`, `RMS_norm`, `AttentionBlock`, `Resample`) are
kept compatible with that earlier code; only the composite block hierarchy
changes.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

CACHE_T = 2

# Default per-channel normalization stats for z_dim=16 (overridden by from_config)
DEFAULT_VAE_MEAN: list[float] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
DEFAULT_VAE_STD: list[float] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]


# ===========================================================================
# Op primitives (match diffusers' WanCausalConv3d / WanRMS_norm / etc.)
# ===========================================================================


class CausalConv3d(nn.Module):
    """3D conv with causal temporal padding. Matches diffusers `WanCausalConv3d`.

    MLX has no native Conv3d, so we slide a Conv2d over the time axis.
    Weights stored as `(O, kT, kH, kW, I)`; PT `(O, I, kT, kH, kW)` is
    transposed once at load time.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple,
        stride: int | tuple = 1,
        padding: int | tuple = 0,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding, padding)

        self.kernel_size = kernel_size
        self.stride = stride
        # Causal padding: dilation*(k-1) + (1-stride). dilation=1 → k-stride.
        self._causal_pad_t = kernel_size[0] - stride[0]
        self._pad_h = padding[1]
        self._pad_w = padding[2]

        self.weight = mx.zeros(
            (out_channels, kernel_size[0], kernel_size[1], kernel_size[2], in_channels)
        )
        self.bias = mx.zeros((out_channels,))

    def __call__(self, x: mx.array, cache_x: mx.array | None = None) -> mx.array:
        """x: [B, C, T, H, W]"""
        b, c, t, h, w = x.shape

        causal_pad = self._causal_pad_t
        if cache_x is not None and causal_pad > 0:
            x = mx.concatenate([cache_x, x], axis=2)
            causal_pad = max(0, causal_pad - cache_x.shape[2])

        if causal_pad > 0:
            pad_t = mx.zeros((b, c, causal_pad, h, w), dtype=x.dtype)
            x = mx.concatenate([pad_t, x], axis=2)

        if self._pad_h > 0 or self._pad_w > 0:
            x = mx.pad(
                x,
                [(0, 0), (0, 0), (0, 0), (self._pad_h, self._pad_h), (self._pad_w, self._pad_w)],
            )

        x = x.transpose(0, 2, 3, 4, 1)  # [B, T, H, W, C]
        out = self._conv3d(x)
        return out.transpose(0, 4, 1, 2, 3)

    def _conv3d(self, x: mx.array) -> mx.array:
        b, t, h, w, c_in = x.shape
        kt, kh, kw = self.kernel_size
        st, sh, sw = self.stride
        t_out = (t - kt) // st + 1

        w_2d = self.weight.transpose(0, 2, 3, 1, 4).reshape(
            self.weight.shape[0], kh, kw, kt * c_in
        )
        outputs = []
        for t_i in range(t_out):
            t_start = t_i * st
            window = x[:, t_start : t_start + kt]
            window = window.transpose(0, 2, 3, 1, 4).reshape(b, h, w, kt * c_in)
            out_2d = mx.conv2d(window, w_2d, stride=(sh, sw)) + self.bias
            outputs.append(out_2d)
        return mx.stack(outputs, axis=1)


class RMS_norm(nn.Module):
    """Channel-first L2-normalize + learned scale. Matches diffusers `WanRMS_norm`."""

    def __init__(self, dim: int, channel_first: bool = True, images: bool = True):
        super().__init__()
        self.channel_first = channel_first
        self.scale = dim**0.5
        if channel_first:
            broadcastable = (1, 1) if images else (1, 1, 1)
            self.gamma = mx.ones((dim, *broadcastable))
        else:
            self.gamma = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        norm_dim = 1 if self.channel_first else -1
        norm = mx.sqrt(
            mx.clip(mx.sum(x * x, axis=norm_dim, keepdims=True), a_min=1e-12, a_max=None)
        )
        return (x / norm) * self.scale * self.gamma


# ===========================================================================
# Composite blocks (match diffusers' WanResidualBlock / WanAttentionBlock /
# WanResample / WanMidBlock / WanUpBlock).
# ===========================================================================


class WanResidualBlock(nn.Module):
    """ResNet block with named norm1/conv1/norm2/conv2/conv_shortcut fields
    (matches diffusers `WanResidualBlock`).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.norm1 = RMS_norm(in_dim, images=False)
        self.conv1 = CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = RMS_norm(out_dim, images=False)
        # diffusers has self.dropout = nn.Dropout — we omit (inference-only, p=0)
        self.conv2 = CausalConv3d(out_dim, out_dim, 3, padding=1)
        # NOTE: PT uses nn.Identity() for no-shortcut; we use None for cleanliness.
        # The PT checkpoint has no `conv_shortcut.weight` key when in_dim == out_dim.
        self.conv_shortcut = CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else None

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        h = x if self.conv_shortcut is None else self.conv_shortcut(x)

        x = nn.silu(self.norm1(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv1(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = nn.silu(self.norm2(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv2(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return x + h


class AttentionBlock(nn.Module):
    """Single-head spatial self-attention (per-frame). Matches diffusers `WanAttentionBlock`.

    **Precision note:** MLX's Metal GPU loses ~3-4 bits of fp32 precision on the
    long-accumulator matmul inside attention (QK^T with D=384 and softmax(...)·V
    with S=256). With identical fp32 inputs PT-CPU produces output matching
    numpy-fp32 to ~5e-6; MLX-GPU diverges by ~1.6e-3, which compounds through
    the decoder's downstream resblocks. The VAE has only two attention blocks
    (one in each mid_block) so we route the SDP call through `mx.stream(mx.cpu)`
    — keeping parity tight at negligible perf cost. The fix only needs to be
    applied to attention; conv layers don't have this issue at fp32.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.norm = RMS_norm(dim, images=True)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def __call__(self, x: mx.array) -> mx.array:
        identity = x
        b, c, t, h, w = x.shape

        x = x.transpose(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.norm(x)
        x = x.transpose(0, 2, 3, 1)  # [BT, H, W, C]

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(b * t, h * w, 3, c).transpose(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q[:, None, :, :]
        k = k[:, None, :, :]
        v = v[:, None, :, :]

        # CPU stream for the QK^T → softmax → ·V chain to keep fp32 precision.
        # See class docstring.
        with mx.stream(mx.cpu):
            scale = c ** -0.5
            attn_logits = (q @ k.transpose(0, 1, 3, 2)) * scale
            attn = mx.softmax(attn_logits, axis=-1)
            out = attn @ v
            mx.eval(out)  # force materialization before leaving the cpu stream

        out = out.squeeze(1).reshape(b * t, h, w, c)
        out = self.proj(out)
        out = out.reshape(b, t, h, w, c).transpose(0, 4, 1, 2, 3)
        return out + identity


class Resample(nn.Module):
    """Upsample / downsample. Matches diffusers `WanResample`.

    Modes: `upsample2d`, `upsample3d`, `downsample2d`, `downsample3d`.
    Upsample modes: `resample[1]` is the Conv2d that halves channels.
    Downsample modes: `resample[1]` is a stride-2 Conv2d that keeps channels.
    `*3d` modes additionally have `time_conv` (CausalConv3d).
    """

    def __init__(self, dim: int, mode: str):
        super().__init__()
        assert mode in ("upsample2d", "upsample3d", "downsample2d", "downsample3d")
        self.mode = mode
        self.dim = dim

        if mode.startswith("upsample"):
            self.resample = [None, nn.Conv2d(dim, dim // 2, 3, padding=1)]
            if mode == "upsample3d":
                self.time_conv = CausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        else:
            self.resample = [None, nn.Conv2d(dim, dim, 3, stride=2)]
            if mode == "downsample3d":
                self.time_conv = CausalConv3d(
                    dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)
                )

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        b, c, t, h, w = x.shape

        if self.mode == "upsample3d":
            # Matches diffusers WanResample.forward (autoencoder_kl_wan.py:269).
            # The "Rep" sentinel in feat_cache makes the FIRST call skip time_conv
            # entirely — so the first decoded latent produces 1 video frame, and
            # each subsequent latent produces 2 (per upsample3d stage). Two such
            # stages stacked give 1, 4, 4, ... video frames per latent.
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    # First call: skip time_conv, mark cache, no temporal change.
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:]
                    if cache_x.shape[2] < 2:
                        if feat_cache[idx] != "Rep":
                            cache_x = mx.concatenate(
                                [feat_cache[idx][:, :, -1:], cache_x], axis=2
                            )
                        else:  # feat_cache[idx] == "Rep"
                            cache_x = mx.concatenate(
                                [mx.zeros_like(cache_x), cache_x], axis=2
                            )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, cache_x=feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    # Double T via reshape-and-stack
                    t = x.shape[2]
                    x = x.reshape(b, 2, c, t, h, w)
                    x = mx.stack([x[:, 0], x[:, 1]], axis=3).reshape(b, c, t * 2, h, w)
                    t = t * 2

        if self.mode.startswith("upsample"):
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
            x = mx.repeat(x, 2, axis=1)
            x = mx.repeat(x, 2, axis=2)
            x = self.resample[1](x)
            c_out = x.shape[-1]
            return x.reshape(b, t, h * 2, w * 2, c_out).transpose(0, 4, 1, 2, 3)
        else:
            x = x.transpose(0, 2, 3, 4, 1).reshape(b * t, h, w, c)
            x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
            x = self.resample[1](x)
            c_out = x.shape[-1]
            h_out, w_out = x.shape[1], x.shape[2]
            x = x.reshape(b, t, h_out, w_out, c_out).transpose(0, 4, 1, 2, 3)

            if self.mode == "downsample3d":
                if feat_cache is not None:
                    idx = feat_idx[0]
                    if feat_cache[idx] is None:
                        feat_cache[idx] = x
                        feat_idx[0] += 1
                    else:
                        cache_x = x[:, :, -1:]
                        x = self.time_conv(x, cache_x=feat_cache[idx][:, :, -1:])
                        feat_cache[idx] = cache_x
                        feat_idx[0] += 1
                else:
                    x = self.time_conv(x)
            return x


class WanMidBlock(nn.Module):
    """Middle block: [resnet, attn, resnet]. Matches diffusers `WanMidBlock`."""

    def __init__(self, dim: int, num_layers: int = 1):
        super().__init__()
        resnets = [WanResidualBlock(dim, dim)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(AttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim))
        self.attentions = attentions
        self.resnets = resnets

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


class WanUpBlock(nn.Module):
    """Decoder up-block: `resnets[..R]` + optional `upsamplers[0]`.

    Matches diffusers `WanUpBlock` (is_residual=False variant).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        upsample_mode: str | None = None,
    ):
        super().__init__()
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim))
            current_dim = out_dim
        self.resnets = resnets

        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = [Resample(out_dim, mode=upsample_mode)]

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x


# ===========================================================================
# Encoder / Decoder
# ===========================================================================


class WanEncoder3d(nn.Module):
    """3D VAE Encoder. Matches diffusers' `WanEncoder3d` with `is_residual=False`.

    `down_blocks` is a FLAT list mixing `WanResidualBlock`, optional
    `AttentionBlock`, and `Resample(downsample*)` per stage. This is the
    diffusers convention for Wan 2.1 (`is_residual=False`).
    """

    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list | None = None,
        num_res_blocks: int = 2,
        attn_scales: list | None = None,
        temperal_downsample: list | None = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        if attn_scales is None:
            attn_scales = []

        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        self.conv_in = CausalConv3d(in_channels, dims[0], 3, padding=1)

        down_blocks: list = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            for _ in range(num_res_blocks):
                down_blocks.append(WanResidualBlock(in_dim, out_dim))
                if scale in attn_scales:
                    down_blocks.append(AttentionBlock(out_dim))
                in_dim = out_dim
            if i != len(dim_mult) - 1:
                mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                down_blocks.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.down_blocks = down_blocks

        self.mid_block = WanMidBlock(out_dim, num_layers=1)

        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, z_dim, 3, padding=1)

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None and isinstance(layer, (WanResidualBlock, Resample)):
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        x = nn.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


class WanDecoder3d(nn.Module):
    """3D VAE Decoder. Matches diffusers' `WanDecoder3d` with `is_residual=False`.

    `up_blocks` is a NESTED list of `WanUpBlock` instances (one per stage).
    Each `WanUpBlock` wraps `resnets[..R]` plus optional `upsamplers[0]`.
    """

    def __init__(
        self,
        out_channels: int = 3,
        dim: int = 96,
        z_dim: int = 16,
        dim_mult: list | None = None,
        num_res_blocks: int = 2,
        temperal_upsample: list | None = None,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_upsample is None:
            temperal_upsample = [True, True, False]

        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]

        self.conv_in = CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0], num_layers=1)

        up_blocks: list = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # Wan 2.1: starting from stage 1, the previous upsample halves channels.
            if i > 0:
                in_dim = in_dim // 2

            up_flag = i != len(dim_mult) - 1
            upsample_mode: str | None = None
            if up_flag:
                upsample_mode = "upsample3d" if temperal_upsample[i] else "upsample2d"

            up_blocks.append(
                WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    upsample_mode=upsample_mode,
                )
            )
        self.up_blocks = up_blocks

        self.norm_out = RMS_norm(out_dim, images=False)
        self.conv_out = CausalConv3d(out_dim, out_channels, 3, padding=1)

    def __call__(self, x: mx.array, feat_cache=None, feat_idx=None) -> mx.array:
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv_in(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx)

        x = nn.silu(self.norm_out(x))
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:]
            if cache_x.shape[2] < CACHE_T and feat_cache[idx] is not None:
                cache_x = mx.concatenate([feat_cache[idx][:, :, -1:], cache_x], axis=2)
            x = self.conv_out(x, cache_x=feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)

        return x


# ===========================================================================
# Top-level AutoencoderKLWan
# ===========================================================================


class AutoencoderKLWan(nn.Module):
    """Wan 2.1 VAE matching diffusers 0.38 `AutoencoderKLWan` schema.

    Top-level params:
      - `encoder` (WanEncoder3d) → outputs 2*z_dim (mu+logvar)
      - `quant_conv` (CausalConv3d 2*z_dim → 2*z_dim, 1×1×1) — post-encoder mixer
      - `post_quant_conv` (CausalConv3d z_dim → z_dim, 1×1×1) — pre-decoder mixer
      - `decoder` (WanDecoder3d) → outputs 3-channel video
    """

    def __init__(
        self,
        z_dim: int = 16,
        base_dim: int = 96,
        dim_mult: list | None = None,
        num_res_blocks: int = 2,
        attn_scales: list | None = None,
        temperal_downsample: list | None = None,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
        encoder: bool = True,
    ):
        super().__init__()
        if dim_mult is None:
            dim_mult = [1, 2, 4, 4]
        if temperal_downsample is None:
            temperal_downsample = [False, True, True]
        if attn_scales is None:
            attn_scales = []
        if latents_mean is None:
            latents_mean = DEFAULT_VAE_MEAN
        if latents_std is None:
            latents_std = DEFAULT_VAE_STD

        temporal_upsample = list(reversed(temperal_downsample))
        self.z_dim = z_dim
        self.mean = mx.array(latents_mean)
        self.std = mx.array(latents_std)
        self.inv_std = 1.0 / self.std

        # Decoder is always present
        self.decoder = WanDecoder3d(
            out_channels=3,
            dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            temperal_upsample=temporal_upsample,
        )
        self.post_quant_conv = CausalConv3d(z_dim, z_dim, 1)

        if encoder:
            self.encoder = WanEncoder3d(
                in_channels=3,
                dim=base_dim,
                z_dim=z_dim * 2,
                dim_mult=dim_mult,
                num_res_blocks=num_res_blocks,
                attn_scales=attn_scales,
                temperal_downsample=temperal_downsample,
            )
            self.quant_conv = CausalConv3d(z_dim * 2, z_dim * 2, 1)

    @classmethod
    def from_config(cls, config: dict, *, encoder: bool = True) -> "AutoencoderKLWan":
        """Construct from a Meituan-style `vae/config.json` dict."""
        return cls(
            z_dim=config.get("z_dim", 16),
            base_dim=config.get("base_dim", 96),
            dim_mult=config.get("dim_mult"),
            num_res_blocks=config.get("num_res_blocks", 2),
            attn_scales=config.get("attn_scales"),
            temperal_downsample=config.get("temperal_downsample"),
            latents_mean=config.get("latents_mean"),
            latents_std=config.get("latents_std"),
            encoder=encoder,
        )

    def _count_encoder_cache_slots(self) -> int:
        """Count CausalConv3d slots that participate in chunked-encoding cache."""
        count = 1  # encoder.conv_in
        for layer in self.encoder.down_blocks:
            if isinstance(layer, WanResidualBlock):
                count += 2
            elif isinstance(layer, Resample) and layer.mode == "downsample3d":
                count += 1
        for resnet in self.encoder.mid_block.resnets:
            count += 2
        count += 1  # encoder.conv_out
        return count

    def _count_decoder_cache_slots(self) -> int:
        """Count CausalConv3d slots that participate in chunked-decoding cache."""
        count = 1  # decoder.conv_in
        for resnet in self.decoder.mid_block.resnets:
            count += 2
        for up_block in self.decoder.up_blocks:
            for _resnet in up_block.resnets:
                count += 2
            if up_block.upsamplers is not None:
                # Both upsample2d and upsample3d advance feat_idx by 1 (upsample3d
                # uses the slot for the time_conv cache or the "Rep" sentinel;
                # upsample2d... actually doesn't use feat_cache at all). We only
                # reserve a slot when temporal: mode == upsample3d.
                if up_block.upsamplers[0].mode == "upsample3d":
                    count += 1
        count += 1  # decoder.conv_out
        return count

    def encode(self, x: mx.array) -> mx.array:
        """Video [B, 3, T, H, W] in [-1, 1] → raw latent mean [B, z_dim, T_lat, H_lat, W_lat].

        Matches diffusers convention: the output is the posterior mean before
        per-channel normalization. Use `normalize_latents` to scale into the
        DiT's expected distribution.
        """
        num_slots = self._count_encoder_cache_slots()
        feat_cache = [None] * num_slots

        t = x.shape[2]
        num_chunks = 1 + (t - 1) // 4

        out = None
        for i in range(num_chunks):
            feat_idx = [0]
            chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i]
            chunk_out = self.encoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
            out = chunk_out if out is None else mx.concatenate([out, chunk_out], axis=2)

        mu, _ = mx.split(self.quant_conv(out), 2, axis=1)
        return mu

    def normalize_latents(self, mu: mx.array) -> mx.array:
        """(mu - latents_mean) / latents_std — apply BEFORE feeding to the DiT."""
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        return (mu - mean) * inv_std

    def denormalize_latents(self, z: mx.array) -> mx.array:
        """z * latents_std + latents_mean — apply BEFORE decoding the DiT output."""
        mean = self.mean.reshape(1, -1, 1, 1, 1)
        inv_std = self.inv_std.reshape(1, -1, 1, 1, 1)
        return z / inv_std + mean

    def decode(self, z: mx.array) -> mx.array:
        """Raw latent (post-denormalization) → video [B, 3, T, H, W] in [-1, 1].

        Matches diffusers `AutoencoderKLWan._decode`: caller is responsible for
        applying `denormalize_latents` first if their `z` came from the DiT's
        normalized output. Iterates one latent frame at a time, with feat_cache
        propagated across calls. Output frame count: `1 + 4*(num_latent_frames - 1)`.
        """
        x = self.post_quant_conv(z)

        num_slots = self._count_decoder_cache_slots()
        feat_cache: list = [None] * num_slots

        num_frame = x.shape[2]
        out: mx.array | None = None
        for i in range(num_frame):
            feat_idx = [0]
            chunk = x[:, :, i : i + 1]
            chunk_out = self.decoder(chunk, feat_cache=feat_cache, feat_idx=feat_idx)
            out = chunk_out if out is None else mx.concatenate([out, chunk_out], axis=2)

        return mx.clip(out, -1, 1)
