"""MLX port of LongCat-Video DiT (base; no Avatar/audio overlay).

PyTorch reference: `refs/longcat-video/longcat_video/modules/longcat_video_dit.py`.

Class names + module attribute names mirror the PT source 1:1 so weights load
from `meituan-longcat/LongCat-Video/dit/*.safetensors` with only the standard
Conv*d-layout transpose at load time, no key remapping.

Avatar overlay (audio cross-attn, Reference Skip) is in
`longcat_video_avatar/models/avatar/` and ports the Avatar deltas only.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.models.attention import Attention, MultiHeadCrossAttention
from longcat_video_avatar.models.blocks import (
    CaptionEmbedder,
    FeedForwardSwiGLU,
    FinalLayer_FP32,
    LayerNorm_FP32,
    PatchEmbed3D,
    TimestepEmbedder,
    modulate_fp32,
)


class LongCatSingleStreamBlock(nn.Module):
    """One DiT block: self-attn + text cross-attn + SwiGLU FFN.

    AdaLN-Zero modulation on self-attn (6-param: shift/scale/gate × msa/mlp).
    Text cross-attn is NOT AdaLN-modulated (residual w/ pre-norm only).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # adaLN_modulation: nn.Sequential(SiLU, Linear(adaln_tembed_dim, 6*hidden_size))
        # PT keys: adaLN_modulation.0 (SiLU, no params), adaLN_modulation.1 (Linear)
        self.adaLN_modulation = [
            None,  # SiLU
            nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True),
        ]

        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(dim=hidden_size, num_heads=num_heads)
        self.cross_attn = MultiHeadCrossAttention(dim=hidden_size, num_heads=num_heads)
        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        t: mx.array,
        y_seqlen: list[int],
        latent_shape: Tuple[int, int, int],
        num_cond_latents: Optional[int] = None,
        return_kv: bool = False,
        kv_cache: Optional[tuple] = None,
        skip_crs_attn: bool = False,
    ):
        """Args:
            x: visual tokens [B, N, C]
            y: packed text [1, sum(y_seqlen), C]
            t: timestep embedding [B, T, C_t] (fp32)
            y_seqlen: per-batch text lengths
            latent_shape: (T, H, W)
        """
        x_dtype = x.dtype
        B, N, C = x.shape
        T, _, _ = latent_shape

        # adaLN params (fp32). t already fp32 by convention.
        t_in = nn.silu(t)
        ada = self.adaLN_modulation[1](t_in)  # [B, T, 6*C]
        ada = ada[:, :, None, :]  # [B, T, 1, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(ada, 6, axis=-1)

        # Self-attn with AdaLN modulation
        x_m = modulate_fp32(self.mod_norm_attn, x.reshape(B, T, -1, C), shift_msa, scale_msa).reshape(B, N, C)

        if kv_cache is not None:
            x_s = self.attn.forward_with_kv_cache(
                x_m, shape=latent_shape, num_cond_latents=num_cond_latents, kv_cache=kv_cache
            )
            new_kv = kv_cache  # forward_with_kv_cache doesn't return new cache
        elif return_kv:
            x_s, new_kv = self.attn(x_m, latent_shape, num_cond_latents=num_cond_latents, return_kv=True)
        else:
            x_s = self.attn(x_m, latent_shape, num_cond_latents=num_cond_latents)
            new_kv = None

        # Residual with gate (fp32 multiply, then back to x dtype)
        gate_msa_b = gate_msa.astype(mx.float32)
        x_s_f = x_s.reshape(B, T, -1, C).astype(mx.float32)
        x = (x.astype(mx.float32) + (gate_msa_b * x_s_f).reshape(B, N, C)).astype(x_dtype)

        # Text cross-attn (no AdaLN modulation; pre-norm + residual)
        if not skip_crs_attn:
            ncl_for_cross = None if kv_cache is not None else num_cond_latents
            x = x + self.cross_attn(
                self.pre_crs_attn_norm(x),
                y,
                y_seqlen,
                num_cond_latents=ncl_for_cross,
                shape=latent_shape,
            )

        # FFN with AdaLN modulation
        x_m = modulate_fp32(self.mod_norm_ffn, x.reshape(B, T, -1, C), shift_mlp, scale_mlp).reshape(B, N, C)
        x_s = self.ffn(x_m)
        gate_mlp_b = gate_mlp.astype(mx.float32)
        x_s_f = x_s.reshape(B, T, -1, C).astype(mx.float32)
        x = (x.astype(mx.float32) + (gate_mlp_b * x_s_f).reshape(B, N, C)).astype(x_dtype)

        if return_kv:
            return x, new_kv
        return x


class LongCatVideoTransformer3DModel(nn.Module):
    """The base LongCat-Video DiT (no audio overlay).

    Class name matches diffusers' `_class_name: LongCatVideoTransformer3DModel`
    so the safetensors and config load cleanly via `from_config(dict)`.
    """

    def __init__(
        self,
        in_channels: int = 16,
        out_channels: int = 16,
        hidden_size: int = 4096,
        depth: int = 48,
        num_heads: int = 32,
        caption_channels: int = 4096,
        mlp_ratio: int = 4,
        adaln_tembed_dim: int = 512,
        frequency_embedding_size: int = 256,
        patch_size: Tuple[int, int, int] = (1, 2, 2),
        text_tokens_zero_pad: bool = False,
    ):
        super().__init__()
        assert patch_size[0] == 1, "Temporal patchify dim must be 1 (matches Meituan assertion)"

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_tokens_zero_pad = text_tokens_zero_pad
        self.depth = depth

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)

        self.blocks = [
            LongCatSingleStreamBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                adaln_tembed_dim=adaln_tembed_dim,
            )
            for _ in range(depth)
        ]
        self.final_layer = FinalLayer_FP32(
            hidden_size,
            int(math.prod(self.patch_size)),
            out_channels,
            adaln_tembed_dim,
        )

    @classmethod
    def from_config(cls, config: dict) -> "LongCatVideoTransformer3DModel":
        return cls(
            in_channels=config.get("in_channels", 16),
            out_channels=config.get("out_channels", 16),
            hidden_size=config.get("hidden_size", 4096),
            depth=config.get("depth", 48),
            num_heads=config.get("num_heads", 32),
            caption_channels=config.get("caption_channels", 4096),
            mlp_ratio=config.get("mlp_ratio", 4),
            adaln_tembed_dim=config.get("adaln_tembed_dim", 512),
            frequency_embedding_size=config.get("frequency_embedding_size", 256),
            patch_size=tuple(config.get("patch_size", [1, 2, 2])),
            text_tokens_zero_pad=config.get("text_tokens_zero_pad", False),
        )

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
        num_cond_latents: int = 0,
    ) -> mx.array:
        """Args:
            hidden_states: [B, C_in, T, H, W] noisy latent
            timestep: [B] or [B, T] (per-frame timestep)
            encoder_hidden_states: [B, 1, N_text, C_text] text embeddings from umT5
            encoder_attention_mask: [B, 1, 1, N_text] or [B, N_text] valid-token mask

        Returns: [B, C_out, T, H, W] predicted noise (or velocity, per scheduler)
        """
        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        # Expand timestep from [B] to [B, N_t] if needed
        if timestep.ndim == 1:
            timestep = mx.broadcast_to(timestep[:, None], (B, N_t))

        # Take the embedder weight dtype as the inference dtype
        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.astype(dtype)
        timestep = timestep.astype(dtype)
        encoder_hidden_states = encoder_hidden_states.astype(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        # t_embedder runs in fp32 by convention (matches PT amp.autocast(fp32))
        t = self.t_embedder(timestep.astype(mx.float32).flatten(), dtype=mx.float32).reshape(B, N_t, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_text, C]

        # Apply text_tokens_zero_pad (zero out padding tokens, set mask to all-1s)
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            mask_b = encoder_attention_mask
            if mask_b.ndim == 4:
                mask_b = mask_b.squeeze(1).squeeze(1)  # -> [B, N_text]
            encoder_hidden_states = encoder_hidden_states * mask_b[:, None, :, None]
            encoder_attention_mask = mx.ones_like(mask_b)

        # Pack text across batch: [B, 1, N_text, C] -> [1, sum(valid_per_batch), C]
        if encoder_attention_mask is not None:
            mask_2d = encoder_attention_mask
            if mask_2d.ndim == 4:
                mask_2d = mask_2d.squeeze(1).squeeze(1)
            elif mask_2d.ndim == 3:
                mask_2d = mask_2d.squeeze(1)
            # Compute valid lengths per batch (Python list for the mask-builder)
            y_seqlens = [int(mx.sum(mask_2d[b]).item()) for b in range(B)]
            # Pack: gather only valid tokens. We use a simple boolean-mask path.
            # encoder_hidden_states is [B, 1, N_text, C]; squeeze the 1.
            ehs = encoder_hidden_states.squeeze(1)  # [B, N_text, C]
            # Concatenate valid slices from each batch into a single packed tensor.
            # (For B=1 inference this is just a slice up to y_seqlens[0].)
            packed_parts = []
            for b in range(B):
                ki = y_seqlens[b]
                packed_parts.append(ehs[b, :ki])
            encoder_hidden_states = mx.concatenate(packed_parts, axis=0)[None, :, :]  # [1, sum_k, C]
        else:
            ehs = encoder_hidden_states.squeeze(1)  # [B, N_text, C]
            y_seqlens = [ehs.shape[1]] * B
            encoder_hidden_states = ehs.reshape(1, -1, ehs.shape[-1])

        # Run blocks
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                t,
                y_seqlens,
                (N_t, N_h, N_w),
                num_cond_latents=num_cond_latents,
            )

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))
        # [B, N, C=T_p*H_p*W_p*C_out] -> [B, C_out, T_p*N_t, H_p*N_h, W_p*N_w]
        hidden_states = self._unpatchify(hidden_states, N_t, N_h, N_w)
        return hidden_states.astype(mx.float32)

    def _unpatchify(self, x: mx.array, N_t: int, N_h: int, N_w: int) -> mx.array:
        T_p, H_p, W_p = self.patch_size
        B = x.shape[0]
        # [B, N_t*N_h*N_w, T_p*H_p*W_p*C_out] -> [B, N_t, N_h, N_w, T_p, H_p, W_p, C_out]
        x = x.reshape(B, N_t, N_h, N_w, T_p, H_p, W_p, self.out_channels)
        # Permute to [B, C_out, N_t, T_p, N_h, H_p, N_w, W_p]
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        # Reshape to [B, C_out, N_t*T_p, N_h*H_p, N_w*W_p]
        return x.reshape(B, self.out_channels, N_t * T_p, N_h * H_p, N_w * W_p)
