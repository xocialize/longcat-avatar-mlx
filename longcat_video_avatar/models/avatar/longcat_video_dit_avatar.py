"""MLX port of `LongCatVideoAvatarTransformer3DModel` (Avatar 1.5).

PyTorch reference: `refs/longcat-video/longcat_video/modules/avatar/longcat_video_dit_avatar.py`.

The Avatar transformer extends the base DiT by inserting an audio cross-attn
layer between text cross-attn and the FFN, gated by per-block AdaLN
(`audio_adaLN_modulation`). The audio path runs once at the top of forward
via `AudioProjModel`, producing 32 audio context tokens per latent frame.

Module hierarchy mirrors PT — `blocks.{B}.attn`, `blocks.{B}.cross_attn`,
`blocks.{B}.audio_cross_attn`, `blocks.{B}.audio_adaLN_modulation.1`, etc.
"""

from __future__ import annotations

from typing import Optional, Tuple

import math

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.models.attention import MultiHeadCrossAttention
from longcat_video_avatar.models.avatar.attention import Attention, SingleStreamAttention
from longcat_video_avatar.models.avatar.blocks import AudioProjModel
from longcat_video_avatar.models.blocks import (
    CaptionEmbedder,
    FeedForwardSwiGLU,
    FinalLayer_FP32,
    LayerNorm_FP32,
    PatchEmbed3D,
    TimestepEmbedder,
    modulate_fp32,
)


class LongCatAvatarSingleStreamBlock(nn.Module):
    """Avatar DiT block: self-attn + text cross-attn + audio cross-attn + FFN.

    AdaLN-Zero on self-attn (6-param) AND on audio cross-attn output (3-param,
    via `audio_adaLN_modulation`). Text cross-attn is residual-only (no AdaLN).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: int,
        adaln_tembed_dim: int,
        output_dim: int = 768,  # audio token dim from AudioProjModel
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        # PT: adaLN_modulation = nn.Sequential(SiLU, Linear(t_dim, 6*hidden))
        # PT: audio_adaLN_modulation = nn.Sequential(SiLU, Linear(t_dim, 3*hidden))
        self.adaLN_modulation = [None, nn.Linear(adaln_tembed_dim, 6 * hidden_size, bias=True)]
        self.audio_adaLN_modulation = [None, nn.Linear(adaln_tembed_dim, 3 * hidden_size, bias=True)]

        # Norms
        self.mod_norm_attn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mod_norm_ffn = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=False)
        self.pre_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)

        # Pre-norms specific to the audio path
        self.pre_video_crs_attn_norm = LayerNorm_FP32(hidden_size, eps=1e-6, elementwise_affine=True)
        # PT: pre_audio_crs_attn_norm is LayerNorm(output_dim, affine=True) when
        # audio_prenorm=True, else nn.Identity (no params). We mirror with conditional attr.
        self.audio_prenorm = audio_prenorm
        if audio_prenorm:
            self.pre_audio_crs_attn_norm = LayerNorm_FP32(output_dim, eps=1e-6, elementwise_affine=True)

        self.attn = Attention(dim=hidden_size, num_heads=num_heads)
        self.cross_attn = MultiHeadCrossAttention(dim=hidden_size, num_heads=num_heads)
        self.audio_cross_attn = SingleStreamAttention(
            dim=hidden_size,
            encoder_hidden_states_dim=output_dim,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            class_range=class_range,
            class_interval=class_interval,
        )

        self.ffn = FeedForwardSwiGLU(dim=hidden_size, hidden_dim=int(hidden_size * mlp_ratio))

    def _maybe_pre_audio_norm(self, x: mx.array) -> mx.array:
        return self.pre_audio_crs_attn_norm(x) if self.audio_prenorm else x

    def __call__(
        self,
        x: mx.array,
        y: mx.array,
        t: mx.array,
        y_seqlen: list[int],
        latent_shape: Tuple[int, int, int],
        audio_hidden_states: mx.array,
        num_cond_latents: Optional[int] = None,
        skip_crs_attn: bool = False,
        human_num: Optional[int] = None,
        num_ref_latents: Optional[int] = None,
        ref_img_index: Optional[int] = None,
        mask_frame_range: Optional[int] = None,
        ref_target_masks: Optional[mx.array] = None,
    ) -> mx.array:
        """One Avatar DiT block.

        Args:
            audio_hidden_states: `[B*N_t, audio_tokens, audio_dim]` per-frame audio tokens.
            num_ref_latents/ref_img_index/mask_frame_range: Reference Skip Q-slicing.
            ref_target_masks: MultiTalk per-human spatial masks (None for single talker).
        """
        x_dtype = x.dtype
        B, N, C = x.shape
        T, _, _ = latent_shape

        # AdaLN params (fp32)
        t_in = nn.silu(t)
        ada = self.adaLN_modulation[1](t_in).astype(mx.float32)[:, :, None, :]  # [B, T, 1, 6*C]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(ada, 6, axis=-1)

        ncl = 0 if num_cond_latents is None else num_cond_latents
        # Audio AdaLN params: only for the noise region (t[:, num_cond_latents:])
        audio_t_in = nn.silu(t[:, ncl:]) if ncl > 0 else t_in
        audio_ada = self.audio_adaLN_modulation[1](audio_t_in).astype(mx.float32)[:, :, None, :]
        audio_shift_mca, audio_scale_mca, audio_gate_mca = mx.split(audio_ada, 3, axis=-1)

        # Self-attn (Avatar's Attention returns x_ref_attn_map for MultiTalk routing)
        x_m = modulate_fp32(self.mod_norm_attn, x.reshape(B, T, -1, C), shift_msa, scale_msa).reshape(B, N, C)
        x_s, x_ref_attn_map = self.attn(
            x_m,
            latent_shape,
            num_cond_latents=num_cond_latents,
            num_ref_latents=num_ref_latents,
            ref_img_index=ref_img_index,
            mask_frame_range=mask_frame_range,
            ref_target_masks=ref_target_masks,
        )
        gate_msa_f = gate_msa.astype(mx.float32)
        x_s_f = x_s.reshape(B, T, -1, C).astype(mx.float32)
        x = (x.astype(mx.float32) + (gate_msa_f * x_s_f).reshape(B, N, C)).astype(x_dtype)

        # Text cross-attn (no AdaLN)
        if not skip_crs_attn:
            x = x + self.cross_attn(
                self.pre_crs_attn_norm(x), y, y_seqlen, num_cond_latents=num_cond_latents, shape=latent_shape
            )

        # Audio cross-attn (AdaLN-gated output)
        if not skip_crs_attn:
            audio_out_cond, audio_out_noise = self.audio_cross_attn(
                self.pre_video_crs_attn_norm(x),
                self._maybe_pre_audio_norm(audio_hidden_states),
                shape=latent_shape,
                num_cond_latents=num_cond_latents,
                x_ref_attn_map=x_ref_attn_map,
                human_num=human_num,
            )
            T_noise = T - ncl
            audio_out_noise = modulate_fp32(
                self.mod_norm_attn,
                audio_out_noise.reshape(B, T_noise, -1, C),
                audio_shift_mca,
                audio_scale_mca,
            ).reshape(B, -1, C)
            gate_a_f = audio_gate_mca.astype(mx.float32)
            audio_add = (gate_a_f * audio_out_noise.reshape(B, T_noise, -1, C).astype(mx.float32)).reshape(B, -1, C)
            if audio_out_cond is not None:
                audio_add = mx.concatenate([audio_out_cond.astype(mx.float32), audio_add], axis=1)
            x = (x.astype(mx.float32) + audio_add).astype(x_dtype)

        # FFN
        x_m = modulate_fp32(self.mod_norm_ffn, x.reshape(B, T, -1, C), shift_mlp, scale_mlp).reshape(B, N, C)
        x_s = self.ffn(x_m)
        gate_mlp_f = gate_mlp.astype(mx.float32)
        x_s_f = x_s.reshape(B, T, -1, C).astype(mx.float32)
        x = (x.astype(mx.float32) + (gate_mlp_f * x_s_f).reshape(B, N, C)).astype(x_dtype)

        return x


class LongCatVideoAvatarTransformer3DModel(nn.Module):
    """Avatar-1.5 DiT: base DiT + audio path + per-block audio cross-attn.

    Class name matches diffusers `_class_name: LongCatVideoAvatarTransformer3DModel`.
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
        # Audio config
        audio_window: int = 5,
        audio_block: int = 5,
        audio_channel: int = 1280,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        vae_scale: int = 4,
        audio_prenorm: bool = False,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__()
        assert patch_size[0] == 1, "Temporal patchify must be 1"

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.text_tokens_zero_pad = text_tokens_zero_pad
        self.depth = depth
        self.vae_scale = vae_scale
        self.audio_window = audio_window

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(
            t_embed_dim=adaln_tembed_dim, frequency_embedding_size=frequency_embedding_size
        )
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size)

        self.blocks = [
            LongCatAvatarSingleStreamBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                adaln_tembed_dim=adaln_tembed_dim,
                output_dim=output_dim,
                audio_prenorm=audio_prenorm,
                class_range=class_range,
                class_interval=class_interval,
            )
            for _ in range(depth)
        ]

        self.audio_proj = AudioProjModel(
            seq_len=audio_window,
            seq_len_vf=audio_window + vae_scale - 1,
            blocks=audio_block,
            channels=audio_channel,
            intermediate_dim=intermediate_dim,
            output_dim=output_dim,
            context_tokens=context_tokens,
        )

        self.final_layer = FinalLayer_FP32(
            hidden_size, int(math.prod(self.patch_size)), out_channels, adaln_tembed_dim
        )

    @classmethod
    def from_config(cls, config: dict) -> "LongCatVideoAvatarTransformer3DModel":
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
            audio_window=config.get("audio_window", 5),
            audio_block=config.get("audio_block", 5),
            audio_channel=config.get("audio_channel", 1280),
            intermediate_dim=config.get("intermediate_dim", 512),
            output_dim=config.get("output_dim", 768),
            context_tokens=config.get("context_tokens", 32),
            vae_scale=config.get("vae_scale", 4),
            audio_prenorm=config.get("audio_prenorm", False),
            class_range=config.get("class_range", 24),
            class_interval=config.get("class_interval", 4),
        )

    def _prepare_audio_hidden_states(self, audio_embs: mx.array, num_ref_latents: int, N_t: int) -> mx.array:
        """Mirror the audio windowing in PT (longcat_video_dit_avatar.py:420-441).

        audio_embs comes from the LongCat audio pipeline with shape
        `[B, T_frames, W=5, S=blocks, C]`. We split first-frame vs later frames,
        re-window per-vae-scale, run AudioProjModel, and reshape to per-latent-frame.
        """
        B, T_frames = audio_embs.shape[0], audio_embs.shape[1]
        first_frame_audio = audio_embs[:, :1]  # [B, 1, W, S, C]
        latter_frame_audio = audio_embs[:, 1:]  # [B, T-1, W, S, C]

        # Re-window per VAE temporal stride
        v = self.vae_scale
        w = self.audio_window
        middle = w // 2
        # Reshape (T-1) along vae_scale chunks: [B, (n_t * v), W, S, C] -> [B, n_t, v, W, S, C]
        b_, t_, w_, s_, c_ = latter_frame_audio.shape
        n_t = t_ // v
        latter = latter_frame_audio.reshape(b_, n_t, v, w_, s_, c_)
        # First sub-frame: first 'middle+1' tokens of its W window
        l_first = latter[:, :, :1, : middle + 1]  # [B, n_t, 1, middle+1, S, C]
        l_first = l_first.reshape(b_, n_t, (middle + 1) * 1, s_, c_)
        # Middle frames: just the middle token of their W window
        l_mid = latter[:, :, 1:-1, middle : middle + 1]
        l_mid = l_mid.reshape(b_, n_t, 1 * max(0, v - 2), s_, c_) if v > 2 else mx.zeros((b_, n_t, 0, s_, c_))
        # Last sub-frame: last 'W - middle' tokens of its W window
        l_last = latter[:, :, -1:, middle:]
        l_last = l_last.reshape(b_, n_t, (w_ - middle) * 1, s_, c_)
        latter_frame_audio_s = mx.concatenate([l_first, l_mid, l_last], axis=2)
        # Shape now: [B, n_t, W' = (middle+1)+(v-2)+(W-middle) = W + v - 1, S, C]

        audio_hidden_states = self.audio_proj(first_frame_audio, latter_frame_audio_s)
        # [B, video_length, context_tokens, output_dim]

        if num_ref_latents is not None and num_ref_latents > 0:
            # Pad with a copy of the first frame for the reference latent
            audio_start_ref = audio_hidden_states[:, :1]
            audio_hidden_states = mx.concatenate([audio_start_ref, audio_hidden_states], axis=1)

        audio_hidden_states = audio_hidden_states[:, -N_t:]
        return audio_hidden_states

    def __call__(
        self,
        hidden_states: mx.array,
        timestep: mx.array,
        encoder_hidden_states: mx.array,
        audio_embs: mx.array,
        encoder_attention_mask: Optional[mx.array] = None,
        num_cond_latents: int = 0,
        num_ref_latents: Optional[int] = None,
        human_num: Optional[int] = None,
        ref_img_index: Optional[int] = None,
        mask_frame_range: Optional[int] = None,
    ) -> mx.array:
        """Avatar DiT forward.

        Args:
            hidden_states: [B, C_in, T, H, W]
            timestep: [B] or [B, T]
            encoder_hidden_states: [B, 1, N_text, C_text]
            audio_embs: [B, T_audio, W, S, C_a] Whisper-pooled audio features
            num_cond_latents: int, count of conditioning latents at the temporal head
            num_ref_latents: optional reference image latent count (for video continuation)
        """
        B, _, T, H, W = hidden_states.shape
        N_t = T // self.patch_size[0]
        N_h = H // self.patch_size[1]
        N_w = W // self.patch_size[2]

        if timestep.ndim == 1:
            timestep = mx.broadcast_to(timestep[:, None], (B, N_t))

        dtype = self.x_embedder.proj.weight.dtype
        hidden_states = hidden_states.astype(dtype)
        timestep = timestep.astype(dtype)
        encoder_hidden_states = encoder_hidden_states.astype(dtype)
        audio_embs = audio_embs.astype(dtype)

        hidden_states = self.x_embedder(hidden_states)  # [B, N, C]

        t = self.t_embedder(timestep.astype(mx.float32).flatten(), dtype=mx.float32).reshape(B, N_t, -1)

        encoder_hidden_states = self.y_embedder(encoder_hidden_states)  # [B, 1, N_text, C]

        # Audio: produce per-latent-frame audio context tokens
        audio_hidden_states = self._prepare_audio_hidden_states(audio_embs, num_ref_latents or 0, N_t)
        # Reshape [B, T_lat, M, C] -> [B*T_lat, M, C] for per-frame cross-attn
        audio_hidden_states = audio_hidden_states.reshape(B * N_t, audio_hidden_states.shape[2], audio_hidden_states.shape[3])

        # text_tokens_zero_pad + pack
        if self.text_tokens_zero_pad and encoder_attention_mask is not None:
            mask_b = encoder_attention_mask
            if mask_b.ndim == 4:
                mask_b = mask_b.squeeze(1).squeeze(1)
            encoder_hidden_states = encoder_hidden_states * mask_b[:, None, :, None]
            encoder_attention_mask = mx.ones_like(mask_b)

        if encoder_attention_mask is not None:
            mask_2d = encoder_attention_mask
            if mask_2d.ndim == 4:
                mask_2d = mask_2d.squeeze(1).squeeze(1)
            elif mask_2d.ndim == 3:
                mask_2d = mask_2d.squeeze(1)
            y_seqlens = [int(mx.sum(mask_2d[b]).item()) for b in range(B)]
            ehs = encoder_hidden_states.squeeze(1)
            packed_parts = [ehs[b, : y_seqlens[b]] for b in range(B)]
            encoder_hidden_states = mx.concatenate(packed_parts, axis=0)[None, :, :]
        else:
            ehs = encoder_hidden_states.squeeze(1)
            y_seqlens = [ehs.shape[1]] * B
            encoder_hidden_states = ehs.reshape(1, -1, ehs.shape[-1])

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states,
                t,
                y_seqlens,
                (N_t, N_h, N_w),
                audio_hidden_states=audio_hidden_states,
                num_cond_latents=num_cond_latents,
                human_num=human_num,
                num_ref_latents=num_ref_latents,
                ref_img_index=ref_img_index,
                mask_frame_range=mask_frame_range,
            )

        hidden_states = self.final_layer(hidden_states, t, (N_t, N_h, N_w))
        hidden_states = self._unpatchify(hidden_states, N_t, N_h, N_w)
        return hidden_states.astype(mx.float32)

    def _unpatchify(self, x: mx.array, N_t: int, N_h: int, N_w: int) -> mx.array:
        T_p, H_p, W_p = self.patch_size
        B = x.shape[0]
        x = x.reshape(B, N_t, N_h, N_w, T_p, H_p, W_p, self.out_channels)
        x = x.transpose(0, 7, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, self.out_channels, N_t * T_p, N_h * H_p, N_w * W_p)
