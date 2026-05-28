"""MLX port of LongCat-Video-Avatar-1.5 audio cross-attention.

PyTorch reference: `refs/longcat-video/longcat_video/modules/avatar/attention.py`.

Two classes:
- `Attention` — visual self-attention. For S1.5 (current), this is a thin
  re-export of the base `Attention` class. The Reference Skip Q-slicing
  branching (when `mask_frame_range > 0` for video continuation) will land
  in S1.6 as additional forward branches that take effect ONLY when those
  args are provided. Parameter names are identical.
- `SingleStreamAttention` — the audio cross-attention. Visual tokens are Q,
  audio context tokens (from `AudioProjModel`) are K/V. Supports MultiTalk
  L-RoPE positional routing for 2-person conversations.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.models.attention import Attention as _BaseAttention
from longcat_video_avatar.models.blocks import RMSNorm_FP32
from longcat_video_avatar.models.rope_3d import RotaryPositionalEmbedding1D


class Attention(_BaseAttention):
    """Avatar self-attention with Reference Skip Q-slicing.

    Subclasses the base `Attention` (identical parameters). When invoked with
    `num_cond_latents > 0` AND `mask_frame_range > 0`, splits the noise-region
    Q into front / maskref / back chunks; the maskref chunk attends only to
    non-reference K/V to prevent the reference image from inducing repetitive
    motion in nearby frames.

    Reference: `avatar/attention.py:Attention.forward` lines 165-194 in PT.
    Single-talker inference (no `ref_target_masks`) returns `x_ref_attn_map=None`.
    """

    def __call__(
        self,
        x,
        shape,
        num_cond_latents=None,
        return_kv: bool = False,
        num_ref_latents=None,
        ref_img_index=None,
        mask_frame_range=None,
        ref_target_masks=None,
    ):
        """Returns either `(x, x_ref_attn_map)` or `(x, kv_cache, x_ref_attn_map)`."""
        import mlx.core as mx

        B, N, C = x.shape
        T_lat = shape[0]
        tokens_per_frame = N // T_lat

        # Standard QKV + QKNorm + 3D RoPE (mirrors base, but we need q/k/v locally
        # for the Reference Skip branching).
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).transpose(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)

        kv_cache_out = None
        if return_kv:
            kv_cache_out = (k, v)

        # 3D RoPE (Avatar variant supports frame_index/num_ref_latents; we pass None
        # for the base case and use those args when the reference image is at a
        # specific temporal position)
        q, k = self.rope_3d(q, k, shape, frame_index=ref_img_index, num_ref_latents=num_ref_latents)

        # Reference Skip Q-slicing path (video continuation with masked range).
        # Activates only when num_cond_latents > 1 AND mask_frame_range > 0.
        if (
            num_cond_latents is not None
            and num_cond_latents > 1
            and num_ref_latents is not None
            and ref_img_index is not None
            and mask_frame_range is not None
            and mask_frame_range > 0
        ):
            num_ref_thw = tokens_per_frame  # one ref frame's worth of tokens
            ncl_thw = num_cond_latents * tokens_per_frame

            q_ref = q[:, :, :num_ref_thw]
            k_ref = k[:, :, :num_ref_thw]
            v_ref = v[:, :, :num_ref_thw]
            x_ref = self._process_attn(q_ref, k_ref, v_ref)

            q_cond = q[:, :, num_ref_thw:ncl_thw]
            k_cond = k[:, :, num_ref_thw:ncl_thw]
            v_cond = v[:, :, num_ref_thw:ncl_thw]
            x_cond = self._process_attn(q_cond, k_cond, v_cond)

            num_noisy_frames = T_lat - num_cond_latents
            if num_cond_latents == T_lat:
                # No noise queries — short-circuit
                out = mx.concatenate([x_ref, x_cond], axis=2)
            else:
                q_noise = q[:, :, ncl_thw:]

                start_noise = ref_img_index - mask_frame_range - num_cond_latents + num_ref_latents
                end_noise = ref_img_index + mask_frame_range - num_cond_latents + num_ref_latents + 1

                if start_noise >= 0 and end_noise > start_noise and end_noise <= num_noisy_frames:
                    start_pos = start_noise * tokens_per_frame
                    end_pos = end_noise * tokens_per_frame

                    q_noise_front = q_noise[:, :, :start_pos]
                    q_noise_maskref = q_noise[:, :, start_pos:end_pos]
                    q_noise_back = q_noise[:, :, end_pos:]

                    k_non_ref = k[:, :, num_ref_thw:]
                    v_non_ref = v[:, :, num_ref_thw:]

                    x_noise_front = self._process_attn(q_noise_front, k, v)
                    x_noise_back = self._process_attn(q_noise_back, k, v)
                    x_noise_maskref = self._process_attn(q_noise_maskref, k_non_ref, v_non_ref)
                    x_noise = mx.concatenate(
                        [x_noise_front, x_noise_maskref, x_noise_back], axis=2
                    )
                else:
                    x_noise = self._process_attn(q_noise, k, v)
                out = mx.concatenate([x_ref, x_cond, x_noise], axis=2)
        elif num_cond_latents is not None and num_cond_latents > 0:
            # Standard cond branching (matches base)
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

        out = out.transpose(0, 2, 1, 3).reshape(B, N, C)
        out = self.proj(out)

        # `x_ref_attn_map` is computed only when `ref_target_masks` is provided
        # (MultiTalk routing). For single-talker inference it's None.
        x_ref_attn_map = None
        if ref_target_masks is not None:
            # Not implemented yet — single-talker is the v3 priority. Surfacing
            # the API hook so the cross-attention can detect MultiTalk mode.
            raise NotImplementedError(
                "MultiTalk x_ref_attn_map computation deferred — "
                "implement get_attn_map_with_target in audio_process when needed."
            )

        if return_kv:
            return out, kv_cache_out, x_ref_attn_map
        return out, x_ref_attn_map


def _normalize_and_scale(column, source_range, target_range, epsilon=1e-8):
    """Map values in `column` from `source_range` linearly into `target_range`."""
    src_lo, src_hi = source_range
    new_lo, new_hi = target_range
    norm = (column - src_lo) / (src_hi - src_lo + epsilon)
    return norm * (new_hi - new_lo) + new_lo


class SingleStreamAttention(nn.Module):
    """Audio cross-attention with optional MultiTalk L-RoPE routing.

    Visual tokens are Q (from the DiT hidden state, dim=hidden_size). Audio
    context tokens are K/V (from AudioProjModel, dim=output_dim=768). Each
    video latent frame attends to its own 32 audio tokens.

    When `x_ref_attn_map` is provided (MultiTalk mode), 1D RoPE positions
    are applied to Q and K to spatially route human1/human2/background audio
    to the right regions.
    """

    def __init__(
        self,
        dim: int,
        encoder_hidden_states_dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        qk_norm: bool = True,
        eps: float = 1e-6,
        class_range: int = 24,
        class_interval: int = 4,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.dim = dim
        self.encoder_hidden_states_dim = encoder_hidden_states_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.q_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else None

        self.proj = nn.Linear(dim, dim)

        self.kv_linear = nn.Linear(encoder_hidden_states_dim, dim * 2, bias=qkv_bias)
        self.k_norm = RMSNorm_FP32(self.head_dim, eps=eps) if qk_norm else None

        # MultiTalk L-RoPE — per-block 1D rotary applied at human-routed positions.
        # `class_range`, `class_interval` define which positions encode human1 vs
        # human2 vs background.
        self.class_interval = class_interval
        self.class_range = class_range
        self.rope_h1 = (0, class_interval)
        self.rope_h2 = (class_range - class_interval, class_range)
        self.rope_bak = int(class_range // 2)
        self.rope_1d = RotaryPositionalEmbedding1D(self.head_dim)

    def _process_cross_attn(
        self,
        x: mx.array,
        cond: mx.array,
        frames_num: int,
        x_ref_attn_map: Optional[mx.array] = None,
        human_num: Optional[int] = None,
    ) -> mx.array:
        """x: [B, N_t * S, C] (visual). cond: [B*N_t, M, C_a] (audio tokens per frame)."""
        N_t = frames_num
        out_dtype = x.dtype

        # Reshape so each frame's visual tokens are separate
        # [B, N_t*S, C] -> [B*N_t, S, C]
        x = x.reshape(x.shape[0] * N_t, -1, x.shape[-1])
        B, N, C = x.shape

        # Q from visual tokens
        q = self.q_linear(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        if self.q_norm is not None:
            q = self.q_norm(q)

        # MultiTalk routing: apply 1D RoPE to Q at positions derived from x_ref_attn_map
        if x_ref_attn_map is not None:
            # x_ref_attn_map: [num_humans, S] — per-human spatial probability maps.
            # We map each visual position to a human/background bucket and apply the
            # rotary at the corresponding position.
            human_num_eff = x_ref_attn_map.shape[0] if human_num is None else human_num
            max_v = x_ref_attn_map.max(axis=1)  # [num_humans]
            min_v = x_ref_attn_map.min(axis=1)
            h1_max, h1_min = float(max_v[0].item()), float(min_v[0].item())
            h2_max, h2_min = float(max_v[1].item()), float(min_v[1].item())
            human1 = _normalize_and_scale(x_ref_attn_map[0], (h1_min, h1_max), self.rope_h1)
            human2 = _normalize_and_scale(x_ref_attn_map[1], (h2_min, h2_max), self.rope_h2)
            background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
            back = mx.full((x_ref_attn_map.shape[1],), background_pos, dtype=human1.dtype)
            max_indices = mx.minimum(mx.argmax(x_ref_attn_map, axis=0), 2)
            stacked = mx.stack([human1, human2, back], axis=1)
            normalized_pos = mx.take_along_axis(stacked, max_indices[:, None], axis=1).squeeze(-1)

            q = q.reshape(B // N_t, self.num_heads, N_t * N, self.head_dim)
            q = self.rope_1d(q, normalized_pos)
            q = q.reshape(B, self.num_heads, N, self.head_dim)

        # K, V from audio context tokens
        N_a = cond.shape[1]
        encoder_kv = self.kv_linear(cond).reshape(B, N_a, 2, self.num_heads, self.head_dim)
        encoder_kv = encoder_kv.transpose(2, 0, 3, 1, 4)  # [2, B, H, N_a, D]
        encoder_k, encoder_v = encoder_kv[0], encoder_kv[1]
        if self.k_norm is not None:
            encoder_k = self.k_norm(encoder_k)

        if x_ref_attn_map is not None:
            per_frame = mx.zeros((N_a,), dtype=encoder_k.dtype)
            human1_pos = (self.rope_h1[0] + self.rope_h1[1]) / 2
            human2_pos = (self.rope_h2[0] + self.rope_h2[1]) / 2
            if human_num is not None and human_num > 2:
                background_pos = self.rope_bak if x_ref_attn_map.shape[0] <= 3 else 100
                tokens_per_human = N_a // human_num
                # Build per_frame via slice assignment
                per_frame_list = [human1_pos] * tokens_per_human
                per_frame_list += [human2_pos] * tokens_per_human
                per_frame_list += [background_pos] * (N_a - 2 * tokens_per_human)
                per_frame = mx.array(per_frame_list, dtype=encoder_k.dtype)
            else:
                half = N_a // 2
                per_frame = mx.concatenate(
                    [
                        mx.full((half,), human1_pos, dtype=encoder_k.dtype),
                        mx.full((N_a - half,), human2_pos, dtype=encoder_k.dtype),
                    ],
                    axis=0,
                )
            encoder_pos = mx.concatenate([per_frame] * N_t, axis=0)
            encoder_k = encoder_k.reshape(B // N_t, self.num_heads, N_t * N_a, self.head_dim)
            encoder_k = self.rope_1d(encoder_k, encoder_pos)
            encoder_k = encoder_k.reshape(B, self.num_heads, N_a, self.head_dim)

        # SDPA: attention over audio K/V
        x_attn = mx.fast.scaled_dot_product_attention(q, encoder_k, encoder_v, scale=self.scale)
        # [B*N_t, H, S, D] -> [B*N_t, S, H, D] -> [B*N_t, S, C]
        x_attn = x_attn.transpose(0, 2, 1, 3).reshape(B, N, C)
        x_attn = self.proj(x_attn)

        # Reshape back: [B*N_t, S, C] -> [B_orig, N_t*S, C]
        x_attn = x_attn.reshape(B // N_t, N_t * N, C)
        return x_attn.astype(out_dtype)

    def __call__(
        self,
        x: mx.array,
        cond: mx.array,
        shape: tuple[int, int, int],
        num_cond_latents: Optional[int] = None,
        x_ref_attn_map: Optional[mx.array] = None,
        human_num: Optional[int] = None,
    ):
        """Args:
            x: [B, N_visual, C]
            cond: [B*N_t, audio_tokens, C_audio] (per-frame audio tokens)
            shape: (T, H, W)

        Returns: (audio_output_cond, audio_output_noise). When
        num_cond_latents=0, audio_output_cond is None and audio_output_noise
        covers all visual tokens. When num_cond_latents > 0, audio_output_cond
        is a tensor of zeros for the cond region and audio_output_noise covers
        only the noise region.
        """
        B, N, C = x.shape

        if num_cond_latents is None or num_cond_latents == 0:
            output = self._process_cross_attn(x, cond, shape[0], x_ref_attn_map, human_num)
            return None, output

        assert num_cond_latents > 0
        ncl_thw = num_cond_latents * (N // shape[0])
        x_noise = x[:, ncl_thw:]
        # Drop the cond rows from cond too: cond is [B*N_t, M, C], rearrange
        # [B*N_t, M, C] -> [B, N_t, M, C] -> drop first num_cond_latents frames -> [B*(N_t-ncl), M, C]
        cond_r = cond.reshape(x.shape[0], shape[0], cond.shape[1], cond.shape[2])
        cond_r = cond_r[:, num_cond_latents:]
        cond_r = cond_r.reshape(x.shape[0] * (shape[0] - num_cond_latents), cond.shape[1], cond.shape[2])

        frames_num = shape[0] - num_cond_latents
        if human_num is not None and human_num >= 2:
            output_noise = self._process_cross_attn(x_noise, cond_r, frames_num, x_ref_attn_map, human_num)
        else:
            output_noise = self._process_cross_attn(x_noise, cond_r, frames_num)
        output_cond = mx.zeros((B, ncl_thw, C), dtype=output_noise.dtype)
        return output_cond, output_noise
