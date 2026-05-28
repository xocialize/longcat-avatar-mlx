"""MLX port of LongCat-Video-Avatar-1.5 audio adapter (AudioProjModel).

PyTorch reference: `refs/longcat-video/longcat_video/modules/avatar/blocks.py`.

`AudioProjModel` is a 3-layer MLP that ingests grouped-pooled Whisper hidden
states and produces 32 audio context tokens per VAE latent frame, at dim
`output_dim=768`. Two parallel projections (`proj1` for the first frame's
audio window, `proj1_vf` for VAE-scale-aligned subsequent frames) feed into
a shared `proj2 → proj3` stack, with optional output LayerNorm.

v1.5 config (from notes/config-snapshot/avatar-1.5--base_model-config.json):
- seq_len = audio_window = 5         (Whisper feature window per latent)
- blocks = audio_block = 5           (group-pooled Whisper layers, NOT 12!)
- channels = audio_channel = 1280    (Whisper-large hidden dim)
- intermediate_dim = 512
- output_dim = 768
- context_tokens = 32
- vae_scale = 4

The v1.0 / Wav2Vec2 defaults differ (blocks=12, channels=768). Construct
via `from_config` to pick up the right values.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class AudioProjModel(nn.Module):
    """3-layer MLP audio adapter. Output: `(B, T_latent, context_tokens, output_dim)`."""

    def __init__(
        self,
        seq_len: int = 5,
        seq_len_vf: int = 8,  # default seq_len + vae_scale - 1 = 5 + 4 - 1
        blocks: int = 5,
        channels: int = 1280,
        intermediate_dim: int = 512,
        output_dim: int = 768,
        context_tokens: int = 32,
        norm_output_audio: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.seq_len_vf = seq_len_vf
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.input_dim_vf = seq_len_vf * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim)
        self.proj1_vf = nn.Linear(self.input_dim_vf, intermediate_dim)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim)
        # PT: `self.norm = nn.LayerNorm(output_dim) if norm_output_audio else nn.Identity()`.
        # When `norm_output_audio=False`, PT uses Identity — no params, no key in checkpoint.
        # We mirror that by conditional attribute.
        self.norm_output_audio = norm_output_audio
        if norm_output_audio:
            self.norm = nn.LayerNorm(output_dim)

    def __call__(self, audio_embeds: mx.array, audio_embeds_vf: mx.array) -> mx.array:
        """Args:
            audio_embeds: `[B, 1, W=seq_len, S=blocks, C=channels]` — first frame's audio window
            audio_embeds_vf: `[B, T-1, W'=seq_len_vf, S=blocks, C=channels]` — subsequent windows

        Returns: `[B, video_length, context_tokens, output_dim]` where
        `video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]`.
        """
        video_length = audio_embeds.shape[1] + audio_embeds_vf.shape[1]
        B = audio_embeds.shape[0]

        # First frame branch — flatten (B, F, W, S, C) -> (B*F, W*S*C)
        bz, f, w, s, c = audio_embeds.shape
        ae = audio_embeds.reshape(bz * f, w * s * c)
        ae = nn.relu(self.proj1(ae))  # [B*F, intermediate]

        # Latter frame branch
        bz_v, f_v, w_v, s_v, c_v = audio_embeds_vf.shape
        ae_vf = audio_embeds_vf.reshape(bz_v * f_v, w_v * s_v * c_v)
        ae_vf = nn.relu(self.proj1_vf(ae_vf))

        # Reshape back to (B, F, intermediate) and concat over time
        ae = ae.reshape(B, f, self.intermediate_dim)
        ae_vf = ae_vf.reshape(B, f_v, self.intermediate_dim)
        ae_c = mx.concatenate([ae, ae_vf], axis=1)  # [B, T_latent, intermediate]

        # Shared proj2 and proj3 (per-token application, reshape for batched Linear)
        Bc, N_t, C_a = ae_c.shape
        ae_c = ae_c.reshape(Bc * N_t, C_a)
        ae_c = nn.relu(self.proj2(ae_c))
        ctx = self.proj3(ae_c).reshape(Bc * N_t, self.context_tokens, self.output_dim)

        if self.norm_output_audio:
            ctx = self.norm(ctx)

        ctx = ctx.reshape(B, video_length, self.context_tokens, self.output_dim)
        return ctx
