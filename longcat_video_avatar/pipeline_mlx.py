"""High-level inference pipeline for LongCat-Video-Avatar-1.5 on MLX.

Wires together:
- umT5-XXL text encoder
- Whisper-large-v3 audio encoder (with group-mean-pool)
- Wan VAE (encode reference image; decode final latent)
- LongCatVideoAvatarTransformer3DModel DiT
- FlowMatchEulerDiscreteScheduler (from mlx-arsenal) with DMD distilled sigmas
- Pre-merged DMD LoRA
- 3-pass disentangled CFG

The primary v1 mode is **AI2V (Image-Audio-to-Video)**: a single reference
image + speech audio + text prompt → 93-frame video at 30 fps (≈3.1s).

This file is purposely synchronous and single-batch focused. Multi-chunk
continuation, MultiTalk routing, video continuation (VC), and async batching
are out of scope for v1 (deferred to v1.1).
"""

from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import numpy as np

from longcat_video_avatar.audio_process import (
    build_avatar_audio_embeddings,
    whisper_encode_audio_to_groups,
)
from longcat_video_avatar.guidance import (
    cfg_split_outputs,
    disentangled_cfg_combine,
    flip_velocity_for_scheduler,
    get_dmd_distilled_sigmas,
)
from longcat_video_avatar.lora import merge_lora_into_model
from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
    LongCatVideoAvatarTransformer3DModel,
)
from longcat_video_avatar.models.umt5 import UMT5EncoderModel
from longcat_video_avatar.models.whisper import WhisperEncoder


@dataclass
class PipelineConfig:
    """Resolved config for a pipeline instance. Built from the union of the
    per-component config.json files snapshotted in `notes/config-snapshot/`."""

    # DiT
    dit_hidden_size: int = 4096
    dit_depth: int = 48
    dit_num_heads: int = 32
    dit_patch_size: tuple = (1, 2, 2)
    dit_in_channels: int = 16
    dit_out_channels: int = 16

    # Sampler
    num_sampling_steps: int = 8
    num_train_timesteps: int = 1000
    scheduler_shift: float = 7.0

    # CFG defaults (per DMD)
    text_guidance_scale: float = 4.0
    audio_guidance_scale: float = 4.0

    # Video
    num_frames: int = 93
    target_fps: int = 25

    # Audio
    whisper_enc_fps: int = 50

    # Latent space
    vae_scale_temporal: int = 4
    vae_scale_spatial: int = 8


class LongCatAvatarPipeline:
    """Composite inference pipeline.

    Construct via `from_pretrained(weights_dir, ...)` once weights are
    downloaded. For now we expose a programmatic constructor for callers
    that have already loaded each component.

    Usage:
        pipeline = LongCatAvatarPipeline(
            vae=vae,
            text_encoder=umt5,
            audio_encoder=whisper,
            dit=avatar_dit,
            config=PipelineConfig(),
        )
        video = pipeline(
            image=ref_image_chw,      # [3, H, W] in [-1, 1]
            audio_mel=mel,            # [128, T_mel] Whisper-large-v3 mel features
            text_embeds=text_embeds,  # [1, N_text, 4096]
            text_mask=text_mask,      # [1, N_text]
            uncond_embeds=neg_embeds, # [1, N_text, 4096]
            uncond_mask=neg_mask,
            seed=0,
        )

    See `tests/smoke/test_pipeline_smoke.py` for the synthetic-weight wiring
    smoke test. End-to-end with real weights lands in S1.11.
    """

    def __init__(
        self,
        vae: AutoencoderKLWan,
        text_encoder: UMT5EncoderModel,
        audio_encoder: WhisperEncoder,
        dit: LongCatVideoAvatarTransformer3DModel,
        config: Optional[PipelineConfig] = None,
        scheduler=None,
    ):
        self.vae = vae
        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder
        self.dit = dit
        self.config = config or PipelineConfig()

        if scheduler is None:
            try:
                from mlx_arsenal.diffusion import FlowMatchEulerDiscreteScheduler

                scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.config.num_train_timesteps,
                    shift=self.config.scheduler_shift,
                )
            except ImportError as e:
                raise ImportError(
                    "mlx-arsenal is required for the default scheduler — `pip install mlx-arsenal`"
                ) from e
        self.scheduler = scheduler

    # ----- LoRA pre-merge ----------------------------------------------------

    def merge_dmd_lora(self, lora_state_dict: dict, multiplier: float = 1.0) -> dict:
        """Pre-merge the DMD LoRA into the DiT weights. Returns the
        `merge_lora_into_model` result (applied/unmapped lists).
        """
        return merge_lora_into_model(self.dit, lora_state_dict, multiplier=multiplier)

    # ----- Inference ---------------------------------------------------------

    def _make_initial_noise(self, batch_size: int, num_frames: int, height: int, width: int, seed: int) -> mx.array:
        """Sample a Gaussian noise tensor in latent space.

        Latent shape: `[B, in_channels=16, T_lat, H_lat, W_lat]` where
        `T_lat = 1 + (num_frames - 1) // vae_scale_temporal` and
        `H_lat = height // vae_scale_spatial, W_lat = width // vae_scale_spatial`.
        """
        v = self.config.vae_scale_temporal
        s = self.config.vae_scale_spatial
        T_lat = 1 + (num_frames - 1) // v
        H_lat = height // s
        W_lat = width // s
        mx.random.seed(seed)
        return mx.random.normal(
            (batch_size, self.config.dit_in_channels, T_lat, H_lat, W_lat)
        )

    def _encode_reference_image(self, image: mx.array) -> mx.array:
        """`image`: `[B, 3, 1, H, W]` in `[-1, 1]` (single-frame video).

        Returns: `[B, z_dim, 1, H_lat, W_lat]` normalized latent.
        """
        raw_mu = self.vae.encode(image)  # raw (unnormalized) latent mean
        return self.vae.normalize_latents(raw_mu)

    def _prepare_audio_embs(self, audio_mel: mx.array) -> mx.array:
        """`audio_mel`: `[B, 128, T_mel]`. Returns `[B, T_video, 5, 1280]`."""
        groups = whisper_encode_audio_to_groups(self.audio_encoder, audio_mel)
        return build_avatar_audio_embeddings(
            groups, fps=self.config.target_fps, enc_fps=self.config.whisper_enc_fps
        )

    def _cfg_forward(
        self,
        latents: mx.array,
        timestep: mx.array,
        text_embeds_cat: mx.array,
        text_mask_cat: mx.array,
        audio_embs: mx.array,
        uncond_text_embeds: mx.array,
        uncond_text_mask: mx.array,
        uncond_audio_embs: mx.array,
        num_cond_latents: int,
    ) -> mx.array:
        """One CFG step: 3-pass forward + disentangled combine + velocity flip."""
        # Pass 1: batched [latents, latents] with [neg_text, pos_text] and pos audio
        latents_2 = mx.concatenate([latents, latents], axis=0)
        # MLX uses `mx.repeat(arr, n, axis=...)` — no `.repeat()` instance method.
        # Ensure timestep is at least 1D before repeat (scheduler may return 0-dim).
        if timestep.ndim == 0:
            timestep = timestep[None]
        timestep_2 = mx.repeat(timestep, 2, axis=0)
        audio_embs_2 = mx.repeat(audio_embs, 2, axis=0)
        pred_2 = self.dit(
            latents_2,
            timestep_2,
            text_embeds_cat,
            audio_embs_2,
            encoder_attention_mask=text_mask_cat,
            num_cond_latents=num_cond_latents,
        )
        noise_uncond_text, noise_cond = cfg_split_outputs(pred_2)

        # Pass 2: fully unconditional (no text, no audio)
        pred_uncond = self.dit(
            latents,
            timestep,
            uncond_text_embeds,
            uncond_audio_embs,
            encoder_attention_mask=uncond_text_mask,
            num_cond_latents=num_cond_latents,
        )

        # Combine + flip velocity sign
        combined = disentangled_cfg_combine(
            noise_cond,
            noise_uncond_text,
            pred_uncond,
            text_guidance_scale=self.config.text_guidance_scale,
            audio_guidance_scale=self.config.audio_guidance_scale,
        )
        return flip_velocity_for_scheduler(combined)

    def __call__(
        self,
        image: mx.array,
        audio_mel: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_frames: Optional[int] = None,
        height: int = 480,
        width: int = 832,
        seed: int = 0,
    ) -> mx.array:
        """Run the full denoising loop and return decoded video.

        Returns video tensor `[1, 3, num_frames, H_out, W_out]` in `[-1, 1]`.
        """
        num_frames = num_frames or self.config.num_frames

        # 1. Encode reference image to latent (conditioning)
        ref_latent = self._encode_reference_image(image)  # [1, 16, 1, H_lat, W_lat]
        num_cond_latents = 1  # one reference frame at the temporal head

        # 2. Prepare audio embeddings
        audio_embs = self._prepare_audio_embs(audio_mel)

        # 3. Initial noise
        noise = self._make_initial_noise(1, num_frames, height, width, seed)
        # Concat: [ref_latent, noise] along time axis
        latents = mx.concatenate([ref_latent, noise], axis=2)
        T_lat_full = latents.shape[2]

        # 3a. Trim/pad audio_embs to match the latent extent.
        # The DiT's audio path requires T_audio = 1 + vae_scale * (T_lat_full - 1)
        # frames (one audio frame per fps step of the corresponding video).
        v = self.config.vae_scale_temporal
        required_audio_T = 1 + v * (T_lat_full - 1)
        if audio_embs.shape[1] >= required_audio_T:
            audio_embs = audio_embs[:, :required_audio_T]
        else:
            # Pad with the last frame replicated
            shortfall = required_audio_T - audio_embs.shape[1]
            last_frame = audio_embs[:, -1:]
            pad = mx.broadcast_to(last_frame, (audio_embs.shape[0], shortfall, *audio_embs.shape[2:]))
            audio_embs = mx.concatenate([audio_embs, pad], axis=1)

        # 4. Build uncond audio (zeros, same shape as audio_embs)
        uncond_audio = mx.zeros_like(audio_embs)

        # 5. Set up the DMD distilled scheduler with the 8-step sigma schedule.
        # `mlx_arsenal.FlowMatchEulerDiscreteScheduler.set_timesteps` accepts
        # `sigmas: list[float] | np.ndarray` for custom schedules.
        sigmas = get_dmd_distilled_sigmas(
            sampling_steps=self.config.num_sampling_steps,
            num_train_timesteps=self.config.num_train_timesteps,
        )
        self.scheduler.set_timesteps(
            self.config.num_sampling_steps,
            sigmas=np.asarray(sigmas).tolist(),
        )
        # mlx-arsenal appends 1.0 (high noise) to the sigmas array as the
        # sentinel "step beyond the last." For inference we need the last
        # step to drive sigma → 0 (clean), matching PT diffusers' convention
        # of appending 0.0. Otherwise the 8th denoising step does
        # `sample + (1.0 - 0.124) * v` and re-adds ~88% of the noise back.
        self.scheduler.sigmas = mx.concatenate(
            [self.scheduler.sigmas[:-1], mx.array([0.0], dtype=mx.float32)]
        )
        timesteps = self.scheduler.timesteps

        # 6. Stack uncond + cond text embeddings for the batched CFG pass
        text_embeds_cat = mx.concatenate([uncond_embeds, text_embeds], axis=0)
        text_mask_cat = mx.concatenate([uncond_mask, text_mask], axis=0)

        # 7. Denoising loop
        for i, t in enumerate(timesteps):
            t_arr = mx.array([t], dtype=mx.float32) if not isinstance(t, mx.array) else t
            noise_pred = self._cfg_forward(
                latents,
                t_arr,
                text_embeds_cat,
                text_mask_cat,
                audio_embs,
                uncond_embeds,
                uncond_mask,
                uncond_audio,
                num_cond_latents=num_cond_latents,
            )
            # NOTE: mlx-arsenal's scheduler.step returns mx.array directly
            # (NOT a (sample, ...) tuple like PT diffusers). Don't index [0].
            latents = self.scheduler.step(noise_pred, t, latents)

        # 8. Strip the reference latent before decoding
        denoised = latents[:, :, num_cond_latents:]

        # 9. Decode through VAE (denormalize first → match PT _decode convention)
        z_denorm = self.vae.denormalize_latents(denoised)
        video = self.vae.decode(z_denorm)
        return video
