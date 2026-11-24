"""High-level inference pipeline for LongCat-Video-Avatar 1.5 / 1.6 on MLX.

Wires together:
- umT5-XXL text encoder
- Whisper-large-v3 audio encoder (with group-mean-pool)
- Wan VAE (encode reference image; decode final latent)
- LongCatVideoAvatarTransformer3DModel DiT
- FlowMatchEulerDiscreteScheduler (from mlx-arsenal) with DMD distilled sigmas
- Pre-merged DMD LoRA (1.5), or the DMD + OPSD runtime LoRA schedule (1.6)
- 3-pass disentangled CFG

Modes, each mirroring the PT pipeline method of the same name:
- `generate_ai2v`: reference image + speech + prompt → one 93-frame segment at
  25 fps (3.72 s). The image is the first frame.
- `generate_avc`: audio-driven video continuation. The next segment is
  conditioned on the last 13 frames of the previous one plus the reference
  latent, without a KV cache (PT uses none when the 1.6 LoRA mix is on).
- `__call__`: the whole clip — `generate_ai2v`, then `num_segments - 1`
  continuations, each adding 80 new frames (3.2 s).

MultiTalk routing (several speakers) is not ported.
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
    build_frame_audio_features,
    whisper_encode_audio_to_groups,
    window_audio_features,
)
from longcat_video_avatar.guidance import (
    cfg_split_outputs,
    disentangled_cfg_combine,
    flip_velocity_for_scheduler,
    get_dmd_distilled_sigmas,
)
from longcat_video_avatar.lora import LoRAMix, attach_runtime_loras, merge_lora_into_model
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

    # Avatar 1.6 LoRA schedule (ignored unless `attach_lora_mix` was called):
    # the first `mix_steps` steps run mix_alpha*DMD + (1-mix_alpha)*OPSD, the
    # rest OPSD only. Upstream's documented default is 4 / 1.0.
    mix_steps: int = 4
    mix_alpha: float = 1.0

    # Guidance. The distilled 8-step path runs WITHOUT classifier-free guidance:
    # upstream's demo forces both scales to 1.0 under `--use_distill`
    # (run_demo_avatar_single_audio_to_video.py), which makes
    # `do_classifier_free_guidance` false and each step one conditional DiT
    # pass. 4.0 / 4.0 is the 50-step undistilled default; set a scale above 1.0
    # to get the 3-pass disentangled CFG back.
    text_guidance_scale: float = 1.0
    audio_guidance_scale: float = 1.0

    # Video
    num_frames: int = 93
    target_fps: int = 25

    # Continuation (PT demo defaults for Avatar 1.6): frames of the previous
    # segment a new one is conditioned on, the reference image's temporal RoPE
    # index, and the Reference Skip half-width (0 = off).
    num_cond_frames: int = 13
    ref_img_index: int = 10
    mask_frame_range: int = 0

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
        self.lora_mix: Optional[LoRAMix] = None

    # ----- LoRA pre-merge ----------------------------------------------------

    def merge_dmd_lora(self, lora_state_dict: dict, multiplier: float = 1.0) -> dict:
        """Pre-merge the DMD LoRA into the DiT weights. Returns the
        `merge_lora_into_model` result (applied/unmapped lists).
        """
        return merge_lora_into_model(self.dit, lora_state_dict, multiplier=multiplier)

    def attach_lora_mix(
        self,
        dmd_state_dict: dict,
        opsd_state_dict: dict,
        mix_alpha: Optional[float] = None,
        mix_steps: Optional[int] = None,
    ) -> LoRAMix:
        """Avatar 1.6: attach the DMD and OPSD LoRAs as runtime side-paths and
        switch between them per denoising step (see `lora.LoRAMix`). Call after
        the DiT's weights are loaded. Works over a bf16 or a quantized DiT.
        """
        modules = attach_runtime_loras(self.dit, {"dmd": dmd_state_dict, "opsd": opsd_state_dict})
        self.lora_mix = LoRAMix(
            modules,
            mix_alpha=self.config.mix_alpha if mix_alpha is None else mix_alpha,
            mix_steps=self.config.mix_steps if mix_steps is None else mix_steps,
        )
        return self.lora_mix

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

    @property
    def do_classifier_free_guidance(self) -> bool:
        """PT: `text_guidance_scale > 1.0 or audio_guidance_scale > 1.0`."""
        return self.config.text_guidance_scale > 1.0 or self.config.audio_guidance_scale > 1.0

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
        **cond_kwargs,
    ) -> mx.array:
        """One denoising forward + velocity flip. With both guidance scales at
        1.0 or below (the distilled default) this is a single conditional pass,
        as in PT (`do_classifier_free_guidance` false); otherwise the 3-pass
        disentangled CFG.
        """
        if timestep.ndim == 0:
            timestep = timestep[None]
        # PT: the conditioning latents are clean, so their timestep is 0.
        T_lat = latents.shape[2]
        timestep = mx.broadcast_to(timestep.reshape(-1, 1), (timestep.shape[0], T_lat))
        frame_is_noise = (mx.arange(T_lat) >= num_cond_latents).astype(timestep.dtype)
        timestep = timestep * frame_is_noise[None, :]
        if not self.do_classifier_free_guidance:
            # `text_embeds_cat` is [negative, positive]; the conditional pass takes the positive half.
            pred = self.dit(
                latents,
                timestep,
                text_embeds_cat[1:],
                audio_embs,
                encoder_attention_mask=text_mask_cat[1:],
                num_cond_latents=num_cond_latents,
                **cond_kwargs,
            )
            return flip_velocity_for_scheduler(pred)
        # Pass 1: batched [latents, latents] with [neg_text, pos_text] and pos audio
        latents_2 = mx.concatenate([latents, latents], axis=0)
        # MLX uses `mx.repeat(arr, n, axis=...)` — no `.repeat()` instance method.
        # Ensure timestep is at least 1D before repeat (scheduler may return 0-dim).
        timestep_2 = mx.repeat(timestep, 2, axis=0)
        audio_embs_2 = mx.repeat(audio_embs, 2, axis=0)
        pred_2 = self.dit(
            latents_2,
            timestep_2,
            text_embeds_cat,
            audio_embs_2,
            encoder_attention_mask=text_mask_cat,
            num_cond_latents=num_cond_latents,
            **cond_kwargs,
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
            **cond_kwargs,
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

    # ----- Audio ------------------------------------------------------------

    def encode_audio(self, audio_mel: mx.array, audio_num_samples: Optional[int] = None, sample_rate: int = 16000) -> mx.array:
        """Whisper mel `[1, 128, T_mel]` (30 s chunks concatenated) → one
        feature per video frame, `[T_video, 5, 1280]`, for the whole audio.

        `audio_num_samples` is the driving audio's real length; without it the
        mel's full (padded) extent is used.
        """
        groups = whisper_encode_audio_to_groups(self.audio_encoder, audio_mel)
        video_length = None
        if audio_num_samples is not None:
            video_length = int(audio_num_samples / sample_rate * self.config.target_fps)
        features = build_frame_audio_features(
            groups, fps=self.config.target_fps, enc_fps=self.config.whisper_enc_fps, video_length=video_length
        )
        mx.eval(features)
        return features

    # ----- Sampling ---------------------------------------------------------

    def _set_distilled_timesteps(self):
        sigmas = get_dmd_distilled_sigmas(
            sampling_steps=self.config.num_sampling_steps,
            num_train_timesteps=self.config.num_train_timesteps,
        )
        self.scheduler.set_timesteps(self.config.num_sampling_steps, sigmas=np.asarray(sigmas).tolist())
        # mlx-arsenal appends 1.0 (high noise) as the "step beyond the last"
        # sentinel; PT diffusers appends 0.0. Without this the 8th step does
        # `sample + (1.0 - 0.124) * v` and re-adds ~88% of the noise.
        self.scheduler.sigmas = mx.concatenate(
            [self.scheduler.sigmas[:-1], mx.array([0.0], dtype=mx.float32)]
        )
        return self.scheduler.timesteps

    def _denoise(
        self,
        latents: mx.array,
        num_cond_latents: int,
        audio_embs: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        on_step=None,
        **cond_kwargs,
    ) -> mx.array:
        """The 8-step loop. The first `num_cond_latents` latent frames are
        clean conditioning: they run at timestep 0 and the scheduler never
        touches them (PT steps `latents[:, :, num_cond_latents:]` only).
        """
        timesteps = self._set_distilled_timesteps()
        text_embeds_cat = mx.concatenate([uncond_embeds, text_embeds], axis=0)
        text_mask_cat = mx.concatenate([uncond_mask, text_mask], axis=0)
        uncond_audio = mx.zeros_like(audio_embs)
        cond = latents[:, :, :num_cond_latents]
        for i, t in enumerate(timesteps):
            if self.lora_mix is not None:
                self.lora_mix.set_step(i)
            if on_step is not None:
                on_step(i, len(timesteps))
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
                **cond_kwargs,
            )
            # mlx-arsenal's scheduler.step returns the sample (not a tuple).
            stepped = self.scheduler.step(
                noise_pred[:, :, num_cond_latents:], t, latents[:, :, num_cond_latents:]
            )
            latents = mx.concatenate([cond, stepped], axis=2)
            mx.eval(latents)
        return latents

    def _decode(self, latents: mx.array) -> mx.array:
        return self.vae.decode(self.vae.denormalize_latents(latents))

    def generate_ai2v(
        self,
        image: mx.array,
        audio_embs: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_frames: int,
        height: int,
        width: int,
        seed: int = 0,
        on_step=None,
    ) -> tuple[mx.array, mx.array]:
        """One segment from a reference image (PT `generate_ai2v`).

        `image`: `[1, 3, 1, H, W]` in `[-1, 1]`. `audio_embs`:
        `[1, num_frames, 5, 5, 1280]`. The image's latent replaces the first
        of the `1 + (num_frames - 1) // 4` latent frames, so the image is the
        clip's first frame.

        Returns `(video [1, 3, num_frames, H, W] in [-1, 1], latents)`.
        """
        noise = self._make_initial_noise(1, num_frames, height, width, seed)
        ref_latent = self._encode_reference_image(image)
        latents = mx.concatenate([ref_latent.astype(noise.dtype), noise[:, :, 1:]], axis=2)
        assert audio_embs.shape[1] == num_frames, (audio_embs.shape, num_frames)
        latents = self._denoise(
            latents, 1, audio_embs, text_embeds, text_mask, uncond_embeds, uncond_mask, on_step=on_step
        )
        video = self._decode(latents)
        mx.eval(video)
        return video, latents

    def generate_avc(
        self,
        video: mx.array,
        ref_latent: mx.array,
        audio_embs: mx.array,
        text_embeds: mx.array,
        text_mask: mx.array,
        uncond_embeds: mx.array,
        uncond_mask: mx.array,
        num_frames: int,
        height: int,
        width: int,
        seed: int = 0,
        on_step=None,
    ) -> tuple[mx.array, mx.array]:
        """The next segment, continuing `video` (PT `generate_avc` without a KV
        cache, which is the path the 1.6 LoRA mix uses).

        The last `num_cond_frames` frames of `video` are re-encoded as the
        segment's leading latents; `ref_latent` (the reference image's latent)
        rides in front of them at temporal RoPE index `ref_img_index` and is
        dropped again before decoding. The returned video repeats those
        `num_cond_frames` frames at its head: append `video[:, :, num_cond_frames:]`.
        """
        cfg = self.config
        cond_frames = video[:, :, -cfg.num_cond_frames :]
        cond_latents = self.vae.normalize_latents(self.vae.encode(cond_frames))
        n_cond = 1 + (cfg.num_cond_frames - 1) // cfg.vae_scale_temporal
        assert cond_latents.shape[2] == n_cond, (cond_latents.shape, n_cond)
        noise = self._make_initial_noise(1, num_frames, height, width, seed)
        latents = mx.concatenate(
            [ref_latent.astype(noise.dtype), cond_latents.astype(noise.dtype), noise[:, :, n_cond:]], axis=2
        )
        assert audio_embs.shape[1] == num_frames, (audio_embs.shape, num_frames)
        latents = self._denoise(
            latents,
            n_cond + 1,
            audio_embs,
            text_embeds,
            text_mask,
            uncond_embeds,
            uncond_mask,
            on_step=on_step,
            num_ref_latents=1,
            ref_img_index=cfg.ref_img_index,
            mask_frame_range=cfg.mask_frame_range,
        )
        latents = latents[:, :, 1:]  # drop the reference latent
        video_out = self._decode(latents)
        mx.eval(video_out)
        return video_out, latents

    def frames_for_segments(self, num_segments: int, num_frames: Optional[int] = None) -> int:
        """Frames a clip of `num_segments` segments has."""
        num_frames = num_frames or self.config.num_frames
        return num_frames + max(0, num_segments - 1) * (num_frames - self.config.num_cond_frames)

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
        num_segments: int = 1,
        audio_num_samples: Optional[int] = None,
        on_step=None,
    ) -> mx.array:
        """The whole clip: one image-to-video segment, then `num_segments - 1`
        continuations. Segment `i` draws its noise from `seed + i`.

        `on_step(segment, step, total_steps)` is called before each denoising step.

        Returns video `[1, 3, frames_for_segments(num_segments), H, W]` in `[-1, 1]`.
        """
        num_frames = num_frames or self.config.num_frames
        frame_audio = self.encode_audio(audio_mel, audio_num_samples)

        def step_callback(segment):
            if on_step is None:
                return None
            return lambda step, total: on_step(segment, step, total)

        text = (text_embeds, text_mask, uncond_embeds, uncond_mask)
        video, latents = self.generate_ai2v(
            image,
            window_audio_features(frame_audio, 0, num_frames),
            *text,
            num_frames=num_frames,
            height=height,
            width=width,
            seed=seed,
            on_step=step_callback(0),
        )
        if num_segments <= 1:
            return video

        ref_latent = latents[:, :, :1]
        stride = num_frames - self.config.num_cond_frames
        segments = [video]
        current = video
        for segment in range(1, num_segments):
            current, _ = self.generate_avc(
                current,
                ref_latent,
                window_audio_features(frame_audio, segment * stride, num_frames),
                *text,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=seed + segment,
                on_step=step_callback(segment),
            )
            segments.append(current[:, :, self.config.num_cond_frames :])
        return mx.concatenate(segments, axis=2)
