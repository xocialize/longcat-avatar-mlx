"""Pipeline-level CFG combiner + DMD distilled sigma schedule.

References:
- CFG combiner: `pipeline_longcat_video_avatar.py:826-844` in PT.
- DMD sigmas: `pipeline_longcat_video_avatar.py:380-398` (`get_timesteps_sigmas`).
- Negative velocity flip: `pipeline_longcat_video_avatar.py:841`.
"""

from __future__ import annotations

import math

import mlx.core as mx


def disentangled_cfg_combine(
    noise_pred_cond: mx.array,
    noise_pred_uncond_text: mx.array,
    noise_pred_uncond: mx.array,
    text_guidance_scale: float = 4.0,
    audio_guidance_scale: float = 4.0,
) -> mx.array:
    """3-pass disentangled CFG for the LongCat Avatar.

    Per Meituan's pipeline:
        noise_pred = uncond
                   + text_scale  * (cond           - uncond_text)
                   + audio_scale * (uncond_text    - uncond)

    Interpretation: `text_scale` weights the text contribution *given* audio is
    present; `audio_scale` weights the audio contribution against the fully
    unconditional baseline.

    Defaults from the DMD-distilled model: text=4.0, audio=4.0.
    """
    return (
        noise_pred_uncond
        + text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text)
        + audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond)
    )


def flip_velocity_for_scheduler(noise_pred: mx.array) -> mx.array:
    """LongCat DiT predicts negative velocity (`ε − x_0`). Flip the sign before
    handing to `FlowMatchEulerDiscreteScheduler.step()`. Don't miss this:
    same line 841 in PT pipeline, easy to forget on the MLX port.
    """
    return -noise_pred


def get_dmd_distilled_sigmas(
    sampling_steps: int = 8,
    num_train_timesteps: int = 1000,
    num_distill_sample_steps: int = 8,
    model_type: str = "avatar-v1.5",
) -> mx.array:
    """Compute the distilled sigma schedule for the 8-step DMD path.

    Matches `pipeline_longcat_video_avatar.py:get_timesteps_sigmas(use_distill=True)`.
    For `model_type='avatar-v1.5'` (the v1.5 release path), the schedule is:
        idx = arange(1, num_distill_sample_steps + 1)
        idx = round(idx * (num_train_timesteps / num_distill_sample_steps))
        idx = num_train_timesteps - idx
        sigmas = flip(linspace(0, 1, num_train_timesteps))[idx]
        sigmas = flip(sigmas)
    """
    if model_type != "avatar-v1.5":
        raise NotImplementedError(
            f"Only model_type='avatar-v1.5' is implemented; got {model_type!r}. "
            "Other LongCat variants use a different distillation schedule."
        )

    # Build the 8 distill indices
    step_size = num_train_timesteps // num_distill_sample_steps  # e.g. 125 for 8 steps × 1000 train
    distill_idx = [round((i + 1) * step_size) for i in range(num_distill_sample_steps)]
    distill_idx = [num_train_timesteps - i for i in distill_idx]

    # Build the full linspace(0, 1, N) reversed, then index, then reverse again
    full = [1.0 - i / num_train_timesteps for i in range(num_train_timesteps)]  # reversed linspace
    # Note: PT does `torch.linspace(0, 1, N)` which gives [0, 1/(N-1), ..., 1], then `flip` → [1, ..., 0].
    # Use a numerically equivalent computation:
    full = [(num_train_timesteps - 1 - i) / (num_train_timesteps - 1) for i in range(num_train_timesteps)]
    sigmas = [full[i] for i in distill_idx]
    # Final flip so the schedule is descending (high sigma -> low sigma)
    sigmas = list(reversed(sigmas))
    return mx.array(sigmas, dtype=mx.float32)


def cfg_split_outputs(noise_pred_2batch: mx.array) -> tuple[mx.array, mx.array]:
    """Split a doubled-batch CFG forward output into `(uncond_text, cond)`.

    The pipeline stacks `[latents, latents]` paired with text
    `cat([negative_prompt_embeds, positive_prompt_embeds], dim=0)` — negative
    FIRST. So the first half of the DiT output is `noise_pred_uncond_text`
    and the second half is `noise_pred_cond` (matches PT
    `noise_pred.chunk(2)` ordering).
    """
    uncond_text, cond = mx.split(noise_pred_2batch, 2, axis=0)
    return uncond_text, cond
