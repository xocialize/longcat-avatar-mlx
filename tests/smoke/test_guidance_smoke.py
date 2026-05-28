"""Smoke tests for the CFG combiner + DMD sigma schedule."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from longcat_video_avatar.guidance import (
    cfg_split_outputs,
    disentangled_cfg_combine,
    flip_velocity_for_scheduler,
    get_dmd_distilled_sigmas,
)


def test_cfg_combiner_when_all_predictions_equal_is_passthrough():
    """If cond == uncond_text == uncond, the combined output is just uncond
    (regardless of guidance scales). Sanity check on the formula.
    """
    base = mx.random.normal((1, 4, 8, 8, 8))
    out = disentangled_cfg_combine(base, base, base, text_guidance_scale=4.0, audio_guidance_scale=4.0)
    mx.eval(out)
    diff = float(mx.abs(out - base).max())
    assert diff < 1e-6, f"passthrough check failed: diff={diff}"


def test_cfg_combiner_text_only_contribution():
    """When uncond_text == uncond (no audio effect), the audio term drops out
    and we get standard text CFG: uncond + s_t * (cond - uncond_text).
    """
    uncond = mx.zeros((1, 4, 4, 4, 4))
    uncond_text = mx.zeros((1, 4, 4, 4, 4))
    cond = mx.ones((1, 4, 4, 4, 4))
    out = disentangled_cfg_combine(cond, uncond_text, uncond, text_guidance_scale=4.0, audio_guidance_scale=4.0)
    mx.eval(out)
    # uncond + 4.0 * (cond - uncond_text) + 4.0 * (uncond_text - uncond)
    # = 0 + 4.0 * 1 + 4.0 * 0 = 4.0
    expected = 4.0
    diff = float(mx.abs(out - expected).max())
    assert diff < 1e-6, f"text-only: got {float(out.max())}, expected {expected}"


def test_cfg_combiner_audio_only_contribution():
    """When cond == uncond_text (audio adds nothing to text), the text term
    drops and we get: uncond + s_a * (uncond_text - uncond).
    """
    uncond = mx.zeros((1, 4, 4, 4, 4))
    uncond_text = mx.ones((1, 4, 4, 4, 4))
    cond = mx.ones((1, 4, 4, 4, 4))
    out = disentangled_cfg_combine(cond, uncond_text, uncond, text_guidance_scale=4.0, audio_guidance_scale=4.0)
    mx.eval(out)
    # = 0 + 4.0 * 0 + 4.0 * 1 = 4.0
    diff = float(mx.abs(out - 4.0).max())
    assert diff < 1e-6


def test_flip_velocity():
    x = mx.array([1.0, -2.0, 3.0])
    y = flip_velocity_for_scheduler(x)
    mx.eval(y)
    assert (mx.abs(y - mx.array([-1.0, 2.0, -3.0])) < 1e-6).all()


def test_dmd_sigmas_have_8_descending_values():
    sigmas = get_dmd_distilled_sigmas(sampling_steps=8, num_train_timesteps=1000, num_distill_sample_steps=8)
    mx.eval(sigmas)
    arr = np.asarray(sigmas)
    assert arr.shape == (8,), f"expected 8 sigmas, got {arr.shape}"
    # Should be descending (high sigma → low sigma over the sampling loop)
    for i in range(len(arr) - 1):
        assert arr[i] >= arr[i + 1], f"sigmas not monotonic at idx {i}: {arr.tolist()}"
    # First sigma should be near 1.0 (high noise), last near 0 (clean)
    assert arr[0] > 0.5, f"first sigma should be high noise, got {arr[0]}"
    assert arr[-1] < 0.5, f"last sigma should be low noise, got {arr[-1]}"


def test_cfg_split_outputs():
    """The 2-batch CFG output splits into (uncond_text, cond)."""
    # Two distinct tensors stacked: uncond_text first, cond second
    uncond_text = mx.zeros((1, 4, 4, 4, 4))
    cond = mx.ones((1, 4, 4, 4, 4))
    # Pipeline does `cat([latents]*2)` with text_embeds = `cat([neg, pos])`, so
    # the OUTPUT is `[noise_pred_uncond_text; noise_pred_cond]`. PT then does
    # `noise_pred.chunk(2)` which returns them in CHUNK order = (first half, second half).
    stacked = mx.concatenate([uncond_text, cond], axis=0)
    a, b = cfg_split_outputs(stacked)
    mx.eval(a, b)
    assert (a == uncond_text).all()
    assert (b == cond).all()
