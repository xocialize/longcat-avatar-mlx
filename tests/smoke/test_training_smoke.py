"""Smoke tests for LoRA training and the fast DiT patches, on a tiny random DiT (seconds, no downloads)."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

from longcat_video_avatar import fast
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from training.train_lora import Engine, add_loras, reference_step

from tests.smoke.test_continuation_smoke import TINY


def _rand(shape, seed):
    return mx.array(np.random.default_rng(seed).standard_normal(shape).astype(np.float32))


def _sample(T=3):
    lat = _rand((4, T, 4, 4), 1)
    audio = _rand((1 + 4 * (T - 1), 5, 5, 32), 2)
    text, mask = _rand((1, 6, 32), 3), mx.array([1, 1, 1, 1, 0, 0])
    return lat, audio, text, mask, 0.8, _rand((1, 4, T, 4, 4), 4)


def _dit_with_loras(seed=0):
    mx.random.seed(seed)
    dit = LongCatVideoAvatarTransformer3DModel.from_config(TINY)
    add_loras(dit, rank=4, alpha=4)
    for _, m in dit.named_modules():  # non-zero lora_up so every gradient is exercised
        if hasattr(m, "lora_up"):
            m.lora_up = _rand(m.lora_up.shape, 5) * 0.01
    mx.eval(dit.parameters())
    return dit


def _max_rel(g1, g2):
    a, b = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    assert a.keys() == b.keys()
    return max(float(mx.abs(a[k] - b[k]).max() / (mx.abs(b[k]).max() + 1e-12)) for k in a)


def test_engine_exact_matches_reference_step():
    dit = _dit_with_loras()
    loss_r, g_r = reference_step(dit)(*_sample())
    loss_e, g_e = Engine(dit, compile=False, exact=True)(*_sample())
    assert abs(loss_r.item() - loss_e.item()) < 1e-6
    assert _max_rel(g_e, g_r) < 1e-5


def test_engine_compiled_close_to_reference_step():
    dit = _dit_with_loras()
    loss_r, g_r = reference_step(dit)(*_sample())
    loss_e, g_e = Engine(fast.apply(dit, fast_norms=True))(*_sample())
    assert abs(loss_r.item() - loss_e.item()) / loss_r.item() < 1e-2
    assert _max_rel(g_e, g_r) < 0.1  # bf16-level differences from fused norms and the mask-free attention


def test_fast_patches_without_norms_are_exact():
    mx.random.seed(0)
    dit = LongCatVideoAvatarTransformer3DModel.from_config(TINY)
    mx.eval(dit.parameters())
    h, t = _rand((1, 4, 3, 4, 4), 7), mx.array([[0.0, 500.0, 500.0]])
    text, mask, audio = _rand((1, 1, 6, 32), 8), mx.array([[1, 1, 1, 1, 0, 0]]), _rand((1, 9, 5, 5, 32), 9)
    ref = dit(h, t, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    out = fast.apply(dit)(h, t, text, audio, encoder_attention_mask=mask, num_cond_latents=1)
    comp = fast.apply(dit, compile=True, fast_norms=True)(h, t, text, audio, encoder_attention_mask=mask,
                                                          num_cond_latents=1)
    mx.eval(ref, out, comp)
    assert float(mx.abs(ref - out).max()) == 0.0
    assert float(mx.abs(ref - comp).max()) / float(mx.abs(ref).max()) < 2e-2
    fast.remove(dit)
