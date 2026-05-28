"""Quantization smoke tests — validate the q4/q8 DiT load path WITHOUT
the actual ~33 GB quantized weights.

Covers:
- The conversion-time predicate (`_should_quantize_dit_linear`) skips the
  right modules (final_layer.linear, t_embedder, y_embedder, adaLN…).
- The inference-time loader (`_quantize_dit_for_load`) reads
  `dit/config.json:quantization` and installs `QuantizedLinear` at the
  right positions in a small stand-in model. Verifies the predicate's
  skip rule matches conversion-time behavior so weights line up at load.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest


# ---------------------------------------------------------------------------
# Conversion-time predicate
# ---------------------------------------------------------------------------


def test_should_quantize_dit_linear_skips_documented_patterns():
    from recipes.convert_longcat_avatar import (
        DIT_QUANT_SKIP_PATTERNS,
        _should_quantize_dit_linear,
    )

    skips = [
        "final_layer.linear",
        "t_embedder.mlp.0",
        "t_embedder.mlp.2",
        "y_embedder.mlp.0",
        "blocks.0.adaLN_modulation.1",
        "blocks.5.audio_adaLN_modulation.1",
    ]
    for path in skips:
        m = nn.Linear(8, 8)
        assert (
            _should_quantize_dit_linear(path, m) is False
        ), f"path {path!r} should be skipped but was selected for quantization"

    # Sanity: every documented skip pattern was actually exercised above
    covered = set()
    for path in skips:
        for pat in DIT_QUANT_SKIP_PATTERNS:
            if pat in path:
                covered.add(pat)
    assert covered == set(DIT_QUANT_SKIP_PATTERNS), (
        f"test missed skip patterns: {set(DIT_QUANT_SKIP_PATTERNS) - covered}"
    )


def test_should_quantize_dit_linear_keeps_normal_linears():
    from recipes.convert_longcat_avatar import _should_quantize_dit_linear

    keeps = [
        "blocks.0.attn.qkv",
        "blocks.0.attn.proj",
        "blocks.0.ffn.0",
        "blocks.0.ffn.2",
        "blocks.0.text_cross_attn.kv",
        "blocks.0.audio_cross_attn.q",
    ]
    for path in keeps:
        m = nn.Linear(8, 8)
        assert (
            _should_quantize_dit_linear(path, m) is True
        ), f"path {path!r} should be quantized but predicate skipped it"


def test_should_quantize_dit_linear_refuses_non_linear():
    from recipes.convert_longcat_avatar import _should_quantize_dit_linear

    # Quantize predicate must only act on nn.Linear — other modules
    # (Embedding, LayerNorm, Conv*, etc.) are not handled by the recipe.
    assert _should_quantize_dit_linear("blocks.0.norm1", nn.RMSNorm(8)) is False
    assert _should_quantize_dit_linear("x_embedder.proj", nn.Conv1d(4, 8, 3)) is False


# ---------------------------------------------------------------------------
# Inference-time loader
# ---------------------------------------------------------------------------


class _ToyDiT(nn.Module):
    """Minimal stand-in mirroring the *paths* of the real DiT for the
    skip-pattern test. Real DiT instantiation is gated on weights + much
    larger memory, so we exercise the predicate against a tiny tree that
    mirrors the relevant module names.
    """

    def __init__(self) -> None:
        super().__init__()
        # Should be quantized
        self.blocks = [_ToyBlock() for _ in range(2)]
        # Should be skipped
        self.t_embedder = _ToyMLP()
        self.y_embedder = _ToyMLP()
        self.final_layer = _ToyFinal()


# Dimensions chosen to satisfy `mx.quantize`'s "last dim divisible by
# group_size" rule for the default group_size=64 used by the real recipe.
class _ToyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_qkv = nn.Linear(128, 384)
        self.attn_proj = nn.Linear(128, 128)
        self.adaLN_modulation = [nn.SiLU(), nn.Linear(128, 768)]
        self.audio_adaLN_modulation = [nn.SiLU(), nn.Linear(128, 256)]


class _ToyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = [nn.Linear(128, 512), nn.SiLU(), nn.Linear(512, 128)]


class _ToyFinal(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(128, 128)


def _count_quantized(module: nn.Module) -> tuple[int, int]:
    """Return (# QuantizedLinear, # Linear) in the tree."""
    n_q = n_lin = 0
    for _, m in module.named_modules():
        if isinstance(m, nn.QuantizedLinear):
            n_q += 1
        elif isinstance(m, nn.Linear):
            n_lin += 1
    return n_q, n_lin


@pytest.mark.parametrize("bits", [4, 8])
def test_quantize_dit_for_load_skips_documented_paths(bits):
    """The runtime loader must produce the same QuantizedLinear-vs-Linear
    pattern that the recipe used at conversion time — otherwise
    `load_weights` would land bit-packed tensors on the wrong modules.
    """
    from scripts.run_inference import _quantize_dit_for_load

    model = _ToyDiT()
    n_q_before, n_lin_before = _count_quantized(model)
    assert n_q_before == 0
    # 2 blocks × 4 Linears each + 2 MLPs × 2 Linears + 1 final = 13
    assert n_lin_before == 13

    quant_cfg = {"bits": bits, "group_size": 64}
    _quantize_dit_for_load(model, quant_cfg)

    n_q_after, n_lin_after = _count_quantized(model)
    # 2 blocks × {attn_qkv, attn_proj} = 4 quantized Linears
    assert n_q_after == 4, f"expected 4 QuantizedLinear, got {n_q_after}"
    # Skipped: 2 × (adaLN, audio_adaLN) + 2 × 2 embedder Linears + 1 final = 9
    assert n_lin_after == 9, f"expected 9 Linear remaining, got {n_lin_after}"


def test_quantize_dit_for_load_respects_custom_skip_patterns():
    """If a checkpoint carries a non-default skip list, the loader must
    honor it rather than fall back to its hard-coded defaults.
    """
    from scripts.run_inference import _quantize_dit_for_load

    model = _ToyDiT()
    # Override to skip nothing — everything Linear should become Quantized.
    _quantize_dit_for_load(model, {"bits": 4, "group_size": 64, "skip_patterns": []})

    n_q, n_lin = _count_quantized(model)
    assert n_lin == 0, f"with empty skip list, no Linear should remain; got {n_lin}"
    assert n_q == 13, f"expected 13 QuantizedLinear, got {n_q}"


def test_quantize_dit_for_load_default_skips_match_recipe():
    """The runtime predicate's *defaults* (used when config doesn't carry
    skip_patterns) must equal the conversion-time skip patterns — this
    catches drift if someone edits one list and forgets the other.
    """
    from recipes.convert_longcat_avatar import DIT_QUANT_SKIP_PATTERNS
    from scripts.run_inference import _quantize_dit_for_load

    model = _ToyDiT()
    # Pass no skip_patterns; the loader should fall back to its hard-coded
    # defaults. Outcome must match what the recipe predicate produces.
    _quantize_dit_for_load(model, {"bits": 4, "group_size": 64})
    runtime_q, runtime_lin = _count_quantized(model)

    # Rebuild a fresh model and run the recipe predicate manually
    from recipes.convert_longcat_avatar import _should_quantize_dit_linear

    fresh = _ToyDiT()
    nn.quantize(
        fresh,
        group_size=64,
        bits=4,
        class_predicate=_should_quantize_dit_linear,
    )
    recipe_q, recipe_lin = _count_quantized(fresh)

    assert (runtime_q, runtime_lin) == (recipe_q, recipe_lin), (
        f"runtime defaults {runtime_q=}/{runtime_lin=} drifted from recipe "
        f"{recipe_q=}/{recipe_lin=} — keep DIT_QUANT_SKIP_PATTERNS in sync "
        f"with the fallback in scripts/run_inference.py"
    )
