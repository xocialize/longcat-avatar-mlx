"""Recipe smoke tests — validate the conversion logic WITHOUT downloading
~60 GB of weights. Uses tiny synthetic dicts to exercise the layout, dtype
casting, and materialization code paths.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import mlx.core as mx
import numpy as np
import pytest

from recipes.convert_longcat_avatar import (
    _materialize_and_save,
    _np_to_mx_with_layout,
    _save_sharded_safetensors,
)


def test_np_to_mx_layout_2d_passthrough():
    """Linear weight (O, I): same shape in PT and MLX, no transpose."""
    arr = np.random.randn(8, 4).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="blocks.0.attn.qkv.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 4)
    assert out.dtype == mx.bfloat16


def test_np_to_mx_layout_conv1d_transpose():
    """Conv1d (O, I, K) -> (O, K, I)."""
    arr = np.random.randn(8, 4, 3).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="conv1.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 3, 4)


def test_np_to_mx_layout_conv2d_transpose():
    """Conv2d (O, I, H, W) -> (O, H, W, I)."""
    arr = np.random.randn(8, 4, 3, 3).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="proj.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 3, 3, 4)


def test_np_to_mx_layout_conv3d_transpose():
    """Conv3d (O, I, T, H, W) -> (O, T, H, W, I)."""
    arr = np.random.randn(8, 4, 3, 3, 3).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="x_embedder.proj.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (8, 3, 3, 3, 4)


def test_np_to_mx_layout_gamma_no_transpose():
    """RMS norm gamma (e.g. (C, 1, 1, 1)): no transpose despite being 4D."""
    arr = np.random.randn(384, 1, 1, 1).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="layer_norm.gamma", is_gamma=True, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.shape == (384, 1, 1, 1)


def test_adaLN_modulation_stays_fp32_even_when_dtype_bf16():
    """AdaLN modulation weights should NOT be downcast (per CLAUDE.md L11)."""
    arr = np.random.randn(8, 4).astype(np.float32)
    out = _np_to_mx_with_layout(arr, name="blocks.0.adaLN_modulation.1.weight", is_gamma=False, dtype=mx.bfloat16)
    mx.eval(out)
    assert out.dtype == mx.float32, f"adaLN weights must stay fp32, got {out.dtype}"


def test_adaLN_modulation_upcast_when_source_is_bf16():
    """If the source checkpoint is bf16 (as Avatar DiT is) and the key is
    adaLN_modulation, we must UPCAST to fp32 — not just keep bf16. This
    catches a subtle directional bug in the cast logic.
    """
    from recipes.convert_longcat_avatar import _layout_and_cast

    bf16_src = mx.random.normal((8, 4)).astype(mx.bfloat16)
    out = _layout_and_cast(
        bf16_src, name="blocks.0.adaLN_modulation.1.weight", is_gamma=False, dtype=mx.bfloat16
    )
    mx.eval(out)
    assert out.dtype == mx.float32, f"bf16 → fp32 upcast for adaLN required, got {out.dtype}"


def test_layout_and_cast_handles_bf16_source():
    """mx.load returns bf16 for bf16 safetensors — our layout fn must not
    crash and must produce bf16 output for non-adaLN keys.
    """
    from recipes.convert_longcat_avatar import _layout_and_cast

    bf16_src = mx.random.normal((6, 4, 3, 3, 3)).astype(mx.bfloat16)  # Conv3d weight
    out = _layout_and_cast(
        bf16_src, name="blocks.0.attn.qkv.weight", is_gamma=False, dtype=mx.bfloat16
    )
    mx.eval(out)
    assert out.dtype == mx.bfloat16
    # Conv3d (O, I, T, H, W) -> (O, T, H, W, I)
    assert out.shape == (6, 3, 3, 3, 4)


def test_materialize_and_save_writes_nonzero_tensors(tmp_path):
    """Smoke check that `_materialize_and_save` actually writes real data —
    catches the silent-zero trap (lazy MLX tensors serialized as zeros).
    """
    # Build a state dict where some tensors are lazy
    a = mx.ones((4, 4))  # eager-ish (constant init)
    b = mx.random.normal((8, 8))  # lazy until eval
    c = a @ b[: 4, :4]  # definitely lazy (matmul result)
    sd = {"a": a, "b": b, "c": c}

    out_path = tmp_path / "test.safetensors"
    _materialize_and_save(sd, out_path)

    # Re-load and assert non-zero
    from safetensors import safe_open

    with safe_open(str(out_path), framework="numpy") as f:
        for k in ("a", "b", "c"):
            arr = f.get_tensor(k)
            assert np.abs(arr).sum() > 0.0, f"{k} was saved as all zeros — materialization failed"


def test_save_sharded_writes_index_and_multiple_shards(tmp_path):
    """Sharding splits a large state dict across multiple files with an index."""
    # Build a state dict that will exceed 1 MB / shard with multiple tensors
    sd = {f"k{i}": mx.random.normal((512, 512)) for i in range(8)}
    _save_sharded_safetensors(sd, tmp_path / "weights", base_name="model", max_shard_size_bytes=2 * 1024 * 1024)

    out_dir = tmp_path / "weights"
    shards = sorted(out_dir.glob("model-*.safetensors"))
    assert len(shards) > 1, f"expected multiple shards, got {len(shards)}"

    idx_path = out_dir / "model.safetensors.index.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text())
    assert "weight_map" in idx
    assert len(idx["weight_map"]) == 8  # all 8 keys mapped to some shard
