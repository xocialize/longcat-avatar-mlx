"""Smoke tests for the DMD LoRA loader.

Two kinds:
- Decoder + merge math tests (no network) — synthesize fake LoRA tensors
  and verify the merge produces the right delta.
- Keymap test (uses safetensors index only, no weights download) — confirms
  every LoRA target module exists in our Avatar MLX model.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from longcat_video_avatar.lora import (
    compute_merged_delta,
    decode_module_name,
    group_lora_tensors,
    list_lora_targets,
    merge_lora_into_model,
)


def test_decode_module_name_basic():
    key = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight"
    mp, tail = decode_module_name(key)
    assert mp == "blocks.0.attn.qkv"
    assert tail == "lora_down.weight"


def test_decode_module_name_split_up():
    key = "lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight"
    mp, tail = decode_module_name(key)
    assert mp == "blocks.0.attn.qkv"
    assert tail == "lora_up.blocks.1.weight"


def test_decode_alpha_scale():
    key = "lora___lorahyphen___blocks___lorahyphen___5___lorahyphen___ffn___lorahyphen___w2.alpha_scale"
    mp, tail = decode_module_name(key)
    assert mp == "blocks.5.ffn.w2"
    assert tail == "alpha_scale"


def test_compute_merged_delta_unsplit():
    """Single up/down: delta = multiplier * alpha * (up @ down)."""
    rank = 4
    in_dim, out_dim = 8, 6
    rng = np.random.default_rng(0)
    down = mx.array(rng.standard_normal((rank, in_dim)).astype(np.float32))
    up = mx.array(rng.standard_normal((out_dim, rank)).astype(np.float32))
    alpha = mx.array(0.5)
    group = {
        "lora_down.weight": down,
        "lora_up.weight": up,
        "alpha_scale": alpha,
    }
    delta = compute_merged_delta(group, multiplier=2.0)
    mx.eval(delta)
    expected = 2.0 * 0.5 * (up @ down)
    mx.eval(expected)
    diff = float(mx.abs(delta - expected).max())
    assert diff < 1e-6
    assert delta.shape == (out_dim, in_dim)


def test_compute_merged_delta_split_fused_qkv():
    """3-way split (QKV): shared down of (3*rank, in_dim), 3 separate ups.

    Expected merged shape: (3 * out_per_split, in_dim).
    Each row block i is `up_i @ down[i*rank:(i+1)*rank, :]`.
    """
    rank = 4
    in_dim, out_per_split = 8, 6
    rng = np.random.default_rng(0)
    down = mx.array(rng.standard_normal((3 * rank, in_dim)).astype(np.float32))
    ups = [mx.array(rng.standard_normal((out_per_split, rank)).astype(np.float32)) for _ in range(3)]
    alpha = mx.array(1.0)
    group = {
        "lora_down.weight": down,
        "lora_up.blocks.0.weight": ups[0],
        "lora_up.blocks.1.weight": ups[1],
        "lora_up.blocks.2.weight": ups[2],
        "alpha_scale": alpha,
    }
    delta = compute_merged_delta(group, multiplier=1.0)
    mx.eval(delta)
    assert delta.shape == (3 * out_per_split, in_dim)
    # Verify each block matches up_i @ down_i_slice
    for i in range(3):
        slice_ = delta[i * out_per_split : (i + 1) * out_per_split]
        expected = ups[i] @ down[i * rank : (i + 1) * rank]
        mx.eval(slice_, expected)
        diff = float(mx.abs(slice_ - expected).max())
        assert diff < 1e-6, f"split {i}: diff={diff}"


def test_merge_lora_into_model():
    """End-to-end: synthesize a tiny model + LoRA, merge, verify weight delta."""

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 6, bias=False)

    model = Toy()
    rng = np.random.default_rng(0)
    base_w = mx.array(rng.standard_normal((6, 8)).astype(np.float32))
    model.proj.weight = base_w
    mx.eval(model.parameters())

    # LoRA: down (rank=2 → in=8), up (out=6 → rank=2), alpha 1.0, multiplier 0.5
    down = mx.array(rng.standard_normal((2, 8)).astype(np.float32))
    up = mx.array(rng.standard_normal((6, 2)).astype(np.float32))
    lora_sd = {
        "lora___lorahyphen___proj.lora_down.weight": down,
        "lora___lorahyphen___proj.lora_up.weight": up,
        "lora___lorahyphen___proj.alpha_scale": mx.array(1.0),
    }

    result = merge_lora_into_model(model, lora_sd, multiplier=0.5)
    assert result["applied"] == ["proj"]
    assert result["unmapped"] == []

    expected_new_w = base_w + 0.5 * 1.0 * (up @ down)
    mx.eval(expected_new_w, model.proj.weight)
    diff = float(mx.abs(model.proj.weight - expected_new_w).max())
    assert diff < 1e-6, f"merge diff: {diff}"


def test_dmd_lora_keymap_against_avatar_model():
    """Fetch the DMD LoRA safetensors header (no weights download) and verify
    every target module exists in our Avatar DiT model.
    """
    import struct
    import urllib.request

    url = "https://huggingface.co/meituan-longcat/LongCat-Video-Avatar-1.5/resolve/main/lora/dmd_lora.safetensors"
    req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
    with urllib.request.urlopen(req, timeout=30) as r:
        size_bytes = r.read()
    header_size = struct.unpack("<Q", size_bytes)[0]
    req2 = urllib.request.Request(url, headers={"Range": f"bytes=8-{8 + header_size - 1}"})
    with urllib.request.urlopen(req2, timeout=30) as r:
        hdr = json.loads(r.read().decode())
    lora_keys = [k for k in hdr.keys() if k != "__metadata__"]

    # Group into modules without loading actual tensors — just walk the keys
    modules = set()
    for k in lora_keys:
        mp, _ = decode_module_name(k)
        modules.add(mp)

    # Build the Avatar MLX model and check each LoRA target is in its parameters
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )

    REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
    cfg = json.loads(
        (REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "avatar-1.5--base_model-config.json").read_text()
    )
    model = LongCatVideoAvatarTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(model.parameters())}

    unmapped = [m for m in modules if f"{m}.weight" not in mx_keys]
    assert not unmapped, (
        f"{len(unmapped)} LoRA targets not in MLX model. First 10:\n  "
        + "\n  ".join(sorted(unmapped)[:10])
    )

    # Sanity: per-block LoRA targets. The DMD LoRA targets:
    #   attn.{qkv, proj}, cross_attn.{q_linear, kv_linear},
    #   ffn.{w1, w2, w3} = 7 modules per block.
    # Notably it does NOT target audio_cross_attn (DMD adjusts the sampling-
    # step-sensitive layers; audio path is unchanged).
    block_0_targets = sorted(m for m in modules if m.startswith("blocks.0."))
    assert len(block_0_targets) == 7, (
        f"block 0 LoRA targets diverge from expected 7: {block_0_targets}"
    )
    expected_block_0 = {
        "blocks.0.attn.qkv",
        "blocks.0.attn.proj",
        "blocks.0.cross_attn.q_linear",
        "blocks.0.cross_attn.kv_linear",
        "blocks.0.ffn.w1",
        "blocks.0.ffn.w2",
        "blocks.0.ffn.w3",
    }
    assert set(block_0_targets) == expected_block_0, (
        f"unexpected per-block LoRA target set: {set(block_0_targets) ^ expected_block_0}"
    )
