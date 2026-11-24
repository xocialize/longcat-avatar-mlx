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


# ---------------------------------------------------------------------------
# Runtime side-paths + the Avatar 1.6 DMD/OPSD schedule
# ---------------------------------------------------------------------------


def _close(got: mx.array, want: mx.array, rtol: float = 1e-3) -> bool:
    """Metal's fp32 matmul is not bit-faithful (skill-lessons L10), so compare
    relative to the output's magnitude rather than at fp32 epsilon."""
    scale = max(1.0, float(mx.abs(want).max()))
    return float(mx.abs(got - want).max()) <= rtol * scale


def _fake_lora(rng, targets: dict[str, tuple[int, int, int]], rank: int = 4, alpha_scale: float = 0.5):
    """`targets`: {module_path: (in_dim, out_dim, n_splits)} → Meituan-encoded state dict."""
    sd = {}
    for path, (in_dim, out_dim, n_splits) in targets.items():
        name = "lora___lorahyphen___" + path.replace(".", "___lorahyphen___")
        sd[f"{name}.lora_down.weight"] = mx.array(
            rng.standard_normal((rank * n_splits, in_dim)).astype(np.float32)
        )
        if n_splits == 1:
            sd[f"{name}.lora_up.weight"] = mx.array(rng.standard_normal((out_dim, rank)).astype(np.float32))
        else:
            for i in range(n_splits):
                sd[f"{name}.lora_up.blocks.{i}.weight"] = mx.array(
                    rng.standard_normal((out_dim // n_splits, rank)).astype(np.float32)
                )
        sd[f"{name}.alpha_scale"] = mx.array(alpha_scale)
    return sd


class _TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = [_TinyBlock(), _TinyBlock()]


_TINY_TARGETS = {
    "blocks.0.qkv": (8, 24, 3),
    "blocks.0.proj": (8, 8, 1),
    "blocks.1.qkv": (8, 24, 3),
    "blocks.1.proj": (8, 8, 1),
}


def test_runtime_lora_matches_merged_delta():
    """The side-path output equals the base Linear with the merged delta added,
    for single and split-fused targets and for a fractional multiplier."""
    from mlx.utils import tree_flatten

    from longcat_video_avatar.lora import attach_runtime_loras

    rng = np.random.default_rng(0)
    model = _TinyModel()
    mx.eval(model.parameters())
    keys_before = sorted(k for k, _ in tree_flatten(model.parameters()))
    lora = _fake_lora(rng, _TINY_TARGETS)
    x = mx.array(rng.standard_normal((2, 5, 8)).astype(np.float32))

    grouped = group_lora_tensors(lora)
    expected = {}
    for path, attr in (("blocks.0.qkv", model.blocks[0].qkv), ("blocks.1.proj", model.blocks[1].proj)):
        delta = compute_merged_delta(grouped[path], multiplier=0.7)
        expected[path] = x @ (attr.weight + delta).T + attr.bias

    modules = attach_runtime_loras(model, {"dmd": lora}, dtype=mx.float32)
    assert len(modules) == 4
    for m in modules:
        m.set_multiplier("dmd", 0.7)
    assert _close(model.blocks[0].qkv(x), expected["blocks.0.qkv"])
    assert _close(model.blocks[1].proj(x), expected["blocks.1.proj"])

    # Adapter tensors stay out of the parameter tree; only the base moved under `.base`.
    keys_after = sorted(k for k, _ in tree_flatten(model.parameters()))
    assert keys_after == sorted(k.replace(".weight", ".base.weight").replace(".bias", ".base.bias") for k in keys_before)


def test_runtime_lora_zero_multiplier_is_base():
    from longcat_video_avatar.lora import attach_runtime_loras

    rng = np.random.default_rng(1)
    model = _TinyModel()
    x = mx.array(rng.standard_normal((1, 3, 8)).astype(np.float32))
    base_out = model.blocks[0].qkv(x)
    attach_runtime_loras(model, {"dmd": _fake_lora(rng, _TINY_TARGETS)}, dtype=mx.float32)
    assert float(mx.abs(model.blocks[0].qkv(x) - base_out).max()) == 0.0


def test_runtime_lora_missing_target_raises():
    import pytest

    from longcat_video_avatar.lora import attach_runtime_loras

    rng = np.random.default_rng(2)
    with pytest.raises(ValueError, match="missing module"):
        attach_runtime_loras(_TinyModel(), {"dmd": _fake_lora(rng, {"blocks.0.nope": (8, 8, 1)})})


def test_lora_mix_schedule_default_switches_dmd_to_opsd():
    """mix_steps=4, mix_alpha=1.0: steps 0-3 are DMD only, steps 4-7 OPSD only."""
    from longcat_video_avatar.lora import LoRAMix, attach_runtime_loras

    rng = np.random.default_rng(3)
    model = _TinyModel()
    dmd, opsd = _fake_lora(rng, _TINY_TARGETS), _fake_lora(rng, _TINY_TARGETS)
    x = mx.array(rng.standard_normal((1, 3, 8)).astype(np.float32))
    lin = model.blocks[0].qkv
    want = {}
    for key, sd in (("dmd", dmd), ("opsd", opsd)):
        delta = compute_merged_delta(group_lora_tensors(sd)["blocks.0.qkv"])
        want[key] = x @ (lin.weight + delta).T + lin.bias

    mix = LoRAMix(attach_runtime_loras(model, {"dmd": dmd, "opsd": opsd}, dtype=mx.float32))
    assert [mix.multipliers_for_step(i) for i in (0, 3, 4, 7)] == [(1.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 1.0)]
    for step in range(8):
        mix.set_step(step)
        key = "dmd" if step < 4 else "opsd"
        assert _close(model.blocks[0].qkv(x), want[key]), step
    # A second generation starts from step 0 again.
    mix.set_step(0)
    assert _close(model.blocks[0].qkv(x), want["dmd"])


def test_lora_mix_fractional_alpha_blends_both():
    from longcat_video_avatar.lora import LoRAMix, attach_runtime_loras

    rng = np.random.default_rng(4)
    model = _TinyModel()
    dmd, opsd = _fake_lora(rng, _TINY_TARGETS), _fake_lora(rng, _TINY_TARGETS)
    x = mx.array(rng.standard_normal((1, 3, 8)).astype(np.float32))
    lin = model.blocks[1].proj
    d_dmd = compute_merged_delta(group_lora_tensors(dmd)["blocks.1.proj"])
    d_opsd = compute_merged_delta(group_lora_tensors(opsd)["blocks.1.proj"])
    want = x @ (lin.weight + 0.25 * d_dmd + 0.75 * d_opsd).T + lin.bias
    mix = LoRAMix(attach_runtime_loras(model, {"dmd": dmd, "opsd": opsd}, dtype=mx.float32), mix_alpha=0.25, mix_steps=3)
    assert mix.multipliers_for_step(2) == (0.25, 0.75) and mix.multipliers_for_step(3) == (0.0, 1.0)
    mix.set_step(1)
    assert _close(model.blocks[1].proj(x), want)


def test_runtime_lora_over_quantized_base():
    """The side-path rides a QuantizedLinear: out = quantized_base(x) + lora(x)."""
    from longcat_video_avatar.lora import attach_runtime_loras

    rng = np.random.default_rng(5)

    class Q(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(64, 64)

    model = Q()
    nn.quantize(model, group_size=64, bits=8)
    x = mx.array(rng.standard_normal((1, 2, 64)).astype(np.float32))
    base_out = model.proj(x)
    lora = _fake_lora(rng, {"proj": (64, 64, 1)})
    (wrapper,) = attach_runtime_loras(model, {"dmd": lora}, dtype=mx.float32)
    assert isinstance(wrapper.base, nn.QuantizedLinear)
    wrapper.set_multiplier("dmd", 1.0)
    delta = compute_merged_delta(group_lora_tensors(lora)["proj"])
    assert _close(model.proj(x), (base_out + x @ delta.T))
