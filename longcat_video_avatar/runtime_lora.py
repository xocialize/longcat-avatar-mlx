"""Runtime (un-merged) LoRAs with per-step multipliers.

`lora.py` pre-merges a single LoRA into the base weights, which is right for an always-on DMD LoRA.
LongCat-Video-Avatar 1.6 ships two LoRAs and switches between them during sampling
(`enable_lora_mix` / `set_lora_mix_step` in Meituan's `longcat_video_dit_avatar.py`): the DMD LoRA
for the first `mix_steps` steps, the OPSD LoRA afterwards. Merging and un-merging at every switch
would drift in bf16, so here each target Linear keeps its base weights and applies the LoRAs on the
fly, like Meituan's `multi_lora_forward`:

    y = base(x) + sum_k  multiplier_k * alpha_scale_k * up_k(down_k(x))

Fused projections (qkv, kv_linear) have one shared `lora_down` and one `lora_up.blocks.{i}` per
split, exactly as in the checkpoints. A user LoRA in the same format (e.g. one trained with
`training/train_lora.py`) can be attached alongside under its own key.
"""

from __future__ import annotations

import pathlib

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.lora import decode_module_name


class LoRALinear(nn.Module):
    """A Linear plus any number of LoRAs, each with its own multiplier (0 = off, no cost)."""

    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.loras: dict[str, dict] = {}
        self.mult: dict[str, float] = {}

    def add(self, key: str, down: mx.array, ups: list[mx.array], scale: float):
        self.loras[key] = {"down": down, "ups": ups, "scale": scale}
        self.mult[key] = 0.0

    def __call__(self, x):
        y = self.base(x)
        for key, lo in self.loras.items():
            m = self.mult[key]
            if m == 0.0:
                continue
            h = x.astype(lo["down"].dtype) @ lo["down"].T
            if len(lo["ups"]) == 1:
                d = h @ lo["ups"][0].T
            else:
                hs = mx.split(h, len(lo["ups"]), axis=-1)
                d = mx.concatenate([hi @ up.T for hi, up in zip(hs, lo["ups"])], axis=-1)
            y = y + (d * (m * lo["scale"])).astype(y.dtype)
        return y


def _get_module(root, path: str):
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module(root, path: str, new):
    parent_path, last = path.rsplit(".", 1)
    parent = _get_module(root, parent_path)
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def attach_lora(dit, lora_path: pathlib.Path | str, key: str, dtype=mx.bfloat16,
                default_alpha_scale: float = 0.5) -> int:
    """Attach a Meituan-format LoRA file to `dit` under `key` (multiplier starts at 0).

    `default_alpha_scale` = network_alpha / rank = 64 / 128, Meituan's value when the file has no
    `alpha_scale` tensors. Returns the number of patched modules.
    """
    sd = mx.load(str(lora_path))
    groups: dict[str, dict] = {}
    for k, v in sd.items():
        mod, tail = decode_module_name(k)
        groups.setdefault(mod, {})[tail] = v
    n = 0
    for mod, g in groups.items():
        target = _get_module(dit, mod)
        if not isinstance(target, LoRALinear):
            if not isinstance(target, nn.Linear):
                raise TypeError(f"LoRA target {mod} is {type(target).__name__}, expected Linear")
            target = LoRALinear(target)
            _set_module(dit, mod, target)
        down = g["lora_down.weight"].astype(dtype)
        split = sorted((k for k in g if k.startswith("lora_up.blocks.")), key=lambda s: int(s.split(".")[2]))
        ups = [g[k].astype(dtype) for k in split] if split else [g["lora_up.weight"].astype(dtype)]
        scale = float(g["alpha_scale"].item()) if "alpha_scale" in g else default_alpha_scale
        target.add(key, down, ups, scale)
        n += 1
    return n


def set_lora_multipliers(dit, mults: dict[str, float]) -> None:
    """Set the multiplier of each named LoRA everywhere it is attached."""
    for _, m in dit.named_modules():
        if isinstance(m, LoRALinear):
            for k, v in mults.items():
                if k in m.mult:
                    m.mult[k] = v
