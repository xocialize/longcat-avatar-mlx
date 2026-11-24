"""LoRA loading for LongCat-Video-Avatar: pre-merge (1.5) and runtime side-paths (1.6).

Avatar 1.6 ships two distillation LoRAs over one base DiT (`loraA` = DMD,
`loraB` = OPSD) and switches between them *during* the 8-step denoise: the
first `mix_steps` steps run `mix_alpha * DMD + (1 - mix_alpha) * OPSD`, the
rest run OPSD alone. A pre-merge cannot express that, so 1.6 uses the same
mechanism as PT (`_create_multi_lora_forward`): the base Linear's output plus
`lora_up(lora_down(x)) * multiplier * alpha_scale` per active LoRA. See
`LoRALinear`, `attach_runtime_loras` and `LoRAMix` at the bottom of this file.
The side-path also works over a quantized base, which a merge does not.

The rest of this docstring describes the 1.5 pre-merge path.

PyTorch reference: `pipeline_longcat_video_avatar.py` calls
`self.dit.load_lora(...)` + `self.dit.enable_loras([...])` to runtime-patch
each LoRA'd module's `forward` (see `longcat_video_dit.py:197-249`). For MLX
inference, we instead **pre-merge**: load LoRA tensors once, compute
    W' = W_base + multiplier * alpha_scale * (lora_up @ lora_down)
and overwrite the corresponding base weights. This avoids runtime monkey-
patching, costs no extra forward-pass FLOPs, and works for any always-on
inference path (DMD distillation, the current use case).

LoRA key encoding (Meituan-specific, from
`longcat_video_dit.py:319` source):
- Tensor name format: `lora___lorahyphen___<MODULE_PATH>.<TENSOR>`
- `___lorahyphen___` decodes to `.` in MODULE_PATH (since `.` is reserved
  for safetensors hierarchy).
- TENSOR is one of `lora_down.weight`, `lora_up.weight`, `alpha_scale`, or
  for fused projections (qkv, kv_linear): `lora_up.blocks.{idx}.weight`.

For fused-QKV / fused-KV layers (where the base Linear's output is a
concatenation of N projections):
- `lora_down.weight` shape: `(N * rank, in_features)` — shared down projection
- `lora_up.blocks.{i}.weight` shape: `(out_per_split, rank)` — one up per split
The merged delta for split i is `up_i @ down[i*rank:(i+1)*rank, :]`,
concatenated along output axis 0.

This module ONLY handles the merge math; the lifecycle (download + dtype
handling + parameter update) is in `pipeline_mlx.py`.
"""

from __future__ import annotations

from typing import Iterable, Optional

import mlx.core as mx
import mlx.nn as nn


def decode_module_name(lora_key: str) -> tuple[str, str]:
    """Split a Meituan-encoded LoRA tensor name into `(module_path, tensor)`.

    Example:
        `lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_down.weight`
        →  `("blocks.0.attn.qkv", "lora_down.weight")`

        `lora___lorahyphen___blocks___lorahyphen___0___lorahyphen___attn___lorahyphen___qkv.lora_up.blocks.1.weight`
        →  `("blocks.0.attn.qkv", "lora_up.blocks.1.weight")`
    """
    # Strip the `lora___lorahyphen___` prefix and decode the rest.
    PREFIX = "lora___lorahyphen___"
    SEP = "___lorahyphen___"
    if not lora_key.startswith(PREFIX):
        raise ValueError(f"Unexpected LoRA key prefix: {lora_key}")
    body = lora_key[len(PREFIX) :]
    # The body has form `<encoded_module>.<tail>` where `tail` starts with
    # `lora_down` / `lora_up` / `alpha_scale`. Find the first `.` AFTER all
    # `___lorahyphen___` segments.
    decoded = body.replace(SEP, ".")
    # Now decoded looks like `blocks.0.attn.qkv.lora_down.weight`. The tail
    # always starts with `lora_down`, `lora_up`, or `alpha_scale` — find that.
    for marker in (".lora_down.", ".lora_up.", ".alpha_scale"):
        idx = decoded.rfind(marker)
        if idx >= 0:
            module_path = decoded[:idx]
            tensor = decoded[idx + 1 :]
            return module_path, tensor
    raise ValueError(f"Cannot find LoRA tail marker in: {decoded}")


def group_lora_tensors(state_dict: dict[str, mx.array]) -> dict[str, dict[str, mx.array]]:
    """Group LoRA tensors by target module_path.

    Returns `{module_path: {tensor_subkey: tensor}}`. `tensor_subkey` is e.g.
    `lora_down.weight`, `lora_up.weight`, `lora_up.blocks.0.weight`,
    `alpha_scale`.
    """
    grouped: dict[str, dict[str, mx.array]] = {}
    for k, v in state_dict.items():
        module_path, tail = decode_module_name(k)
        grouped.setdefault(module_path, {})[tail] = v
    return grouped


def compute_merged_delta(group: dict[str, mx.array], multiplier: float = 1.0) -> mx.array:
    """Compute the weight delta `multiplier * alpha_scale * (up @ down)` for
    one LoRA target module. Handles both single and split-fused variants.

    Returns the delta in PT weight layout `(out_features, in_features)`. The
    caller is responsible for adding this to the base module's weight tensor.
    """
    if "lora_down.weight" not in group:
        raise KeyError(f"LoRA group missing lora_down.weight; keys: {list(group)}")
    down = group["lora_down.weight"]  # (rank * N_splits, in_features)
    alpha_scale = float(group.get("alpha_scale", mx.array(1.0)).item())

    # Detect split: presence of `lora_up.blocks.{i}.weight`
    split_keys = sorted(k for k in group if k.startswith("lora_up.blocks."))
    if split_keys:
        n_splits = len(split_keys)
        rank = down.shape[0] // n_splits
        assert down.shape[0] == n_splits * rank, (
            f"down shape {down.shape} doesn't divide into {n_splits} splits"
        )
        delta_parts = []
        for i, sk in enumerate(split_keys):
            up_i = group[sk]  # (out_per_split, rank)
            down_i = down[i * rank : (i + 1) * rank]  # (rank, in_features)
            # PT W = up @ down. Same shape: (out, rank) @ (rank, in) = (out, in)
            delta_i = up_i @ down_i  # (out_per_split, in_features)
            delta_parts.append(delta_i)
        delta = mx.concatenate(delta_parts, axis=0)  # (sum out_per_split, in_features)
    else:
        if "lora_up.weight" not in group:
            raise KeyError(f"LoRA group missing lora_up.weight; keys: {list(group)}")
        up = group["lora_up.weight"]  # (out_features, rank)
        delta = up @ down  # (out_features, in_features)

    return (multiplier * alpha_scale) * delta


def merge_lora_into_model(
    mx_model,
    lora_state_dict: dict[str, mx.array],
    multiplier: float = 1.0,
    target_prefix: str = "",
) -> dict[str, list[str]]:
    """Pre-merge a LoRA into the MLX model's base weights in-place.

    Args:
        mx_model: an nn.Module whose `parameters()` includes the LoRA target weights.
        lora_state_dict: the loaded safetensors dict from `dmd_lora.safetensors`.
        multiplier: per-LoRA strength multiplier (PT's `lora.multiplier`, default 1.0).
        target_prefix: optional path prefix prepended to each decoded module_path
            before lookup in `mx_model.parameters()`. Use `""` for a flat model;
            use `"dit"` if your wrapper nests the DiT under `self.dit`.

    Returns:
        Dict with `applied` (list of merged module paths) and `unmapped`
        (LoRA keys whose target module wasn't found in the MLX model).
    """
    from mlx.utils import tree_flatten, tree_unflatten

    grouped = group_lora_tensors(lora_state_dict)

    # Snapshot existing params for in-place update
    params = dict(tree_flatten(mx_model.parameters()))

    applied: list[str] = []
    unmapped: list[str] = []

    for module_path, group in grouped.items():
        weight_key = (
            f"{target_prefix}.{module_path}.weight" if target_prefix else f"{module_path}.weight"
        )
        if weight_key not in params:
            unmapped.append(module_path)
            continue
        base_w = params[weight_key]
        delta = compute_merged_delta(group, multiplier=multiplier).astype(base_w.dtype)
        if delta.shape != base_w.shape:
            raise ValueError(
                f"shape mismatch merging {module_path}: "
                f"delta {tuple(delta.shape)} vs base {tuple(base_w.shape)}"
            )
        params[weight_key] = base_w + delta
        applied.append(module_path)

    # Push the updated params back
    mx_model.update(tree_unflatten(list(params.items())))
    mx.eval(mx_model.parameters())  # materialize the merged tensors

    return {"applied": applied, "unmapped": unmapped}


def list_lora_targets(lora_state_dict: dict[str, mx.array]) -> list[str]:
    """Diagnostic: list all module paths the LoRA targets. Useful for
    pre-flight validation against the MLX model's parameter tree.
    """
    return sorted(group_lora_tensors(lora_state_dict).keys())


# ---------------------------------------------------------------------------
# Runtime LoRA side-paths (Avatar 1.6)
# ---------------------------------------------------------------------------


class LoRALinear(nn.Module):
    """A Linear (or QuantizedLinear) plus switchable LoRA side-paths.

    Mirrors PT `_create_multi_lora_forward`:
        out = base(x) + sum_k up_k(down_k(x)) * multiplier_k * alpha_scale_k
    over the adapters whose multiplier is non-zero. Split-fused targets (qkv,
    kv_linear) chunk the down projection's output and run one up per chunk
    (PT `LoRAUPParallel`).

    Adapter tensors live in underscore attributes so they stay out of
    `parameters()`: the base model's weight tree and `load_weights` keys are
    unchanged by attaching LoRAs.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self._adapters: dict[str, tuple[mx.array, list[mx.array], float]] = {}
        self._multipliers: dict[str, float] = {}

    def add_adapter(self, key: str, group: dict[str, mx.array], dtype=mx.bfloat16) -> None:
        if "lora_down.weight" not in group:
            raise KeyError(f"LoRA group missing lora_down.weight; keys: {list(group)}")
        split_keys = sorted(
            (k for k in group if k.startswith("lora_up.blocks.")),
            key=lambda k: int(k.split(".")[2]),
        )
        ups = [group[k] for k in split_keys] if split_keys else [group["lora_up.weight"]]
        down = group["lora_down.weight"]
        if down.shape[0] % len(ups) != 0:
            raise ValueError(f"down shape {down.shape} doesn't divide into {len(ups)} splits")
        alpha_scale = float(group.get("alpha_scale", mx.array(1.0)).astype(mx.float32).item())
        # Stored transposed (in, rank) / (rank, out) so the forward is two plain matmuls.
        down_t = down.T.astype(dtype)
        ups_t = [u.T.astype(dtype) for u in ups]
        mx.eval(down_t, ups_t)
        self._adapters[key] = (down_t, ups_t, alpha_scale)
        self._multipliers.setdefault(key, 0.0)

    def set_multiplier(self, key: str, multiplier: float) -> None:
        if key in self._adapters:
            self._multipliers[key] = float(multiplier)

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        for key, (down_t, ups_t, alpha_scale) in self._adapters.items():
            multiplier = self._multipliers.get(key, 0.0)
            if multiplier == 0.0:
                continue
            lx = x.astype(down_t.dtype) @ down_t
            if len(ups_t) == 1:
                lx = lx @ ups_t[0]
            else:
                chunks = mx.split(lx, len(ups_t), axis=-1)
                lx = mx.concatenate([c @ u for c, u in zip(chunks, ups_t)], axis=-1)
            out = out + lx.astype(out.dtype) * (multiplier * alpha_scale)
        return out


def _resolve_parent(mx_model, module_path: str):
    """Return `(parent, name)` for a dotted module path; integer parts index lists."""
    parts = module_path.split(".")
    obj = mx_model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj, parts[-1]


def attach_runtime_loras(
    mx_model,
    loras: dict[str, dict[str, mx.array]],
    dtype=mx.bfloat16,
) -> list[LoRALinear]:
    """Wrap every LoRA target Linear of `mx_model` in a `LoRALinear` carrying
    one adapter per entry of `loras` (`{key: lora_state_dict}`). All adapters
    start at multiplier 0. Call AFTER the base weights are loaded (and after
    `nn.quantize` for a quantized base): wrapping changes the module path of
    the base weight, so it must not happen before `load_weights`.

    Raises if a LoRA targets a module the model doesn't have: a silently
    skipped target would run that layer undistilled.
    """
    wrapped: dict[str, LoRALinear] = {}
    for key, state_dict in loras.items():
        for module_path, group in group_lora_tensors(state_dict).items():
            wrapper = wrapped.get(module_path)
            if wrapper is None:
                try:
                    parent, name = _resolve_parent(mx_model, module_path)
                    target = parent[int(name)] if name.isdigit() else getattr(parent, name)
                except (AttributeError, IndexError, KeyError) as e:
                    raise ValueError(f"LoRA {key!r} targets missing module {module_path}") from e
                if isinstance(target, LoRALinear):
                    wrapper = target
                else:
                    wrapper = LoRALinear(target)
                    if name.isdigit():
                        parent[int(name)] = wrapper
                    else:
                        setattr(parent, name, wrapper)
                wrapped[module_path] = wrapper
            wrapper.add_adapter(key, group, dtype=dtype)
    return list(wrapped.values())


class LoRAMix:
    """Per-step DMD/OPSD schedule of Avatar 1.6 (PT `enable_lora_mix` /
    `set_lora_mix_step`).

    Steps `< mix_steps` run `mix_alpha * DMD + (1 - mix_alpha) * OPSD`; later
    steps run OPSD alone. Upstream's documented default is `mix_steps=4,
    mix_alpha=1.0`: four DMD-only steps, then four OPSD-only steps.
    """

    def __init__(
        self,
        modules: Iterable[LoRALinear],
        dmd_key: str = "dmd",
        opsd_key: str = "opsd",
        mix_alpha: float = 1.0,
        mix_steps: int = 4,
    ):
        if not 0.0 <= mix_alpha <= 1.0:
            raise ValueError(f"mix_alpha must be in [0, 1], got {mix_alpha}")
        if mix_steps < 0:
            raise ValueError(f"mix_steps must be >= 0, got {mix_steps}")
        self.modules = list(modules)
        self.dmd_key = dmd_key
        self.opsd_key = opsd_key
        self.mix_alpha = float(mix_alpha)
        self.mix_steps = int(mix_steps)
        self._active: Optional[bool] = None
        self.set_step(0)

    def multipliers_for_step(self, step_index: int) -> tuple[float, float]:
        """`(dmd_multiplier, opsd_multiplier)` for a denoising step."""
        if step_index < self.mix_steps:
            return self.mix_alpha, 1.0 - self.mix_alpha
        return 0.0, 1.0

    def set_step(self, step_index: int) -> None:
        is_mixed = step_index < self.mix_steps
        if is_mixed == self._active:
            return
        dmd, opsd = self.multipliers_for_step(step_index)
        for module in self.modules:
            module.set_multiplier(self.dmd_key, dmd)
            module.set_multiplier(self.opsd_key, opsd)
        self._active = is_mixed
