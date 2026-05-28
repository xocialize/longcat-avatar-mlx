"""DMD LoRA loader (pre-merge) for LongCat-Video-Avatar-1.5.

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

from typing import Iterable

import mlx.core as mx


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
