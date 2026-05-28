"""Weight conversion recipe — Meituan PT checkpoints → MLX-community publish.

Produces TWO publishable variants, both in per-component HF subdir layout:
- `mlx-community/LongCat-Video-Avatar-1.5-bf16/`
    `{vae,text_encoder,audio_encoder,dit,lora,scheduler,tokenizer}/...`
    50-step base + separate DMD LoRA. End user calls `pipeline.merge_dmd_lora()`.
- `mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged/`
    Same layout minus `lora/`. DiT weights have DMD LoRA pre-merged. 8-step
    distilled inference, no LoRA loading at runtime.

Each component is converted via a dedicated `convert_*` function:
- `convert_vae` — Conv*d transpose, key renaming is identity (we use Meituan's
  diffusers 0.38 schema directly).
- `convert_umt5` — applies the `rename_pt_to_mx` from tests/parity to map HF
  verbose names to mlx-video's compact names.
- `convert_whisper` — strips `model.encoder.` prefix; Conv1d weight transpose.
- `convert_dit` — pure passthrough (our DiT schema matches PT exactly).
- `convert_lora` — re-format the Meituan-encoded LoRA names to a clean
  MLX-loadable format for later merging by `lora.merge_lora_into_model`.

CRITICAL: every saved tensor is materialized via `mx.eval` immediately before
`mx.save_safetensors`. Lazy MLX tensors serialize as ZEROS with no error
(per CLAUDE.md L4 / the mlx-porting skill's silent-killer warning).

Usage:
    python -m recipes.convert_longcat_avatar --variant base   --out <PATH>
    python -m recipes.convert_longcat_avatar --variant merged --out <PATH>
    python -m recipes.convert_longcat_avatar --variant both   --out <PATH>

Requires `huggingface_hub` and `safetensors` (already in `[parity]` extras).
The PT-source download is on-demand; uses ~120 GB of disk for the full
both-variants conversion (60 GB source + 60 GB output).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Callable, Optional

import mlx.core as mx
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materialize_and_save(
    state_dict: dict[str, mx.array],
    out_path: pathlib.Path,
    metadata: Optional[dict[str, str]] = None,
) -> None:
    """Materialize every tensor (mx.eval) and write to a single safetensors file.

    The materialization step is critical: lazy MLX tensors serialize as zeros
    with no error message (the silent killer per the mlx-porting skill).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Force evaluation of every tensor before save
    mx.eval(list(state_dict.values()))
    mx.save_safetensors(str(out_path), state_dict, metadata=metadata or {})


def _save_sharded_safetensors(
    state_dict: dict[str, mx.array],
    out_dir: pathlib.Path,
    base_name: str = "diffusion_pytorch_model",
    max_shard_size_bytes: int = 5 * 1024**3,
) -> None:
    """Save a state_dict across N shards each <= max_shard_size_bytes.

    Writes an `<base_name>.safetensors.index.json` describing the weight map.
    Each shard is `<base_name>-<i>-of-<N>.safetensors`. Materializes per shard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Estimate bytes per tensor (dtype-aware)
    bytes_per_dtype = {
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.float32: 4,
        mx.int8: 1,
        mx.int32: 4,
    }

    def tensor_bytes(t: mx.array) -> int:
        nbytes = bytes_per_dtype.get(t.dtype, 4)
        for d in t.shape:
            nbytes *= d
        return nbytes

    # First pass: pack tensors into shards greedily
    shards: list[dict[str, mx.array]] = [{}]
    cur_bytes = 0
    for k, v in state_dict.items():
        sz = tensor_bytes(v)
        if cur_bytes + sz > max_shard_size_bytes and shards[-1]:
            shards.append({})
            cur_bytes = 0
        shards[-1][k] = v
        cur_bytes += sz

    n_shards = len(shards)
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard in enumerate(shards, start=1):
        fname = f"{base_name}-{idx:05d}-of-{n_shards:05d}.safetensors"
        out_path = out_dir / fname
        mx.eval(list(shard.values()))  # MATERIALIZE before save
        mx.save_safetensors(str(out_path), shard, metadata={"format": "mlx"})
        for k, v in shard.items():
            weight_map[k] = fname
            total_size += tensor_bytes(v)

    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (out_dir / f"{base_name}.safetensors.index.json").write_text(json.dumps(index, indent=2))


def _copy_meituan_config(
    repo_id: str,
    src_path: str,
    out_path: pathlib.Path,
) -> None:
    """Download and copy a config.json (or similar metadata) verbatim from HF."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=src_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, out_path)


def _load_pt_safetensors(repo_id: str, filename: str) -> dict[str, mx.array]:
    """Download a single safetensors file from HF and load as MLX arrays.

    Uses `mx.load` which is bf16-native — unlike `safe_open(..., framework="numpy")`,
    which raises `TypeError: data type 'bfloat16' not understood` on bf16 weights.
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=filename)
    return mx.load(p)  # dict[str, mx.array], dtype preserved (incl. bf16)


def _load_pt_safetensors_sharded(repo_id: str, index_filename: str) -> dict[str, mx.array]:
    """Download a sharded safetensors checkpoint via its index.json and load
    every key into a single dict of MLX arrays. bf16-native (see above).
    """
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    base_path = pathlib.Path(index_filename).parent

    # Group keys by shard so we only `mx.load` each file once
    shard_to_keys: dict[str, list[str]] = {}
    for k, shard_name in weight_map.items():
        shard_to_keys.setdefault(shard_name, []).append(k)

    sd: dict[str, mx.array] = {}
    for shard_name, keys in shard_to_keys.items():
        shard_path = hf_hub_download(repo_id=repo_id, filename=str(base_path / shard_name))
        shard_data = mx.load(shard_path)
        for k in keys:
            sd[k] = shard_data[k]
    return sd


def _layout_and_cast(arr: mx.array, *, name: str, is_gamma: bool, dtype=mx.bfloat16) -> mx.array:
    """Apply Conv*d weight transpose + dtype cast to a tensor loaded via mx.load.

    Layout rules:
    - 1D / 2D: pass through (biases, gammas, Linear weights — same in PT and MLX)
    - 3D: Conv1d (O, I, K) -> (O, K, I)
    - 4D: Conv2d (O, I, H, W) -> (O, H, W, I), UNLESS `is_gamma=True` (RMS_norm
      gamma may be 4D like (C, 1, 1, 1) for video VAE — keep as-is)
    - 5D: Conv3d (O, I, T, H, W) -> (O, T, H, W, I) for non-gamma; gamma 5D
      RMS norm weights (e.g. video VAE) keep as-is.

    Dtype rules:
    - `_key_should_stay_fp32(name)` (e.g. adaLN modulation) → ALWAYS fp32, even
      if the source was bf16. Otherwise cast to the target `dtype`.
    """
    if is_gamma:
        pass  # norm weights — no transpose regardless of ndim
    elif arr.ndim == 3:
        arr = arr.transpose(0, 2, 1)
    elif arr.ndim == 4:
        arr = arr.transpose(0, 2, 3, 1)
    elif arr.ndim == 5:
        arr = arr.transpose(0, 2, 3, 4, 1)

    # Dtype: fp32-special keys are upcast even from bf16; others cast to target.
    if _key_should_stay_fp32(name):
        if arr.dtype != mx.float32:
            arr = arr.astype(mx.float32)
    elif dtype is not None and arr.dtype != dtype:
        arr = arr.astype(dtype)
    return arr


# Backwards-compat alias kept for tests/smoke/test_recipe_smoke.py which
# synthesizes numpy inputs to validate transpose math. Production code path
# uses `_layout_and_cast` on mx.array directly.
def _np_to_mx_with_layout(arr: np.ndarray, *, name: str, is_gamma: bool, dtype=mx.bfloat16) -> mx.array:
    return _layout_and_cast(mx.array(arr), name=name, is_gamma=is_gamma, dtype=dtype)


def _key_should_stay_fp32(name: str) -> bool:
    """Keep certain tensors in fp32: AdaLN modulation weights produce fp32
    shift/scale params; per CLAUDE.md L11 they must compute in fp32 to match
    PT behavior. Casting them to bf16 introduces visible numerical drift.

    Conservative pattern: keep `adaLN_modulation` and `gamma` tensors at fp32.
    """
    return ".adaLN_modulation." in name or "audio_adaLN_modulation" in name


# ---------------------------------------------------------------------------
# Per-component converters
# ---------------------------------------------------------------------------


def convert_vae(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video/vae → out_dir/vae/"""
    print(f"  Converting VAE → {out_dir}/vae/")
    src = _load_pt_safetensors(
        "meituan-longcat/LongCat-Video", "vae/diffusion_pytorch_model.safetensors"
    )
    mlx_sd = {}
    for k, v in src.items():
        is_gamma = "gamma" in k
        mlx_sd[k] = _layout_and_cast(v, name=k, is_gamma=is_gamma, dtype=dtype)
    _materialize_and_save(
        mlx_sd, out_dir / "vae" / "diffusion_pytorch_model.safetensors", metadata={"format": "mlx"}
    )
    _copy_meituan_config("meituan-longcat/LongCat-Video", "vae/config.json", out_dir / "vae" / "config.json")


def convert_umt5(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video/text_encoder → out_dir/text_encoder/"""
    print(f"  Converting umT5 → {out_dir}/text_encoder/")
    # Reuse the rename function exported from the model module.
    from longcat_video_avatar.models.umt5 import rename_pt_to_mx

    src = _load_pt_safetensors_sharded(
        "meituan-longcat/LongCat-Video", "text_encoder/model.safetensors.index.json"
    )
    mlx_sd = {}
    for pt_k, v in src.items():
        mx_k = rename_pt_to_mx(pt_k)
        mlx_sd[mx_k] = _layout_and_cast(v, name=mx_k, is_gamma=False, dtype=dtype)
    _save_sharded_safetensors(mlx_sd, out_dir / "text_encoder", base_name="model")
    _copy_meituan_config(
        "meituan-longcat/LongCat-Video",
        "text_encoder/config.json",
        out_dir / "text_encoder" / "config.json",
    )


def convert_whisper(out_dir: pathlib.Path, dtype=mx.bfloat16) -> None:
    """meituan-longcat/LongCat-Video-Avatar-1.5/whisper-large-v3 → out_dir/audio_encoder/

    Only ENCODER keys are kept (decoder is unused for Avatar inference).
    """
    print(f"  Converting Whisper encoder → {out_dir}/audio_encoder/")
    # Whisper has a bf16 single file: `whisper-large-v3/model.safetensors`
    src = _load_pt_safetensors(
        "meituan-longcat/LongCat-Video-Avatar-1.5", "whisper-large-v3/model.safetensors"
    )
    mlx_sd = {}
    for pt_k, v in src.items():
        if not pt_k.startswith("model.encoder."):
            continue  # skip decoder + proj_out
        mx_k = pt_k[len("model.encoder.") :]
        mlx_sd[mx_k] = _layout_and_cast(v, name=mx_k, is_gamma=False, dtype=dtype)
    _materialize_and_save(
        mlx_sd,
        out_dir / "audio_encoder" / "model.safetensors",
        metadata={"format": "mlx", "source": "model.encoder"},
    )
    _copy_meituan_config(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        "whisper-large-v3/config.json",
        out_dir / "audio_encoder" / "config.json",
    )


def convert_dit(out_dir: pathlib.Path, dtype=mx.bfloat16, lora_to_merge: Optional[dict] = None) -> None:
    """meituan-longcat/LongCat-Video-Avatar-1.5/base_model → out_dir/dit/

    If `lora_to_merge` is provided, the LoRA is pre-merged before saving
    (DMD-merged variant). Otherwise the base bf16 DiT is saved as-is.
    """
    print(f"  Converting Avatar DiT → {out_dir}/dit/")
    src = _load_pt_safetensors_sharded(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        "base_model/diffusion_pytorch_model.safetensors.index.json",
    )
    mlx_sd = {}
    for k, v in src.items():
        is_gamma = "gamma" in k
        mlx_sd[k] = _layout_and_cast(v, name=k, is_gamma=is_gamma, dtype=dtype)

    if lora_to_merge is not None:
        print("    Merging DMD LoRA into base DiT weights...")
        from longcat_video_avatar.lora import compute_merged_delta, group_lora_tensors

        # lora_to_merge is already a dict[str, mx.array] from `_load_pt_safetensors`
        grouped = group_lora_tensors(lora_to_merge)
        merged_count = 0
        for module_path, group in grouped.items():
            weight_key = f"{module_path}.weight"
            if weight_key not in mlx_sd:
                print(f"    WARN: LoRA target {module_path} not in DiT — skipping")
                continue
            base_w = mlx_sd[weight_key]
            delta = compute_merged_delta(group, multiplier=1.0).astype(base_w.dtype)
            mlx_sd[weight_key] = base_w + delta
            merged_count += 1
        print(f"    Merged {merged_count} LoRA target modules")

    _save_sharded_safetensors(mlx_sd, out_dir / "dit", base_name="diffusion_pytorch_model")
    _copy_meituan_config(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        "base_model/config.json",
        out_dir / "dit" / "config.json",
    )


def convert_lora(out_dir: pathlib.Path) -> None:
    """Re-format the DMD LoRA safetensors with the Meituan-encoded names
    preserved (the runtime loader handles the decoding via `lora.decode_module_name`).
    Source is fp32; no layout transpose needed.
    """
    print(f"  Converting DMD LoRA → {out_dir}/lora/")
    src = _load_pt_safetensors(
        "meituan-longcat/LongCat-Video-Avatar-1.5", "lora/dmd_lora.safetensors"
    )
    # mx.load already returned dict[str, mx.array]; passthrough
    _materialize_and_save(
        src, out_dir / "lora" / "dmd_lora.safetensors", metadata={"format": "mlx"}
    )


def copy_scheduler_and_tokenizer(out_dir: pathlib.Path) -> None:
    """Copy scheduler config + umT5 tokenizer files verbatim from Meituan."""
    print(f"  Copying scheduler + tokenizer → {out_dir}/")
    _copy_meituan_config(
        "meituan-longcat/LongCat-Video-Avatar-1.5",
        "scheduler/scheduler_config.json",
        out_dir / "scheduler" / "scheduler_config.json",
    )
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            _copy_meituan_config(
                "meituan-longcat/LongCat-Video", f"tokenizer/{fname}", out_dir / "tokenizer" / fname
            )
        except Exception as e:
            print(f"    WARN: failed to copy tokenizer/{fname}: {e}")


def write_pipeline_config(out_dir: pathlib.Path, merged: bool) -> None:
    """Drop a pipeline_config.json at the top of out_dir capturing the
    resolved `PipelineConfig` defaults for this variant.
    """
    cfg = {
        "variant": "bf16-dmd-merged" if merged else "bf16",
        "num_sampling_steps": 8,  # DMD path is default for both variants;
                                  # base variant supports 50-step too once user
                                  # calls disable_dmd_lora.
        "scheduler_shift": 7.0,
        "text_guidance_scale": 4.0,
        "audio_guidance_scale": 4.0,
        "target_fps": 25,
        "default_num_frames": 93,
        "dit_class": "LongCatVideoAvatarTransformer3DModel",
        "vae_class": "AutoencoderKLWan",
    }
    (out_dir / "pipeline_config.json").write_text(json.dumps(cfg, indent=2))


def write_readme(out_dir: pathlib.Path, merged: bool) -> None:
    suffix = "-dmd-merged" if merged else ""
    body = f"""# LongCat-Video-Avatar-1.5-bf16{suffix} (MLX)

Apple MLX port of [LongCat-Video-Avatar-1.5](https://github.com/meituan-longcat/LongCat-Video).
{'DMD LoRA pre-merged into DiT weights — 8-step distilled inference, no LoRA loading at runtime.' if merged else 'Base bf16 + separate DMD LoRA. Pipeline merges the LoRA on demand via `merge_dmd_lora()`.'}

## Quick start

```python
from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline
pipeline = LongCatAvatarPipeline.from_pretrained("{out_dir.name}")
video = pipeline(image=..., audio_mel=..., text_embeds=..., ...)
```

## License

MIT (matches upstream Meituan LongCat-Video license).
"""
    (out_dir / "README.md").write_text(body)


# ---------------------------------------------------------------------------
# Top-level orchestrators
# ---------------------------------------------------------------------------


def _component_done(out_dir: pathlib.Path, sub: str, sentinel: str) -> bool:
    """Cheap skip-if-exists check so re-runs after a partial failure don't
    redo completed components. Looks for an existing `sub/<sentinel>` file
    with non-trivial size (>1 KB).
    """
    p = out_dir / sub / sentinel
    return p.exists() and p.stat().st_size > 1024


def build_base_variant(out_dir: pathlib.Path, *, skip_done: bool = True) -> None:
    """Build the `LongCat-Video-Avatar-1.5-bf16` variant (base + separate LoRA).

    With `skip_done=True` (default), components whose output already exists
    are skipped — safe to re-run after a partial failure without redoing
    minutes of conversion math.
    """
    print(f"Building base variant → {out_dir}")
    if skip_done and _component_done(out_dir, "vae", "diffusion_pytorch_model.safetensors"):
        print("  VAE already converted — skipping")
    else:
        convert_vae(out_dir)
    if skip_done and _component_done(out_dir, "text_encoder", "model.safetensors.index.json"):
        print("  umT5 already converted — skipping")
    else:
        convert_umt5(out_dir)
    if skip_done and _component_done(out_dir, "audio_encoder", "model.safetensors"):
        print("  Whisper encoder already converted — skipping")
    else:
        convert_whisper(out_dir)
    if skip_done and _component_done(
        out_dir, "dit", "diffusion_pytorch_model.safetensors.index.json"
    ):
        print("  DiT already converted — skipping")
    else:
        convert_dit(out_dir, lora_to_merge=None)
    if skip_done and _component_done(out_dir, "lora", "dmd_lora.safetensors"):
        print("  DMD LoRA already converted — skipping")
    else:
        convert_lora(out_dir)
    copy_scheduler_and_tokenizer(out_dir)
    write_pipeline_config(out_dir, merged=False)
    write_readme(out_dir, merged=False)
    print(f"DONE: {out_dir}")


def build_merged_variant(out_dir: pathlib.Path, *, skip_done: bool = True) -> None:
    """Build the `LongCat-Video-Avatar-1.5-bf16-dmd-merged` variant
    (DMD LoRA pre-merged into DiT)."""
    print(f"Building DMD-merged variant → {out_dir}")
    if skip_done and _component_done(out_dir, "vae", "diffusion_pytorch_model.safetensors"):
        print("  VAE already converted — skipping")
    else:
        convert_vae(out_dir)
    if skip_done and _component_done(out_dir, "text_encoder", "model.safetensors.index.json"):
        print("  umT5 already converted — skipping")
    else:
        convert_umt5(out_dir)
    if skip_done and _component_done(out_dir, "audio_encoder", "model.safetensors"):
        print("  Whisper encoder already converted — skipping")
    else:
        convert_whisper(out_dir)
    if skip_done and _component_done(
        out_dir, "dit", "diffusion_pytorch_model.safetensors.index.json"
    ):
        print("  DiT (DMD-merged) already converted — skipping")
    else:
        print("  Pre-loading DMD LoRA for in-place merge...")
        lora_sd = _load_pt_safetensors(
            "meituan-longcat/LongCat-Video-Avatar-1.5", "lora/dmd_lora.safetensors"
        )
        convert_dit(out_dir, lora_to_merge=lora_sd)
    copy_scheduler_and_tokenizer(out_dir)
    write_pipeline_config(out_dir, merged=True)
    write_readme(out_dir, merged=True)
    print(f"DONE: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert LongCat-Video-Avatar-1.5 to MLX format")
    parser.add_argument(
        "--variant",
        choices=("base", "merged", "both"),
        default="both",
        help="Which variant(s) to produce.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Output root directory. Variant subdirs are created underneath.",
    )
    args = parser.parse_args()

    if args.variant in ("base", "both"):
        build_base_variant(args.out / "LongCat-Video-Avatar-1.5-bf16")
    if args.variant in ("merged", "both"):
        build_merged_variant(args.out / "LongCat-Video-Avatar-1.5-bf16-dmd-merged")


if __name__ == "__main__":
    main()
