"""Dump fp32 references for the Avatar 1.6 LoRA side-paths, for the Swift parity test.

The end-to-end pipeline fixture cannot pin the LoRA math down: bf16 kernels
drift between MLX bindings by about as much as the DMD -> OPSD switch changes
the output. So this fixture isolates the LoRA term itself, in fp32, where the
two runtimes agree closely: for a few target modules (one of each kind,
including both split-fused ones) it records a fixed input `x` and
`up(down(x)) * alpha_scale` for the real DMD and OPSD LoRAs.

A Swift port that mis-ordered the split-fused blocks, mis-read `alpha_scale`,
or swapped the two files would miss these by orders of magnitude more than
the fp32 tolerance.

Usage:
    .venv/bin/python scripts/dump_lora_swift_fixtures.py \\
        --variant-dir <weights>/LongCat-Video-Avatar-1.6-bf16 \\
        --out <swift repo>/Tests/LongCatVideoAvatarTests/Resources/lora-parity-v16/
"""

from __future__ import annotations

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np

from longcat_video_avatar.lora import compute_merged_delta, group_lora_tensors

# One of each target kind; qkv (3 splits) and kv_linear (2 splits) are the split-fused ones.
MODULES = [
    "blocks.0.attn.qkv",
    "blocks.17.attn.proj",
    "blocks.23.cross_attn.q_linear",
    "blocks.31.cross_attn.kv_linear",
    "blocks.40.ffn.w1",
    "blocks.47.ffn.w2",
]
TOKENS = 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    loras = {
        name: group_lora_tensors(mx.load(str(args.variant_dir / "lora" / f"{name}_lora.safetensors")))
        for name in ("dmd", "opsd")
    }
    manifest = {
        "generator": "longcat-avatar-mlx/scripts/dump_lora_swift_fixtures.py",
        "source_weights": args.variant_dir.name,
        "seed": args.seed,
        "target_modules": {name: len(groups) for name, groups in loras.items()},
        "modules": {},
    }
    rng = np.random.default_rng(args.seed)
    for index, module in enumerate(MODULES):
        in_dim = loras["dmd"][module]["lora_down.weight"].shape[1]
        x = rng.standard_normal((1, TOKENS, in_dim)).astype(np.float32)
        np.save(args.out / f"m{index}_x.npy", x)
        entry = {"path": module, "in": in_dim}
        for name, groups in loras.items():
            group = {k: v.astype(mx.float32) for k, v in groups[module].items()}
            # CPU stream: exact fp32 (Metal's fp32 matmul is not bit-faithful, skill-lessons L10).
            with mx.stream(mx.cpu):
                y = mx.array(x) @ compute_merged_delta(group).T
                mx.eval(y)
            y = np.asarray(y)
            np.save(args.out / f"m{index}_{name}.npy", y)
            entry[name] = {"abs_max": float(np.abs(y).max()), "out": int(y.shape[-1])}
            entry["splits"] = sum(1 for k in group if k.startswith("lora_up.blocks.")) or 1
            entry["alpha_scale"] = float(group["alpha_scale"].item())
        dmd, opsd = np.load(args.out / f"m{index}_dmd.npy"), np.load(args.out / f"m{index}_opsd.npy")
        entry["dmd_vs_opsd_abs_max"] = float(np.abs(dmd - opsd).max())
        manifest["modules"][f"m{index}"] = entry
        print(f"  {module}: in {in_dim}, out {dmd.shape[-1]}, splits {entry['splits']}, "
              f"|dmd| {entry['dmd']['abs_max']:.3g}, |opsd| {entry['opsd']['abs_max']:.3g}, "
              f"|dmd - opsd| {entry['dmd_vs_opsd_abs_max']:.3g}")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    total = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"Total fixture size: {total / 1024:.1f} KB")


if __name__ == "__main__":
    main()
