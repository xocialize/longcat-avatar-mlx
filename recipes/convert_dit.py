"""Convert a Meituan `base_model/` DiT (PyTorch safetensors) to this port's MLX layout, one shard at a time.

    python -m recipes.convert_dit <LongCat-Video-Avatar-1.6>/base_model <out>/dit

Same rules as `convert_longcat_avatar.convert_dit` (conv weights to channels-last, AdaLN modulation in
fp32, everything else bf16) but from a local folder and without holding the whole 31 GB in memory, so
it also works for checkpoints that have no MLX conversion yet (Avatar 1.6, whose base_model differs
from 1.5's). Keys are unchanged; the LoRAs (loraA = DMD, loraB = OPSD) load as they are.
"""

import json
import pathlib
import shutil
import sys

import mlx.core as mx


def to_mlx(name: str, a: mx.array) -> mx.array:
    if "gamma" not in name:
        if a.ndim == 3:
            a = a.transpose(0, 2, 1)
        elif a.ndim == 4:
            a = a.transpose(0, 2, 3, 1)
        elif a.ndim == 5:
            a = a.transpose(0, 2, 3, 4, 1)
    if ".adaLN_modulation." in name or "audio_adaLN_modulation" in name:
        return a.astype(mx.float32)
    return a.astype(mx.bfloat16)


def convert(src: pathlib.Path, dst: pathlib.Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    idx = json.loads((src / "diffusion_pytorch_model.safetensors.index.json").read_text())
    weight_map = {}
    for shard in sorted(set(idx["weight_map"].values())):
        sd = mx.load(str(src / shard))
        out = {k: to_mlx(k, v) for k, v in sd.items()}
        mx.eval(list(out.values()))  # lazy arrays would be saved as zeros
        mx.save_safetensors(str(dst / shard), out)
        weight_map.update({k: shard for k in out})
        print(f"  {shard}: {len(out)} tensors", flush=True)
        del sd, out
        mx.clear_cache()
    (dst / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": weight_map}, indent=1))
    shutil.copy(src / "config.json", dst / "config.json")


if __name__ == "__main__":
    convert(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
