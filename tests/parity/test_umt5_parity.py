"""PT↔MLX parity for UMT5-XXL (Stage 1.2 gate).

Two tests:
- `test_umt5_keymap_renames_completely`: cheap; uses just the safetensors INDEX
  (~22 KB) to verify our rename function maps every PT key to a key that
  exists in our MLX model.
- `test_umt5_forward_parity`: full parity. Requires 22 GB of umT5 weights via
  `LONGCAT_UMT5_AUTO_DOWNLOAD=1`.

Pass criteria (full parity): max_abs < 5e-3 on the encoder's final hidden
state for a seeded token sequence. We use 5e-3 (vs 1e-3 for VAE encode)
because UMT5 has 24 blocks of unscaled-attention fp32-softmax chains —
larger accumulation surface than the VAE encoder.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

try:
    import torch
    from transformers import UMT5EncoderModel as PTUMT5EncoderModel

    _PT_AVAILABLE = True
except Exception:
    _PT_AVAILABLE = False

import mlx.core as mx

from longcat_video_avatar.models.umt5 import UMT5EncoderModel
from tests.parity._helpers import assert_parity, make_seeded_input

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--text_encoder-config.json"

HF_REPO = "meituan-longcat/LongCat-Video"
HF_TEXT_ENCODER_INDEX = "text_encoder/model.safetensors.index.json"


# Re-export from the model module (single source of truth)
from longcat_video_avatar.models.umt5 import rename_pt_to_mx  # noqa: F401


def test_umt5_keymap_renames_completely():
    """Verify every PT key in the umT5 checkpoint maps to a key our MLX
    model exposes via `parameters()`. Cheap — uses only the safetensors INDEX.
    """
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_TEXT_ENCODER_INDEX)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    pt_keys = sorted(weight_map.keys())

    cfg = json.loads(CONFIG_PATH.read_text())
    mx_model = UMT5EncoderModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    unmapped = []
    for pt_k in pt_keys:
        mx_k = rename_pt_to_mx(pt_k)
        if mx_k == pt_k or mx_k not in mx_keys:
            unmapped.append((pt_k, mx_k))

    assert not unmapped, (
        f"{len(unmapped)} PT keys do not map to an MLX parameter:\n  "
        + "\n  ".join(f"{pt} -> {mx}" for pt, mx in unmapped[:10])
    )


def _locate_weights() -> str | None:
    env_dir = os.environ.get("LONGCAT_UMT5_WEIGHTS_DIR")
    if env_dir and (pathlib.Path(env_dir) / "model.safetensors.index.json").exists():
        return env_dir
    if os.environ.get("LONGCAT_UMT5_AUTO_DOWNLOAD") == "1":
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=HF_REPO, allow_patterns=["text_encoder/*"])
    return None


def _load_mx_umt5_from_pt_state_dict(state_dict: dict) -> UMT5EncoderModel:
    cfg = json.loads(CONFIG_PATH.read_text())
    mx_model = UMT5EncoderModel.from_config(cfg)
    from mlx.utils import tree_unflatten

    flat = []
    for pt_k, pt_v in state_dict.items():
        mx_k = rename_pt_to_mx(pt_k)
        arr = pt_v.detach().cpu().float().numpy()
        # umT5 has only Linear + Embedding weights — no Conv*d transpose needed.
        # PT Linear weight: (out, in). MLX Linear weight: (out, in). Identical.
        # PT Embedding weight: (vocab, dim). MLX Embedding weight: (vocab, dim). Identical.
        flat.append((mx_k, mx.array(arr)))
    mx_model.update(tree_unflatten(flat))
    mx.eval(mx_model.parameters())
    return mx_model


@pytest.mark.skipif(
    not _PT_AVAILABLE, reason="parity dep missing — install with `pip install -e '.[parity]'`"
)
def test_umt5_forward_parity():
    weights_dir = _locate_weights()
    if not weights_dir:
        pytest.skip(
            "UMT5 weights not located (~22 GB). Set LONGCAT_UMT5_WEIGHTS_DIR "
            "or LONGCAT_UMT5_AUTO_DOWNLOAD=1 to enable."
        )

    cfg = json.loads(CONFIG_PATH.read_text())
    pt_model = PTUMT5EncoderModel.from_pretrained(
        os.path.join(weights_dir, "text_encoder") if not weights_dir.endswith("text_encoder") else weights_dir,
        torch_dtype=torch.float32,
    )
    pt_model.eval()

    # Build MLX version with the same weights.
    from safetensors.torch import load_file
    import glob

    state_dict: dict = {}
    text_encoder_dir = (
        weights_dir if weights_dir.endswith("text_encoder") else os.path.join(weights_dir, "text_encoder")
    )
    for shard in sorted(glob.glob(os.path.join(text_encoder_dir, "model-*-of-*.safetensors"))):
        state_dict.update(load_file(shard))
    mx_model = _load_mx_umt5_from_pt_state_dict(state_dict)

    # Drive both with the same seeded token sequence.
    rng = np.random.default_rng(42)
    ids_np = rng.integers(0, cfg["vocab_size"], size=(1, 24), dtype=np.int64)
    mask_np = np.ones((1, 24), dtype=np.int64)

    with torch.no_grad():
        pt_out = pt_model(
            input_ids=torch.from_numpy(ids_np),
            attention_mask=torch.from_numpy(mask_np),
        ).last_hidden_state

    mx_out = mx_model(mx.array(ids_np), mask=mx.array(mask_np))

    assert_parity(pt_out, mx_out, threshold=5e-3, name="umt5.encoder")
