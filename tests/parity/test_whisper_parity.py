"""PT↔MLX parity for Whisper-large-v3 encoder (Stage 1.3 gate).

Two tests:
- `test_whisper_keymap_no_rename_needed`: validates that EVERY PT key in the
  Whisper encoder checkpoint exists in our MLX model's parameters (after the
  `model.encoder.` strip — we don't carry the decoder, and we strip
  `model.encoder.` because we only ported the encoder). No download — uses
  just the safetensors INDEX (~120 KB).
- `test_whisper_forward_parity`: full parity. Gated on `LONGCAT_WHISPER_AUTO_DOWNLOAD=1`.
"""

from __future__ import annotations

import json
import os
import pathlib

import numpy as np
import pytest

try:
    import torch
    from transformers import WhisperModel

    _PT_AVAILABLE = True
except Exception:
    _PT_AVAILABLE = False

import mlx.core as mx

from longcat_video_avatar.models.whisper import WhisperEncoder
from tests.parity._helpers import assert_parity

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "avatar-1.5--whisper-large-v3-config.json"

HF_REPO = "meituan-longcat/LongCat-Video-Avatar-1.5"
HF_WHISPER_INDEX = "whisper-large-v3/model.safetensors.index.fp32.json"


def rename_pt_to_mx(pt_key: str) -> str | None:
    """Strip `model.encoder.` prefix from Meituan's full-Whisper checkpoint
    to get our encoder-only key. Return None for decoder/proj_out/etc. keys we don't load.
    """
    if pt_key.startswith("model.encoder."):
        return pt_key[len("model.encoder.") :]
    return None  # decoder.*, proj_out, etc. — not part of our encoder-only port


def test_whisper_keymap_no_rename_needed():
    """Every encoder key in Meituan's Whisper checkpoint maps to an MLX
    parameter via simple `model.encoder.` prefix strip.
    """
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_WHISPER_INDEX)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    pt_keys = sorted(weight_map.keys())

    encoder_keys = [k for k in pt_keys if k.startswith("model.encoder.")]
    cfg = json.loads(CONFIG_PATH.read_text())
    mx_model = WhisperEncoder.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    unmapped = []
    for pt_k in encoder_keys:
        mx_k = rename_pt_to_mx(pt_k)
        if mx_k is None or mx_k not in mx_keys:
            unmapped.append((pt_k, mx_k))

    assert not unmapped, (
        f"{len(unmapped)} encoder PT keys not in MLX model:\n  "
        + "\n  ".join(f"{pt} -> {mx}" for pt, mx in unmapped[:15])
    )

    # Also confirm the count matches what we expect for 32-layer encoder
    # 7 top-level (conv1.w/b, conv2.w/b, embed_positions.w, layer_norm.w/b)
    # + 32 layers × 15 keys/layer = 487
    assert len(encoder_keys) == 487, f"expected 487 encoder keys, got {len(encoder_keys)}"


def _locate_weights() -> str | None:
    env_dir = os.environ.get("LONGCAT_WHISPER_WEIGHTS_DIR")
    if env_dir and (pathlib.Path(env_dir) / "model.safetensors").exists():
        return env_dir
    if os.environ.get("LONGCAT_WHISPER_AUTO_DOWNLOAD") == "1":
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=HF_REPO,
            allow_patterns=["whisper-large-v3/model.safetensors", "whisper-large-v3/config.json"],
        )
    return None


def _load_mx_whisper_from_pt_state_dict(state_dict: dict) -> WhisperEncoder:
    cfg = json.loads(CONFIG_PATH.read_text())
    mx_model = WhisperEncoder.from_config(cfg)
    from mlx.utils import tree_unflatten

    flat = []
    for pt_k, pt_v in state_dict.items():
        mx_k = rename_pt_to_mx(pt_k)
        if mx_k is None:
            continue  # skip decoder / unused keys
        arr = pt_v.detach().cpu().float().numpy()
        # Conv1d weights: PT `(O, I, K)` -> MLX `(O, K, I)`
        if arr.ndim == 3:
            arr = arr.transpose(0, 2, 1)
        flat.append((mx_k, mx.array(arr)))
    mx_model.update(tree_unflatten(flat))
    mx.eval(mx_model.parameters())
    return mx_model


@pytest.mark.skipif(
    not _PT_AVAILABLE, reason="parity dep missing — install with `pip install -e '.[parity]'`"
)
def test_whisper_forward_parity():
    weights_dir = _locate_weights()
    if not weights_dir:
        pytest.skip(
            "Whisper weights not located (~3 GB). Set LONGCAT_WHISPER_WEIGHTS_DIR "
            "or LONGCAT_WHISPER_AUTO_DOWNLOAD=1 to enable."
        )

    # Resolve the whisper-large-v3 subdir if weights_dir is a snapshot root
    whisper_dir = (
        weights_dir
        if pathlib.Path(weights_dir).name == "whisper-large-v3"
        else str(pathlib.Path(weights_dir) / "whisper-large-v3")
    )

    pt_model = WhisperModel.from_pretrained(whisper_dir, torch_dtype=torch.float32).encoder
    pt_model.eval()

    # MLX side: load same checkpoint via our rename function.
    from safetensors.torch import load_file

    state_dict = load_file(str(pathlib.Path(whisper_dir) / "model.safetensors"))
    mx_model = _load_mx_whisper_from_pt_state_dict(state_dict)

    # Seeded mel spectrogram: 1 batch, 128 bins, 3000 frames (Whisper standard)
    rng = np.random.default_rng(42)
    mel_np = rng.standard_normal((1, 128, 3000)).astype(np.float32)

    with torch.no_grad():
        pt_hidden = pt_model(
            input_features=torch.from_numpy(mel_np),
            output_hidden_states=True,
        ).hidden_states  # tuple of 33 tensors

    mx_hidden = mx_model(mx.array(mel_np), return_all_hidden_states=True)
    assert len(mx_hidden) == 33

    # Compare each hidden state. The first one (post-conv+pos) is the
    # easiest case (only conv + pos_embed math). The deepest layers
    # accumulate fp32-precision drift through 32 attention blocks.
    for i in (0, 1, 16, 31, 32):
        # Per CLAUDE.md L10: realistic threshold for a 32-block encoder on
        # Metal GPU vs PT CPU fp32 is ~5e-3 for early layers, ~1e-2 for last.
        threshold = 1e-2 if i >= 16 else 5e-3
        assert_parity(pt_hidden[i], mx_hidden[i], threshold=threshold, name=f"whisper.hidden[{i}]")
