"""Smoke test: construct WhisperEncoder from Whisper-large-v3 config and
confirm forward-pass shapes (no weights download)."""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.whisper import WhisperEncoder

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "avatar-1.5--whisper-large-v3-config.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def test_whisper_constructs_from_config():
    cfg = _load_config()
    model = WhisperEncoder.from_config(cfg)
    assert len(model.layers) == 32
    assert model.d_model == 1280
    assert model.max_source_positions == 1500
    assert model.layers[0].self_attn.num_heads == 20
    assert model.layers[0].self_attn.head_dim == 64
    # K has no bias (Whisper-specific): MLX represents this by simply not
    # having a `.bias` attribute on the Linear (vs PT's `bias=None`).
    assert not hasattr(model.layers[0].self_attn.k_proj, "bias"), (
        "k_proj should have NO bias (Whisper-specific)"
    )
    assert hasattr(model.layers[0].self_attn.q_proj, "bias"), "q_proj must have bias"
    assert hasattr(model.layers[0].self_attn.v_proj, "bias"), "v_proj must have bias"
    assert hasattr(model.layers[0].self_attn.out_proj, "bias"), "out_proj must have bias"


def test_whisper_forward_last_hidden_state():
    cfg = _load_config()
    model = WhisperEncoder.from_config(cfg)

    # Mock mel spectrogram: 1 batch, 128 mel bins, 3000 mel frames
    # (Whisper standard 30-sec input).
    mel = mx.random.normal((1, 128, 3000))
    out = model(mel, return_all_hidden_states=False)
    mx.eval(out)

    # After conv2 stride=2: T_enc = 3000 // 2 = 1500 (the max_source_positions)
    assert out.shape == (1, 1500, 1280), f"got {out.shape}"


def test_whisper_forward_all_hidden_states():
    cfg = _load_config()
    model = WhisperEncoder.from_config(cfg)

    mel = mx.random.normal((1, 128, 3000))
    hidden = model(mel, return_all_hidden_states=True)

    # 1 (post-conv+pos) + 32 (one per encoder layer) = 33
    assert isinstance(hidden, list)
    assert len(hidden) == 33, f"expected 33 hidden states, got {len(hidden)}"
    for i, h in enumerate(hidden):
        mx.eval(h)
        assert h.shape == (1, 1500, 1280), f"layer {i}: got {h.shape}"


if __name__ == "__main__":
    test_whisper_constructs_from_config()
    print("test_whisper_constructs_from_config: PASS")
    test_whisper_forward_last_hidden_state()
    print("test_whisper_forward_last_hidden_state: PASS")
    test_whisper_forward_all_hidden_states()
    print("test_whisper_forward_all_hidden_states: PASS")
