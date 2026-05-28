"""Smoke test: construct UMT5EncoderModel from Meituan's text_encoder config.json
and confirm forward-pass shapes are correct.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.umt5 import UMT5EncoderModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--text_encoder-config.json"


def _load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def test_umt5_constructs_from_config():
    cfg = _load_config()
    model = UMT5EncoderModel.from_config(cfg)
    # Sanity: 24 blocks, dim 4096, vocab 256384
    assert len(model.blocks) == 24
    assert model.dim == 4096
    # UMT5 uses per-block bias (shared_pos=False), so top-level pos_embedding is None
    assert model.pos_embedding is None
    # Each block has its own relative bias
    assert model.blocks[0].pos_embedding is not None
    assert model.blocks[0].pos_embedding.embedding.weight.shape == (32, 64)


def test_umt5_forward_shape():
    cfg = _load_config()
    model = UMT5EncoderModel.from_config(cfg)

    # 2 sequences of length 16
    ids = mx.random.randint(0, cfg["vocab_size"], shape=(2, 16))
    mask = mx.ones((2, 16))
    out = model(ids, mask=mask)
    mx.eval(out)

    assert out.shape == (2, 16, 4096)


if __name__ == "__main__":
    test_umt5_constructs_from_config()
    print("test_umt5_constructs_from_config: PASS")
    test_umt5_forward_shape()
    print("test_umt5_forward_shape: PASS")
