"""Smoke tests for the base LongCatVideoTransformer3DModel."""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.longcat_video_dit import LongCatVideoTransformer3DModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DIT_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--dit-config.json"


def _load_config() -> dict:
    return json.loads(DIT_CONFIG_PATH.read_text())


def test_dit_constructs_from_config():
    """Construct at full LongCat-Video defaults (48 layers, 4096 hidden).

    Just allocates the parameter tensors; no forward pass.
    """
    cfg = _load_config()
    model = LongCatVideoTransformer3DModel.from_config(cfg)
    assert model.depth == 48
    assert len(model.blocks) == 48
    assert model.x_embedder.embed_dim == 4096
    assert model.patch_size == (1, 2, 2)
    # x_embedder.proj has weight shape (O=4096, kT=1, kH=2, kW=2, I=16)
    assert model.x_embedder.proj.weight.shape == (4096, 1, 2, 2, 16)
    # First block's FFN inner dim should be 11008 (SwiGLU 2/3 reduction)
    assert model.blocks[0].ffn.hidden_dim == 11008


def test_dit_forward_tiny():
    """Tiny forward pass on a 4-block 64-dim model with small spatial size.

    Confirms the full block chain composes correctly without OOM or shape bugs.
    """
    # Override config for a tiny-but-realistic instance
    cfg = {
        "in_channels": 4,
        "out_channels": 4,
        "hidden_size": 64,
        "depth": 2,
        "num_heads": 4,
        "caption_channels": 32,
        "mlp_ratio": 4,
        "adaln_tembed_dim": 32,
        "frequency_embedding_size": 32,
        "patch_size": [1, 2, 2],
        "text_tokens_zero_pad": False,
    }
    model = LongCatVideoTransformer3DModel.from_config(cfg)

    # 1 batch, 4 channels, 2 frames, 4x4 spatial
    h = mx.random.normal((1, 4, 2, 4, 4))
    t = mx.array([0.5], dtype=mx.float32)  # single timestep, will broadcast to [B=1, N_t=2]
    # encoder_hidden_states: [B, 1, N_text, C_text]
    text = mx.random.normal((1, 1, 6, 32))
    # encoder_attention_mask: [B, 1, 1, N_text] all 1s
    mask = mx.ones((1, 1, 1, 6))

    out = model(h, t, text, encoder_attention_mask=mask)
    mx.eval(out)
    # Output: [B, out_channels, T*T_p, H*H_p, W*W_p]. T_p=1, H_p=W_p=2.
    # N_t=2, T_p=1 -> 2 frames. N_h=2, H_p=2 -> 4. N_w=2, W_p=2 -> 4.
    assert out.shape == (1, 4, 2, 4, 4)


def test_dit_unpatchify_explicit_shape():
    """Direct test of the unpatchify rearrangement."""
    cfg = _load_config()
    model = LongCatVideoTransformer3DModel.from_config(cfg)
    # Make a synthetic post-final-layer output:
    # [B, N_t*N_h*N_w, T_p*H_p*W_p*C_out] = [1, 12, 1*2*2*16] = [1, 12, 64]
    # for N_t=3, N_h=2, N_w=2.
    x = mx.random.normal((1, 12, 64))
    out = model._unpatchify(x, N_t=3, N_h=2, N_w=2)
    mx.eval(out)
    # Output: [1, 16, 3, 4, 4]
    assert out.shape == (1, 16, 3, 4, 4)


if __name__ == "__main__":
    test_dit_constructs_from_config()
    print("test_dit_constructs_from_config: PASS")
    test_dit_forward_tiny()
    print("test_dit_forward_tiny: PASS")
    test_dit_unpatchify_explicit_shape()
    print("test_dit_unpatchify_explicit_shape: PASS")
