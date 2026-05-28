"""Smoke tests for the Avatar overlay: AudioProjModel + SingleStreamAttention +
LongCatVideoAvatarTransformer3DModel.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.avatar.blocks import AudioProjModel
from longcat_video_avatar.models.avatar.attention import SingleStreamAttention
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
    LongCatVideoAvatarTransformer3DModel,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AVATAR_DIT_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "avatar-1.5--base_model-config.json"


def _load_config() -> dict:
    return json.loads(AVATAR_DIT_CONFIG_PATH.read_text())


def test_audio_proj_model_output_shape_v15():
    """AudioProjModel with v1.5 config: 5 groups × 1280 Whisper hidden, output 32×768."""
    cfg = _load_config()
    ap = AudioProjModel(
        seq_len=cfg["audio_window"],
        seq_len_vf=cfg["audio_window"] + cfg["vae_scale"] - 1,
        blocks=cfg["audio_block"],
        channels=cfg["audio_channel"],
        intermediate_dim=cfg["intermediate_dim"],
        output_dim=cfg["output_dim"],
        context_tokens=cfg["context_tokens"],
    )
    # First frame audio: [B=1, F=1, W=5, S=5, C=1280]
    first = mx.random.normal((1, 1, 5, 5, 1280))
    # Latter frame audio: [B=1, F=2, W'=8, S=5, C=1280]
    latter = mx.random.normal((1, 2, 8, 5, 1280))
    out = ap(first, latter)
    mx.eval(out)
    # Expected: [B=1, video_length=1+2=3, context_tokens=32, output_dim=768]
    assert out.shape == (1, 3, 32, 768)


def test_single_stream_attention_basic():
    """SingleStreamAttention smoke: visual Q (B*N_t, S, hidden_size),
    audio K/V (B*N_t, audio_tokens, output_dim).
    """
    ssa = SingleStreamAttention(
        dim=64, encoder_hidden_states_dim=32, num_heads=4, qkv_bias=True, qk_norm=True
    )
    # B=1, latent T=2, spatial S=4 per frame, C=64
    x = mx.random.normal((1, 8, 64))  # 1 batch, 2*4 = 8 tokens, 64-dim
    # Per-frame audio: B*N_t = 1*2 = 2, audio_tokens=8, dim=32
    audio = mx.random.normal((2, 8, 32))

    audio_cond, audio_noise = ssa(x, audio, shape=(2, 2, 2), num_cond_latents=0)
    mx.eval(audio_noise)
    assert audio_cond is None
    assert audio_noise.shape == (1, 8, 64)


def test_avatar_dit_constructs_from_v15_config():
    cfg = _load_config()
    model = LongCatVideoAvatarTransformer3DModel.from_config(cfg)
    assert model.depth == 48
    assert len(model.blocks) == 48
    # Critical v1.5 audio config (vs v1.0)
    assert model.audio_proj.blocks == 5, f"v1.5 has 5 Whisper layer groups, got {model.audio_proj.blocks}"
    assert model.audio_proj.channels == 1280, f"v1.5 has 1280 hidden, got {model.audio_proj.channels}"
    # Audio cross-attn dims
    assert model.blocks[0].audio_cross_attn.encoder_hidden_states_dim == 768
    # Audio adaLN modulation outputs 3*hidden_size
    assert model.blocks[0].audio_adaLN_modulation[1].weight.shape == (3 * 4096, 512)


def test_avatar_dit_forward_tiny():
    """End-to-end Avatar DiT forward with a tiny config."""
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
        "audio_window": 5,
        "audio_block": 5,
        "audio_channel": 32,
        "intermediate_dim": 32,
        "output_dim": 32,
        "context_tokens": 8,
        "vae_scale": 4,
        "audio_prenorm": False,
        "class_range": 24,
        "class_interval": 4,
    }
    model = LongCatVideoAvatarTransformer3DModel.from_config(cfg)
    # B=1, C=4, T=5 (1 first + 1 vae_scale of 4 = 5 latent frames after our windowing)
    # Actually T=5 here is the number of audio temporal frames at video FPS.
    # For T_latent=2, vae_scale=4, T_audio = 1 + 4*(T_latent - 1) = 5.
    h = mx.random.normal((1, 4, 2, 4, 4))  # latent: 2 frames
    t = mx.array([0.5], dtype=mx.float32)
    text = mx.random.normal((1, 1, 6, 32))
    mask = mx.ones((1, 1, 1, 6))
    # Audio: B=1, T_audio=5, W=5, S=5, C=32
    audio = mx.random.normal((1, 5, 5, 5, 32))

    out = model(h, t, text, audio, encoder_attention_mask=mask)
    mx.eval(out)
    assert out.shape == (1, 4, 2, 4, 4)


if __name__ == "__main__":
    test_audio_proj_model_output_shape_v15()
    test_single_stream_attention_basic()
    test_avatar_dit_constructs_from_v15_config()
    test_avatar_dit_forward_tiny()
    print("avatar smoke tests passed")
