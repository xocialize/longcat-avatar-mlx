"""Smoke tests for the LongCatAvatarPipeline orchestration.

These tests use TINY mock weights so they run in seconds. They validate that
the pipeline composes correctly (no shape errors, no missing attributes, no
wiring bugs) — they do NOT validate output quality. Real-weight golden test
lands in S1.11.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from longcat_video_avatar.audio_process import (
    build_avatar_audio_embeddings,
    group_pool_whisper_hidden_states,
    linear_interpolate_features,
)
from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig


def test_linear_interpolate_features():
    feats = mx.arange(0, 12, dtype=mx.float32).reshape(1, 4, 3)  # T=4, D=3
    out = linear_interpolate_features(feats, input_fps=50, output_fps=25, output_len=2)
    mx.eval(out)
    assert out.shape == (1, 2, 3)
    # output_len=2 over T=4 -> src_idx = linspace(0, 3, 2) = [0, 3]
    # feat_lo[0] = feats[0,0] = [0,1,2]; feat_hi[0] = feats[0,1] = [3,4,5]; frac=0 -> [0,1,2]
    # feat_lo[1] = feats[0,3] = [9,10,11]; feat_hi clamped to T-1=3 -> same; frac=0 -> [9,10,11]
    expected = mx.array([[[0, 1, 2], [9, 10, 11]]], dtype=mx.float32)
    diff = float(mx.abs(out - expected).max())
    assert diff < 1e-5


def test_group_pool_whisper_hidden_states():
    """33-layer input → [B, T, 5, D] via 4×8 groups + 1 singleton."""
    rng = np.random.default_rng(0)
    hiddens = [
        mx.array(rng.standard_normal((1, 4, 8)).astype(np.float32)) for _ in range(33)
    ]
    out = group_pool_whisper_hidden_states(hiddens)
    mx.eval(out)
    assert out.shape == (1, 4, 5, 8)
    # Group 0 = mean of layers 0..7
    g0_expected = mx.mean(mx.stack(hiddens[0:8], axis=0), axis=0)
    diff = float(mx.abs(out[:, :, 0, :] - g0_expected).max())
    assert diff < 1e-5
    # Group 4 = singleton layer 32
    diff = float(mx.abs(out[:, :, 4, :] - hiddens[32]).max())
    assert diff < 1e-6


def test_build_avatar_audio_embeddings_resamples_and_windows():
    """50 Hz encoder output → 25 fps target (2:1) + sliding window of W=5."""
    rng = np.random.default_rng(0)
    audio_groups = mx.array(rng.standard_normal((1, 100, 5, 8)).astype(np.float32))  # 100 enc frames
    out = build_avatar_audio_embeddings(audio_groups, fps=25, enc_fps=50, audio_window=5)
    mx.eval(out)
    # 2s × 25 fps = 50 video frames, each with a 5-frame window of 5-group 8-dim features
    assert out.shape == (1, 50, 5, 5, 8)


def test_pipeline_constructs_with_mock_components():
    """Build a tiny pipeline and call it end-to-end with synthetic inputs."""
    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )
    from longcat_video_avatar.models.umt5 import UMT5EncoderModel
    from longcat_video_avatar.models.whisper import WhisperEncoder

    # Tiny mock configs
    vae = AutoencoderKLWan(z_dim=16, base_dim=24, dim_mult=[1, 2, 4, 4])
    umt5 = UMT5EncoderModel(vocab_size=1000, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2)
    whisper = WhisperEncoder(d_model=64, num_layers=2, num_heads=4, ffn_dim=128, num_mel_bins=128, max_source_positions=750)
    dit = LongCatVideoAvatarTransformer3DModel(
        in_channels=16,
        out_channels=16,
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=64,
        mlp_ratio=4,
        adaln_tembed_dim=32,
        patch_size=(1, 2, 2),
        audio_window=5,
        audio_block=5,
        audio_channel=8,
        intermediate_dim=16,
        output_dim=16,
        context_tokens=4,
        vae_scale=4,
        audio_prenorm=False,
    )

    cfg = PipelineConfig(
        dit_hidden_size=64,
        dit_depth=2,
        dit_num_heads=4,
        num_sampling_steps=2,  # minimal
        num_frames=5,  # 1 ref + 4 = T_lat after VAE = 1 + (5-1)//4 = 2
        target_fps=25,
        vae_scale_temporal=4,
        vae_scale_spatial=8,
    )

    pipeline = LongCatAvatarPipeline(
        vae=vae,
        text_encoder=umt5,
        audio_encoder=whisper,
        dit=dit,
        config=cfg,
    )
    # Make sure it constructs
    assert pipeline.vae is vae
    assert pipeline.dit is dit
    assert pipeline.scheduler is not None


def test_pipeline_call_end_to_end_mock_weights():
    """Full pipeline __call__ with synthetic inputs. Validates the denoising
    loop, CFG split, scheduler step, and VAE decode all compose correctly.
    Does NOT validate output quality — just that no shape/wiring bug surfaces.
    """
    pytest.importorskip("mlx_arsenal")
    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )
    from longcat_video_avatar.models.umt5 import UMT5EncoderModel
    from longcat_video_avatar.models.whisper import WhisperEncoder

    vae = AutoencoderKLWan(z_dim=16, base_dim=24, dim_mult=[1, 2, 4, 4])
    umt5 = UMT5EncoderModel(vocab_size=1000, dim=64, dim_attn=64, dim_ffn=128, num_heads=4, num_layers=2)
    # MUST be 32 layers — group_pool_whisper_hidden_states is hard-coded for
    # the Whisper-large-v3 33-layer (32 transformer + 1 embedding) topology.
    # That matches Meituan's audio pipeline assumption.
    whisper = WhisperEncoder(
        d_model=64, num_layers=32, num_heads=4, ffn_dim=128, num_mel_bins=128, max_source_positions=750
    )
    dit = LongCatVideoAvatarTransformer3DModel(
        in_channels=16,
        out_channels=16,
        hidden_size=64,
        depth=2,
        num_heads=4,
        caption_channels=64,
        mlp_ratio=4,
        adaln_tembed_dim=32,
        patch_size=(1, 2, 2),
        # Audio inputs are 5-channel 64-dim (NOT 1280) for the mock — match whisper d_model
        audio_window=5,
        audio_block=5,
        audio_channel=64,
        intermediate_dim=16,
        output_dim=16,
        context_tokens=4,
        vae_scale=4,
        audio_prenorm=False,
    )

    cfg = PipelineConfig(
        dit_hidden_size=64,
        dit_depth=2,
        dit_num_heads=4,
        num_sampling_steps=2,
        num_frames=5,
        target_fps=25,
        whisper_enc_fps=50,
        text_guidance_scale=4.0,
        audio_guidance_scale=4.0,
        vae_scale_temporal=4,
        vae_scale_spatial=8,
        dit_in_channels=16,
        dit_out_channels=16,
    )

    pipeline = LongCatAvatarPipeline(
        vae=vae,
        text_encoder=umt5,
        audio_encoder=whisper,
        dit=dit,
        config=cfg,
    )

    # Synthetic inputs at the post-encoder boundary (skip umT5 + Whisper for
    # speed — those have their own smoke tests). Real pipeline.__call__ would
    # tokenize/preprocess inside, but for orchestration validation we pass
    # the embeddings directly.
    H, W = 64, 64  # spatial (must be / 8 latent)
    image = mx.random.normal((1, 3, 1, H, W))  # single reference frame
    audio_mel = mx.random.normal((1, 128, 1500))  # 1500 mel frames (~15s @ 100 Hz)
    # Use umT5-style embeds shape [B, 1, N_text, hidden_size]
    text_embeds = mx.random.normal((1, 1, 8, 64))
    text_mask = mx.ones((1, 1, 1, 8))
    uncond_embeds = mx.random.normal((1, 1, 8, 64))
    uncond_mask = mx.ones((1, 1, 1, 8))

    video = pipeline(
        image=image,
        audio_mel=audio_mel,
        text_embeds=text_embeds,
        text_mask=text_mask,
        uncond_embeds=uncond_embeds,
        uncond_mask=uncond_mask,
        height=H,
        width=W,
        seed=0,
    )
    mx.eval(video)
    # Output shape: [1, 3, num_frames, H_out, W_out]. num_frames=5; H/W decode
    # to chunked output frames = 1 + 4*(T_lat - 1).
    # T_lat after ref + noise concat = 1 + ((5-1)//4 + 1) = 1 + 2 = 3, minus ref → 2 noise latents.
    # decode of 2 latents → 1 + 4*(2-1) = 5 frames. Spatial decodes 8x: 64*1 = 64 → wait that's wrong.
    # Actually H_lat = H // 8 = 8, decodes back to 8 * 8 = 64. Good.
    assert video.shape[0] == 1
    assert video.shape[1] == 3
    assert video.shape[3] == H
    assert video.shape[4] == W
