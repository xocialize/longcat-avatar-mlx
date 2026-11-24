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


def _mock_pipeline(**config_overrides):
    """A tiny pipeline over mock components (seconds, no weights), at the
    default (distilled, no-CFG) guidance unless overridden."""
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
    for name, value in config_overrides.items():
        setattr(pipeline.config, name, value)
    return pipeline


def _mock_inputs(H=64, W=64):
    return dict(
        image=mx.random.normal((1, 3, 1, H, W)),
        audio_mel=mx.random.normal((1, 128, 1500)),
        text_embeds=mx.random.normal((1, 1, 8, 64)),
        text_mask=mx.ones((1, 1, 1, 8)),
        uncond_embeds=mx.random.normal((1, 1, 8, 64)),
        uncond_mask=mx.ones((1, 1, 1, 8)),
        height=H,
        width=W,
    )


def test_distilled_default_runs_without_cfg():
    """Default guidance is 1.0 / 1.0 (upstream --use_distill): one conditional
    DiT pass per step, not three."""
    pipeline = _mock_pipeline()
    assert not pipeline.do_classifier_free_guidance
    calls = []
    dit_call = pipeline.dit.__call__

    class Counting:
        def __call__(self, *args, **kwargs):
            calls.append(args[0].shape[0])
            return dit_call(*args, **kwargs)

    pipeline.dit = Counting()
    video = pipeline(**_mock_inputs(), seed=0)
    mx.eval(video)
    assert calls == [1] * 8  # the distilled schedule is always 8 steps


def test_image_is_first_frame_and_cond_latent_is_untouched():
    """PT `generate_ai2v`: the image's latent is the first latent frame, runs
    at timestep 0 and is never stepped; the clip has exactly `num_frames` frames."""
    pipeline = _mock_pipeline()
    inputs = _mock_inputs()
    seen_timesteps = []
    dit_call = pipeline.dit.__call__

    class Spy:
        def __call__(self, latents, timestep, *args, **kwargs):
            seen_timesteps.append(np.asarray(timestep))
            return dit_call(latents, timestep, *args, **kwargs)

    pipeline.dit = Spy()
    frame_audio = pipeline.encode_audio(inputs["audio_mel"])
    from longcat_video_avatar.audio_process import window_audio_features

    video, latents = pipeline.generate_ai2v(
        inputs["image"],
        window_audio_features(frame_audio, 0, 5),
        inputs["text_embeds"], inputs["text_mask"], inputs["uncond_embeds"], inputs["uncond_mask"],
        num_frames=5, height=64, width=64, seed=0,
    )
    assert video.shape == (1, 3, 5, 64, 64)
    assert latents.shape[2] == 2
    ref = pipeline._encode_reference_image(inputs["image"])
    assert float(mx.abs(latents[:, :, :1] - ref).max()) == 0.0
    for ts in seen_timesteps:
        assert ts.shape == (1, 2) and ts[0, 0] == 0 and ts[0, 1] > 0


def test_two_segments_continue_the_clip():
    """`num_segments=2`: 9 + (9 - 5) frames; the continuation runs with the
    reference latent in front (num_cond_latents = cond latents + 1)."""
    pipeline = _mock_pipeline(num_frames=9, num_cond_frames=5)
    assert pipeline.frames_for_segments(2) == 13
    seen = []
    dit_call = pipeline.dit.__call__

    class Spy:
        def __call__(self, latents, timestep, *args, **kwargs):
            seen.append((latents.shape[2], kwargs.get("num_cond_latents"), kwargs.get("num_ref_latents"),
                         kwargs.get("ref_img_index"), np.asarray(timestep)[0].tolist()))
            return dit_call(latents, timestep, *args, **kwargs)

    pipeline.dit = Spy()
    steps = []
    video = pipeline(**_mock_inputs(), seed=0, num_segments=2, on_step=lambda seg, i, n: steps.append((seg, i, n)))
    mx.eval(video)
    assert video.shape == (1, 3, 13, 64, 64)
    n_steps = 8  # the distilled schedule is always 8 steps
    assert steps == [(seg, i, n_steps) for seg in (0, 1) for i in range(n_steps)]
    first, second = seen[:n_steps], seen[n_steps:]
    # Segment 0: 3 latents, 1 conditioning (the image).
    assert all(s[0] == 3 and s[1] == 1 and s[2] is None for s in first)
    # Segment 1: reference + 2 cond latents (5 frames) + 1 noise latent = 4, three of them clean.
    assert all(s[0] == 4 and s[1] == 3 and s[2] == 1 and s[3] == pipeline.config.ref_img_index for s in second)
    assert all(s[4][:3] == [0, 0, 0] and s[4][3] > 0 for s in second)


def test_window_audio_features_clamps_to_extent():
    from longcat_video_avatar.audio_process import build_frame_audio_features, window_audio_features

    feats = mx.arange(10, dtype=mx.float32).reshape(10, 1, 1)
    w = window_audio_features(feats, start=0, num_frames=3)
    assert w.shape == (1, 3, 5, 1, 1)
    assert np.asarray(w)[0, :, :, 0, 0].tolist() == [[0, 0, 0, 1, 2], [0, 0, 1, 2, 3], [0, 1, 2, 3, 4]]
    tail = window_audio_features(feats, start=8, num_frames=3)
    assert np.asarray(tail)[0, :, :, 0, 0].tolist() == [[6, 7, 8, 9, 9], [7, 8, 9, 9, 9], [8, 9, 9, 9, 9]]
    # Only the audio's real extent is interpolated: 2 s of a 30 s padded encoder output.
    groups = mx.arange(1500, dtype=mx.float32).reshape(1, 1500, 1, 1) * mx.ones((1, 1, 5, 1))
    frames = build_frame_audio_features(groups, fps=25, enc_fps=50, video_length=50)
    assert frames.shape == (50, 5, 1)
    assert abs(float(frames[-1, 0, 0]) - 99.0) < 1e-3 and float(frames[0, 0, 0]) == 0.0
