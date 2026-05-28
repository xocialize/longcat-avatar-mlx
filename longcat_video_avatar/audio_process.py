"""Audio preprocessing for LongCat-Video-Avatar inference.

Mirrors the pipeline path in `pipeline_longcat_video_avatar.py:557-612`:
  1. wav → mel spectrogram via WhisperFeatureExtractor (deterministic)
  2. mel → Whisper encoder hidden states (33 layers, 50 Hz)
  3. group-mean-pool 33 → 5 channels (4 groups of 8 + 1 singleton)
  4. linear-interp 50 Hz → target fps (default 25)

The vocal separator step (MelBand RoFormer in PT) is deferred for v1 — we
assume the caller pre-cleans the audio. Speech loudness normalization is
inline (PT `_loudness_norm`, -23 LUFS).

Dependencies:
- `transformers.WhisperFeatureExtractor` for mel spectrogram extraction.
  This is a `[parity]` dev dep today; if you want runtime-only inference,
  vendor the FE config and reimplement the ~50-line mel transform.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import numpy as np


def linear_interpolate_features(features: mx.array, input_fps: float, output_fps: float, output_len: Optional[int] = None) -> mx.array:
    """Resample features along time axis. `features`: [B, T, D]. Returns [B, T_out, D]."""
    B, T, D = features.shape
    if output_len is None:
        output_len = int(T / input_fps * output_fps)
    if output_len == T:
        return features
    # Compute fractional source indices for each output position
    src_idx = mx.linspace(0.0, T - 1, output_len)
    # Floor + fractional weights for linear interp
    lo = mx.minimum(mx.floor(src_idx).astype(mx.int32), T - 1)
    hi = mx.minimum(lo + 1, T - 1)
    frac = (src_idx - lo).astype(features.dtype)[None, :, None]  # [1, T_out, 1]
    # Gather: [B, T_out, D] from each end
    feat_lo = features[:, lo, :]
    feat_hi = features[:, hi, :]
    return feat_lo * (1.0 - frac) + feat_hi * frac


def loudness_normalize(audio: np.ndarray, sample_rate: int = 16000, target_lufs: float = -23.0) -> np.ndarray:
    """Loudness-normalize audio to -23 LUFS. Best-effort; uses pyloudnorm if
    available, else falls back to a simple RMS normalization."""
    try:
        import pyloudnorm as pyln

        meter = pyln.Meter(sample_rate)
        loudness = meter.integrated_loudness(audio)
        return pyln.normalize.loudness(audio, loudness, target_lufs)
    except (ImportError, ValueError):
        # Fallback: target RMS proportional to -23 LUFS (approximation)
        rms = float(np.sqrt(np.mean(audio**2)) + 1e-8)
        target_rms = 10 ** ((target_lufs - 0) / 20)
        return audio * (target_rms / rms)


def group_pool_whisper_hidden_states(hidden_states: list[mx.array]) -> mx.array:
    """Apply Meituan's specific 33-layer group-mean-pool.

    Args:
        hidden_states: list of 33 MLX arrays, each [B, T_enc, D=1280].

    Returns:
        Stacked feature: [B, T_enc, 5, D] — 5 groups (4×8 + 1 singleton).
    """
    assert len(hidden_states) == 33, f"expected 33 layers, got {len(hidden_states)}"

    # Compute group means: indices [0:8], [8:16], [16:24], [24:32], and singleton [32]
    feats = []
    for start, end in [(0, 8), (8, 16), (16, 24), (24, 32)]:
        # Stack along new axis, then mean
        stacked = mx.stack(hidden_states[start:end], axis=0)  # [8, B, T, D]
        feats.append(mx.mean(stacked, axis=0))  # [B, T, D]
    feats.append(hidden_states[32])  # singleton last layer

    return mx.stack(feats, axis=2)  # [B, T, 5, D]


def whisper_encode_audio_to_groups(
    whisper_encoder,
    mel_features: mx.array,
    enc_chunk: int = 3000,
) -> mx.array:
    """Run Whisper encoder over mel features (possibly long) and return the
    group-pooled 5-channel features.

    Args:
        whisper_encoder: WhisperEncoder MLX model.
        mel_features: [B, num_mel_bins, T_mel] mel spectrogram.
        enc_chunk: chunk size in mel-frame units (3000 = 30s @ 100 Hz).

    Returns: [B, T_enc, 5, D] where T_enc = T_mel // 2 (the conv2 stride).
    """
    B, _, T_mel = mel_features.shape
    out_chunks: list[mx.array] = []
    for start in range(0, T_mel, enc_chunk):
        chunk = mel_features[:, :, start : start + enc_chunk]
        hidden_states = whisper_encoder(chunk, return_all_hidden_states=True)
        pooled = group_pool_whisper_hidden_states(hidden_states)  # [B, T_enc_chunk, 5, D]
        out_chunks.append(pooled)
    return mx.concatenate(out_chunks, axis=1) if len(out_chunks) > 1 else out_chunks[0]


def build_avatar_audio_embeddings(
    audio_groups: mx.array,
    fps: int = 25,
    enc_fps: int = 50,
    audio_window: int = 5,
) -> mx.array:
    """Convert encoder-rate group features to per-video-frame audio embeddings,
    with a sliding window of size `audio_window` over the time axis (the W=5
    dimension the DiT consumes via AudioProjModel.proj1).

    Args:
        audio_groups: [B, T_enc, 5, D] — output of group_pool_whisper_hidden_states.
        fps: target video frame rate.
        enc_fps: Whisper encoder output frame rate (50 Hz).
        audio_window: number of audio frames per video frame's window.

    Returns:
        [B, T_video, W=audio_window, S=5, D] — `audio_embs` shape the
        LongCatVideoAvatarTransformer3DModel expects.
    """
    B, T_enc, G, D = audio_groups.shape
    # 1. Resample T_enc → T_video using linear interpolation per-(group, dim)
    flat = audio_groups.reshape(B, T_enc, G * D)
    target_len = int(T_enc / enc_fps * fps)
    resampled = linear_interpolate_features(
        flat, input_fps=float(enc_fps), output_fps=float(fps), output_len=target_len
    )  # [B, T_video, G*D]
    per_frame = resampled.reshape(B, target_len, G, D)  # [B, T_video, 5, D]

    # 2. Sliding window of size `audio_window` over time, with edge replication
    #    for boundary frames. Output: [B, T_video, W, 5, D].
    half = audio_window // 2
    # Pad with edge replicas along time axis: pad `half` on each side.
    if half > 0:
        first = per_frame[:, :1]  # [B, 1, G, D]
        last = per_frame[:, -1:]
        left_pad = mx.broadcast_to(first, (B, half, G, D))
        right_pad = mx.broadcast_to(last, (B, half, G, D))
        padded = mx.concatenate([left_pad, per_frame, right_pad], axis=1)
        # padded: [B, T_video + 2*half, G, D]
    else:
        padded = per_frame

    # Build the window by gathering per-position slices.
    windows = []
    for offset in range(audio_window):
        windows.append(padded[:, offset : offset + target_len])  # [B, T_video, G, D]
    # Stack along the window axis -> [B, T_video, W, G, D]
    return mx.stack(windows, axis=2)
