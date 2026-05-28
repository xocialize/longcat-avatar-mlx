"""Smoke test: construct AutoencoderKLWan from Meituan's vae/config.json
and run a tiny encode+decode round trip with random weights.

This does NOT validate numerical correctness — it only catches shape /
config wiring errors. Numerical parity is checked in `tests/parity/`.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
VAE_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--vae-config.json"


def _load_vae_config() -> dict:
    return json.loads(VAE_CONFIG_PATH.read_text())


def test_vae_constructs_from_meituan_config():
    cfg = _load_vae_config()
    vae = AutoencoderKLWan.from_config(cfg, encoder=True)

    # Sanity: latents_mean / latents_std propagate
    assert vae.mean.shape == (16,)
    assert vae.std.shape == (16,)
    assert abs(float(vae.mean[0]) - cfg["latents_mean"][0]) < 1e-6
    assert abs(float(vae.std[0]) - cfg["latents_std"][0]) < 1e-6


def test_vae_decode_shapes():
    """Random latent → video. Confirms decoder wiring + spatial/temporal scales."""
    cfg = _load_vae_config()
    vae = AutoencoderKLWan.from_config(cfg, encoder=False)

    # Tiny input: 1 batch, 16 latent channels, 3 latent frames, 16×16 latent spatial
    z = mx.random.normal((1, 16, 3, 16, 16))
    video = vae.decode(z)
    mx.eval(video)

    b, c, t, h, w = video.shape
    assert b == 1
    assert c == 3, f"decoder must output 3-channel video, got {c}"
    # Chunked decode (per-latent-frame, matches PT _decode): 1 + 4*(N-1) video
    # frames from N latents. The "Rep" sentinel in upsample3d skips the temporal
    # double on the first call.
    expected_t = 1 + 4 * (3 - 1)  # 9
    assert t == expected_t, f"expected {expected_t} frames out of 3 latent frames, got {t}"
    # 8× spatial upsample (3 spatial upsamples × 2)
    assert h == 16 * 8 and w == 16 * 8, f"expected 128×128, got {h}×{w}"
    # Output is clipped to [-1, 1]
    arr = mx.array(video)
    assert float(arr.min()) >= -1.0 - 1e-5
    assert float(arr.max()) <= 1.0 + 1e-5


def test_vae_encode_shapes():
    """Random video → latent. Confirms encoder wiring + spatial/temporal scales."""
    cfg = _load_vae_config()
    vae = AutoencoderKLWan.from_config(cfg, encoder=True)

    # 1 batch, 3 RGB channels, 9 frames (= 1 + 2*4), 32×32 spatial
    video = mx.random.uniform(shape=(1, 3, 9, 32, 32), low=-1.0, high=1.0)
    z = vae.encode(video)
    mx.eval(z)

    b, c, t, h, w = z.shape
    assert b == 1
    assert c == 16, f"encoder must output 16 latent channels, got {c}"
    # 4× temporal compression: 9 frames → 1 (first) + 2 (chunks of 4) = 3 latent frames
    assert t == 3, f"expected 3 latent frames out of 9, got {t}"
    # 8× spatial downsample
    assert h == 32 // 8 and w == 32 // 8, f"expected 4×4, got {h}×{w}"


if __name__ == "__main__":
    test_vae_constructs_from_meituan_config()
    print("test_vae_constructs_from_meituan_config: PASS")
    test_vae_decode_shapes()
    print("test_vae_decode_shapes: PASS")
    test_vae_encode_shapes()
    print("test_vae_encode_shapes: PASS")
