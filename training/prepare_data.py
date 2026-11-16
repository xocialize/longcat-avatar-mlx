"""Turn short talking clips (video + audio) into LoRA training samples for LongCat-Video-Avatar.

    python -m training.prepare_data --weights weights/LongCat-Video-Avatar-1.6-bf16 \
        --clips my_clips/ --out data/person --size 640x512

For each `*.mp4|*.mov` in `--clips` (>= 2 s, one person talking to the camera, with its audio):
- 49 frames at 25 fps -> normalized VAE latents [16, 13, H/8, W/8] (the first latent is the reference frame);
- the audio -> Whisper features windowed per frame [49, 5, 5, 1280] (-23 LUFS by default, like generation);
- the caption -> umT5 [512, 4096]. Caption = `<clip>.txt` next to the clip, else `--caption`. Use the same
  wording you will use as the generation prompt.
Writes `<out>/<clip>.safetensors`. ~40 clips of ~2 s were enough for a person-specific motion/mouth LoRA.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess

import mlx.core as mx
import numpy as np

from longcat_video_avatar.generation import (
    audio_embedding,
    audio_window,
    encode_prompt,
    frames_to_tensor,
    load_vae,
    load_whisper,
    vae_encode,
)

FPS = 25


def read_frames(clip: pathlib.Path, n: int, h: int, w: int) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vf",
                        f"fps={FPS},scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h}",
                        "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True, stdin=subprocess.DEVNULL)
    f = np.frombuffer(r.stdout, np.uint8).reshape(-1, h, w, 3)
    if len(f) < n:
        raise ValueError(f"{clip.name}: {len(f)} frames at {FPS} fps, need {n}")
    return f


def read_audio(clip: pathlib.Path, seconds: float) -> np.ndarray:
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(clip), "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le",
                        "-"], capture_output=True, check=True, stdin=subprocess.DEVNULL)
    x = np.frombuffer(r.stdout, np.float32).astype(np.float64)
    need = int((seconds + 1.0) * 16000)
    return np.concatenate([x, np.zeros(max(0, need - len(x)))])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=pathlib.Path, required=True, help="MLX variant folder (vae, text/audio encoders)")
    ap.add_argument("--clips", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--caption", help="caption for clips without a .txt")
    ap.add_argument("--size", default="640x512", help="HxW, multiples of 16")
    ap.add_argument("--frames", type=int, default=49, help="4k+1 frames per sample")
    ap.add_argument("--no-loudness-norm", action="store_true")
    a = ap.parse_args()

    h, w = (int(x) for x in a.size.split("x"))
    a.out.mkdir(parents=True, exist_ok=True)
    vae = load_vae(a.weights, fp32=True)
    whisper = load_whisper(a.weights)
    cache = a.out / ".t5-cache"
    clips = sorted(p for p in a.clips.iterdir() if p.suffix.lower() in (".mp4", ".mov"))
    for clip in clips:
        f = a.out / f"{clip.stem}.safetensors"
        if f.exists():
            continue
        cap_file = clip.with_suffix(".txt")
        caption = cap_file.read_text().strip() if cap_file.exists() else a.caption
        if not caption:
            raise SystemExit(f"{clip.name}: no caption (add {cap_file.name} or --caption)")
        frames = read_frames(clip, a.frames, h, w)
        lat = vae.normalize_latents(vae_encode(vae, frames_to_tensor(frames))).astype(mx.float32)
        full = audio_embedding(a.weights, read_audio(clip, a.frames / FPS), 16000, fps=FPS,
                               loudness_norm=not a.no_loudness_norm, whisper=whisper)
        audio = audio_window(full, 0, a.frames).astype(mx.bfloat16)
        emb, mask = encode_prompt(a.weights, caption, cache)
        mx.save_safetensors(str(f), {"latents": lat[0], "audio": audio[0], "text": emb[0], "mask": mask[0]})
        print(f"{clip.stem}: latents {tuple(lat.shape[1:])}", flush=True)


if __name__ == "__main__":
    main()
