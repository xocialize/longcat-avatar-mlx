"""Talking-head video of any length from a portrait (or a few frames of real footage) and a voice track.

    # LongCat-Video-Avatar 1.6 (base DiT + DMD/OPSD LoRAs), 480p bucket
    python scripts/generate.py --weights ./weights/LongCat-Video-Avatar-1.6-bf16 \
        --image portrait.png --audio voice.wav --prompt "A young man is talking to the camera ..." \
        --out out.mp4

    # 1.5 with the DMD LoRA pre-merged (mlx-community bf16-dmd-merged)
    python scripts/generate.py --weights ./weights/LongCat-Video-Avatar-1.5-bf16-dmd-merged ...

Writes `out.mp4` (25 fps, H.264, with the audio) and `out.json` (settings, timings, peak memory).
Needs ffmpeg on PATH. See `longcat_video_avatar/generation.py` for what each option does.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import subprocess
import time

from longcat_video_avatar.generation import (
    FPS,
    NUM_COND_FRAMES,
    AvatarGenerator,
    ModelPaths,
    Settings,
    bucket_for,
)


def write_mp4(frames, audio: pathlib.Path, out: pathlib.Path, fps: int = FPS):
    out.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-i", str(audio), "-t", f"{len(frames) / fps:.3f}", "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    p.stdin.write(frames.tobytes())
    p.stdin.close()
    p.wait()


def read_start_frames(video: pathlib.Path, start: float, size: tuple[int, int], crop: str | None):
    import numpy as np

    H, W = size
    vf = f"fps={FPS}," + (f"crop={crop}," if crop else "") + f"scale={W}:{H}:flags=lanczos"
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", str(start), "-i", str(video), "-vf", vf,
                        "-frames:v", str(NUM_COND_FRAMES), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                       capture_output=True, check=True, stdin=subprocess.DEVNULL)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, H, W, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", type=pathlib.Path, required=True, help="an MLX variant folder")
    ap.add_argument("--image", type=pathlib.Path, help="reference portrait")
    ap.add_argument("--audio", type=pathlib.Path, required=True)
    ap.add_argument("--prompt", required=True, help="text, or a path to a .txt file")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--resolution", choices=["480p", "720p"], default="480p")
    ap.add_argument("--size", help="explicit HxW, e.g. 1120x832 (multiples of 32)")
    ap.add_argument("--duration", type=float, help="seconds to generate (default: the whole audio)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start-video", type=pathlib.Path, help="continue from 13 frames of real footage")
    ap.add_argument("--start-at", type=float, default=0.0, help="second of --start-video to take the frames from")
    ap.add_argument("--start-crop", help="ffmpeg crop w:h:x:y applied to --start-video before scaling")
    ap.add_argument("--lora", type=pathlib.Path, help="extra (user) LoRA in Meituan format, always on")
    ap.add_argument("--lora-strength", type=float, default=1.0, help="multiplier of the DMD/OPSD LoRAs")
    ap.add_argument("--user-lora-strength", type=float, default=1.0)
    ap.add_argument("--mix-steps", type=int, default=4)
    ap.add_argument("--mix-alpha", type=float, default=1.0)
    ap.add_argument("--audio-scale", type=float, default=1.0)
    ap.add_argument("--audio-steps", type=int, default=8)
    ap.add_argument("--text-scale", type=float, default=1.0)
    ap.add_argument("--ref-img-index", type=int, default=10)
    ap.add_argument("--mask-frame-range", type=int, default=0)
    ap.add_argument("--loudness-norm", choices=["on", "off"], help="-23 LUFS (default: Meituan's choice per version)")
    ap.add_argument("--vae-tile", type=int, default=0, help="fp32 VAE decoded in tiles of N latents (0 = bf16, whole)")
    ap.add_argument("--vae-overlap", type=int, default=12)
    ap.add_argument("--no-kv-cache", action="store_true")
    ap.add_argument("--no-compile", action="store_true", help="no mx.compile / fast norms: bit-identical to the port")
    ap.add_argument("--cache-dir", type=pathlib.Path, default=pathlib.Path.home() / ".cache" / "longcat-avatar-mlx")
    a = ap.parse_args()

    prompt = pathlib.Path(a.prompt).read_text().strip() if a.prompt.endswith(".txt") else a.prompt
    paths = ModelPaths.from_variant(a.weights)
    if a.lora:
        paths.loras["user"] = a.lora
    size = tuple(int(x) for x in a.size.split("x")) if a.size else None
    st = Settings(resolution=a.resolution, size=size, seed=a.seed, text_scale=a.text_scale,
                  audio_scale=a.audio_scale, audio_steps=a.audio_steps, ref_img_index=a.ref_img_index,
                  mask_frame_range=a.mask_frame_range, mix_steps=a.mix_steps, mix_alpha=a.mix_alpha,
                  lora_strength=a.lora_strength, user_lora_strength=a.user_lora_strength,
                  loudness_norm=None if a.loudness_norm is None else a.loudness_norm == "on",
                  kv_cache=not a.no_kv_cache)
    start_frames = None
    if a.start_video:
        if size is None:
            from PIL import Image

            ref = Image.open(a.image) if a.image else None
            w0, h0 = ref.size if ref else (int(a.start_crop.split(":")[0]), int(a.start_crop.split(":")[1]))
            size = bucket_for(h0, w0, a.resolution)
        start_frames = read_start_frames(a.start_video, a.start_at, size, a.start_crop)

    t0 = time.time()
    gen = AvatarGenerator(paths, cache_dir=a.cache_dir, vae_tile=a.vae_tile, vae_overlap=a.vae_overlap,
                          compile=not a.no_compile)
    partial = a.out.with_name(a.out.stem + "-partial.mp4")
    frames, info = gen.generate(a.audio, prompt, st, image=a.image, start_frames=start_frames, duration=a.duration,
                                on_segment=lambda f, n: write_mp4(f, a.audio, partial))
    write_mp4(frames, a.audio, a.out)
    partial.unlink(missing_ok=True)
    info.update(settings=dataclasses.asdict(st), weights=str(a.weights), prompt=prompt,
                total_seconds=round(time.time() - t0, 1), frames=len(frames))
    a.out.with_suffix(".json").write_text(json.dumps(info, indent=2, default=str))
    print(json.dumps({k: v for k, v in info.items() if k != "prompt"}, indent=2, default=str))


if __name__ == "__main__":
    main()
