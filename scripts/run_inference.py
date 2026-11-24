"""End-to-end inference smoke test using Meituan's shipped demo inputs.

Reads:
- `refs/longcat-video/assets/avatar/single/man.png` — reference portrait
- `refs/longcat-video/assets/avatar/single/man.mp3` — speech audio
- The single_example_1.json prompt
- Converted MLX weights from the recipe output

Outputs an MP4 to `scripts/output_<timestamp>.mp4` (or .npy if ffmpeg missing).

NOTE: this is the S1.11 e2e smoke. The first run primarily validates that
the full chain runs without crashing. Visual quality validation comes later.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Optional

import mlx.core as mx
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_DIR = REPO_ROOT / "refs" / "longcat-video" / "assets" / "avatar"


def load_demo_inputs():
    """Load Meituan's shipped demo: man.png + man.mp3 + scene prompt."""
    cfg_path = DEMO_DIR / "single_example_1.json"
    cfg = json.loads(cfg_path.read_text())
    image_path = REPO_ROOT / "refs" / "longcat-video" / cfg["cond_image"]
    audio_path = REPO_ROOT / "refs" / "longcat-video" / cfg["cond_audio"]["person1"]
    prompt = cfg["prompt"]
    return image_path, audio_path, prompt


# PT `ASPECT_RATIO_627` (longcat_video/utils/bukcet_config.py): the 480p buckets,
# `height / width ratio -> (height, width)`. PT picks the bucket closest to the
# reference image's ratio (`get_condition_shape`).
BUCKETS_480P = {
    0.26: (320, 1216), 0.31: (352, 1120), 0.38: (384, 1024), 0.43: (416, 960), 0.52: (448, 864),
    0.58: (480, 832), 0.67: (512, 768), 0.74: (544, 736), 0.86: (576, 672), 0.95: (608, 640),
    1.05: (640, 608), 1.17: (672, 576), 1.29: (704, 544), 1.35: (736, 544), 1.50: (768, 512),
    1.67: (800, 480), 1.73: (832, 480), 2.00: (896, 448), 2.31: (960, 416), 2.58: (992, 384),
    2.75: (1056, 384), 3.09: (1088, 352), 3.70: (1184, 320), 3.80: (1216, 320), 3.90: (1248, 320),
    4.00: (1280, 320),
}


def bucket_size(image_path: pathlib.Path) -> tuple[int, int]:
    """(height, width) of the 480p bucket closest to the image's aspect ratio."""
    from PIL import Image

    with Image.open(image_path) as img:
        ratio = img.height / img.width
    return BUCKETS_480P[min(BUCKETS_480P, key=lambda r: abs(r - ratio))]


def preprocess_image(image_path: pathlib.Path, height: int = 480, width: int = 832) -> mx.array:
    """Load, resize to cover `width x height`, centre-crop (PT resize_mode
    "crop"), normalize to [B=1, 3, T=1, H, W] in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB")
    scale = max(width / img.width, height / img.height)
    resized = img.resize((max(width, round(img.width * scale)), max(height, round(img.height * scale))), Image.BICUBIC)
    left, top = (resized.width - width) // 2, (resized.height - height) // 2
    img = resized.crop((left, top, left + width, top + height))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return mx.array(arr[None, :, None, :, :])  # [1, 3, 1, H, W]


MEL_CHUNK_SAMPLES = 750 * 640  # PT `MEL_CHUNK`: 30 s at 16 kHz per feature-extractor call


def preprocess_audio_mel(
    audio_path: pathlib.Path, seconds: Optional[float] = None, sample_rate: int = 16000
) -> tuple[mx.array, int]:
    """Load audio, cut or zero-pad it to `seconds`, and compute the Whisper mel
    in 30 s chunks (each padded and normalized on its own, as PT does).

    Returns `(mel [1, 128, 3000 * chunks], num_samples)`. No loudness
    normalization (PT skips it for Avatar 1.6) and no vocal separation: give
    the model clean speech.
    """
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required to load audio. `pip install librosa`") from e
    from transformers import WhisperFeatureExtractor

    audio, _ = librosa.load(str(audio_path), sr=sample_rate)
    if seconds is not None:
        count = int(np.ceil(seconds * sample_rate))
        audio = audio[:count] if len(audio) >= count else np.concatenate([audio, np.zeros(count - len(audio), np.float32)])
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    chunks = [
        fe(audio[i : i + MEL_CHUNK_SAMPLES], sampling_rate=sample_rate, return_tensors="np").input_features
        for i in range(0, len(audio), MEL_CHUNK_SAMPLES)
    ]
    mel = np.concatenate(chunks, axis=-1)  # [1, 128, T_mel]
    return mx.array(mel.astype(np.float32)), len(audio)


def tokenize_prompt(prompt: str, weights_dir: pathlib.Path) -> tuple[mx.array, mx.array]:
    """Tokenize a text prompt via the umT5 tokenizer. Returns (ids, mask).

    Looks for the tokenizer in `<weights_dir>/tokenizer/`.
    """
    from transformers import T5TokenizerFast

    tok = T5TokenizerFast.from_pretrained(str(weights_dir / "tokenizer"))
    enc = tok(prompt, return_tensors="np", padding="max_length", max_length=512, truncation=True)
    ids = mx.array(enc.input_ids)
    mask = mx.array(enc.attention_mask)
    return ids, mask


VARIANT_DIRNAMES: dict[str, str] = {
    "merged": "LongCat-Video-Avatar-1.5-bf16-dmd-merged",
    "base": "LongCat-Video-Avatar-1.5-bf16",
    "q4-merged": "LongCat-Video-Avatar-1.5-q4-dmd-merged",
    "q8-merged": "LongCat-Video-Avatar-1.5-q8-dmd-merged",
    # Avatar 1.6: base DiT + DMD and OPSD LoRAs switched per denoising step.
    "v1.6-bf16": "LongCat-Video-Avatar-1.6-bf16",
    "v1.6-q8": "LongCat-Video-Avatar-1.6-q8",
    "v1.6-q4": "LongCat-Video-Avatar-1.6-q4",
}


def _quantize_dit_for_load(dit, quant_cfg: dict) -> None:
    """Apply `mlx.nn.quantize` to a freshly-constructed DiT *before* loading
    quantized weights — required so `QuantizedLinear` modules are installed
    in the right places before the bit-packed `weight`/`scales`/`biases`
    tensors land via `load_weights`.

    The skip rule must match what the conversion recipe used (see
    `recipes.convert_longcat_avatar.DIT_QUANT_SKIP_PATTERNS`); we duplicate
    it inline rather than import to keep the runtime free of conversion-time
    dependencies (huggingface_hub, etc.).
    """
    import mlx.nn as nn

    skip_patterns: list[str] = quant_cfg.get(
        "skip_patterns",
        [
            "final_layer.linear",
            "t_embedder.",
            "y_embedder.",
            "adaLN_modulation.",
            "audio_adaLN_modulation.",
        ],
    )

    def predicate(path: str, module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        for pat in skip_patterns:
            if pat in path:
                return False
        return True

    nn.quantize(
        dit,
        group_size=int(quant_cfg.get("group_size", 64)),
        bits=int(quant_cfg["bits"]),
        class_predicate=predicate,
    )


def _load_mlx_safetensors(path: pathlib.Path) -> dict:
    """`mx.load` keeps bf16 (numpy-backed safetensors readers do not)."""
    return mx.load(str(path))


def build_pipeline(
    weights_dir: pathlib.Path,
    variant: str = "merged",
    mix_steps: Optional[int] = None,
    mix_alpha: Optional[float] = None,
):
    """Construct LongCatAvatarPipeline by loading each component from the
    converted MLX weights.

    Supports the 1.5 variants {"merged", "base", "q4-merged", "q8-merged"} and
    the 1.6 variants {"v1.6-bf16", "v1.6-q8", "v1.6-q4"}. A 1.6 variant attaches
    its DMD and OPSD LoRAs as runtime side-paths; `mix_steps` / `mix_alpha`
    override the schedule in its `pipeline_config.json` (default 4 / 1.0).
    Quantized variants are detected via the `quantization` block in
    `dit/config.json` and matched with `mlx.nn.quantize` *before* loading
    weights.
    """
    from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
    from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
        LongCatVideoAvatarTransformer3DModel,
    )
    from longcat_video_avatar.models.umt5 import UMT5EncoderModel
    from longcat_video_avatar.models.whisper import WhisperEncoder
    from longcat_video_avatar.pipeline_mlx import LongCatAvatarPipeline, PipelineConfig

    if variant not in VARIANT_DIRNAMES:
        raise ValueError(
            f"Unknown variant {variant!r}. Choose from {list(VARIANT_DIRNAMES)}"
        )
    variant_dir = weights_dir / VARIANT_DIRNAMES[variant]

    print(f"Loading from {variant_dir}")

    # VAE
    vae_cfg = json.loads((variant_dir / "vae" / "config.json").read_text())
    vae = AutoencoderKLWan.from_config(vae_cfg)
    vae.load_weights(str(variant_dir / "vae" / "diffusion_pytorch_model.safetensors"), strict=False)

    # umT5
    umt5_cfg = json.loads((variant_dir / "text_encoder" / "config.json").read_text())
    umt5 = UMT5EncoderModel.from_config(umt5_cfg)
    # umT5 is sharded
    umt5_idx = json.loads((variant_dir / "text_encoder" / "model.safetensors.index.json").read_text())
    for shard_name in sorted(set(umt5_idx["weight_map"].values())):
        umt5.load_weights(str(variant_dir / "text_encoder" / shard_name), strict=False)

    # Whisper
    whisper_cfg = json.loads((variant_dir / "audio_encoder" / "config.json").read_text())
    whisper = WhisperEncoder.from_config(whisper_cfg)
    whisper.load_weights(str(variant_dir / "audio_encoder" / "model.safetensors"), strict=False)

    # DiT (possibly quantized)
    dit_cfg = json.loads((variant_dir / "dit" / "config.json").read_text())
    quant_cfg = dit_cfg.get("quantization")
    dit = LongCatVideoAvatarTransformer3DModel.from_config(dit_cfg)
    if quant_cfg is not None:
        print(
            f"  DiT quantization detected: {quant_cfg['bits']}-bit, "
            f"group_size={quant_cfg.get('group_size', 64)}, "
            f"skipping {len(quant_cfg.get('skip_patterns', []))} pattern(s) "
            f"— applying nn.quantize before load_weights"
        )
        _quantize_dit_for_load(dit, quant_cfg)
    dit_idx = json.loads((variant_dir / "dit" / "diffusion_pytorch_model.safetensors.index.json").read_text())
    for shard_name in sorted(set(dit_idx["weight_map"].values())):
        dit.load_weights(str(variant_dir / "dit" / shard_name), strict=False)

    mx.eval(vae.parameters(), umt5.parameters(), whisper.parameters(), dit.parameters())

    cfg = PipelineConfig()
    pipeline = LongCatAvatarPipeline(vae=vae, text_encoder=umt5, audio_encoder=whisper, dit=dit, config=cfg)

    # For the base variant, merge the LoRA on the fly
    if variant == "base":
        from safetensors.torch import load_file as torch_load_file

        # We saved with mx.save_safetensors; reload as mx
        from safetensors import safe_open

        lora_sd: dict = {}
        with safe_open(str(variant_dir / "lora" / "dmd_lora.safetensors"), framework="numpy") as f:
            for k in f.keys():
                lora_sd[k] = mx.array(f.get_tensor(k))
        result = pipeline.merge_dmd_lora(lora_sd, multiplier=1.0)
        print(f"  LoRA merge: applied {len(result['applied'])}, unmapped {len(result['unmapped'])}")

    # Avatar 1.6: both LoRAs stay unmerged and are switched per denoising step.
    pipe_cfg_path = variant_dir / "pipeline_config.json"
    lora_cfg = (json.loads(pipe_cfg_path.read_text()) if pipe_cfg_path.exists() else {}).get("lora")
    if lora_cfg and lora_cfg.get("mode") == "runtime_mix":
        mix = pipeline.attach_lora_mix(
            _load_mlx_safetensors(variant_dir / lora_cfg["dmd"]),
            _load_mlx_safetensors(variant_dir / lora_cfg["opsd"]),
            mix_alpha=lora_cfg.get("mix_alpha", 1.0) if mix_alpha is None else mix_alpha,
            mix_steps=lora_cfg.get("mix_steps", 4) if mix_steps is None else mix_steps,
        )
        print(
            f"  LoRA mix: {len(mix.modules)} target modules, "
            f"mix_steps={mix.mix_steps}, mix_alpha={mix.mix_alpha}"
        )

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANT_DIRNAMES),
        default="merged",
    )
    parser.add_argument("--height", type=int, default=None, help="default: the 480p bucket closest to the image's aspect ratio")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=93, help="frames per segment (4k + 1)")
    parser.add_argument("--num-segments", type=int, default=1, help="1 image-to-video segment + N-1 continuations (80 new frames each)")
    parser.add_argument("--text-guidance-scale", type=float, default=None, help="default 1.0 (distilled: no CFG)")
    parser.add_argument("--audio-guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "scripts" / "output.mp4")
    parser.add_argument("--mix-steps", type=int, default=None, help="1.6 only: leading steps using mix_alpha*DMD + (1-mix_alpha)*OPSD (default 4)")
    parser.add_argument("--mix-alpha", type=float, default=None, help="1.6 only: DMD weight during the mixed steps (default 1.0)")
    parser.add_argument("--image", type=pathlib.Path, default=None, help="reference image (default: Meituan's demo)")
    parser.add_argument("--audio", type=pathlib.Path, default=None, help="speech audio (default: Meituan's demo)")
    parser.add_argument("--prompt", type=str, default=None, help="scene prompt (default: Meituan's demo)")
    args = parser.parse_args()

    print("=== LongCat-Avatar MLX e2e inference ===")
    image_path, audio_path, prompt = load_demo_inputs()
    image_path = args.image or image_path
    audio_path = args.audio or audio_path
    prompt = args.prompt or prompt
    print(f"Image:  {image_path}")
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    print("\n[1/5] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline = build_pipeline(
        args.weights, variant=args.variant, mix_steps=args.mix_steps, mix_alpha=args.mix_alpha
    )
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    if args.text_guidance_scale is not None:
        pipeline.config.text_guidance_scale = args.text_guidance_scale
    if args.audio_guidance_scale is not None:
        pipeline.config.audio_guidance_scale = args.audio_guidance_scale
    if args.height is None or args.width is None:
        args.height, args.width = bucket_size(image_path)
    total_frames = pipeline.frames_for_segments(args.num_segments, args.num_frames)
    print(f"  {args.width}x{args.height}, {args.num_segments} segment(s), {total_frames} frames "
          f"({total_frames / pipeline.config.target_fps:.2f} s)")

    print("[2/5] Preprocessing image...")
    image = preprocess_image(image_path, height=args.height, width=args.width)

    print("[3/5] Preprocessing audio...")
    # PT: the audio covers the generated duration plus one second.
    audio_mel, audio_num_samples = preprocess_audio_mel(
        audio_path, seconds=total_frames / pipeline.config.target_fps + 1.0
    )

    print("[4/5] Encoding prompt...")
    # All variants ship an identical tokenizer (verbatim umT5 copy)
    ids, mask = tokenize_prompt(prompt, args.weights / VARIANT_DIRNAMES[args.variant])
    text_hidden = pipeline.text_encoder(ids, mask=mask)
    text_embeds = text_hidden[:, None, :, :]  # [B, 1, N_text, C]
    text_mask = mask[:, None, None, :]
    # Empty prompt for uncond
    empty_ids = mx.zeros_like(ids)
    empty_mask = mx.zeros_like(mask)
    uncond_hidden = pipeline.text_encoder(empty_ids, mask=empty_mask)
    uncond_embeds = uncond_hidden[:, None, :, :]
    uncond_mask = empty_mask[:, None, None, :]

    print("[5/5] Running denoising loop (8 distilled steps per segment)...")
    step_t = [time.time()]

    def on_step(segment, step, total):
        now = time.time()
        if step > 0:
            print(f"    segment {segment + 1} step {step}/{total}: {now - step_t[0]:.1f}s", flush=True)
        step_t[0] = now

    t1 = time.time()
    video = pipeline(
        image=image,
        audio_mel=audio_mel,
        text_embeds=text_embeds,
        text_mask=text_mask,
        uncond_embeds=uncond_embeds,
        uncond_mask=uncond_mask,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        num_segments=args.num_segments,
        audio_num_samples=audio_num_samples,
        on_step=on_step,
    )
    mx.eval(video)
    elapsed = time.time() - t1
    print(f"  inference complete in {elapsed:.1f}s ({elapsed / total_frames * 1000:.0f} ms/frame), "
          f"peak memory {mx.get_peak_memory() / 1e9:.1f} GB")

    # Save as numpy first (always works), then try MP4 if ffmpeg available
    arr = (np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), arr)
    print(f"\nSaved frames to {npy_path}  (shape {arr.shape})")

    # The model's motion is 25 fps (PipelineConfig.target_fps); the driving audio
    # is muxed in, cut to the clip's length.
    fps = pipeline.config.target_fps
    try:
        import subprocess

        cmd = [
            "ffmpeg", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-video_size", f"{arr.shape[2]}x{arr.shape[1]}", "-framerate", str(fps),
            "-i", "pipe:0", "-i", str(audio_path),
            "-t", f"{arr.shape[0] / fps:.6f}",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(args.out),
        ]
        subprocess.run(cmd, input=np.ascontiguousarray(arr).tobytes(), check=True)
        print(f"Saved MP4 to {args.out}")
    except Exception as e:
        print(f"(MP4 export skipped: {e})")


if __name__ == "__main__":
    main()
