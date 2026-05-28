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


def preprocess_image(image_path: pathlib.Path, height: int = 480, width: int = 832) -> mx.array:
    """Load + resize + normalize image to [B=1, 3, T=1, H, W] in [-1, 1]."""
    from PIL import Image

    img = Image.open(image_path).convert("RGB").resize((width, height), Image.BICUBIC)
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0  # [H, W, 3] in [-1, 1]
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return mx.array(arr[None, :, None, :, :])  # [1, 3, 1, H, W]


def preprocess_audio_mel(audio_path: pathlib.Path, sample_rate: int = 16000) -> mx.array:
    """Load audio + compute Whisper mel spectrogram via transformers FE.
    Returns [1, 128, T_mel]."""
    try:
        import librosa
    except ImportError as e:
        raise ImportError("librosa is required to load audio. `pip install librosa`") from e
    from transformers import WhisperFeatureExtractor

    audio, sr = librosa.load(str(audio_path), sr=sample_rate)
    fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-large-v3")
    inputs = fe(audio, sampling_rate=sample_rate, return_tensors="np")
    mel = inputs.input_features  # [1, 128, T_mel] numpy
    return mx.array(mel.astype(np.float32))


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


def build_pipeline(weights_dir: pathlib.Path, variant: str = "merged"):
    """Construct LongCatAvatarPipeline by loading each component from the
    converted MLX weights.

    Supports `variant` in {"merged", "base", "q4-merged", "q8-merged"}.
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

    return pipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=pathlib.Path, required=True)
    parser.add_argument(
        "--variant",
        choices=("base", "merged", "q4-merged", "q8-merged"),
        default="merged",
    )
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=93)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=pathlib.Path, default=REPO_ROOT / "scripts" / "output.mp4")
    args = parser.parse_args()

    print("=== LongCat-Avatar MLX e2e inference ===")
    image_path, audio_path, prompt = load_demo_inputs()
    print(f"Image:  {image_path}")
    print(f"Audio:  {audio_path}")
    print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

    print("\n[1/5] Building pipeline (loading converted weights)...")
    t0 = time.time()
    pipeline = build_pipeline(args.weights, variant=args.variant)
    print(f"  pipeline loaded in {time.time() - t0:.1f}s")

    print("[2/5] Preprocessing image...")
    image = preprocess_image(image_path, height=args.height, width=args.width)

    print("[3/5] Preprocessing audio...")
    audio_mel = preprocess_audio_mel(audio_path)

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

    print("[5/5] Running denoising loop (8 DMD steps)...")
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
    )
    mx.eval(video)
    elapsed = time.time() - t1
    print(f"  inference complete in {elapsed:.1f}s ({elapsed / args.num_frames * 1000:.0f} ms/frame)")

    # Save as numpy first (always works), then try MP4 if ffmpeg available
    arr = (np.asarray(video).transpose(0, 2, 3, 4, 1)[0] * 127.5 + 127.5).clip(0, 255).astype(np.uint8)
    npy_path = args.out.with_suffix(".npy")
    np.save(str(npy_path), arr)
    print(f"\nSaved frames to {npy_path}  (shape {arr.shape})")

    try:
        import imageio

        writer = imageio.get_writer(str(args.out), fps=30, codec="libx264", quality=8)
        for frame in arr:
            writer.append_data(frame)
        writer.close()
        print(f"Saved MP4 to {args.out}")
    except Exception as e:
        print(f"(MP4 export skipped: {e})")


if __name__ == "__main__":
    main()
