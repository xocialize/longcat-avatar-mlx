"""Long-form audio-driven generation for LongCat-Video-Avatar 1.5 / 1.6 on MLX.

`pipeline_mlx.py` covers one 93-frame clip. This module follows Meituan's reference implementation
(`run_demo_avatar_single_audio_to_video.py` + `pipeline_longcat_video_avatar.py`) end to end:

- 25 fps, 93-frame segments; each segment after the first continues from the last 13 frames of the
  previous one plus the reference image (video continuation, `generate_avc`), so a whole voice track
  comes out as one continuous shot;
- with the distilled model (8 steps) text and audio guidance are 1.0, i.e. a single DiT pass per
  step (Meituan forces this for `--use_distill`), instead of the 3-pass CFG at 4.0;
- the first latent IS the reference image (timestep 0, never updated by the scheduler) and the
  decoded video starts with it, as in `generate_ai2v`;
- 1.6: base DiT + DMD LoRA for the first `mix_steps` steps, then the OPSD LoRA
  (`--mix_steps 4 --mix_alpha 1.0` in Meituan's README), applied at runtime (`runtime_lora.py`);
- continuation segments reuse the conditioning keys/values (`kv_cache.py`, exact);
- blocks run through `mx.compile` with fused norms/RoPE (`fast.py`, ~1.25x; `compile=False` is exact);
- the VAE decodes one latent at a time (optionally in spatial tiles), which keeps 480p at ~47 GB
  peak and makes 720p-class resolutions fit in 64 GB.

Typical use: see `scripts/generate.py`.
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from longcat_video_avatar.audio_process import linear_interpolate_features
from longcat_video_avatar.models.autoencoder_kl_wan import AutoencoderKLWan
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from longcat_video_avatar.models.umt5 import UMT5EncoderModel
from longcat_video_avatar.models.whisper import WhisperEncoder
from longcat_video_avatar.runtime_lora import attach_lora, set_lora_multipliers

FPS = 25  # the model is trained at 25 fps; generating "at 30" breaks lip sync, interpolate instead
NUM_FRAMES = 93
NUM_COND_FRAMES = 13
NUM_TRAIN_TIMESTEPS = 1000
SHIFT = 7.0

NEGATIVE_PROMPT = (
    "Close-up, Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
    "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
    "messy background, three legs, many people in the background, walking backwards"
)

# Meituan's buckets (longcat_video/utils/bukcet_config.py), ratio = H / W. Portrait and square subset.
BUCKETS = {
    "480p": {0.82: (480, 592), 1.0: (576, 576), 1.17: (672, 576), 1.29: (704, 544), 1.35: (736, 544),
             1.5: (768, 512), 1.78: (832, 480)},
    "720p": {1.0: (960, 960), 1.22: (1056, 864), 1.31: (1088, 832), 1.35: (1120, 832), 1.5: (1152, 768),
             1.78: (1280, 720)},
}


def bucket_for(height: int, width: int, resolution: str) -> tuple[int, int]:
    ratio = height / width
    key = min(BUCKETS[resolution], key=lambda r: abs(r - ratio))
    return BUCKETS[resolution][key]


# ----------------------------------------------------------------------------- loading


def _load_sharded(module, folder: pathlib.Path, index_name: str):
    idx = json.loads((folder / index_name).read_text())
    for shard in sorted(set(idx["weight_map"].values())):
        module.load_weights(str(folder / shard), strict=False)


def load_dit(folder: pathlib.Path) -> LongCatVideoAvatarTransformer3DModel:
    cfg = json.loads((folder / "config.json").read_text())
    dit = LongCatVideoAvatarTransformer3DModel.from_config(cfg)
    _load_sharded(dit, folder, "diffusion_pytorch_model.safetensors.index.json")
    mx.eval(dit.parameters())
    return dit


def load_vae(components: pathlib.Path, fp32: bool = False) -> AutoencoderKLWan:
    """bf16 like Meituan (torch_dtype=bfloat16); fp32 only makes sense with tiled decoding."""
    cfg = json.loads((components / "vae" / "config.json").read_text())
    vae = AutoencoderKLWan.from_config(cfg)
    vae.load_weights(str(components / "vae" / "diffusion_pytorch_model.safetensors"), strict=False)
    vae.set_dtype(mx.float32 if fp32 else mx.bfloat16)
    mx.eval(vae.parameters())
    return vae


def load_whisper(components: pathlib.Path) -> WhisperEncoder:
    cfg = json.loads((components / "audio_encoder" / "config.json").read_text())
    whisper = WhisperEncoder.from_config(cfg)
    whisper.load_weights(str(components / "audio_encoder" / "model.safetensors"), strict=False)
    return whisper


# ----------------------------------------------------------------------------- text and audio


def _prompt_clean(text: str) -> str:
    import html
    import re

    try:
        import ftfy

        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


def encode_prompt(components: pathlib.Path, prompt: str, cache_dir: Optional[pathlib.Path] = None
                  ) -> tuple[mx.array, mx.array]:
    """umT5 -> ([1, 1, 512, 4096] bf16, [1, 512]). Cached on disk by prompt hash: umT5 is 11 GB and is
    freed right after encoding, so it never sits next to the DiT."""
    key = hashlib.sha1(_prompt_clean(prompt).encode()).hexdigest()[:16]
    f = cache_dir / f"t5-{key}.safetensors" if cache_dir else None
    if f is not None and f.exists():
        d = mx.load(str(f))
        return d["emb"], d["mask"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(components / "tokenizer"))
    enc = tok([_prompt_clean(prompt)], padding="max_length", max_length=512, truncation=True,
              add_special_tokens=True, return_attention_mask=True, return_tensors="np")
    cfg = json.loads((components / "text_encoder" / "config.json").read_text())
    umt5 = UMT5EncoderModel.from_config(cfg)
    _load_sharded(umt5, components / "text_encoder", "model.safetensors.index.json")
    ids, mask = mx.array(enc.input_ids), mx.array(enc.attention_mask)
    emb = umt5(ids, mask=mask).astype(mx.bfloat16)[:, None]
    mx.eval(emb)
    if f is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        mx.save_safetensors(str(f), {"emb": emb, "mask": mask})
    del umt5
    mx.clear_cache()
    return emb, mask


def audio_embedding(components: pathlib.Path, speech: np.ndarray, sr: int = 16000, fps: int = FPS,
                    loudness_norm: bool = True, whisper: Optional[WhisperEncoder] = None) -> mx.array:
    """Meituan's `get_audio_embedding_whisper`: 30 s mel chunks, 3000-frame encoder chunks, the 33 hidden
    states pooled into 5 groups, linearly resampled 50 Hz -> `fps`. Returns [T, 5, 1280].

    Meituan normalizes to -23 LUFS for 1.5 and not for 1.6 (`use_loudness_norm`)."""
    from transformers import WhisperFeatureExtractor

    mel_chunk, enc_chunk, enc_fps = 750 * 640, 3000, 50
    video_length = int(len(speech) / sr * fps)
    if loudness_norm:
        import pyloudnorm as pyln

        loud = pyln.Meter(sr).integrated_loudness(speech)
        if abs(loud) <= 100:
            speech = pyln.normalize.loudness(speech, loud, -23)
    fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, chunk_length=30, n_fft=400)
    mels = [fe(speech[i:i + mel_chunk], sampling_rate=sr, return_tensors="np").input_features
            for i in range(0, len(speech), mel_chunk)]
    mel = mx.array(np.concatenate(mels, axis=-1).astype(np.float32))
    own = whisper is None
    whisper = whisper or load_whisper(components)
    enc = []
    for i in range(0, mel.shape[-1], enc_chunk):
        hs = whisper(mel[:, :, i:i + enc_chunk], return_all_hidden_states=True)
        enc.append(mx.stack(hs, axis=2))  # [1, T, 33, D]
        mx.eval(enc[-1])
    states = mx.concatenate(enc, axis=1)[:, : video_length * 2]
    groups = [states[:, :, a:b].mean(axis=2) for a, b in ((0, 8), (8, 16), (16, 24), (24, 32))]
    groups.append(states[:, :, 32])
    feats = [linear_interpolate_features(g.astype(mx.float32), enc_fps, fps, video_length) for g in groups]
    out = mx.stack(feats, axis=2)[0]
    mx.eval(out)
    if own:
        del whisper
        mx.clear_cache()
    return out


def audio_window(full: mx.array, start: int, num_frames: int = NUM_FRAMES) -> mx.array:
    """Meituan's `center_indices`: a -2..2 window per video frame. Returns [1, F, 5, 5, D]."""
    centers = np.arange(start, start + num_frames)[:, None] + np.arange(-2, 3)[None, :]
    centers = np.clip(centers, 0, full.shape[0] - 1)
    return full[mx.array(centers)][None]


# ----------------------------------------------------------------------------- images and VAE


def load_image(path: pathlib.Path, height: int, width: int) -> mx.array:
    """Cover-resize + center crop (Meituan's resize_mode='crop'), [-1, 1], [1, 3, 1, H, W]."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    W, H = img.size
    if width / height > W / H:
        nw, nh = width, int(H * width / W)
    else:
        nh, nw = height, int(W * height / H)
    img = img.resize((nw, nh), Image.LANCZOS)
    top, left = (nh - height) // 2, (nw - width) // 2
    img = img.crop((left, top, left + width, top + height))
    arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
    return mx.array(arr.transpose(2, 0, 1)[None, :, None])


def frames_to_tensor(frames: np.ndarray) -> mx.array:
    """uint8 [F, H, W, 3] -> [1, 3, F, H, W] in [-1, 1]."""
    arr = frames.astype(np.float32) / 127.5 - 1.0
    return mx.array(arr.transpose(3, 0, 1, 2)[None])


def _vae_dtype(vae) -> mx.Dtype:
    return vae.post_quant_conv.weight.dtype


def _decode_float(vae, z: mx.array) -> np.ndarray:
    """One latent frame at a time, evaluated as it goes. `AutoencoderKLWan.decode` builds the graph of
    all latents before evaluating, which costs ~+26 GB at 480p. Returns float32 [F, H, W, 3] in [-1, 1]."""
    x = vae.post_quant_conv(z.astype(_vae_dtype(vae)))
    feat_cache = [None] * vae._count_decoder_cache_slots()
    out = []
    for i in range(x.shape[2]):
        y = mx.clip(vae.decoder(x[:, :, i:i + 1], feat_cache=feat_cache, feat_idx=[0]), -1, 1)
        mx.eval(y, [c for c in feat_cache if isinstance(c, mx.array)])
        out.append(np.asarray(y[0].astype(mx.float32)).transpose(1, 2, 3, 0))
        del y
        mx.clear_cache()
    return np.concatenate(out)


def _tile_starts(n: int, tile: int, overlap: int) -> list[int]:
    if n <= tile:
        return [0]
    return sorted(set(list(range(0, n - tile, tile - overlap)) + [n - tile]))


def vae_decode(vae, z: mx.array, tile: int = 0, overlap: int = 12) -> np.ndarray:
    """Denormalized latents [1, 16, T, h, w] -> uint8 [F, H, W, 3].

    With `tile` > 0 the latent plane is decoded in `tile` x `tile` tiles overlapping by `overlap`
    latents and blended linearly, so the decoder fits in fp32 at 720p-class sizes."""
    _, _, _, hl, wl = z.shape
    if tile <= 0 or (hl <= tile and wl <= tile):
        v = _decode_float(vae, z)
    else:
        acc = wsum = None
        o8 = overlap * 8
        for h0 in _tile_starts(hl, tile, overlap):
            for w0 in _tile_starts(wl, tile, overlap):
                th, tw = min(tile, hl - h0), min(tile, wl - w0)
                part = _decode_float(vae, z[:, :, :, h0:h0 + th, w0:w0 + tw])
                if acc is None:
                    acc = np.zeros((part.shape[0], hl * 8, wl * 8, 3), np.float32)
                    wsum = np.zeros((hl * 8, wl * 8), np.float32)
                rh, rw = np.ones(th * 8, np.float32), np.ones(tw * 8, np.float32)
                ramp = np.linspace(0, 1, o8 + 2, dtype=np.float32)[1:-1]
                if h0 > 0:
                    rh[:o8] = ramp
                if h0 + th < hl:
                    rh[-o8:] = ramp[::-1]
                if w0 > 0:
                    rw[:o8] = ramp
                if w0 + tw < wl:
                    rw[-o8:] = ramp[::-1]
                m = rh[:, None] * rw[None, :]
                acc[:, h0 * 8:(h0 + th) * 8, w0 * 8:(w0 + tw) * 8] += part * m[None, :, :, None]
                wsum[h0 * 8:(h0 + th) * 8, w0 * 8:(w0 + tw) * 8] += m
        v = acc / wsum[None, :, :, None]
    return (np.clip(v / 2 + 0.5, 0, 1) * 255).astype(np.uint8)


def vae_encode(vae, x: mx.array) -> mx.array:
    """`AutoencoderKLWan.encode`, evaluated chunk by chunk. [1, 3, F, H, W] -> raw latent mean (fp32)."""
    feat_cache = [None] * vae._count_encoder_cache_slots()
    x = x.astype(_vae_dtype(vae))
    outs = []
    for i in range(1 + (x.shape[2] - 1) // 4):
        chunk = x[:, :, :1] if i == 0 else x[:, :, 1 + 4 * (i - 1): 1 + 4 * i]
        o = vae.encoder(chunk, feat_cache=feat_cache, feat_idx=[0])
        mx.eval(o, [c for c in feat_cache if isinstance(c, mx.array)])
        outs.append(o)
    mu, _ = mx.split(vae.quant_conv(mx.concatenate(outs, axis=2)), 2, axis=1)
    mu = mu.astype(mx.float32)
    mx.eval(mu)
    return mu


# ----------------------------------------------------------------------------- sampler


def shift_sigmas(s: np.ndarray, shift: float = SHIFT) -> np.ndarray:
    return shift * s / (1 + (shift - 1) * s)


def distill_sigmas(steps: int = 8) -> np.ndarray:
    """`get_timesteps_sigmas(use_distill=True)` for avatar-v1.5/1.6, then the scheduler's shift."""
    idx = np.round(np.arange(1, steps + 1, dtype=np.float32) * (NUM_TRAIN_TIMESTEPS // steps)).astype(np.int64)
    idx = NUM_TRAIN_TIMESTEPS - idx
    full = np.linspace(0, 1, NUM_TRAIN_TIMESTEPS, dtype=np.float32)[::-1]
    return shift_sigmas(full[idx][::-1].astype(np.float32))


def base_sigmas(steps: int) -> np.ndarray:
    return shift_sigmas(np.linspace(1, 0.001, steps, dtype=np.float32))


# ----------------------------------------------------------------------------- generation


@dataclass
class ModelPaths:
    """`components`: a variant folder with vae/, text_encoder/, audio_encoder/, tokenizer/.
    `dit`: the DiT folder. `loras`: {key: path}, keys "dmd"/"opsd" are driven by the sampler,
    any other key (a user LoRA) is applied at `user_lora_strength` on every step."""

    components: pathlib.Path
    dit: pathlib.Path
    version: str = "1.6"  # "1.5" or "1.6"
    loras: dict = field(default_factory=dict)
    distilled: bool = True

    @classmethod
    def from_variant(cls, folder: pathlib.Path, version: Optional[str] = None) -> "ModelPaths":
        """A published MLX variant folder (`-bf16-dmd-merged`, or the 1.6 one with lora/)."""
        folder = pathlib.Path(folder)
        loras = {}
        if (folder / "lora" / "dmd_loraA.safetensors").exists():
            loras = {"dmd": folder / "lora" / "dmd_loraA.safetensors",
                     "opsd": folder / "lora" / "opsd_loraB.safetensors"}
            version = version or "1.6"
        elif (folder / "lora" / "dmd_lora.safetensors").exists():
            loras = {"dmd": folder / "lora" / "dmd_lora.safetensors"}
        return cls(components=folder, dit=folder / "dit", version=version or "1.5", loras=loras)


@dataclass
class Settings:
    resolution: str = "480p"
    size: Optional[tuple[int, int]] = None  # explicit (H, W) instead of the bucket
    seed: int = 42
    steps: int = 8
    text_scale: float = 1.0  # > 1 enables Meituan's 3-pass disentangled CFG (base, non-distilled use)
    audio_scale: float = 1.0  # with text_scale == 1: audio-only guidance, 2 passes; < 1 damps lip motion
    audio_steps: int = 8  # apply the audio-only guidance on the first N steps only
    ref_img_index: int = 10
    mask_frame_range: int = 0
    mix_steps: int = 4  # 1.6: steps with the DMD LoRA before switching to OPSD
    mix_alpha: float = 1.0
    lora_strength: float = 1.0  # multiplier of the DMD/OPSD LoRAs (1.5 static merge default in Meituan: 0.8)
    user_lora_strength: float = 1.0
    loudness_norm: Optional[bool] = None  # None = like Meituan (on for 1.5, off for 1.6)
    kv_cache: bool = True
    retries: int = 0  # with `segment_check`: re-roll a rejected segment with another seed


class AvatarGenerator:
    def __init__(self, paths: ModelPaths, cache_dir: Optional[pathlib.Path] = None, vae_tile: int = 0,
                 vae_overlap: int = 12, compile: bool = True):
        self.paths = paths
        self.compile = compile  # mx.compile'd blocks + mx.fast norms (fast.py); False = bit-identical to the port
        self.cache_dir = cache_dir
        self.vae_tile, self.vae_overlap = vae_tile, vae_overlap
        self.vae = load_vae(paths.components, fp32=vae_tile > 0)
        self.dit = None
        self.lora_keys: list[str] = []

    def _load_dit(self):
        if self.dit is not None:
            return
        self.dit = load_dit(self.paths.dit)
        for key, path in self.paths.loras.items():
            attach_lora(self.dit, path, key)
            self.lora_keys.append(key)
        mx.eval(self.dit.parameters())
        from longcat_video_avatar import fast

        fast.apply(self.dit, compile=self.compile, fast_norms=self.compile)

    def _set_loras_for_step(self, i: int, st: Settings):
        if not self.lora_keys:
            return
        k = st.lora_strength
        mults = {key: st.user_lora_strength for key in self.lora_keys if key not in ("dmd", "opsd")}
        if "opsd" in self.lora_keys:
            mixed = i < st.mix_steps
            mults["dmd"] = k * (st.mix_alpha if mixed else 0.0)
            mults["opsd"] = k * ((1 - st.mix_alpha) if mixed else 1.0)
        elif "dmd" in self.lora_keys:
            mults["dmd"] = k
        set_lora_multipliers(self.dit, mults)

    def _denoise(self, latents, num_cond, text, text_mask, neg, neg_mask, audio, st: Settings,
                 num_ref_latents=None, log: Callable = print):
        sig = distill_sigmas(st.steps) if self.paths.distilled else base_sigmas(st.steps)
        sig = np.append(sig, 0.0).astype(np.float32)
        cfg = st.text_scale > 1.0
        T = latents.shape[2]
        use_cache = bool(num_ref_latents) and st.kv_cache and not cfg and st.mask_frame_range == 0
        cache, cache_phase = None, None
        for i in range(st.steps):
            t0 = time.time()
            self._set_loras_for_step(i, st)
            t = float(sig[i] * NUM_TRAIN_TIMESTEPS)
            dt = float(sig[i + 1]) - float(sig[i])
            audio_guided = not cfg and st.audio_scale != 1.0 and i < st.audio_steps
            if use_cache:
                from longcat_video_avatar.kv_cache import build_cache, forward_noise

                phase = (i < st.mix_steps) if "opsd" in self.lora_keys else None
                if cache is None or phase != cache_phase:
                    cache = build_cache(self.dit, latents[:, :, :num_cond], latents.shape[3], latents.shape[4], T,
                                        st.ref_img_index, num_ref_latents)
                    cache_phase = phase
                noisy = latents[:, :, num_cond:]
                pred = forward_noise(self.dit, noisy, t, text, text_mask, audio, cache, T, st.ref_img_index,
                                     num_ref_latents)
                if audio_guided:
                    p0 = forward_noise(self.dit, noisy, t, text, text_mask, mx.zeros_like(audio), cache, T,
                                       st.ref_img_index, num_ref_latents)
                    pred = p0 + st.audio_scale * (pred - p0)
                new = noisy + dt * (-pred)
            else:
                ts = mx.full((1, T), t, dtype=mx.float32)
                ts[:, :num_cond] = 0
                kw = dict(num_cond_latents=num_cond, num_ref_latents=num_ref_latents,
                          ref_img_index=st.ref_img_index if num_ref_latents else None,
                          mask_frame_range=st.mask_frame_range if num_ref_latents else None)
                if cfg:
                    pred = self.dit(mx.concatenate([latents, latents]), mx.concatenate([ts, ts]),
                                    mx.concatenate([neg, text]), mx.concatenate([audio, audio]),
                                    encoder_attention_mask=mx.concatenate([neg_mask, text_mask]), **kw)
                    p_unc_text, p_cond = pred[:1], pred[1:]
                    p_unc = self.dit(latents, ts, neg, mx.zeros_like(audio), encoder_attention_mask=neg_mask, **kw)
                    pred = p_unc + st.text_scale * (p_cond - p_unc_text) + st.audio_scale * (p_unc_text - p_unc)
                elif audio_guided:
                    pred2 = self.dit(mx.concatenate([latents, latents]), mx.concatenate([ts, ts]),
                                     mx.concatenate([text, text]), mx.concatenate([audio, mx.zeros_like(audio)]),
                                     encoder_attention_mask=mx.concatenate([text_mask, text_mask]), **kw)
                    pred = pred2[1:] + st.audio_scale * (pred2[:1] - pred2[1:])
                else:
                    pred = self.dit(latents, ts, text, audio, encoder_attention_mask=text_mask, **kw)
                # the DiT predicts the negative velocity; only the noisy latents move
                new = latents[:, :, num_cond:] + dt * (-pred[:, :, num_cond:])
            latents = mx.concatenate([latents[:, :, :num_cond], new.astype(latents.dtype)], axis=2)
            mx.eval(latents)
            log(f"    step {i + 1}/{st.steps} {time.time() - t0:.1f}s")
        return latents

    def generate(self, audio_path: pathlib.Path, prompt: str, st: Settings, image: Optional[pathlib.Path] = None,
                 start_frames: Optional[np.ndarray] = None, duration: Optional[float] = None,
                 on_segment: Optional[Callable[[np.ndarray, int], None]] = None,
                 segment_check: Optional[Callable[[np.ndarray], tuple[bool, float]]] = None,
                 log: Callable = print) -> tuple[np.ndarray, dict]:
        """Generate the talking video for `audio_path`. Returns (uint8 frames [F, H, W, 3] at 25 fps, info).

        - `image`: reference portrait (ai2v). Without `start_frames` the first segment starts from it.
        - `start_frames`: uint8 [13, H, W, 3] of real footage at 25 fps. The first segment continues
          from them (video continuation) and they are not part of the output; `image`, if given, is the
          reference (identity) frame, otherwise the first of `start_frames`.
        - `segment_check(new_frames) -> (ok, score)`: optional per-segment quality gate; a rejected segment
          is re-generated with another seed up to `st.retries` times and the lowest score is kept.
        """
        import librosa
        from PIL import Image

        if start_frames is not None:
            H, W = start_frames.shape[1:3]
        else:
            if image is None:
                raise ValueError("need `image` or `start_frames`")
            W0, H0 = Image.open(image).size
            H, W = st.size or bucket_for(H0, W0, st.resolution)
        speech, sr = librosa.load(str(audio_path), sr=16000)
        duration = duration or len(speech) / sr
        step = NUM_FRAMES - NUM_COND_FRAMES
        if start_frames is not None:
            # silence under the 13 real frames: the voice starts on the first generated frame
            speech = np.concatenate([np.zeros(int(round(NUM_COND_FRAMES / FPS * sr)), dtype=speech.dtype), speech])
            n_seg = max(1, math.ceil(duration * FPS / step))
            gen_dur = (NUM_COND_FRAMES + n_seg * step) / FPS
        else:
            n_seg = max(1, math.ceil((duration * FPS - NUM_FRAMES) / step) + 1)
            gen_dur = (NUM_FRAMES + (n_seg - 1) * step) / FPS
        need = int(math.ceil((gen_dur + 1.0) * sr))
        if len(speech) < need:
            speech = np.concatenate([speech, np.zeros(need - len(speech), dtype=speech.dtype)])
        info = {"height": H, "width": W, "segments": n_seg, "duration": duration}
        log(f"  {H}x{W}, {n_seg} segment(s) of {NUM_FRAMES} frames")

        loud = (self.paths.version != "1.6") if st.loudness_norm is None else st.loudness_norm
        full_audio = audio_embedding(self.paths.components, speech.astype(np.float64), sr, loudness_norm=loud)
        text, text_mask = encode_prompt(self.paths.components, prompt, self.cache_dir)
        neg = neg_mask = None
        if st.text_scale > 1.0:
            neg, neg_mask = encode_prompt(self.paths.components, NEGATIVE_PROMPT, self.cache_dir)
        mx.clear_cache()
        self._load_dit()
        mx.reset_peak_memory()
        shape = (1, 16, (NUM_FRAMES - 1) // 4 + 1, H // 8, W // 8)
        t_all = time.time()
        checks = []

        def noise(seg: int, attempt: int) -> mx.array:
            return mx.random.normal(shape, key=mx.random.key(st.seed + 7919 * seg + 100003 * attempt))

        def best_of(seg: int, make, new_frames: slice):
            chosen = None
            for attempt in range(st.retries + 1):
                lat, frames = make(noise(seg, attempt))
                if segment_check is None:
                    return lat, frames
                ok, score = segment_check(frames[new_frames])
                checks.append({"segment": seg + 1, "attempt": attempt, "ok": ok, "score": score})
                if chosen is None or score < chosen[2]:
                    chosen = (lat, frames, score)
                if ok:
                    break
            return chosen[0], chosen[1]

        def decode(lat):
            mx.eval(lat)
            mx.clear_cache()
            return vae_decode(self.vae, self.vae.denormalize_latents(lat), self.vae_tile, self.vae_overlap)

        if start_frames is not None:
            ref = load_image(image, H, W) if image is not None else frames_to_tensor(start_frames[:1])
            ref_latent = self.vae.normalize_latents(vae_encode(self.vae, ref))
            frames, out, first_avc = start_frames, [], 0
        else:
            ref = self.vae.normalize_latents(vae_encode(self.vae, load_image(image, H, W)))
            audio = audio_window(full_audio, 0)

            def first(nz):
                lat = mx.concatenate([ref, nz[:, :, 1:]], axis=2)
                lat = self._denoise(lat, 1, text, text_mask, neg, neg_mask, audio, st, log=log)
                return lat, decode(lat)

            log(f"  segment 1/{n_seg}")
            latents, frames = best_of(0, first, slice(None))
            ref_latent = latents[:, :, :1]
            mx.eval(ref_latent)
            out, first_avc = [frames], 1
            if on_segment:
                on_segment(np.concatenate(out), 1)

        n_cond_lat = 1 + (NUM_COND_FRAMES - 1) // 4
        start = 0 if first_avc else -step
        for s in range(first_avc, n_seg):
            log(f"  segment {s + 1}/{n_seg}")
            start += step
            audio = audio_window(full_audio, start)
            cond_lat = self.vae.normalize_latents(vae_encode(self.vae, frames_to_tensor(frames[-NUM_COND_FRAMES:])))

            def cont(nz, cond_lat=cond_lat, audio=audio):
                lat = mx.concatenate([ref_latent, cond_lat, nz[:, :, n_cond_lat:]], axis=2)
                lat = self._denoise(lat, n_cond_lat + 1, text, text_mask, neg, neg_mask, audio, st,
                                    num_ref_latents=1, log=log)
                lat = mx.concatenate([cond_lat, lat[:, :, 1 + n_cond_lat:]], axis=2)
                return lat, decode(lat)

            _, frames = best_of(s, cont, slice(NUM_COND_FRAMES, None))
            out.append(frames[NUM_COND_FRAMES:])
            if on_segment:
                on_segment(np.concatenate(out), s + 1)

        info["checks"] = checks
        info["seconds"] = round(time.time() - t_all, 1)
        info["peak_gb"] = round(mx.get_peak_memory() / 1e9, 1)
        return np.concatenate(out)[: int(round(duration * FPS))], info
