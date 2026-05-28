"""Keymap test for the Avatar-1.5 LongCatVideoAvatarTransformer3DModel.

No download — uses the safetensors INDEX (~150 KB) for the Avatar base_model.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx

from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import (
    LongCatVideoAvatarTransformer3DModel,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
AVATAR_DIT_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "avatar-1.5--base_model-config.json"

HF_REPO = "meituan-longcat/LongCat-Video-Avatar-1.5"
HF_AVATAR_INDEX = "base_model/diffusion_pytorch_model.safetensors.index.json"


def test_avatar_keymap_no_rename_needed():
    """Every PT key in the Avatar checkpoint maps to an MLX parameter directly."""
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_AVATAR_INDEX)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    pt_keys = sorted(weight_map.keys())

    cfg = json.loads(AVATAR_DIT_CONFIG_PATH.read_text())
    mx_model = LongCatVideoAvatarTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    unmapped = []
    for pt_k in pt_keys:
        if pt_k not in mx_keys:
            unmapped.append(pt_k)

    assert not unmapped, (
        f"{len(unmapped)} PT keys not in MLX model. First 25:\n  "
        + "\n  ".join(unmapped[:25])
        + (f"\n  ... {len(unmapped) - 25} more" if len(unmapped) > 25 else "")
    )


def test_avatar_keymap_audio_path_specific():
    """Spot-check that the Avatar-specific audio path keys exist."""
    cfg = json.loads(AVATAR_DIT_CONFIG_PATH.read_text())
    mx_model = LongCatVideoAvatarTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    expected_audio = [
        "audio_proj.proj1.weight",
        "audio_proj.proj1.bias",
        "audio_proj.proj1_vf.weight",
        "audio_proj.proj1_vf.bias",
        "audio_proj.proj2.weight",
        "audio_proj.proj2.bias",
        "audio_proj.proj3.weight",
        "audio_proj.proj3.bias",
        # audio_proj.norm (LayerNorm) — only if norm_output_audio=True; default True
        "audio_proj.norm.weight",
        "audio_proj.norm.bias",
        # Per-block audio cross-attn
        "blocks.0.audio_adaLN_modulation.1.weight",
        "blocks.0.audio_adaLN_modulation.1.bias",
        "blocks.0.pre_video_crs_attn_norm.weight",
        "blocks.0.pre_video_crs_attn_norm.bias",
        "blocks.0.audio_cross_attn.q_linear.weight",
        "blocks.0.audio_cross_attn.q_linear.bias",
        "blocks.0.audio_cross_attn.kv_linear.weight",
        "blocks.0.audio_cross_attn.kv_linear.bias",
        "blocks.0.audio_cross_attn.proj.weight",
        "blocks.0.audio_cross_attn.proj.bias",
        "blocks.0.audio_cross_attn.q_norm.weight",
        "blocks.0.audio_cross_attn.k_norm.weight",
        # Last block sanity
        "blocks.47.audio_cross_attn.q_linear.weight",
    ]
    missing = [k for k in expected_audio if k not in mx_keys]
    assert not missing, "Missing expected Avatar audio keys:\n  " + "\n  ".join(missing)
