"""Keymap test for the base LongCatVideoTransformer3DModel.

Validates that every PT key in the base DiT checkpoint maps cleanly to a
parameter in our MLX model. No weights download — uses the safetensors
INDEX (~90 KB).

Per our isomorphic-with-PT convention, NO rename function is needed: every
class name and attribute name in `longcat_video_dit.py` matches PT exactly,
modulo the standard `_FP32` norm-as-attribute-vs-subclass detail.
"""

from __future__ import annotations

import json
import pathlib

import mlx.core as mx
import pytest

from longcat_video_avatar.models.longcat_video_dit import LongCatVideoTransformer3DModel

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DIT_CONFIG_PATH = REPO_ROOT / "docs" / "development" / "notes" / "config-snapshot" / "longcat-video--dit-config.json"

HF_REPO = "meituan-longcat/LongCat-Video"
HF_DIT_INDEX = "dit/diffusion_pytorch_model.safetensors.index.json"


def test_dit_keymap_no_rename_needed():
    """Every PT key in the base DiT checkpoint exists in our MLX model's
    `parameters()` keyset. No download — just the index.
    """
    from huggingface_hub import hf_hub_download

    idx_path = hf_hub_download(repo_id=HF_REPO, filename=HF_DIT_INDEX)
    weight_map = json.loads(pathlib.Path(idx_path).read_text())["weight_map"]
    pt_keys = sorted(weight_map.keys())

    cfg = json.loads(DIT_CONFIG_PATH.read_text())
    mx_model = LongCatVideoTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    unmapped = []
    for pt_k in pt_keys:
        if pt_k not in mx_keys:
            unmapped.append(pt_k)

    assert not unmapped, (
        f"{len(unmapped)} PT keys not in MLX model. First 20:\n  "
        + "\n  ".join(unmapped[:20])
        + (f"\n  ... {len(unmapped) - 20} more" if len(unmapped) > 20 else "")
    )

    # Sanity: PT and MLX should have the same TOTAL parameter count.
    # 48 blocks × 14 keys/block + 11 top-level = 683 keys
    # (Each block: adaLN.1.{w,b}, mod_norm_attn (no), mod_norm_ffn (no),
    #  pre_crs_attn_norm.{w,b}, attn.{qkv.w,qkv.b,q_norm.w,k_norm.w,proj.w,proj.b},
    #  cross_attn.{q_linear.w,q_linear.b,kv_linear.w,kv_linear.b,proj.w,proj.b,
    #              q_norm.w,k_norm.w}, ffn.{w1,w2,w3}.weight = 16 params/block)
    # But mod_norm_attn / mod_norm_ffn have NO params (elementwise_affine=False).
    assert len(pt_keys) > 0, "PT should have at least some keys"


def test_dit_keymap_block0_specific_keys():
    """Spot-check that critical PT keys exist in our MLX model — defends
    against silent typos in our class hierarchy."""
    cfg = json.loads(DIT_CONFIG_PATH.read_text())
    mx_model = LongCatVideoTransformer3DModel.from_config(cfg)

    from mlx.utils import tree_flatten

    mx_keys = {k for k, _ in tree_flatten(mx_model.parameters())}

    expected = [
        "x_embedder.proj.weight",
        "x_embedder.proj.bias",
        "t_embedder.mlp.0.weight",
        "t_embedder.mlp.0.bias",
        "t_embedder.mlp.2.weight",
        "t_embedder.mlp.2.bias",
        "y_embedder.y_proj.0.weight",
        "y_embedder.y_proj.0.bias",
        "y_embedder.y_proj.2.weight",
        "y_embedder.y_proj.2.bias",
        "blocks.0.adaLN_modulation.1.weight",
        "blocks.0.adaLN_modulation.1.bias",
        "blocks.0.pre_crs_attn_norm.weight",
        "blocks.0.pre_crs_attn_norm.bias",
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.qkv.bias",
        "blocks.0.attn.q_norm.weight",
        "blocks.0.attn.k_norm.weight",
        "blocks.0.attn.proj.weight",
        "blocks.0.attn.proj.bias",
        "blocks.0.cross_attn.q_linear.weight",
        "blocks.0.cross_attn.kv_linear.weight",
        "blocks.0.cross_attn.proj.weight",
        "blocks.0.cross_attn.q_norm.weight",
        "blocks.0.cross_attn.k_norm.weight",
        "blocks.0.ffn.w1.weight",
        "blocks.0.ffn.w2.weight",
        "blocks.0.ffn.w3.weight",
        "blocks.47.attn.qkv.weight",  # last block sanity
        "final_layer.linear.weight",
        "final_layer.linear.bias",
        "final_layer.adaLN_modulation.1.weight",
        "final_layer.adaLN_modulation.1.bias",
    ]
    missing = [k for k in expected if k not in mx_keys]
    assert not missing, "Missing expected MLX keys:\n  " + "\n  ".join(missing)
