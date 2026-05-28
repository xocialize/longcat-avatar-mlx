"""Dump intermediate values from Python-MLX umT5 for Swift diagnosis.

Used during S3.4 to narrow down where Swift-MLX vs Python-MLX divergence
starts. Outputs:

  block0_pos_bias.npy  — [1, num_heads, 16, 16], block 0's pos_embedding output
  block0_norm1_out.npy — [1, 16, dim],          post-norm1, pre-attention
  block0_out.npy       — [1, 16, dim],          after block 0
  block23_out.npy      — [1, 16, dim],          after final block
  final_out.npy        — [1, 16, dim],          after final norm (== output.npy)

Run after dump_umt5_swift_fixtures.py so the same input_ids fixture is used.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import mlx.core as mx
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        default="mlx-community/LongCat-Video-Avatar-1.5-bf16-dmd-merged",
    )
    parser.add_argument(
        "--fixtures",
        type=pathlib.Path,
        required=True,
        help="Dir containing input_ids.npy + input_mask.npy from the main dump.",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        required=True,
        help="Output dir for diag fixtures.",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import hf_hub_download

    from longcat_video_avatar.models.umt5 import UMT5EncoderModel

    cfg_path = hf_hub_download(repo_id=args.weights, filename="text_encoder/config.json")
    config = json.loads(pathlib.Path(cfg_path).read_text())
    model = UMT5EncoderModel.from_config(config)

    idx_path = hf_hub_download(
        repo_id=args.weights, filename="text_encoder/model.safetensors.index.json"
    )
    idx = json.loads(pathlib.Path(idx_path).read_text())
    shards = sorted(set(idx["weight_map"].values()))
    for shard in shards:
        shard_path = hf_hub_download(repo_id=args.weights, filename=f"text_encoder/{shard}")
        model.load_weights(shard_path, strict=False)
    mx.eval(model.parameters())

    ids = mx.array(np.load(args.fixtures / "input_ids.npy"))
    mask = mx.array(np.load(args.fixtures / "input_mask.npy"))
    L = ids.shape[1]

    def save(name: str, arr_mx):
        arr = np.asarray(arr_mx.astype(mx.float32))
        out = args.out / name
        np.save(out, arr)
        print(f"  wrote {out}  shape={arr.shape}, abs.max={np.abs(arr).max():.4g}")

    # ----- token embedding -----
    x = model.token_embedding(ids)
    save("token_embedding_out.npy", x)

    # ----- block 0 pos_bias -----
    blk0 = model.blocks[0]
    pos_bias = blk0.pos_embedding(L, L)
    save("block0_pos_bias.npy", pos_bias)

    # ----- block 0 norm1 -----
    n1 = blk0.norm1(x)
    save("block0_norm1_out.npy", n1)

    # ----- block 0 attention output -----
    attn_out = blk0.attn(n1, mask=mask, pos_bias=pos_bias)
    save("block0_attn_out.npy", attn_out)

    # ----- INSIDE attention: dump q/k/v projections, qk^T, softmax, attn@v -----
    attn_mod = blk0.attn
    b = n1.shape[0]
    n_heads = attn_mod.num_heads
    hd = attn_mod.head_dim
    q = attn_mod.q(n1).reshape(b, -1, n_heads, hd).transpose(0, 2, 1, 3)
    k = attn_mod.k(n1).reshape(b, -1, n_heads, hd).transpose(0, 2, 1, 3)
    v = attn_mod.v(n1).reshape(b, -1, n_heads, hd).transpose(0, 2, 1, 3)
    save("block0_attn_q.npy", q)
    save("block0_attn_k.npy", k)
    save("block0_attn_v.npy", v)

    qkt = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
    save("block0_attn_qkt.npy", qkt)
    biased = qkt + pos_bias.astype(mx.float32)
    save("block0_attn_qkt_biased.npy", biased)
    sm = mx.softmax(biased, axis=-1).astype(q.dtype)
    save("block0_attn_softmax.npy", sm)
    attn_v = sm @ v
    save("block0_attn_av.npy", attn_v)
    attn_out_2 = attn_mod.o(attn_v.transpose(0, 2, 1, 3).reshape(b, -1, n_heads * hd))
    save("block0_attn_out_recomputed.npy", attn_out_2)

    # ----- block 0 full -----
    x = blk0(x, mask=mask, pos_bias=None)  # umT5 ignores pos_bias arg
    save("block0_out.npy", x)

    # ----- all remaining blocks -----
    for blk in model.blocks[1:]:
        x = blk(x, mask=mask, pos_bias=None)
    save("block23_out.npy", x)

    x = model.norm(x)
    save("final_out.npy", x)


if __name__ == "__main__":
    main()

# Note: lines below are appended by inline diag for S3.4
