"""MLX port of UMT5-XXL text encoder for LongCat-Video.

PyTorch reference: HF transformers `UMT5EncoderModel`
(`transformers/models/umt5/modeling_umt5.py`).

This port is adapted from `Blaizzy/mlx-video`'s `mlx_video/models/wan_2/text_encoder.py`
(MIT) — already a parity-tested MLX implementation of UMT5 used by Wan 2.2.
Math is identical to HF; only parameter names differ. We adopt the compact
mlx-video names internally and remap PT checkpoint keys at load time
(see `tests/parity/test_umt5_parity.py` for the rename function).

Compared to vanilla T5, UMT5 has:
- Per-block (not shared) relative position bias → `shared_pos=False` here
- Gated GeLU FFN with `wi_0, wi_1, wo` (T5 1.1 style)
- RMSNorm (still called T5LayerNorm in transformers, but it's RMSNorm)
- No bias on Linear projections
- No attention scaling (unscaled QK^T, fp32 softmax)
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class T5LayerNorm(nn.Module):
    """RMS-based layer normalization (T5/UMT5 style)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class T5RelativeEmbedding(nn.Module):
    """T5-style relative position bias with bucketing.

    For UMT5, one of these lives per block (set `shared_pos=False` at the
    encoder level). For vanilla T5 / Wan 2.2, a single shared instance lives
    at the encoder level.
    """

    def __init__(
        self,
        num_buckets: int,
        num_heads: int,
        bidirectional: bool = True,
        max_dist: int = 128,
    ):
        super().__init__()
        self.num_buckets = num_buckets
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        self.max_dist = max_dist
        self.embedding = nn.Embedding(num_buckets, num_heads)

    def _relative_position_bucket(self, rel_pos: mx.array) -> mx.array:
        if self.bidirectional:
            num_buckets = self.num_buckets // 2
            rel_buckets = (rel_pos > 0).astype(mx.int32) * num_buckets
            rel_pos = mx.abs(rel_pos)
        else:
            num_buckets = self.num_buckets
            rel_buckets = mx.zeros_like(rel_pos, dtype=mx.int32)
            rel_pos = mx.maximum(-rel_pos, mx.zeros_like(rel_pos))

        max_exact = num_buckets // 2
        is_small = rel_pos < max_exact

        rel_pos_f = rel_pos.astype(mx.float32)
        rel_pos_large = max_exact + (
            mx.log(rel_pos_f / max_exact)
            / math.log(self.max_dist / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        rel_pos_large = mx.minimum(
            rel_pos_large,
            mx.full(rel_pos_large.shape, num_buckets - 1, dtype=mx.int32),
        )

        rel_buckets = rel_buckets + mx.where(is_small, rel_pos.astype(mx.int32), rel_pos_large)
        return rel_buckets

    def __call__(self, lq: int, lk: int) -> mx.array:
        positions_k = mx.arange(lk)[None, :]
        positions_q = mx.arange(lq)[:, None]
        rel_pos = positions_k - positions_q
        buckets = self._relative_position_bucket(rel_pos)
        embeds = self.embedding(buckets)
        embeds = embeds.transpose(2, 0, 1)[None, :, :, :]
        return embeds


class T5Attention(nn.Module):
    """T5/UMT5 self-attention. No scaling (unscaled QK^T), fp32 softmax."""

    def __init__(self, dim: int, dim_attn: int, num_heads: int):
        super().__init__()
        assert dim_attn % num_heads == 0
        self.dim = dim
        self.dim_attn = dim_attn
        self.num_heads = num_heads
        self.head_dim = dim_attn // num_heads

        self.q = nn.Linear(dim, dim_attn, bias=False)
        self.k = nn.Linear(dim, dim_attn, bias=False)
        self.v = nn.Linear(dim, dim_attn, bias=False)
        self.o = nn.Linear(dim_attn, dim, bias=False)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        b, n, c = x.shape[0], self.num_heads, self.head_dim

        q = self.q(x).reshape(b, -1, n, c)
        k = self.k(x).reshape(b, -1, n, c)
        v = self.v(x).reshape(b, -1, n, c)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        # T5: no 1/sqrt(d) scaling. Compute QK^T directly in fp32.
        attn = q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 1, 3, 2)
        if pos_bias is not None:
            attn = attn + pos_bias.astype(mx.float32)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask[:, None, None, :]
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]
            additive_mask = mx.where(mask == 0, -3.389e38, 0.0).astype(mx.float32)
            attn = attn + additive_mask
        attn = mx.softmax(attn, axis=-1).astype(q.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(b, -1, n * c)
        return self.o(out)


class T5FeedForward(nn.Module):
    """Gated GeLU FFN (T5 1.1 / UMT5): wi_0 (gate_proj) gates wi_1 (fc1) → wo (fc2)."""

    def __init__(self, dim: int, dim_ffn: int):
        super().__init__()
        self.dim = dim
        self.dim_ffn = dim_ffn
        self.gate_proj = nn.Linear(dim, dim_ffn, bias=False)
        self.gate_act = nn.GELU(approx="tanh")
        self.fc1 = nn.Linear(dim, dim_ffn, bias=False)
        self.fc2 = nn.Linear(dim_ffn, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.fc1(x) * self.gate_act(self.gate_proj(x)))


class T5SelfAttentionBlock(nn.Module):
    """One UMT5 encoder block: pre-LN self-attn + pre-LN gated FFN."""

    def __init__(
        self,
        dim: int,
        dim_attn: int,
        dim_ffn: int,
        num_heads: int,
        num_buckets: int,
        shared_pos: bool = True,
    ):
        super().__init__()
        self.shared_pos = shared_pos
        self.norm1 = T5LayerNorm(dim)
        self.attn = T5Attention(dim, dim_attn, num_heads)
        self.norm2 = T5LayerNorm(dim)
        self.ffn = T5FeedForward(dim, dim_ffn)
        self.pos_embedding = (
            None if shared_pos else T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True)
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        pos_bias: mx.array | None = None,
    ) -> mx.array:
        e = pos_bias if self.shared_pos else self.pos_embedding(x.shape[1], x.shape[1])
        x = x + self.attn(self.norm1(x), mask=mask, pos_bias=e)
        x = x + self.ffn(self.norm2(x))
        return x


class UMT5EncoderModel(nn.Module):
    """UMT5-XXL encoder. Class name matches transformers' for clarity."""

    def __init__(
        self,
        vocab_size: int = 256384,
        dim: int = 4096,
        dim_attn: int = 4096,
        dim_ffn: int = 10240,
        num_heads: int = 64,
        num_layers: int = 24,
        num_buckets: int = 32,
        shared_pos: bool = False,  # UMT5 default — per-block relative bias
    ):
        super().__init__()
        self.dim = dim
        self.shared_pos = shared_pos

        self.token_embedding = nn.Embedding(vocab_size, dim)
        # Top-level shared bias only when `shared_pos=True` (vanilla T5).
        self.pos_embedding = (
            T5RelativeEmbedding(num_buckets, num_heads, bidirectional=True) if shared_pos else None
        )
        self.blocks = [
            T5SelfAttentionBlock(dim, dim_attn, dim_ffn, num_heads, num_buckets, shared_pos)
            for _ in range(num_layers)
        ]
        self.norm = T5LayerNorm(dim)

    @classmethod
    def from_config(cls, config: dict) -> "UMT5EncoderModel":
        """Construct from a HF-style umT5 config.json dict.

        Recognized keys: `vocab_size`, `d_model`, `d_kv`*`num_heads` (→ `dim_attn`),
        `d_ff`, `num_heads`, `num_layers`, `relative_attention_num_buckets`.
        """
        num_heads = config.get("num_heads", 64)
        d_kv = config.get("d_kv", 64)
        return cls(
            vocab_size=config.get("vocab_size", 256384),
            dim=config.get("d_model", 4096),
            dim_attn=num_heads * d_kv,
            dim_ffn=config.get("d_ff", 10240),
            num_heads=num_heads,
            num_layers=config.get("num_layers", 24),
            num_buckets=config.get("relative_attention_num_buckets", 32),
            shared_pos=False,  # UMT5 hard-coded — every block has its own bias
        )

    def __call__(self, ids: mx.array, mask: mx.array | None = None) -> mx.array:
        """Args: ids `[B, L]`, mask `[B, L]` (1=keep, 0=pad). Returns `[B, L, dim]`."""
        x = self.token_embedding(ids)
        e = self.pos_embedding(x.shape[1], x.shape[1]) if self.pos_embedding is not None else None
        for block in self.blocks:
            x = block(x, mask=mask, pos_bias=e)
        x = self.norm(x)
        return x


def rename_pt_to_mx(pt_key: str) -> str:
    """Rename a single HF/transformers UMT5 checkpoint key to our compact MLX
    hierarchy. Public so the conversion recipe doesn't need to import from
    tests. Behavior mirrors tests/parity/test_umt5_parity.rename_pt_to_mx.
    """
    if pt_key == "shared.weight":
        return "token_embedding.weight"
    if pt_key == "encoder.final_layer_norm.weight":
        return "norm.weight"
    if pt_key.startswith("encoder.block."):
        rest = pt_key[len("encoder.block.") :]
        b_str, rest = rest.split(".", 1)
        if rest.startswith("layer.0."):
            sub = rest[len("layer.0.") :]
            if sub.startswith("SelfAttention.relative_attention_bias."):
                tail = sub[len("SelfAttention.relative_attention_bias.") :]
                return f"blocks.{b_str}.pos_embedding.embedding.{tail}"
            if sub.startswith("SelfAttention."):
                tail = sub[len("SelfAttention.") :]
                return f"blocks.{b_str}.attn.{tail}"
            if sub.startswith("layer_norm."):
                tail = sub[len("layer_norm.") :]
                return f"blocks.{b_str}.norm1.{tail}"
        elif rest.startswith("layer.1."):
            sub = rest[len("layer.1.") :]
            if sub.startswith("DenseReluDense.wi_0."):
                tail = sub[len("DenseReluDense.wi_0.") :]
                return f"blocks.{b_str}.ffn.gate_proj.{tail}"
            if sub.startswith("DenseReluDense.wi_1."):
                tail = sub[len("DenseReluDense.wi_1.") :]
                return f"blocks.{b_str}.ffn.fc1.{tail}"
            if sub.startswith("DenseReluDense.wo."):
                tail = sub[len("DenseReluDense.wo.") :]
                return f"blocks.{b_str}.ffn.fc2.{tail}"
            if sub.startswith("layer_norm."):
                tail = sub[len("layer_norm.") :]
                return f"blocks.{b_str}.norm2.{tail}"
    return pt_key
