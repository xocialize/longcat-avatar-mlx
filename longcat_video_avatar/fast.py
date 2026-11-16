"""Drop-in speedups for the Avatar DiT that keep the math, mounted on an already-loaded model with `apply(dit)`.

Measured on an M5 Max (see the PR description): generation 1.2-1.3x faster, LoRA training 1.4x faster and
7 GB lower at full size when combined with `training/train_lora.py`'s per-block engine.

- `MaskFreeTextCrossAttention`: with batch size 1 the port's block-diagonal text mask is all zeros; adding it
  changes nothing, but a masked SDPA can't use the fused kernel in the backward pass. Bit-identical forward.
- `CachedRope`: cos/sin tables computed once per grid for all 48 blocks (the port recomputes them in every
  block on every call) and a single-pass pair swap instead of slice/negate/stack. Bit-identical.
- `FastLayerNorm`, `FastRMSNorm`: fp32 norms through `mx.fast.layer_norm` / `mx.fast.rms_norm` (one kernel
  instead of ~7 ops); summation order differs, so a few values in 1e5 move by one bf16 ulp.
- `CompiledBlockFn` / `FastBlock`: the block forward through `mx.compile`, one compiled function shared by all
  48 blocks (the block's weights are arguments), traced once per input shape, constants and LoRA multipliers
  (1.6's DMD -> OPSD switch retraces once). Elementwise chains (AdaLN, RoPE, residuals, LoRA) fuse.

`apply(dit)` (no compile, no fast norms) is bit-identical to the port; `apply(dit, compile=True,
fast_norms=True)` is the fast default used by `generation.py` (PSNR ~36 dB vs. the exact path).
"""

from __future__ import annotations

import copy

import mlx.core as mx
import mlx.nn as nn

from longcat_video_avatar.models.attention import MultiHeadCrossAttention
from longcat_video_avatar.models.avatar.longcat_video_dit_avatar import LongCatAvatarSingleStreamBlock
from longcat_video_avatar.models.blocks import LayerNorm_FP32, RMSNorm_FP32
from longcat_video_avatar.models.rope_3d import RotaryPositionalEmbedding


class MaskFreeTextCrossAttention(MultiHeadCrossAttention):
    def _process_cross_attn(self, x, cond, kv_seqlen):
        B, N, C = x.shape
        if B != 1:
            return super()._process_cross_attn(x, cond, kv_seqlen)
        q = self.q_norm(self.q_linear(x).reshape(1, N, self.num_heads, self.head_dim))
        kv = self.kv_linear(cond).reshape(1, -1, 2, self.num_heads, self.head_dim)
        k, v = self.k_norm(kv[:, :, 0]), kv[:, :, 1]
        out = mx.fast.scaled_dot_product_attention(q.transpose(0, 2, 1, 3), k.transpose(0, 2, 1, 3),
                                                   v.transpose(0, 2, 1, 3), scale=self.scale)
        return self.proj(out.transpose(0, 2, 1, 3).reshape(B, N, C))


_TABLES: dict[tuple, tuple] = {}


def rope_tables(rope, grid, frame_index=None, num_ref=None):
    """[tokens, head_dim] cos and signed sin for a grid, shared by all blocks. The sin already carries the
    rotation sign: - on even slots, + on odd ones."""
    key = (rope.head_dim, rope.base, tuple(grid), frame_index, num_ref)
    if key not in _TABLES:
        f = rope._precompute_freqs(grid, frame_index, num_ref)
        tab = (mx.cos(f), mx.sin(f) * mx.array([-1.0, 1.0] * (f.shape[-1] // 2)))
        mx.eval(tab)
        _TABLES[key] = tab
    return _TABLES[key]


def rotate(x, cos, sin):
    """`x * cos + rotate_half(x) * sin` with the same products and sum (bit-identical forward and backward),
    the pair swap done in one reshape/reverse."""
    *p, d = x.shape
    x32 = x.astype(mx.float32)
    swapped = x32.reshape(*p, d // 2, 2)[..., ::-1].reshape(*p, d)
    return (x32 * cos + swapped * sin).astype(x.dtype)


class CachedRope(RotaryPositionalEmbedding):
    def __call__(self, q, k, grid_size, frame_index=None, num_ref_latents=None):
        cos, sin = rope_tables(self, grid_size, frame_index, num_ref_latents)
        return rotate(q, cos, sin), rotate(k, cos, sin)


class FastLayerNorm(LayerNorm_FP32):
    def __call__(self, x):
        w = b = None
        if self.elementwise_affine:
            w, b = self.weight.astype(mx.float32), self.bias.astype(mx.float32)
        return mx.fast.layer_norm(x.astype(mx.float32), w, b, self.eps).astype(x.dtype)


class FastRMSNorm(RMSNorm_FP32):
    """The weight is applied afterwards in the input dtype, as in the port."""

    def __call__(self, x):
        return mx.fast.rms_norm(x.astype(mx.float32), None, self.eps).astype(x.dtype) * self.weight


def clone_structure(m):
    """Same module structure and same arrays, nothing copied."""
    if isinstance(m, nn.Module):
        n = copy.copy(m)
        n._no_grad = set(m._no_grad)
        for k, v in m.items():
            n[k] = clone_structure(v)
        return n
    if isinstance(m, list):
        return [clone_structure(v) for v in m]
    if isinstance(m, dict):
        return {k: clone_structure(v) for k, v in m.items()}
    return m


def _lora_state(block):
    """What a block's forward reads besides its weights: the runtime LoRA multipliers."""
    mult = {}
    for _, m in block.named_modules():
        mult.update(getattr(m, "mult", {}))
    return tuple(sorted(mult.items()))


class CompiledBlockFn:
    """`f(block, *arrays, *constants)` through `mx.compile`, one function for all blocks: the current block's
    weights are loaded into a template block with the same structure on every call."""

    def __init__(self, block, f):
        self.template = clone_structure(block)
        if isinstance(self.template, FastBlock):
            self.template.__class__ = LongCatAvatarSingleStreamBlock
        self.f = f
        self.compiled = mx.compile(self._trace)

    def _trace(self, params, args, consts, lora_state):
        m = self.template
        m.update(params)
        mult = dict(lora_state)
        for _, lo in m.named_modules():
            for k in getattr(lo, "mult", {}):
                lo.mult[k] = mult[k]
        return self.f(m, *args, *consts)

    def __call__(self, block, *args, consts=()):
        return self.compiled(block.parameters(), list(args), consts, _lora_state(block))


class FastBlock(LongCatAvatarSingleStreamBlock):
    def __call__(self, x, y, t, y_seqlen, latent_shape, audio_hidden_states=None, **kw):
        return self._compiled(self, x, y, t, audio_hidden_states,
                              consts=(tuple(y_seqlen), tuple(latent_shape), tuple(sorted(kw.items()))))


def _port_block(m, x, y, t, a, seqlen, shape, kw):
    return LongCatAvatarSingleStreamBlock.__call__(m, x, y, t, list(seqlen), shape, audio_hidden_states=a, **dict(kw))


def compiled(dit, name, f):
    """The compiled version of block function `f` if `dit` was mounted with `apply(..., compile=True)`."""
    cache = getattr(dit, "_compiled_fns", None)
    if cache is None:
        return lambda block, *args, consts=(): f(block, *args, *consts)
    if name not in cache:
        cache[name] = CompiledBlockFn(dit.blocks[0], f)
    return cache[name]


SWAPS = {MultiHeadCrossAttention: MaskFreeTextCrossAttention, RotaryPositionalEmbedding: CachedRope}
NORM_SWAPS = {LayerNorm_FP32: FastLayerNorm, RMSNorm_FP32: FastRMSNorm}


def apply(dit, compile: bool = False, fast_norms: bool = False):
    """Mount the speedups on `dit` (also after runtime LoRAs are attached). `compile=True` compiles the block
    forward for inference; training compiles its own per-block functions (`training/train_lora.py`)."""
    swaps = {**SWAPS, **(NORM_SWAPS if fast_norms else {})}
    for _, m in dit.named_modules():
        if type(m) in swaps:
            m.__class__ = swaps[type(m)]
    if compile:
        object.__setattr__(dit, "_compiled_fns", {})  # outside the parameter tree
        fn = compiled(dit, "block", _port_block)
        for b in dit.blocks:
            b.__class__ = FastBlock
            b._compiled = fn
    return dit


def remove(dit):
    """Restore the port's classes (used by the equivalence tests)."""
    back = {v: k for k, v in {**SWAPS, **NORM_SWAPS, LongCatAvatarSingleStreamBlock: FastBlock}.items()}
    for _, m in dit.named_modules():
        if type(m) in back:
            m.__class__ = back[type(m)]
    if hasattr(dit, "_compiled_fns"):
        object.__delattr__(dit, "_compiled_fns")
    return dit
