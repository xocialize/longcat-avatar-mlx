"""LoRA training for LongCat-Video-Avatar on Apple Silicon (MLX), e.g. to adapt mouth/motion style to one person.

    python -m training.train_lora --dit weights/LongCat-Video-Avatar-1.6-bf16/dit --data data/person \
        --steps 250 --save-every 125 --out loras/person

    # resumable in chunks (e.g. to share the GPU with other jobs): run the same command repeatedly
    python -m training.train_lora ... --chunk 10

Objective: Meituan's image+audio -> video task with flow matching. The first latent is the clean reference
frame (timestep 0), the other latents are noised to sigma; the DiT predicts x0 - eps (the sampler negates it
into a velocity). sigma ~ logit-normal shifted by 7 (the scheduler's shift), so mostly 0.5-1, where the 8
distilled steps sample. Train on the base DiT (without the DMD/OPSD LoRAs); at inference the user LoRA is
attached next to them (`scripts/generate.py --lora`).

LoRA (default rank 32, alpha 32) on self-attention (qkv, proj) and audio cross-attention (q, kv, proj):
that's where lip sync, mouth amplitude and motion live; identity comes from the reference frame anyway.
Checkpoints are written in Meituan's LoRA key format, so `runtime_lora.attach_lora` loads them.

Memory/speed on an M5 Max 64 GB, 49 frames at 640x512 (16.6k tokens): ~49.6 GB peak, ~65 s/step with the
default engine (blocks recomputed in the backward one at a time, each block a single `mx.compile`d function
shared by all 48 blocks). `--reference` runs the straightforward `nn.value_and_grad` + `mx.checkpoint`
version (~91 s/step, 56.5 GB), which the engine matches bit-for-bit in its exact mode (tests).
Randomness depends only on the step number, so chunked/resumed runs reproduce an uninterrupted run exactly.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_unflatten

from longcat_video_avatar import fast
from longcat_video_avatar.generation import load_dit

TARGETS = ["attn.qkv", "attn.proj", "audio_cross_attn.q_linear", "audio_cross_attn.kv_linear",
           "audio_cross_attn.proj"]


class TrainableLoRA(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        out_dim, in_dim = base.weight.shape
        self.base = base
        self.scale = alpha / rank
        bound = 1 / math.sqrt(in_dim)
        self.lora_down = mx.random.uniform(-bound, bound, (rank, in_dim)).astype(mx.float32)
        self.lora_up = mx.zeros((out_dim, rank), mx.float32)

    def __call__(self, x):
        y = self.base(x)
        d = (x.astype(mx.float32) @ self.lora_down.T) @ self.lora_up.T
        return y + (d * self.scale).astype(y.dtype)


class CheckpointedBlock(nn.Module):
    """Recomputes the block in the backward pass (reference step). The engine calls `self.block` directly; the
    wrapper stays in the tree so parameter names (`blocks.N.block.*`) match saved training states."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def __call__(self, x, y, t, y_seqlen, shape, audio_hidden_states=None, **kw):
        block = self.block

        def f(params, x, y, t, a):
            block.update(params)
            return block(x, y, t, y_seqlen, shape, audio_hidden_states=a, **kw)

        return mx.checkpoint(f)(block.trainable_parameters(), x, y, t, audio_hidden_states)


def add_loras(dit, rank: int, alpha: float):
    dit.freeze()
    for b in dit.blocks:
        for path in TARGETS:
            parent_path, name = path.rsplit(".", 1)
            parent = b
            for p in parent_path.split("."):
                parent = getattr(parent, p)
            setattr(parent, name, TrainableLoRA(getattr(parent, name), rank, alpha))
    dit.blocks = [CheckpointedBlock(b) for b in dit.blocks]
    for _, m in dit.named_modules():
        if isinstance(m, TrainableLoRA):
            m.unfreeze(keys=["lora_down", "lora_up"], recurse=False)
    return sum(v.size for _, v in tree_flatten(dit.trainable_parameters()))


def save_lora(dit, f: pathlib.Path, scale: float):
    """Meituan's LoRA format: `lora___lorahyphen___blocks___lorahyphen___N___...<path>.lora_{down,up}.weight`."""
    out = {}
    for i, b in enumerate(dit.blocks):
        for path in TARGETS:
            m = b.block
            for p in path.split("."):
                m = getattr(m, p)
            name = "lora___lorahyphen___" + "___lorahyphen___".join(["blocks", str(i)] + path.split("."))
            out[f"{name}.lora_down.weight"] = m.lora_down.astype(mx.bfloat16)
            out[f"{name}.lora_up.weight"] = m.lora_up.astype(mx.bfloat16)
            out[f"{name}.alpha_scale"] = mx.array(scale, mx.float32)
    mx.eval(out)
    f.parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(f), out)


def noisy_input(lat, sigma, eps):
    """DiT input: the clean reference latent, the others at sigma, with their timesteps."""
    x0 = lat[None]
    T = x0.shape[2]
    xt = mx.concatenate([x0[:, :, :1], (1 - sigma) * x0[:, :, 1:] + sigma * eps[:, :, 1:]], axis=2)
    ts = mx.concatenate([mx.zeros((1, 1)), mx.full((1, T - 1), sigma * 1000.0)], axis=1)
    return x0, xt, ts


def loss_from_output(out, x0, eps):
    return mx.mean((out[:, :, 1:] - (x0 - eps)[:, :, 1:]) ** 2)


def reference_step(dit):
    """One graph, `mx.checkpoint` on every block. Returns f(latents, ...) -> (loss, grads)."""

    def loss_fn(model, lat, audio, text, mask, sigma, eps):
        x0, xt, ts = noisy_input(lat, sigma, eps)
        out = model(xt, ts, text[None], audio[None], encoder_attention_mask=mask[None], num_cond_latents=1)
        return loss_from_output(out, x0, eps)

    loss_and_grad = nn.value_and_grad(dit, loss_fn)

    def step(lat, audio, text, mask, sigma, eps):
        loss, grads = loss_and_grad(dit, lat, audio, text, mask, sigma, eps)
        mx.eval(loss, grads)
        return loss, grads

    return step


class Engine:
    """The training step by hand, one block at a time.

    Forward: blocks in order, keeping each block's input. Backward: last to first, each block recomputes its
    forward and yields the gradients of its LoRAs and of its input. Same math as `reference_step`, but each
    block is one `mx.compile`d function (one for all 48: weights are arguments) evaluated on its own, so a
    block's memory is free before the next one starts.
    """

    def __init__(self, dit, compile: bool = True, exact: bool = False):
        self.dit = dit
        self.exact = exact
        self.blocks = [b.block for b in dit.blocks]
        self.template = fast.clone_structure(self.blocks[0])
        self.names = [k for k, _ in tree_flatten(self.blocks[0].trainable_parameters())]
        self._fwd = mx.compile(self._f_forward) if compile else self._f_forward
        self._bwd = mx.compile(self._f_backward) if compile else self._f_backward

    def _block(self, x, y, t, a, grid):
        shape = (t.shape[1], *grid)
        return self.template(x, y, t, [y.shape[1]], shape, audio_hidden_states=a, num_cond_latents=1)

    def _f_forward(self, params, x, y, t, a, grid):
        self.template.update(params)
        if self.exact:
            # inside a derivative MLX computes masked attention differently; taking the forward through a vjp
            # makes it bit-identical to the reference step (tests only)
            return mx.vjp(lambda x: self._block(x, y, t, a, grid), [x], [mx.zeros(x.shape, x.dtype)])[0][0]
        return self._block(x, y, t, a, grid)

    def _f_backward(self, params, x, y, t, a, cot, grid):
        flat = dict(tree_flatten(params))

        def f(*args):
            self.template.update(params)
            self.template.update(tree_unflatten(list(zip(self.names, args[:-1]))))
            return self._block(args[-1], y, t, a, grid)

        _, g = mx.vjp(f, [flat[n] for n in self.names] + [x], [cot])
        return g

    def _inputs(self, xt, ts, text, mask, audio):
        """The start of `LongCatVideoAvatarTransformer3DModel.__call__`, up to the blocks."""
        dit = self.dit
        B, _, T, H, W = xt.shape
        grid = (H // dit.patch_size[1], W // dit.patch_size[2])
        dtype = dit.x_embedder.proj.weight.dtype
        x = dit.x_embedder(xt.astype(dtype))
        t = dit.t_embedder(ts.astype(dtype).astype(mx.float32).flatten(), dtype=mx.float32).reshape(B, T, -1)
        a = dit._prepare_audio_hidden_states(audio.astype(dtype), 0, T)
        a = a.reshape(B * T, a.shape[2], a.shape[3])
        # text packed like the DiT does with B = 1 and text_tokens_zero_pad: padding tokens zeroed, all 512 kept
        assert B == 1 and dit.text_tokens_zero_pad
        y = (dit.y_embedder(text.astype(dtype)) * mask[:, None, :, None]).squeeze(1)
        return x, t, y, a, grid

    def __call__(self, lat, audio, text, mask, sigma, eps):
        dit = self.dit
        x0, xt, ts = noisy_input(lat, sigma, eps)
        x, t, y, a, grid = self._inputs(xt, ts, text[None], mask[None], audio[None])
        inputs = []
        for b in self.blocks:
            inputs.append(x)
            x = self._fwd(b.parameters(), x, y, t, a, grid)
            mx.eval(x)

        def head(x):
            T = t.shape[1]
            out = dit._unpatchify(dit.final_layer(x, t, (T, *grid)), T, *grid).astype(mx.float32)
            return loss_from_output(out, x0, eps)

        loss, cot = mx.value_and_grad(head)(x)
        mx.eval(loss, cot)
        del x
        grads = [None] * len(self.blocks)
        for i in reversed(range(len(self.blocks))):
            g = self._bwd(self.blocks[i].parameters(), inputs.pop(), y, t, a, cot, grid)
            mx.eval(g if i else g[:-1])  # the first block's input gradient is not needed
            grads[i], cot = g[:-1], g[-1]
        return loss, {"blocks": [{"block": tree_unflatten(list(zip(self.names, g)))} for g in grads]}


def sample_for_step(data, step: int, seed: int, shift: float, latents: int):
    """Clip, sigma and noise depend only on the step number, so resumed runs replay the same sequence."""
    epoch, k = divmod(step - 1, len(data))
    order = list(range(len(data)))
    random.Random(seed + epoch).shuffle(order)
    d = dict(data[order[k]])
    d["latents"] = d["latents"][:, :latents]
    d["audio"] = d["audio"][:1 + 4 * (latents - 1)]
    u = random.Random(seed * 1000 + step).gauss(0, 1)
    s = 1 / (1 + math.exp(-u))
    sigma = shift * s / (1 + (shift - 1) * s)
    eps = mx.random.normal((1,) + d["latents"].shape, key=mx.random.key(seed * 100000 + step))
    return d, sigma, eps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dit", type=pathlib.Path, required=True, help="base DiT folder (no DMD/OPSD merged)")
    ap.add_argument("--data", type=pathlib.Path, required=True, help="output of training.prepare_data")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--name", default="lora")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=125)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--shift", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latents", type=int, default=13, help="latents per sample (13 = 49 frames)")
    ap.add_argument("--chunk", type=int, default=0, help="run at most N steps, save state and exit (0 = all)")
    ap.add_argument("--reference", action="store_true", help="plain value_and_grad + mx.checkpoint step")
    a = ap.parse_args()

    mx.random.seed(a.seed)
    data = [mx.load(str(c)) for c in sorted(a.data.glob("*.safetensors"))]
    print(f"{len(data)} samples", flush=True)
    dit = load_dit(a.dit)
    n = add_loras(dit, a.rank, a.alpha)
    print(f"trainable parameters: {n / 1e6:.1f} M", flush=True)
    opt = optim.Adam(learning_rate=a.lr)
    step_fn = reference_step(dit) if a.reference else Engine(fast.apply(dit, fast_norms=True))

    a.out.mkdir(parents=True, exist_ok=True)
    state = a.out / "state"
    f_hist = a.out / "history.json"
    history = json.loads(f_hist.read_text()) if f_hist.exists() else []
    start = 0
    if (state / "state.json").exists():
        start = json.loads((state / "state.json").read_text())["step"]
        dit.update(tree_unflatten(list(mx.load(str(state / "lora.safetensors")).items())))
        opt.init(dit.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(state / "adam.safetensors")).items()))
        history = [h for h in history if h["step"] <= start]
        print(f"resuming from step {start}", flush=True)
    end = min(a.steps, start + a.chunk) if a.chunk else a.steps
    t0 = time.time()
    for step in range(start + 1, end + 1):
        d, sigma, eps = sample_for_step(data, step, a.seed, a.shift, a.latents)
        ts = time.time()
        loss, grads = step_fn(d["latents"], d["audio"], d["text"], d["mask"], sigma, eps)
        grads, norm = optim.clip_grad_norm(grads, 1.0)
        opt.update(dit, grads)
        mx.eval(dit.trainable_parameters(), opt.state, loss)
        history.append({"step": step, "loss": round(loss.item(), 5), "sigma": round(sigma, 3),
                        "grad_norm": round(norm.item(), 4), "s": round(time.time() - ts, 1)})
        print(f"step {step}/{a.steps} loss {loss.item():.4f} sigma {sigma:.2f} grad {norm.item():.3f} "
              f"{time.time() - ts:.0f}s, peak {mx.get_peak_memory() / 1e9:.1f} GB", flush=True)
        if step % a.save_every == 0 or step == a.steps:
            save_lora(dit, a.out / f"{a.name}-{step:04d}.safetensors", a.alpha / a.rank)
    state.mkdir(exist_ok=True)
    mx.save_safetensors(str(state / "lora.safetensors"), dict(tree_flatten(dit.trainable_parameters())))
    mx.save_safetensors(str(state / "adam.safetensors"),
                        {k: v for k, v in tree_flatten(opt.state) if isinstance(v, mx.array)})
    (state / "state.json").write_text(json.dumps({"step": end}))
    f_hist.write_text(json.dumps(history, indent=0))
    print(f"done up to step {end} ({(time.time() - t0) / 60:.0f} min)", flush=True)


if __name__ == "__main__":
    main()
