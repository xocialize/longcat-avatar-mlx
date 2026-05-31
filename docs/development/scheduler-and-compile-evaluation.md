# Scheduler & `mx.compile` evaluation for LongCat

**Date:** 2026-05-30
**Triggered by:** A flow-matching DPM-Solver++(2M) scheduler landed
upstream in [`xocialize/lance-mlx` PR #4](https://github.com/xocialize/lance-mlx/pull/4)
gave Lance image generation a ~2× wall-clock win on its 30-step Euler
path. Question: does that transfer to LongCat, and are there other
per-step optimizations worth pursuing?

**TL;DR:** Both DPM++(2M) and `mx.compile()` are *plausibly applicable*
but **lower ROI than they look** for LongCat specifically. The
DMD-distilled 8-step paths (the actually-shipping user variants) get
~zero benefit from either. Only the `bf16` 50-step base path benefits,
and that path is positioned as "for experiments" not production. **No
immediate action recommended.** Capturing the evaluation here so we
don't re-derive it.

## Part 1 — DPM-Solver++(2M) / variable-step Adams-Bashforth 2

### Architectural fit

LongCat's denoising loop ([`pipeline_mlx.py:309-324`](../../longcat_video_avatar/pipeline_mlx.py))
runs through `mlx_arsenal.FlowMatchEulerDiscreteScheduler` — a
Diffusers-style abstraction, NOT inline Euler math:

```python
for i, t in enumerate(timesteps):
    noise_pred = self._cfg_forward(latents, t_arr, ...)
    latents = self.scheduler.step(noise_pred, t, latents)
```

This differs from Lance (where the Euler step was `latents = latents -
velocity * dt` inline, making the PR #4 fix a 1-line swap). For LongCat,
adding DPM++ means **adding a new scheduler class** with the same
`set_timesteps` / `step` API. Two paths:

| Path | Pros | Cons |
|---|---|---|
| **A — Upstream to mlx-arsenal** | Cleanest; helps all downstream consumers. PR #4's 70 LOC is the starting point. | PR + review cycle. Need to match `FlowMatchEulerDiscreteScheduler` API exactly. |
| **B — Private to LongCat** | Immediate; no external dependency. | Drift risk; mlx-arsenal may add DPM++ later. |

### Where the win actually lands

| Variant | Steps | DPM++ helps? |
|---|---|---|
| `bf16` | 50-step flow matching | ✓ Yes. 2-2.5× speedup likely (50 → ~20 steps). ~6 min → ~2.5 min on 29-frame benchmark. |
| `bf16-dmd-merged` | 8-step DMD distilled | ✗ Probably hurts. |
| `q4-dmd-merged` | 8-step DMD distilled | ✗ Probably hurts. |
| `q8-dmd-merged` | 8-step DMD distilled | ✗ Probably hurts. |

### Why DPM++(2M) doesn't help DMD-merged paths

1. **DMD distillation explicitly trains for 8-step convergence** with a
   specific sigma schedule. The "velocity field" between adjacent
   sigmas isn't smooth — each step is more like "denoise → re-noise"
   than a continuous ODE. Multistep methods (AB2 / DPM++) assume
   smooth velocity fields; that assumption fails on distilled samplers.
2. **First-step Euler warm-up dominates at low step counts.** At 8
   steps the first-step Euler fallback is 12.5% of compute; at 50
   steps it's 2%. The compounding 2nd-order accuracy benefit doesn't
   have many steps to accumulate over.

### LongCat-specific wrinkles to handle if you do this

- **Three-pass CFG with velocity flip.** `_cfg_forward` ([`pipeline_mlx.py:183-232`](../../longcat_video_avatar/pipeline_mlx.py))
  does a 3-pass forward (one batched 2x `[neg_text, pos_text]`, one fully
  uncond) then `flip_velocity_for_scheduler(combined)` at line 232. The
  multistep history (`_v_prev`) needs to store the **post-flip** combined
  signal consistently. Easy to get right but worth a unit test (mirror
  Lance's `test_first_step_is_euler` pattern).
- **DMD sigma last-step replacement.** Lines 296-301 explicitly replace
  the last sigma with `0.0` to drive sigma → 0 at termination (PT
  diffusers convention). The AB2 step at the boundary needs to handle
  `dt` to a zero-sigma final state. PR #4's solver assumes `dt > 0`;
  verify behavior at the final boundary step.
- **`num_cond_latents` baked-in.** Fixed across all steps within one
  generation; safe to capture as a Python int closure variable.

### Honest user-impact assessment

The README positions DMD-merged variants as the **recommended user
paths** ("recommended for 64 GB+ Macs" / "recommended for 32-48 GB
Macs"). The bf16 base is positioned as "for runtime-merge /
multi-strength experiments." So **DPM++ would benefit a minority user
segment.** That's why this is a lower-priority follow-up rather than a
must-do.

## Part 2 — `mx.compile()` on the per-step body

### Compile candidate locations

Inside the denoising loop, three nesting levels could be compiled:

1. **`_cfg_forward` (the 2 DiT calls + CFG combine + velocity flip)** —
   biggest practical compile unit. Pure function modulo the closure
   captures (`self.dit`, `self.config.*_guidance_scale`). All inputs
   `mx.array`; `num_cond_latents: int` is fine as Python closure.
2. **`self.dit(...)` (the DiT forward)** — clean compile target. All
   inputs/outputs are `mx.array`. Shape-stable across steps. The
   internal dtype cast (`hidden_states.astype(dtype)` at
   [`longcat_video_dit.py:232`](../../longcat_video_avatar/models/longcat_video_dit.py))
   is fine for compile (closure-captured dtype).
3. **`scheduler.step` (mlx-arsenal)** — **NOT a compile target.**
   Library code we don't own; has internal state (sigma index, prev
   sigmas). Stateful methods break `mx.compile` referential transparency.

Recommended placement: **wrap `_cfg_forward`** (Option 1). Captures most
per-step cost (2 DiT forwards dominate) with one compile boundary.

### Expected wins (modest, NOT slam-dunk)

| Variant | Step count | Per-step time | Compile benefit expected |
|---|---|---|---|
| `bf16-dmd-merged` 8-step | 8 | ~13 s/step | 5-15% wall-clock (small dispatch fraction at 13s/step; warm-up costs 1-2 steps' worth on first call) |
| `bf16` 50-step | 50 | similar | 10-20% wall-clock (more steps to amortize warm-up; may approach memory-bound) |
| `q4-dmd-merged` 8-step | 8 | similar | Likely similar to bf16-dmd (Linear ops dominate; mx.fast.quantized_matmul already fused) |

These are **5-20% wins, not 2-3×.** The LongCat DiT is 13.6B params;
per-step is largely compute-bound on the matmul side. `mx.compile`
helps most when dispatch / temp-allocation overhead is the bottleneck,
which is true for small models or many small ops, less true here.

### Risks specific to LongCat

1. **First-call compile time.** A 13.6B model with complex control
   flow may take **30s-2min** to trace on first invocation. For a
   single inference of 105s (DMD-merged) that warm-up could be net
   loss. For batch inference (multiple generations in one pipeline
   load) it amortizes well.
2. **Shape-stability verification needed.** `timestep` is `[B]` or
   `[B, T]`. After the `timestep[None]` guard at
   [`pipeline_mlx.py:200-201`](../../longcat_video_avatar/pipeline_mlx.py)
   the shape is at least 1D. Across steps within one generation
   shapes should be stable; verify before committing.
3. **`bound method` vs free function.** `mx.compile` is cleanest on
   free functions. Wrapping `self._cfg_forward` likely needs a small
   refactor — extract the per-step body into a free function that takes
   `(dit_module, config, latents, ...)` and is wrapped, OR use the
   `lambda self_, *a: self_._cfg_forward(*a)` pattern (less clean).
4. **Numerical equivalence test required.** `mx.compile` is graph
   optimization, not precision change — outputs MUST match exactly
   (modulo `mx.fast.scaled_dot_product_attention` nondeterminism). Add
   a pixel-MAD pre/post test as a regression guard.

### Recommended experiment if pursued

```python
# scripts/experiments/compile_ab.py
# 1. Baseline: time 3 consecutive generations (single + 2 reuse)
# 2. Wrap _cfg_forward with mx.compile
# 3. Time same 3 generations — record:
#    - first-call wall-clock (warm-up included)
#    - second-call wall-clock (compile cache hit)
#    - third-call wall-clock (steady state)
# 4. Output equivalence: SHA or pixel-MAD on a fixed-seed reference
# 5. Verdict matrix:
#    - 2nd/3rd call ≥10% faster AND 1st call <baseline+30%: ship
#    - 2nd/3rd call <5% faster: skip
#    - mid-range: depends on usage pattern (batch vs single)
```

Effort: ~1-2 hours including the equivalence + timing harness.

## Part 3 — combined ranking

**For LongCat specifically, current ranking of optimization candidates
(highest-leverage first):**

1. **Neither DPM++ nor mx.compile makes the shortlist for the
   user-recommended DMD-merged paths.** Those are already
   near-optimal at 8 steps; the bottlenecks are elsewhere (VAE decode,
   audio embedder, weight load).
2. **mx.compile** is the lower-risk of the two for the bf16 50-step
   path: smaller win but no algorithmic change, can be A/B'd in an
   hour, and benefits ALL variants if it works (modest gain on
   DMD-merged too).
3. **DPM++ for bf16 50-step** is the bigger potential win but only on
   the minority user variant, and the scheduler-class work makes it a
   larger investment than a pure `@mx.compile` decorator.
4. **Better targets if perf becomes a priority:** look at VAE decode
   wall-clock contribution, weight load amortization (cache loaded
   pipeline), and Whisper audio embedder cost. These are likely
   bigger fractions of total user-visible latency than the DiT
   per-step micro-optimizations.

## What was checked vs not

**Checked:**
- `pipeline_mlx.py` denoising loop structure ([line 309-324](../../longcat_video_avatar/pipeline_mlx.py))
- `_cfg_forward` CFG pattern ([line 183-232](../../longcat_video_avatar/pipeline_mlx.py))
- DiT `__call__` signature ([line 205-239 in `models/longcat_video_dit.py`](../../longcat_video_avatar/models/longcat_video_dit.py))
- Scheduler abstraction (mlx-arsenal `FlowMatchEulerDiscreteScheduler`)
- README variant table (DMD-merged vs bf16 base)
- Existing `mx.compile` usage in the codebase (none yet)

**Not checked (would need before committing to either):**
- `mlx_arsenal.FlowMatchEulerDiscreteScheduler.step()` internals — to
  confirm what state it mutates and whether a Drop-in DPM++ class can
  match its API surface exactly
- `mlx_arsenal.get_dmd_distilled_sigmas()` shape/values — for the
  zero-sigma boundary handling in any DPM++ port
- Per-component wall-clock breakdown of one full generation (DiT vs
  VAE vs audio vs scheduler) — to verify the DiT forward really
  dominates and is the right optimization target
- Whether `longcat-avatar-mlx-swift` has an equivalent scheduler
  abstraction — relevant if mx.compile / DPM++ work needs Swift mirror

## Resume points if either becomes worth doing

**For DPM++(2M):** start with [`xocialize/lance-mlx/src/lance_mlx/scheduler/solvers.py`](https://github.com/xocialize/lance-mlx/blob/main/src/lance_mlx/scheduler/solvers.py)
(~70 LOC). Adapt to `mlx_arsenal.FlowMatchEulerDiscreteScheduler`'s
`set_timesteps` / `step` API. Handle the post-flip velocity history
and the zero-sigma boundary. Land as a private LongCat class first,
upstream to mlx-arsenal once stable.

**For `mx.compile()`:** wrap `_cfg_forward` per the experiment outline
in Part 2. Land behind a `pipeline.compile=True` config kwarg so users
can opt in (and warm-up cost is explicit). Default to off until A/B
data justifies on-by-default.

## Source repo for context

The Lance evaluation that motivated this lives at
[`xocialize/lance-mlx`](https://github.com/xocialize/lance-mlx/blob/main/notes/perf_optimization_todos.md)
— the `notes/perf_optimization_todos.md` there has the same Perf-1 /
TODO-1 analysis for image generation. LongCat's situation differs
because DMD-distilled paths are the user-recommended defaults; Lance
has no equivalent distilled fast path so its 30-step Euler IS the
shipping path.
