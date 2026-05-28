# Post-project MLX-port skill enhancements

**This section grows as we discover lessons during the port.** When the
port is complete, distill these into a PR to the `mlx-porting` skill
references (likely `common-pitfalls.md` and `repo-layout.md`).

Each entry: short title → what we hit → the rule for next time.

**`[toolkit candidate]` tag** marks lessons whose artifact (script, helper,
template) is a strong candidate for extraction into a shared
`mlx-port-toolkit` repo at end of port. Tag is applied when we see a pattern
likely to recur across DiT / VAE / text-encoder / audio-encoder ports.
Confirm by surviving at least 3 of the 4 architecture families in Stage 1
(VAE / umT5 / Whisper / DiT) before extracting.

Tagged so far:
- L4 (Conv*d transpose, gamma-skip pattern in safetensors loader) `[toolkit candidate]`
- L7 (HF Range-request safetensors header inspection) `[toolkit candidate]`
- The `diag_*.py` bisection template pattern (parity_helpers + per-layer
  output comparison loop) `[toolkit candidate]`
- The `[parity]` optional-extras pattern in pyproject.toml `[toolkit candidate]`
- The smoke/parity test split with HF auto-download env var `[toolkit candidate]`

## L1. "Existing MLX port of the same architecture" is not a drop-in for a different checkpoint

**What we hit:** `Blaizzy/mlx-video/mlx_video/models/wan_2/vae.py` is a
parity-tested MLX port of "the Wan 2.1 VAE." We assumed it was a drop-in
for Meituan's checkpoint (also `_class_name: AutoencoderKLWan`) and copied
it with three minor adaptations. At the first parity test it failed
structurally — mlx-video targets the original Wan-AI checkpoint with a
**different channel pattern and module hierarchy** than diffusers 0.38's
canonical `AutoencoderKLWan` that Meituan trained against. Channels per
decoder stage differ (`192→384` vs `192→192` for stage 1's first resblock).
Required a full structural refactor.

**Rule for next port:** When adopting an existing MLX port of "the same
architecture," do NOT assume checkpoint compatibility. Verify by loading
the target checkpoint at the *first* integration point (parity test) —
not just by smoke-testing shapes against random weights. Same class name +
same paper reference + same architectural family does NOT imply the
checkpoint loads. Different repos (Wan-AI vs diffusers vs custom forks)
ship structurally divergent variants.

**Skill update target:** new entry in `common-pitfalls.md` under a new
"adoption traps" heading, plus a checklist item in the workflow:
> Before adapting an existing MLX port: download the target checkpoint
> header, diff the top-level module keys against the MLX module's
> `parameters()` keys. If they don't match, refactor or re-port — don't
> rename your way out.

## L2. diffusers' module hierarchy changes across versions (config.json's `_diffusers_version` is a critical hint)

**What we hit:** Meituan's `vae/config.json` has `_diffusers_version:
0.33.0.dev0`. The shipped checkpoint uses diffusers 0.38's canonical
nested `up_blocks[B].resnets[R]` / `down_blocks` / `mid_block.resnets` /
`quant_conv` / `post_quant_conv` schema. mlx-video's port matches an older
flat `upsamples` / `downsamples` / `middle` schema. These are NOT
key-renameable — the channel arithmetic differs.

**Rule for next port:** When a HF config has `_diffusers_version`,
**install that exact diffusers version** (or the closest minor) and
import the canonical class from it. Read its `__init__` / forward
side-by-side with the MLX target. The "official" diffusers schema at the
time of training is the binding contract for the checkpoint.

**Skill update target:** `weight-conversion.md` should add a step:
> Pin `diffusers` to within one minor of the `_diffusers_version` in the
> checkpoint's config.json when doing parity tests.

## L3. Pipeline-level numerical tricks live outside the model definition

**What we hit:** Three forward-pass behaviors are NOT in
`modules/avatar/longcat_video_dit_avatar.py` but live in
`pipeline_longcat_video_avatar.py`:
1. `noise_pred = -noise_pred` (negative velocity flip before scheduler step)
2. The 3-pass disentangled CFG combiner formula
3. The `"Rep"` string sentinel that the WanResample upsample3d uses in
   `feat_cache` to skip temporal-doubling on the first chunk

Any of these missed at port time → silent numerical garbage downstream
that's impossible to bisect without intermediate-value diffing.

**Rule for next port:** Grep the inference pipeline file for: sign flips
(`= -`), magic-string sentinels (`"Rep"`, `"None"`, etc. used as dict
values), and conditional cache mutations (`if feat_cache[idx] is None`).
Document them in `notes/architecture-spec.md` BEFORE writing any forward
code. These are the "unobvious" surprises that the model-definition file
won't reveal.

**Skill update target:** new `common-pitfalls.md` entry — "pipeline-level
gotchas." Should reference the LongCat `"Rep"` case and the velocity
sign flip.

## L4. Conv weight transpose is `ndim >= 3`, not just 5D Conv3d

**What we hit:** Loaded PT VAE weights into MLX. Transposed 5D (Conv3d)
weights from `(O, I, T, H, W)` → `(O, T, H, W, I)` per the skill's
`transpose_pt_conv`. Forgot 4D Conv2d weights need the same treatment:
`(O, I, H, W)` → `(O, H, W, I)`. Hit at runtime with a confusing
"input shape (1, 33, 33, 96) vs weight (96, 96, 3, 3)" error.

**Rule for next port:** In the safetensors loader, classify by
`arr.ndim`, not by tensor name:
- `ndim == 3` → Conv1d, transpose `(0, 2, 1)`
- `ndim == 4` → Conv2d, transpose `(0, 2, 3, 1)`
- `ndim == 5` → Conv3d, transpose `(0, 2, 3, 4, 1)`

WITH an exception list for special 4D tensors that aren't conv weights
(e.g. RMS_norm `gamma` with shape `(C, 1, 1, 1)` — same ndim as Conv2d
weight but must not be transposed). Detect those by parameter-name
patterns (`"gamma"`, `"weight_scale"`, etc.) or by attribute-walking the
MLX module before applying transposes.

**Skill update target:** `parity-testing.md` — extend the
`transpose_pt_conv` helper docstring to mention Conv2d, and document the
gamma-shape gotcha.

## L5. Smoke tests should encode the architectural contract, not the initial implementation guess

**What we hit:** Our first VAE smoke test asserted `decode(N latent
frames) → 4*N video frames` based on a naive reading of the upsample
config. The correct contract (matching the chunked-decode logic) is
`1 + 4*(N - 1)` video frames. When the decoder was refactored to match
PT, the smoke test failed loudly — which was great signal, BUT it
revealed that the test had encoded an implementation guess, not the
canonical contract.

**Rule for next port:** Smoke-test assertions on shapes, frame counts,
and channel counts should be derived from the *reference model's
documented behavior* (or its forward pass on a known input), not from
the MLX port's initial shape guess. When the smoke test "fails" because
of a refactor, ask: was the test wrong, or is the code wrong? Often the
test was the initial guess.

## L6. Per-channel symmetric INT8 is not MLX-quantize-compatible

**What we hit:** Meituan ships an INT8 DiT checkpoint with
`quantization_method: int8_per_channel_symmetric`, one F32 scale per
output channel + I8 weight + BF16 bias. MLX's `mx.quantize` produces a
grouped layout (scales and biases per group of `group_size` weights,
weights bit-packed into int32). These are NOT directly cross-mappable —
loading Meituan's INT8 into an MLX-quantized `nn.Linear` will silently
fail or produce garbage.

**Rule for next port:** Before claiming "we can use the vendor's INT8
weights directly," look at the quantization metadata:
- `int8_per_channel_symmetric` / `int8_per_tensor_symmetric` → vendor
  layout. Write a custom MLX linear that does dequant-on-the-fly:
  `(x @ (w_i8.astype(bf16) * scale[None, :])) + bias`.
- `awq` / `gptq` / MLX-style grouped → MLX's `nn.quantize` may load it
  directly, but verify the group_size matches.

When in doubt, re-quantize from bf16 via `mx.nn.quantize(bits=N,
group_size=K)` — that's the canonical MLX path and gets you kernel
acceleration. Document the choice in `notes/int8-format.md` (or the
equivalent).

**Skill update target:** `weight-conversion.md` — extend the
quantization-scope section with a "vendor INT8 formats" subsection.

## L7. Cheap HF schema discovery via Range requests

**What we hit:** Wanted to inspect Meituan's INT8 checkpoint layout
without downloading 16 GB of weights. Solution: HTTP Range request for
the first 64 KB of one shard, parse the safetensors header (uint64 size
prefix + JSON metadata), inspect tensor names + dtypes + shapes. Total
~200 KB downloaded for full schema visibility.

**Rule for next port:** Before kicking off a multi-GB download to "look
at the weights," sketch a Python script that:
1. Lists the HF repo tree via `https://huggingface.co/api/models/{repo}/tree/main`
2. Fetches the safetensors `.index.json` (typically <300 KB) for the
   weight_map
3. Range-requests the first 64 KB of one shard to read the safetensors
   header

See [notes/config-snapshot/peek_int8.py](notes/config-snapshot/peek_int8.py)
for a working example.

**Skill update target:** `references/weight-conversion.md` — add a
"cheap schema discovery" section near the top of the recipe-writing
flow.

## L8. Stage 0 (recon) is where you actually save time, not where you spend it

**What we hit:** v2's plan jumped directly to "MLX Python port." We
inserted a Stage 0 recon pass (~3 hours for 7 deliverables) that:
- Resolved Open Question #1 (audio-injection wiring) from source — no
  speculation needed
- Demystified two "innovations" v2 had attributed to LongCat (Cross-Chunk
  Latent Stitching, Disentangled UCG) that turned out to be standard
  CFG combiner + standard KV-cache chunking
- Surfaced critical numerical conventions (fp32 modulation, sign flip,
  "Rep" sentinel, audio_block=5 in v1.5 vs 12 in v1.0)
- Caught the VAE schema mismatch INSIDE the recon (via INT8 format peek
  showing the diffusers 0.38 hierarchy) — which would have cost a day if
  hit during the port

**Rule for next port:** Make Stage 0 explicit in the plan. Default
deliverables:
1. Audio/conditioning injection wiring spec (`notes/audio-injection-wiring.md` or equiv)
2. Architecture spec with config defaults and all known numerical conventions
3. Tech report read (extract the PDF, grep for sign flips & magic strings)
4. HF config snapshot (all configs from both code & weights repos)
5. INT8 / quant format inspection
6. MLX venv + Metal verification
7. Reference-MLX-port diff (precedent code, with explicit reuse / divergence list)

Time-box at 4–8 hours. The output is **durable notes**, not code.

**Skill update target:** workflow section of `SKILL.md` — add Stage 0
between "Read reference" (step 1) and "Scaffold -mlx fork" (step 2),
with the 7-item deliverable list above as default.

## L10. MLX Metal-GPU fp32 ≠ true fp32 for long-accumulator ops

**What we hit:** With identical fp32 q/k/v inputs (matching to 5e-6 between PT
and MLX), `mx.fast.scaled_dot_product_attention` AND a hand-written
`softmax(QK^T)·V` chain BOTH produced output differing from PT
`F.scaled_dot_product_attention` by `max_abs ≈ 1.6e-3`. Running the *same MLX
code* under `mx.stream(mx.cpu)` brought it back to `max_abs ≈ 6e-6`. Same
pattern with Conv2d at large spatial sizes (64×64 with 384 in / 192 out):
GPU ≈ 5e-2 drift, CPU ≈ 6e-5.

Metal's fp32 ALU appears to use a reduced internal precision (TF32-like, ~19
bits of mantissa rather than 23) for matmul/conv accumulators. The hit is
proportional to the number of accumulated multiply-adds — small for short
dot products, very visible for long ones (attention QK^T with D=384,
post-upsample Conv2d at 64×64 spatial).

**Rule for next port:**
1. **Set parity thresholds that match the hardware, not the textbook.** The
   skill's `< 1e-3 single layer / < 1e-2 full pipeline` assumes CPU-fp32-vs-
   CPU-fp32 precision. For MLX-Metal-GPU vs PT-CPU comparison, realistic
   numbers are `~1e-3 single layer / ~2e-2 full pipeline`.
2. **For strict bit-correctness verification**, run the suspect layer (or
   the whole model) under `with mx.stream(mx.cpu):` and compare against PT.
   If that passes at tight thresholds, the implementation is correct and any
   remaining drift on GPU is Metal precision (acceptable).
3. **Cap individual high-precision ops to CPU stream when easy** — e.g. the
   2 attention blocks in a Wan VAE go on CPU stream at negligible perf cost.
   Heavier modules (full DiT) should stay on GPU and accept the precision.
4. **Verify visually, not just numerically.** A max_abs of 1.5e-2 on a
   `[-1, 1]`-bounded output is ~0.75% per-pixel — below any human perceptual
   threshold. Don't waste hours chasing strict numeric parity when the actual
   output quality is fine.

**Skill update target:** `parity-testing.md` — extend the threshold table
with a "MLX-GPU vs PT-CPU" column and a note about the CPU-stream escape
hatch. Add `mx.stream(mx.cpu)` example to the `assert_parity` helper.

## L11. Spurious "huge parity failure" from normalization-convention divergence

**What we hit:** Initial decoder parity reported `max_abs = 1.385` (failure)
even though the IMPLEMENTATION was correct. Root cause: our MLX
`AutoencoderKLWan.decode()` had inherited a convention from mlx-video that
denormalizes input `z` internally (`z = z * std + mean`), but PT's
`AutoencoderKLWan.decode()` (diffusers convention) treats input z as already
denormalized — the pipeline caller is responsible for the
denormalization step. Result: same `z_np` produced very different inputs to
the actual decoder forward pass. We chased the error all the way through
attention bisection before noticing the difference at the API surface.

**Rule for next port:** When designing a class's public API (encode /
decode / forward), **match the reference repo's input/output convention
exactly**, even if a different convention seems more "convenient" for
end-user pipeline code. Internal helpers like `normalize_latents` /
`denormalize_latents` can still expose the convenience — but the core
forward pass must take the same input scaling as the reference. Otherwise
the first parity test of a new component runs with effectively different
inputs on the two sides, producing huge spurious errors that mask the real
implementation status.

A specific test for this trap: compute `out_pt - out_mx` at a known-good
intermediate point (e.g. after `post_quant_conv` for a VAE decode). If it's
within 1e-6 there but blows up immediately after, the bug is in your forward
chain. If it's already blown up at the FIRST op in the forward pass, suspect
input-convention divergence at the API surface, not a forward-pass bug.

**Skill update target:** `common-pitfalls.md` — new entry under "input
convention traps." Reference Hugging Face diffusers' "VAE outputs raw z,
pipeline scales it" pattern as the canonical case.

## L17. mlx-arsenal FlowMatchEulerDiscreteScheduler appends the WRONG sigma sentinel `[toolkit candidate]`

**What we hit:** First real-weight inference produced pure noise. Diagnosed:
`mlx_arsenal.FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...)`
appends `np.ones(1)` (= 1.0, *high noise*) as the final sigma boundary.
PT diffusers' equivalent appends `0.0` (*clean target*). The Euler step
formula is identical (`prev_sample = sample + (sigma_next - sigma) * v`),
so the LAST denoising step in our 8-step DMD schedule did
`sample + (1.0 - 0.124) * v` = **re-added ~88% of the noise** at the
final step, undoing all prior denoising. Output was pure noise textured
by the VAE decoder.

**Fix:** Overwrite the trailing sentinel after `set_timesteps`:
```python
self.scheduler.set_timesteps(N, sigmas=my_sigmas)
self.scheduler.sigmas = mx.concatenate(
    [self.scheduler.sigmas[:-1], mx.array([0.0], dtype=mx.float32)]
)
```

**Rule for next port:** When wiring an mlx-arsenal scheduler, read
`set_timesteps` source and verify the appended sentinel matches your
inference direction. PT diffusers and mlx-arsenal disagree on this
specific value. Until upstream aligns, every inference pipeline using
mlx-arsenal's FlowMatchEulerDiscreteScheduler needs this 3-line fix.

**Skill update target:** `references/common-pitfalls.md` — add a
"scheduler sentinel sigma" note. Also worth upstreaming an issue to
`mlx-arsenal` proposing alignment with PT diffusers' convention (or at
minimum a `sigma_min` argument so the user can override).

## L18. `mx.fast.scaled_dot_product_attention` rejects fp32 mask with bf16 Q/K/V `[toolkit candidate]`

**What we hit:** Cross-attention mask was built in fp32 (for additive-mask
sentinel precision: `-3.389e38`). When loaded weights produced bf16 Q/K/V,
SDPA raised:
```
ValueError: [scaled_dot_product_attention] Mask type must promote to output type bfloat16.
```
because fp32 → bf16 is a downcast and MLX rejects it in the SDPA dispatch.

**Fix:** Build the mask in fp32 for sentinel precision, then cast to
`q.dtype` immediately before passing to SDPA:
```python
mask = mx.full((B*N, sum_kv), -3.389e38, dtype=mx.float32)  # fp32 build
# ... populate mask ...
mask = mask[None, None, :, :].astype(q.dtype)  # cast to bf16 for SDPA
out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
```
bf16's representable max is the same as fp32's (~3.39e38), so the sentinel
value survives the downcast. Only precision in the mantissa changes,
which doesn't affect the additive-mask gating behavior (we just need
"large negative" — any value < -1e30 works in softmax).

**Rule for next port:** SDPA masks should always be cast to `q.dtype` at
the SDPA boundary. Build in whatever precision you need, but cast
explicitly. Defensive helper:
```python
def _sdpa_mask(mask: mx.array, q: mx.array) -> mx.array:
    return mask.astype(q.dtype)
```

**Skill update target:** `references/attention-patterns.md` — add an
"SDPA mask dtype promotion" subsection. Pair with the `mlx.fast.sdpa`
parity gotcha from L10 (GPU fp32 precision).

## L15. safetensors numpy framework cannot read bf16 — use mx.load instead `[toolkit candidate]`

**What we hit:** Recipe used `safetensors.safe_open(path, framework="numpy")` to
load PT shards. Worked fine for VAE (fp32), umT5 (fp32), and Whisper (fp16 —
numpy supports it). Crashed on the Avatar DiT shards with:
```
TypeError: data type 'bfloat16' not understood
```
because numpy has no native bf16 dtype, even as of numpy 2.x. The shards
loaded fine for the smaller models but the recipe died ~3 components into
the conversion, after 2+ hours of weight downloads completed.

**Fix:** Use `mx.load(path)` which is bf16-native — returns
`dict[str, mx.array]` with the source dtype preserved. Eliminates the
numpy round-trip entirely.

**Rule for next port:** Default to `mx.load` for safetensors loading in
recipes. Reserve `safe_open(..., framework="numpy")` for cases where you
explicitly need numpy arrays AND you've verified the source dtype is one
numpy supports (`fp32`/`fp16`/`int*`). bf16 source weights are increasingly
common (all recent video and image gen models ship them) — and the failure
mode is hours-late, not at load time.

**Skill update target:** `references/weight-conversion.md` — add a "load
function selection" subsection at the top of the recipe pattern.
`tests/smoke/test_recipe_smoke.py::test_layout_and_cast_handles_bf16_source`
is the template for the regression test.

## L16. Dtype-preserve casts must distinguish "already at target" from "force upcast" `[toolkit candidate]`

**What we hit:** Recipe's cast logic was:
```python
if dtype is not None and arr.dtype != dtype and not _key_should_stay_fp32(name):
    arr = arr.astype(dtype)
```
Reads as "cast to target dtype unless the key should stay fp32." But the
control flow accidentally meant: if `_key_should_stay_fp32(name)` returns
True, the cast is SKIPPED entirely — including when we needed an UPCAST
from bf16 source to fp32 target. With Meituan's source weights at bf16,
adaLN modulation weights would have been silently kept at bf16 instead of
the required fp32, producing the same numerical drift the `_FP32` suffix
was designed to prevent.

**Fix:** Split the logic into "fp32-special keys → always fp32" vs.
"everything else → target dtype":
```python
if _key_should_stay_fp32(name):
    if arr.dtype != mx.float32:
        arr = arr.astype(mx.float32)
elif dtype is not None and arr.dtype != dtype:
    arr = arr.astype(dtype)
```

**Rule for next port:** When you have "should stay at X precision" keys,
write the cast logic with an explicit X branch, not as a negation of the
default cast. Negated branches don't compose with upcasts. Regression test
template: pass a bf16 source through your cast fn and assert the
fp32-special keys come out fp32 (not bf16, not "unchanged").

## L14. PT diffusers ↔ MLX-arsenal API divergences worth knowing `[toolkit candidate]`

**What we hit:** Two silent-failure pitfalls when wiring `mlx_arsenal`'s
scheduler:
1. `FlowMatchEulerDiscreteScheduler.step()` returns `mx.array` directly,
   NOT a `(prev_sample, ...)` tuple like PT `diffusers`. Code copied from a
   diffusers pipeline that does `latents = scheduler.step(...)[0]` silently
   strips the batch axis from the array (PT's `[0]` would unpack the tuple;
   MLX's `[0]` indexes into the array).
2. `mx.array` has no `.repeat()` instance method (PT has `tensor.repeat(n)`);
   use the functional form `mx.repeat(arr, n, axis=...)`.
3. MLX has no scalar 0-d → 1-d auto-broadcast for repeat; ensure
   `arr.ndim >= 1` first (e.g. `if t.ndim == 0: t = t[None]`).

**Rule for next port:** When you port code that uses PT-diffusers APIs to
MLX-arsenal:
- Wrap every scheduler call site in a unit test that asserts the OUTPUT
  shape equals the INPUT shape. Catches the `.step()` tuple-vs-array bug
  immediately.
- Find-and-replace `.repeat(` (instance method) → `mx.repeat(` (functional)
  in your initial port.
- Add an `assert arr.ndim >= K` immediately before any `mx.repeat` or
  `mx.reshape` that assumes a minimum dimensionality.

**Skill update target:** `references/common-pitfalls.md` — add a new
"API drift between PT and MLX equivalents" entry. Bundled-script
candidate: a tiny `tests/smoke/test_scheduler_shape_preservation.py` that
asserts shape preservation across scheduler.step on a randomly-sized tensor.

## L13. Validate rename functions WITHOUT downloading weights `[toolkit candidate]`

**What we hit:** Stage 1.2 (umT5) port needed a PT→MLX key-rename function
(mlx-video uses compact names; HF transformers uses verbose hierarchy).
Validating it normally requires downloading 22 GB of umT5 weights. Instead,
we fetched only the safetensors `.index.json` (~22 KB) which contains every
PT key, applied the rename function to each, and asserted the result exists
in our MLX model's `parameters()`. Caught real issues (initial pass left
`pos_embedding.embedding` keys unmapped) with no large download.

**Rule for next port:** Whenever your port needs PT→MLX key renames, write
a `test_keymap_renames_completely` smoke test that:
1. Fetches the checkpoint's `*.index.json` (always small, even for multi-GB models)
2. Iterates every PT key, applies your rename function
3. Asserts the output is in `tree_flatten(mx_model.parameters())`'s key set

This catches every unmapped key in <2 seconds with no network beyond a tiny
index file. Run BEFORE downloading actual weights. If any key fails, fix
the rename function (or refactor the MLX module) until all 100% map.

**Skill update target:** `parity-testing.md` — add this as a separate test
type alongside the layer-by-layer parity pattern. Bundled script candidate:
`tests/parity/test_umt5_parity.py::test_umt5_keymap_renames_completely` is
the template.

## L12. The "Rep" sentinel pattern for skip-first-step in causal-convolution chunks

**What we hit:** Wan VAE's `WanResample.upsample3d` uses an inline string
sentinel `"Rep"` stored in `feat_cache[idx]` to mark "this slot has been
visited once but no temporal context has been accumulated yet — skip the
time-conv on this call but apply it normally next time." This makes the
first chunk in chunked decode emit 1 video frame (no temporal doubling),
while subsequent chunks emit 4 video frames each (full temporal doubling
through both upsample3d stages stacked). Without this sentinel, you get
`4 × N` video frames out of N latent frames instead of the canonical
`1 + 4(N−1)`.

The sentinel is NOT documented anywhere. It only appears in the source code
of `WanResample.forward`. Easy to miss when reading the model definition file
alone, since the model defn doesn't show the chunked-decode iteration.

**Rule for next port:** When porting any model with chunked/streaming
inference (audio, video, long-context LLMs), grep the inference code for
string sentinels in cache structures. They're often used to mark
"initialized-but-empty" states distinct from "uninitialized" (None). Search
patterns: `feat_cache[idx] = "..."`, `if cache[idx] == "..."`, `cache_x ==
"..."`. Document them as part of Stage 0's architecture spec.

**Skill update target:** `common-pitfalls.md` — extend the "pipeline-level
gotchas" entry (added in L3) with the string-sentinel pattern.

## L9. Refactor cost during a port is paid twice — but RE-port cost when the precedent is wrong is paid once

**What we hit:** The skill rightly emphasizes "no refactoring during the
port" (preserves diff-ability with upstream). But we hit a case where
the existing MLX precedent was structurally wrong for our checkpoint.
There the choice was: (a) rename-our-way-around it (fails — channel
arithmetic differs), (b) refactor MLX module hierarchy to match
canonical PT (chosen — works). The refactor LOOKS like rule-breaking but
it's actually *increasing* isomorphism with the right upstream.

**Rule for next port:** The "no refactoring" rule applies to code that
already matches upstream. It does NOT mean "preserve any MLX precedent
you started from at all costs." If the MLX precedent diverges from your
checkpoint, refactor it to match the checkpoint — that IS the
isomorphism the rule was protecting.

**Skill update target:** `SKILL.md` — clarify the "no refactoring" rule
with: "preserve isomorphism *with the checkpoint's canonical reference*
— refactor away from MLX precedents that target a different reference."

## L19. HF CLI 1.x dropped `hf upload --create-repo` — use `hf repo create --exist-ok` `[toolkit candidate]`

**What we hit:** Our publish script used `hf upload <repo> <file> --create-repo`
which worked at the time of the initial bf16 publish. By the time we
went to publish the q4/q8 quant variants the local CLI had bumped to
1.16.4 and that flag is gone:

```
Error: No such option '--create-repo'.
Did you mean: --create-pr, --no-create-pr?
```

The new flow is `hf repo create <repo> --type model --exist-ok` (idempotent)
followed by a plain `hf upload`. `--exist-ok` makes the create call safe
to re-run after first publish without erroring on "already exists".

Bonus failure mode: when running `bash publish.sh quant 2>&1 | tee log`,
the script's internal `set -euo pipefail` masks `hf upload`'s exit code
through the outer pipe — the *driving* shell sees `tee`'s exit status
(0), so the task reports success despite the inner failure. Always
check the log content, not just the exit code.

**Skill update target:** `common-pitfalls.md` — add a "HF CLI quirks"
section: the `--create-repo` deprecation + the bash-pipe exit-code
masking gotcha. Recommend the split `hf repo create --exist-ok` + `hf
upload` pattern for all future publish scripts.

## L20. `hf upload` of large multi-GB repos stalls in `CLOSE_WAIT` without retry `[toolkit candidate]`

**What we hit:** Running `hf upload mlx-community/<repo> <local-dir> .` for
~31 GB of sharded safetensors (the q8-merged DiT). After ~10 minutes the
process was still alive (RSS 4.7 GB, ~37s CPU time) but had completely
stopped making progress:

```bash
lsof -p <PID> | grep TCP
# Python ... TCP 10.46.15.54:... -> cloudfront ...:https (CLOSE_WAIT)

netstat -ib | awk '/en0/ {print $7}'
# Outbound bytes: ~1.4 KB/s
```

`CLOSE_WAIT` means the remote (CloudFront / xet CDN) tore the connection
down but our `hf upload` never `close()`'d it on our side and never
retried. The CLI logs nothing — looks alive from the outside, including
to a parent `bash` watcher.

Made worse by: when run via `bash publish.sh quant 2>&1 | tee log.txt`,
`set -euo pipefail` does not propagate up through the outer pipeline
either, so any harness that watches "exit code 0 = success" gets fooled.

**Diagnostic recipe:**
1. `ps -o pid,rss,time,etime -p <PID>` — alive but CPU time << elapsed → IO-bound
2. `lsof -p <PID> | grep TCP` — any `CLOSE_WAIT` is the smoking gun
3. `netstat -ib` 5-sec delta on the relevant interface — <10 KB/s = stalled

**Workaround:** `kill <PID>` and re-run the same command from an
interactive terminal. `hf upload` is content-addressed (xet), so files
that *did* upload are skipped on retry — net cost is just the time to
re-hash and re-stream what was in flight. Running interactively also
gives Ctrl-C if a second stall happens.

**Skill update target:** `common-pitfalls.md` — extend the "HF CLI quirks"
section started in L19 with: (a) `hf upload` lacks a watchdog timeout for
CLOSE_WAIT sockets on multi-GB repos; (b) for any upload >10 GB, drive
it from an interactive terminal where the user can Ctrl-C; (c) tee with
`set -euo pipefail` does NOT propagate through outer pipelines — use
`> log.txt 2>&1` for background uploads instead.

## L21. `swift test` doesn't bundle `default.metallib`; use `xcodebuild test` instead `[toolkit candidate]`

**Where this bites:** any mlx-swift-based port. `swift test` from the CLI
on macOS crashes on the FIRST MLX op dispatch with:

```
MLX error: Failed to load the default metallib. library not found
  at .../mlx-swift/Source/Cmlx/mlx-c/mlx/c/stream.cpp:106
```

The crash is in `stream.cpp:106` during default-stream lookup — it fires
even if you immediately call `Device.setDefault(device: .cpu)` because
that call itself triggers the stream lookup. `MLXRandom.normal` /
`MLXArray.zeros` of any non-trivial shape will also blow up.

Root cause: `Cmlx` declares the Metal shaders as Swift Package resources
(`.process` rule producing `default.metallib`), but SwiftPM's CLI test
runner on macOS doesn't run the metallib compile pass that Xcode's build
system does. The metallib doesn't exist on disk after `swift test`'s
build phase, so it can't be loaded.

**Fix:** invoke tests via xcodebuild instead.

```bash
xcodebuild test \
    -scheme <Package>-Package \
    -destination "platform=macOS"
```

This builds the metallib correctly and runs the same XCTest binaries.
24-test run took 1.4 s wall clock in our case — comparable to `swift test`
when it works.

**Skill update target:** `common-pitfalls.md` — add a "Swift port quirks"
section with this + the swift-transformers Hub-target-not-exposed
quirk from L19's surrounding context. Recommend Swift ports document
this in the top-level README so external contributors don't lose 30
minutes to "why are my tests crashing on the first kernel call."
