# Cross-port analysis: lance-mlx PRs → LongCat

**Date:** 2026-06-05
**Updated:** 2026-06-05 (same day) with revised findings after a
second-pass code read across all four Wan-family VAE modules.
**Triggered by:** Three substantive optimization PRs landed on
`xocialize/lance-mlx` recently. This doc enumerates what would
transfer to LongCat, what the effort actually looks like, and which
items rank highest if we invest the hours.

> **⚠️ Read first:** the streaming-decode portion of this analysis
> has been superseded by
> [phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md](../../../phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md).
> The second-pass code read uncovered two surprises that change the
> recommended sequencing: (1) mlx-video stock `wan_2/vae.py` (used by
> phantom-wan and bernini-r) is incomplete for streaming — its
> `Resample.upsample3d` doesn't honor `feat_cache`; (2) LongCat's
> `autoencoder_kl_wan.py` is MORE complete than mlx-video stock —
> its `Resample.upsample3d` already has the diffusers `"Rep"` sentinel.
> Net effect: the highest-leverage port target is mlx-video stock
> (benefits phantom-wan + bernini-r + upstreamable), not LongCat's
> private fork. LongCat streaming port effort drops from 4-8 hr to
> 2-4 hr because the orchestrator is the only missing piece.
> The LongCat-specific sections below are still accurate for the
> LongCat port itself; only the sequencing recommendation has changed.

## Source PRs

| PR | Description | Mechanism | Lance win |
|---|---|---|---|
| [#4](https://github.com/xocialize/lance-mlx/pull/4) | `DPM-Solver++(2M)` scheduler | Variable-step Adams-Bashforth 2 on the flow-matching ODE | ~2× wall-clock (30 → 12 steps) |
| [#6](https://github.com/xocialize/lance-mlx/pull/6) | `memory_mode` (tower relay + tiled VAE decode) | Shed-cascade: load → run phase → free → load next phase | 32 GB→16 GB envelope |
| [#7](https://github.com/xocialize/lance-mlx/pull/7) | Lossless streaming VAE decode | Temporal causal-cache (Wan-family encoder pattern, mirrored on decode) + spatial halo-tile crop | 12 GB → 8 GB peak decode |

## Architectural reality

What I assumed earlier turned out to be wrong on one key point: **LongCat
does NOT use Lance's `Wan22VAEDecoder`**. We use our own port at
`longcat_video_avatar/models/autoencoder_kl_wan.py` which targets the
diffusers `AutoencoderKLWan` weight layout (16-channel, channels-first
NCHWD tensor convention). Lance's `vae_stream.py` imports from
`mlx_video.models.wan_2.vae22`, a different module with a different
class hierarchy and tensor layout (48-channel, NHWC).

The streaming **algorithm** still applies (both are causal-in-time Wan
VAE family with `CACHE_T = 2` boundary frames and a `feat_cache /
feat_idx` chunked-encode pattern). The **code** does not transfer
unchanged.

## Per-PR applicability + effort

### PR #4 — DPM-Solver++(2M) scheduler

| | Detail |
|---|---|
| Algorithmic fit | Sound for the bf16 50-step base path; **does NOT help DMD-distilled 8-step paths** (distilled samplers violate the smooth-velocity assumption multistep methods need) |
| User-impact ceiling | Bf16 50-step is the minority "experiments" variant; DMD-merged is the recommended user path |
| Implementation home | Adding `DPM-Solver++(2M)` means a new scheduler class in `mlx-arsenal` (cleanest) OR private to LongCat. Either way `scheduler.step()` API needs to match `mlx_arsenal.FlowMatchEulerDiscreteScheduler` |
| Effort | Moderate — 100-200 LOC for the scheduler class + integration + tests, plus the upstream decision (mlx-arsenal PR vs vendor) |
| **Verdict** | **Low priority.** Helps only the minority bf16 50-step path. Already enumerated in `scheduler-and-compile-evaluation.md`; conclusion stands. |

### PR #6 — `memory_mode` tower relay + tiled VAE decode

| | Detail |
|---|---|
| Architectural fit | **High structural similarity.** LongCat has FOUR disjoint phases (umT5-XXL text encode → Whisper audio encode → DiT denoise → AutoencoderKLWan decode), one more than Lance. Each phase's weights can be freed when the next phase starts. |
| Pattern transfer | The shed-cascade IDEA transfers directly. Lance's specific method names (`prefill_prefix`, `free_und_tower`, `materialize_gen_tower`) need LongCat equivalents (`encode_text_and_free_umt5`, `encode_audio_and_free_whisper`, `materialize_dit`, `free_dit`, `materialize_vae`) |
| The `resolve_memory_mode` budget helper | Lance's `resolve_memory_mode(...)` (querying `mx.device_info().max_recommended_working_set_size`) transfers verbatim modulo the threshold (Lance picks ~18 GiB; LongCat needs its own threshold) |
| Effort | **Substantial** — 400-800 LOC across `pipeline_mlx.py`, the model loader, and tests. Bigger lift than Lance's because we have 4 phases instead of 3, plus an audio encoder which Lance doesn't have |
| **Verdict** | **Highest leverage but biggest effort.** If anyone is fitting LongCat into 32 GB or smaller Macs, this is the unlock. Plan a focused 1-2 day session for the port; consider doing it after the Swift S4.x work completes. |

### PR #7 — Lossless streaming VAE decode

| | Detail |
|---|---|
| Algorithm fit | **Direct.** Wan2.2 VAE is causal in time across the entire family. Lance's algorithm — temporal causal-cache streaming (decode one latent frame at a time while carrying CACHE_T=2 boundary frames) + spatial halo-tile crop (≥ measured receptive field, then crop) — works identically against our `AutoencoderKLWan` |
| Code fit | **Not direct.** Symbol mismatch between Lance's `Wan22VAEDecoder` and our `AutoencoderKLWan`: |
| | • `ResidualBlock` → `WanResidualBlock` (rename) |
| | • `Up_ResidualBlock` → `WanUpBlock` (rename + structure check) |
| | • `DupUp3D` → not present in our port (Lance-specific dual-up 3D shortcut) |
| | • `Wan22VAEDecoder` → `WanDecoder3d` (rename) |
| | • `_unpatchify` → may not exist or may be inlined |
| | • Tensor layout: Lance NHWC `(B, T, H, W, C)` → ours NCHWD `(B, C, T, H, W)` — every axis index in the streaming code needs translation |
| Bit-identity test pattern | Lance's `tests/test_decode_stream.py` (280 LOC, 50 cases, weights-free, max\|Δ\|=0) is the gold-standard test pattern. Direct mirror should work — build a tiny random-init `AutoencoderKLWan` and assert `decode_streaming(dec, z) == dec(z)` bit-exact. |
| Effort (**revised 2026-06-05 — second correction same day**) | **Near-zero. The work is already done.** Both `Resample.upsample3d` (with `"Rep"` sentinel) AND the top-level orchestrator in `AutoencoderKLWan.decode()` (per-frame loop, `feat_cache` threading, output concatenation along temporal axis) are already in place in `autoencoder_kl_wan.py:716-735`. The streaming pattern is the **default** decode path, not an opt-in. `encode()` (line 681) is also already streaming. The earlier "missing orchestrator" claim was wrong — I read the inner blocks (Resample) but didn't audit the top-level entry point until writing the longcat-video-mlx deferred-port handoff. **Residual scope:** one optional self-consistency bit-identity test (~30 min, weights-free, mirror of `lance-mlx/tests/test_decode_stream.py`). See [`longcat-video-mlx/docs/development/streaming-vae-decode-deferred-port-handoff.md`](../../../longcat-video-mlx/docs/development/streaming-vae-decode-deferred-port-handoff.md) for the full audit. |
| Win on LongCat | Decode peak flat in frame count + halo-tile spatial = the long-video paths benefit most. Concretely the same Lance numbers (256² × 121f: 15.4 GB whole → 8.0 GB streaming) should hold on our VAE modulo channel count (16 vs 48 ratio modifies peaks proportionally) |
| **Verdict** | **Defer to end** (per user direction). When picked up, cheaper than originally estimated. |

## Recommended sequence (revised 2026-06-05)

The second-pass code read reshuffled the priority. The headline change:
**mlx-video stock `wan_2/vae.py` is the highest-leverage port target**
because it benefits phantom-wan + bernini-r simultaneously (both ride
the same upstream) and is upstreamable to `Blaizzy/mlx-video`. LongCat's
private fork is the *cheapest* port (because the `"Rep"` sentinel is
already wired) but lowest leverage (single port). User direction:
**defer LongCat regardless**.

1. **Update the `mx.compile` recommendation in
   `scheduler-and-compile-evaluation.md`** — done 2026-06-05; Lance
   empirical refutation makes the speculative LongCat estimate stale.
2. **PR #7 → mlx-video stock `wan_2/vae.py`** — moderate effort
   (~4-6 hr), two beneficiaries (phantom-wan + bernini-r), upstreamable.
   See [phantom-wan-mlx handoff](../../../phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md)
   for full work-breakdown.
3. **PR #7 → LongCat `AutoencoderKLWan`** — **AUDIT COMPLETE 2026-06-05:
   no port needed.** Both the `"Rep"` sentinel (in `Resample.upsample3d`)
   and the streaming orchestrator (in `AutoencoderKLWan.decode()` /
   `.encode()`) are already in place as the default code path — the
   original LongCat VAE port author implemented streaming-by-default
   when matching the diffusers reference. Residual work: one optional
   self-consistency bit-identity test (~30 min). Full audit + pickup
   plan in
   [`longcat-video-mlx/docs/development/streaming-vae-decode-deferred-port-handoff.md`](../../../longcat-video-mlx/docs/development/streaming-vae-decode-deferred-port-handoff.md).
4. **PR #6 → LongCat memory_mode** — high leverage but a 1-2 day
   focused session. Worth doing once the Swift port S4.x work
   stabilizes so we don't ship two changing things at once.
5. **Un-fork LongCat onto mlx-video stock?** — investigated; **not
   directly possible**. The channel arithmetic at `dim_mult=[1,2,4,4]`
   differs structurally between mlx-video's halve-on-input pattern and
   diffusers/Meituan's keep-and-project pattern (see
   `notes/vae-schema-mismatch.md`). The *opposite* direction works:
   upstream our `AutoencoderKLWan` to mlx-video as a sibling class or
   `schema="diffusers"` flag on `WanVAE`. Effort ~4-6 hr, benefit
   ceiling = single port (us) + any future ports needing the diffusers
   schema. **Defer indefinitely** unless a second consumer surfaces.
6. **PR #4 DPM scheduler** — low priority; only helps bf16 50-step
   minority variant. Defer indefinitely unless a user case appears
   that needs faster bf16 base path generation.

## Concrete file map for PR #7 port (if/when we do it)

If we do the streaming-decode port, the file layout would be:

```
longcat_video_avatar/models/
  ├── autoencoder_kl_wan.py        (existing)
  └── autoencoder_kl_wan_stream.py (new — mirror of vae_stream.py)
tests/smoke/
  └── test_decode_stream.py        (new — mirror of Lance's test, weights-free, ~50 cases)
longcat_video_avatar/pipeline_mlx.py
  └── add `lossless_decode: bool = True` kwarg to generate()
  └── dispatch `vae.decode(z)` vs `decode_streaming(vae.decoder, z, ...)` on flag
```

Lance reference files (for the algorithm + tests):
- `xocialize/lance-mlx/src/lance_mlx/model/vae_stream.py` (426 LOC)
- `xocialize/lance-mlx/tests/test_decode_stream.py` (280 LOC)
- `xocialize/lance-mlx/results/decode_lossless/DECODE_FOOTPRINT_SWEEP.md`
  (the measurement methodology — useful for our own validation)

## What doesn't transfer

- **Lance's `lance-mlx-studio` mx.compile wrapper** — referenced in old
  Lance README, then deleted when the external repo 404'd. The Lance
  team's empirical investigation showed mx.compile is a regression at
  this model scale; we shouldn't pursue it here either. See
  `scheduler-and-compile-evaluation.md` for the data and reasoning.

## Cross-reference

Bernini-R uses **mlx-video stock** `wan_2.vae.WanVAE` (verified
2026-06-05 via grep on `bernini_r_mlx/sampling.py`), **NOT** a private
fork. So a streaming-decode port at the mlx-video upstream level
benefits bernini-r and phantom-wan with one effort. PR #6 memory_mode
still requires a per-port port (bernini-r has 3 phases: umT5 → DiT →
VAE) but is cheaper than LongCat's 4-phase variant.

For the full corrected sequencing and the streaming-decode
work-breakdown, see
[phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md](../../../phantom-wan-mlx/docs/development/streaming-vae-decode-port-handoff.md)
— that doc supersedes the original sequencing recommendation in this
file.
