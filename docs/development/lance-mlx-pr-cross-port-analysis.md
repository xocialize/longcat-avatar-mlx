# Cross-port analysis: lance-mlx PRs → LongCat

**Date:** 2026-06-05
**Triggered by:** Three substantive optimization PRs landed on
`xocialize/lance-mlx` recently. This doc enumerates what would
transfer to LongCat, what the effort actually looks like, and which
items rank highest if we invest the hours.

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
| Effort | **Moderate.** 400-500 LOC stream module + 250 LOC test, careful axis-index translation. Maybe 4-8 hours including parity validation. |
| Win on LongCat | Decode peak flat in frame count + halo-tile spatial = the long-video paths benefit most. Concretely the same Lance numbers (256² × 121f: 15.4 GB whole → 8.0 GB streaming) should hold on our VAE modulo channel count (16 vs 48 ratio modifies peaks proportionally) |
| **Verdict** | **Best single-PR win for the effort.** Recommended next port if you allocate optimization time. |

## Recommended sequence (refined)

1. **Update the `mx.compile` recommendation in
   `scheduler-and-compile-evaluation.md`** — done 2026-06-05; Lance
   empirical refutation makes the speculative LongCat estimate stale.
2. **PR #7 → LongCat lossless streaming decode** — moderate effort,
   biggest single-PR win, well-bounded scope, has a gold-standard test
   pattern to mirror.
3. **PR #6 → LongCat memory_mode** — high leverage but a 1-2 day
   focused session. Worth doing once the Swift port S4.x work
   stabilizes so we don't ship two changing things at once.
4. **PR #4 DPM scheduler** — low priority; only helps bf16 50-step
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

A parallel analysis for `bernini-r-mlx` (Wan2.2-A14B renderer) is
likely worth writing if anyone touches that port — Bernini-R uses
`AutoencoderKLWan` too but with a different channel count, and uses
`FlowUniPCScheduler` (already a sophisticated multistep solver) so PR
#4 doesn't transfer. PR #7 and PR #6 transfer similarly with
similar effort to LongCat.
