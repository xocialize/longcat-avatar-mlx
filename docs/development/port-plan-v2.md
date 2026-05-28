# LongCat-Video-Avatar-1.5 → Apple MLX Port: Technical Feasibility Report for RosettaCast

*Version 2 — port-first sequencing, anime validation deferred to bonus stage.*

## TL;DR
- **Yes, this is worth porting — but as a Wan-2.1-family video DiT, not as a "talking head" model.** LongCat-Video-Avatar-1.5 is a 13.6B-parameter Diffusion Transformer built on the LongCat-Video foundation, sharing the Wan 2.1 VAE and umT5-XXL text encoder, with a Whisper-Large-v3 audio side-tower and a 2.5 GB DMD distillation LoRA for 8-step inference. Licensed MIT (weights + code), so commercial use in RosettaCast is unambiguous.
- **Anime/stylized is a first-class supported scenario in v1.5, not a side demo.** The official v1.5 project page ships a dedicated "Animation" carousel of seven MP4 clips and a head-to-head "3D Animation" comparison vs. HeyGen / Kling Avatar 2.0 / OmniHuman-1.5. The model card explicitly lists "anime, animals, and complex real-world conditions" as a key feature, and the published EvalTalker benchmark deliberately spans two visual styles (Realistic/Animated) across 508 image-audio source pairs, with 770 crowdsourced evaluators producing 13,240 1-to-5 human-likeness judgments plus a 10-domain-expert objective track across Physical Rationality, Harmony, Temporal Stability, and Identity Consistency. This is a meaningful differentiator vs. EchoMimic/Hallo3/Sonic/EMO, which are largely real-portrait models — but it's a bonus, not a gate, for the port itself.
- **MLX home: `Blaizzy/mlx-video`, with weights published under `mlx-community/LongCat-Video-Avatar-1.5-{bf16,int8,q4}`.** mlx-video already ships Wan2.1/Wan2.2 (same VAE, same umT5-XXL, same flow-matching scheduler family) and an audio-to-video pipeline (LTX-2 A2V), so ~70% of the primitives exist. The novel work is the audio cross-attention adapter, Whisper-Large-v3 encoder in MLX (mlx-audio already has Whisper), and the LongCat-specific 3D block-sparse attention. Realistic effort: **~3-4 weeks of focused Python/MLX work** to get a usable bf16/int4 inference path on a 128 GB M5 Max, validated against the standard real-portrait case. Anime validation deferred to a bonus stage after the port lands.

## Key Findings

**Architecture is Wan-2.1-shaped, not exotic.** LongCat-Video (the base) is a 13.6B-parameter dense DiT with 48 layers, hidden size 4096, FFN 16384, 32 attention heads, and an AdaLN embedding of 512. Each block is single-stream: 3D self-attention → cross-attention (for umT5 text) → SwiGLU FFN, with 3D RoPE, RMSNorm + QKNorm, AdaLN-Zero modulation. VAE is the Wan 2.1 `AutoencoderKLWan` (4×8×8 latent compression) plus a 1×2×2 patchify (overall 4×16×16). Text encoder is umT5-XXL bilingual (zh/en). Scheduler is `FlowMatchEulerDiscreteScheduler`. This is the same family mlx-video already supports for Wan2.1/Wan2.2.

**Avatar-1.5 = base DiT + audio adapter + LoRA, not a separate transformer.** The HuggingFace repo (74.9 GB total) ships: `base_model/` (six safetensors shards, 31.7 GB bf16 — the modified `LongCatVideoAvatarTransformer3DModel`), `base_model_int8/` (INT8 DiT for low-VRAM inference, v1.5 only), `lora/dmd_lora.safetensors` (2.52 GB — the DMD2 step-distillation LoRA, required for v1.5's 8-step sampler), `whisper-large-v3/` (the audio encoder shipped in the repo), `vocal_separator/` (MelBand RoFormer for cleaning vocals from BGM), and `scheduler/`. Notably the umT5 text encoder and Wan VAE are **not** in this repo — the inference scripts load them from the sibling `meituan-longcat/LongCat-Video` repo. **Architectural innovations in the Avatar variant**: Reference Skip Attention (prevents identity "copy-paste"), Disentangled Unconditional Guidance (decouples speech from body motion so silence ≠ frozen), and Cross-Chunk Latent Stitching (skips VAE re-encode between chunks to prevent drift in long sequences). The Avatar adapter operates at the pipeline layer without modifying the underlying transformer architecture (per DeepWiki's reading of the repo).

**v1.0 → v1.5 deltas, ranked by importance for the port.** (1) Audio encoder upgraded from `chinese-wav2vec2-base` (English-leaning) to Whisper-Large-v3 (680k-hour multilingual) — this is the headline change and directly improves lip-sync on Japanese, English, and accented speech. (2) DMD2 step distillation collapses 50 sampler steps → 8 NFE. Meituan's official v1.5 release announcement (aibase.com/news/28241, May 22 2026) states "approximately 15 times faster inference efficiency, with generating a 10-second video taking about 1 minute" — hardware unspecified. (3) INT8 DiT released. (4) Explicit training on stylized data so v1.5 generalizes to anime/animals/3D animation, with a new "Animation" demo carousel and a 3D-anime commercial comparison on the project page (neither existed on v1.0). The base DiT weights themselves were re-trained, not just adapter-swapped.

**License is genuinely permissive.** Both model weights and inference code are released under MIT (LICENSE file at the HF repo root and GitHub repo root). The model card states: *"The model weights are released under the MIT License. … This license does not grant any rights to use Meituan trademarks or patents."* That's about the cleanest commercial-OK license you can get for a 13B video model. Compare to Wan-AI (Apache 2.0, also fine) and Hunyuan-Video (custom Tencent community license with usage restrictions). **Verdict for MVS Collective: zero license blockers** — you can ship MLX weights under mlx-community, build RosettaCast on top, and charge users.

**Anime viability — context for the bonus stage, not a gate.** Four pieces of evidence converge:
1. *Official demos* — meigen-ai.github.io/LongCat-Video-Avatar-1.5-Page hosts seven dedicated 2D-animation demos under `videos/animations/` and a four-way commercial comparison under `videos/comparison_commerical/3d_animation/` (LongCat vs. HeyGen vs. Kling vs. OmniHuman). The HF model card explicitly enumerates "anime" as a stylized domain.
2. *Benchmark design* — the EvalTalker harness is deliberately split across 6 application scenarios, 2 languages, and 2 visual styles (Realistic/Animated) across 508 image-audio pairs.
3. *Architectural* — Because the audio path is a separate cross-attention adapter on top of a general-purpose video DiT, the model isn't anchored to real-face geometry priors the way Sonic (SVD-based) or EchoMimic (landmark-driven) are. The OmniSync paper (arXiv 2505.21448) specifically says: *"methods such as EchoMimic, Hallo3, and Sonic face challenges in maintaining identity, which may result in less realistic outcomes…This holds true even when applied to out-of-distribution subjects, such as non-human or highly stylized characters."* LongCat-Video-Avatar-1.5 was designed to escape that failure mode.
4. *Published headline scores* — Meituan's official launch announcement reports EvalTalker subjective single-person score of 3.336, "outperforms Kling Avatar 2.0 by 65.9%, OmniHuman-1.5 by 61.1%, and HeyGen by 54.3%", and multi-person score of 2.730 "greatly surpassing InfiniteTalk (2.339)", along with subject-deformation rate 23.1%, frame-skipping rate 0.8%, and lip-sync issue rate 29.8%. These aggregate Realistic and Animated together; **a separate anime-only number is not published**. That's what Stage 4 validates empirically.

**Comparable model baseline for stylized inputs is poor — LongCat is well-positioned.** EMO, Hallo, Hallo2, Hallo3, Sonic, V-Express, Loopy, and EchoMimic are all real-portrait-trained; Sonic uses SVD and degrades on stylized inputs; Hallo3 and EchoMimic show identity drift on out-of-distribution subjects (per the OmniSync paper's qualitative comparison). The closest direct competitors that explicitly handle anime are OmniHuman-1.5 (ByteDance, closed-source API) and InfiniteTalk (also based on Wan 2.1, open-source) — Meituan benchmarks LongCat-Video-Avatar-1.5 above both. If anime stylization works as advertised, LongCat is the only open-weight option in the category.

**Inputs / outputs / sampling.** Inputs are: a reference image (for ATI2V/AI2V), one or two audio clips (single or dual-stream conversation, with `para` merge mode or `add` concatenation), and a text prompt. Outputs: 480P or 720P video at 30 fps (default 93 frames ≈ 3.1 s, with `--num_segments=N` chaining for longer continuation; Cross-Chunk Latent Stitching enables 5-minute+ sequences without VAE-decode drift). Sampling: flow matching with `FlowMatchEulerDiscreteScheduler`, 8 NFE with `--use_distill --model_type avatar-v1.5`, audio CFG 3-5, text CFG separate. Reference image index (`--ref_img_index`, default 10) and mask frame range (`--mask_frame_range`, default 3) control repetitive motion. This is **strictly reference-image-to-video**, not video-to-video relighting — you cannot feed an existing anime clip and have it re-lip-sync directly. For RosettaCast (a Stage 4 concern), the integration pattern is therefore: extract a representative anime keyframe per shot, drive it with the English dub audio + a scene-describing text prompt, then composite back into the source video where mouth-region replacement is desired (a non-trivial post step).

## MLX port — recommended target package: `Blaizzy/mlx-video` (publishing weights under `mlx-community`)

| Option | Fit | Reason |
|---|---|---|
| `Blaizzy/mlx-video` | ★★★★★ | Already has Wan2.1, Wan2.2, LTX-2 (same VAE family, same umT5, same flow matching, audio-to-video plumbing already exists in LTX-2 A2V). Prince Canuma actively merges new architectures. This is the obvious home. |
| `Blaizzy/mlx-audio` | ★★ | Has the Whisper encoder and already hosts `mlx-community/LongCat-AudioDiT-1B-bf16` (Meituan's audio DiT). Useful as a *dependency* for the Whisper-Large-v3 audio embedding step, but mlx-audio's primary scope is TTS/STT, not video. |
| `Blaizzy/mlx-vlm` | ★ | Wrong abstraction — mlx-vlm is for VLMs (input-side image/audio/video understanding), not generative video diffusion. |
| Standalone repo (`longcat-avatar-mlx`) | ★★★ | Viable interim path before merging into mlx-video; mirrors what `dgrauet/ltx-2-mlx` and `osama-ata/Wan2.2-mlx` did. Reasonable "stopping point" for a Swift port handoff. |

**Recommended publish strategy:** Build the Python port as a `longcat_video_avatar` module **inside or as a fork of mlx-video**, then publish weight repos: `mlx-community/LongCat-Video-Avatar-1.5-bf16`, `-int8`, and `-q4`. This matches the precedent (mlx-video's `mlx_video.wan_2.generate`, `mlx_video.ltx_2.generate` namespaces). If Prince Canuma is responsive, upstream a PR.

**MLX primitives needed — most already exist.** Reusable from mlx-video's Wan2.1/Wan2.2 implementation: AutoencoderKLWan (3D causal video VAE), umT5-XXL encoder, FlowMatchEulerDiscreteScheduler, 3D RoPE, flow-matching CFG, LoRA loader (for the DMD LoRA). Reusable from mlx-audio: Whisper-large-v3 encoder. **What needs to be written from scratch:**
1. The `LongCatVideoAvatarTransformer3DModel` block — mostly Wan2.1-shaped but with the Avatar's audio cross-attention insertion (the Whisper features must be projected and fed via an additional cross-attention path or AdaLN modulation in each DiT block — the exact wiring needs to be read from `longcat_video/` PyTorch source).
2. Reference Skip Attention layer.
3. Disentangled Unconditional Guidance at the sampler level (modify the standard CFG combiner).
4. Cross-Chunk Latent Stitching helper for the continuation path.
5. Block Sparse Attention — **defer to v2.** Dense attention via `mx.fast.scaled_dot_product_attention` at 480P will be acceptable for the port. Revisit only if 720P performance is unacceptable.

**Memory footprint on Apple Silicon.** Disk: 31.7 GB bf16 DiT + ~10 GB umT5-XXL + ~2 GB Wan VAE + ~3 GB Whisper-large-v3 + 2.5 GB DMD LoRA ≈ **50 GB on-disk for the full bf16 stack**. Inference memory is dominated by DiT activations + KV cache for the 3D self-attention. The base LongCat-Video reportedly OOMs on a 48 GB Ada A6000 at the refinement stage even with T5/VAE CPU-offloaded (per GitHub issue meituan-longcat/LongCat-Video#7). fal.ai's official Longcat Video Prompt Guide (by Brad Rose, last updated 12/17/2025) states verbatim: *"For local deployment, Longcat Video requires approximately 80GB of VRAM on an NVIDIA GPU system"* — but that figure is for the **base** model, not the Avatar-1.5 INT8 + 8-step distillation path. **My estimate for Apple Silicon:** the 128 GB M5 Max should comfortably run bf16 at 480P with no offloading (unified memory means VAE/text-encoder offload is free), and likely 720P with sequential offload. A 64 GB Mac will need int4/int8 weights and is plausible for 480P only. 32 GB Macs and below are not realistic targets.

**Inference speed expectations.** Meituan's "10-second video in ~1 minute" claim is on unspecified hardware (the official announcement names no GPU). mlx-video's Wan2.2-T2V-A14B on an M2 Ultra takes single-digit minutes for 5 seconds at 480p with 4-step Lightning LoRA; LongCat is similar size and structure. With 8-step DMD distillation, **expect roughly 3-8 minutes per 5-second 480P clip on an M5 Max 128 GB**, slower at 720P. Block Sparse Attention on Metal would help but is a research-grade port; dense attention is the pragmatic v1 path.

**Quantization viability.** bf16 (Apple Silicon-native), Q8 (mlx 8-bit), Q4 (mlx 4-bit) all proven for Wan2.x in mlx-video. Meituan ships an INT8 DiT they trained — those weights can likely be remapped to MLX 8-bit if the scales align. FP8 not natively supported on Apple Silicon yet. Recommended starter quants: `bf16` (reference / quality), `int8` (use the Meituan-provided INT8 weights directly where possible), `q4` (aggressive, for 64 GB Macs).

**Porting gotchas specific to this model.**
1. **No FlashAttention on Metal** — Meituan's default `flash_attn==2.7.4.post1` and Block Sparse Attention Triton kernel must be replaced with `mx.fast.scaled_dot_product_attention` (dense). Quality should match; speed will be 3-5× slower at 720P. The Wan2.x mlx-video implementation already does this swap.
2. **Multi-GPU `context_parallel_size=2` is baked into all example commands.** Meituan's reference recipe is `torchrun --nproc_per_node=2`; single-GPU works (the same scripts) but requires removing the context-parallel sharding. mlx-video doesn't have distributed inference, so this becomes a single-device run on unified memory.
3. **3D RoPE** — Wan2.x port in mlx-video already handles this; reuse.
4. **Two-repo dependency** — inference scripts load text_encoder/vae/scheduler from `LongCat-Video` (separate HF repo) and DiT/LoRA/Whisper from `LongCat-Video-Avatar-1.5`. The MLX port should consolidate these into a single mlx-community repo for usability.
5. **Audio cross-attention wiring is undocumented in the model card** — the Avatar tech report (the 17.6 MB PDF in `assets/`) likely has it; otherwise you must read `LongCatVideoAvatarTransformer3DModel` in `longcat_video/` PyTorch source to find where the audio embeddings are injected (typical pattern: per-block AdaLN modulation + an extra cross-attention to audio tokens after the text cross-attention).
6. **Whisper-large-v3 audio feature alignment** — Avatar's audio path consumes Whisper encoder hidden states, not Whisper's decoded text. The audio encoder is shipped as model weights in the v1.5 repo (under `whisper-large-v3/`), so faithful reproduction is straightforward; mlx-audio already has Whisper. You'll need a small adapter MLP whose weights live inside the DiT checkpoint.

**Existing MLX video diffusion precedents for reference.**
- `Blaizzy/mlx-video` (LTX-2, Wan2.1, Wan2.2) — same architectural family, has audio-to-video already (LTX-2 A2V), has LoRA support. **This is the direct template.**
- `osama-ata/Wan2.2-mlx` and `antonpetrovmain/Wan2.2-mlx` — pure MLX Wan2.2 ports.
- `dgrauet/ltx-2-mlx` — 19B parameter video model in MLX with full T2V/I2V/A2V/Retake/Extend pipelines.
- `notapalindrome/ltx2-mlx-av` — joint audio-video MLX weights.
- `mlx-community/LongCat-AudioDiT-1B-bf16` (in mlx-audio) — **Meituan's own audio DiT is already in mlx-community**, which is helpful precedent and contact.

## Recommended next steps and decision gates

**Stage 1 — MLX Python port. ~3-4 weeks.**
1. Fork `Blaizzy/mlx-video`; add a `mlx_video/longcat_avatar/` module mirroring the existing `wan_2/` and `ltx_2/` modules.
2. Port the DiT block (Wan-shaped + audio cross-attention), Reference Skip Attention, Disentangled UCG, Cross-Chunk Latent Stitching. Reuse VAE/umT5/scheduler/Whisper from existing mlx-video/mlx-audio. Defer Block Sparse Attention to v2 — dense attention at 480P will be acceptable on the M5 Max.
3. Validate against the model's primary case: real-portrait head-and-shoulders + clean English speech, 5s at 480P. This is the case Meituan trained against and benchmarks against; it's also what any third-party reviewer will compare you to.

**Decision gate:** Visual quality within ~10% of PyTorch reference on real-portrait inputs at 480P, and inference completes in <20 minutes for a 5-second clip on a 128 GB M5 Max.

**Stage 2 — mlx-community publish (the "stopping point"). ~1 week.**
Publish weight repos: `mlx-community/LongCat-Video-Avatar-1.5-bf16` (consolidated: DiT + LoRA + audio adapter + Whisper-large-v3 + umT5 + VAE + scheduler), then `-int8` (re-mapping Meituan's shipped INT8 DiT where scales permit) and `-q4`. README with inference example, parity notes vs. PyTorch reference, and the consolidated-repo layout rationale. If Prince Canuma is responsive, upstream the `longcat_avatar/` module as a PR to mlx-video.

**Stage 3 — Swift / MLXEngine port. Deferred.**
Only after Python MLX inference works end-to-end. The Swift port is straightforward once the Python reference exists (port shape for shape), and `mlx-swift-examples` already has Wan-family precedent via DrawThings's open Swift implementations. Lands inside the MLXEngine umbrella alongside the other xocialize-code engine packages.

**Stage 4 (bonus) — Anime validation + RosettaCast integration. Scoped after Stages 1-2.**
Run 5 representative anime inputs against the working MLX port:
1. Clean anime keyframe (single character, head-and-shoulders, neutral expression) + 5s English speech audio.
2. The same character in a complex pose (3/4 view, hand-near-face).
3. A 3D-rendered animated character (since the v1.5 page has a 3D anime comparison).
4. A cat or stylized animal character (the original inspiration case).
5. A multi-character scene with two anime portraits + dual audio.

**Honesty bar:** if at least 3/5 produce usable lip motion with preserved identity and no catastrophic geometry collapse, anime is real and worth building the RosettaCast compositing layer for. If 0-2/5 work, the port still has standalone value for real-portrait avatar work and the time wasn't wasted.

The RosettaCast integration (anime video → shot detection → per-shot keyframe → audio alignment → LongCat-MLX inference → mouth-region compositing back to source frames) is contingent on this gate. The compositing step — replacing only the mouth region of the source frames with the LongCat output, keeping all other content from the original anime — is critical for production quality and **outside LongCat's scope**. Consider re-using ProPainter or a per-frame face-region blend.

## Open questions

### Stage 1 / port-specific (resolve before or during the port)

1. **What is the exact audio-injection wiring in the DiT?** Cross-attention layer per block? AdaLN modulation? Single injection or per-block? Read `LongCatVideoAvatarTransformer3DModel` source and the v1.5 tech report PDF (17.6 MB on GitHub) to confirm before starting the port. This is the #1 blocker.
2. **Can the INT8 weights be loaded into MLX 8-bit directly, or must you re-quantize from bf16?** Check whether Meituan's INT8 is per-tensor symmetric (MLX-compatible) or uses some custom format. Saves a quantization round-trip if directly mappable.
3. **Block Sparse Attention — defer or port?** Default: defer to v2; dense attention at 480P should be fast enough on M5 Max. Revisit only if 720P performance is unacceptable after Stage 1 lands.

### Stage 4 / anime + dubbing-specific (deferred validation)

4. **Does anime lip-sync actually look right?** The model card says it does, the project page shows curated demos, but no independent third-party anime test results surfaced in any forum searched (no Reddit r/StableDiffusion threads on anime-input LongCat-Avatar). The 5-input test in Stage 4 is the answer.
5. **Does it cope with side-view / non-frontal anime poses?** Talking-head models traditionally degrade hard off-axis, and anime scenes are full of profile shots.
6. **Does Whisper-Large-v3 handle Japanese-source-with-English-dub audio reasonably?** For dubbing, the input audio is the English dub — Whisper-v3 is multilingual so this should work, but worth confirming with accented English voice samples representative of dub VAs.
7. **How does the model behave when the reference image and the target speech are mismatched in style?** Important for RosettaCast (Japanese-source character, English audio).

## Caveats

- **VRAM numbers are partial.** Meituan does not publish an official "minimum VRAM" for v1.5 in the model card, project page, or English/Chinese launch announcements. The "approximately 80 GB" figure (fal.ai's prompt guide) is for the **base** LongCat-Video, not Avatar-1.5, and predates the Avatar v1.5 INT8 + 8-step distillation release. Community ComfyUI/Kijai FP8 ports of v1.0 run on 16-24 GB cards with heavy block-swapping (slow); a community RTX 4090D report cites ~60 s/step at 480P 10-step 105-frame on v1.0 (no v1.5-specific number available yet).
- **Speed projections for Apple Silicon are extrapolated** from mlx-video's Wan2.2 throughput, not measured directly on LongCat. Real numbers will require running Stage 1.
- **The "Animation" demos on the official page are curated by Meituan** and almost certainly cherry-picked. Expect real-world quality to be lower than the marketing reel.
- **Cross-Chunk Latent Stitching is the most novel piece** and is the highest-risk porting item. If you skip it, you lose long-video stability (color drift, identity drift after ~10s). It's needed for any >10s clip, anime or not, so it stays in Stage 1.
- **The Diffusers code snippet on the HF model card is wrong** — it's a generic SD-style `DiffusionPipeline.from_pretrained("…").images[0]` template that does not match the actual model. The real inference path is the LongCat-Video repo's `run_demo_avatar_single_audio_to_video.py`. Don't trust the auto-generated snippet.
- **v1.5 Technical Report PDF was not extractable to plain text** via tooling and was not read end-to-end for this analysis. Download from `github.com/meituan-longcat/LongCat-Video/blob/main/assets/LongCat-Video-Avatar-1.5-Tech-Report.pdf` and read it directly before starting Stage 1 — it likely contains hyperparameter tables and the audio-injection wiring needed for the port.
- **Reference-image-to-video only.** This model does **not** do video-to-video re-lip-sync. For an anime dubbing pipeline where the source video already exists, you will need a separate compositing/inpainting step to merge the LongCat output back into the original frames — that's an open engineering problem in Stage 4, outside the scope of the MLX port.
- **Published EvalTalker scores aggregate Realistic and Animated styles into single numbers.** Meituan's headline claims (single-person 3.336, multi-person 2.730, beating Kling Avatar 2.0 by 65.9%, OmniHuman-1.5 by 61.1%, HeyGen by 54.3%, and InfiniteTalk on multi-person 2.730 vs. 2.339) do not separate anime from realistic performance, so the anime-specific quality remains unconfirmed by any party other than Meituan's curated demo page.

## Changelog

**v2 (this version)** — Restructured staging: dropped pre-port CUDA validation; sequence is now (1) MLX Python port validated against real-portrait baseline, (2) mlx-community publish as the explicit stopping point, (3) Swift/MLXEngine port deferred, (4) anime validation + RosettaCast integration as a bonus stage gated on a 5-input honesty test. Effort estimate tightened from 3-6 weeks to ~3-4 weeks given the narrower Stage 1 scope. Block Sparse Attention explicitly deferred to v2.

**v1** — Initial report with anime-first validation as Stage 0.
