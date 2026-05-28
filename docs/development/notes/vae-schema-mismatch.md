# VAE schema mismatch — mlx-video's Wan VAE is NOT a drop-in for Meituan

**Discovered:** during Stage 1.1 parity test (test_vae_decode_parity).

## Symptom

Loading Meituan's `vae/diffusion_pytorch_model.safetensors` (194 tensors,
diffusers 0.38 `AutoencoderKLWan` schema) into the MLX module copied from
mlx-video's `wan_2/vae.py` fails: `ValueError: Module does not have parameter
named "conv_in"`.

## Root cause

Two distinct structural divergences between mlx-video's MLX port and
diffusers-0.38 / Meituan's checkpoint:

### 1. Module hierarchy

| diffusers 0.38 (Meituan ships) | mlx-video |
|---|---|
| `encoder.conv_in` | `encoder.conv1` |
| `encoder.conv_out` | `encoder.head.2` |
| `encoder.norm_out` | `encoder.head.0` |
| `encoder.down_blocks[B].resnets[R].{conv1,conv2,norm1,norm2}` | `encoder.downsamples[idx].residual.{2,6,0,3}` |
| `encoder.down_blocks[B].downsamplers[0].{resample.1,time_conv}` | `encoder.downsamples[idx].{resample.1,time_conv}` (flat) |
| `encoder.mid_block.resnets[0,1]` + `attentions[0]` | `encoder.middle.{0,1,2}` |
| `quant_conv` / `post_quant_conv` | `conv1` / `conv2` |
| (decoder mirror) | (decoder mirror) |

Renames are mechanical and would fix this layer.

### 2. Channel arithmetic (the harder problem)

For `dim_mult=[1, 2, 4, 4]` (Meituan's setting), the decoder block channel pattern is:

| Stage | diffusers 0.38 / Meituan | mlx-video |
|---|---|---|
| 0 | 384→384 ×3, upsample 384→192 | 384→384 ×3, upsample 384→192 ✓ |
| 1 | **192→384**, 384→384, 384→384, upsample 384→192 (with conv_shortcut) | 192→192 ×3, upsample 192→96 ✗ |
| 2 | 192→192 ×3, upsample 192→96 | 96→96 ×3, upsample 96→48 ✗ |
| 3 | 96→96 ×3 | 48→96, 96→96, 96→96 ✗ |

The PT shape `decoder.up_blocks.1.resnets.0.conv1.weight: (384, 192, 3, 3, 3)`
with companion `conv_shortcut: (384, 192, 1, 1, 1)` confirms diffusers'
canonical pattern.

mlx-video's halve-on-input pattern is **architecturally incompatible** — the
intermediate tensor shapes literally differ, so you can't reconcile via key
renaming or tensor reshape. It must target a different Wan VAE checkpoint
(perhaps an earlier Wan-AI release with a different `dim_mult` interpretation,
or a custom Blaizzy variant).

## What this means for Stage 1.1

The port needs to be **re-implemented to match the diffusers 0.38
`AutoencoderKLWan` reference**, not adapted from mlx-video. Estimated
~500–700 lines of MLX, modeled directly after
`diffusers/models/autoencoders/autoencoder_kl_wan.py` (1400 lines PT).

Module structure to produce (top-level prefixes that must match PT keys):

```
quant_conv, post_quant_conv                 # top-level 1×1×1 CausalConv3d
encoder.conv_in, encoder.conv_out, encoder.norm_out
encoder.down_blocks[B].resnets[R]           # B in 0..3, R in 0..1 (num_res_blocks)
encoder.down_blocks[B].downsamplers[0]      # B in 0..2 (no downsampler on last)
encoder.mid_block.resnets[0,1]
encoder.mid_block.attentions[0]
decoder.conv_in, decoder.conv_out, decoder.norm_out
decoder.up_blocks[B].resnets[R]             # B in 0..3, R in 0..2 (num_res_blocks + 1)
decoder.up_blocks[B].upsamplers[0]          # B in 0..2
decoder.mid_block.resnets[0,1]
decoder.mid_block.attentions[0]
```

Inside each ResNet block (matching PT names directly):

```
norm1.gamma, conv1.weight, conv1.bias, norm2.gamma, conv2.weight, conv2.bias
conv_shortcut.weight, conv_shortcut.bias   # only when in_dim != out_dim
```

## Recommendation

**Option C (recommended): refactor in place to canonical diffusers schema.**
Keep our existing `CausalConv3d`, `RMS_norm`, `AttentionBlock`, `Resample` op
primitives (those are correct). Replace the `Encoder3d` / `Decoder3d` /
`AutoencoderKLWan` classes with reimplementations whose module hierarchy
matches diffusers 0.38. Effort: ~4–8 hours including the parity gate.

**Option A (re-port from PT diffusers source):** read
`diffusers/models/autoencoders/autoencoder_kl_wan.py` end-to-end and write a
new MLX port from scratch, matching the PT structure exactly. Most rigorous,
most lines of new code (~600), but yields a future-proof file.

**Option B (find a compatible Wan checkpoint elsewhere):** Wan-AI's HF
checkpoints might use mlx-video's expected schema. Could swap our weight
source to those. **Not viable** — Meituan's DiT was trained against
Meituan's VAE, so swapping VAE checkpoints will break the joint distribution
and produce garbage video latents through the rest of the pipeline. Don't do this.
