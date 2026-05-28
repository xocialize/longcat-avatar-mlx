# Meituan INT8 checkpoint format (resolves v2 Open Question #2)

**Source:**
- [`base_model_int8/quantization_config.json`](config-snapshot/avatar-1.5--base_model_int8-quantization_config.json)
- [`base_model_int8/quantized_model.safetensors.index.json`](config-snapshot/avatar-1.5--base_model_int8-quantized_model.safetensors.index.json)
- safetensors header of `quantized_model-00001-of-00004.safetensors` (sampled, 610 tensors)

## Format

```json
{
  "quantization_method": "int8_per_channel_symmetric",
  "skip_patterns": ["final_layer.linear"],
  "description": "Weight-only INT8 quantization with per-channel symmetric scaling"
}
```

Per Linear, two tensors:
- `{module}.weight_int8`  — `dtype=I8`, shape `(out_features, in_features)`
- `{module}.weight_scale` — `dtype=F32`, shape `(out_features,)`  ← per **output** channel
- `{module}.bias`         — `dtype=BF16` (unchanged from base_model)

Dequant formula (verbatim):
```
W_fp = W_int8.astype(fp16) * scale[:, None]   # shape (out, in)
y = x @ W_fp.T + bias                          # or use fused op
```

Sampled from shard 1:
- 172 quantized Linears → 172 (I8) + 172 (F32 scale) tensors
- 266 BF16 tensors for biases, norms, embeddings, and non-quantized layers
- Total in shard 1: **610 tensors**, dtype distribution `{F32: 172, BF16: 266, I8: 172}`
- 4 shards × ~4 GB each = **~15.9 GB total INT8**  (vs ~31.7 GB bf16)

## MLX compatibility verdict

**NOT directly loadable into `mlx.nn.quantize` format.** MLX's grouped quantization (`bits=8, group_size=128`) uses a different layout: weights bit-packed into int32, plus `scales` AND `biases` tensors *per group of `group_size` weights*. Meituan's format is **per output channel** (one scale per row), which is coarser.

**Two viable Stage 2 paths:**

### Option A — Custom MLX QuantLinear, load Meituan's INT8 verbatim
- Write a ~30-line `MeituanInt8Linear(mlx.nn.Module)` that holds `weight_int8` + `weight_scale` + `bias` and dequant-on-the-fly:
  ```python
  def __call__(self, x):
      w = self.weight_int8.astype(mx.bfloat16) * self.weight_scale[:, None]
      return x @ w.T + self.bias
  ```
- Pros: byte-for-byte faithful to Meituan; preserves their quant calibration; smallest porting risk.
- Cons: no kernel acceleration — dequant runs as a separate elementwise multiply each forward pass.
- Replace `nn.Linear` with this only for the 172 quantized Linears; keep `final_layer.linear` unquantized per skip-pattern.

### Option B — Re-quantize bf16 → `mlx.nn.quantize(bits=8, group_size=128)`
- Faster at inference (MLX has tuned QMV kernels for grouped quant).
- Cons: a second approximation on top of bf16; might drift visibly. Need parity test against bf16 reference.
- This is the path for the `-q8` and `-q4` mlx-community variants.

## Recommended Stage 2 order

1. **`mlx-community/LongCat-Video-Avatar-1.5-bf16`** — full bf16, no quantization, the parity reference.
2. **`mlx-community/LongCat-Video-Avatar-1.5-int8-meituan`** — Option A, byte-faithful Meituan port. Same disk footprint (~16 GB), same numerics.
3. **`mlx-community/LongCat-Video-Avatar-1.5-q4`** — Option B with MLX 4-bit grouped, for 64 GB Macs. Aggressive, ship after smoke test.

## Tensor-name mapping (sampled)

```
blocks.0.adaLN_modulation.1            quantized   (24576 = 6 * 4096)
blocks.0.audio_adaLN_modulation.1      quantized   (12288 = 3 * 4096)
blocks.0.attn.qkv                      quantized   (12288 = 3 * 4096)
blocks.0.attn.proj                     quantized   (4096)
blocks.0.cross_attn.q_linear           quantized   (4096)
blocks.0.cross_attn.kv_linear          quantized   (8192 = 2 * 4096)
blocks.0.cross_attn.proj               quantized   (4096)
blocks.0.audio_cross_attn.q_linear     quantized   (4096)
blocks.0.audio_cross_attn.kv_linear    quantized   (8192)
blocks.0.audio_cross_attn.proj         quantized   (4096)
blocks.0.ffn.w1                        quantized   (11008)   ← FFN inner = 11008, NOT 16384
blocks.0.ffn.w2                        quantized   (4096)
blocks.0.ffn.w3                        quantized   (11008)
x_embedder.proj.weight                 BF16 (Conv3d)
t_embedder.mlp.0                       quantized   (512)
t_embedder.mlp.2                       quantized   (512)
y_embedder.y_proj.0                    quantized   (4096)
y_embedder.y_proj.2                    quantized   (4096)
final_layer.linear                     BF16 (skip pattern)
```

Confirms 13 quantized Linears per block × 48 blocks + ~16 non-block quantized Linears.

## Critical incidental finding

**FFN inner dimension is 11008, not 16384.** The `FeedForwardSwiGLU.__init__` does:
```python
hidden_dim = int(2 * (hidden_size * mlp_ratio) / 3)   # = int(2*16384/3) = 10922
hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)  # = 256 * 43 = 11008
```
So `mlp_ratio=4` with `multiple_of=256` gives 11008. The architecture spec has been updated. **Easy to get wrong in the port if you naïvely use `int(hidden * mlp_ratio)`.**
