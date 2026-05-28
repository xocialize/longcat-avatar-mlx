"""MLX port of Whisper-large-v3 encoder (HF transformers schema).

LongCat-Video-Avatar 1.5 consumes the **stack of all encoder hidden states**
(embedding layer + 32 transformer layers = 33 tensors), grouped into 5 pooled
feature groups and projected into per-frame audio context tokens. See
[notes/audio-injection-wiring.md](../../notes/audio-injection-wiring.md) and
[notes/architecture-spec.md](../../notes/architecture-spec.md) for the
downstream consumption.

Module hierarchy matches `transformers.WhisperEncoder` so Meituan's
shipped weights load directly with the usual Conv*d transpose, no key
remapping. Decoder is not ported (not needed — Avatar uses encoder
hidden states, not decoded text).

Reference math sanity-checked against `Blaizzy/mlx-audio`'s AudioEncoder
(OpenAI-style naming, otherwise identical).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class WhisperAttention(nn.Module):
    """Whisper self-attention.

    HF-style parameter names: `q_proj`, `k_proj` (no bias!), `v_proj`,
    `out_proj`. Uses `1/sqrt(head_dim)` scale via `mx.fast.scaled_dot_product_attention`.
    """

    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Whisper has no bias on k_proj only (HF historical detail).
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        b, n, _ = x.shape
        q = self.q_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(b, n, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        scale = self.head_dim**-0.5
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = out.transpose(0, 2, 1, 3).reshape(b, n, self.d_model)
        return self.out_proj(out)


class WhisperEncoderLayer(nn.Module):
    """One Whisper encoder layer: pre-norm self-attn + pre-norm FFN."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.self_attn = WhisperAttention(d_model, num_heads)
        self.final_layer_norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, ffn_dim, bias=True)
        self.fc2 = nn.Linear(ffn_dim, d_model, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # Pre-norm self-attn + residual
        x = x + self.self_attn(self.self_attn_layer_norm(x))
        # Pre-norm FFN + residual
        x = x + self.fc2(nn.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class WhisperEncoder(nn.Module):
    """Whisper-large-v3 audio encoder.

    Input: mel spectrogram `[B, num_mel_bins=128, T_mel]`.
    Output: stack of all 33 hidden states `[(B, T_enc, d_model), …]`
    when called with `return_all_hidden_states=True`. Otherwise the final
    post-LayerNorm hidden state.

    The 2× temporal compression from `conv2` (stride=2) is the source of
    the 50 Hz internal frame rate referenced in the tech report — the
    upstream LongCat audio pipeline assumes T_enc = T_mel // 2.
    """

    def __init__(
        self,
        d_model: int = 1280,
        num_layers: int = 32,
        num_heads: int = 20,
        ffn_dim: int = 5120,
        num_mel_bins: int = 128,
        max_source_positions: int = 1500,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_source_positions = max_source_positions

        self.conv1 = nn.Conv1d(num_mel_bins, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)

        # Learned (loaded from checkpoint). HF initializes to sinusoidal but
        # the actual stored weights are learned.
        self.embed_positions = nn.Embedding(max_source_positions, d_model)

        self.layers = [WhisperEncoderLayer(d_model, num_heads, ffn_dim) for _ in range(num_layers)]
        self.layer_norm = nn.LayerNorm(d_model)

    @classmethod
    def from_config(cls, config: dict) -> "WhisperEncoder":
        """Build from HF `WhisperConfig` (or its JSON dict)."""
        return cls(
            d_model=config.get("d_model", 1280),
            num_layers=config.get("encoder_layers", 32),
            num_heads=config.get("encoder_attention_heads", 20),
            ffn_dim=config.get("encoder_ffn_dim", 5120),
            num_mel_bins=config.get("num_mel_bins", 128),
            max_source_positions=config.get("max_source_positions", 1500),
        )

    def __call__(
        self,
        mel_features: mx.array,
        *,
        return_all_hidden_states: bool = False,
    ):
        """Args: `mel_features` `[B, num_mel_bins, T_mel]` (channel-second).

        Returns the post-LayerNorm last hidden state by default. If
        `return_all_hidden_states=True`, returns a list of `num_layers + 1`
        tensors matching HF's `WhisperEncoder.forward(output_hidden_states=True).hidden_states`.
        """
        # MLX Conv1d wants channels-last (B, T, C). Transpose first.
        x = mel_features.transpose(0, 2, 1)
        x = nn.gelu(self.conv1(x))
        x = nn.gelu(self.conv2(x))
        # After conv2 stride=2: T_enc = T_mel // 2.

        # Slice positional embedding to match sequence length.
        t_enc = x.shape[1]
        # embed_positions.weight has shape (max_source_positions, d_model)
        pos = self.embed_positions.weight[:t_enc][None, :, :]
        x = x + pos

        all_hidden: list[mx.array] = []
        if return_all_hidden_states:
            all_hidden.append(x)

        for layer in self.layers:
            x = layer(x)
            if return_all_hidden_states:
                all_hidden.append(x)

        if return_all_hidden_states:
            # HF convention: hidden_states is pre-final-LN; .last_hidden_state is post.
            # The LongCat pipeline consumes hidden_states[i] for i in 0..32.
            return all_hidden

        return self.layer_norm(x)
