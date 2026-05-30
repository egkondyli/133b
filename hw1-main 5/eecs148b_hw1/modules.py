"""Transformer language model components, built from scratch.

Only `torch.nn.Parameter` and container classes (`nn.Module`, `nn.ModuleList`)
are used from `torch.nn`; everything else (linear, embedding, layernorm,
attention, FFN, sinusoidal PE) is implemented manually.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nn_utils import softmax


# ----------------------------------------------------------------------
# Linear (no bias)
# ----------------------------------------------------------------------
class Linear(nn.Module):
    """y = W x, where W has shape (out_features, in_features). No bias term."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Stored as (d_out, d_in) for the right memory layout.
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # Initialization: truncated normal with std = sqrt(2 / (d_in + d_out)).
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., d_in)  ->  out: (..., d_out)
        # Use F.linear so we go through the same BLAS path the reference
        # snapshot was generated with; `x @ self.weight.T` is mathematically
        # identical but can dispatch to a different GEMM kernel.
        return F.linear(x, self.weight)


# ----------------------------------------------------------------------
# Embedding
# ----------------------------------------------------------------------
class Embedding(nn.Module):
    """Token-id -> dense vector lookup table of shape (num_embeddings, embedding_dim)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        # Per spec: N(0, 1) truncated to [-3, 3].
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (...,) -> (..., embedding_dim)
        return self.weight[token_ids]


# ----------------------------------------------------------------------
# LayerNorm (standard, with affine weight + bias)
# ----------------------------------------------------------------------
class LayerNorm(nn.Module):
    """Standard LayerNorm: normalize last dim, then affine (weight, bias).

    Mean/variance are computed in float32 to avoid precision loss when the
    upstream tensor is float16 / bfloat16.
    """

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x32 = x.to(torch.float32)
        mu = x32.mean(dim=-1, keepdim=True)
        var = x32.var(dim=-1, keepdim=True, unbiased=False)  # biased var (1/N)
        normed = (x32 - mu) / torch.sqrt(var + self.eps)
        out = normed * self.weight + self.bias
        return out.to(in_dtype)


# ----------------------------------------------------------------------
# Sinusoidal positional embeddings (no learnable params)
# ----------------------------------------------------------------------
class SinusoidalPositionalEncoding(nn.Module):
    """Vaswani et al. (2017) sinusoidal PE.

    For position p and dimension i:
        PE(p, 2i)   = sin(p / 10000^(2i / d_model))
        PE(p, 2i+1) = cos(p / 10000^(2i / d_model))

    Precomputed once into a (max_seq_len, d_model) buffer; forward indexes it
    by `token_positions`.
    """

    def __init__(self, d_model: int, max_seq_len: int, device=None, dtype=None):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be even for sinusoidal PE."
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)  # (L,)
        # Frequencies for each of the d_model/2 pairs of dims.
        two_i = torch.arange(0, d_model, 2, dtype=torch.float32, device=device)  # (d/2,)
        div_term = torch.exp(-math.log(10000.0) * two_i / d_model)  # (d/2,)

        angles = positions[:, None] * div_term[None, :]  # (L, d/2)
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32, device=device)
        pe[:, 0::2] = torch.sin(angles)
        pe[:, 1::2] = torch.cos(angles)

        if dtype is not None:
            pe = pe.to(dtype)
        # Non-persistent buffer: stays with the module but isn't saved in state_dict.
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, token_positions: Tensor) -> Tensor:
        # token_positions: (..., L)  ->  (..., L, d_model)
        return self.pe[token_positions]


# ----------------------------------------------------------------------
# Position-wise feed-forward network: 2-layer with ReLU
# ----------------------------------------------------------------------
class PositionwiseFeedForward(nn.Module):
    """FFN(x) = W2 * ReLU(W1 * x). No bias terms anywhere."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.fc1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def _relu(x: Tensor) -> Tensor:
        # F.relu goes through the same fused kernel the reference uses.
        return F.relu(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self._relu(self.fc1(x)))


# ----------------------------------------------------------------------
# Scaled dot-product attention
# ----------------------------------------------------------------------
def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Args:
        Q: (..., n_queries, d_k)
        K: (..., n_keys,    d_k)
        V: (..., n_keys,    d_v)
        mask: optional bool tensor broadcastable to (..., n_queries, n_keys).
              True  = attend (information may flow);
              False = mask  (attention prob becomes 0).
    """
    d_k = Q.shape[-1]
    # (..., n_queries, n_keys)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)

    if mask is not None:
        # Set disallowed positions to -inf so they go to 0 after softmax.
        scores = scores.masked_fill(~mask, float("-inf"))

    attn = softmax(scores, dim=-1)
    return attn @ V


# ----------------------------------------------------------------------
# Causal multi-head self-attention
# ----------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking.

    All four projections (Q, K, V, output) are linear maps from / to d_model.
    Q, K, V are reshaped into (num_heads, d_head) before attention.
    """

    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads."
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        # x: (..., seq_len, d_model)
        *batch_dims, seq_len, _ = x.shape

        # Project to Q/K/V.  Each is (..., seq_len, d_model).
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split into heads: (..., seq_len, num_heads, d_head) -> (..., num_heads, seq_len, d_head)
        def split_heads(t: Tensor) -> Tensor:
            t = t.view(*batch_dims, seq_len, self.num_heads, self.d_head)
            # Move the heads dim before the sequence dim.
            return t.transpose(-2, -3)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Causal mask: shape (seq_len, seq_len), True on/below the diagonal.
        # Broadcasts over batch and head dims.
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        )

        # (..., num_heads, seq_len, d_head)
        attn_out = scaled_dot_product_attention(q, k, v, mask=causal_mask)

        # Recombine heads: (..., seq_len, num_heads * d_head) = (..., seq_len, d_model).
        attn_out = attn_out.transpose(-2, -3).contiguous()
        attn_out = attn_out.view(*batch_dims, seq_len, self.d_model)

        return self.output_proj(attn_out)


# ----------------------------------------------------------------------
# Pre-norm Transformer block
# ----------------------------------------------------------------------
class TransformerBlock(nn.Module):
    """A pre-norm Transformer block:

        y = x + Attn(LN1(x))
        z = y + FFN(LN2(y))
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.ln1 = LayerNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)
        self.ln2 = LayerNorm(d_model, device=device, dtype=dtype)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ----------------------------------------------------------------------
# Full Transformer LM
# ----------------------------------------------------------------------
class TransformerLM(nn.Module):
    """Decoder-only Transformer language model.

    State dict layout (matches the reference):
        token_embeddings.weight                  (vocab_size, d_model)
        layers.{i}.attn.q_proj.weight            (d_model, d_model)
        layers.{i}.attn.k_proj.weight            (d_model, d_model)
        layers.{i}.attn.v_proj.weight            (d_model, d_model)
        layers.{i}.attn.output_proj.weight       (d_model, d_model)
        layers.{i}.ln1.weight, layers.{i}.ln1.bias   (d_model,)
        layers.{i}.ffn.fc1.weight                (d_ff, d_model)
        layers.{i}.ffn.fc2.weight                (d_model, d_ff)
        layers.{i}.ln2.weight, layers.{i}.ln2.bias   (d_model,)
        ln_final.weight, ln_final.bias               (d_model,)
        lm_head.weight                            (vocab_size, d_model)
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        use_pos_embedding: bool = True,
        use_layernorm: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_pos_embedding = use_pos_embedding
        self.use_layernorm = use_layernorm

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        if use_pos_embedding:
            self.pos_embedding = SinusoidalPositionalEncoding(
                d_model, context_length, device=device, dtype=dtype
            )
        else:
            self.pos_embedding = None

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )

        if use_layernorm:
            self.ln_final = LayerNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = nn.Identity()

        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor) -> Tensor:
        """in_indices: (batch, seq_len)  ->  logits: (batch, seq_len, vocab_size)."""
        _, seq_len = in_indices.shape

        x = self.token_embeddings(in_indices)  # (B, L, d_model)
        if self.pos_embedding is not None:
            positions = torch.arange(seq_len, device=in_indices.device)
            x = x + self.pos_embedding(positions)  # broadcasts over batch

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits
