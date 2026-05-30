from __future__ import annotations

import os
from typing import Any

import numpy.typing as npt
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor

from eecs148b_hw1.bpe import train_bpe
from eecs148b_hw1.data import get_batch
from eecs148b_hw1.modules import (
    Embedding,
    LayerNorm,
    Linear,
    MultiHeadSelfAttention,
    PositionwiseFeedForward,
    SinusoidalPositionalEncoding,
    TransformerBlock,
    TransformerLM,
    scaled_dot_product_attention,
)
from eecs148b_hw1.nn_utils import cross_entropy, softmax
from eecs148b_hw1.tokenizer import Tokenizer


def run_linear(
    d_in: int,
    d_out: int,
    weights: Float[Tensor, " d_out d_in"],
    in_features: Float[Tensor, " ... d_in"],
) -> Float[Tensor, " ... d_out"]:
    layer = Linear(d_in, d_out)
    layer.load_state_dict({"weight": weights})
    return layer(in_features)


def run_embedding(
    vocab_size: int,
    d_model: int,
    weights: Float[Tensor, " vocab_size d_model"],
    token_ids: Int[Tensor, " ..."],
) -> Float[Tensor, " ... d_model"]:
    layer = Embedding(vocab_size, d_model)
    layer.load_state_dict({"weight": weights})
    return layer(token_ids)


def run_ffn(
    d_model: int,
    d_ff: int,
    w1_weight: Float[Tensor, " d_ff d_model"],
    w2_weight: Float[Tensor, " d_model d_ff"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ffn = PositionwiseFeedForward(d_model, d_ff)
    ffn.load_state_dict({"fc1.weight": w1_weight, "fc2.weight": w2_weight})
    return ffn(in_features)


def run_layernorm(
    d_model: int,
    eps: float,
    weight: Float[Tensor, " d_model"],
    bias: Float[Tensor, " d_model"],
    in_features: Float[Tensor, " ... d_model"],
) -> Float[Tensor, " ... d_model"]:
    ln = LayerNorm(d_model, eps=eps)
    ln.load_state_dict({"weight": weight, "bias": bias})
    return ln(in_features)


def run_sinusoidal_pe(
    d_model: int,
    max_seq_len: int,
    token_positions: Int[Tensor, " ... sequence_length"],
) -> Float[Tensor, " ... sequence_length d_model"]:
    pe = SinusoidalPositionalEncoding(d_model, max_seq_len)
    return pe(token_positions)


def run_scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:
    return scaled_dot_product_attention(Q, K, V, mask=mask)


def run_multihead_self_attention(
    d_model: int,
    num_heads: int,
    q_proj_weight: Float[Tensor, " d_k d_in"],
    k_proj_weight: Float[Tensor, " d_k d_in"],
    v_proj_weight: Float[Tensor, " d_v d_in"],
    o_proj_weight: Float[Tensor, " d_model d_v"],
    in_features: Float[Tensor, " ... sequence_length d_in"],
) -> Float[Tensor, " ... sequence_length d_out"]:
    mha = MultiHeadSelfAttention(d_model, num_heads)
    mha.load_state_dict(
        {
            "q_proj.weight": q_proj_weight,
            "k_proj.weight": k_proj_weight,
            "v_proj.weight": v_proj_weight,
            "output_proj.weight": o_proj_weight,
        }
    )
    return mha(in_features)


def run_transformer_block(
    d_model: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_features: Float[Tensor, " batch sequence_length d_model"],
) -> Float[Tensor, " batch sequence_length d_model"]:
    block = TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
    block.load_state_dict(weights)
    return block(in_features)


def run_transformer_lm(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    weights: dict[str, Tensor],
    in_indices: Int[Tensor, " batch_size sequence_length"],
) -> Float[Tensor, " batch_size sequence_length vocab_size"]:
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    # The sinusoidal PE buffer is non-persistent and isn't in `weights`, so
    # use strict=False to allow that and skip any other extra keys.
    model.load_state_dict(weights, strict=False)
    return model(in_indices)


def run_get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_batch(dataset, batch_size, context_length, device)


def run_softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    return softmax(in_features, dim=dim)


def run_cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    return cross_entropy(inputs, targets)


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    return Tokenizer(vocab, merges, special_tokens)


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    return train_bpe(input_path=input_path, vocab_size=vocab_size, special_tokens=special_tokens)
