import numpy
import torch
from einops import rearrange

from .adapters import (
    run_embedding,
    run_ffn,
    run_layernorm,
    run_linear,
    run_multihead_self_attention,
    run_scaled_dot_product_attention,
    run_sinusoidal_pe,
    run_transformer_block,
    run_transformer_lm,
)


def test_linear(numpy_snapshot, ts_state_dict, in_embeddings, d_model, d_ff):
    w1_weight = ts_state_dict[0]["layers.0.ffn.fc1.weight"]
    output = run_linear(
        d_in=d_model,
        d_out=d_ff,
        weights=w1_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(output)


def test_embedding(numpy_snapshot, ts_state_dict, in_indices, vocab_size, d_model):
    embedding_weight = ts_state_dict[0]["token_embeddings.weight"]
    output = run_embedding(
        vocab_size=vocab_size,
        d_model=d_model,
        weights=embedding_weight,
        token_ids=in_indices,
    )
    numpy_snapshot.assert_match(output)


def test_ffn(numpy_snapshot, ffn_in_features, ffn_w1_weight, ffn_w2_weight, d_model, d_ff):
    actual_output = run_ffn(
        d_model=d_model,
        d_ff=d_ff,
        w1_weight=ffn_w1_weight,
        w2_weight=ffn_w2_weight,
        in_features=ffn_in_features,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_4d_scaled_dot_product_attention(numpy_snapshot, q, k, v, mask):
    # Shape: (batch_size, num_heads, seq_len, d_k)
    q, k, v = (rearrange(x, "(batch head) seq d -> batch head seq d", head=2) for x in (q, k, v))
    mask = rearrange(mask, "(batch head) query key -> batch head query key", head=2)

    actual_output = run_scaled_dot_product_attention(Q=q, K=k, V=v, mask=mask)
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_multihead_self_attention(numpy_snapshot, in_embeddings, d_model, n_heads, ts_state_dict):
    d, _ = ts_state_dict
    q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight = [
        d[f"layers.0.attn.{k}_proj.weight"] for k in ["q", "k", "v", "output"]
    ]
    actual_output = run_multihead_self_attention(
        d_model=d_model,
        num_heads=n_heads,
        q_proj_weight=q_proj_weight,
        k_proj_weight=k_proj_weight,
        v_proj_weight=v_proj_weight,
        o_proj_weight=o_proj_weight,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_transformer_lm(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, ts_state_dict, in_indices
):
    state_dict, _ = ts_state_dict

    actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        weights=state_dict,
        in_indices=in_indices,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-4, rtol=1e-2)


def test_transformer_lm_truncated_input(
    numpy_snapshot, vocab_size, n_keys, d_model, n_layers, n_heads, d_ff, ts_state_dict, in_indices
):
    in_indices_truncated = in_indices[..., : in_indices.shape[-1] // 2]
    truncated_actual_output = run_transformer_lm(
        vocab_size=vocab_size,
        context_length=n_keys,
        d_model=d_model,
        num_layers=n_layers,
        num_heads=n_heads,
        d_ff=d_ff,
        weights=ts_state_dict[0],
        in_indices=in_indices_truncated,
    )

    numpy_snapshot.assert_match(
        truncated_actual_output,
        atol=1e-4,
    )


def test_transformer_block(numpy_snapshot, ts_state_dict, in_embeddings, d_model, n_heads, d_ff):
    block_weights = {k.replace("layers.0.", ""): v for k, v in ts_state_dict[0].items() if "layers.0." in k}

    actual_output = run_transformer_block(
        d_model=d_model,
        num_heads=n_heads,
        d_ff=d_ff,
        weights=block_weights,
        in_features=in_embeddings,
    )
    numpy_snapshot.assert_match(
        actual_output,
        atol=1e-6,
    )


def test_layernorm(numpy_snapshot, layernorm_in_features, layernorm_weight, layernorm_bias, d_model):
    actual_output = run_layernorm(
        d_model=d_model,
        eps=1e-5,
        weight=layernorm_weight,
        bias=layernorm_bias,
        in_features=layernorm_in_features,
    )
    numpy_snapshot.assert_match(actual_output, atol=1e-6)


def test_sinusoidal_pe(numpy_snapshot, d_model, n_queries, pos_ids):
    token_positions = rearrange(pos_ids, "seq -> 1 seq")
    actual_output = run_sinusoidal_pe(d_model=d_model, max_seq_len=n_queries, token_positions=token_positions)
    numpy_snapshot.assert_match(actual_output, atol=1e-6)
