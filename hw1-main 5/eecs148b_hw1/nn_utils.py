"""Numerically-stable softmax and cross-entropy.

These are implemented from scratch (no torch.nn.functional.softmax /
cross_entropy) per the assignment constraints.
"""

from __future__ import annotations

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along `dim`.

    Standard trick: subtract the max before exp so we never compute exp(+inf).
    The result is identical to the naive formula since
        softmax(x) = softmax(x - c)   for any constant c.
    """
    max_vals = x.max(dim=dim, keepdim=True).values
    shifted = x - max_vals
    exps = torch.exp(shifted)
    return exps / exps.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy between unnormalized logits and integer targets.

    Args:
        inputs:  (..., vocab_size) logits.
        targets: (...) integer class indices in [0, vocab_size).

    Returns scalar mean loss across the flattened batch.

    We use the identity
        - log softmax(o)[t] = -o[t] + log sum_j exp(o[j])
                           = -o[t] + (max_o + log sum_j exp(o[j] - max_o))
    which is numerically stable for any logit magnitudes.
    """
    # Subtract the per-example max for stability.
    max_vals = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_vals  # (..., V)

    # log Z = log sum_j exp(shifted_j)
    log_sum_exp = torch.log(torch.exp(shifted).sum(dim=-1))  # (...,)

    # Gather the shifted logit for the target class.
    # `gather` needs targets to have the same number of dims as `shifted`.
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    losses = -target_logits + log_sum_exp  # (...,)
    return losses.mean()
