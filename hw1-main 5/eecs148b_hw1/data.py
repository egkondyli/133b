"""Data loading for language model training."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch from a 1-D array of token IDs.

    Returns (x, y), both of shape (batch_size, context_length), where
    y[b, t] = x[b, t+1] for every position.

    Valid starting indices are 0 .. len(dataset) - context_length - 1
    (so y's last element x[start + context_length] is still in-bounds).
    """
    n = len(dataset)
    # max_start is exclusive; allows starts in [0, n - context_length).
    max_start = n - context_length
    starts = np.random.randint(0, max_start, size=batch_size)

    # Build x and y by stacking sliced windows.
    x_np = np.stack([np.asarray(dataset[s : s + context_length]) for s in starts])
    y_np = np.stack(
        [np.asarray(dataset[s + 1 : s + 1 + context_length]) for s in starts]
    )

    # Convert to long tensors on the requested device.
    x = torch.from_numpy(x_np.astype(np.int64)).to(device)
    y = torch.from_numpy(y_np.astype(np.int64)).to(device)
    return x, y
