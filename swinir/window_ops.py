"""Window partition and reverse operations."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition ``[B, H, W, C]`` features into non-overlapping windows."""
    if x.ndim != 4:
        raise ValueError(f"expected a 4D tensor [B, H, W, C], got shape {tuple(x.shape)}")
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")

    batch, height, width, channels = x.shape
    if height % window_size or width % window_size:
        raise ValueError(
            f"height and width ({height}, {width}) must be divisible by "
            f"window_size={window_size}"
        )

    x = x.reshape(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(-1, window_size, window_size, channels)
    )


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Reverse window partitioning into a ``[B, H, W, C]`` tensor."""
    if windows.ndim != 4:
        raise ValueError(
            "expected a 4D window tensor [B*nW, M, M, C], "
            f"got shape {tuple(windows.shape)}"
        )
    if height % window_size or width % window_size:
        raise ValueError(
            f"height and width ({height}, {width}) must be divisible by "
            f"window_size={window_size}"
        )

    windows_per_image = (height // window_size) * (width // window_size)
    if windows.shape[0] % windows_per_image:
        raise ValueError(
            f"window batch {windows.shape[0]} is not divisible by "
            f"{windows_per_image} windows per image"
        )

    batch = windows.shape[0] // windows_per_image
    x = windows.reshape(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .reshape(batch, height, width, -1)
    )
