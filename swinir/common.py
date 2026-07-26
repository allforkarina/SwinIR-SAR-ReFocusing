"""Small utilities used by the SwinIR implementation."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
from torch import nn


def to_2tuple(value: int | tuple[int, int]) -> tuple[int, int]:
    """Return an integer as a square pair and preserve a two-element tuple."""
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"expected a two-element tuple, got {value!r}")
        return value
    return (value, value)


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
) -> torch.Tensor:
    """Drop residual paths independently for each batch sample."""
    if drop_prob == 0.0 or not training:
        return x
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")

    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """Per-sample stochastic depth."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


def initialize_swinir_weights(module: nn.Module) -> None:
    """Apply the initialization rules used by the official SwinIR model."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
