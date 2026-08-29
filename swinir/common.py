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
    """
    Randomly dropout some path of residual connection.
    Sometimes, y = x + F(x). And sometimes y = x + 0.(F(x) is dropped out.)

    Args:
        x (torch.Tensor): input tensor. Only care about the first dimension B, batch size.
        drop_prob (float, optional): Dropout prob for current residual path. Defaults to 0.0.(Totally keep)
        training (bool, optional): Whether in training mode. Defaults to False.
            Drop_path only works during training, random dropout to reduce overfitting.
    """
    # not dropout.
    if drop_prob == 0.0 or not training:
        return x

    # illegal drop_prob
    if not 0.0 <= drop_prob < 1.0:
        raise ValueError(f"drop_prob must be in [0, 1), got {drop_prob}")

    # prob to keep the residual path.
    keep_prob = 1.0 - drop_prob

    # if x.shape = [B, C, H, W], so (x.shape[0],) = (B,).
    # then add (x.ndim - 1) counts of 1 to the shape, shape = [B, 1, 1, 1] to fit x.shape.
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # each batch has a prob to drop.

    # each batch generate a random prob. With a keep_prob bias.
    random_tensor = keep_prob + torch.rand(
        shape,
        dtype=x.dtype,
        device=x.device
    )
    random_tensor.floor_()  # floor each batch's prob to 0 or 1. 0 means drop, 1 means keep.(mask)

    # If x * random_tensor(mask) to randomly drop, it will cause the expectation of output reduced.
    # etc: x = 10, and random_tensor = 0.5. E(x) = 10, but y = x * random_tensor, E(y) = 5. Cause the expectation of feature reduced.
    # So, first divide x by random_tensor, x = x/random_tensor(enlarge), then y = x * random_tensor, E(y) = E(x).
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    """
    Whole model contains multiple RSTB, and each block contains multiple STL.
    Each STL's output y = x + F(x). Randomly drop the F(x) value, which means model leave alone this layers' feature.

    Swin Transformer Block 原本学习一个修正量 F(x)
    DropPath 在训练过程中随机禁止部分 Block 提供这个修正量，
    使网络不能过度依赖任何单个 Block 从而起到正则化作用。
    """

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
