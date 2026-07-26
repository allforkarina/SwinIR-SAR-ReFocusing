"""Patch embedding operations for the patch-size-one SwinIR backbone."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
from torch import nn

from .common import to_2tuple


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x) if self.norm is not None else x

    def flops(self) -> int:
        if self.norm is None:
            return 0
        return self.img_size[0] * self.img_size[1] * self.embed_dim


class PatchUnEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patches_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
    ) -> torch.Tensor:
        batch, token_count, channels = x.shape
        if token_count != x_size[0] * x_size[1]:
            raise ValueError(
                f"token count {token_count} does not match x_size={x_size}"
            )
        if channels != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, got {channels}")
        return x.transpose(1, 2).reshape(
            batch,
            self.embed_dim,
            x_size[0],
            x_size[1],
        )

    def flops(self) -> int:
        return 0
