"""Residual Swin Transformer block (RSTB)."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
from torch import nn

from .basic_layer import BasicLayer
from .patch_ops import PatchEmbed, PatchUnEmbed


def _three_conv_residual(dim: int) -> nn.Sequential:
    reduced_dim = dim // 4
    if reduced_dim == 0:
        raise ValueError("3conv residual connection requires dim >= 4")
    return nn.Sequential(
        nn.Conv2d(dim, reduced_dim, 3, 1, 1),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(reduced_dim, reduced_dim, 1, 1, 0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(reduced_dim, dim, 3, 1, 1),
    )


class RSTB(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
        img_size: int | tuple[int, int] = 224,
        patch_size: int | tuple[int, int] = 4,
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=self.input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )
        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv = _three_conv_residual(dim)
        else:
            raise ValueError(
                f"unsupported resi_connection={resi_connection!r}; "
                "expected '1conv' or '3conv'"
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
    ) -> torch.Tensor:
        residual = self.residual_group(x, x_size)
        residual = self.patch_unembed(residual, x_size)
        residual = self.conv(residual)
        return self.patch_embed(residual) + x

    def flops(self) -> int:
        height, width = self.input_resolution
        return (
            self.residual_group.flops()
            + height * width * self.dim * self.dim * 9
            + self.patch_embed.flops()
            + self.patch_unembed.flops()
        )
