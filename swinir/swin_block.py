"""Swin Transformer block with regular and shifted-window attention."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
from torch import nn

from .common import DropPath, to_2tuple
from .mlp import Mlp
from .window_attention import WindowAttention
from .window_ops import window_partition, window_reverse


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,                                           # feature channel dimension
        input_resolution: tuple[int, int],                  # input feature size: [512, 512]
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
        norm_layer: type[nn.Module] = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim                                      # feature channel dimension
        self.input_resolution = tuple(input_resolution)     # input feature size: [512, 512]
        self.num_heads = num_heads                          # number of attention heads, each head refer to one feature subspace.
        self.window_size = window_size                      # window size for window-based self-attention, etc: 7x7.
        self.shift_size = shift_size                        # window shift size.
        self.mlp_ratio = mlp_ratio

        # Raw patch size should be larger than the window size.
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        # each shift should be smaller than the window size.
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size={self.shift_size} must be in [0, {self.window_size})"
            )

        self.norm1 = norm_layer(dim)                        # LayerNorm for each input feature channel.

        # Window Attention module, with relative position bias and dropout.
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # random branch dropout for each batch, to reduce overfitting.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        # MLP, hidden features = dim * mlp_ratio.
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

        # Compute the shifted-window attention mask when needed.
        attn_mask = (
            self.calculate_mask(self.input_resolution) if self.shift_size > 0 else None
        )
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size: tuple[int, int]) -> torch.Tensor:
        """Create the region-isolation mask used by shifted-window attention."""
        height, width = x_size
        if height % self.window_size or width % self.window_size:
            raise ValueError(
                f"x_size={x_size} must be divisible by window_size={self.window_size}"
            )

        image_mask = torch.zeros((1, height, width, 1))
        height_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        width_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        region = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                image_mask[:, height_slice, width_slice, :] = region
                region += 1

        mask_windows = window_partition(image_mask, self.window_size)
        mask_windows = mask_windows.reshape(
            -1,
            self.window_size * self.window_size,
        )
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
            attn_mask == 0,
            0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
    ) -> torch.Tensor:
        height, width = x_size
        batch, token_count, channels = x.shape
        if token_count != height * width:
            raise ValueError(
                f"token count {token_count} does not match x_size={x_size}"
            )
        if channels != self.dim:
            raise ValueError(f"expected channel dimension {self.dim}, got {channels}")

        shortcut = x
        x = self.norm1(x).reshape(batch, height, width, channels)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2),
            )
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.reshape(
            -1,
            self.window_size * self.window_size,
            channels,
        )
        if self.input_resolution == tuple(x_size):
            mask = self.attn_mask
        elif self.shift_size > 0:
            mask = self.calculate_mask(x_size).to(device=x.device, dtype=x.dtype)
        else:
            mask = None

        attn_windows = self.attn(x_windows, mask=mask)
        attn_windows = attn_windows.reshape(
            -1,
            self.window_size,
            self.window_size,
            channels,
        )
        shifted_x = window_reverse(
            attn_windows,
            self.window_size,
            height,
            width,
        )

        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        else:
            x = shifted_x
        x = x.reshape(batch, height * width, channels)

        x = shortcut + self.drop_path(x)
        return x + self.drop_path(self.mlp(self.norm2(x)))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"num_heads={self.num_heads}, window_size={self.window_size}, "
            f"shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

    def flops(self) -> int:
        height, width = self.input_resolution
        flops = self.dim * height * width
        num_windows = height * width / self.window_size / self.window_size
        flops += num_windows * self.attn.flops(self.window_size**2)
        flops += 2 * height * width * self.dim * self.dim * self.mlp_ratio
        flops += self.dim * height * width
        return int(flops)
