"""A stage of alternating regular and shifted Swin Transformer blocks."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
import torch.utils.checkpoint as checkpoint
from torch import nn

from .swin_block import SwinTransformerBlock


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,                                       # input channel: 2->96
        input_resolution: tuple[int, int],              # input size of feature map: [512, 512]
        depth: int,                                     # number of Swin Transformer blocks.
        num_heads: int,                                 # number of heads for each block.
        window_size: int,                               # size of the attention window.
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop: float = 0.0,                              # dropout rate.
        attn_drop: float = 0.0,                         # attention dropout rate.
        drop_path: float | list[float] = 0.0,           # stochastic depth rate.（随机跳过残差连接过程）
        norm_layer: type[nn.Module] = nn.LayerNorm,     # normalization: LayerNorm.
        downsample: type[nn.Module] | None = None,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = tuple(input_resolution)
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # if the drop_path is a list, length must equal to depth, each block need a rate.
        if isinstance(drop_path, list) and len(drop_path) != depth:
            raise ValueError(
                f"drop_path list length {len(drop_path)} must equal depth={depth}"
            )

        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=self.input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for index in range(depth)
            ]
        )
        self.downsample = (
            downsample(self.input_resolution, dim=dim, norm_layer=norm_layer)
            if downsample is not None
            else None
        )

    def forward(
        self,
        x: torch.Tensor,
        x_size: tuple[int, int],
    ) -> torch.Tensor:
        for block in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(
                    block,
                    x,
                    x_size,
                    use_reentrant=False,
                )
            else:
                x = block(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, "
            f"depth={self.depth}"
        )

    def flops(self) -> int:
        flops = sum(block.flops() for block in self.blocks)
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops
