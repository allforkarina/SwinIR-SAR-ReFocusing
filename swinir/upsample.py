"""PixelShuffle upsampling modules retained for SwinIR compatibility."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import math

from torch import nn


class Upsample(nn.Sequential):
    def __init__(self, scale: int, num_feat: int) -> None:
        modules: list[nn.Module] = []
        if scale > 0 and (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                modules.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                modules.append(nn.PixelShuffle(2))
        elif scale == 3:
            modules.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            modules.append(nn.PixelShuffle(3))
        else:
            raise ValueError(
                f"scale {scale} is not supported; expected a power of two or 3"
            )
        super().__init__(*modules)


class UpsampleOneStep(nn.Sequential):
    def __init__(
        self,
        scale: int,
        num_feat: int,
        num_out_ch: int,
        input_resolution: tuple[int, int] | None = None,
    ) -> None:
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        super().__init__(
            nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1),
            nn.PixelShuffle(scale),
        )

    def flops(self) -> int:
        if self.input_resolution is None:
            raise ValueError("input_resolution is required to compute FLOPs")
        height, width = self.input_resolution
        return height * width * self.num_feat * 3 * 9
