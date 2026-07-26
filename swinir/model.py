"""Independent SwinIR model implementation for same-size SAR restoration.

The structure follows the SwinIR paper and the official Apache-2.0 reference
implementation, while runtime code is split into project-owned modules and does
not import the official repository.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .common import initialize_swinir_weights
from .patch_ops import PatchEmbed, PatchUnEmbed
from .rstb import RSTB, _three_conv_residual
from .upsample import Upsample, UpsampleOneStep


class SwinIR(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int] = 64,
        patch_size: int | tuple[int, int] = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        depths: tuple[int, ...] | list[int] = (6, 6, 6, 6),
        num_heads: tuple[int, ...] | list[int] = (6, 6, 6, 6),
        window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        ape: bool = False,
        patch_norm: bool = True,
        use_checkpoint: bool = False,
        upscale: int = 2,
        img_range: float = 1.0,
        upsampler: str = "",
        resi_connection: str = "1conv",
        **kwargs,
    ) -> None:
        super().__init__()
        if len(depths) != len(num_heads):
            raise ValueError(
                f"depths and num_heads must have equal lengths, got "
                f"{len(depths)} and {len(num_heads)}"
            )
        if window_size <= 0:
            raise ValueError("window_size must be positive")

        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = float(img_range)
        if in_chans == 3:
            self.mean = torch.tensor((0.4488, 0.4371, 0.4040)).reshape(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None,
        )

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            nn.init.trunc_normal_(self.absolute_pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_values = [
            value.item()
            for value in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        self.layers = nn.ModuleList()
        block_offset = 0
        for depth, heads in zip(depths, num_heads):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(
                    patches_resolution[0],
                    patches_resolution[1],
                ),
                depth=depth,
                num_heads=heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_values[block_offset : block_offset + depth],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
            block_offset += depth

        self.norm = norm_layer(self.num_features)
        if resi_connection == "1conv":
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == "3conv":
            self.conv_after_body = _three_conv_residual(embed_dim)
        else:
            raise ValueError(
                f"unsupported resi_connection={resi_connection!r}; "
                "expected '1conv' or '3conv'"
            )

        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif upsampler == "pixelshuffledirect":
            self.upsample = UpsampleOneStep(
                upscale,
                embed_dim,
                num_out_ch,
                (patches_resolution[0], patches_resolution[1]),
            )
        elif upsampler == "nearest+conv":
            if upscale not in (2, 4):
                raise ValueError("nearest+conv supports upscale 2 or 4")
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif upsampler == "":
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        else:
            raise ValueError(f"unsupported upsampler={upsampler!r}")

        self.apply(initialize_swinir_weights)

    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self) -> set[str]:
        return {"relative_position_bias_table"}

    def check_image_size(self, x: torch.Tensor) -> torch.Tensor:
        """Pad spatial dimensions to window multiples.

        Reflection matches the official path for normal inputs. Replication is
        used only when a very small input would make reflection padding invalid.
        """
        _, _, height, width = x.shape
        pad_height = (self.window_size - height % self.window_size) % self.window_size
        pad_width = (self.window_size - width % self.window_size) % self.window_size
        if pad_height == 0 and pad_width == 0:
            return x
        can_reflect = (pad_height == 0 or pad_height < height) and (
            pad_width == 0 or pad_width < width
        )
        mode = "reflect" if can_reflect else "replicate"
        return F.pad(x, (0, pad_width, 0, pad_height), mode=mode)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            if x.shape[1] != self.absolute_pos_embed.shape[1]:
                raise ValueError(
                    "absolute position embedding requires the padded input size "
                    "to match img_size"
                )
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x, x_size)
        x = self.norm(x)
        return self.patch_unembed(x, x_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_height, original_width = x.shape[2:]
        x = self.check_image_size(x)

        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        if self.upsampler == "pixelshuffle":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == "pixelshuffledirect":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == "nearest+conv":
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(
                self.conv_up1(F.interpolate(x, scale_factor=2, mode="nearest"))
            )
            if self.upscale == 4:
                x = self.lrelu(
                    self.conv_up2(F.interpolate(x, scale_factor=2, mode="nearest"))
                )
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            x_first = self.conv_first(x)
            residual = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(residual)

        x = x / self.img_range + self.mean
        return x[
            :,
            :,
            : original_height * self.upscale,
            : original_width * self.upscale,
        ]

    def flops(self) -> int:
        height, width = self.patches_resolution
        flops = height * width * self.in_chans * self.embed_dim * 9
        flops += self.patch_embed.flops()
        flops += sum(layer.flops() for layer in self.layers)
        flops += height * width * self.embed_dim * self.embed_dim * 9
        if hasattr(self, "upsample") and hasattr(self.upsample, "flops"):
            flops += self.upsample.flops()
        return int(flops)
