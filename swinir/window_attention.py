"""Window multi-head self-attention with relative position bias."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

import torch
from torch import nn


class WindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_positions, num_heads)
        )

        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_windows, token_count, channels = x.shape
        expected_tokens = self.window_size[0] * self.window_size[1]
        if token_count != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} tokens per window, got {token_count}"
            )

        qkv = (
            self.qkv(x)
            .reshape(
                batch_windows,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(token_count, token_count, self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            if batch_windows % num_windows:
                raise ValueError(
                    f"attention batch {batch_windows} must be divisible by "
                    f"mask window count {num_windows}"
                )
            attn = attn.reshape(
                batch_windows // num_windows,
                num_windows,
                self.num_heads,
                token_count,
                token_count,
            )
            attn = attn + mask.unsqueeze(0).unsqueeze(2)
            attn = attn.reshape(
                batch_windows,
                self.num_heads,
                token_count,
                token_count,
            )

        attn = self.attn_drop(self.softmax(attn))
        x = (attn @ v).transpose(1, 2).reshape(
            batch_windows,
            token_count,
            channels,
        )
        return self.proj_drop(self.proj(x))

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, window_size={self.window_size}, "
            f"num_heads={self.num_heads}"
        )

    def flops(self, token_count: int) -> int:
        flops = token_count * self.dim * 3 * self.dim
        flops += self.num_heads * token_count * (self.dim // self.num_heads) * token_count
        flops += self.num_heads * token_count * token_count * (self.dim // self.num_heads)
        flops += token_count * self.dim * self.dim
        return flops
