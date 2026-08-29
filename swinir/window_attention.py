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

        # etc: 96 % 6 = 16, each head contains 16 channels for one feature subspace.
        if dim % num_heads:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5             # scale attention.

        """
            possible relative position: -(Ww - 1), -(Ww - 2), ..., 0, ..., (Ww - 2), (Ww - 1)
            etc: window(0, 0) with other positions' relative positions: 0, 1, 2, ..., Ww - 1
            As the same, height also contains 2 * Wh - 1 relative positions.
        """
        relative_positions = (2 * window_size[0] - 1) * (2 * window_size[1] - 1)
        # each head refer to one feature. each feature using window-attention with one relative tables.
        # init as zero, and learn 
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(relative_positions, num_heads)
        )

        coords_h = torch.arange(window_size[0])     # etc: [0, 1, 2, 3, 4, 5, 6] for window size = 7
        coords_w = torch.arange(window_size[1])     # etc: [0, 1, 2, 3, 4, 5, 6] for window size = 7
        # coords.shape = [2, 7, 7], coords[0] -> row coordinates, coords[1] -> column coordinates.
        # coords[:, 0, 0] = [0, 0].
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij")) # window coordinate
        """
        coords_flatten =
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 1, 2, 0, 1, 2, 0, 1, 2],
        ]
        """
        coords_flatten = torch.flatten(coords, 1)

        # calculate relative coordinates, etc: [0, 0] - [0, 0] = [0, 0]; [0, 1] - [0, 0] = [0, 1]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        # permute to [Ww, Wh, 2]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        # relative_coords can be negative, add bias to make it all positive.
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1

        """
            relative_positions kind of relative positions in total.
            then for pixel(i, j) attention need a unique bias from relative_position_bias_table, found by the index.
            here generate the unique index for each pixel.
                etc: window [3, 3], contains 9 * 9 = 81 attention pairs
                     but only 5 * 5 = 25 unique relative positions.
                     so each attention pair need a index to find the bias from relative_position_bias_table.
            relative_coords shape equal to attention matrix, 
            each attention pair find index from relative_coords, then find bias from relative_position_bias_table.
        """
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1  # row concat, (1, 0) -> 5(0->4 is first row)
        self.register_buffer(
            "relative_position_index",
            relative_coords.sum(-1),
        )

        # Linear projection for qkv, hide layer with more channels, etc: 96 -> 288.
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # Dropout for attention weights and output projection.
        self.attn_drop = nn.Dropout(attn_drop)
        # one to one, self-learning weights with dropout, some channel is important, some is not.
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # activate. avoid linear for more expressive power.
        self.softmax = nn.Softmax(dim=-1)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # input tensor X: [batch, pixel_num, channel_num], pixel_num = window_size * window_size.
        batch_windows, token_count, channels = x.shape
        expected_tokens = self.window_size[0] * self.window_size[1]
        if token_count != expected_tokens:
            raise ValueError(
                f"expected {expected_tokens} tokens per window, got {token_count}"
            )

        # one tensor x project to three tensors qkv, reshape to [B, N, 3(qkv), H(heads), C/H].
        qkv = (
            self.qkv(x)
            .reshape(
                batch_windows,
                token_count,
                3,
                self.num_heads,
                channels // self.num_heads,
            )
            # permute to [qkv, B, H, N, C/H].
            .permute(2, 0, 3, 1, 4) 
        )
        # extract q, k, v.
        q, k, v = qkv.unbind(0)
        # scale q for dot product.
        q = q * self.scale
        # attention calculate.
        attn = q @ k.transpose(-2, -1)

        # get bias, q shape: [B, H, N, C/H], and attn shape: [B, H, N, N].
        # reshape to [N, N, H], then permute to [H, N, N] for each head.
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ].reshape(token_count, token_count, self.num_heads)
        relative_bias = relative_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_bias.unsqueeze(0)    # squeeze to batch dimension.

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

        # softmax activate and dropout for attention weights.
        attn = self.attn_drop(self.softmax(attn))

        # [B, H, N, N] @ [B, H, N, C/H] -> [B, H, N, C/H]
        # then permute to [B, N, H, C/H], reshape to [B, N, C].
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
