"""Feed-forward network used inside a Swin Transformer block."""

# Independent modular refactor based on the Apache-2.0 official SwinIR reference.

from __future__ import annotations

from torch import nn


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)      # full connection layer
        self.act = act_layer()                                  # GELU activate function
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    # in -> hidden -> activate & dropout -> out -> dropout
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)
