from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from swinir.training import (
    complex_charbonnier_loss,
    make_ema_model,
    normalized_complex_rmse,
    update_ema,
)


def test_complex_charbonnier_is_joint_over_real_and_imaginary_channels() -> None:
    prediction = torch.tensor([[[[3.0]], [[4.0]]]], dtype=torch.float16)
    target = torch.zeros_like(prediction)

    loss = complex_charbonnier_loss(prediction, target, epsilon=1e-3)

    assert loss.dtype is torch.float32
    assert loss.item() == pytest.approx(math.sqrt(25.0 + 1e-6))


def test_normalized_complex_rmse_uses_both_channels() -> None:
    prediction = torch.tensor([[[[3.0]], [[4.0]]]])
    target = torch.zeros_like(prediction)

    assert normalized_complex_rmse(prediction, target).item() == pytest.approx(
        math.sqrt((9.0 + 16.0) / 2.0)
    )


def test_ema_updates_floating_parameters_and_copies_non_floating_buffers() -> None:
    model = nn.BatchNorm2d(1)
    ema_model = make_ema_model(model)
    with torch.no_grad():
        model.weight.fill_(3.0)
        model.running_mean.fill_(5.0)
        model.num_batches_tracked.fill_(7)
        ema_model.weight.fill_(1.0)

    update_ema(ema_model, model, decay=0.75)

    assert ema_model.weight.item() == pytest.approx(1.5)
    assert ema_model.running_mean.item() == pytest.approx(5.0)
    assert ema_model.num_batches_tracked.item() == 7
