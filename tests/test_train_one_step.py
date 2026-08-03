from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from main import train_one_step
from swinir.training import make_ema_model, make_grad_scaler, resolve_precision


def test_train_one_step_updates_model_scheduler_and_ema_on_cpu() -> None:
    torch.manual_seed(7)
    model = nn.Conv2d(2, 2, kernel_size=1, bias=False)
    ema_model = make_ema_model(model)
    optimizer = Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.99))
    scheduler = MultiStepLR(optimizer, milestones=[1], gamma=0.5)
    precision = resolve_precision(torch.device("cpu"))
    scaler = make_grad_scaler(precision)
    initial_parameter = model.weight.detach().clone()
    batch = {
        "input": torch.ones(1, 2, 4, 4),
        "target": torch.zeros(1, 2, 4, 4),
    }

    result = train_one_step(
        model,
        ema_model,
        optimizer,
        scheduler,
        scaler,
        batch,
        device=torch.device("cpu"),
        precision=precision,
        loss_epsilon=1e-3,
        ema_decay=0.5,
    )

    assert result.did_optimizer_step is True
    assert result.loss > 0.0
    assert result.gradient_norm > 0.0
    assert result.scaler_scale_before is None
    assert result.scaler_scale_after is None
    assert not torch.equal(model.weight.detach(), initial_parameter)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(5e-3)
    expected_ema = initial_parameter * 0.5 + model.weight.detach() * 0.5
    torch.testing.assert_close(ema_model.weight.detach(), expected_ema)
