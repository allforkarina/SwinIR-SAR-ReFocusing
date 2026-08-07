from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import pytest
import torch


ENTRYPOINT_PATH = Path(__file__).parents[1] / "test.py"
SPEC = importlib.util.spec_from_file_location("sar_test_entrypoint", ENTRYPOINT_PATH)
assert SPEC is not None and SPEC.loader is not None
ENTRYPOINT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ENTRYPOINT
SPEC.loader.exec_module(ENTRYPOINT)
MetricTotals = ENTRYPOINT.MetricTotals
is_independent_coordinate = ENTRYPOINT.is_independent_coordinate


def test_metric_totals_reports_identity_baseline_improvement() -> None:
    prediction = torch.tensor([[[[3.0]], [[4.0]]]])
    target = torch.zeros_like(prediction)
    echo = torch.tensor([[[[6.0]], [[8.0]]]])
    totals = MetricTotals()

    totals.update(prediction, target, echo, loss_epsilon=1e-3)

    metrics = totals.as_dict()
    assert metrics["patch_count"] == 1
    assert metrics["charbonnier"] == pytest.approx(math.sqrt(25.0 + 1e-6))
    assert metrics["echo_baseline_charbonnier"] == pytest.approx(math.sqrt(100.0 + 1e-6))
    assert metrics["complex_rmse"] == pytest.approx(math.sqrt(12.5))
    assert metrics["echo_baseline_complex_rmse"] == pytest.approx(math.sqrt(50.0))
    assert metrics["complex_rmse_relative_improvement_percent"] == pytest.approx(50.0)


def test_independent_coordinate_uses_the_shared_origin_and_stride() -> None:
    assert is_independent_coordinate(
        10000, 3000, origin_row=10000, origin_col=3000, stride=600
    )
    assert is_independent_coordinate(
        10600, 3600, origin_row=10000, origin_col=3000, stride=600
    )
    assert not is_independent_coordinate(
        10100, 3000, origin_row=10000, origin_col=3000, stride=600
    )
