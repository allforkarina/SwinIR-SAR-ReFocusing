from __future__ import annotations

import math

import numpy as np
import pytest

from swinir.sar_metrics import (
    evaluate_complex_prediction,
    log_magnitude_image,
    peak_signal_to_noise_ratio,
    structural_similarity,
)


def structured_complex_matrix(size: int = 16) -> np.ndarray:
    rows, cols = np.mgrid[:size, :size]
    magnitude = 0.2 + rows / size + 0.5 * cols / size
    phase = 0.1 * rows - 0.07 * cols
    return magnitude * np.exp(1j * phase)


def test_identical_complex_prediction_reaches_metric_optima() -> None:
    target = structured_complex_matrix()

    metrics = evaluate_complex_prediction(target, target, floor_db=-60.0)

    assert metrics["normalized_complex_rmse"] == pytest.approx(0.0)
    assert metrics["complex_coherence"] == pytest.approx(1.0)
    assert metrics["magnitude_correlation"] == pytest.approx(1.0)
    assert metrics["rms_ratio_target"] == pytest.approx(1.0)
    assert metrics["log_magnitude_psnr_db"] == math.inf
    assert metrics["log_magnitude_ssim"] == pytest.approx(1.0)


def test_zero_prediction_does_not_receive_good_structure_metrics() -> None:
    target = structured_complex_matrix()
    prediction = np.zeros_like(target)

    metrics = evaluate_complex_prediction(prediction, target, floor_db=-60.0)

    assert metrics["normalized_complex_rmse"] > 0.1
    assert metrics["complex_coherence"] == 0.0
    assert metrics["magnitude_correlation"] == 0.0
    assert metrics["rms_ratio_target"] == 0.0
    assert metrics["log_magnitude_psnr_db"] < 10.0
    assert metrics["log_magnitude_ssim"] < 0.1


def test_log_magnitude_metrics_use_one_shared_target_peak() -> None:
    target = structured_complex_matrix()
    prediction = 0.5 * target
    target_peak = float(np.abs(target).max())
    target_log = log_magnitude_image(target, reference_peak=target_peak)
    prediction_log = log_magnitude_image(prediction, reference_peak=target_peak)

    assert not np.array_equal(prediction_log, target_log)
    assert peak_signal_to_noise_ratio(prediction_log, target_log) < math.inf
    assert structural_similarity(prediction_log, target_log) < 1.0


def test_ssim_rejects_arrays_smaller_than_the_agreed_window() -> None:
    values = np.zeros((10, 10), dtype=np.float64)

    with pytest.raises(ValueError, match="at least 11x11"):
        structural_similarity(values, values)
