"""Metrics for complex SAR predictions and shared-reference log magnitudes."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter


def _complex_matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix, got shape={array.shape}")
    if not np.iscomplexobj(array):
        raise ValueError(f"{name} must contain complex values")
    if not bool((np.isfinite(array.real) & np.isfinite(array.imag)).all()):
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.complex128, copy=False)


def _paired_complex_matrices(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    prediction_array = _complex_matrix(prediction, "prediction")
    target_array = _complex_matrix(target, "target")
    if prediction_array.shape != target_array.shape:
        raise ValueError(
            "prediction and target shapes differ: "
            f"{prediction_array.shape} vs {target_array.shape}"
        )
    return prediction_array, target_array


def log_magnitude_image(
    values: np.ndarray,
    *,
    reference_peak: float,
    floor_db: float = -60.0,
) -> np.ndarray:
    """Return a [0, 1] log-magnitude image using one externally shared peak."""

    array = _complex_matrix(values, "values")
    if not math.isfinite(reference_peak) or reference_peak <= 0:
        raise ValueError("reference_peak must be finite and positive")
    if not math.isfinite(floor_db) or floor_db >= 0:
        raise ValueError("floor_db must be finite and negative")

    relative_floor = 10.0 ** (floor_db / 20.0)
    relative_magnitude = np.abs(array) / reference_peak
    decibels = 20.0 * np.log10(np.maximum(relative_magnitude, relative_floor))
    clipped = np.clip(decibels, floor_db, 0.0)
    return ((clipped - floor_db) / -floor_db).astype(np.float64, copy=False)


def peak_signal_to_noise_ratio(
    prediction: np.ndarray, target: np.ndarray, *, data_range: float = 1.0
) -> float:
    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction_array.size == 0:
        raise ValueError("PSNR requires at least one value")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if not bool(np.isfinite(prediction_array).all() and np.isfinite(target_array).all()):
        raise ValueError("PSNR inputs must be finite")

    mean_squared_error = float(np.mean((prediction_array - target_array) ** 2))
    if mean_squared_error == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range**2 / mean_squared_error)


def structural_similarity(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute single-channel SSIM with an 11x11 Gaussian valid region."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction_array.ndim != 2:
        raise ValueError("SSIM inputs must be two-dimensional")
    if min(prediction_array.shape) < window_size:
        raise ValueError(
            f"SSIM inputs must be at least {window_size}x{window_size}, "
            f"got {prediction_array.shape}"
        )
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    if not math.isfinite(data_range) or data_range <= 0:
        raise ValueError("data_range must be finite and positive")
    if not bool(np.isfinite(prediction_array).all() and np.isfinite(target_array).all()):
        raise ValueError("SSIM inputs must be finite")

    radius = window_size // 2
    truncate = radius / sigma
    mu_prediction = gaussian_filter(prediction_array, sigma=sigma, truncate=truncate)
    mu_target = gaussian_filter(target_array, sigma=sigma, truncate=truncate)
    mu_prediction_sq = mu_prediction**2
    mu_target_sq = mu_target**2
    mu_cross = mu_prediction * mu_target

    sigma_prediction_sq = (
        gaussian_filter(prediction_array**2, sigma=sigma, truncate=truncate)
        - mu_prediction_sq
    )
    sigma_target_sq = (
        gaussian_filter(target_array**2, sigma=sigma, truncate=truncate) - mu_target_sq
    )
    sigma_cross = (
        gaussian_filter(prediction_array * target_array, sigma=sigma, truncate=truncate)
        - mu_cross
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_cross + c1) * (2.0 * sigma_cross + c2)
    denominator = (mu_prediction_sq + mu_target_sq + c1) * (
        sigma_prediction_sq + sigma_target_sq + c2
    )
    score_map = numerator / denominator
    valid = score_map[radius:-radius, radius:-radius]
    return float(valid.mean())


def evaluate_complex_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    target_peak: float | None = None,
    floor_db: float = -60.0,
) -> dict[str, float]:
    """Evaluate one complex prediction against one complex target matrix."""

    prediction_array, target_array = _paired_complex_matrices(prediction, target)
    difference = prediction_array - target_array
    normalized_complex_rmse = math.sqrt(float(np.mean(np.abs(difference) ** 2)) / 2.0)

    prediction_norm = float(np.linalg.norm(prediction_array.ravel()))
    target_norm = float(np.linalg.norm(target_array.ravel()))
    coherence_denominator = prediction_norm * target_norm
    complex_coherence = (
        float(abs(np.vdot(prediction_array.ravel(), target_array.ravel())))
        / coherence_denominator
        if coherence_denominator > 0
        else 0.0
    )

    prediction_magnitude = np.abs(prediction_array).ravel()
    target_magnitude = np.abs(target_array).ravel()
    prediction_centered = prediction_magnitude - prediction_magnitude.mean()
    target_centered = target_magnitude - target_magnitude.mean()
    magnitude_denominator = float(
        np.linalg.norm(prediction_centered) * np.linalg.norm(target_centered)
    )
    magnitude_correlation = (
        float(np.dot(prediction_centered, target_centered)) / magnitude_denominator
        if magnitude_denominator > 0
        else 0.0
    )

    prediction_rms = math.sqrt(float(np.mean(np.abs(prediction_array) ** 2)))
    target_rms = math.sqrt(float(np.mean(np.abs(target_array) ** 2)))
    rms_ratio = prediction_rms / target_rms if target_rms > 0 else math.inf

    resolved_target_peak = (
        float(np.abs(target_array).max()) if target_peak is None else float(target_peak)
    )
    target_log_magnitude = log_magnitude_image(
        target_array, reference_peak=resolved_target_peak, floor_db=floor_db
    )
    prediction_log_magnitude = log_magnitude_image(
        prediction_array, reference_peak=resolved_target_peak, floor_db=floor_db
    )

    return {
        "normalized_complex_rmse": normalized_complex_rmse,
        "complex_coherence": complex_coherence,
        "magnitude_correlation": magnitude_correlation,
        "rms_ratio_target": rms_ratio,
        "log_magnitude_psnr_db": peak_signal_to_noise_ratio(
            prediction_log_magnitude, target_log_magnitude
        ),
        "log_magnitude_ssim": structural_similarity(
            prediction_log_magnitude, target_log_magnitude
        ),
    }
