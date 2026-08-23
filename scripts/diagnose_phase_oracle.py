"""E007: measure per-patch phase-correction oracle ceilings on Scene4."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import yaml
from scipy.optimize import minimize

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.diagnose_shared_complex_filter import (
    _evenly_spaced,
    evaluate_focus_prediction,
    load_normalized_pair,
    summarize_metrics,
)
from scripts.overfit_single_patch import write_json
from swinir.sar_dataset import (
    CoordinateRegion,
    DatasetManifest,
    PairRecord,
    SplitName,
    build_manifest,
)
from swinir.sar_metrics import log_magnitude_image


REPORT_SCHEMA_VERSION = 1
METHODS = (
    "echo_identity",
    "shift_oracle",
    "quadratic_phase_oracle",
    "unrestricted_phase_oracle",
)


@dataclass(frozen=True)
class PhaseFit:
    coefficients: tuple[float, float, float, float, float, float]
    initial_objective: float
    final_objective: float
    optimizer_success: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int
    initial_shift: tuple[int, int]
    selected_frequency_count: int


@dataclass(frozen=True)
class AuditSelection:
    filename: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate target-informed per-patch shift, quadratic-phase, and "
            "unrestricted phase-only oracle ceilings."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/diagnose_phase_oracle.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume an interrupted run from a complete phase_fits.json cache. "
            "The dataset fingerprint and oracle selection must match."
        ),
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be a mapping")
    for section in ("data", "phase", "evaluation", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    phase = config["phase"]
    evaluation = config["evaluation"]
    runtime = config["runtime"]
    shape = tuple(int(value) for value in data["expected_shape"])
    if len(shape) != 2 or min(shape) < 11:
        raise ValueError("data.expected_shape must contain two dimensions >= 11")
    if int(data["oracle_sample_count"]) <= 0:
        raise ValueError("oracle_sample_count must be positive")
    if phase.get("fft_norm") not in ("ortho", "backward", "forward"):
        raise ValueError("phase.fft_norm is unsupported")
    for name in (
        "maximum_shift_pixels",
        "maximum_frequency_samples",
        "optimizer_max_iterations",
    ):
        if int(phase[name]) <= 0:
            raise ValueError(f"phase.{name} must be positive")
    for name in (
        "quadratic_coefficient_bound_radians",
        "optimizer_ftol",
        "optimizer_gtol",
    ):
        value = float(phase[name])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"phase.{name} must be finite and positive")
    radius = float(evaluation["high_frequency_radius_fraction"])
    if not 0 < radius < math.sqrt(0.5):
        raise ValueError("high_frequency_radius_fraction is outside the FFT radius")
    if int(evaluation["audit_sample_count"]) <= 0:
        raise ValueError("audit_sample_count must be positive")
    if float(evaluation["log_magnitude_floor_db"]) >= 0:
        raise ValueError("log_magnitude_floor_db must be negative")
    for name in ("progress_interval_samples", "figure_dpi", "contact_sheet_page_size"):
        if int(runtime[name]) <= 0:
            raise ValueError(f"runtime.{name} must be positive")


def select_spatial_oracle_records(
    records: Sequence[PairRecord], sample_count: int
) -> tuple[PairRecord, ...]:
    if not records:
        raise ValueError("oracle selection requires validation records")
    if not 0 < sample_count <= len(records):
        raise ValueError(f"oracle sample count must be in [1, {len(records)}]")
    lookup = {(record.row, record.col): record for record in records}
    rows = sorted({record.row for record in records})
    cols = sorted({record.col for record in records})
    row_count = max(1, int(math.floor(math.sqrt(sample_count))))
    col_count = int(math.ceil(sample_count / row_count))
    selected_rows = _evenly_spaced(rows, min(row_count, len(rows)))
    selected_cols = _evenly_spaced(cols, min(col_count, len(cols)))
    selected = [
        lookup[(row, col)]
        for row in selected_rows
        for col in selected_cols
        if (row, col) in lookup
    ]
    if len(selected) < sample_count:
        selected_names = {record.echo_path.name for record in selected}
        for record in _evenly_spaced(
            sorted(records, key=lambda item: (item.row, item.col, item.key)),
            len(records),
        ):
            if record.echo_path.name not in selected_names:
                selected.append(record)
                selected_names.add(record.echo_path.name)
            if len(selected) == sample_count:
                break
    return tuple(selected[:sample_count])


def optimal_complex_gain(prediction: np.ndarray, target: np.ndarray) -> complex:
    denominator = float(np.sum(np.abs(prediction) ** 2))
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0j
    return complex(np.sum(target * np.conj(prediction)) / denominator)


def estimate_magnitude_shift(
    echo: np.ndarray, target: np.ndarray, maximum_shift: int
) -> tuple[int, int]:
    if echo.shape != target.shape:
        raise ValueError("Echo and target shapes must match")
    source = np.log1p(np.abs(echo))
    destination = np.log1p(np.abs(target))
    source -= float(source.mean())
    destination -= float(destination.mean())
    correlation = np.fft.fftshift(
        np.abs(
            np.fft.ifft2(
                np.fft.fft2(destination) * np.conj(np.fft.fft2(source))
            )
        )
    )
    center = np.asarray(correlation.shape) // 2
    row_low = max(0, int(center[0]) - maximum_shift)
    row_high = min(correlation.shape[0], int(center[0]) + maximum_shift + 1)
    col_low = max(0, int(center[1]) - maximum_shift)
    col_high = min(correlation.shape[1], int(center[1]) + maximum_shift + 1)
    window = correlation[row_low:row_high, col_low:col_high]
    local = np.unravel_index(int(np.argmax(window)), window.shape)
    row_index = row_low + int(local[0])
    col_index = col_low + int(local[1])
    return row_index - int(center[0]), col_index - int(center[1])


def shift_oracle(
    echo: np.ndarray, target: np.ndarray, maximum_shift: int
) -> tuple[np.ndarray, tuple[int, int], complex]:
    shift = estimate_magnitude_shift(echo, target, maximum_shift)
    shifted = np.roll(echo, shift=shift, axis=(0, 1))
    gain = optimal_complex_gain(shifted, target)
    return gain * shifted, shift, gain


def _frequency_terms(shape: tuple[int, int]) -> tuple[np.ndarray, ...]:
    row = np.fft.fftfreq(shape[0])[:, None]
    col = np.fft.fftfreq(shape[1])[None, :]
    row_grid = np.broadcast_to(row, shape)
    col_grid = np.broadcast_to(col, shape)
    return (
        np.ones(shape, dtype=np.float64),
        row_grid,
        col_grid,
        row_grid**2,
        row_grid * col_grid,
        col_grid**2,
    )


def apply_phase_polynomial(
    echo: np.ndarray,
    coefficients: Sequence[float],
    *,
    fft_norm: str,
) -> np.ndarray:
    if len(coefficients) != 6:
        raise ValueError("phase polynomial requires six coefficients")
    terms = _frequency_terms(echo.shape)
    phase = sum(float(coefficient) * term for coefficient, term in zip(coefficients, terms))
    spectrum = np.fft.fft2(echo, norm=fft_norm)
    return np.fft.ifft2(np.exp(1j * phase) * spectrum, norm=fft_norm)


def fit_quadratic_phase(
    echo: np.ndarray,
    target: np.ndarray,
    *,
    fft_norm: str,
    maximum_shift: int,
    maximum_frequency_samples: int,
    quadratic_bound: float,
    maximum_iterations: int,
    ftol: float,
    gtol: float,
) -> PhaseFit:
    if echo.shape != target.shape:
        raise ValueError("Echo and target shapes must match")
    echo_spectrum = np.fft.fft2(echo, norm=fft_norm)
    target_spectrum = np.fft.fft2(target, norm=fft_norm)
    cross = target_spectrum * np.conj(echo_spectrum)
    weights = np.abs(echo_spectrum) * np.abs(target_spectrum)
    flat_weights = weights.ravel()
    positive = np.flatnonzero(flat_weights > np.finfo(np.float64).tiny)
    if positive.size == 0:
        raise ValueError("phase fitting requires nonzero cross-spectral energy")
    selected_count = min(int(maximum_frequency_samples), int(positive.size))
    if selected_count < positive.size:
        candidate_weights = flat_weights[positive]
        chosen = np.argpartition(candidate_weights, -selected_count)[-selected_count:]
        selected = positive[chosen]
    else:
        selected = positive
    terms = _frequency_terms(echo.shape)
    design = np.column_stack([term.ravel()[selected] for term in terms])
    observed_phase = np.angle(cross.ravel()[selected])
    selected_weights = flat_weights[selected].astype(np.float64, copy=True)
    selected_weights /= float(selected_weights.sum())

    initial_shift = estimate_magnitude_shift(echo, target, maximum_shift)
    initial = np.zeros(6, dtype=np.float64)
    initial[1] = -2.0 * np.pi * initial_shift[0]
    initial[2] = -2.0 * np.pi * initial_shift[1]
    residual_without_constant = observed_phase - design[:, 1:] @ initial[1:]
    circular_mean = np.sum(selected_weights * np.exp(1j * residual_without_constant))
    initial[0] = float(np.angle(circular_mean))

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        residual = observed_phase - design @ coefficients
        value = float(np.sum(selected_weights * (1.0 - np.cos(residual))))
        gradient = -(design.T @ (selected_weights * np.sin(residual)))
        return value, gradient

    initial_objective = objective(initial)[0]
    linear_bound = 2.0 * np.pi * maximum_shift
    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=(
            (-np.pi, np.pi),
            (-linear_bound, linear_bound),
            (-linear_bound, linear_bound),
            (-quadratic_bound, quadratic_bound),
            (-quadratic_bound, quadratic_bound),
            (-quadratic_bound, quadratic_bound),
        ),
        options={"maxiter": maximum_iterations, "ftol": ftol, "gtol": gtol},
    )
    coefficients = tuple(float(value) for value in result.x)
    return PhaseFit(
        coefficients=coefficients,  # type: ignore[arg-type]
        initial_objective=initial_objective,
        final_objective=float(result.fun),
        optimizer_success=bool(result.success),
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_iterations=int(result.nit),
        initial_shift=initial_shift,
        selected_frequency_count=selected_count,
    )


def quadratic_phase_oracle(
    echo: np.ndarray,
    target: np.ndarray,
    **fit_kwargs: Any,
) -> tuple[np.ndarray, PhaseFit, complex]:
    fit = fit_quadratic_phase(echo, target, **fit_kwargs)
    prediction = apply_phase_polynomial(
        echo, fit.coefficients, fft_norm=str(fit_kwargs["fft_norm"])
    )
    gain = optimal_complex_gain(prediction, target)
    return gain * prediction, fit, gain


def unrestricted_phase_oracle(
    echo: np.ndarray, target: np.ndarray, *, fft_norm: str
) -> tuple[np.ndarray, complex]:
    echo_spectrum = np.fft.fft2(echo, norm=fft_norm)
    target_spectrum = np.fft.fft2(target, norm=fft_norm)
    cross = target_spectrum * np.conj(echo_spectrum)
    correction = np.ones_like(cross, dtype=np.complex128)
    nonzero = np.abs(cross) > np.finfo(np.float64).tiny
    correction[nonzero] = cross[nonzero] / np.abs(cross[nonzero])
    prediction = np.fft.ifft2(correction * echo_spectrum, norm=fft_norm)
    gain = optimal_complex_gain(prediction, target)
    return gain * prediction, gain


def _method_comparison(
    metrics: dict[str, dict[str, dict[str, float]]],
    summaries: dict[str, dict[str, Any]],
    method: str,
) -> dict[str, float]:
    echo = summaries["echo_identity"]["aggregate"]
    candidate = summaries[method]["aggregate"]
    filenames = tuple(metrics[method])
    return {
        "rmse_win_fraction_vs_echo": float(
            np.mean(
                [
                    metrics[method][name]["normalized_complex_rmse"]
                    < metrics["echo_identity"][name]["normalized_complex_rmse"]
                    for name in filenames
                ]
            )
        ),
        "mean_complex_coherence_delta_vs_echo": float(
            candidate["complex_coherence"]["mean"]
            - echo["complex_coherence"]["mean"]
        ),
        "mean_log_ssim_delta_vs_echo": float(
            candidate["log_magnitude_ssim"]["mean"]
            - echo["log_magnitude_ssim"]["mean"]
        ),
        "mean_edge_correlation_delta_vs_echo": float(
            candidate["edge_correlation"]["mean"]
            - echo["edge_correlation"]["mean"]
        ),
        "median_high_frequency_energy_ratio": float(
            candidate["high_frequency_energy_ratio"]["median"]
        ),
    }


def compare_oracles(
    metrics: dict[str, dict[str, dict[str, float]]],
    summaries: dict[str, dict[str, Any]],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    unrestricted = _method_comparison(
        metrics, summaries, "unrestricted_phase_oracle"
    )
    quadratic = _method_comparison(metrics, summaries, "quadratic_phase_oracle")
    unrestricted_ssim_gain = unrestricted["mean_log_ssim_delta_vs_echo"]
    unrestricted_edge_gain = unrestricted["mean_edge_correlation_delta_vs_echo"]
    quadratic_ssim_fraction = (
        quadratic["mean_log_ssim_delta_vs_echo"] / unrestricted_ssim_gain
        if unrestricted_ssim_gain > 0
        else 0.0
    )
    quadratic_edge_fraction = (
        quadratic["mean_edge_correlation_delta_vs_echo"] / unrestricted_edge_gain
        if unrestricted_edge_gain > 0
        else 0.0
    )
    unrestricted_checks = {
        "rmse_win_fraction_vs_echo": unrestricted["rmse_win_fraction_vs_echo"]
        >= float(criteria["unrestricted_rmse_win_fraction_vs_echo_min"]),
        "mean_complex_coherence_delta_vs_echo": unrestricted[
            "mean_complex_coherence_delta_vs_echo"
        ]
        >= float(criteria["unrestricted_mean_coherence_delta_vs_echo_min"]),
        "mean_log_ssim_delta_vs_echo": unrestricted[
            "mean_log_ssim_delta_vs_echo"
        ]
        >= float(criteria["unrestricted_mean_log_ssim_delta_vs_echo_min"]),
        "mean_edge_correlation_delta_vs_echo": unrestricted[
            "mean_edge_correlation_delta_vs_echo"
        ]
        >= float(criteria["unrestricted_mean_edge_correlation_delta_vs_echo_min"]),
        "median_high_frequency_energy_ratio": float(
            criteria["unrestricted_median_high_frequency_energy_ratio_min"]
        )
        <= unrestricted["median_high_frequency_energy_ratio"]
        <= float(criteria["unrestricted_median_high_frequency_energy_ratio_max"]),
    }
    quadratic_checks = {
        "rmse_win_fraction_vs_echo": quadratic["rmse_win_fraction_vs_echo"]
        >= float(criteria["quadratic_rmse_win_fraction_vs_echo_min"]),
        "mean_complex_coherence_delta_vs_echo": quadratic[
            "mean_complex_coherence_delta_vs_echo"
        ]
        >= float(criteria["quadratic_mean_coherence_delta_vs_echo_min"]),
        "mean_log_ssim_delta_vs_echo": quadratic[
            "mean_log_ssim_delta_vs_echo"
        ]
        >= float(criteria["quadratic_mean_log_ssim_delta_vs_echo_min"]),
        "mean_edge_correlation_delta_vs_echo": quadratic[
            "mean_edge_correlation_delta_vs_echo"
        ]
        >= float(criteria["quadratic_mean_edge_correlation_delta_vs_echo_min"]),
        "median_high_frequency_energy_ratio": float(
            criteria["quadratic_median_high_frequency_energy_ratio_min"]
        )
        <= quadratic["median_high_frequency_energy_ratio"]
        <= float(criteria["quadratic_median_high_frequency_energy_ratio_max"]),
        "fraction_of_unrestricted_ssim_gain": quadratic_ssim_fraction
        >= float(criteria["quadratic_fraction_of_unrestricted_ssim_gain_min"]),
        "fraction_of_unrestricted_edge_gain": quadratic_edge_fraction
        >= float(criteria["quadratic_fraction_of_unrestricted_edge_gain_min"]),
    }
    unrestricted["checks"] = unrestricted_checks
    unrestricted["metric_supported"] = all(unrestricted_checks.values())
    quadratic["fraction_of_unrestricted_ssim_gain"] = quadratic_ssim_fraction
    quadratic["fraction_of_unrestricted_edge_gain"] = quadratic_edge_fraction
    quadratic["checks"] = quadratic_checks
    quadratic["metric_supported"] = all(quadratic_checks.values())
    return {"unrestricted_phase": unrestricted, "quadratic_phase": quadratic}


def select_audit_samples(
    records: Sequence[PairRecord],
    metrics: dict[str, dict[str, dict[str, float]]],
    sample_count: int,
) -> tuple[AuditSelection, ...]:
    if not 0 < sample_count <= len(records):
        raise ValueError(f"audit sample count must be in [1, {len(records)}]")
    selected: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(filename: str, reason: str) -> None:
        if filename not in reasons:
            reasons[filename] = []
            selected.append(filename)
        reasons[filename].append(reason)

    for method, label in (
        ("quadratic_phase_oracle", "quadratic"),
        ("unrestricted_phase_oracle", "unrestricted"),
    ):
        for metric, metric_label in (
            ("normalized_complex_rmse", "rmse"),
            ("edge_correlation", "edge"),
        ):
            ordered = sorted(
                metrics[method],
                key=lambda name: (metrics[method][name][metric], name),
            )
            add(ordered[0], f"best_{label}_{metric_label}")
            add(ordered[-1], f"worst_{label}_{metric_label}")
    spatial = sorted(records, key=lambda item: (item.row, item.col, item.key))
    for index, record in enumerate(_evenly_spaced(spatial, sample_count * 3)):
        add(record.echo_path.name, f"spatial_quantile_{index:02d}")
        if len(selected) >= sample_count:
            break
    return tuple(
        AuditSelection(filename=name, reasons=tuple(reasons[name]))
        for name in selected[:sample_count]
    )


def _display(values: np.ndarray, peak: float, floor_db: float) -> np.ndarray:
    return log_magnitude_image(values, reference_peak=peak, floor_db=floor_db)


def export_sample_figure(
    values: tuple[np.ndarray, ...],
    path: Path,
    *,
    filename: str,
    reasons: Sequence[str],
    metrics: dict[str, dict[str, float]],
    phase_fit: PhaseFit,
    floor_db: float,
    dpi: int,
) -> None:
    titles = (
        "Echo",
        "Shift oracle",
        "Quadratic phase oracle",
        "Unrestricted phase oracle",
        "Image target",
    )
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    target_peak = max(float(np.abs(values[-1]).max()), np.finfo(np.float64).tiny)
    for column, (array, title) in enumerate(zip(values, titles, strict=True)):
        own_peak = max(float(np.abs(array).max()), np.finfo(np.float64).tiny)
        axes[0, column].imshow(_display(array, own_peak, floor_db), cmap="gray", vmin=0, vmax=1)
        axes[1, column].imshow(_display(array, target_peak, floor_db), cmap="gray", vmin=0, vmax=1)
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Independent peak")
    axes[1, 0].set_ylabel("Shared Image peak")
    quadratic = metrics["quadratic_phase_oracle"]
    unrestricted = metrics["unrestricted_phase_oracle"]
    figure.suptitle(
        f"{filename}  selection={','.join(reasons)}\n"
        f"quadratic: RMSE={quadratic['normalized_complex_rmse']:.3f}, "
        f"SSIM={quadratic['log_magnitude_ssim']:.3f}, edge={quadratic['edge_correlation']:.3f}; "
        f"unrestricted: RMSE={unrestricted['normalized_complex_rmse']:.3f}, "
        f"SSIM={unrestricted['log_magnitude_ssim']:.3f}, edge={unrestricted['edge_correlation']:.3f}; "
        f"phase loss={phase_fit.initial_objective:.3f}->{phase_fit.final_objective:.3f}",
        fontsize=9,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[tuple[str, tuple[np.ndarray, ...], dict[str, dict[str, float]], Sequence[str]]],
    path: Path,
    *,
    floor_db: float,
    dpi: int,
) -> None:
    titles = ("Echo", "Shift", "Quadratic", "Unrestricted", "Image")
    figure, axes = plt.subplots(
        len(rows), 5, figsize=(18, 3.1 * len(rows)), constrained_layout=True, squeeze=False
    )
    for row_index, (filename, values, metrics, reasons) in enumerate(rows):
        target_peak = max(float(np.abs(values[-1]).max()), np.finfo(np.float64).tiny)
        for column, array in enumerate(values):
            axes[row_index, column].imshow(
                _display(array, target_peak, floor_db), cmap="gray", vmin=0, vmax=1
            )
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column])
        quadratic = metrics["quadratic_phase_oracle"]
        unrestricted = metrics["unrestricted_phase_oracle"]
        axes[row_index, 0].set_ylabel(
            f"{Path(filename).stem}\nQ edge={quadratic['edge_correlation']:.2f} "
            f"U edge={unrestricted['edge_correlation']:.2f}\n{','.join(reasons)}",
            fontsize=7,
        )
    figure.suptitle(
        "E007 target-informed phase oracle audit\nAll columns share the Image target peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_phase_diagnostics(
    fits: dict[str, PhaseFit],
    metrics: dict[str, dict[str, dict[str, float]]],
    path: Path,
    *,
    dpi: int,
) -> None:
    names = tuple(fits)
    coefficients = np.asarray([fits[name].coefficients for name in names])
    losses = np.asarray(
        [[fits[name].initial_objective, fits[name].final_objective] for name in names]
    )
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    axes[0, 0].scatter(coefficients[:, 1], coefficients[:, 2], s=18, alpha=0.7)
    axes[0, 0].set_title("Fitted linear phase coefficients")
    axes[0, 0].set_xlabel("row-frequency coefficient")
    axes[0, 0].set_ylabel("col-frequency coefficient")
    axes[0, 1].boxplot([coefficients[:, index] for index in (3, 4, 5)])
    axes[0, 1].set_xticks([1, 2, 3])
    axes[0, 1].set_xticklabels(["fr²", "fr fc", "fc²"])
    axes[0, 1].set_title("Quadratic phase coefficients")
    axes[1, 0].scatter(losses[:, 0], losses[:, 1], s=18, alpha=0.7)
    limit = max(float(losses.max()), np.finfo(np.float64).eps)
    axes[1, 0].plot([0, limit], [0, limit], "--", color="gray")
    axes[1, 0].set_xlabel("initial circular loss")
    axes[1, 0].set_ylabel("final circular loss")
    axes[1, 0].set_title("Quadratic phase fit objective")
    q_edge = [metrics["quadratic_phase_oracle"][name]["edge_correlation"] for name in names]
    u_edge = [metrics["unrestricted_phase_oracle"][name]["edge_correlation"] for name in names]
    axes[1, 1].scatter(q_edge, u_edge, s=18, alpha=0.7)
    axes[1, 1].set_xlabel("quadratic phase edge correlation")
    axes[1, 1].set_ylabel("unrestricted phase edge correlation")
    axes[1, 1].set_title("Low-order versus phase-only ceiling")
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _resolved_config(
    config: dict[str, Any], args: argparse.Namespace, manifest: DatasetManifest
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["schema_version"] = 1
    result["config_file"] = str(args.config.resolve())
    result["data"]["echo_dir"] = str(args.echo_dir.resolve())
    result["data"]["image_dir"] = str(args.image_dir.resolve())
    result["data"]["manifest_fingerprint"] = manifest.fingerprint
    return result


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"resume file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"resume file must contain a JSON object: {path}")
    return payload


def _phase_fit_from_json(payload: dict[str, Any]) -> PhaseFit:
    return PhaseFit(
        coefficients=tuple(float(value) for value in payload["coefficients"]),  # type: ignore[arg-type]
        initial_objective=float(payload["initial_objective"]),
        final_objective=float(payload["final_objective"]),
        optimizer_success=bool(payload["optimizer_success"]),
        optimizer_status=int(payload["optimizer_status"]),
        optimizer_message=str(payload["optimizer_message"]),
        optimizer_iterations=int(payload["optimizer_iterations"]),
        initial_shift=tuple(int(value) for value in payload["initial_shift"]),  # type: ignore[arg-type]
        selected_frequency_count=int(payload["selected_frequency_count"]),
    )


def _write_phase_fit_cache(
    path: Path,
    phase_fits: dict[str, PhaseFit],
    oracle_metadata: dict[str, dict[str, Any]],
) -> None:
    write_json(
        path,
        {
            "model": "c0+c1*fr+c2*fc+c3*fr^2+c4*fr*fc+c5*fc^2",
            "fits": {
                name: {**asdict(fit), **oracle_metadata.get(name, {})}
                for name, fit in phase_fits.items()
            },
        },
    )


def load_resume_cache(
    output_dir: Path,
    manifest: DatasetManifest,
    oracle_records: Sequence[PairRecord],
) -> tuple[dict[str, PhaseFit], dict[str, dict[str, Any]]]:
    resolved = _load_json(output_dir / "resolved_config.json")
    cached_fingerprint = resolved.get("data", {}).get("manifest_fingerprint")
    if cached_fingerprint != manifest.fingerprint:
        raise RuntimeError(
            "resume dataset fingerprint does not match the interrupted run"
        )
    selection = _load_json(output_dir / "oracle_selection.json")
    cached_names = [sample.get("filename") for sample in selection.get("samples", [])]
    expected_names = [record.echo_path.name for record in oracle_records]
    if cached_names != expected_names:
        raise RuntimeError("resume oracle sample selection does not match")
    cache = _load_json(output_dir / "phase_fits.json")
    raw_fits = cache.get("fits")
    if not isinstance(raw_fits, dict):
        raise ValueError("phase_fits.json is missing the fits mapping")
    missing = sorted(set(expected_names) - set(raw_fits))
    unexpected = sorted(set(raw_fits) - set(expected_names))
    if missing or unexpected:
        raise RuntimeError(
            "resume phase-fit cache is incomplete or inconsistent: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    phase_fits = {
        name: _phase_fit_from_json(raw_fits[name]) for name in expected_names
    }
    metadata = {
        name: {
            key: raw_fits[name][key]
            for key in (
                "shift",
                "shift_gain",
                "quadratic_gain",
                "unrestricted_gain",
            )
            if key in raw_fits[name]
        }
        for name in expected_names
    }
    return phase_fits, metadata


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_config(config)
    if not args.echo_dir.is_dir() or not args.image_dir.is_dir():
        raise FileNotFoundError("Echo and Image directories must both exist")
    resume = bool(getattr(args, "resume", False))
    if args.output_dir.exists() and not resume:
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new directory"
        )
    if resume and not args.output_dir.is_dir():
        raise FileNotFoundError(
            f"resume output directory does not exist: {args.output_dir}"
        )
    data = config["data"]
    phase = config["phase"]
    evaluation = config["evaluation"]
    runtime = config["runtime"]
    expected_shape = tuple(int(value) for value in data["expected_shape"])
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**data["validation_region"]),
        CoordinateRegion(**data["guard_region"]),
        expected_counts=data["expected_split_counts"],
    )
    validation_records = manifest.records_for(SplitName.VALIDATION)
    oracle_records = select_spatial_oracle_records(
        validation_records, int(data["oracle_sample_count"])
    )
    print(
        f"split train={manifest.split_counts['train']} guard={manifest.split_counts['guard']} "
        f"validation={len(validation_records)} oracle_selected={len(oracle_records)}",
        flush=True,
    )

    if resume:
        phase_fits, oracle_metadata = load_resume_cache(
            args.output_dir, manifest, oracle_records
        )
        print(
            f"resume loaded {len(phase_fits)}/{len(oracle_records)} cached phase fits",
            flush=True,
        )
    else:
        args.output_dir.mkdir(parents=True)
        phase_fits = {}
        oracle_metadata = {}
        write_json(
            args.output_dir / "resolved_config.json",
            _resolved_config(config, args, manifest),
        )
        manifest.write_json(args.output_dir / "split_manifest.json")
        write_json(
            args.output_dir / "oracle_selection.json",
            {
                "sample_count": len(oracle_records),
                "selection": "deterministic spatial grid over validation only",
                "target_usage": "per-sample diagnostic oracle; not a trainable or deployable estimator",
                "samples": [
                    {"filename": record.echo_path.name, "row": record.row, "col": record.col}
                    for record in oracle_records
                ],
            },
        )

    metric_maps: dict[str, dict[str, dict[str, float]]] = {method: {} for method in METHODS}
    floor_db = float(evaluation["log_magnitude_floor_db"])
    radius = float(evaluation["high_frequency_radius_fraction"])
    for index, record in enumerate(oracle_records, start=1):
        echo, image, _ = load_normalized_pair(
            record, expected_shape=expected_shape, rms_epsilon=float(data["rms_epsilon"])
        )
        shift_prediction, shift, shift_gain = shift_oracle(
            echo, image, int(phase["maximum_shift_pixels"])
        )
        filename = record.echo_path.name
        if filename in phase_fits:
            fit = phase_fits[filename]
            quadratic_prediction = apply_phase_polynomial(
                echo, fit.coefficients, fft_norm=str(phase["fft_norm"])
            )
            quadratic_gain = optimal_complex_gain(quadratic_prediction, image)
            quadratic_prediction *= quadratic_gain
        else:
            quadratic_prediction, fit, quadratic_gain = quadratic_phase_oracle(
                echo,
                image,
                fft_norm=str(phase["fft_norm"]),
                maximum_shift=int(phase["maximum_shift_pixels"]),
                maximum_frequency_samples=int(phase["maximum_frequency_samples"]),
                quadratic_bound=float(phase["quadratic_coefficient_bound_radians"]),
                maximum_iterations=int(phase["optimizer_max_iterations"]),
                ftol=float(phase["optimizer_ftol"]),
                gtol=float(phase["optimizer_gtol"]),
            )
        unrestricted_prediction, unrestricted_gain = unrestricted_phase_oracle(
            echo, image, fft_norm=str(phase["fft_norm"])
        )
        phase_fits[filename] = fit
        oracle_metadata[filename] = {
            "shift": list(shift),
            "shift_gain": {"real": shift_gain.real, "imag": shift_gain.imag},
            "quadratic_gain": {"real": quadratic_gain.real, "imag": quadratic_gain.imag},
            "unrestricted_gain": {"real": unrestricted_gain.real, "imag": unrestricted_gain.imag},
        }
        for method, prediction in (
            ("echo_identity", echo),
            ("shift_oracle", shift_prediction),
            ("quadratic_phase_oracle", quadratic_prediction),
            ("unrestricted_phase_oracle", unrestricted_prediction),
        ):
            metric_maps[method][filename] = evaluate_focus_prediction(
                prediction,
                image,
                floor_db=floor_db,
                high_frequency_radius_fraction=radius,
            )
        interval = int(runtime["progress_interval_samples"])
        if index % interval == 0 or index == len(oracle_records):
            _write_phase_fit_cache(
                args.output_dir / "phase_fits.json", phase_fits, oracle_metadata
            )
            source = "cached" if resume else "fitted"
            print(f"oracle {index}/{len(oracle_records)} phase={source}", flush=True)

    summaries = {method: summarize_metrics(values) for method, values in metric_maps.items()}
    comparison = compare_oracles(
        metric_maps, summaries, evaluation["success_criteria"]
    )
    unrestricted_supported = bool(comparison["unrestricted_phase"]["metric_supported"])
    quadratic_supported = bool(comparison["quadratic_phase"]["metric_supported"])
    if not unrestricted_supported:
        status = "phase_only_oracle_not_supported"
    elif quadratic_supported:
        status = "quadratic_phase_oracle_metric_supported"
    else:
        status = "phase_only_oracle_supported_quadratic_not_supported"

    _write_phase_fit_cache(
        args.output_dir / "phase_fits.json", phase_fits, oracle_metadata
    )
    export_phase_diagnostics(
        phase_fits,
        metric_maps,
        args.output_dir / "phase_diagnostics.png",
        dpi=int(runtime["figure_dpi"]),
    )

    audit = select_audit_samples(
        oracle_records, metric_maps, min(int(evaluation["audit_sample_count"]), len(oracle_records))
    )
    audit_dir = args.output_dir / "audit"
    sample_dir = audit_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    records_by_name = {record.echo_path.name: record for record in oracle_records}
    contact_rows = []
    page_size = int(runtime["contact_sheet_page_size"])
    for audit_index, selection in enumerate(audit, start=1):
        record = records_by_name[selection.filename]
        echo, image, _ = load_normalized_pair(
            record, expected_shape=expected_shape, rms_epsilon=float(data["rms_epsilon"])
        )
        shift_prediction, _, _ = shift_oracle(
            echo, image, int(phase["maximum_shift_pixels"])
        )
        fit = phase_fits[selection.filename]
        quadratic_prediction = apply_phase_polynomial(
            echo, fit.coefficients, fft_norm=str(phase["fft_norm"])
        )
        quadratic_prediction *= optimal_complex_gain(quadratic_prediction, image)
        unrestricted_prediction, _ = unrestricted_phase_oracle(
            echo, image, fft_norm=str(phase["fft_norm"])
        )
        values = (echo, shift_prediction, quadratic_prediction, unrestricted_prediction, image)
        sample_metrics = {method: metric_maps[method][selection.filename] for method in METHODS}
        export_sample_figure(
            values,
            sample_dir / f"{Path(selection.filename).stem}.png",
            filename=selection.filename,
            reasons=selection.reasons,
            metrics=sample_metrics,
            phase_fit=fit,
            floor_db=floor_db,
            dpi=int(runtime["figure_dpi"]),
        )
        contact_rows.append((selection.filename, values, sample_metrics, selection.reasons))
        if len(contact_rows) == page_size or audit_index == len(audit):
            page = int(math.ceil(audit_index / page_size))
            export_contact_sheet(
                contact_rows,
                audit_dir / f"audit_page_{page:03d}.png",
                floor_db=floor_db,
                dpi=int(runtime["figure_dpi"]),
            )
            contact_rows = []

    optimizer_success_fraction = float(
        np.mean([fit.optimizer_success for fit in phase_fits.values()])
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "experiment": config.get("experiment"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "interpretation_guardrail": (
            "All phase methods use each target Image at inference time and are oracle ceilings, "
            "not deployable estimators. Metric support still requires manual confirmation of "
            "edge/scatterer refocusing."
        ),
        "dataset": {
            "manifest_fingerprint": manifest.fingerprint,
            "split_counts": manifest.split_counts,
            "validation_count": len(validation_records),
            "oracle_sample_count": len(oracle_records),
        },
        "optimizer_success_fraction": optimizer_success_fraction,
        "summaries": summaries,
        "comparison": comparison,
        "per_sample": metric_maps,
        "phase_fits_file": str((args.output_dir / "phase_fits.json").resolve()),
        "audit": [asdict(selection) for selection in audit],
    }
    write_json(args.output_dir / "report.json", report)
    print(
        f"status={status} unrestricted_metric_supported={unrestricted_supported} "
        f"quadratic_metric_supported={quadratic_supported} "
        f"report={(args.output_dir / 'report.json').resolve()}",
        flush=True,
    )
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
