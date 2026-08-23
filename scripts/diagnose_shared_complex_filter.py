"""E006: diagnose a shared complex frequency filter on a spatial holdout."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import yaml
from scipy import ndimage

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.overfit_single_patch import write_json
from swinir.sar_dataset import (
    CoordinateRegion,
    DatasetManifest,
    PairRecord,
    SplitName,
    build_manifest,
    load_complex_patch,
)
from swinir.sar_metrics import evaluate_complex_prediction, log_magnitude_image


REPORT_SCHEMA_VERSION = 1
FILTER_SCHEMA_VERSION = 1
METRIC_NAMES = (
    "normalized_complex_rmse",
    "complex_coherence",
    "magnitude_correlation",
    "rms_ratio_target",
    "log_magnitude_psnr_db",
    "log_magnitude_ssim",
    "edge_correlation",
    "gradient_energy_ratio",
    "high_frequency_energy_ratio",
)


@dataclass(frozen=True)
class SharedFilter:
    transfer: np.ndarray
    complex_gain: complex
    cross_spectral_coherence: np.ndarray
    cross_spectral_weight: np.ndarray
    ridge: float
    fit_sample_count: int


@dataclass(frozen=True)
class AuditSelection:
    filename: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one shared complex Wiener filter on spatially isolated training "
            "patches and evaluate all validation patches."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/diagnose_shared_complex_filter.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be a mapping")
    for section in ("data", "filter", "evaluation", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    data = config["data"]
    filtering = config["filter"]
    evaluation = config["evaluation"]
    runtime = config["runtime"]
    shape = tuple(int(value) for value in data["expected_shape"])
    if len(shape) != 2 or min(shape) < 11:
        raise ValueError("data.expected_shape must contain two dimensions >= 11")
    if int(data["fit_min_coordinate_spacing"]) <= 0:
        raise ValueError("fit_min_coordinate_spacing must be positive")
    if int(data["fit_max_samples"]) <= 0:
        raise ValueError("fit_max_samples must be positive")
    if filtering.get("fft_norm") not in ("ortho", "backward", "forward"):
        raise ValueError("filter.fft_norm is unsupported")
    ridge_fraction = float(filtering["ridge_fraction_of_mean_power"])
    if not math.isfinite(ridge_fraction) or ridge_fraction <= 0:
        raise ValueError("ridge_fraction_of_mean_power must be finite and positive")
    radius_fraction = float(evaluation["high_frequency_radius_fraction"])
    if not 0 < radius_fraction < math.sqrt(0.5):
        raise ValueError("high_frequency_radius_fraction is outside the FFT radius")
    if int(evaluation["audit_sample_count"]) <= 0:
        raise ValueError("audit_sample_count must be positive")
    floor_db = float(evaluation["log_magnitude_floor_db"])
    if not math.isfinite(floor_db) or floor_db >= 0:
        raise ValueError("log_magnitude_floor_db must be finite and negative")
    if int(runtime["progress_interval_samples"]) <= 0:
        raise ValueError("progress_interval_samples must be positive")
    if int(runtime["figure_dpi"]) <= 0:
        raise ValueError("figure_dpi must be positive")
    if int(runtime["contact_sheet_page_size"]) <= 0:
        raise ValueError("contact_sheet_page_size must be positive")


def _spaced_axis(values: Sequence[int], minimum_spacing: int) -> tuple[int, ...]:
    selected = []
    for value in sorted(set(values)):
        if not selected or value - selected[-1] >= minimum_spacing:
            selected.append(value)
    return tuple(selected)


def _evenly_spaced(items: Sequence[Any], count: int) -> tuple[Any, ...]:
    if count >= len(items):
        return tuple(items)
    indices = np.rint(np.linspace(0, len(items) - 1, count)).astype(int)
    unique_indices = []
    for index in indices:
        if int(index) not in unique_indices:
            unique_indices.append(int(index))
    for index in range(len(items)):
        if len(unique_indices) >= count:
            break
        if index not in unique_indices:
            unique_indices.append(index)
    return tuple(items[index] for index in sorted(unique_indices[:count]))


def select_spatially_separated_fit_records(
    records: Sequence[PairRecord],
    *,
    minimum_spacing: int,
    maximum_samples: int,
) -> tuple[PairRecord, ...]:
    if not records:
        raise ValueError("fit record selection requires training records")
    rows = _spaced_axis([record.row for record in records], minimum_spacing)
    cols = _spaced_axis([record.col for record in records], minimum_spacing)
    row_set = set(rows)
    col_set = set(cols)
    candidates = tuple(
        record
        for record in sorted(records, key=lambda item: (item.row, item.col, item.key))
        if record.row in row_set and record.col in col_set
    )
    if not candidates:
        raise RuntimeError("spatial fit selection produced no samples")
    return _evenly_spaced(candidates, min(maximum_samples, len(candidates)))


def load_normalized_pair(
    record: PairRecord,
    *,
    expected_shape: tuple[int, int],
    rms_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    echo = load_complex_patch(record.echo_path, expected_shape)
    image = load_complex_patch(record.image_path, expected_shape)
    scale = math.sqrt(float(np.mean(np.abs(echo) ** 2)) + rms_epsilon)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid Echo RMS scale for {record.echo_path.name}")
    return echo / scale, image / scale, scale


def fit_shared_filter(
    records: Sequence[PairRecord],
    *,
    expected_shape: tuple[int, int],
    rms_epsilon: float,
    fft_norm: str,
    ridge_fraction: float,
    progress_interval: int = 0,
) -> SharedFilter:
    if not records:
        raise ValueError("shared filter fitting requires at least one record")
    sxx = np.zeros(expected_shape, dtype=np.float64)
    syy = np.zeros(expected_shape, dtype=np.float64)
    syx = np.zeros(expected_shape, dtype=np.complex128)
    gain_numerator = 0.0j
    gain_denominator = 0.0
    for index, record in enumerate(records, start=1):
        echo, image, _ = load_normalized_pair(
            record,
            expected_shape=expected_shape,
            rms_epsilon=rms_epsilon,
        )
        echo_spectrum = np.fft.fft2(echo, norm=fft_norm)
        image_spectrum = np.fft.fft2(image, norm=fft_norm)
        sxx += np.abs(echo_spectrum) ** 2
        syy += np.abs(image_spectrum) ** 2
        syx += image_spectrum * np.conj(echo_spectrum)
        gain_numerator += np.sum(image * np.conj(echo))
        gain_denominator += float(np.sum(np.abs(echo) ** 2))
        if progress_interval > 0 and (
            index % progress_interval == 0 or index == len(records)
        ):
            print(f"fit {index}/{len(records)}", flush=True)

    mean_power = float(sxx.mean())
    ridge = ridge_fraction * mean_power
    if not math.isfinite(ridge) or ridge <= 0:
        raise ValueError("resolved ridge is not finite and positive")
    transfer = syx / (sxx + ridge)
    denominator = sxx * syy
    coherence = np.divide(
        np.abs(syx) ** 2,
        denominator,
        out=np.zeros_like(sxx),
        where=denominator > 0,
    )
    coherence = np.clip(coherence, 0.0, 1.0)
    cross_spectral_weight = np.sqrt(denominator)
    complex_gain = gain_numerator / max(gain_denominator, np.finfo(np.float64).tiny)
    return SharedFilter(
        transfer=transfer,
        complex_gain=complex(complex_gain),
        cross_spectral_coherence=coherence,
        cross_spectral_weight=cross_spectral_weight,
        ridge=ridge,
        fit_sample_count=len(records),
    )


def apply_shared_filter(
    echo: np.ndarray,
    transfer: np.ndarray,
    *,
    fft_norm: str,
) -> np.ndarray:
    if echo.shape != transfer.shape:
        raise ValueError("Echo and transfer function shapes must match")
    spectrum = np.fft.fft2(echo, norm=fft_norm)
    return np.fft.ifft2(transfer * spectrum, norm=fft_norm)


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=np.float64).ravel()
    second_values = np.asarray(second, dtype=np.float64).ravel()
    first_centered = first_values - float(first_values.mean())
    second_centered = second_values - float(second_values.mean())
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    return (
        float(np.dot(first_centered, second_centered) / denominator)
        if denominator > 0
        else 0.0
    )


def _gradient_magnitude(values: np.ndarray) -> np.ndarray:
    row = ndimage.sobel(values, axis=0, mode="reflect")
    col = ndimage.sobel(values, axis=1, mode="reflect")
    return np.hypot(row, col)


def _high_frequency_energy(values: np.ndarray, radius_fraction: float) -> float:
    centered = values - float(values.mean())
    spectrum = np.fft.fftshift(np.fft.fft2(centered, norm="ortho"))
    rows = np.fft.fftshift(np.fft.fftfreq(values.shape[0]))
    cols = np.fft.fftshift(np.fft.fftfreq(values.shape[1]))
    radius = np.hypot(rows[:, None], cols[None, :])
    mask = radius >= radius_fraction
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def evaluate_focus_prediction(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    floor_db: float,
    high_frequency_radius_fraction: float,
) -> dict[str, float]:
    metrics = evaluate_complex_prediction(
        prediction,
        target,
        target_peak=float(np.abs(target).max()),
        floor_db=floor_db,
    )
    target_peak = float(np.abs(target).max())
    prediction_log = log_magnitude_image(
        prediction, reference_peak=target_peak, floor_db=floor_db
    )
    target_log = log_magnitude_image(
        target, reference_peak=target_peak, floor_db=floor_db
    )
    prediction_gradient = _gradient_magnitude(prediction_log)
    target_gradient = _gradient_magnitude(target_log)
    target_gradient_energy = float(np.mean(target_gradient**2))
    prediction_gradient_energy = float(np.mean(prediction_gradient**2))
    target_high_frequency_energy = _high_frequency_energy(
        target_log, high_frequency_radius_fraction
    )
    prediction_high_frequency_energy = _high_frequency_energy(
        prediction_log, high_frequency_radius_fraction
    )
    metrics.update(
        {
            "edge_correlation": _pearson(prediction_gradient, target_gradient),
            "gradient_energy_ratio": (
                prediction_gradient_energy / target_gradient_energy
                if target_gradient_energy > 0
                else math.inf
            ),
            "high_frequency_energy_ratio": (
                prediction_high_frequency_energy / target_high_frequency_energy
                if target_high_frequency_energy > 0
                else math.inf
            ),
        }
    )
    return metrics


def summarize_metrics(
    per_sample: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not per_sample:
        raise ValueError("cannot summarize empty metrics")
    aggregate = {}
    for metric_name in METRIC_NAMES:
        values = np.asarray(
            [metrics[metric_name] for metrics in per_sample.values()], dtype=np.float64
        )
        aggregate[metric_name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return {"sample_count": len(per_sample), "aggregate": aggregate}


def transfer_consistency(shared_filter: SharedFilter) -> dict[str, float]:
    coherence = shared_filter.cross_spectral_coherence
    weights = shared_filter.cross_spectral_weight
    weighted_mean = float(
        np.sum(coherence * weights) / np.sum(weights)
        if float(np.sum(weights)) > 0
        else 0.0
    )
    return {
        "mean": float(coherence.mean()),
        "median": float(np.median(coherence)),
        "p05": float(np.percentile(coherence, 5)),
        "p95": float(np.percentile(coherence, 95)),
        "weighted_mean": weighted_mean,
        "fraction_at_least_0_5": float(np.mean(coherence >= 0.5)),
        "fraction_at_least_0_9": float(np.mean(coherence >= 0.9)),
    }


def compare_filter(
    metrics: dict[str, dict[str, dict[str, float]]],
    summaries: dict[str, dict[str, Any]],
    consistency: dict[str, float],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    echo = summaries["echo_identity"]["aggregate"]
    gain = summaries["complex_gain"]["aggregate"]
    filtered = summaries["shared_filter"]["aggregate"]
    filenames = tuple(metrics["shared_filter"])
    rmse_win_vs_echo = float(
        np.mean(
            [
                metrics["shared_filter"][name]["normalized_complex_rmse"]
                < metrics["echo_identity"][name]["normalized_complex_rmse"]
                for name in filenames
            ]
        )
    )
    rmse_win_vs_gain = float(
        np.mean(
            [
                metrics["shared_filter"][name]["normalized_complex_rmse"]
                < metrics["complex_gain"][name]["normalized_complex_rmse"]
                for name in filenames
            ]
        )
    )
    coherence_delta = float(
        filtered["complex_coherence"]["mean"] - echo["complex_coherence"]["mean"]
    )
    ssim_delta = float(
        filtered["log_magnitude_ssim"]["mean"]
        - echo["log_magnitude_ssim"]["mean"]
    )
    edge_delta_echo = float(
        filtered["edge_correlation"]["mean"] - echo["edge_correlation"]["mean"]
    )
    edge_delta_gain = float(
        filtered["edge_correlation"]["mean"] - gain["edge_correlation"]["mean"]
    )
    high_frequency_ratio = float(
        filtered["high_frequency_energy_ratio"]["median"]
    )
    checks = {
        "rmse_win_fraction_vs_echo": rmse_win_vs_echo
        >= float(criteria["validation_rmse_win_fraction_vs_echo_min"]),
        "rmse_win_fraction_vs_gain": rmse_win_vs_gain
        >= float(criteria["validation_rmse_win_fraction_vs_gain_min"]),
        "complex_coherence_delta_vs_echo": coherence_delta
        >= float(criteria["mean_complex_coherence_delta_vs_echo_min"]),
        "log_ssim_delta_vs_echo": ssim_delta
        >= float(criteria["mean_log_ssim_delta_vs_echo_min"]),
        "edge_correlation_delta_vs_echo": edge_delta_echo
        >= float(criteria["mean_edge_correlation_delta_vs_echo_min"]),
        "edge_correlation_delta_vs_gain": edge_delta_gain
        >= float(criteria["mean_edge_correlation_delta_vs_gain_min"]),
        "median_high_frequency_energy_ratio": float(
            criteria["median_high_frequency_energy_ratio_min"]
        )
        <= high_frequency_ratio
        <= float(criteria["median_high_frequency_energy_ratio_max"]),
        "fit_weighted_transfer_coherence": consistency["weighted_mean"]
        >= float(criteria["fit_weighted_transfer_coherence_min"]),
    }
    return {
        "rmse_win_fraction_vs_echo": rmse_win_vs_echo,
        "rmse_win_fraction_vs_gain": rmse_win_vs_gain,
        "mean_complex_coherence_delta_vs_echo": coherence_delta,
        "mean_log_ssim_delta_vs_echo": ssim_delta,
        "mean_edge_correlation_delta_vs_echo": edge_delta_echo,
        "mean_edge_correlation_delta_vs_gain": edge_delta_gain,
        "median_high_frequency_energy_ratio": high_frequency_ratio,
        "checks": checks,
        "metric_supported": all(checks.values()),
    }


def select_audit_samples(
    records: Sequence[PairRecord],
    filter_metrics: dict[str, dict[str, float]],
    sample_count: int,
) -> tuple[AuditSelection, ...]:
    if not 0 < sample_count <= len(records):
        raise ValueError(f"audit sample count must be in [1, {len(records)}]")
    records_by_name = {record.echo_path.name: record for record in records}
    if records_by_name.keys() != filter_metrics.keys():
        raise RuntimeError("filter metrics do not match validation records")
    selected: list[str] = []
    reasons: dict[str, list[str]] = {}

    def add(filename: str, reason: str) -> None:
        if filename not in reasons:
            reasons[filename] = []
            selected.append(filename)
        reasons[filename].append(reason)

    for metric, lower_label, upper_label in (
        ("normalized_complex_rmse", "best_complex_rmse", "worst_complex_rmse"),
        ("edge_correlation", "lowest_edge_correlation", "highest_edge_correlation"),
        (
            "high_frequency_energy_ratio",
            "lowest_high_frequency_ratio",
            "highest_high_frequency_ratio",
        ),
    ):
        ordered = sorted(
            filter_metrics,
            key=lambda name: (float(filter_metrics[name][metric]), name),
        )
        add(ordered[0], lower_label)
        add(ordered[-1], upper_label)
    spatial_order = sorted(
        records,
        key=lambda record: (record.row, record.col, record.echo_path.name),
    )
    for index, record in enumerate(
        _evenly_spaced(spatial_order, min(len(spatial_order), sample_count * 3))
    ):
        add(record.echo_path.name, f"spatial_quantile_{index:02d}")
        if len(selected) >= sample_count:
            break
    for record in spatial_order:
        if len(selected) >= sample_count:
            break
        add(record.echo_path.name, "spatial_fill")
    return tuple(
        AuditSelection(filename, tuple(reasons[filename]))
        for filename in selected[:sample_count]
    )


def _display(values: np.ndarray, *, peak: float, floor_db: float) -> np.ndarray:
    return log_magnitude_image(values, reference_peak=peak, floor_db=floor_db)


def export_sample_figure(
    values: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    path: Path,
    *,
    filename: str,
    reasons: Sequence[str],
    filter_metrics: dict[str, float],
    floor_db: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    titles = ("Echo", "Fixed complex gain", "Shared frequency filter", "Image target")
    target_peak = float(np.abs(values[-1]).max())
    for column, (array, title) in enumerate(zip(values, titles, strict=True)):
        own_peak = max(float(np.abs(array).max()), np.finfo(np.float64).tiny)
        axes[0, column].imshow(
            _display(array, peak=own_peak, floor_db=floor_db),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[1, column].imshow(
            _display(array, peak=target_peak, floor_db=floor_db),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Independent peak")
    axes[1, 0].set_ylabel("Shared Image peak")
    figure.suptitle(
        f"{filename}  selection={','.join(reasons)}\n"
        f"filter RMSE={filter_metrics['normalized_complex_rmse']:.4f}, "
        f"coherence={filter_metrics['complex_coherence']:.4f}, "
        f"SSIM={filter_metrics['log_magnitude_ssim']:.4f}, "
        f"edge corr={filter_metrics['edge_correlation']:.4f}, "
        f"HF ratio={filter_metrics['high_frequency_energy_ratio']:.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[
        tuple[
            str,
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
            dict[str, float],
            Sequence[str],
        ]
    ],
    path: Path,
    *,
    floor_db: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(16, 3.2 * len(rows)), constrained_layout=True, squeeze=False
    )
    titles = ("Echo", "Fixed complex gain", "Shared filter", "Image target")
    for row_index, (filename, values, metrics, reasons) in enumerate(rows):
        target_peak = float(np.abs(values[-1]).max())
        for column, array in enumerate(values):
            axes[row_index, column].imshow(
                _display(array, peak=target_peak, floor_db=floor_db),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column])
        axes[row_index, 0].set_ylabel(
            f"{Path(filename).stem}\nRMSE={metrics['normalized_complex_rmse']:.3f} "
            f"edge={metrics['edge_correlation']:.3f}\n{','.join(reasons)}",
            fontsize=7,
        )
    figure.suptitle(
        "E006 shared complex frequency-filter audit\n"
        "All columns in each row share the Image target peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_filter_diagnostics(
    shared_filter: SharedFilter,
    path: Path,
    *,
    dpi: int,
) -> None:
    transfer = np.fft.fftshift(shared_filter.transfer)
    coherence = np.fft.fftshift(shared_filter.cross_spectral_coherence)
    magnitude_db = 20.0 * np.log10(
        np.maximum(np.abs(transfer), np.finfo(np.float64).tiny)
    )
    lower, upper = np.percentile(magnitude_db, (1, 99))
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    magnitude_handle = axes[0].imshow(magnitude_db, cmap="viridis", vmin=lower, vmax=upper)
    axes[0].set_title("Shared filter magnitude (dB)")
    phase_handle = axes[1].imshow(np.angle(transfer), cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes[1].set_title("Shared filter phase (radian)")
    coherence_handle = axes[2].imshow(coherence, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("Cross-spectral coherence across fit patches")
    for axis in axes:
        axis.axis("off")
    figure.colorbar(magnitude_handle, ax=axes[0], shrink=0.8)
    figure.colorbar(phase_handle, ax=axes[1], shrink=0.8)
    figure.colorbar(coherence_handle, ax=axes[2], shrink=0.8)
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_config(config)
    if not args.echo_dir.is_dir() or not args.image_dir.is_dir():
        raise FileNotFoundError("Echo and Image directories must both exist")
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new directory"
        )
    data = config["data"]
    filtering = config["filter"]
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
    fit_records = select_spatially_separated_fit_records(
        manifest.records_for(SplitName.TRAIN),
        minimum_spacing=int(data["fit_min_coordinate_spacing"]),
        maximum_samples=int(data["fit_max_samples"]),
    )
    validation_records = manifest.records_for(SplitName.VALIDATION)
    print(
        f"split train={manifest.split_counts['train']} guard={manifest.split_counts['guard']} "
        f"validation={len(validation_records)} fit_selected={len(fit_records)}",
        flush=True,
    )

    args.output_dir.mkdir(parents=True)
    resolved = _resolved_config(config, args, manifest)
    write_json(args.output_dir / "resolved_config.json", resolved)
    manifest.write_json(args.output_dir / "split_manifest.json")
    write_json(
        args.output_dir / "fit_selection.json",
        {
            "sample_count": len(fit_records),
            "minimum_coordinate_spacing": int(data["fit_min_coordinate_spacing"]),
            "samples": [
                {
                    "filename": record.echo_path.name,
                    "row": record.row,
                    "col": record.col,
                }
                for record in fit_records
            ],
        },
    )
    shared_filter = fit_shared_filter(
        fit_records,
        expected_shape=expected_shape,
        rms_epsilon=float(data["rms_epsilon"]),
        fft_norm=str(filtering["fft_norm"]),
        ridge_fraction=float(filtering["ridge_fraction_of_mean_power"]),
        progress_interval=int(runtime["progress_interval_samples"]),
    )
    consistency = transfer_consistency(shared_filter)
    np.savez_compressed(
        args.output_dir / "shared_filter.npz",
        schema_version=np.asarray(FILTER_SCHEMA_VERSION),
        transfer=shared_filter.transfer,
        complex_gain=np.asarray(shared_filter.complex_gain),
        cross_spectral_coherence=shared_filter.cross_spectral_coherence,
        cross_spectral_weight=shared_filter.cross_spectral_weight,
        ridge=np.asarray(shared_filter.ridge),
        fit_sample_count=np.asarray(shared_filter.fit_sample_count),
    )
    export_filter_diagnostics(
        shared_filter,
        args.output_dir / "transfer_diagnostics.png",
        dpi=int(runtime["figure_dpi"]),
    )

    metric_maps: dict[str, dict[str, dict[str, float]]] = {
        "echo_identity": {},
        "complex_gain": {},
        "shared_filter": {},
    }
    floor_db = float(evaluation["log_magnitude_floor_db"])
    radius_fraction = float(evaluation["high_frequency_radius_fraction"])
    for index, record in enumerate(validation_records, start=1):
        echo, image, _ = load_normalized_pair(
            record,
            expected_shape=expected_shape,
            rms_epsilon=float(data["rms_epsilon"]),
        )
        gain_prediction = shared_filter.complex_gain * echo
        filter_prediction = apply_shared_filter(
            echo, shared_filter.transfer, fft_norm=str(filtering["fft_norm"])
        )
        filename = record.echo_path.name
        for method, prediction in (
            ("echo_identity", echo),
            ("complex_gain", gain_prediction),
            ("shared_filter", filter_prediction),
        ):
            metric_maps[method][filename] = evaluate_focus_prediction(
                prediction,
                image,
                floor_db=floor_db,
                high_frequency_radius_fraction=radius_fraction,
            )
        progress_interval = int(runtime["progress_interval_samples"])
        if index % progress_interval == 0 or index == len(validation_records):
            print(f"validation {index}/{len(validation_records)}", flush=True)

    summaries = {
        method: summarize_metrics(per_sample)
        for method, per_sample in metric_maps.items()
    }
    comparison = compare_filter(
        metric_maps,
        summaries,
        consistency,
        evaluation["success_criteria"],
    )
    selections = select_audit_samples(
        validation_records,
        metric_maps["shared_filter"],
        int(evaluation["audit_sample_count"]),
    )
    audit_dir = args.output_dir / "audit"
    samples_dir = audit_dir / "samples"
    samples_dir.mkdir(parents=True)
    validation_by_filename = {
        record.echo_path.name: record for record in validation_records
    }
    contact_rows = []
    audit_entries = []
    for selection_index, selection in enumerate(selections):
        record = validation_by_filename[selection.filename]
        echo, image, _ = load_normalized_pair(
            record,
            expected_shape=expected_shape,
            rms_epsilon=float(data["rms_epsilon"]),
        )
        gain_prediction = shared_filter.complex_gain * echo
        filter_prediction = apply_shared_filter(
            echo, shared_filter.transfer, fft_norm=str(filtering["fft_norm"])
        )
        values = (echo, gain_prediction, filter_prediction, image)
        metrics = metric_maps["shared_filter"][selection.filename]
        figure_name = f"{selection_index:02d}_{Path(selection.filename).stem}.png"
        export_sample_figure(
            values,
            samples_dir / figure_name,
            filename=selection.filename,
            reasons=selection.reasons,
            filter_metrics=metrics,
            floor_db=floor_db,
            dpi=int(runtime["figure_dpi"]),
        )
        contact_rows.append((selection.filename, values, metrics, selection.reasons))
        audit_entries.append(
            {
                "selection_index": selection_index,
                "filename": selection.filename,
                "selection_reasons": list(selection.reasons),
                "figure": str(Path("samples") / figure_name),
                "metrics": {
                    method: metric_maps[method][selection.filename]
                    for method in metric_maps
                },
            }
        )
    page_size = int(runtime["contact_sheet_page_size"])
    contact_sheets = []
    for page_index, start in enumerate(range(0, len(contact_rows), page_size), start=1):
        path = audit_dir / f"audit_page_{page_index:03d}.png"
        export_contact_sheet(
            contact_rows[start : start + page_size],
            path,
            floor_db=floor_db,
            dpi=int(runtime["figure_dpi"]),
        )
        contact_sheets.append(path.name)
    write_json(
        audit_dir / "audit_manifest.json",
        {
            "schema_version": 1,
            "sample_count": len(audit_entries),
            "contact_sheets": contact_sheets,
            "display_normalization": (
                "individual figures show independent and shared Image-peak scales; "
                "contact sheets use the shared Image peak"
            ),
            "samples": audit_entries,
        },
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config.get("experiment"),
        "status": (
            "shared_filter_metric_supported"
            if comparison["metric_supported"]
            else "shared_filter_not_supported"
        ),
        "manual_audit_required": True,
        "split_counts": manifest.split_counts,
        "manifest_fingerprint": manifest.fingerprint,
        "fit": {
            "sample_count": len(fit_records),
            "ridge": shared_filter.ridge,
            "complex_gain": {
                "real": shared_filter.complex_gain.real,
                "imag": shared_filter.complex_gain.imag,
                "magnitude": abs(shared_filter.complex_gain),
                "phase_radian": float(np.angle(shared_filter.complex_gain)),
            },
            "transfer_consistency": consistency,
        },
        "validation": {
            "sample_count": len(validation_records),
            "summaries": summaries,
            "comparison": comparison,
            "per_sample": {
                filename: {
                    method: metric_maps[method][filename]
                    for method in metric_maps
                }
                for filename in metric_maps["shared_filter"]
            },
        },
        "artifacts": {
            "shared_filter": "shared_filter.npz",
            "transfer_diagnostics": "transfer_diagnostics.png",
            "audit_manifest": "audit/audit_manifest.json",
            "audit_contact_sheets": [str(Path("audit") / name) for name in contact_sheets],
        },
    }
    write_json(args.output_dir / "report.json", report)
    print(
        f"status={report['status']} metric_supported={comparison['metric_supported']} "
        f"report={(args.output_dir / 'report.json').resolve()}",
        flush=True,
    )
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
