"""Black-box diagnosis of the processing stage relating SAR Echo and PFA images.

The paired MAT files available to this project contain only complex matrices; they
do not contain angle, frequency, trajectory, or PFA interpolation metadata.  This
tool therefore compares several *hypotheses* on a fit split and reports their
performance on a disjoint holdout split.  It cannot uniquely recover the original
processor or its physical geometry.

The hypotheses are:

* image-domain data, up to orientation/conjugation/gain/translation;
* a one-dimensional FFT intermediate;
* a Cartesian two-dimensional spectrum;
* a polar ``(theta, radial-frequency)`` spectrum followed by approximate PFA
  regridding and a two-dimensional Fourier transform.

Example::

    python scripts/diagnose_pfa_stage.py \
        --echo-dir /data/dyn/capella/output/echo \
        --image-dir /data/dyn/capella/output/image \
        --output-dir pfa_stage_diagnosis
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

try:  # Direct script execution: python scripts/diagnose_pfa_stage.py
    from analyze_sar_dataset import FilePair, discover_pairs, inspect_patch_file
except ImportError:  # Package import from tests or another module.
    from scripts.analyze_sar_dataset import FilePair, discover_pairs, inspect_patch_file


REPORT_SCHEMA_VERSION = 1
STAGE_LABELS = {
    "image_domain": "图像域或仅有简单排列差异",
    "one_dimensional_fft": "仅完成一个维度 FFT/IFFT 的中间阶段",
    "cartesian_2d_spectrum": "笛卡尔二维频谱",
    "polar_frequency": "方向角/慢时间与距离频率构成的极坐标频域",
    "unknown": "未知阶段或缺少必要处理/元数据",
}


@dataclass(frozen=True)
class TransformSpec:
    """One globally fixed Echo-to-Image transformation hypothesis."""

    name: str
    stage: str
    family: str
    axes: tuple[int, ...] = ()
    direction: str = "identity"
    pre_shift: bool = False
    post_shift: bool = False
    theta_axis: int | None = None
    theta_reversed: bool = False
    radial_reversed: bool = False
    theta_span_deg: float | None = None
    radius_center_ratio: float | None = None
    orientation: str = "identity"
    conjugate: bool = False


@dataclass(frozen=True)
class SampleMetrics:
    score: float
    complex_coherence: float
    magnitude_correlation: float
    log_magnitude_correlation: float
    relative_nrmse: float
    gain_real: float
    gain_imag: float
    shift_row: int
    shift_col: int


@dataclass(frozen=True)
class CandidateResult:
    spec: TransformSpec
    split: str
    sample_count: int
    metrics: dict[str, float]
    per_sample: tuple[SampleMetrics, ...]

    @property
    def score(self) -> float:
        return self.metrics["score_median"]


def parse_float_list(raw: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise argparse.ArgumentTypeError(f"{name} values must be finite and positive")
    return values


def evenly_spaced(items: Sequence[Any], requested: int) -> list[Any]:
    if requested <= 0:
        raise ValueError("requested sample count must be positive")
    if requested >= len(items):
        return list(items)
    indices = np.rint(np.linspace(0, len(items) - 1, requested)).astype(int)
    return [items[int(index)] for index in indices]


def split_fit_holdout(pairs: Sequence[FilePair], fit_count: int) -> tuple[list[FilePair], list[FilePair]]:
    """Interleave coordinates so both splits span the sorted dataset."""

    if len(pairs) < 2:
        raise ValueError("at least two selected pairs are required")
    if not 1 <= fit_count < len(pairs):
        raise ValueError("fit_count must leave at least one holdout pair")
    fit_indices = set(
        int(index)
        for index in np.rint(np.linspace(0, len(pairs) - 1, fit_count)).astype(int)
    )
    # Rounding can theoretically duplicate an index; fill deterministically.
    for index in range(len(pairs)):
        if len(fit_indices) >= fit_count:
            break
        fit_indices.add(index)
    fit = [pair for index, pair in enumerate(pairs) if index in fit_indices]
    holdout = [pair for index, pair in enumerate(pairs) if index not in fit_indices]
    return fit, holdout


def load_pair(pair: FilePair) -> tuple[np.ndarray, np.ndarray]:
    echo_info = inspect_patch_file(pair.echo, "source", load_values=True)
    image_info = inspect_patch_file(pair.image, "target", load_values=True)
    if echo_info.values is None or image_info.values is None:
        raise RuntimeError(f"failed to load numeric matrices for {pair.key}")
    echo = np.asarray(echo_info.values, dtype=np.complex128)
    image = np.asarray(image_info.values, dtype=np.complex128)
    if echo.shape != image.shape or echo.ndim != 2:
        raise ValueError(
            f"pair {pair.key} must contain equal 2-D shapes, got {echo.shape} and {image.shape}"
        )
    finite = (
        np.isfinite(echo.real)
        & np.isfinite(echo.imag)
        & np.isfinite(image.real)
        & np.isfinite(image.imag)
    )
    if not bool(finite.all()):
        raise ValueError(f"pair {pair.key} contains non-finite complex values")
    return echo, image


def resize_image_complex(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Band-limited image resizing used only for the coarse polar search."""

    if values.shape == shape:
        return np.asarray(values, dtype=np.complex128)
    resized = signal.resample(values, shape[0], axis=0)
    resized = signal.resample(resized, shape[1], axis=1)
    return np.asarray(resized, dtype=np.complex128)


def resize_coordinate_grid(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Interpolate an unknown coordinate grid without assuming it is an image."""

    if values.shape == shape:
        return np.asarray(values, dtype=np.complex128)
    zoom = (shape[0] / values.shape[0], shape[1] / values.shape[1])
    real = ndimage.zoom(values.real, zoom, order=1, mode="nearest", prefilter=False)
    imag = ndimage.zoom(values.imag, zoom, order=1, mode="nearest", prefilter=False)
    return np.asarray(real + 1j * imag, dtype=np.complex128)


def load_resized_pairs(
    pairs: Sequence[FilePair], search_size: int
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    loaded = []
    for pair in pairs:
        echo, image = load_pair(pair)
        shape = (min(search_size, echo.shape[0]), min(search_size, echo.shape[1]))
        loaded.append(
            (
                pair.key,
                resize_coordinate_grid(echo, shape),
                resize_image_complex(image, shape),
            )
        )
    return loaded


def apply_fourier_transform(values: np.ndarray, spec: TransformSpec) -> np.ndarray:
    if spec.direction == "identity":
        return np.asarray(values, dtype=np.complex128)
    transformed = np.asarray(values, dtype=np.complex128)
    if spec.pre_shift:
        transformed = np.fft.ifftshift(transformed, axes=spec.axes)
    transform = np.fft.fftn if spec.direction == "fft" else np.fft.ifftn
    transformed = transform(transformed, axes=spec.axes, norm="ortho")
    if spec.post_shift:
        transformed = np.fft.fftshift(transformed, axes=spec.axes)
    return transformed


def orientation_names(shape: tuple[int, int]) -> tuple[str, ...]:
    names = ("identity", "flip_row", "flip_col", "flip_both")
    if shape[0] == shape[1]:
        names += ("transpose", "transpose_flip_row", "transpose_flip_col", "transpose_flip_both")
    return names


def apply_orientation(values: np.ndarray, name: str) -> np.ndarray:
    transpose = name.startswith("transpose")
    oriented = values.T if transpose else values
    suffix = name.removeprefix("transpose_") if transpose else name
    if name == "transpose":
        suffix = "identity"
    if suffix in {"flip_row", "flip_both"}:
        oriented = np.flip(oriented, axis=0)
    if suffix in {"flip_col", "flip_both"}:
        oriented = np.flip(oriented, axis=1)
    return oriented


def _polar_bounds(
    theta_span_deg: float, radius_center_ratio: float
) -> tuple[float, float, float, float]:
    theta = np.deg2rad(np.linspace(-theta_span_deg / 2.0, theta_span_deg / 2.0, 1025))
    radial = np.asarray([radius_center_ratio - 0.5, radius_center_ratio + 0.5])
    if radial[0] <= 0:
        raise ValueError("radius_center_ratio must exceed 0.5")
    x = radial[:, None] * np.cos(theta)[None, :]
    y = radial[:, None] * np.sin(theta)[None, :]
    return float(x.min()), float(x.max()), float(y.min()), float(y.max())


def polar_to_cartesian(
    echo: np.ndarray,
    *,
    theta_axis: int,
    theta_reversed: bool,
    radial_reversed: bool,
    theta_span_deg: float,
    radius_center_ratio: float,
) -> np.ndarray:
    """Approximate polar/sector spectrum regridding with normalized geometry."""

    if theta_axis not in {0, 1}:
        raise ValueError("theta_axis must be 0 or 1")
    output_shape = echo.shape
    polar = np.asarray(echo, dtype=np.complex128)
    if theta_axis == 1:
        polar = polar.T
    if theta_reversed:
        polar = np.flip(polar, axis=0)
    if radial_reversed:
        polar = np.flip(polar, axis=1)
    theta_count, radial_count = polar.shape
    x_min, x_max, y_min, y_max = _polar_bounds(theta_span_deg, radius_center_ratio)
    x = np.linspace(x_min, x_max, output_shape[1])
    y = np.linspace(y_min, y_max, output_shape[0])
    cart_x, cart_y = np.meshgrid(x, y)
    query_theta = np.rad2deg(np.arctan2(cart_y, cart_x))
    query_radius = np.hypot(cart_x, cart_y)
    theta_index = (query_theta + theta_span_deg / 2.0) / theta_span_deg * (theta_count - 1)
    radial_index = (
        query_radius - (radius_center_ratio - 0.5)
    ) * (radial_count - 1)
    valid = (
        (theta_index >= 0)
        & (theta_index <= theta_count - 1)
        & (radial_index >= 0)
        & (radial_index <= radial_count - 1)
    )
    coordinates = np.stack([theta_index, radial_index])
    real = ndimage.map_coordinates(polar.real, coordinates, order=1, mode="constant", cval=0.0)
    imag = ndimage.map_coordinates(polar.imag, coordinates, order=1, mode="constant", cval=0.0)
    cartesian = real + 1j * imag
    cartesian[~valid] = 0.0
    return cartesian


def transform_echo(echo: np.ndarray, spec: TransformSpec) -> np.ndarray:
    if spec.family == "simple":
        reconstructed = apply_fourier_transform(echo, spec)
    elif spec.family == "polar":
        if (
            spec.theta_axis is None
            or spec.theta_span_deg is None
            or spec.radius_center_ratio is None
        ):
            raise ValueError("polar transform is missing geometry parameters")
        cartesian = polar_to_cartesian(
            echo,
            theta_axis=spec.theta_axis,
            theta_reversed=spec.theta_reversed,
            radial_reversed=spec.radial_reversed,
            theta_span_deg=spec.theta_span_deg,
            radius_center_ratio=spec.radius_center_ratio,
        )
        reconstructed = apply_fourier_transform(cartesian, spec)
    else:
        raise ValueError(f"unknown transform family: {spec.family}")
    reconstructed = apply_orientation(reconstructed, spec.orientation)
    return np.conj(reconstructed) if spec.conjugate else reconstructed


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=np.float64).ravel()
    second_values = np.asarray(second, dtype=np.float64).ravel()
    first_values = first_values - first_values.mean()
    second_values = second_values - second_values.mean()
    denominator = np.linalg.norm(first_values) * np.linalg.norm(second_values)
    if denominator <= np.finfo(np.float64).tiny:
        return 0.0
    return float(np.dot(first_values, second_values) / denominator)


def best_circular_shift(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int]:
    prediction_mag = np.log1p(np.abs(prediction))
    target_mag = np.log1p(np.abs(target))
    prediction_mag -= prediction_mag.mean()
    target_mag -= target_mag.mean()
    correlation = np.fft.ifft2(
        np.fft.fft2(target_mag) * np.conj(np.fft.fft2(prediction_mag))
    ).real
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shifts = tuple(
        int(index if index <= size // 2 else index - size)
        for index, size in zip(peak, correlation.shape, strict=True)
    )
    return shifts[0], shifts[1]


def score_sample(prediction: np.ndarray, target: np.ndarray) -> SampleMetrics:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction/target shape mismatch: {prediction.shape} vs {target.shape}")
    shift_row, shift_col = best_circular_shift(prediction, target)
    aligned = np.roll(prediction, (shift_row, shift_col), axis=(0, 1))
    denominator = np.vdot(aligned, aligned).real
    gain = np.vdot(aligned, target) / denominator if denominator > 0 else 0.0j
    fitted = gain * aligned
    norm_product = np.linalg.norm(aligned) * np.linalg.norm(target)
    coherence = float(abs(np.vdot(aligned, target)) / norm_product) if norm_product > 0 else 0.0
    target_norm = np.linalg.norm(target)
    relative_nrmse = float(np.linalg.norm(fitted - target) / target_norm) if target_norm > 0 else 1.0
    magnitude_correlation = _pearson(np.abs(fitted), np.abs(target))
    log_magnitude_correlation = _pearson(np.log1p(np.abs(fitted)), np.log1p(np.abs(target)))
    score = (
        0.50 * np.clip(coherence, 0.0, 1.0)
        + 0.25 * np.clip(magnitude_correlation, 0.0, 1.0)
        + 0.15 * np.clip(log_magnitude_correlation, 0.0, 1.0)
        + 0.10 * (1.0 - np.clip(relative_nrmse, 0.0, 1.0))
    )
    return SampleMetrics(
        score=float(score),
        complex_coherence=coherence,
        magnitude_correlation=magnitude_correlation,
        log_magnitude_correlation=log_magnitude_correlation,
        relative_nrmse=relative_nrmse,
        gain_real=float(np.real(gain)),
        gain_imag=float(np.imag(gain)),
        shift_row=shift_row,
        shift_col=shift_col,
    )


def summarize_metrics(rows: Sequence[SampleMetrics]) -> dict[str, float]:
    if not rows:
        raise ValueError("cannot summarize an empty metric list")
    summary: dict[str, float] = {}
    for field in (
        "score",
        "complex_coherence",
        "magnitude_correlation",
        "log_magnitude_correlation",
        "relative_nrmse",
    ):
        values = np.asarray([getattr(row, field) for row in rows], dtype=np.float64)
        summary[f"{field}_mean"] = float(values.mean())
        summary[f"{field}_median"] = float(np.median(values))
        summary[f"{field}_p05"] = float(np.percentile(values, 5))
        summary[f"{field}_p95"] = float(np.percentile(values, 95))
    return summary


def evaluate_fixed_spec(
    spec: TransformSpec,
    samples: Iterable[tuple[str, np.ndarray, np.ndarray]],
    split: str,
) -> CandidateResult:
    rows = []
    for _, echo, target in samples:
        rows.append(score_sample(transform_echo(echo, spec), target))
    return CandidateResult(
        spec=spec,
        split=split,
        sample_count=len(rows),
        metrics=summarize_metrics(rows),
        per_sample=tuple(rows),
    )


def _thumbnail_magnitude(values: np.ndarray, size: int = 32) -> np.ndarray:
    magnitude = np.log1p(np.abs(values))
    if magnitude.shape[0] % size == 0 and magnitude.shape[1] % size == 0:
        row_block = magnitude.shape[0] // size
        col_block = magnitude.shape[1] // size
        return magnitude.reshape(size, row_block, size, col_block).mean(axis=(1, 3))
    zoom = (size / magnitude.shape[0], size / magnitude.shape[1])
    return ndimage.zoom(magnitude, zoom, order=1)


def optimize_orientation(
    spec: TransformSpec,
    samples: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> CandidateResult:
    """Choose one global orientation cheaply, then test both conjugation conventions."""

    raw_predictions = [
        (target, transform_echo(echo, replace(spec, orientation="identity")))
        for _, echo, target in samples
    ]
    orientations = orientation_names(samples[0][1].shape)
    orientation_scores = {}
    for orientation in orientations:
        correlations = []
        for target, prediction in raw_predictions:
            oriented = _thumbnail_magnitude(apply_orientation(prediction, orientation))
            target_thumbnail = _thumbnail_magnitude(target)
            shift = best_circular_shift(oriented, target_thumbnail)
            aligned = np.roll(oriented, shift, axis=(0, 1))
            correlations.append(_pearson(aligned, target_thumbnail))
        orientation_scores[orientation] = float(np.median(correlations))
    best_orientation = max(orientation_scores, key=orientation_scores.get)
    results = []
    for conjugate in (False, True):
        fitted_spec = replace(spec, orientation=best_orientation, conjugate=conjugate)
        rows = []
        for target, prediction in raw_predictions:
            oriented = apply_orientation(prediction, best_orientation)
            if conjugate:
                oriented = np.conj(oriented)
            rows.append(score_sample(oriented, target))
        results.append(
            CandidateResult(
                spec=fitted_spec,
                split="fit",
                sample_count=len(rows),
                metrics=summarize_metrics(rows),
                per_sample=tuple(rows),
            )
        )
    return max(results, key=lambda result: result.score)


def simple_specs() -> tuple[TransformSpec, ...]:
    specs = [
        TransformSpec(
            name="identity",
            stage="image_domain",
            family="simple",
        )
    ]
    shift_modes = (
        (False, False, "raw"),
        (True, True, "centered"),
        (True, False, "input_shifted"),
        (False, True, "output_shifted"),
    )
    for axes, stage, axis_name in (
        ((0,), "one_dimensional_fft", "axis0"),
        ((1,), "one_dimensional_fft", "axis1"),
        ((0, 1), "cartesian_2d_spectrum", "2d"),
    ):
        for direction in ("fft", "ifft"):
            for pre_shift, post_shift, shift_name in shift_modes:
                specs.append(
                    TransformSpec(
                        name=f"{direction}_{axis_name}_{shift_name}",
                        stage=stage,
                        family="simple",
                        axes=axes,
                        direction=direction,
                        pre_shift=pre_shift,
                        post_shift=post_shift,
                    )
                )
    return tuple(specs)


def polar_specs(
    theta_spans: Sequence[float], radius_center_ratios: Sequence[float]
) -> Iterable[TransformSpec]:
    shift_modes = (
        (False, False, "raw"),
        (True, True, "centered"),
    )
    for theta_axis in (0, 1):
        for theta_reversed in (False, True):
            for radial_reversed in (False, True):
                for theta_span in theta_spans:
                    for radius_ratio in radius_center_ratios:
                        if radius_ratio <= 0.5:
                            continue
                        for direction in ("fft", "ifft"):
                            for pre_shift, post_shift, shift_name in shift_modes:
                                yield TransformSpec(
                                    name=(
                                        f"polar_theta{theta_axis}_tr{int(theta_reversed)}_"
                                        f"rr{int(radial_reversed)}_span{theta_span:g}_"
                                        f"radius{radius_ratio:g}_{direction}_{shift_name}"
                                    ),
                                    stage="polar_frequency",
                                    family="polar",
                                    axes=(0, 1),
                                    direction=direction,
                                    pre_shift=pre_shift,
                                    post_shift=post_shift,
                                    theta_axis=theta_axis,
                                    theta_reversed=theta_reversed,
                                    radial_reversed=radial_reversed,
                                    theta_span_deg=float(theta_span),
                                    radius_center_ratio=float(radius_ratio),
                                )


def rank_specs(
    specs: Iterable[TransformSpec],
    samples: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    progress_every: int = 0,
    label: str = "candidates",
) -> list[CandidateResult]:
    results = []
    for index, spec in enumerate(specs, start=1):
        results.append(optimize_orientation(spec, samples))
        if progress_every > 0 and index % progress_every == 0:
            print(f"evaluated {index:,} {label}", flush=True)
    return sorted(results, key=lambda result: result.score, reverse=True)


def retain_per_stage(results: Sequence[CandidateResult], count: int) -> list[CandidateResult]:
    retained: list[CandidateResult] = []
    stage_counts: dict[str, int] = {}
    for result in results:
        used = stage_counts.get(result.spec.stage, 0)
        if used < count:
            retained.append(result)
            stage_counts[result.spec.stage] = used + 1
    return retained


def full_resolution_samples(
    pairs: Sequence[FilePair],
) -> Iterable[tuple[str, np.ndarray, np.ndarray]]:
    for pair in pairs:
        echo, image = load_pair(pair)
        yield pair.key, echo, image


def validate_candidates(
    candidates: Sequence[CandidateResult],
    pairs: Sequence[FilePair],
) -> list[CandidateResult]:
    """Evaluate fixed fit-selected parameters lazily on original-size holdout data."""

    metric_rows: list[list[SampleMetrics]] = [[] for _ in candidates]
    for _, echo, target in full_resolution_samples(pairs):
        for index, candidate in enumerate(candidates):
            prediction = transform_echo(echo, candidate.spec)
            metric_rows[index].append(score_sample(prediction, target))
    validated = [
        CandidateResult(
            spec=candidate.spec,
            split="holdout_full_resolution",
            sample_count=len(rows),
            metrics=summarize_metrics(rows),
            per_sample=tuple(rows),
        )
        for candidate, rows in zip(candidates, metric_rows, strict=True)
    ]
    return sorted(validated, key=lambda result: result.score, reverse=True)


def infer_conclusion(results: Sequence[CandidateResult]) -> dict[str, Any]:
    if not results:
        raise ValueError("conclusion requires candidate results")
    stage_best: dict[str, CandidateResult] = {}
    for result in results:
        stage_best.setdefault(result.spec.stage, result)
    ranked_stages = sorted(stage_best.values(), key=lambda result: result.score, reverse=True)
    best = ranked_stages[0]
    second_score = ranked_stages[1].score if len(ranked_stages) > 1 else 0.0
    margin = best.score - second_score
    coherence = best.metrics["complex_coherence_median"]
    if best.score >= 0.75 and coherence >= 0.70 and margin >= 0.08:
        strength = "strong_support"
        selected_stage = best.spec.stage
    elif best.score >= 0.40 and coherence >= 0.25 and margin >= 0.03:
        strength = "weak_support"
        selected_stage = best.spec.stage
    else:
        strength = "unidentified"
        selected_stage = "unknown"
    return {
        "selected_stage": selected_stage,
        "selected_stage_zh": STAGE_LABELS[selected_stage],
        "evidence_strength": strength,
        "best_candidate": best.spec.name,
        "best_score_median": best.score,
        "best_complex_coherence_median": coherence,
        "stage_margin": margin,
        "stage_ranking": [
            {
                "stage": result.spec.stage,
                "stage_zh": STAGE_LABELS[result.spec.stage],
                "score_median": result.score,
                "candidate": result.spec.name,
            }
            for result in ranked_stages
        ],
        "warning": (
            "This is a black-box hypothesis ranking, not unique proof of the original PFA stage. "
            "Physical angle/frequency/trajectory metadata and source processing code are absent."
        ),
    }


def result_dict(result: CandidateResult, include_samples: bool = False) -> dict[str, Any]:
    payload = {
        "spec": asdict(result.spec),
        "split": result.split,
        "sample_count": result.sample_count,
        "metrics": result.metrics,
    }
    if include_samples:
        payload["per_sample"] = [asdict(row) for row in result.per_sample]
    return payload


def write_ranking_csv(path: Path, results: Sequence[CandidateResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "stage",
        "candidate",
        "score_median",
        "complex_coherence_median",
        "magnitude_correlation_median",
        "log_magnitude_correlation_median",
        "relative_nrmse_median",
        "orientation",
        "conjugate",
        "theta_axis",
        "theta_span_deg",
        "radius_center_ratio",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "stage": result.spec.stage,
                    "candidate": result.spec.name,
                    "score_median": result.score,
                    "complex_coherence_median": result.metrics["complex_coherence_median"],
                    "magnitude_correlation_median": result.metrics["magnitude_correlation_median"],
                    "log_magnitude_correlation_median": result.metrics[
                        "log_magnitude_correlation_median"
                    ],
                    "relative_nrmse_median": result.metrics["relative_nrmse_median"],
                    "orientation": result.spec.orientation,
                    "conjugate": result.spec.conjugate,
                    "theta_axis": result.spec.theta_axis,
                    "theta_span_deg": result.spec.theta_span_deg,
                    "radius_center_ratio": result.spec.radius_center_ratio,
                }
            )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    conclusion = report["conclusion"]
    lines = [
        "PFA 阶段黑盒诊断摘要",
        "=" * 24,
        f"结论：{conclusion['selected_stage_zh']}",
        f"证据等级：{conclusion['evidence_strength']}",
        f"最佳候选：{conclusion['best_candidate']}",
        f"留出集综合分数中位数：{conclusion['best_score_median']:.6f}",
        f"留出集复数相干性中位数：{conclusion['best_complex_coherence_median']:.6f}",
        f"相对第二阶段的分数间隔：{conclusion['stage_margin']:.6f}",
        "",
        "各阶段最佳结果：",
    ]
    for row in conclusion["stage_ranking"]:
        lines.append(
            f"- {row['stage_zh']}: score={row['score_median']:.6f}, "
            f"candidate={row['candidate']}"
        )
    lines.extend(
        [
            "",
            "限制：MAT 文件没有角度、频率、轨迹或 PFA 插值元数据。此结论只是跨样本的",
            "候选模型比较，不能唯一证明原始处理程序具体在哪一步执行了 FFT。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_representative_figure(
    path: Path,
    pair: FilePair,
    candidate: CandidateResult,
) -> None:
    echo, target = load_pair(pair)
    prediction = transform_echo(echo, candidate.spec)
    metrics = score_sample(prediction, target)
    prediction = np.roll(prediction, (metrics.shift_row, metrics.shift_col), axis=(0, 1))
    gain = metrics.gain_real + 1j * metrics.gain_imag
    prediction = gain * prediction
    magnitude_arrays = [np.log1p(np.abs(values)) for values in (echo, prediction, target)]
    mag_min = min(float(values.min()) for values in magnitude_arrays)
    mag_max = max(float(values.max()) for values in magnitude_arrays)
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    titles = ("Echo", "Best hypothesis reconstruction", "PFA Image target")
    for axis, values, title in zip(axes[0], magnitude_arrays, titles, strict=True):
        image = axis.imshow(values, cmap="gray", vmin=mag_min, vmax=mag_max)
        axis.set_title(f"{title}: log(1+|z|)")
        axis.axis("off")
    figure.colorbar(image, ax=axes[0].tolist(), shrink=0.75)
    for axis, values, title in zip(axes[1], (echo, prediction, target), titles, strict=True):
        phase = axis.imshow(np.angle(values), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        axis.set_title(f"{title}: phase")
        axis.axis("off")
    figure.colorbar(phase, ax=axes[1].tolist(), shrink=0.75)
    figure.suptitle(
        f"{pair.key} | {candidate.spec.stage} | score={metrics.score:.4f} | "
        f"coherence={metrics.complex_coherence:.4f}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def diagnose(args: argparse.Namespace) -> dict[str, Any]:
    pairs, pairing_summary = discover_pairs(args.echo_dir, args.image_dir)
    if len(pairs) < 2:
        raise RuntimeError("diagnosis requires at least two strictly paired MAT files")
    selected = evenly_spaced(pairs, min(args.sample_count, len(pairs)))
    fit_count = args.fit_count if args.fit_count is not None else len(selected) // 2
    fit_pairs, holdout_pairs = split_fit_holdout(selected, fit_count)
    coarse_fit_pairs = evenly_spaced(fit_pairs, min(args.coarse_fit_count, len(fit_pairs)))
    simple_fit_samples = list(full_resolution_samples(coarse_fit_pairs))
    polar_fit_samples = load_resized_pairs(fit_pairs, args.search_size)

    print(
        f"selected {len(selected)} pairs: fit={len(fit_pairs)}, holdout={len(holdout_pairs)}; "
        f"polar coarse size={polar_fit_samples[0][1].shape}",
        flush=True,
    )
    simple_fit = rank_specs(simple_specs(), simple_fit_samples, label="simple candidates")
    simple_retained = retain_per_stage(simple_fit, args.simple_top_per_stage)

    coarse_pairs = evenly_spaced(
        polar_fit_samples, min(args.coarse_fit_count, len(polar_fit_samples))
    )
    polar_coarse = rank_specs(
        polar_specs(args.theta_spans, args.radius_center_ratios),
        coarse_pairs,
        progress_every=args.progress_every,
        label="polar candidates",
    )
    polar_refine_specs = [result.spec for result in polar_coarse[: args.polar_refine_count]]
    polar_fit = rank_specs(
        polar_refine_specs, polar_fit_samples, label="refined polar candidates"
    )
    polar_retained = polar_fit[: args.polar_full_count]

    fit_candidates = simple_retained + polar_retained
    holdout_results = validate_candidates(fit_candidates, holdout_pairs)
    conclusion = infer_conclusion(holdout_results)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "sample_count_requested": args.sample_count,
            "selected_pair_count": len(selected),
            "fit_pair_count": len(fit_pairs),
            "simple_orientation_fit_pair_count": len(simple_fit_samples),
            "holdout_pair_count": len(holdout_pairs),
            "search_size": args.search_size,
            "coarse_fit_count": len(coarse_pairs),
            "theta_spans_deg": list(args.theta_spans),
            "radius_center_ratios": list(args.radius_center_ratios),
            "pair_sampling": "evenly_spaced_after_canonical_key_sort",
            "parameter_selection": "fit_only",
            "final_scoring": "original_resolution_holdout_only",
        },
        "pairing": pairing_summary,
        "selected_keys": {
            "fit": [pair.key for pair in fit_pairs],
            "holdout": [pair.key for pair in holdout_pairs],
        },
        "conclusion": conclusion,
        "holdout_ranking": [result_dict(result, include_samples=True) for result in holdout_results],
        "fit_ranking": {
            "simple": [result_dict(result) for result in simple_fit],
            "polar_refined": [result_dict(result) for result in polar_fit],
        },
        "method_limits": [
            "No physical theta/frequency/trajectory/PFA interpolation metadata were available.",
            "The polar model uses normalized sector geometry and bilinear interpolation.",
            "A winning hypothesis demonstrates cross-sample explanatory power, not unique provenance.",
            "Per-sample nuisance fitting is limited to one complex gain and one circular integer shift.",
        ],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "diagnosis_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    write_ranking_csv(args.output_dir / "holdout_ranking.csv", holdout_results)
    write_summary(args.output_dir / "summary_zh.txt", report)
    if holdout_pairs:
        export_representative_figure(
            args.output_dir / "representative_best_candidate.png",
            holdout_pairs[len(holdout_pairs) // 2],
            holdout_results[0],
        )
    print((args.output_dir / "summary_zh.txt").read_text(encoding="utf-8"), flush=True)
    print(f"JSON report: {report_path.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Black-box ranking of candidate processing stages between SAR Echo and PFA images."
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("pfa_stage_diagnosis"))
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--fit-count", type=int, default=None)
    parser.add_argument("--search-size", type=int, default=128)
    parser.add_argument("--coarse-fit-count", type=int, default=4)
    parser.add_argument("--simple-top-per-stage", type=int, default=2)
    parser.add_argument("--polar-refine-count", type=int, default=12)
    parser.add_argument("--polar-full-count", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--theta-spans",
        type=lambda raw: parse_float_list(raw, "theta-spans"),
        default=parse_float_list("2,5,10,20,40,80", "theta-spans"),
    )
    parser.add_argument(
        "--radius-center-ratios",
        type=lambda raw: parse_float_list(raw, "radius-center-ratios"),
        default=parse_float_list("0.75,1,2,4,8,16,32", "radius-center-ratios"),
    )
    args = parser.parse_args()
    for name in (
        "sample_count",
        "search_size",
        "coarse_fit_count",
        "simple_top_per_stage",
        "polar_refine_count",
        "polar_full_count",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.fit_count is not None and args.fit_count <= 0:
        parser.error("--fit-count must be positive")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    if any(value <= 0.5 for value in args.radius_center_ratios):
        parser.error("--radius-center-ratios values must exceed 0.5")
    return args


def main() -> None:
    diagnose(parse_args())


if __name__ == "__main__":
    main()
