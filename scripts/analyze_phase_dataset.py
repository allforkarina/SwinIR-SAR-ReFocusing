"""Read-only audit for paired SAR phase-refocusing datasets.

The source Echo/Image trees are never modified.  All reports and optional
mosaic previews are written below one explicitly separate output directory.
Dataset splitting is deliberately out of scope for this audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import matplotlib
import numpy as np
from scipy.io import whosmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_patch_adjacency import (
    analyze as analyze_adjacency,
    overlap_metrics,
    overlap_regions,
)
from scripts.analyze_sar_dataset import (
    DEFAULT_ECHO_DIR,
    DEFAULT_IMAGE_DIR,
    FilePair,
    analyze as analyze_structure,
    discover_pairs,
    distribution,
    evenly_spaced_indices,
    index_files,
    inspect_patch_file,
    mat_files,
)
from scripts.diagnose_shared_complex_filter import evaluate_focus_prediction
from swinir.sar_metrics import log_magnitude_image


SCHEMA_VERSION = 1
COORDINATE_PATTERN = re.compile(
    r"^patch_row_([+-]?\d+)_col_([+-]?\d+)(?:_(\d+))?\.mat$",
    re.IGNORECASE,
)
MAX_EXAMPLES = 20
PHASE_METRIC_NAMES = (
    "echo_normalized_complex_rmse",
    "oracle_normalized_complex_rmse",
    "rmse_gap_fraction_closed",
    "echo_complex_coherence",
    "oracle_complex_coherence",
    "echo_log_magnitude_ssim",
    "oracle_log_magnitude_ssim",
    "echo_edge_correlation",
    "oracle_edge_correlation",
    "echo_high_frequency_energy_ratio",
    "oracle_high_frequency_energy_ratio",
    "identity_phase_alignment",
    "phase_resultant_length",
    "phase_neighbor_roughness",
    "reliable_frequency_fraction",
    "log_amplitude_ratio_mean",
    "log_amplitude_ratio_std",
)


@dataclass(frozen=True)
class CoordinatePair:
    pair: FilePair
    row: int
    col: int
    suffix: str | None
    parent_id: str


@dataclass(frozen=True)
class PhaseSample:
    record: CoordinatePair
    correction: np.ndarray
    weights: np.ndarray
    metrics: dict[str, float | int | str]
    band_rows: tuple[dict[str, float | str], ...]
    pca_feature: np.ndarray


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_source_output_separation(
    echo_dir: Path, image_dir: Path, output_dir: Path
) -> None:
    echo = echo_dir.resolve()
    image = image_dir.resolve()
    output = output_dir.resolve()
    if echo == image:
        raise ValueError("Echo and Image directories must be distinct")
    for role, source in (("Echo", echo), ("Image", image)):
        if output == source or is_relative_to(output, source):
            raise ValueError(
                f"output directory must not be inside the read-only {role} tree: {source}"
            )


def write_json(path: Path, values: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(values, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_coordinate(path: Path) -> tuple[int, int, str | None] | None:
    match = COORDINATE_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2)), match.group(3)


def coordinate_records(
    pairs: Sequence[FilePair],
) -> tuple[list[CoordinatePair], dict[str, Any]]:
    parsed: list[tuple[FilePair, int, int, str | None]] = []
    unparsed: list[str] = []
    suffixes: Counter[str] = Counter()
    for pair in pairs:
        coordinate = parse_coordinate(pair.echo)
        if coordinate is None:
            unparsed.append(pair.echo.name)
            continue
        row, col, suffix = coordinate
        if suffix is not None:
            suffixes[suffix] += 1
        parsed.append((pair, row, col, suffix))

    distinct_suffixes = sorted(suffixes)
    use_suffix = len(distinct_suffixes) > 1
    records = [
        CoordinatePair(
            pair=pair,
            row=row,
            col=col,
            suffix=suffix,
            parent_id=f"suffix_{suffix}" if use_suffix else "coordinate_grid_0",
        )
        for pair, row, col, suffix in parsed
    ]
    if use_suffix:
        method = "filename_numeric_suffix"
        confidence = "probable"
    elif records:
        method = "single_coordinate_grid_candidate"
        confidence = "ambiguous"
    else:
        method = "unavailable"
        confidence = "unavailable"
    return records, {
        "parsed_pair_count": len(records),
        "unparsed_pair_count": len(unparsed),
        "unparsed_examples": unparsed[:MAX_EXAMPLES],
        "numeric_suffix_counts": dict(sorted(suffixes.items())),
        "grouping_method": method,
        "initial_confidence": confidence,
        "warning": (
            "A filename suffix is only a candidate parent-image identifier until "
            "metadata or overlap evidence confirms it."
            if use_suffix
            else "No explicit multi-image identifier was found in parsed filenames."
        ),
    }


def pair_manifest_rows(echo_dir: Path, image_dir: Path) -> list[dict[str, Any]]:
    echo_index, echo_collisions = index_files(mat_files(echo_dir))
    image_index, image_collisions = index_files(mat_files(image_dir))
    keys = sorted(set(echo_index) | set(image_index) | set(echo_collisions) | set(image_collisions))
    rows: list[dict[str, Any]] = []
    for key in keys:
        echo = echo_index.get(key)
        image = image_index.get(key)
        if key in echo_collisions or key in image_collisions:
            status = "collision"
        elif echo is None:
            status = "image_only"
        elif image is None:
            status = "echo_only"
        else:
            status = "paired"
        rows.append(
            {
                "key": key,
                "status": status,
                "echo_file": echo.name if echo is not None else None,
                "image_file": image.name if image is not None else None,
                "echo_collision_files": "|".join(echo_collisions.get(key, [])),
                "image_collision_files": "|".join(image_collisions.get(key, [])),
            }
        )
    return rows


def modal_positive_step(values: Iterable[int]) -> int | None:
    unique = sorted(set(values))
    counts = Counter(second - first for first, second in zip(unique, unique[1:]))
    positive = [(count, -step, step) for step, count in counts.items() if step > 0]
    return max(positive)[2] if positive else None


def group_grid_summary(records: Sequence[CoordinatePair]) -> dict[str, Any]:
    rows = sorted({record.row for record in records})
    cols = sorted({record.col for record in records})
    coordinates = {(record.row, record.col) for record in records}
    collisions = len(records) - len(coordinates)
    expected = len(rows) * len(cols)
    return {
        "sample_count": len(records),
        "unique_coordinate_count": len(coordinates),
        "coordinate_collision_count": collisions,
        "row_min": rows[0] if rows else None,
        "row_max": rows[-1] if rows else None,
        "col_min": cols[0] if cols else None,
        "col_max": cols[-1] if cols else None,
        "unique_row_count": len(rows),
        "unique_col_count": len(cols),
        "modal_row_step": modal_positive_step(rows),
        "modal_col_step": modal_positive_step(cols),
        "rectangular_grid_cell_count": expected,
        "missing_grid_cell_count": expected - len(coordinates),
        "grid_coverage_fraction": len(coordinates) / expected if expected else 0.0,
    }


def coordinate_components(records: Sequence[CoordinatePair]) -> dict[str, Any]:
    groups: dict[str, list[CoordinatePair]] = defaultdict(list)
    for record in records:
        groups[record.parent_id].append(record)
    report: dict[str, Any] = {}
    for parent_id, group in sorted(groups.items()):
        grid = group_grid_summary(group)
        row_step = grid["modal_row_step"]
        col_step = grid["modal_col_step"]
        coordinates = {(record.row, record.col) for record in group}
        remaining = set(coordinates)
        components: list[list[tuple[int, int]]] = []
        while remaining:
            start = remaining.pop()
            component = [start]
            stack = [start]
            while stack:
                row, col = stack.pop()
                neighbors: list[tuple[int, int]] = []
                if row_step is not None:
                    neighbors.extend(((row - row_step, col), (row + row_step, col)))
                if col_step is not None:
                    neighbors.extend(((row, col - col_step), (row, col + col_step)))
                for neighbor in neighbors:
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.append(neighbor)
                        stack.append(neighbor)
            components.append(component)
        components.sort(key=len, reverse=True)
        report[parent_id] = {
            "component_count": len(components),
            "component_sizes": [len(component) for component in components],
            "largest_component_fraction": (
                len(components[0]) / len(coordinates) if components and coordinates else 0.0
            ),
            "component_examples": [
                {
                    "size": len(component),
                    "row_min": min(row for row, _ in component),
                    "row_max": max(row for row, _ in component),
                    "col_min": min(col for _, col in component),
                    "col_max": max(col for _, col in component),
                }
                for component in components[:MAX_EXAMPLES]
            ],
        }
    return report


def variable_schema(path: Path) -> list[tuple[str, tuple[int, ...], str]]:
    if h5py.is_hdf5(path):
        result: list[tuple[str, tuple[int, ...], str]] = []
        with h5py.File(path, "r") as handle:
            for name, value in handle.items():
                if isinstance(value, h5py.Dataset):
                    result.append((name, tuple(value.shape), str(value.dtype)))
        return result
    return [(name, tuple(shape), matlab_class) for name, shape, matlab_class in whosmat(path)]


def metadata_schema(
    pairs: Sequence[FilePair], requested: int
) -> dict[str, Any]:
    selected = [pairs[index] for index in evenly_spaced_indices(len(pairs), requested)]
    role_counters: dict[str, Counter[str]] = {"echo": Counter(), "image": Counter()}
    shape_counters: dict[str, Counter[str]] = {"echo": Counter(), "image": Counter()}
    errors: list[dict[str, str]] = []
    for pair in selected:
        for role, path in (("echo", pair.echo), ("image", pair.image)):
            try:
                for name, shape, dtype in variable_schema(path):
                    role_counters[role][name] += 1
                    shape_counters[role][f"{name}|{shape}|{dtype}"] += 1
            except Exception as error:
                if len(errors) < MAX_EXAMPLES:
                    errors.append({"file": path.name, "error": repr(error)})
    return {
        "sampled_pair_count": len(selected),
        "variable_presence": {
            role: dict(sorted(counter.items())) for role, counter in role_counters.items()
        },
        "variable_shape_dtype": {
            role: dict(sorted(counter.items())) for role, counter in shape_counters.items()
        },
        "inspection_error_count": len(errors),
        "inspection_error_examples": errors,
        "parent_id_status": (
            "No scalar field is automatically accepted as a parent-image ID; "
            "candidate fields require inspection of this schema and values."
        ),
    }


def load_pair(pair: FilePair) -> tuple[np.ndarray, np.ndarray]:
    echo_info = inspect_patch_file(pair.echo, "source", load_values=True)
    image_info = inspect_patch_file(pair.image, "target", load_values=True)
    if echo_info.values is None or image_info.values is None:
        raise RuntimeError("numeric MAT inspection did not load both arrays")
    echo = np.asarray(echo_info.values)
    image = np.asarray(image_info.values)
    if echo.shape != image.shape:
        raise ValueError(f"Echo/Image shapes differ: {echo.shape} vs {image.shape}")
    if echo.ndim != 2 or not np.iscomplexobj(echo) or not np.iscomplexobj(image):
        raise ValueError("Echo and Image patches must be two-dimensional complex arrays")
    finite = (
        np.isfinite(echo.real)
        & np.isfinite(echo.imag)
        & np.isfinite(image.real)
        & np.isfinite(image.imag)
    )
    if not bool(finite.all()):
        raise ValueError("Echo/Image patch contains non-finite values")
    return echo, image


def normalized_phase_oracle(
    echo: np.ndarray,
    image: np.ndarray,
    *,
    fft_norm: str,
    phasor_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = math.sqrt(float(np.mean(np.abs(echo) ** 2)) + 1.0e-12)
    normalized_echo = echo / scale
    normalized_image = image / scale
    echo_spectrum = np.fft.fftshift(np.fft.fft2(normalized_echo, norm=fft_norm))
    image_spectrum = np.fft.fftshift(np.fft.fft2(normalized_image, norm=fft_norm))
    cross = image_spectrum * np.conj(echo_spectrum)
    energy = np.abs(cross)
    reliable = energy > phasor_epsilon
    correction = np.ones_like(cross, dtype=np.complex128)
    correction[reliable] = cross[reliable] / energy[reliable]
    weights = np.sqrt(energy)
    focused_spectrum = echo_spectrum * correction
    oracle = np.fft.ifft2(np.fft.ifftshift(focused_spectrum), norm=fft_norm)
    return normalized_echo, normalized_image, oracle, correction, weights


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    denominator = float(weights.sum())
    return float(np.sum(values * weights) / denominator) if denominator > 0 else 0.0


def phase_roughness(correction: np.ndarray, weights: np.ndarray) -> float:
    terms: list[float] = []
    for axis in (0, 1):
        first = np.take(correction, range(correction.shape[axis] - 1), axis=axis)
        second = np.take(correction, range(1, correction.shape[axis]), axis=axis)
        first_w = np.take(weights, range(weights.shape[axis] - 1), axis=axis)
        second_w = np.take(weights, range(1, weights.shape[axis]), axis=axis)
        pair_weights = np.sqrt(first_w * second_w)
        terms.append(weighted_mean(1.0 - np.real(second * np.conj(first)), pair_weights))
    return float(np.mean(terms))


def radial_band_rows(
    key: str, correction: np.ndarray, weights: np.ndarray, reliable: np.ndarray
) -> tuple[dict[str, float | str], ...]:
    rows = np.fft.fftshift(np.fft.fftfreq(correction.shape[0]))
    cols = np.fft.fftshift(np.fft.fftfreq(correction.shape[1]))
    radius = np.hypot(rows[:, None], cols[None, :])
    maximum = float(radius.max()) or 1.0
    normalized = radius / maximum
    bands = (("low", 0.0, 1.0 / 3.0), ("mid", 1.0 / 3.0, 2.0 / 3.0), ("high", 2.0 / 3.0, 1.01))
    result: list[dict[str, float | str]] = []
    for name, lower, upper in bands:
        mask = (normalized >= lower) & (normalized < upper)
        band_weights = weights * mask
        resultant = np.sum(correction * band_weights)
        denominator = float(band_weights.sum())
        result.append(
            {
                "key": key,
                "band": name,
                "frequency_fraction": float(mask.mean()),
                "reliable_fraction": float(reliable[mask].mean()) if bool(mask.any()) else 0.0,
                "phase_resultant_length": float(abs(resultant) / denominator) if denominator > 0 else 0.0,
                "identity_phase_alignment": weighted_mean(np.real(correction), band_weights),
            }
        )
    return tuple(result)


def pooled_phase_feature(
    correction: np.ndarray, weights: np.ndarray, grid_size: int
) -> np.ndarray:
    row_edges = np.linspace(0, correction.shape[0], grid_size + 1, dtype=int)
    col_edges = np.linspace(0, correction.shape[1], grid_size + 1, dtype=int)
    pooled = np.ones((grid_size, grid_size), dtype=np.complex128)
    for row_index in range(grid_size):
        for col_index in range(grid_size):
            row_slice = slice(row_edges[row_index], row_edges[row_index + 1])
            col_slice = slice(col_edges[col_index], col_edges[col_index + 1])
            local_weights = weights[row_slice, col_slice]
            value = np.sum(correction[row_slice, col_slice] * local_weights)
            if abs(value) > 0:
                pooled[row_index, col_index] = value / abs(value)
    return np.concatenate((pooled.real.ravel(), pooled.imag.ravel())).astype(np.float32)


def phase_sample(
    record: CoordinatePair,
    *,
    fft_norm: str,
    phasor_epsilon: float,
    floor_db: float,
    high_frequency_radius_fraction: float,
    pca_grid_size: int,
) -> PhaseSample:
    echo, image = load_pair(record.pair)
    normalized_echo, normalized_image, oracle, correction, weights = normalized_phase_oracle(
        echo,
        image,
        fft_norm=fft_norm,
        phasor_epsilon=phasor_epsilon,
    )
    reliable = np.square(weights) > phasor_epsilon
    echo_metrics = evaluate_focus_prediction(
        normalized_echo,
        normalized_image,
        floor_db=floor_db,
        high_frequency_radius_fraction=high_frequency_radius_fraction,
    )
    oracle_metrics = evaluate_focus_prediction(
        oracle,
        normalized_image,
        floor_db=floor_db,
        high_frequency_radius_fraction=high_frequency_radius_fraction,
    )
    echo_rmse = float(echo_metrics["normalized_complex_rmse"])
    oracle_rmse = float(oracle_metrics["normalized_complex_rmse"])
    log_ratio = np.log(
        np.abs(np.fft.fftshift(np.fft.fft2(normalized_image, norm=fft_norm)))
        + 1.0e-12
    ) - np.log(
        np.abs(np.fft.fftshift(np.fft.fft2(normalized_echo, norm=fft_norm)))
        + 1.0e-12
    )
    metrics: dict[str, float | int | str] = {
        "key": record.pair.key,
        "echo_file": record.pair.echo.name,
        "image_file": record.pair.image.name,
        "parent_id": record.parent_id,
        "row": record.row,
        "col": record.col,
        "echo_normalized_complex_rmse": echo_rmse,
        "oracle_normalized_complex_rmse": oracle_rmse,
        "rmse_gap_fraction_closed": (
            (echo_rmse - oracle_rmse) / echo_rmse if echo_rmse > 0 else 0.0
        ),
        "echo_complex_coherence": float(echo_metrics["complex_coherence"]),
        "oracle_complex_coherence": float(oracle_metrics["complex_coherence"]),
        "echo_log_magnitude_ssim": float(echo_metrics["log_magnitude_ssim"]),
        "oracle_log_magnitude_ssim": float(oracle_metrics["log_magnitude_ssim"]),
        "echo_edge_correlation": float(echo_metrics["edge_correlation"]),
        "oracle_edge_correlation": float(oracle_metrics["edge_correlation"]),
        "echo_high_frequency_energy_ratio": float(echo_metrics["high_frequency_energy_ratio"]),
        "oracle_high_frequency_energy_ratio": float(oracle_metrics["high_frequency_energy_ratio"]),
        "identity_phase_alignment": weighted_mean(np.real(correction), weights),
        "phase_resultant_length": float(abs(np.sum(correction * weights)) / weights.sum()) if float(weights.sum()) > 0 else 0.0,
        "phase_neighbor_roughness": phase_roughness(correction, weights),
        "reliable_frequency_fraction": float(reliable.mean()),
        "log_amplitude_ratio_mean": weighted_mean(log_ratio, weights),
        "log_amplitude_ratio_std": math.sqrt(
            max(
                weighted_mean(
                    (log_ratio - weighted_mean(log_ratio, weights)) ** 2,
                    weights,
                ),
                0.0,
            )
        ),
    }
    return PhaseSample(
        record=record,
        correction=correction,
        weights=weights,
        metrics=metrics,
        band_rows=radial_band_rows(record.pair.key, correction, weights, reliable),
        pca_feature=pooled_phase_feature(correction, weights, pca_grid_size),
    )


def aggregate_phase_metrics(samples: Sequence[PhaseSample]) -> dict[str, Any]:
    return {
        name: distribution([float(sample.metrics[name]) for sample in samples])
        for name in PHASE_METRIC_NAMES
    }


def save_phase_sample_figure(
    sample: PhaseSample,
    output_dir: Path,
    *,
    floor_db: float,
) -> None:
    echo, image = load_pair(sample.record.pair)
    normalized_echo, normalized_image, oracle, _, _ = normalized_phase_oracle(
        echo, image, fft_norm="ortho", phasor_epsilon=1.0e-6
    )
    peak = max(float(np.abs(normalized_image).max()), np.finfo(np.float64).tiny)
    images = (
        log_magnitude_image(normalized_echo, reference_peak=peak, floor_db=floor_db),
        log_magnitude_image(oracle, reference_peak=peak, floor_db=floor_db),
        log_magnitude_image(normalized_image, reference_peak=peak, floor_db=floor_db),
        np.angle(sample.correction),
        np.log1p(sample.weights),
    )
    titles = (
        "Echo",
        "Phase-only Oracle",
        "Image",
        "Oracle correction phase",
        "log(1 + phase weight)",
    )
    figure, axes = plt.subplots(1, 5, figsize=(18, 4), constrained_layout=True)
    for index, (axis, values, title) in enumerate(zip(axes, images, titles, strict=True)):
        if index == 3:
            axis.imshow(values, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        elif index == 4:
            axis.imshow(values, cmap="magma")
        else:
            axis.imshow(values, cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{sample.record.pair.key} | RMSE gap closed="
        f"{float(sample.metrics['rmse_gap_fraction_closed']):.4f} | "
        f"phase roughness={float(sample.metrics['phase_neighbor_roughness']):.4f}",
        fontsize=10,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{sample.record.pair.key}.png", dpi=150)
    plt.close(figure)


def save_phase_metric_distributions(
    samples: Sequence[PhaseSample], output_dir: Path
) -> None:
    if not samples:
        return
    metrics = (
        "rmse_gap_fraction_closed",
        "identity_phase_alignment",
        "phase_resultant_length",
        "phase_neighbor_roughness",
        "log_amplitude_ratio_std",
    )
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, name in zip(axes.ravel(), metrics, strict=False):
        values = [float(sample.metrics[name]) for sample in samples]
        axis.hist(values, bins=min(30, max(5, int(math.sqrt(len(values))))))
        axis.set_title(name)
        axis.grid(alpha=0.2)
    for axis in axes.ravel()[len(metrics) :]:
        axis.axis("off")
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / "metric_distributions.png", dpi=160)
    plt.close(figure)


def phase_neighbor_rows(samples: Sequence[PhaseSample]) -> list[dict[str, Any]]:
    by_parent: dict[str, list[PhaseSample]] = defaultdict(list)
    for sample in samples:
        by_parent[sample.record.parent_id].append(sample)
    result: list[dict[str, Any]] = []
    for parent_id, group in sorted(by_parent.items()):
        for first_index, first in enumerate(group):
            candidates = group[first_index + 1 :]
            if not candidates:
                continue
            second = min(
                candidates,
                key=lambda item: abs(item.record.row - first.record.row)
                + abs(item.record.col - first.record.col),
            )
            weights = np.sqrt(first.weights * second.weights)
            similarity = weighted_mean(
                np.real(first.correction * np.conj(second.correction)), weights
            )
            result.append(
                {
                    "parent_id": parent_id,
                    "first_key": first.record.pair.key,
                    "second_key": second.record.pair.key,
                    "row_distance": abs(second.record.row - first.record.row),
                    "col_distance": abs(second.record.col - first.record.col),
                    "manhattan_distance": abs(second.record.row - first.record.row)
                    + abs(second.record.col - first.record.col),
                    "weighted_phase_similarity": similarity,
                }
            )
    return result


def pca_analysis(
    samples: Sequence[PhaseSample], output_dir: Path, grid_size: int
) -> dict[str, Any]:
    if len(samples) < 2:
        write_csv(output_dir / "pca_explained_variance.csv", [])
        return {"status": "insufficient_samples", "sample_count": len(samples)}
    features = np.stack([sample.pca_feature for sample in samples]).astype(np.float64)
    centered = features - features.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(centered, full_matrices=False)
    variance = np.square(singular_values) / max(len(samples) - 1, 1)
    total = float(variance.sum())
    if total <= 0:
        write_csv(output_dir / "pca_explained_variance.csv", [])
        return {
            "status": "zero_variance",
            "sample_count": len(samples),
            "feature_grid_size": grid_size,
            "feature_dimension": int(features.shape[1]),
        }
    ratios = variance / total if total > 0 else np.zeros_like(variance)
    cumulative = np.cumsum(ratios)
    rows = [
        {
            "component": index + 1,
            "explained_variance_ratio": float(ratio),
            "cumulative_explained_variance_ratio": float(cumulative[index]),
        }
        for index, ratio in enumerate(ratios)
    ]
    write_csv(output_dir / "pca_explained_variance.csv", rows)

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    axis.plot(np.arange(1, len(cumulative) + 1), cumulative, marker=".")
    for threshold in (0.90, 0.95, 0.99):
        axis.axhline(threshold, linestyle="--", linewidth=0.8)
    axis.set_xlabel("Component count")
    axis.set_ylabel("Cumulative explained variance")
    axis.set_ylim(0, 1.01)
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "pca_explained_variance.png", dpi=160)
    plt.close(figure)

    component_dir = output_dir / "pca_components"
    component_dir.mkdir(parents=True, exist_ok=True)
    cells = grid_size * grid_size
    for index in range(min(4, components.shape[0])):
        real = components[index, :cells].reshape(grid_size, grid_size)
        imag = components[index, cells:].reshape(grid_size, grid_size)
        magnitude = np.hypot(real, imag)
        phase = np.angle(real + 1j * imag)
        figure, axes = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
        axes[0].imshow(magnitude, cmap="magma")
        axes[0].set_title(f"PC {index + 1} magnitude")
        axes[1].imshow(phase, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[1].set_title(f"PC {index + 1} direction")
        for axis in axes:
            axis.axis("off")
        figure.savefig(component_dir / f"component_{index + 1:02d}.png", dpi=160)
        plt.close(figure)

    def count_for(threshold: float) -> int:
        return min(
            int(np.searchsorted(cumulative, threshold, side="left") + 1),
            len(cumulative),
        )

    scores = centered @ components[: min(8, components.shape[0])].T
    coordinates = np.asarray(
        [[sample.record.row, sample.record.col] for sample in samples], dtype=np.float64
    )
    coordinate_result: list[dict[str, Any]] = []
    if float(coordinates.std(axis=0).min()) > 0:
        normalized_coordinates = (
            coordinates - coordinates.mean(axis=0, keepdims=True)
        ) / coordinates.std(axis=0, keepdims=True)
        design = np.column_stack((np.ones(len(samples)), normalized_coordinates))
        for index in range(scores.shape[1]):
            target = scores[:, index]
            coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
            predicted = design @ coefficients
            total_error = float(np.sum((target - target.mean()) ** 2))
            residual = float(np.sum((target - predicted) ** 2))
            coordinate_result.append(
                {
                    "component": index + 1,
                    "coordinate_linear_r2": 1.0 - residual / total_error if total_error > 0 else 0.0,
                    "row_coefficient": float(coefficients[1]),
                    "col_coefficient": float(coefficients[2]),
                }
            )
    write_csv(output_dir / "coordinate_dependence.csv", coordinate_result)
    return {
        "status": "completed",
        "sample_count": len(samples),
        "feature_grid_size": grid_size,
        "feature_dimension": int(features.shape[1]),
        "components_for_90_percent": count_for(0.90),
        "components_for_95_percent": count_for(0.95),
        "components_for_99_percent": count_for(0.99),
        "coordinate_dependence": coordinate_result,
    }


def selected_group_overlap_rows(
    records: Sequence[CoordinatePair], requested: int
) -> list[dict[str, Any]]:
    by_parent: dict[str, list[CoordinatePair]] = defaultdict(list)
    for record in records:
        by_parent[record.parent_id].append(record)
    rows: list[dict[str, Any]] = []
    for parent_id, group in sorted(by_parent.items()):
        index = {(record.row, record.col): record for record in group}
        grid = group_grid_summary(group)
        edges: list[tuple[str, CoordinatePair, CoordinatePair, int]] = []
        for axis, step in (("row", grid["modal_row_step"]), ("col", grid["modal_col_step"])):
            if step is None:
                continue
            for record in group:
                neighbor_coordinate = (
                    (record.row + step, record.col)
                    if axis == "row"
                    else (record.row, record.col + step)
                )
                neighbor = index.get(neighbor_coordinate)
                if neighbor is not None:
                    edges.append((axis, record, neighbor, int(step)))
        selected = [edges[index] for index in evenly_spaced_indices(len(edges), requested)]
        for axis, first, second, step in selected:
            try:
                first_echo, first_image = load_pair(first.pair)
                second_echo, second_image = load_pair(second.pair)
                for role in ("echo", "image"):
                    first_values = first_echo if role == "echo" else first_image
                    second_values = second_echo if role == "echo" else second_image
                    first_overlap, second_overlap = overlap_regions(
                        first_values, second_values, axis, step
                    )
                    metrics = overlap_metrics(first_overlap, second_overlap)
                    rows.append(
                        {
                            "parent_id": parent_id,
                            "role": role,
                            "axis": axis,
                            "first_file": first.pair.echo.name if role == "echo" else first.pair.image.name,
                            "second_file": second.pair.echo.name if role == "echo" else second.pair.image.name,
                            "coordinate_step": step,
                            **metrics,
                        }
                    )
            except Exception as error:
                rows.append(
                    {
                        "parent_id": parent_id,
                        "role": "pair",
                        "axis": axis,
                        "first_file": first.pair.echo.name,
                        "second_file": second.pair.echo.name,
                        "coordinate_step": step,
                        "error": repr(error),
                    }
                )
    return rows


def group_stitchability(
    records: Sequence[CoordinatePair], overlap_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    groups: dict[str, list[CoordinatePair]] = defaultdict(list)
    for record in records:
        groups[record.parent_id].append(record)
    result: dict[str, Any] = {}
    for parent_id, group in sorted(groups.items()):
        grid = group_grid_summary(group)
        rows = [
            row
            for row in overlap_rows
            if row.get("parent_id") == parent_id and "error" not in row
        ]
        role_result: dict[str, Any] = {}
        for role in ("echo", "image"):
            selected = [row for row in rows if row.get("role") == role]
            complex_corr = distribution([row["complex_correlation"] for row in selected])
            magnitude_corr = distribution([row["magnitude_correlation"] for row in selected])
            relative_rmse = distribution([row["relative_rmse"] for row in selected])
            complex_valid = bool(selected) and float(complex_corr.get("median") or 0.0) >= 0.95 and float(relative_rmse.get("median") or math.inf) <= 0.20
            magnitude_valid = bool(selected) and float(magnitude_corr.get("median") or 0.0) >= 0.90
            role_result[role] = {
                "analyzed_overlap_count": len(selected),
                "complex_correlation": complex_corr,
                "magnitude_correlation": magnitude_corr,
                "relative_rmse": relative_rmse,
                "complex_stitching_valid": complex_valid,
                "magnitude_stitching_valid": magnitude_valid or complex_valid,
            }
        coordinate_valid = (
            grid["coordinate_collision_count"] == 0
            and grid["modal_row_step"] is not None
            and grid["modal_col_step"] is not None
        )
        result[parent_id] = {
            "grid": grid,
            "coordinate_stitching_valid": coordinate_valid,
            "roles": role_result,
            "preview_allowed": coordinate_valid
            and all(role_result[role]["magnitude_stitching_valid"] for role in ("echo", "image")),
        }
    return result


def preview_scale(full_height: int, full_width: int, max_pixels: int) -> int:
    return max(1, int(math.ceil(math.sqrt(full_height * full_width / max_pixels))))


def stitch_previews(
    records: Sequence[CoordinatePair],
    *,
    max_preview_pixels: int,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]]:
    first_echo, first_image = load_pair(records[0].pair)
    patch_shape = first_echo.shape
    min_row = min(record.row for record in records)
    min_col = min(record.col for record in records)
    full_height = max(record.row for record in records) - min_row + patch_shape[0]
    full_width = max(record.col for record in records) - min_col + patch_shape[1]
    downsample_factor = preview_scale(full_height, full_width, max_preview_pixels)
    canvas_height = int(math.ceil(full_height / downsample_factor))
    canvas_width = int(math.ceil(full_width / downsample_factor))
    roles = ("echo", "image", "oracle")
    sums = {
        role: np.zeros((canvas_height, canvas_width), dtype=np.complex64)
        for role in roles
    }
    sums_sq = {
        role: np.zeros((canvas_height, canvas_width), dtype=np.float32)
        for role in roles
    }
    counts = np.zeros((canvas_height, canvas_width), dtype=np.uint16)
    for record in records:
        echo, image = load_pair(record.pair)
        if echo.shape != patch_shape or image.shape != patch_shape:
            raise ValueError("mosaic group contains inconsistent patch shapes")
        oracle_scale = math.sqrt(float(np.mean(np.abs(echo) ** 2)) + 1.0e-12)
        oracle = oracle_scale * normalized_phase_oracle(
            echo, image, fft_norm="ortho", phasor_epsilon=1.0e-6
        )[2]
        sampled = {
            role: values[::downsample_factor, ::downsample_factor].astype(
                np.complex64, copy=False
            )
            for role, values in (("echo", echo), ("image", image), ("oracle", oracle))
        }
        row = int(round((record.row - min_row) / downsample_factor))
        col = int(round((record.col - min_col) / downsample_factor))
        height = min(sampled["echo"].shape[0], canvas_height - row)
        width = min(sampled["echo"].shape[1], canvas_width - col)
        target = (slice(row, row + height), slice(col, col + width))
        for role in roles:
            current = sampled[role][:height, :width]
            sums[role][target] += current
            sums_sq[role][target] += np.square(np.abs(current))
        counts[target] += 1
    covered = counts > 0
    result: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for role in roles:
        mosaic = np.zeros_like(sums[role])
        mosaic[covered] = sums[role][covered] / counts[covered]
        variance = np.zeros_like(sums_sq[role])
        variance[covered] = np.maximum(
            sums_sq[role][covered] / counts[covered]
            - np.square(np.abs(mosaic[covered])),
            0.0,
        )
        disagreement = np.sqrt(variance)
        result[role] = (
            mosaic,
            counts,
            disagreement,
            {
                "role": role,
                "full_resolution_shape": [full_height, full_width],
                "preview_shape": [canvas_height, canvas_width],
                "preview_downsample_factor": downsample_factor,
                "coverage_fraction": float(covered.mean()),
                "overlap_fraction": float((counts > 1).mean()),
                "maximum_coverage": int(counts.max()),
            },
        )
    return result


def save_mosaic_role(
    output_dir: Path,
    role: str,
    mosaic: np.ndarray,
    counts: np.ndarray,
    disagreement: np.ndarray,
    floor_db: float,
) -> None:
    covered = counts > 0
    peak = float(np.abs(mosaic[covered]).max()) if bool(covered.any()) else 1.0
    peak = max(peak, np.finfo(np.float64).tiny)
    log_image = log_magnitude_image(mosaic, reference_peak=peak, floor_db=floor_db)
    log_image[~covered] = 0.0
    plt.imsave(output_dir / f"{role}_log_magnitude.png", log_image, cmap="gray", vmin=0, vmax=1)
    phase = np.angle(mosaic)
    phase[~covered] = 0.0
    plt.imsave(
        output_dir / f"{role}_phase.png",
        phase,
        cmap="twilight",
        vmin=-math.pi,
        vmax=math.pi,
    )
    plt.imsave(output_dir / f"{role}_coverage_count.png", counts, cmap="viridis")
    normalized_disagreement = disagreement / peak
    plt.imsave(
        output_dir / f"{role}_overlap_disagreement.png",
        normalized_disagreement,
        cmap="magma",
        vmin=0,
        vmax=max(float(np.percentile(normalized_disagreement[covered], 99)) if bool(covered.any()) else 1.0, 1.0e-12),
    )
    plt.imsave(
        output_dir / f"{role}_seam_error.png",
        normalized_disagreement * (counts > 1),
        cmap="magma",
        vmin=0,
        vmax=max(float(np.percentile(normalized_disagreement[covered], 99)) if bool(covered.any()) else 1.0, 1.0e-12),
    )


def save_mosaics(
    records: Sequence[CoordinatePair],
    stitchability: dict[str, Any],
    output_dir: Path,
    *,
    max_preview_pixels: int,
    max_complex_mosaic_pixels: int,
    floor_db: float,
) -> dict[str, Any]:
    groups: dict[str, list[CoordinatePair]] = defaultdict(list)
    for record in records:
        groups[record.parent_id].append(record)
    report: dict[str, Any] = {}
    for parent_id, group in sorted(groups.items()):
        gate = stitchability[parent_id]
        group_dir = output_dir / parent_id
        if not gate["preview_allowed"]:
            report[parent_id] = {
                "status": "skipped_by_stitchability_gate",
                "gate": gate,
            }
            continue
        group_dir.mkdir(parents=True, exist_ok=True)
        role_reports: dict[str, Any] = {}
        previews = stitch_previews(
            group,
            max_preview_pixels=max_preview_pixels,
        )
        for role, (mosaic, counts, disagreement, role_report) in previews.items():
            save_mosaic_role(group_dir, role, mosaic, counts, disagreement, floor_db)
            if mosaic.size <= max_complex_mosaic_pixels:
                np.savez_compressed(
                    group_dir / f"{role}_complex_mosaic.npz",
                    mosaic=mosaic,
                    coverage_count=counts,
                    disagreement=disagreement,
                )
                role_report["complex_preview_saved"] = True
            else:
                role_report["complex_preview_saved"] = False
                role_report["complex_preview_skip_reason"] = "preview exceeds max_complex_mosaic_pixels"
            role_reports[role] = role_report
        report[parent_id] = {
            "status": "completed",
            "sample_count": len(group),
            "roles": role_reports,
        }
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    validate_source_output_separation(args.echo_dir, args.image_dir, args.output_dir)
    pairs, pairing = discover_pairs(args.echo_dir, args.image_dir)
    if not pairs:
        raise RuntimeError("no Echo/Image file pairs were found")
    args.output_dir.mkdir(parents=True, exist_ok=False)

    structure = analyze_structure(
        argparse.Namespace(
            echo_dir=args.echo_dir,
            image_dir=args.image_dir,
            sample_count=args.numeric_sample_count,
            alignment_count=args.alignment_sample_count,
            progress_every=args.progress_every,
        )
    )
    write_json(args.output_dir / "dataset_structure.json", structure)

    adjacency = analyze_adjacency(
        argparse.Namespace(
            echo_dir=args.echo_dir,
            image_dir=args.image_dir,
            pairs_per_axis=args.adjacent_pairs_per_axis,
            shift_tolerance=args.shift_tolerance,
            progress_every=args.progress_every,
        )
    )
    write_json(args.output_dir / "patch_adjacency.json", adjacency)

    records, grouping = coordinate_records(pairs)
    grouped: dict[str, list[CoordinatePair]] = defaultdict(list)
    for record in records:
        grouped[record.parent_id].append(record)
    grouping["groups"] = {
        parent_id: group_grid_summary(group)
        for parent_id, group in sorted(grouped.items())
    }
    write_json(args.output_dir / "grouping" / "parent_image_candidates.json", grouping)
    write_csv(
        args.output_dir / "grouping" / "grouping_evidence.csv",
        [
            {
                "parent_id": parent_id,
                **group_grid_summary(group),
                "grouping_method": grouping["grouping_method"],
                "confidence": grouping["initial_confidence"],
            }
            for parent_id, group in sorted(grouped.items())
        ],
    )
    write_csv(
        args.output_dir / "grouping" / "ambiguous_files.csv",
        [{"filename": filename, "reason": "coordinate_filename_unparsed"} for filename in grouping["unparsed_examples"]],
    )

    schema = metadata_schema(pairs, args.metadata_sample_count)
    write_json(args.output_dir / "metadata_schema.json", schema)

    inventory_rows = [
        {
            "key": pair.key,
            "echo_file": pair.echo.name,
            "image_file": pair.image.name,
            "paired": True,
            "coordinate_parsed": parse_coordinate(pair.echo) is not None,
            "row": parse_coordinate(pair.echo)[0] if parse_coordinate(pair.echo) else None,
            "col": parse_coordinate(pair.echo)[1] if parse_coordinate(pair.echo) else None,
            "filename_suffix": parse_coordinate(pair.echo)[2] if parse_coordinate(pair.echo) else None,
            "echo_size_bytes": pair.echo.stat().st_size,
            "image_size_bytes": pair.image.stat().st_size,
        }
        for pair in pairs
    ]
    write_csv(args.output_dir / "file_inventory.csv", inventory_rows)
    write_csv(
        args.output_dir / "pair_manifest.csv",
        pair_manifest_rows(args.echo_dir, args.image_dir),
    )

    phase_records = [
        records[index] for index in evenly_spaced_indices(len(records), args.phase_sample_count)
    ]
    phase_samples: list[PhaseSample] = []
    phase_errors: list[dict[str, str]] = []
    for position, record in enumerate(phase_records, start=1):
        try:
            phase_samples.append(
                phase_sample(
                    record,
                    fft_norm=args.fft_norm,
                    phasor_epsilon=args.phasor_epsilon,
                    floor_db=args.floor_db,
                    high_frequency_radius_fraction=args.high_frequency_radius_fraction,
                    pca_grid_size=args.pca_grid_size,
                )
            )
        except Exception as error:
            if len(phase_errors) < MAX_EXAMPLES:
                phase_errors.append({"key": record.pair.key, "error": repr(error)})
        if args.progress_every and (
            position % args.progress_every == 0 or position == len(phase_records)
        ):
            print(
                f"phase: checked {position:,}/{len(phase_records):,}; "
                f"analyzed={len(phase_samples):,}",
                flush=True,
            )

    phase_dir = args.output_dir / "phase_analysis"
    write_csv(phase_dir / "per_sample_phase_metrics.csv", [dict(sample.metrics) for sample in phase_samples])
    band_rows = [row for sample in phase_samples for row in sample.band_rows]
    write_csv(phase_dir / "frequency_band_statistics.csv", band_rows)
    neighbor_rows = phase_neighbor_rows(phase_samples)
    write_csv(phase_dir / "neighbor_phase_similarity.csv", neighbor_rows)
    figure_samples = [
        phase_samples[index]
        for index in evenly_spaced_indices(len(phase_samples), args.phase_figure_count)
    ]
    for sample in figure_samples:
        save_phase_sample_figure(
            sample,
            phase_dir / "figures" / "representative_samples",
            floor_db=args.floor_db,
        )
    save_phase_metric_distributions(phase_samples, phase_dir / "figures")
    pca = pca_analysis(phase_samples, phase_dir, args.pca_grid_size)
    phase_report = {
        "requested_sample_count": args.phase_sample_count,
        "selected_sample_count": len(phase_records),
        "analyzed_sample_count": len(phase_samples),
        "error_count": len(phase_records) - len(phase_samples),
        "error_examples": phase_errors,
        "aggregate_metrics": aggregate_phase_metrics(phase_samples) if phase_samples else {},
        "neighbor_phase_similarity": distribution(
            [row["weighted_phase_similarity"] for row in neighbor_rows]
        ),
        "pca": pca,
    }
    write_json(phase_dir / "oracle_recoverability.json", phase_report)

    overlap_rows = selected_group_overlap_rows(records, args.group_overlap_pair_count)
    write_csv(args.output_dir / "coordinates" / "adjacent_patch_metrics.csv", overlap_rows)
    stitchability = group_stitchability(records, overlap_rows)
    write_json(args.output_dir / "coordinates" / "coordinate_grids.json", stitchability)
    write_json(
        args.output_dir / "coordinates" / "overlap_components.json",
        coordinate_components(records),
    )
    mosaic_report = save_mosaics(
        records,
        stitchability,
        args.output_dir / "mosaics",
        max_preview_pixels=args.max_preview_pixels,
        max_complex_mosaic_pixels=args.max_complex_mosaic_pixels,
        floor_db=args.floor_db,
    )

    median_oracle_closed = (
        phase_report.get("aggregate_metrics", {})
        .get("rmse_gap_fraction_closed", {})
        .get("median")
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "read_only_contract": {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "source_mutation": "forbidden",
            "all_writes_below": str(args.output_dir.resolve()),
            "dataset_splitting": "not_implemented_by_explicit_request",
        },
        "pairing": pairing,
        "grouping": grouping,
        "phase_analysis": phase_report,
        "stitchability": stitchability,
        "mosaics": mosaic_report,
        "decision_gates": {
            "pairing_complete": pairing["echo_only_count"] == 0
            and pairing["image_only_count"] == 0
            and pairing["echo_collision_count"] == 0
            and pairing["image_collision_count"] == 0,
            "parent_grouping_confirmed": False,
            "parent_grouping_requires_review": True,
            "phase_oracle_recoverable": (
                median_oracle_closed is not None and float(median_oracle_closed) >= 0.50
            ),
            "mosaic_preview_count": sum(
                result.get("status") == "completed" for result in mosaic_report.values()
            ),
        },
        "required_human_review": [
            "Confirm whether filename suffixes or MAT metadata identify original images/angles.",
            "Inspect overlap metrics before accepting any mosaic as a physical reconstruction.",
            "Inspect phase-only Oracle figures/metrics before starting phase-model training.",
        ],
    }
    write_json(args.output_dir / "summary.json", summary)
    print_summary(summary, args.output_dir)
    return summary


def print_summary(report: dict[str, Any], output_dir: Path) -> None:
    pairing = report["pairing"]
    phase = report["phase_analysis"]
    gates = report["decision_gates"]
    print("\nPhase dataset audit")
    print("=" * 19)
    print(
        f"pairs={pairing['paired_files']:,} echo_only={pairing['echo_only_count']:,} "
        f"image_only={pairing['image_only_count']:,}"
    )
    print(
        f"phase samples={phase['analyzed_sample_count']:,}/"
        f"{phase['selected_sample_count']:,} errors={phase['error_count']:,}"
    )
    closed = phase.get("aggregate_metrics", {}).get("rmse_gap_fraction_closed", {})
    print(
        "phase-only Oracle RMSE gap closed median="
        f"{closed.get('median', 'n/a')} p05={closed.get('p05', 'n/a')}"
    )
    print(
        f"mosaic previews={gates['mosaic_preview_count']} "
        f"parent grouping confirmed={gates['parent_grouping_confirmed']}"
    )
    print("No dataset split was generated. Source MAT files were read only.")
    print(f"Report: {(output_dir / 'summary.json').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only paired SAR audit with phase-only Oracle, phase structure, "
            "candidate parent grouping, overlap gates, and mosaic previews."
        )
    )
    parser.add_argument("--echo-dir", type=Path, default=DEFAULT_ECHO_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--numeric-sample-count", type=int, default=3000)
    parser.add_argument("--alignment-sample-count", type=int, default=200)
    parser.add_argument("--metadata-sample-count", type=int, default=200)
    parser.add_argument("--phase-sample-count", type=int, default=256)
    parser.add_argument("--phase-figure-count", type=int, default=12)
    parser.add_argument("--adjacent-pairs-per-axis", type=int, default=100)
    parser.add_argument("--group-overlap-pair-count", type=int, default=100)
    parser.add_argument("--shift-tolerance", type=int, default=2)
    parser.add_argument("--pca-grid-size", type=int, default=32)
    parser.add_argument("--fft-norm", choices=("ortho", "backward", "forward"), default="ortho")
    parser.add_argument("--phasor-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--floor-db", type=float, default=-60.0)
    parser.add_argument("--high-frequency-radius-fraction", type=float, default=0.25)
    parser.add_argument("--max-preview-pixels", type=int, default=16_000_000)
    parser.add_argument("--max-complex-mosaic-pixels", type=int, default=16_000_000)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    nonnegative = (
        "numeric_sample_count",
        "alignment_sample_count",
        "metadata_sample_count",
        "phase_sample_count",
        "phase_figure_count",
        "adjacent_pairs_per_axis",
        "group_overlap_pair_count",
        "shift_tolerance",
        "progress_every",
    )
    for name in nonnegative:
        if int(getattr(args, name)) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")
    positive = ("pca_grid_size", "max_preview_pixels", "max_complex_mosaic_pixels")
    for name in positive:
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.phasor_epsilon) or args.phasor_epsilon <= 0:
        parser.error("--phasor-epsilon must be finite and positive")
    if not math.isfinite(args.floor_db) or args.floor_db >= 0:
        parser.error("--floor-db must be finite and negative")
    if not 0 < args.high_frequency_radius_fraction < 1:
        parser.error("--high-frequency-radius-fraction must be in (0, 1)")
    return args


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
