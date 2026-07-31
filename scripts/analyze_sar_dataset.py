"""Analyze paired defocused/focused SAR patches without modifying source data.

The server dataset is expected to contain MATLAB 7.3 (HDF5) files.  Echo files
provide ``coarse_patch`` and image files provide ``pfa_patch``.  The script
checks every pair structurally, then reads a deterministic, evenly spaced
subset for numerical statistics.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_ECHO_DIR = Path("/data/dyn/capella/output/echo")
DEFAULT_IMAGE_DIR = Path("/data/dyn/capella/output/image")
ROLE_SUFFIX = re.compile(r"_(?:echo|image|iamge)_(\d+)$", re.IGNORECASE)
MAX_EXAMPLES = 20


@dataclass(frozen=True)
class FilePair:
    key: str
    echo: Path
    image: Path


class PixelMoments:
    """Accumulate exact first- and second-order pixel statistics."""

    def __init__(self) -> None:
        self.total = 0
        self.finite = 0
        self.nonfinite = 0
        self.zeros = 0
        self.real_sum = 0.0
        self.real_sum_sq = 0.0
        self.imag_sum = 0.0
        self.imag_sum_sq = 0.0
        self.mag_sum = 0.0
        self.mag_sum_sq = 0.0
        self.mag_min = math.inf
        self.mag_max = -math.inf

    def update(self, values: np.ndarray) -> None:
        self.total += values.size
        mask = np.isfinite(values.real) & np.isfinite(values.imag)
        self.finite += int(mask.sum())
        self.nonfinite += int(values.size - mask.sum())
        if not mask.any():
            return

        finite_values = values[mask]
        real = finite_values.real.astype(np.float64, copy=False)
        imag = finite_values.imag.astype(np.float64, copy=False)
        magnitude = np.hypot(real, imag)
        self.zeros += int(np.count_nonzero(magnitude == 0))
        self.real_sum += float(real.sum(dtype=np.float64))
        self.real_sum_sq += float(np.dot(real, real))
        self.imag_sum += float(imag.sum(dtype=np.float64))
        self.imag_sum_sq += float(np.dot(imag, imag))
        self.mag_sum += float(magnitude.sum(dtype=np.float64))
        self.mag_sum_sq += float(np.dot(magnitude, magnitude))
        self.mag_min = min(self.mag_min, float(magnitude.min()))
        self.mag_max = max(self.mag_max, float(magnitude.max()))

    def as_dict(self) -> dict[str, Any]:
        if self.finite == 0:
            return {
                "total_pixels": self.total,
                "finite_pixels": 0,
                "nonfinite_pixels": self.nonfinite,
            }

        def mean_std(total: float, total_sq: float) -> tuple[float, float]:
            mean = total / self.finite
            variance = max(total_sq / self.finite - mean * mean, 0.0)
            return mean, math.sqrt(variance)

        real_mean, real_std = mean_std(self.real_sum, self.real_sum_sq)
        imag_mean, imag_std = mean_std(self.imag_sum, self.imag_sum_sq)
        mag_mean, mag_std = mean_std(self.mag_sum, self.mag_sum_sq)
        return {
            "total_pixels": self.total,
            "finite_pixels": self.finite,
            "nonfinite_pixels": self.nonfinite,
            "zero_magnitude_pixels": self.zeros,
            "real_mean": real_mean,
            "real_std": real_std,
            "imag_mean": imag_mean,
            "imag_std": imag_std,
            "magnitude_mean": mag_mean,
            "magnitude_std": mag_std,
            "magnitude_rms": math.sqrt(self.mag_sum_sq / self.finite),
            "magnitude_min": self.mag_min,
            "magnitude_max": self.mag_max,
        }


def canonical_pair_key(path: Path) -> str:
    """Remove the role token while preserving the sample index."""
    stem = path.stem
    match = ROLE_SUFFIX.search(stem)
    if match is None:
        return stem
    return f"{stem[:match.start()]}__sample_{match.group(1)}"


def mat_files(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() == ".mat"
    )


def index_files(files: list[Path]) -> tuple[dict[str, Path], dict[str, list[str]]]:
    grouped: dict[str, list[Path]] = {}
    for path in files:
        grouped.setdefault(canonical_pair_key(path), []).append(path)

    unique = {key: paths[0] for key, paths in grouped.items() if len(paths) == 1}
    collisions = {
        key: [path.name for path in paths]
        for key, paths in grouped.items()
        if len(paths) > 1
    }
    return unique, collisions


def discover_pairs(
    echo_dir: Path, image_dir: Path
) -> tuple[list[FilePair], dict[str, Any]]:
    echo_files = mat_files(echo_dir)
    image_files = mat_files(image_dir)
    echo_index, echo_collisions = index_files(echo_files)
    image_index, image_collisions = index_files(image_files)

    paired_keys = sorted(echo_index.keys() & image_index.keys())
    echo_only = sorted(echo_index.keys() - image_index.keys())
    image_only = sorted(image_index.keys() - echo_index.keys())
    pairs = [
        FilePair(key=key, echo=echo_index[key], image=image_index[key])
        for key in paired_keys
    ]
    summary = {
        "echo_mat_files": len(echo_files),
        "image_mat_files": len(image_files),
        "paired_files": len(pairs),
        "echo_only_count": len(echo_only),
        "image_only_count": len(image_only),
        "echo_only_examples": echo_only[:MAX_EXAMPLES],
        "image_only_examples": image_only[:MAX_EXAMPLES],
        "echo_collision_count": len(echo_collisions),
        "image_collision_count": len(image_collisions),
        "echo_collision_examples": dict(list(echo_collisions.items())[:MAX_EXAMPLES]),
        "image_collision_examples": dict(list(image_collisions.items())[:MAX_EXAMPLES]),
    }
    return pairs, summary


def evenly_spaced_indices(total: int, requested: int) -> list[int]:
    if total == 0:
        return []
    if requested == 0 or requested >= total:
        return list(range(total))
    if requested < 0:
        raise ValueError("sample count must be non-negative")
    return np.rint(np.linspace(0, total - 1, requested)).astype(int).tolist()


def dtype_signature(dataset: h5py.Dataset) -> str:
    names = dataset.dtype.names
    if names and {"real", "imag"}.issubset(names):
        real_type = dataset.dtype.fields["real"][0]
        imag_type = dataset.dtype.fields["imag"][0]
        return f"compound(real={real_type},imag={imag_type})"
    return str(dataset.dtype)


def read_complex(dataset: h5py.Dataset) -> np.ndarray:
    raw = dataset[()]
    if np.iscomplexobj(raw):
        return np.asarray(raw, dtype=np.complex64)
    names = raw.dtype.names
    if names and {"real", "imag"}.issubset(names):
        return raw["real"].astype(np.float32) + 1j * raw["imag"].astype(np.float32)
    raise TypeError(f"expected a complex dataset, got dtype={raw.dtype}")


def scalar_value(file: h5py.File, name: str) -> float | None:
    if name not in file:
        return None
    dataset = file[name]
    if not isinstance(dataset, h5py.Dataset) or dataset.size != 1:
        return None
    value = np.asarray(dataset[()]).reshape(-1)[0]
    return float(value)


def per_array_metrics(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not finite.all():
        values = values[finite]
    if values.size == 0:
        raise ValueError("array contains no finite complex values")

    magnitude = np.abs(values).astype(np.float64, copy=False)
    power = magnitude * magnitude
    median = float(np.median(magnitude))
    return {
        "nonfinite": int(finite.size - finite.sum()),
        "rms": math.sqrt(float(power.mean())),
        "magnitude_median": median,
        "magnitude_p95": float(np.percentile(magnitude, 95)),
        "magnitude_p99": float(np.percentile(magnitude, 99)),
        "magnitude_p999": float(np.percentile(magnitude, 99.9)),
        "magnitude_max": float(magnitude.max()),
        "max_over_median": float(magnitude.max() / median) if median > 0 else None,
        "peak_power_fraction": float(power.max() / power.sum()) if power.sum() > 0 else None,
    }


def correlation_metrics(source: np.ndarray, target: np.ndarray) -> dict[str, float]:
    source_flat = source.ravel().astype(np.complex128, copy=False)
    target_flat = target.ravel().astype(np.complex128, copy=False)
    denominator = np.linalg.norm(source_flat) * np.linalg.norm(target_flat)
    complex_correlation = (
        float(abs(np.vdot(source_flat, target_flat)) / denominator)
        if denominator > 0
        else 0.0
    )

    source_mag = np.abs(source_flat)
    target_mag = np.abs(target_flat)
    source_centered = source_mag - source_mag.mean()
    target_centered = target_mag - target_mag.mean()
    magnitude_denominator = np.linalg.norm(source_centered) * np.linalg.norm(target_centered)
    magnitude_correlation = (
        float(np.dot(source_centered, target_centered) / magnitude_denominator)
        if magnitude_denominator > 0
        else 0.0
    )
    return {
        "complex_correlation": complex_correlation,
        "magnitude_correlation": magnitude_correlation,
    }


def best_circular_shift(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source_mag = np.abs(source).astype(np.float64)
    target_mag = np.abs(target).astype(np.float64)
    source_std = float(source_mag.std())
    target_std = float(target_mag.std())
    if source_std == 0 or target_std == 0:
        return {"correlation": 0.0, "shift": [0, 0]}

    source_norm = (source_mag - source_mag.mean()) / source_std
    target_norm = (target_mag - target_mag.mean()) / target_std
    correlation = np.fft.ifft2(
        np.fft.fft2(source_norm) * np.conj(np.fft.fft2(target_norm))
    ).real / source_norm.size
    peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    shift = [
        int(value if value <= size // 2 else value - size)
        for value, size in zip(peak, correlation.shape)
    ]
    return {"correlation": float(correlation[peak]), "shift": shift}


def distribution(values: list[float | int | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [value for value in values if value is not None and math.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "mean": None, "std": None}
    quantiles = np.percentile(finite, [0, 5, 25, 50, 75, 95, 100])
    return {
        "count": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "min": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "max": float(quantiles[6]),
    }


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def format_number(value: float | int | None) -> str:
    return "n/a" if value is None else f"{float(value):.6g}"


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    if not args.echo_dir.is_dir():
        raise FileNotFoundError(f"echo directory does not exist: {args.echo_dir}")
    if not args.image_dir.is_dir():
        raise FileNotFoundError(f"image directory does not exist: {args.image_dir}")

    pairs, pairing = discover_pairs(args.echo_dir, args.image_dir)
    if not pairs:
        raise RuntimeError("no file pairs were found; inspect unmatched filename examples")

    numeric_indices = set(evenly_spaced_indices(len(pairs), args.sample_count))
    alignment_positions = set(
        evenly_spaced_indices(len(numeric_indices), min(args.alignment_count, len(numeric_indices)))
    )
    numeric_order = {pair_index: order for order, pair_index in enumerate(sorted(numeric_indices))}

    source_shapes: Counter[str] = Counter()
    target_shapes: Counter[str] = Counter()
    source_dtypes: Counter[str] = Counter()
    target_dtypes: Counter[str] = Counter()
    source_compression: Counter[str] = Counter()
    target_compression: Counter[str] = Counter()
    target_x_shapes: Counter[str] = Counter()
    target_y_shapes: Counter[str] = Counter()
    source_sizes: list[int] = []
    target_sizes: list[int] = []
    structure_errors: list[dict[str, str]] = []
    numeric_errors: list[dict[str, str]] = []
    structure_error_count = 0
    numeric_error_count = 0
    shape_match_count = 0
    complex_layout_count = 0
    metadata_match_count = 0
    metadata_mismatch_count = 0
    metadata_mismatch_examples: list[dict[str, Any]] = []
    readable_pair_count = 0
    source_pixels = PixelMoments()
    target_pixels = PixelMoments()
    sample_rows: list[dict[str, Any]] = []

    metadata_pairs = {
        "x0": "x0",
        "y0": "y0",
        "z0": "z0",
        "row1_c": "row1_p",
        "row2_c": "row2_p",
        "col1_c": "col1_p",
        "col2_c": "col2_p",
    }

    for index, pair in enumerate(pairs):
        structure_complete = False
        try:
            source_sizes.append(pair.echo.stat().st_size)
            target_sizes.append(pair.image.stat().st_size)
            with h5py.File(pair.echo, "r") as echo_file, h5py.File(pair.image, "r") as image_file:
                if "coarse_patch" not in echo_file:
                    raise KeyError("echo file is missing coarse_patch")
                if "pfa_patch" not in image_file:
                    raise KeyError("image file is missing pfa_patch")
                source_dataset = echo_file["coarse_patch"]
                target_dataset = image_file["pfa_patch"]
                if not isinstance(source_dataset, h5py.Dataset):
                    raise TypeError("coarse_patch is not an HDF5 dataset")
                if not isinstance(target_dataset, h5py.Dataset):
                    raise TypeError("pfa_patch is not an HDF5 dataset")

                source_shape = str(tuple(source_dataset.shape))
                target_shape = str(tuple(target_dataset.shape))
                source_shapes[source_shape] += 1
                target_shapes[target_shape] += 1
                source_dtypes[dtype_signature(source_dataset)] += 1
                target_dtypes[dtype_signature(target_dataset)] += 1
                source_compression[str(source_dataset.compression or "none")] += 1
                target_compression[str(target_dataset.compression or "none")] += 1
                if source_dataset.shape == target_dataset.shape:
                    shape_match_count += 1

                source_names = set(source_dataset.dtype.names or ())
                target_names = set(target_dataset.dtype.names or ())
                if {"real", "imag"}.issubset(source_names) and {"real", "imag"}.issubset(target_names):
                    complex_layout_count += 1

                if "my_x_patch" in image_file:
                    target_x_shapes[str(tuple(image_file["my_x_patch"].shape))] += 1
                if "my_y_patch" in image_file:
                    target_y_shapes[str(tuple(image_file["my_y_patch"].shape))] += 1

                metadata_mismatches = []
                metadata_compared = 0
                for source_name, target_name in metadata_pairs.items():
                    source_value = scalar_value(echo_file, source_name)
                    target_value = scalar_value(image_file, target_name)
                    if source_value is None or target_value is None:
                        continue
                    metadata_compared += 1
                    if not math.isclose(source_value, target_value, rel_tol=0.0, abs_tol=1e-9):
                        metadata_mismatches.append(
                            f"{source_name}={source_value} vs {target_name}={target_value}"
                        )
                if metadata_compared:
                    if metadata_mismatches:
                        metadata_mismatch_count += 1
                        if len(metadata_mismatch_examples) < MAX_EXAMPLES:
                            metadata_mismatch_examples.append(
                                {"key": pair.key, "differences": metadata_mismatches}
                            )
                    else:
                        metadata_match_count += 1

                readable_pair_count += 1
                structure_complete = True
                if index not in numeric_indices:
                    continue

                source = read_complex(source_dataset)
                target = read_complex(target_dataset)
                if source.shape != target.shape:
                    raise ValueError(
                        f"numeric shapes differ: source={source.shape}, target={target.shape}"
                    )
                source_pixels.update(source)
                target_pixels.update(target)
                source_metrics = per_array_metrics(source)
                target_metrics = per_array_metrics(target)
                row: dict[str, Any] = {
                    "key": pair.key,
                    "echo_file": pair.echo.name,
                    "image_file": pair.image.name,
                    "source_rms": source_metrics["rms"],
                    "target_rms": target_metrics["rms"],
                    "rms_ratio_target_over_source": (
                        target_metrics["rms"] / source_metrics["rms"]
                        if source_metrics["rms"] > 0
                        else None
                    ),
                    "source_magnitude_median": source_metrics["magnitude_median"],
                    "target_magnitude_median": target_metrics["magnitude_median"],
                    "source_magnitude_p99": source_metrics["magnitude_p99"],
                    "target_magnitude_p99": target_metrics["magnitude_p99"],
                    "source_max_over_median": source_metrics["max_over_median"],
                    "target_max_over_median": target_metrics["max_over_median"],
                    "source_peak_power_fraction": source_metrics["peak_power_fraction"],
                    "target_peak_power_fraction": target_metrics["peak_power_fraction"],
                    "source_nonfinite": source_metrics["nonfinite"],
                    "target_nonfinite": target_metrics["nonfinite"],
                }
                row.update(correlation_metrics(source, target))
                if numeric_order[index] in alignment_positions:
                    row["alignment"] = best_circular_shift(source, target)
                sample_rows.append(row)
        except Exception as error:  # Continue so one corrupt file does not lose the report.
            if structure_complete and index in numeric_indices:
                numeric_error_count += 1
                if len(numeric_errors) < MAX_EXAMPLES:
                    numeric_errors.append({"key": pair.key, "error": repr(error)})
            else:
                structure_error_count += 1
                if len(structure_errors) < MAX_EXAMPLES:
                    structure_errors.append({"key": pair.key, "error": repr(error)})

        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(
                f"checked {index + 1:,}/{len(pairs):,} pairs; "
                f"numeric {len(sample_rows):,}/{len(numeric_indices):,}",
                file=sys.stderr,
                flush=True,
            )

    metric_names = [
        "source_rms",
        "target_rms",
        "rms_ratio_target_over_source",
        "source_magnitude_median",
        "target_magnitude_median",
        "source_magnitude_p99",
        "target_magnitude_p99",
        "source_max_over_median",
        "target_max_over_median",
        "source_peak_power_fraction",
        "target_peak_power_fraction",
        "complex_correlation",
        "magnitude_correlation",
    ]
    metric_distributions = {
        name: distribution([row[name] for row in sample_rows])
        for name in metric_names
    }
    alignment_rows = [row["alignment"] for row in sample_rows if "alignment" in row]
    shift_counts = Counter(str(tuple(row["shift"])) for row in alignment_rows)

    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "sample_count_requested": args.sample_count,
            "alignment_count_requested": args.alignment_count,
            "sampling_strategy": "evenly_spaced_after_sorting_canonical_pair_keys",
            "source_dataset": "coarse_patch",
            "target_dataset": "pfa_patch",
        },
        "pairing": pairing,
        "structure": {
            "pairs_checked": len(pairs),
            "readable_pairs": readable_pair_count,
            "structure_error_count": structure_error_count,
            "structure_error_examples": structure_errors,
            "matching_matrix_shapes": shape_match_count,
            "matlab_complex_layout_pairs": complex_layout_count,
            "metadata_match_pairs": metadata_match_count,
            "metadata_mismatch_pairs": metadata_mismatch_count,
            "metadata_mismatch_examples": metadata_mismatch_examples,
            "source_shapes": counter_dict(source_shapes),
            "target_shapes": counter_dict(target_shapes),
            "source_dtypes": counter_dict(source_dtypes),
            "target_dtypes": counter_dict(target_dtypes),
            "source_compression": counter_dict(source_compression),
            "target_compression": counter_dict(target_compression),
            "target_my_x_patch_shapes": counter_dict(target_x_shapes),
            "target_my_y_patch_shapes": counter_dict(target_y_shapes),
            "source_file_size_bytes": distribution([float(value) for value in source_sizes]),
            "target_file_size_bytes": distribution([float(value) for value in target_sizes]),
        },
        "numeric_sampling": {
            "selected_pairs": len(numeric_indices),
            "analyzed_pairs": len(sample_rows),
            "numeric_error_count": numeric_error_count,
            "numeric_error_examples": numeric_errors,
            "source_pixels": source_pixels.as_dict(),
            "target_pixels": target_pixels.as_dict(),
            "metric_distributions": metric_distributions,
            "alignment": {
                "analyzed_pairs": len(alignment_rows),
                "correlation": distribution(
                    [float(row["correlation"]) for row in alignment_rows]
                ),
                "shift_counts": dict(shift_counts.most_common()),
            },
            "samples": sample_rows,
        },
    }
    return report


def print_summary(report: dict[str, Any], output: Path) -> None:
    pairing = report["pairing"]
    structure = report["structure"]
    numeric = report["numeric_sampling"]
    metrics = numeric["metric_distributions"]
    source_pixels = numeric["source_pixels"]
    target_pixels = numeric["target_pixels"]

    print("\nSAR dataset analysis summary")
    print("=" * 32)
    print(
        f"files: echo={pairing['echo_mat_files']:,}, "
        f"image={pairing['image_mat_files']:,}, paired={pairing['paired_files']:,}"
    )
    print(
        f"unmatched: echo_only={pairing['echo_only_count']:,}, "
        f"image_only={pairing['image_only_count']:,}"
    )
    print(
        f"structure: readable={structure['readable_pairs']:,}, "
        f"errors={structure['structure_error_count']:,}, "
        f"shape_matches={structure['matching_matrix_shapes']:,}, "
        f"metadata_mismatches={structure['metadata_mismatch_pairs']:,}"
    )
    print(f"source shapes/dtypes: {structure['source_shapes']} / {structure['source_dtypes']}")
    print(f"target shapes/dtypes: {structure['target_shapes']} / {structure['target_dtypes']}")
    print(
        f"numeric: selected={numeric['selected_pairs']:,}, "
        f"analyzed={numeric['analyzed_pairs']:,}, errors={numeric['numeric_error_count']:,}"
    )
    if numeric["analyzed_pairs"]:
        ratio = metrics["rms_ratio_target_over_source"]
        source_rms = metrics["source_rms"]
        target_rms = metrics["target_rms"]
        print(
            "source pixels: "
            f"real_mean={source_pixels['real_mean']:.6g}, "
            f"real_std={source_pixels['real_std']:.6g}, "
            f"imag_mean={source_pixels['imag_mean']:.6g}, "
            f"imag_std={source_pixels['imag_std']:.6g}, "
            f"magnitude_rms={source_pixels['magnitude_rms']:.6g}"
        )
        print(
            "target pixels: "
            f"real_mean={target_pixels['real_mean']:.6g}, "
            f"real_std={target_pixels['real_std']:.6g}, "
            f"imag_mean={target_pixels['imag_mean']:.6g}, "
            f"imag_std={target_pixels['imag_std']:.6g}, "
            f"magnitude_rms={target_pixels['magnitude_rms']:.6g}"
        )
        print(
            "target/source RMS ratio: "
            f"median={format_number(ratio['median'])}, p05={format_number(ratio['p05'])}, "
            f"p95={format_number(ratio['p95'])}, min={format_number(ratio['min'])}, "
            f"max={format_number(ratio['max'])}"
        )
        print(
            "per-pair RMS: "
            f"source median={format_number(source_rms['median'])} "
            f"(p05={format_number(source_rms['p05'])}, "
            f"p95={format_number(source_rms['p95'])}); "
            f"target median={format_number(target_rms['median'])} "
            f"(p05={format_number(target_rms['p05'])}, "
            f"p95={format_number(target_rms['p95'])})"
        )
        print(
            "correlations: "
            f"complex p05/median/p95="
            f"{format_number(metrics['complex_correlation']['p05'])}/"
            f"{format_number(metrics['complex_correlation']['median'])}/"
            f"{format_number(metrics['complex_correlation']['p95'])}; "
            f"magnitude p05/median/p95="
            f"{format_number(metrics['magnitude_correlation']['p05'])}/"
            f"{format_number(metrics['magnitude_correlation']['median'])}/"
            f"{format_number(metrics['magnitude_correlation']['p95'])}"
        )
        print(
            "dynamic range (max/median) median: "
            f"source={format_number(metrics['source_max_over_median']['median'])}, "
            f"target={format_number(metrics['target_max_over_median']['median'])}"
        )
        print(
            f"nonfinite pixels: source={source_pixels['nonfinite_pixels']:,}, "
            f"target={target_pixels['nonfinite_pixels']:,}"
        )
        print(
            f"alignment checks={numeric['alignment']['analyzed_pairs']:,}, "
            f"most_common_shifts={list(numeric['alignment']['shift_counts'].items())[:10]}"
        )
    if pairing["echo_only_count"] or pairing["image_only_count"]:
        print(f"unmatched examples: echo={pairing['echo_only_examples'][:5]}")
        print(f"unmatched examples: image={pairing['image_only_examples'][:5]}")
    if structure["structure_error_count"]:
        print(f"structure error examples: {structure['structure_error_examples'][:5]}")
    if numeric["numeric_error_count"]:
        print(f"numeric error examples: {numeric['numeric_error_examples'][:5]}")
    print(f"JSON report: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check all SAR MAT pairs and numerically analyze a deterministic subset."
    )
    parser.add_argument("--echo-dir", type=Path, default=DEFAULT_ECHO_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=3000,
        help="number of pairs to read numerically; 0 reads every pair (default: 3000)",
    )
    parser.add_argument(
        "--alignment-count",
        type=int,
        default=200,
        help="number of sampled pairs used for FFT alignment checks (default: 200)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="print progress after this many structural checks; 0 disables progress",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_analysis_report.json"),
        help="JSON report path (default: ./dataset_analysis_report.json)",
    )
    args = parser.parse_args()
    if args.sample_count < 0:
        parser.error("--sample-count must be non-negative")
    if args.alignment_count < 0:
        parser.error("--alignment-count must be non-negative")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    report = analyze(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print_summary(report, args.output)


if __name__ == "__main__":
    main()
