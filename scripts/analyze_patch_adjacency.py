"""Audit whether filename coordinate steps correspond to overlapping patch pixels.

The audit is read-only. It scans every filename to describe the coordinate grid,
then loads a deterministic subset of immediate row/column neighbours. For each
numeric pair it reports two independent observations:

1. overlap metrics when the filename coordinate delta is treated as a pixel
   shift; and
2. the strongest circular magnitude-correlation shift estimated by FFT.

The report deliberately exposes measurements instead of declaring that the
coordinates are pixel indices. That conclusion should be made only after the
server report has been inspected.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.analyze_sar_dataset import (
        DEFAULT_ECHO_DIR,
        DEFAULT_IMAGE_DIR,
        best_circular_shift,
        correlation_metrics,
        distribution,
        evenly_spaced_indices,
        inspect_patch_file,
    )
except ModuleNotFoundError:  # Support ``python scripts/analyze_patch_adjacency.py``.
    from analyze_sar_dataset import (  # type: ignore[no-redef]
        DEFAULT_ECHO_DIR,
        DEFAULT_IMAGE_DIR,
        best_circular_shift,
        correlation_metrics,
        distribution,
        evenly_spaced_indices,
        inspect_patch_file,
    )


COORDINATE_PATTERN = re.compile(
    r"^patch_row_([+-]?\d+)_col_([+-]?\d+)\.mat$", re.IGNORECASE
)
MAX_EXAMPLES = 20


@dataclass(frozen=True)
class CoordinateFile:
    row: int
    col: int
    path: Path


@dataclass(frozen=True)
class NeighborEdge:
    axis: str
    first: CoordinateFile
    second: CoordinateFile

    @property
    def coordinate_delta(self) -> int:
        if self.axis == "row":
            return self.second.row - self.first.row
        return self.second.col - self.first.col


def parse_coordinate(path: Path) -> tuple[int, int] | None:
    match = COORDINATE_PATTERN.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def index_coordinate_files(
    directory: Path,
) -> tuple[dict[tuple[int, int], CoordinateFile], dict[str, Any]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"not a directory: {directory}")

    mat_files = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".mat"
    )
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    unparsed: list[str] = []
    for path in mat_files:
        coordinate = parse_coordinate(path)
        if coordinate is None:
            unparsed.append(path.name)
        else:
            grouped[coordinate].append(path)

    collisions = {
        str(coordinate): [path.name for path in paths]
        for coordinate, paths in grouped.items()
        if len(paths) > 1
    }
    index = {
        coordinate: CoordinateFile(coordinate[0], coordinate[1], paths[0])
        for coordinate, paths in grouped.items()
        if len(paths) == 1
    }
    return index, {
        "mat_file_count": len(mat_files),
        "parsed_unique_coordinate_count": len(index),
        "unparsed_count": len(unparsed),
        "unparsed_examples": unparsed[:MAX_EXAMPLES],
        "coordinate_collision_count": len(collisions),
        "coordinate_collision_examples": dict(list(collisions.items())[:MAX_EXAMPLES]),
    }


def positive_step_counts(values: list[int]) -> dict[str, int]:
    return {
        str(step): count
        for step, count in Counter(
            second - first for first, second in zip(values, values[1:])
        ).most_common()
    }


def grid_summary(
    index: dict[tuple[int, int], CoordinateFile], file_summary: dict[str, Any]
) -> dict[str, Any]:
    rows = sorted({row for row, _ in index})
    cols = sorted({col for _, col in index})
    expected_cells = len(rows) * len(cols)
    missing = [
        [row, col]
        for row in rows
        for col in cols
        if (row, col) not in index
    ]
    return {
        **file_summary,
        "unique_row_count": len(rows),
        "unique_col_count": len(cols),
        "row_min": rows[0] if rows else None,
        "row_max": rows[-1] if rows else None,
        "col_min": cols[0] if cols else None,
        "col_max": cols[-1] if cols else None,
        "row_step_counts": positive_step_counts(rows),
        "col_step_counts": positive_step_counts(cols),
        "rectangular_grid_cell_count": expected_cells,
        "missing_grid_cell_count": len(missing),
        "missing_grid_cell_examples": missing[:MAX_EXAMPLES],
        "is_complete_rectangular_grid": bool(index) and not missing,
    }


def neighbor_edges(
    index: dict[tuple[int, int], CoordinateFile], axis: str
) -> list[NeighborEdge]:
    if axis not in {"row", "col"}:
        raise ValueError(f"unsupported axis: {axis!r}")

    groups: dict[int, list[CoordinateFile]] = defaultdict(list)
    for item in index.values():
        fixed_coordinate = item.col if axis == "row" else item.row
        groups[fixed_coordinate].append(item)

    result: list[NeighborEdge] = []
    for fixed_coordinate in sorted(groups):
        items = sorted(
            groups[fixed_coordinate],
            key=(lambda item: item.row) if axis == "row" else (lambda item: item.col),
        )
        result.extend(
            NeighborEdge(axis=axis, first=first, second=second)
            for first, second in zip(items, items[1:])
        )
    return result


def select_evenly(items: list[NeighborEdge], requested: int) -> list[NeighborEdge]:
    return [items[index] for index in evenly_spaced_indices(len(items), requested)]


def overlap_regions(
    first: np.ndarray, second: np.ndarray, axis: str, pixel_shift: int
) -> tuple[np.ndarray, np.ndarray]:
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError(f"expected 2-D patches, got {first.shape} and {second.shape}")
    if first.shape != second.shape:
        raise ValueError(f"patch shapes differ: {first.shape} and {second.shape}")
    array_axis = 0 if axis == "row" else 1
    axis_size = first.shape[array_axis]
    if not 0 < pixel_shift < axis_size:
        raise ValueError(
            f"pixel shift {pixel_shift} is outside valid range 1..{axis_size - 1}"
        )

    if axis == "row":
        return first[pixel_shift:, :], second[:-pixel_shift, :]
    return first[:, pixel_shift:], second[:, :-pixel_shift]


def overlap_metrics(first: np.ndarray, second: np.ndarray) -> dict[str, float | int]:
    first_complex = first.astype(np.complex128, copy=False)
    second_complex = second.astype(np.complex128, copy=False)
    difference = first_complex - second_complex
    first_rms = math.sqrt(float(np.mean(np.abs(first_complex) ** 2)))
    second_rms = math.sqrt(float(np.mean(np.abs(second_complex) ** 2)))
    reference_rms = math.sqrt((first_rms * first_rms + second_rms * second_rms) / 2.0)
    rmse = math.sqrt(float(np.mean(np.abs(difference) ** 2)))

    first_flat = first_complex.ravel()
    second_flat = second_complex.ravel()
    gain_denominator = float(np.vdot(first_flat, first_flat).real)
    gain = (
        np.vdot(first_flat, second_flat) / gain_denominator
        if gain_denominator > 0
        else 0.0j
    )
    gain_residual = gain * first_complex - second_complex
    gain_rmse = math.sqrt(float(np.mean(np.abs(gain_residual) ** 2)))
    correlations = correlation_metrics(first_complex, second_complex)
    return {
        "element_count": int(first_complex.size),
        "exact_equal_fraction": float(np.mean(first_complex == second_complex)),
        "close_fraction_rtol_1e-5_atol_1e-8": float(
            np.mean(np.isclose(first_complex, second_complex, rtol=1e-5, atol=1e-8))
        ),
        "rmse": rmse,
        "relative_rmse": rmse / reference_rms if reference_rms > 0 else 0.0,
        "complex_correlation": correlations["complex_correlation"],
        "magnitude_correlation": correlations["magnitude_correlation"],
        "best_fit_gain_real": float(gain.real),
        "best_fit_gain_imag": float(gain.imag),
        "best_fit_gain_relative_rmse": (
            gain_rmse / second_rms if second_rms > 0 else 0.0
        ),
    }


def analyze_edge(
    edge: NeighborEdge, role: str, shift_tolerance: int
) -> dict[str, Any]:
    first_info = inspect_patch_file(edge.first.path, role=role, load_values=True)
    second_info = inspect_patch_file(edge.second.path, role=role, load_values=True)
    first = first_info.values
    second = second_info.values
    if first is None or second is None:
        raise RuntimeError("numeric inspection did not load both patches")
    if first.shape != second.shape:
        raise ValueError(f"patch shapes differ: {first.shape} and {second.shape}")

    coordinate_delta = edge.coordinate_delta
    array_axis = 0 if edge.axis == "row" else 1
    orthogonal_axis = 1 - array_axis
    alignment = best_circular_shift(first, second)
    estimated_shift = alignment["shift"]
    row: dict[str, Any] = {
        "axis": edge.axis,
        "first_file": edge.first.path.name,
        "second_file": edge.second.path.name,
        "first_coordinate": [edge.first.row, edge.first.col],
        "second_coordinate": [edge.second.row, edge.second.col],
        "coordinate_delta": coordinate_delta,
        "shape": list(first.shape),
        "mat_formats": [first_info.mat_format, second_info.mat_format],
        "variables": [first_info.variable_name, second_info.variable_name],
        "fft_magnitude_correlation": float(alignment["correlation"]),
        "estimated_circular_shift": estimated_shift,
        "estimated_axis_shift": int(estimated_shift[array_axis]),
        "estimated_orthogonal_shift": int(estimated_shift[orthogonal_axis]),
        "axis_shift_minus_coordinate_delta": int(
            estimated_shift[array_axis] - coordinate_delta
        ),
        "coordinate_shift_match": bool(
            abs(estimated_shift[array_axis] - coordinate_delta) <= shift_tolerance
            and abs(estimated_shift[orthogonal_axis]) <= shift_tolerance
        ),
    }
    if 0 < coordinate_delta < first.shape[array_axis]:
        first_overlap, second_overlap = overlap_regions(
            first, second, edge.axis, coordinate_delta
        )
        row["coordinate_delta_as_pixel_shift"] = {
            "valid": True,
            **overlap_metrics(first_overlap, second_overlap),
        }
    else:
        row["coordinate_delta_as_pixel_shift"] = {
            "valid": False,
            "reason": (
                f"coordinate delta {coordinate_delta} is not within the "
                f"array-axis length {first.shape[array_axis]}"
            ),
        }
    return row


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid_overlap_rows = [
        row
        for row in rows
        if row["coordinate_delta_as_pixel_shift"]["valid"]
    ]
    overlap_metric_names = [
        "exact_equal_fraction",
        "close_fraction_rtol_1e-5_atol_1e-8",
        "relative_rmse",
        "complex_correlation",
        "magnitude_correlation",
        "best_fit_gain_relative_rmse",
    ]
    return {
        "analyzed_edge_count": len(rows),
        "coordinate_delta_distribution": distribution(
            [row["coordinate_delta"] for row in rows]
        ),
        "valid_coordinate_as_pixel_shift_count": len(valid_overlap_rows),
        "coordinate_shift_match_fraction": (
            float(np.mean([row["coordinate_shift_match"] for row in rows]))
            if rows
            else None
        ),
        "fft_magnitude_correlation": distribution(
            [row["fft_magnitude_correlation"] for row in rows]
        ),
        "estimated_axis_shift": distribution(
            [row["estimated_axis_shift"] for row in rows]
        ),
        "estimated_orthogonal_shift": distribution(
            [row["estimated_orthogonal_shift"] for row in rows]
        ),
        "axis_shift_minus_coordinate_delta": distribution(
            [row["axis_shift_minus_coordinate_delta"] for row in rows]
        ),
        "most_common_estimated_circular_shifts": dict(
            Counter(str(tuple(row["estimated_circular_shift"])) for row in rows).most_common(
                10
            )
        ),
        "coordinate_delta_overlap_metrics": {
            metric: distribution(
                [row["coordinate_delta_as_pixel_shift"][metric] for row in valid_overlap_rows]
            )
            for metric in overlap_metric_names
        },
    }


def analyze_axis(
    index: dict[tuple[int, int], CoordinateFile],
    axis: str,
    role: str,
    pairs_per_axis: int,
    shift_tolerance: int,
    progress_every: int,
) -> dict[str, Any]:
    edges = neighbor_edges(index, axis)
    selected = select_evenly(edges, pairs_per_axis)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for position, edge in enumerate(selected, start=1):
        try:
            rows.append(analyze_edge(edge, role=role, shift_tolerance=shift_tolerance))
        except Exception as error:  # Continue the audit and retain concrete failures.
            if len(errors) < MAX_EXAMPLES:
                errors.append(
                    {
                        "first_file": edge.first.path.name,
                        "second_file": edge.second.path.name,
                        "error": repr(error),
                    }
                )
        if progress_every and (position % progress_every == 0 or position == len(selected)):
            print(
                f"{role}/{axis}: checked {position:,}/{len(selected):,} adjacent pairs; "
                f"analyzed={len(rows):,}, errors={position - len(rows):,}",
                flush=True,
            )

    return {
        "available_edge_count": len(edges),
        "selected_edge_count": len(selected),
        "numeric_error_count": len(selected) - len(rows),
        "numeric_error_examples": errors,
        "summary": summarize_rows(rows),
        "samples": rows,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    echo_index, echo_files = index_coordinate_files(args.echo_dir)
    image_index, image_files = index_coordinate_files(args.image_dir)
    common_coordinates = sorted(echo_index.keys() & image_index.keys())
    echo_only = sorted(echo_index.keys() - image_index.keys())
    image_only = sorted(image_index.keys() - echo_index.keys())

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "pairs_per_axis_requested": args.pairs_per_axis,
            "sampling_strategy": "evenly_spaced_after_sorting_adjacent_coordinate_edges",
            "shift_tolerance": args.shift_tolerance,
        },
        "coordinate_pairing": {
            "common_coordinate_count": len(common_coordinates),
            "echo_only_coordinate_count": len(echo_only),
            "image_only_coordinate_count": len(image_only),
            "echo_only_coordinate_examples": [list(value) for value in echo_only[:MAX_EXAMPLES]],
            "image_only_coordinate_examples": [list(value) for value in image_only[:MAX_EXAMPLES]],
            "coordinate_sets_equal": not echo_only and not image_only,
        },
        "grid": {
            "echo": grid_summary(echo_index, echo_files),
            "image": grid_summary(image_index, image_files),
        },
        "adjacency": {},
    }

    for report_name, index, role in (
        ("echo", echo_index, "source"),
        ("image", image_index, "target"),
    ):
        report["adjacency"][report_name] = {}
        for axis in ("row", "col"):
            report["adjacency"][report_name][axis] = analyze_axis(
                index=index,
                axis=axis,
                role=role,
                pairs_per_axis=args.pairs_per_axis,
                shift_tolerance=args.shift_tolerance,
                progress_every=args.progress_every,
            )
    return report


def format_number(value: float | int | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def print_summary(report: dict[str, Any], output: Path) -> None:
    pairing = report["coordinate_pairing"]
    print("\nSAR patch adjacency audit")
    print("=" * 29)
    print(
        f"coordinate pairs: common={pairing['common_coordinate_count']:,}, "
        f"echo_only={pairing['echo_only_coordinate_count']:,}, "
        f"image_only={pairing['image_only_coordinate_count']:,}"
    )
    for role in ("echo", "image"):
        grid = report["grid"][role]
        print(
            f"{role} grid: files={grid['mat_file_count']:,}, "
            f"parsed={grid['parsed_unique_coordinate_count']:,}, "
            f"rows={grid['unique_row_count']:,}, cols={grid['unique_col_count']:,}, "
            f"complete={grid['is_complete_rectangular_grid']}"
        )
        print(
            f"{role} coordinate steps: rows={grid['row_step_counts']}, "
            f"cols={grid['col_step_counts']}"
        )
        for axis in ("row", "col"):
            result = report["adjacency"][role][axis]
            summary = result["summary"]
            overlap = summary["coordinate_delta_overlap_metrics"]
            print(
                f"{role}/{axis}: available={result['available_edge_count']:,}, "
                f"analyzed={summary['analyzed_edge_count']:,}, "
                f"errors={result['numeric_error_count']:,}, "
                f"shift_match={format_number(summary['coordinate_shift_match_fraction'])}"
            )
            print(
                "  FFT corr median="
                f"{format_number(summary['fft_magnitude_correlation'].get('median'))}; "
                "coordinate-overlap exact/close/correlation median="
                f"{format_number(overlap['exact_equal_fraction'].get('median'))}/"
                f"{format_number(overlap['close_fraction_rtol_1e-5_atol_1e-8'].get('median'))}/"
                f"{format_number(overlap['complex_correlation'].get('median'))}; "
                "relative RMSE median="
                f"{format_number(overlap['relative_rmse'].get('median'))}"
            )
    print("Interpretation is intentionally deferred until these measurements are reviewed.")
    print(f"JSON report: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the SAR coordinate grid and audit whether adjacent filename "
            "steps behave like pixel shifts."
        )
    )
    parser.add_argument("--echo-dir", type=Path, default=DEFAULT_ECHO_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--pairs-per-axis",
        type=int,
        default=100,
        help="adjacent pairs loaded per role and axis; 0 loads every edge (default: 100)",
    )
    parser.add_argument(
        "--shift-tolerance",
        type=int,
        default=2,
        help="pixel tolerance used only for the reported FFT/coordinate match fraction",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="print progress after this many numeric checks; 0 disables progress",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("patch_adjacency_report.json"),
        help="JSON report path (default: ./patch_adjacency_report.json)",
    )
    args = parser.parse_args()
    if args.pairs_per_axis < 0:
        parser.error("--pairs-per-axis must be non-negative")
    if args.shift_tolerance < 0:
        parser.error("--shift-tolerance must be non-negative")
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
