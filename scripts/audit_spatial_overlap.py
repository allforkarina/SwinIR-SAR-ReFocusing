"""Strictly audit raw-pixel window overlap between two SAR patch grids.

The reference grid is normally the full data directory that supplied training,
validation, and guard patches.  The candidate grid is a proposed test set.
File names may use either ``patch_row_R_col_C.mat`` or the same name with a
numeric suffix such as ``patch_row_R_col_C_2.mat``.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from scripts.analyze_patch_adjacency import (
        MAX_EXAMPLES,
        CoordinateFile,
        index_coordinate_files,
    )
    from scripts.analyze_sar_dataset import DEFAULT_ECHO_DIR
except ModuleNotFoundError:  # Support ``python scripts/audit_spatial_overlap.py``.
    from analyze_patch_adjacency import (  # type: ignore[no-redef]
        MAX_EXAMPLES,
        CoordinateFile,
        index_coordinate_files,
    )
    from analyze_sar_dataset import DEFAULT_ECHO_DIR  # type: ignore[no-redef]


def windows_overlap(
    first: tuple[int, int], second: tuple[int, int], patch_size: int
) -> bool:
    """Return whether two half-open ``patch_size`` square windows intersect."""

    row_delta = abs(first[0] - second[0])
    col_delta = abs(first[1] - second[1])
    return row_delta < patch_size and col_delta < patch_size


def coordinate_bounds(index: dict[tuple[int, int], CoordinateFile]) -> dict[str, int | None]:
    if not index:
        return {"row_min": None, "row_max": None, "col_min": None, "col_max": None}
    rows = [row for row, _ in index]
    cols = [col for _, col in index]
    return {
        "row_min": min(rows),
        "row_max": max(rows),
        "col_min": min(cols),
        "col_max": max(cols),
    }


def spatial_buckets(
    index: dict[tuple[int, int], CoordinateFile], patch_size: int
) -> dict[tuple[int, int], list[CoordinateFile]]:
    buckets: dict[tuple[int, int], list[CoordinateFile]] = defaultdict(list)
    for item in index.values():
        buckets[(item.row // patch_size, item.col // patch_size)].append(item)
    return buckets


def find_window_overlaps(
    reference: dict[tuple[int, int], CoordinateFile],
    candidate: dict[tuple[int, int], CoordinateFile],
    patch_size: int,
) -> dict[str, Any]:
    """Find every overlap using local spatial buckets instead of an O(N*M) scan."""

    buckets = spatial_buckets(reference, patch_size)
    overlap_pair_count = 0
    candidate_coordinates_with_overlap: set[tuple[int, int]] = set()
    reference_coordinates_with_overlap: set[tuple[int, int]] = set()
    examples: list[dict[str, Any]] = []

    for candidate_item in candidate.values():
        bucket_row = candidate_item.row // patch_size
        bucket_col = candidate_item.col // patch_size
        for row_offset in (-1, 0, 1):
            for col_offset in (-1, 0, 1):
                for reference_item in buckets.get(
                    (bucket_row + row_offset, bucket_col + col_offset), []
                ):
                    reference_coordinate = (reference_item.row, reference_item.col)
                    candidate_coordinate = (candidate_item.row, candidate_item.col)
                    if not windows_overlap(reference_coordinate, candidate_coordinate, patch_size):
                        continue
                    overlap_pair_count += 1
                    candidate_coordinates_with_overlap.add(candidate_coordinate)
                    reference_coordinates_with_overlap.add(reference_coordinate)
                    if len(examples) < MAX_EXAMPLES:
                        examples.append(
                            {
                                "reference_file": reference_item.path.name,
                                "candidate_file": candidate_item.path.name,
                                "reference_coordinate": list(reference_coordinate),
                                "candidate_coordinate": list(candidate_coordinate),
                                "row_start_delta": candidate_item.row - reference_item.row,
                                "col_start_delta": candidate_item.col - reference_item.col,
                            }
                        )

    exact_coordinates = sorted(reference.keys() & candidate.keys())
    return {
        "exact_coordinate_count": len(exact_coordinates),
        "exact_coordinate_examples": [list(value) for value in exact_coordinates[:MAX_EXAMPLES]],
        "window_overlap_pair_count": overlap_pair_count,
        "candidate_patch_count_with_overlap": len(candidate_coordinates_with_overlap),
        "reference_patch_count_with_overlap": len(reference_coordinates_with_overlap),
        "window_overlap_examples": examples,
    }


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    reference, reference_files = index_coordinate_files(args.reference_echo_dir)
    candidate, candidate_files = index_coordinate_files(args.candidate_echo_dir)
    overlap = find_window_overlaps(reference, candidate, args.patch_size)
    parsing_complete = all(
        summary["unparsed_count"] == 0 and summary["coordinate_collision_count"] == 0
        for summary in (reference_files, candidate_files)
    )
    passed = parsing_complete and overlap["window_overlap_pair_count"] == 0
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "reference_echo_dir": str(args.reference_echo_dir.resolve()),
            "candidate_echo_dir": str(args.candidate_echo_dir.resolve()),
            "patch_size": args.patch_size,
        },
        "reference": {**reference_files, "coordinate_bounds": coordinate_bounds(reference)},
        "candidate": {**candidate_files, "coordinate_bounds": coordinate_bounds(candidate)},
        "overlap": overlap,
        "audit_passed": passed,
        "failure_reason": (
            None
            if passed
            else "unparsed/colliding filenames or at least one raw-pixel window overlap"
        ),
    }


def print_summary(report: dict[str, Any], output: Path) -> None:
    print("\nSAR spatial overlap audit")
    print("=" * 26)
    for role in ("reference", "candidate"):
        details = report[role]
        print(
            f"{role}: files={details['mat_file_count']:,}, "
            f"parsed={details['parsed_unique_coordinate_count']:,}, "
            f"unparsed={details['unparsed_count']:,}, "
            f"collisions={details['coordinate_collision_count']:,}, "
            f"bounds={details['coordinate_bounds']}"
        )
    overlap = report["overlap"]
    print(
        f"cross-grid: exact_coordinates={overlap['exact_coordinate_count']:,}, "
        f"window_overlap_pairs={overlap['window_overlap_pair_count']:,}, "
        f"candidate_patches_with_overlap={overlap['candidate_patch_count_with_overlap']:,}"
    )
    print(f"audit_passed={report['audit_passed']}")
    print(f"JSON report: {output.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly reject raw-pixel overlap between reference and candidate SAR grids."
    )
    parser.add_argument(
        "--reference-echo-dir",
        type=Path,
        default=DEFAULT_ECHO_DIR,
        help="full original echo directory; default is the training-data echo directory",
    )
    parser.add_argument(
        "--candidate-echo-dir",
        type=Path,
        required=True,
        help="proposed independent-test echo directory",
    )
    parser.add_argument("--patch-size", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("spatial_overlap_report.json"),
    )
    args = parser.parse_args()
    if args.patch_size <= 0:
        parser.error("--patch-size must be positive")
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
    if not report["audit_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
