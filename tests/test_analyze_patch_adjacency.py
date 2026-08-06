from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
from scipy.io import savemat

from scripts.analyze_patch_adjacency import (
    analyze,
    grid_summary,
    index_coordinate_files,
    neighbor_edges,
    overlap_metrics,
    overlap_regions,
    parse_coordinate,
)


def test_parse_coordinate_accepts_signed_server_coordinates() -> None:
    assert parse_coordinate(Path("patch_row_-100_col_+2300.mat")) == (-100, 2300)
    assert parse_coordinate(Path("patch_row_-100_col_+2300_2.mat")) == (-100, 2300)
    assert parse_coordinate(Path("other.mat")) is None


def test_grid_and_edges_are_derived_from_coordinates(tmp_path: Path) -> None:
    directory = tmp_path / "echo"
    directory.mkdir()
    for row in (1, 101):
        for col in (2300, 2400, 2500):
            (directory / f"patch_row_{row}_col_{col}.mat").touch()

    index, files = index_coordinate_files(directory)
    summary = grid_summary(index, files)

    assert summary["is_complete_rectangular_grid"] is True
    assert summary["row_step_counts"] == {"100": 1}
    assert summary["col_step_counts"] == {"100": 2}
    assert len(neighbor_edges(index, "row")) == 3
    assert len(neighbor_edges(index, "col")) == 4


def test_overlap_metrics_detect_exact_coordinate_shift() -> None:
    rng = np.random.default_rng(0)
    full = rng.normal(size=(8, 10)) + 1j * rng.normal(size=(8, 10))
    left = full[:, :8]
    right = full[:, 2:10]

    left_overlap, right_overlap = overlap_regions(left, right, "col", 2)
    metrics = overlap_metrics(left_overlap, right_overlap)

    assert metrics["exact_equal_fraction"] == 1.0
    assert metrics["close_fraction_rtol_1e-5_atol_1e-8"] == 1.0
    assert metrics["relative_rmse"] == 0.0
    assert metrics["complex_correlation"] == 1.0


def write_grid(root: Path, patch_size: int = 16, step: int = 4) -> None:
    rng = np.random.default_rng(7)
    full = rng.normal(size=(patch_size + step, patch_size + step)) + 1j * rng.normal(
        size=(patch_size + step, patch_size + step)
    )
    for role in ("echo", "image"):
        directory = root / role
        directory.mkdir()
        for row_index, row in enumerate((1, 1 + step)):
            for col_index, col in enumerate((2300, 2300 + step)):
                patch = full[
                    row_index * step : row_index * step + patch_size,
                    col_index * step : col_index * step + patch_size,
                ]
                savemat(directory / f"patch_row_{row}_col_{col}.mat", {"patch": patch})


def test_analyze_reports_exact_overlap_and_matching_fft_shift(tmp_path: Path) -> None:
    write_grid(tmp_path)

    report = analyze(
        Namespace(
            echo_dir=tmp_path / "echo",
            image_dir=tmp_path / "image",
            pairs_per_axis=0,
            shift_tolerance=0,
            progress_every=0,
        )
    )

    assert report["coordinate_pairing"]["coordinate_sets_equal"] is True
    assert report["grid"]["echo"]["is_complete_rectangular_grid"] is True
    for role in ("echo", "image"):
        for axis in ("row", "col"):
            summary = report["adjacency"][role][axis]["summary"]
            overlap = summary["coordinate_delta_overlap_metrics"]
            assert summary["analyzed_edge_count"] == 2
            assert summary["coordinate_shift_match_fraction"] == 1.0
            assert overlap["exact_equal_fraction"]["median"] == 1.0
            assert overlap["relative_rmse"]["median"] == 0.0
