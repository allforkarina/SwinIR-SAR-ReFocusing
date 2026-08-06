from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.audit_spatial_overlap import analyze, find_window_overlaps, windows_overlap
from scripts.analyze_patch_adjacency import index_coordinate_files


def touch_patch(directory: Path, row: int, col: int, suffix: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"patch_row_{row}_col_{col}{suffix}.mat").touch()


def test_windows_overlap_uses_half_open_patch_bounds() -> None:
    assert windows_overlap((0, 0), (511, 511), patch_size=512) is True
    assert windows_overlap((0, 0), (512, 0), patch_size=512) is False
    assert windows_overlap((0, 0), (0, 512), patch_size=512) is False


def test_overlap_audit_accepts_optional_numeric_filename_suffix(tmp_path: Path) -> None:
    reference_dir = tmp_path / "echo"
    candidate_dir = tmp_path / "echo_3"
    touch_patch(reference_dir, 0, 0)
    touch_patch(reference_dir, 1000, 1000)
    touch_patch(candidate_dir, 1512, 1000, suffix="_2")

    report = analyze(
        Namespace(
            reference_echo_dir=reference_dir,
            candidate_echo_dir=candidate_dir,
            patch_size=512,
        )
    )

    assert report["audit_passed"] is True
    assert report["overlap"]["window_overlap_pair_count"] == 0
    assert report["candidate"]["unparsed_count"] == 0


def test_overlap_audit_reports_partial_and_exact_overlap(tmp_path: Path) -> None:
    reference_dir = tmp_path / "echo"
    candidate_dir = tmp_path / "echo_3"
    touch_patch(reference_dir, 1000, 1000)
    touch_patch(candidate_dir, 1100, 1100, suffix="_2")
    touch_patch(candidate_dir, 1000, 1000, suffix="_2")

    reference, _ = index_coordinate_files(reference_dir)
    candidate, _ = index_coordinate_files(candidate_dir)
    overlap = find_window_overlaps(reference, candidate, patch_size=512)

    assert overlap["exact_coordinate_count"] == 1
    assert overlap["window_overlap_pair_count"] == 2
