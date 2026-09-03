from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from scripts.analyze_phase_dataset import (
    audit,
    normalized_phase_oracle,
    validate_source_output_separation,
)


def write_paired_grid(root: Path, patch_size: int = 16, step: int = 4) -> None:
    rng = np.random.default_rng(11)
    full = rng.normal(size=(patch_size + step, patch_size + step)) + 1j * rng.normal(
        size=(patch_size + step, patch_size + step)
    )
    echo_dir = root / "echo"
    image_dir = root / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    phase = np.exp(1j * 0.7)
    for row_index, row in enumerate((100, 100 + step)):
        for col_index, col in enumerate((200, 200 + step)):
            image = full[
                row_index * step : row_index * step + patch_size,
                col_index * step : col_index * step + patch_size,
            ]
            echo = image * phase
            filename = f"patch_row_{row}_col_{col}.mat"
            savemat(echo_dir / filename, {"patch": echo})
            savemat(image_dir / filename, {"patch": image})


def audit_args(root: Path) -> Namespace:
    return Namespace(
        echo_dir=root / "echo",
        image_dir=root / "image",
        output_dir=root / "audit",
        numeric_sample_count=4,
        alignment_sample_count=2,
        metadata_sample_count=2,
        phase_sample_count=4,
        phase_figure_count=2,
        adjacent_pairs_per_axis=0,
        group_overlap_pair_count=0,
        shift_tolerance=0,
        pca_grid_size=4,
        fft_norm="ortho",
        phasor_epsilon=1.0e-6,
        floor_db=-60.0,
        high_frequency_radius_fraction=0.25,
        max_preview_pixels=10_000,
        max_complex_mosaic_pixels=10_000,
        progress_every=0,
    )


def test_phase_oracle_closes_constant_phase_error() -> None:
    rng = np.random.default_rng(4)
    image = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    echo = image * np.exp(1j * 1.2)

    _, normalized_image, oracle, correction, weights = normalized_phase_oracle(
        echo,
        image,
        fft_norm="ortho",
        phasor_epsilon=1.0e-9,
    )

    np.testing.assert_allclose(oracle, normalized_image, atol=1.0e-10)
    assert float(weights.sum()) > 0
    assert np.allclose(np.abs(correction), 1.0)


def test_output_directory_cannot_be_inside_read_only_source(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()

    with pytest.raises(ValueError, match="must not be inside"):
        validate_source_output_separation(
            echo_dir, image_dir, echo_dir / "audit"
        )


def test_end_to_end_audit_is_read_only_and_writes_expected_artifacts(
    tmp_path: Path,
) -> None:
    write_paired_grid(tmp_path)
    source_bytes = {
        path: path.read_bytes()
        for directory in (tmp_path / "echo", tmp_path / "image")
        for path in directory.glob("*.mat")
    }

    report = audit(audit_args(tmp_path))

    assert report["read_only_contract"]["source_mutation"] == "forbidden"
    assert report["read_only_contract"]["dataset_splitting"].startswith(
        "not_implemented"
    )
    assert report["pairing"]["paired_files"] == 4
    assert report["phase_analysis"]["analyzed_sample_count"] == 4
    median_closed = report["phase_analysis"]["aggregate_metrics"][
        "rmse_gap_fraction_closed"
    ]["median"]
    assert median_closed > 0.99
    assert report["decision_gates"]["mosaic_preview_count"] == 1
    parent = next(iter(report["stitchability"].values()))
    assert parent["roles"]["echo"]["relative_rmse"]["median"] == pytest.approx(0.0)
    assert parent["roles"]["echo"]["complex_stitching_valid"] is True

    output = tmp_path / "audit"
    for relative in (
        "summary.json",
        "dataset_structure.json",
        "patch_adjacency.json",
        "file_inventory.csv",
        "pair_manifest.csv",
        "metadata_schema.json",
        "grouping/parent_image_candidates.json",
        "coordinates/adjacent_patch_metrics.csv",
        "coordinates/overlap_components.json",
        "phase_analysis/per_sample_phase_metrics.csv",
        "phase_analysis/oracle_recoverability.json",
        "phase_analysis/figures/metric_distributions.png",
        "mosaics/coordinate_grid_0/echo_log_magnitude.png",
        "mosaics/coordinate_grid_0/image_log_magnitude.png",
        "mosaics/coordinate_grid_0/oracle_log_magnitude.png",
    ):
        assert (output / relative).is_file(), relative
    assert not (output / "splits").exists()
    for path, expected in source_bytes.items():
        assert path.read_bytes() == expected
