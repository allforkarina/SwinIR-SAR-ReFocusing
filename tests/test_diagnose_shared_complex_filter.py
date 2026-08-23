from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat

from scripts.diagnose_shared_complex_filter import (
    apply_shared_filter,
    evaluate_focus_prediction,
    fit_shared_filter,
    run,
    select_spatially_separated_fit_records,
)
from swinir.sar_dataset import (
    CoordinateRegion,
    PairRecord,
    SplitName,
    classify_coordinate,
)


def _record(tmp_path: Path, row: int, col: int) -> PairRecord:
    name = f"patch_row_{row}_col_{col}_2.mat"
    return PairRecord(
        key=f"row_{row}_col_{col}",
        row=row,
        col=col,
        split=SplitName.TRAIN,
        echo_path=tmp_path / "echo" / name,
        image_path=tmp_path / "image" / name,
    )


def _known_transfer(shape: tuple[int, int]) -> np.ndarray:
    row_frequency = np.fft.fftfreq(shape[0])[:, None]
    col_frequency = np.fft.fftfreq(shape[1])[None, :]
    phase = 0.35 * row_frequency - 0.2 * col_frequency
    magnitude = 1.1 + 0.1 * np.cos(2.0 * np.pi * row_frequency)
    return magnitude * np.exp(1j * phase)


def _write_pair(record: PairRecord, echo: np.ndarray, transfer: np.ndarray) -> None:
    image = np.fft.ifft2(transfer * np.fft.fft2(echo, norm="ortho"), norm="ortho")
    savemat(record.echo_path, {"patch": echo})
    savemat(record.image_path, {"patch": image})


def test_spatial_fit_selection_respects_coordinate_spacing(tmp_path: Path) -> None:
    records = tuple(
        _record(tmp_path, row, col)
        for row in range(0, 501, 100)
        for col in range(0, 501, 100)
    )

    selected = select_spatially_separated_fit_records(
        records, minimum_spacing=250, maximum_samples=16
    )

    selected_rows = sorted({record.row for record in selected})
    selected_cols = sorted({record.col for record in selected})
    assert all(second - first >= 250 for first, second in zip(selected_rows, selected_rows[1:]))
    assert all(second - first >= 250 for first, second in zip(selected_cols, selected_cols[1:]))
    assert len(selected) <= 16


def test_scene4_fit_contract_selects_256_nonoverlapping_training_patches(
    tmp_path: Path,
) -> None:
    validation = CoordinateRegion(16400, 18400, 7700, 9700)
    guard = CoordinateRegion(15888, 18912, 7188, 10212)
    records = tuple(
        _record(tmp_path, row, col)
        for row in range(10000, 24401, 100)
        for col in range(3000, 14401, 100)
        if classify_coordinate(row, col, validation, guard) is SplitName.TRAIN
    )

    selected = select_spatially_separated_fit_records(
        records, minimum_spacing=512, maximum_samples=256
    )

    assert len(selected) == 256
    for index, first in enumerate(selected):
        for second in selected[index + 1 :]:
            assert abs(first.row - second.row) >= 512 or abs(first.col - second.col) >= 512


def test_shared_filter_recovers_known_complex_frequency_mapping(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    shape = (16, 16)
    transfer = _known_transfer(shape)
    records = []
    for index in range(12):
        record = _record(tmp_path, index * 20, 0)
        rng = np.random.default_rng(index)
        echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
        _write_pair(record, echo, transfer)
        records.append(record)

    fitted = fit_shared_filter(
        records,
        expected_shape=shape,
        rms_epsilon=1.0e-12,
        fft_norm="ortho",
        ridge_fraction=1.0e-9,
    )
    rng = np.random.default_rng(999)
    echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    expected = np.fft.ifft2(transfer * np.fft.fft2(echo, norm="ortho"), norm="ortho")
    prediction = apply_shared_filter(echo, fitted.transfer, fft_norm="ortho")

    relative_error = np.linalg.norm(prediction - expected) / np.linalg.norm(expected)
    assert relative_error < 1.0e-8
    assert fitted.cross_spectral_coherence.mean() > 0.999999


def test_focus_metrics_identical_prediction_has_unit_structure_ratios() -> None:
    rng = np.random.default_rng(42)
    target = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))

    metrics = evaluate_focus_prediction(
        target,
        target,
        floor_db=-60.0,
        high_frequency_radius_fraction=0.25,
    )

    assert metrics["complex_coherence"] == 1.0
    assert metrics["edge_correlation"] > 0.999999
    assert metrics["gradient_energy_ratio"] == 1.0
    assert metrics["high_frequency_energy_ratio"] == 1.0


def test_tiny_end_to_end_run_writes_report_filter_and_audit(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    shape = (16, 16)
    transfer = _known_transfer(shape)
    for row in range(0, 81, 20):
        for col in range(0, 81, 20):
            record = _record(tmp_path, row, col)
            rng = np.random.default_rng(row * 100 + col + 1)
            echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            _write_pair(record, echo, transfer)

    config = {
        "experiment": "test-shared-complex-filter",
        "data": {
            "expected_shape": [16, 16],
            "rms_epsilon": 1.0e-12,
            "validation_region": {
                "row_min": 40,
                "row_max": 40,
                "col_min": 40,
                "col_max": 40,
            },
            "guard_region": {
                "row_min": 20,
                "row_max": 60,
                "col_min": 20,
                "col_max": 60,
            },
            "expected_split_counts": {"train": 16, "guard": 8, "validation": 1},
            "fit_min_coordinate_spacing": 20,
            "fit_max_samples": 8,
        },
        "filter": {
            "fft_norm": "ortho",
            "ridge_fraction_of_mean_power": 1.0e-9,
        },
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "high_frequency_radius_fraction": 0.25,
            "audit_sample_count": 1,
            "success_criteria": {
                "validation_rmse_win_fraction_vs_echo_min": 0.0,
                "validation_rmse_win_fraction_vs_gain_min": 0.0,
                "mean_complex_coherence_delta_vs_echo_min": -1.0,
                "mean_log_ssim_delta_vs_echo_min": -1.0,
                "mean_edge_correlation_delta_vs_echo_min": -1.0,
                "mean_edge_correlation_delta_vs_gain_min": -1.0,
                "median_high_frequency_energy_ratio_min": 0.0,
                "median_high_frequency_energy_ratio_max": 2.0,
                "fit_weighted_transfer_coherence_min": 0.0,
            },
        },
        "runtime": {
            "progress_interval_samples": 4,
            "figure_dpi": 40,
            "contact_sheet_page_size": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "output"
    args = argparse.Namespace(
        config=config_path,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
    )

    report = run(args)

    assert report["status"] == "shared_filter_metric_supported"
    assert report["validation"]["sample_count"] == 1
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "shared_filter.npz").is_file()
    assert (output_dir / "transfer_diagnostics.png").is_file()
    assert (output_dir / "audit" / "audit_page_001.png").is_file()
    assert len(tuple((output_dir / "audit" / "samples").glob("*.png"))) == 1
