from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat

import scripts.diagnose_phase_oracle as phase_oracle_module
from scripts.diagnose_phase_oracle import (
    apply_phase_polynomial,
    estimate_magnitude_shift,
    fit_quadratic_phase,
    run,
    select_spatial_oracle_records,
    unrestricted_phase_oracle,
)
from swinir.sar_dataset import PairRecord, SplitName


def _record(tmp_path: Path, row: int, col: int) -> PairRecord:
    name = f"patch_row_{row}_col_{col}_2.mat"
    return PairRecord(
        key=f"row_{row}_col_{col}",
        row=row,
        col=col,
        split=SplitName.VALIDATION,
        echo_path=tmp_path / "echo" / name,
        image_path=tmp_path / "image" / name,
    )


def test_spatial_oracle_selection_is_deterministic_and_covers_grid(tmp_path: Path) -> None:
    records = tuple(
        _record(tmp_path, row, col)
        for row in range(0, 1001, 100)
        for col in range(0, 1001, 100)
    )

    first = select_spatial_oracle_records(records, 25)
    second = select_spatial_oracle_records(tuple(reversed(records)), 25)

    assert first == second
    assert len(first) == 25
    assert min(record.row for record in first) == 0
    assert max(record.row for record in first) == 1000
    assert min(record.col for record in first) == 0
    assert max(record.col for record in first) == 1000


def test_magnitude_shift_recovers_known_circular_translation() -> None:
    rng = np.random.default_rng(7)
    echo = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    target = np.roll(echo, shift=(5, -7), axis=(0, 1))

    assert estimate_magnitude_shift(echo, target, maximum_shift=10) == (5, -7)


def test_unrestricted_phase_oracle_is_exact_for_phase_only_mapping() -> None:
    rng = np.random.default_rng(11)
    echo = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    coefficients = (0.3, -4.0, 2.5, 3.0, -1.5, 2.0)
    target = apply_phase_polynomial(echo, coefficients, fft_norm="ortho")

    prediction, _ = unrestricted_phase_oracle(echo, target, fft_norm="ortho")

    relative_error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
    assert relative_error < 1.0e-12


def test_quadratic_phase_fit_reduces_circular_objective() -> None:
    rng = np.random.default_rng(23)
    echo = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
    target = apply_phase_polynomial(
        echo, (0.2, -2.0, 1.0, 2.0, -0.8, 1.5), fft_norm="ortho"
    )

    fit = fit_quadratic_phase(
        echo,
        target,
        fft_norm="ortho",
        maximum_shift=8,
        maximum_frequency_samples=1024,
        quadratic_bound=20.0,
        maximum_iterations=400,
        ftol=1.0e-12,
        gtol=1.0e-10,
    )
    prediction = apply_phase_polynomial(echo, fit.coefficients, fft_norm="ortho")
    coherence = abs(np.vdot(prediction, target)) / (
        np.linalg.norm(prediction) * np.linalg.norm(target)
    )

    assert fit.final_objective < 1.0e-10
    assert fit.final_objective <= fit.initial_objective
    assert coherence > 0.999999999


def test_tiny_end_to_end_run_writes_report_audit_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    shape = (16, 16)
    coefficients = (0.15, -1.0, 0.8, 0.6, -0.2, 0.4)
    for row in range(0, 81, 20):
        for col in range(0, 81, 20):
            record = _record(tmp_path, row, col)
            rng = np.random.default_rng(row * 100 + col + 1)
            echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
            image = apply_phase_polynomial(echo, coefficients, fft_norm="ortho")
            savemat(record.echo_path, {"patch": echo})
            savemat(record.image_path, {"patch": image})

    config = {
        "experiment": "test-phase-oracle",
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
            "oracle_sample_count": 1,
        },
        "phase": {
            "fft_norm": "ortho",
            "maximum_shift_pixels": 4,
            "maximum_frequency_samples": 256,
            "quadratic_coefficient_bound_radians": 10.0,
            "optimizer_max_iterations": 200,
            "optimizer_ftol": 1.0e-10,
            "optimizer_gtol": 1.0e-8,
        },
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "high_frequency_radius_fraction": 0.25,
            "audit_sample_count": 1,
            "success_criteria": {
                "unrestricted_rmse_win_fraction_vs_echo_min": 0.0,
                "unrestricted_mean_coherence_delta_vs_echo_min": -1.0,
                "unrestricted_mean_log_ssim_delta_vs_echo_min": -1.0,
                "unrestricted_mean_edge_correlation_delta_vs_echo_min": -1.0,
                "unrestricted_median_high_frequency_energy_ratio_min": 0.0,
                "unrestricted_median_high_frequency_energy_ratio_max": 2.0,
                "quadratic_rmse_win_fraction_vs_echo_min": 0.0,
                "quadratic_mean_coherence_delta_vs_echo_min": -1.0,
                "quadratic_mean_log_ssim_delta_vs_echo_min": -1.0,
                "quadratic_mean_edge_correlation_delta_vs_echo_min": -1.0,
                "quadratic_median_high_frequency_energy_ratio_min": 0.0,
                "quadratic_median_high_frequency_energy_ratio_max": 2.0,
                "quadratic_fraction_of_unrestricted_ssim_gain_min": -10.0,
                "quadratic_fraction_of_unrestricted_edge_gain_min": -10.0,
            },
        },
        "runtime": {
            "progress_interval_samples": 1,
            "figure_dpi": 40,
            "contact_sheet_page_size": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output_dir = tmp_path / "output"

    report = run(
        argparse.Namespace(
            config=config_path,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
        )
    )

    assert report["status"] == "quadratic_phase_oracle_metric_supported"
    assert report["dataset"]["oracle_sample_count"] == 1
    assert (output_dir / "report.json").is_file()
    assert (output_dir / "phase_fits.json").is_file()
    assert (output_dir / "phase_diagnostics.png").is_file()
    assert (output_dir / "audit" / "audit_page_001.png").is_file()
    assert len(tuple((output_dir / "audit" / "samples").glob("*.png"))) == 1

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("resume must not refit cached quadratic phase")

    monkeypatch.setattr(
        phase_oracle_module, "fit_quadratic_phase", fail_if_refit
    )
    resumed = run(
        argparse.Namespace(
            config=config_path,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            resume=True,
        )
    )

    assert resumed["status"] == report["status"]
    assert resumed["comparison"]["quadratic_phase"]["metric_supported"] is True
    assert np.isclose(
        resumed["comparison"]["quadratic_phase"][
            "mean_log_ssim_delta_vs_echo"
        ],
        report["comparison"]["quadratic_phase"][
            "mean_log_ssim_delta_vs_echo"
        ],
        rtol=1.0e-12,
        atol=1.0e-15,
    )
