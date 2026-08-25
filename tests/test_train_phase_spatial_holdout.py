from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat

from scripts.train_phase_spatial_holdout import (
    PhasePatchDataset,
    ValidationBaselines,
    add_generalization_metrics,
    aggregate_metrics,
    compare_to_baselines,
    run,
)
from scripts.visualize_phase_spatial_holdout_checkpoint import (
    load_checkpoint,
    run as run_visualization,
)
from scripts.visualize_phase_spatial_holdout_checkpoint import select_representative
from swinir.sar_dataset import CoordinateRegion, SplitName, build_manifest


def _model_config() -> dict[str, object]:
    return {
        "img_size": 16,
        "patch_size": 1,
        "in_chans": 2,
        "embed_dim": 12,
        "depths": [1],
        "num_heads": [3],
        "window_size": 4,
        "mlp_ratio": 2.0,
        "qkv_bias": True,
        "qk_scale": None,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "ape": False,
        "patch_norm": True,
        "use_checkpoint": False,
        "upscale": 1,
        "img_range": 1.0,
        "upsampler": "",
        "resi_connection": "1conv",
    }


def _criteria(value: float = 1.0) -> dict[str, float]:
    return {
        "mean_phase_alignment_min": value,
        "median_phase_alignment_min": value,
        "p05_phase_alignment_min": value,
        "mean_rmse_oracle_gap_fraction_closed_min": value,
        "median_rmse_oracle_gap_fraction_closed_min": value,
        "rmse_win_fraction_vs_echo_min": value,
        "mean_coherence_fraction_of_oracle_min": value,
        "mean_ssim_gain_fraction_of_oracle_min": value,
        "mean_edge_gain_fraction_of_oracle_min": value,
        "median_high_frequency_energy_ratio_min": 1.0,
        "median_high_frequency_energy_ratio_max": 1.0,
    }


def _write_config(path: Path) -> None:
    config = {
        "experiment": "E010-D001-phase-spatial-holdout",
        "model": _model_config(),
        "data": {
            "expected_shape": [16, 16],
            "rms_epsilon": 1.0e-12,
            "fft_norm": "ortho",
            "representation": "fftshifted_echo_complex_spectrum_to_unit_phase_correction",
            "validation_region": {
                "row_min": 40,
                "row_max": 40,
                "col_min": 40,
                "col_max": 40,
            },
            "guard_region": {
                "row_min": 24,
                "row_max": 56,
                "col_min": 24,
                "col_max": 56,
            },
            "expected_split_counts": {"train": 24, "guard": 0, "validation": 1},
            "num_workers": 0,
            "prefetch_factor": 2,
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": 2.0e-4,
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate_schedule": "constant",
            "total_steps": 1,
            "phase_loss_weight": 1.0,
            "complex_reconstruction_weight": 0.25,
            "log_magnitude_weight": 0.25,
            "phase_energy_weight_power": 0.5,
            "phasor_epsilon": 1.0e-6,
            "charbonnier_epsilon": 1.0e-3,
            "ema_decay": 0.999,
        },
        "runtime": {
            "seed": 42,
            "strict_reproducibility": False,
            "batch_size": 1,
            "validation_interval_steps": 1,
            "archive_interval_steps": 1,
            "log_interval_steps": 1,
            "max_consecutive_fp16_overflows": 8,
            "required_consecutive_successes": 3,
            "early_stopping": {"patience": 10, "min_delta": 0.0},
        },
        "evaluation": {
            "authority": "raw_model_spatial_validation",
            "log_magnitude_floor_db": -60.0,
            "high_frequency_radius_fraction": 0.25,
            "success_criteria": _criteria(),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _phase_pair(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    row = np.fft.fftfreq(shape[0])[:, None]
    col = np.fft.fftfreq(shape[1])[None, :]
    correction = np.exp(1j * (0.15 * seed + 2.0 * row - 1.5 * col))
    spectrum = np.fft.fft2(echo, norm="ortho")
    image = np.fft.ifft2(correction * spectrum, norm="ortho")
    return echo, image


def _write_grid(echo_dir: Path, image_dir: Path) -> None:
    echo_dir.mkdir()
    image_dir.mkdir()
    seed = 0
    for row in range(0, 81, 20):
        for col in range(0, 81, 20):
            seed += 1
            echo, image = _phase_pair((16, 16), seed)
            filename = f"patch_row_{row}_col_{col}_2.mat"
            savemat(echo_dir / filename, {"patch": echo})
            savemat(image_dir / filename, {"patch": image})


def _args(
    config: Path,
    echo_dir: Path,
    image_dir: Path,
    output_dir: Path,
    resume: Path | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        resume=resume,
        device="cpu",
    )


def _base_metrics(*, rmse: float, coherence: float, ssim: float, edge: float) -> dict[str, float]:
    return {
        "weighted_phase_alignment": 1.0,
        "normalized_complex_rmse": rmse,
        "complex_coherence": coherence,
        "magnitude_correlation": 0.8,
        "rms_ratio_target": 1.0,
        "log_magnitude_psnr_db": 25.0,
        "log_magnitude_ssim": ssim,
        "edge_correlation": edge,
        "gradient_energy_ratio": 1.0,
        "high_frequency_energy_ratio": 1.0,
    }


def test_phase_dataset_exposes_echo_spectrum_and_image_supervised_phasor(
    tmp_path: Path,
) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    _write_grid(echo_dir, image_dir)
    manifest = build_manifest(
        echo_dir,
        image_dir,
        CoordinateRegion(40, 40, 40, 40),
        CoordinateRegion(24, 56, 24, 56),
    )
    dataset = PhasePatchDataset(
        manifest.records_for(SplitName.VALIDATION),
        expected_shape=(16, 16),
        data_config={"rms_epsilon": 1.0e-12, "fft_norm": "ortho"},
        optimization={"phasor_epsilon": 1.0e-6, "phase_energy_weight_power": 0.5},
    )

    sample = dataset[0]

    assert sample["input_spectrum"].shape == (2, 16, 16)
    assert sample["target_phasor"].shape == (2, 16, 16)
    assert sample["phase_weights"].shape == (1, 16, 16)
    assert sample["target_image"].shape == (2, 16, 16)
    assert sample["input_spectrum"].dtype is torch.float32
    phasor_norm = sample["target_phasor"].square().sum(dim=0).sqrt()
    assert float(phasor_norm.mean()) == pytest.approx(1.0, abs=1.0e-5)


def test_success_gate_requires_oracle_relative_refocusing_metrics() -> None:
    echo = _base_metrics(rmse=1.0, coherence=0.1, ssim=0.1, edge=0.1)
    oracle = _base_metrics(rmse=0.4, coherence=0.8, ssim=0.5, edge=0.5)
    candidate = _base_metrics(rmse=0.55, coherence=0.7, ssim=0.45, edge=0.42)
    metrics = add_generalization_metrics(candidate, echo, oracle)
    summary = aggregate_metrics({"sample.mat": metrics})
    baselines = ValidationBaselines(
        echo_identity={"per_sample": {"sample.mat": echo}},
        unrestricted_phase_oracle={"per_sample": {"sample.mat": oracle}},
    )

    comparison = compare_to_baselines(summary, baselines, _criteria(0.5))

    assert comparison["passed"] is True
    assert comparison["rmse_win_fraction_vs_echo"] == 1.0
    assert comparison["mean_rmse_oracle_gap_fraction_closed"] == pytest.approx(0.75)
    failed = compare_to_baselines(
        aggregate_metrics(
            {
                "sample.mat": {
                    **metrics,
                    "weighted_phase_alignment": 0.1,
                }
            }
        ),
        baselines,
        _criteria(0.5),
    )
    assert failed["passed"] is False
    assert failed["checks"]["mean_phase_alignment"] is False


def test_representative_selection_includes_phase_extremes() -> None:
    metrics = {}
    coordinates = {}
    for index in range(6):
        name = f"sample_{index}.mat"
        metrics[name] = {
            "weighted_phase_alignment": index / 10,
            "rmse_oracle_gap_fraction_closed": index / 10,
            "coherence_fraction_of_oracle": index / 10,
            "ssim_gain_fraction_of_oracle": index / 10,
            "edge_gain_fraction_of_oracle": index / 10,
        }
        coordinates[name] = (index, index)

    selected = select_representative(metrics, coordinates, 3)
    reasons = {reason for item in selected for reason in item.reasons}

    assert "lowest_phase" in reasons
    assert "highest_phase" in reasons


def test_tiny_cpu_run_resumes_and_exports_unseen_audit(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    _write_grid(echo_dir, image_dir)
    config = tmp_path / "config.yaml"
    _write_config(config)
    output = tmp_path / "run"

    first = run(_args(config, echo_dir, image_dir, output))

    assert first["global_step"] == 1
    assert first["split_counts"] == {"train": 24, "guard": 0, "validation": 1}
    assert first["inference_contract"].startswith("Echo spectrum")
    assert (output / "validation_baselines.json").is_file()
    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "step_000001.pt").is_file()
    checkpoint = torch.load(
        output / "checkpoints" / "latest.pt", map_location="cpu", weights_only=False
    )
    assert "early_best_phase_alignment" in checkpoint
    assert checkpoint["resolved_config"]["runtime"]["initialization"].startswith(
        "random_seeded_no_E009"
    )

    resumed = run(
        _args(
            config,
            echo_dir,
            image_dir,
            output,
            output / "checkpoints" / "latest.pt",
        )
    )
    assert resumed["global_step"] == 1
    assert resumed["manifest_fingerprint"] == first["manifest_fingerprint"]

    audit = tmp_path / "audit"
    visual = run_visualization(
        argparse.Namespace(
            checkpoint=output / "checkpoints" / "final.pt",
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=audit,
            sample_count=1,
            selection="representative",
            weights="raw",
            device="cpu",
            dpi=40,
            contact_sheet_page_size=12,
        )
    )

    assert visual["sample_count"] == 1
    assert visual["validation_sample_count"] == 1
    assert visual["checkpoint_step"] == 1
    assert visual["validation_source"] == "last_validation"
    assert visual["validation_step"] == 1
    assert (audit / "audit_001_validation_samples.png").is_file()
    assert len(tuple((audit / "samples").glob("*.png"))) == 1
    written = json.loads((audit / "audit_manifest.json").read_text(encoding="utf-8"))
    assert written["weights"] == "raw"
    assert "stored_last_validation_raw_metrics" in written["samples"][0]

    mismatched = torch.load(
        output / "checkpoints" / "final.pt", map_location="cpu", weights_only=False
    )
    mismatched["last_validation"]["step"] = 0
    mismatch_path = tmp_path / "mismatched.pt"
    torch.save(mismatched, mismatch_path)
    with pytest.raises(RuntimeError, match="not from the same step"):
        load_checkpoint(mismatch_path)
