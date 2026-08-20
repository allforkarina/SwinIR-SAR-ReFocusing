from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import savemat

from scripts.train_magnitude_spatial_holdout import (
    MagnitudePatchDataset,
    compare_to_identity,
    run,
    summarize_metrics,
)
from swinir.sar_dataset import (
    CoordinateRegion,
    SplitName,
    build_manifest,
    classify_coordinate,
)


def _model_config() -> dict[str, object]:
    return {
        "img_size": 16,
        "patch_size": 1,
        "in_chans": 1,
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


def _success_criteria() -> dict[str, object]:
    return {
        "mean_rmse_relative_improvement_min": 0.10,
        "median_rmse_relative_improvement_min": 0.10,
        "rmse_win_fraction_min": 0.75,
        "require_mean_correlation_improvement": True,
        "require_mean_psnr_improvement": True,
        "require_mean_ssim_improvement": True,
        "median_magnitude_rms_ratio_min": 0.90,
        "median_magnitude_rms_ratio_max": 1.10,
    }


def _write_config(path: Path, expected_counts: dict[str, int]) -> None:
    config = {
        "model": _model_config(),
        "data": {
            "expected_shape": [16, 16],
            "rms_epsilon": 1.0e-12,
            "representation": "echo_rms_log1p_magnitude",
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
            "expected_split_counts": expected_counts,
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
            "total_steps": 2,
            "loss": "magnitude_charbonnier",
            "charbonnier_epsilon": 1.0e-3,
            "ema_decay": 0.999,
        },
        "runtime": {
            "seed": 42,
            "strict_reproducibility": False,
            "batch_size": 1,
            "validation_interval_steps": 1,
            "archive_interval_steps": 2,
            "log_interval_steps": 1,
            "max_consecutive_fp16_overflows": 8,
            "early_stopping": {"patience": 10, "min_delta": 0.0},
        },
        "evaluation": {
            "authority": "raw_model_spatial_validation",
            "success_criteria": _success_criteria(),
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _write_grid(echo_dir: Path, image_dir: Path) -> None:
    echo_dir.mkdir()
    image_dir.mkdir()
    for row in range(0, 81, 20):
        for col in range(0, 81, 20):
            seed = row * 1000 + col + 1
            rng = np.random.default_rng(seed)
            echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
            image = (1.4 + 0.1j) * echo
            name = f"patch_row_{row}_col_{col}_2.mat"
            savemat(echo_dir / name, {"patch": echo})
            savemat(image_dir / name, {"patch": image})


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


def test_scene4_contract_has_expected_counts_and_zero_train_validation_overlap() -> None:
    validation = CoordinateRegion(16400, 18400, 7700, 9700)
    guard = CoordinateRegion(15888, 18912, 7188, 10212)
    coordinates = [
        (row, col)
        for row in range(10000, 24401, 100)
        for col in range(3000, 14401, 100)
    ]
    split_by_coordinate = {
        coordinate: classify_coordinate(*coordinate, validation, guard)
        for coordinate in coordinates
    }

    counts = Counter(split.value for split in split_by_coordinate.values())

    assert dict(counts) == {"train": 15714, "guard": 520, "validation": 441}
    train = [
        coordinate
        for coordinate, split in split_by_coordinate.items()
        if split is SplitName.TRAIN
    ]
    for row, col in train:
        row_gap = (
            validation.row_min - row
            if row < validation.row_min
            else row - validation.row_max
            if row > validation.row_max
            else 0
        )
        col_gap = (
            validation.col_min - col
            if col < validation.col_min
            else col - validation.col_max
            if col > validation.col_max
            else 0
        )
        assert row_gap >= 512 or col_gap >= 512


def test_magnitude_dataset_uses_one_channel_echo_rms_log_representation(
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
    dataset = MagnitudePatchDataset(
        manifest.records_for(SplitName.VALIDATION),
        expected_shape=(16, 16),
        rms_epsilon=1.0e-12,
    )

    sample = dataset[0]

    assert sample["input"].shape == (1, 16, 16)
    assert sample["target"].shape == (1, 16, 16)
    assert sample["input"].dtype is torch.float32
    assert sample["target"].dtype is torch.float32
    assert float(sample["scale"]) > 0


def test_success_requires_distribution_wide_multi_metric_improvement() -> None:
    identity_per_sample = {
        name: {
            "normalized_log_rmse": 1.0,
            "log_magnitude_correlation": 0.1,
            "magnitude_rms_ratio_target": 1.0,
            "log_magnitude_psnr_db": 15.0,
            "log_magnitude_ssim": 0.1,
            "magnitude_charbonnier": 0.5,
        }
        for name in ("a.mat", "b.mat", "c.mat", "d.mat")
    }
    model_per_sample = {
        name: {
            **metrics,
            "normalized_log_rmse": 0.8,
            "log_magnitude_correlation": 0.5,
            "log_magnitude_psnr_db": 20.0,
            "log_magnitude_ssim": 0.5,
            "magnitude_charbonnier": 0.3,
        }
        for name, metrics in identity_per_sample.items()
    }

    comparison = compare_to_identity(
        summarize_metrics(model_per_sample),
        summarize_metrics(identity_per_sample),
        _success_criteria(),
    )

    assert comparison["passed"] is True
    assert np.isclose(comparison["mean_rmse_relative_improvement"], 0.2)
    assert comparison["rmse_win_fraction"] == 1.0


def test_tiny_cpu_run_writes_strict_artifacts_and_resumes(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    _write_grid(echo_dir, image_dir)
    config = tmp_path / "config.yaml"
    _write_config(config, {"train": 24, "guard": 0, "validation": 1})
    output = tmp_path / "run"

    first = run(_args(config, echo_dir, image_dir, output))

    assert first["global_step"] == 2
    assert first["split_counts"] == {"train": 24, "guard": 0, "validation": 1}
    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "latest.pt").is_file()
    assert (output / "checkpoints" / "final.pt").is_file()
    assert (output / "checkpoints" / "step_000002.pt").is_file()
    assert (output / "split_manifest.json").is_file()
    assert (output / "echo_identity_baseline.json").is_file()
    assert (output / "report.json").is_file()

    resumed = run(
        _args(
            config,
            echo_dir,
            image_dir,
            output,
            output / "checkpoints" / "latest.pt",
        )
    )

    assert resumed["global_step"] == 2
    written = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert written["manifest_fingerprint"] == first["manifest_fingerprint"]
