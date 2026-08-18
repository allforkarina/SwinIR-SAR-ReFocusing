from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from scripts.overfit_magnitude_patch_set import (
    load_selected_samples,
    run,
    summarize_metric_map,
    validate_args,
)
from scripts.overfit_single_magnitude_patch import MagnitudeSuccessCriteria
from swinir.sar_dataset import discover_pairs


def _write_config(path: Path) -> None:
    config = {
        "model": {
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
            "drop_path_rate": 0.5,
            "ape": False,
            "patch_norm": True,
            "use_checkpoint": False,
            "upscale": 1,
            "img_range": 1.0,
            "upsampler": "",
            "resi_connection": "1conv",
        },
        "data": {"expected_shape": [16, 16], "rms_epsilon": 1.0e-12},
        "optimization": {"charbonnier_epsilon": 1.0e-3},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def _write_pair(
    echo_dir: Path,
    image_dir: Path,
    filename: str,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (1.8 + 0.2j) * echo
    savemat(echo_dir / filename, {"patch": echo})
    savemat(image_dir / filename, {"patch": image})


def _args(
    *,
    config: Path,
    echo_dir: Path,
    image_dir: Path,
    output_dir: Path,
    anchor_filename: str,
    resume: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        anchor_filename=anchor_filename,
        sample_count=2,
        resume=resume,
        device="cpu",
        steps=2,
        eval_every=2,
        save_every=2,
        required_consecutive_successes=1,
        seed=42,
        learning_rate=2.0e-4,
        ema_decay=0.999,
        charbonnier_epsilon=None,
        rms_epsilon=None,
        success_rmse=0.0,
        success_correlation=1.0,
        success_rms_ratio_min=1.0,
        success_rms_ratio_max=1.0,
        success_psnr=300.0,
        success_ssim=1.0,
    )


def test_loaded_samples_use_one_channel_and_pair_specific_echo_scale(
    tmp_path: Path,
) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    _write_pair(echo_dir, image_dir, "patch_row_0_col_0_2.mat", 1)
    _write_pair(echo_dir, image_dir, "patch_row_100_col_100_2.mat", 2)

    samples = load_selected_samples(
        discover_pairs(echo_dir, image_dir),
        expected_shape=(16, 16),
        rms_epsilon=1.0e-12,
    )

    assert len(samples) == 2
    assert all(sample.inputs.shape == (1, 1, 16, 16) for sample in samples)
    assert all(sample.targets.shape == (1, 1, 16, 16) for sample in samples)
    assert all(sample.scale > 0 for sample in samples)


def test_summary_requires_every_sample_to_pass() -> None:
    criteria = MagnitudeSuccessCriteria()
    passing = {
        "normalized_log_rmse": 0.09,
        "log_magnitude_correlation": 0.96,
        "magnitude_rms_ratio_target": 1.0,
        "log_magnitude_psnr_db": 31.0,
        "log_magnitude_ssim": 0.96,
        "magnitude_charbonnier": 0.02,
    }
    failing = dict(passing, log_magnitude_ssim=0.90)

    summary = summarize_metric_map({"a.mat": passing, "b.mat": failing}, criteria)

    assert summary["pass_count"] == 1
    assert not summary["all_passed"]
    assert summary["aggregate"]["log_magnitude_ssim"]["min"] == pytest.approx(0.90)


def test_validate_args_requires_whole_epoch_evaluations(tmp_path: Path) -> None:
    args = _args(
        config=tmp_path / "config.yaml",
        echo_dir=tmp_path,
        image_dir=tmp_path,
        output_dir=tmp_path / "run",
        anchor_filename="anchor.mat",
        resume=None,
    )
    args.eval_every = 3
    with pytest.raises(ValueError, match="whole epochs"):
        validate_args(args)


def test_tiny_joint_run_writes_manifest_metrics_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    _write_config(config)
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    anchor = "patch_row_0_col_0_2.mat"
    _write_pair(echo_dir, image_dir, anchor, 1)
    _write_pair(echo_dir, image_dir, "patch_row_100_col_100_2.mat", 2)
    output_dir = tmp_path / "run"

    report = run(
        _args(
            config=config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            anchor_filename=anchor,
            resume=None,
        )
    )

    assert report["status"] == "failed"
    assert report["experiment"] == "D002-B2-A-joint-magnitude-patch-set"
    assert report["selection_manifest"]["sample_count"] == 2
    assert report["final"]["raw"]["sample_count"] == 2
    assert report["representation"]["normalization_source"] == "Echo only per pair"
    assert (output_dir / "selected_samples.json").is_file()
    assert (output_dir / "metrics.jsonl").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "final.pt").is_file()
    selected = json.loads(
        (output_dir / "selected_samples.json").read_text(encoding="utf-8")
    )
    assert selected["samples"][0]["filename"] == anchor

    resumed = run(
        _args(
            config=config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            anchor_filename=anchor,
            resume=output_dir / "checkpoints" / "latest.pt",
        )
    )
    assert resumed["step"] == 2
