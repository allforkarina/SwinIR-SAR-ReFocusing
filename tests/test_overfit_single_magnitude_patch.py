from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat

from scripts.overfit_single_magnitude_patch import (
    MagnitudeSuccessCriteria,
    evaluate_log_magnitude_prediction,
    magnitude_charbonnier_loss,
    prepare_log_magnitude_pair,
    run,
    validate_args,
)


def test_preprocessing_uses_echo_rms_for_both_images() -> None:
    echo = np.asarray([[3.0 + 4.0j, 0.0j], [1.0j, 2.0 + 0.0j]])
    image = 7.0 * echo

    input_tensor, target_tensor, scale = prepare_log_magnitude_pair(
        echo, image, epsilon=1.0e-12
    )

    expected_scale = np.sqrt(np.mean(np.abs(echo) ** 2) + 1.0e-12)
    assert scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(
        input_tensor.numpy()[0], np.log1p(np.abs(echo) / expected_scale), rtol=1.0e-6
    )
    np.testing.assert_allclose(
        target_tensor.numpy()[0], np.log1p(np.abs(image) / expected_scale), rtol=1.0e-6
    )


def test_magnitude_loss_and_identity_metrics() -> None:
    target = torch.linspace(0.0, 1.0, 16 * 16).reshape(1, 1, 16, 16)
    prediction = target.clone().requires_grad_(True)

    loss = magnitude_charbonnier_loss(prediction, target)
    loss.backward()
    metrics = evaluate_log_magnitude_prediction(
        prediction.detach(), target, charbonnier_epsilon=1.0e-3
    )

    assert torch.isfinite(loss)
    assert prediction.grad is not None
    assert metrics["normalized_log_rmse"] == pytest.approx(0.0)
    assert metrics["log_magnitude_correlation"] == pytest.approx(1.0)
    assert metrics["magnitude_rms_ratio_target"] == pytest.approx(1.0)
    assert metrics["log_magnitude_psnr_db"] == pytest.approx(float("inf"))
    assert metrics["log_magnitude_ssim"] == pytest.approx(1.0)


def test_success_criteria_require_every_metric() -> None:
    criteria = MagnitudeSuccessCriteria()
    passing = {
        "normalized_log_rmse": 0.09,
        "log_magnitude_correlation": 0.96,
        "magnitude_rms_ratio_target": 1.02,
        "log_magnitude_psnr_db": 31.0,
        "log_magnitude_ssim": 0.96,
    }
    assert criteria.is_satisfied(passing)
    for name in passing:
        failing = dict(passing)
        if name == "normalized_log_rmse":
            failing[name] = 0.2
        elif name == "magnitude_rms_ratio_target":
            failing[name] = 0.2
        else:
            failing[name] = 0.0
        assert not criteria.is_satisfied(failing), name


def _tiny_config(path: Path) -> None:
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


def _args(
    *,
    config: Path,
    echo_file: Path,
    image_file: Path,
    output_dir: Path,
    resume: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        echo_file=echo_file,
        image_file=image_file,
        output_dir=output_dir,
        resume=resume,
        device="cpu",
        steps=1,
        eval_every=1,
        save_every=1,
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


def test_validate_args_rejects_too_few_evaluations(tmp_path: Path) -> None:
    args = _args(
        config=tmp_path / "config.yaml",
        echo_file=tmp_path / "sample.mat",
        image_file=tmp_path / "sample.mat",
        output_dir=tmp_path / "run",
        resume=None,
    )
    args.steps = 200
    args.eval_every = 100
    args.required_consecutive_successes = 3
    with pytest.raises(ValueError, match="required minimum=300"):
        validate_args(args)


def test_tiny_cpu_run_writes_artifacts_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    _tiny_config(config)
    rng = np.random.default_rng(7)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (1.8 + 0.2j) * echo
    filename = "patch_row_17500_col_9400_2.mat"
    echo_file = tmp_path / "echo" / filename
    image_file = tmp_path / "image" / filename
    echo_file.parent.mkdir()
    image_file.parent.mkdir()
    savemat(echo_file, {"patch": echo})
    savemat(image_file, {"patch": image})
    output_dir = tmp_path / "run"

    first_report = run(
        _args(
            config=config,
            echo_file=echo_file,
            image_file=image_file,
            output_dir=output_dir,
            resume=None,
        )
    )

    assert first_report["status"] == "failed"
    assert first_report["experiment"] == "D002-B2-A-single-patch"
    assert first_report["representation"]["normalization_source"] == "Echo only"
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "final.pt").is_file()
    assert (output_dir / "figures" / "step_000001.png").is_file()
    assert (output_dir / "predictions" / "step_000001.mat").is_file()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sample"]["echo_sha256"]

    resumed_report = run(
        _args(
            config=config,
            echo_file=echo_file,
            image_file=image_file,
            output_dir=output_dir,
            resume=output_dir / "checkpoints" / "latest.pt",
        )
    )
    assert resumed_report["step"] == 1
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["model"]["in_chans"] == 1
    assert resolved["model"]["drop_path_rate"] == pytest.approx(0.0)
