from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from scripts.overfit_single_patch import SuccessCriteria, run, validate_args


def test_success_criteria_require_every_agreed_metric() -> None:
    criteria = SuccessCriteria()
    passing = {
        "normalized_complex_rmse": 0.09,
        "complex_coherence": 0.96,
        "magnitude_correlation": 0.97,
        "rms_ratio_target": 1.02,
        "log_magnitude_psnr_db": 31.0,
        "log_magnitude_ssim": 0.96,
    }

    assert criteria.is_satisfied(passing)
    for name in passing:
        failing = dict(passing)
        if name in {"normalized_complex_rmse", "rms_ratio_target"}:
            failing[name] = 0.2
        else:
            failing[name] = 0.0
        assert not criteria.is_satisfied(failing), name


def tiny_config(path: Path) -> None:
    config = {
        "model": {
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


def experiment_args(
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
        db_floor=-60.0,
        success_rmse=0.0,
        success_coherence=1.0,
        success_magnitude_correlation=1.0,
        success_rms_ratio_min=1.0,
        success_rms_ratio_max=1.0,
        success_psnr=300.0,
        success_ssim=1.0,
    )


def test_validate_args_rejects_budget_without_three_post_update_evaluations(
    tmp_path: Path,
) -> None:
    args = experiment_args(
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


def test_tiny_cpu_run_writes_artifacts_and_restores_checkpoint(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    tiny_config(config)
    rng = np.random.default_rng(7)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (0.8 + 0.1j) * echo
    filename = "patch_row_17500_col_9400_2.mat"
    echo_file = tmp_path / "echo" / filename
    image_file = tmp_path / "image" / filename
    echo_file.parent.mkdir()
    image_file.parent.mkdir()
    savemat(echo_file, {"patch": echo})
    savemat(image_file, {"patch": image})
    output_dir = tmp_path / "run"

    first_report = run(
        experiment_args(
            config=config,
            echo_file=echo_file,
            image_file=image_file,
            output_dir=output_dir,
            resume=None,
        )
    )

    assert first_report["status"] == "failed"
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "final.pt").is_file()
    assert (output_dir / "figures" / "step_000001.png").is_file()
    assert (output_dir / "predictions" / "step_000001.mat").is_file()
    report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert report["sample"]["echo_sha256"]

    resumed_report = run(
        experiment_args(
            config=config,
            echo_file=echo_file,
            image_file=image_file,
            output_dir=output_dir,
            resume=output_dir / "checkpoints" / "latest.pt",
        )
    )

    assert resumed_report["step"] == 1
    resolved = json.loads((output_dir / "resolved_config.json").read_text(encoding="utf-8"))
    assert resolved["model"]["drop_path_rate"] == pytest.approx(0.0)
