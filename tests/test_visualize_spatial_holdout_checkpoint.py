from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from scripts.visualize_spatial_holdout_checkpoint import (
    run,
    select_representative_samples,
)
from swinir import SwinIR
from swinir.sar_dataset import CoordinateRegion, build_manifest


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


def _metrics(value: float) -> dict[str, float]:
    return {
        "normalized_log_rmse": value,
        "log_magnitude_correlation": 1.0 - value,
        "magnitude_rms_ratio_target": 0.5 + value,
        "log_magnitude_psnr_db": 20.0 - value,
        "log_magnitude_ssim": 1.0 - value,
        "magnitude_charbonnier": value,
    }


def test_representative_selection_includes_metric_extremes() -> None:
    metrics = {f"sample_{index}.mat": _metrics(index / 10) for index in range(10)}
    coordinates = {
        filename: (index, index)
        for index, filename in enumerate(metrics)
    }

    selected = select_representative_samples(metrics, coordinates, sample_count=5)
    reasons = {
        reason
        for sample in selected
        for reason in sample.reasons
    }

    assert len(selected) == 5
    assert "best_rmse" in reasons
    assert "worst_rmse" in reasons
    assert "median_rmse" in reasons


def test_representative_selection_rejects_manifest_mismatch() -> None:
    with pytest.raises(RuntimeError, match="do not match"):
        select_representative_samples(
            {"a.mat": _metrics(0.1)},
            {"b.mat": (0, 0)},
            sample_count=1,
        )


def test_run_exports_shared_scale_audit_and_individual_figures(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    filenames = []
    for index, coordinate in enumerate(range(0, 81, 20), start=1):
        filename = f"patch_row_{coordinate}_col_{coordinate}_2.mat"
        filenames.append(filename)
        rng = np.random.default_rng(index)
        echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        image = 1.5 * echo
        savemat(echo_dir / filename, {"patch": echo})
        savemat(image_dir / filename, {"patch": image})

    region = CoordinateRegion(0, 80, 0, 80)
    manifest = build_manifest(
        echo_dir,
        image_dir,
        region,
        region,
        expected_counts={"train": 0, "guard": 0, "validation": 5},
    )
    model_config = _model_config()
    model = SwinIR(**model_config)
    per_sample = {
        filename: _metrics(0.1 + index / 100)
        for index, filename in enumerate(filenames)
    }
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "schema_version": 1,
            "global_step": 123,
            "model": model.state_dict(),
            "ema_model": model.state_dict(),
            "manifest_fingerprint": manifest.fingerprint,
            "best_validation": {
                "step": 123,
                "summary": {"per_sample": per_sample},
            },
            "resolved_config": {
                "experiment": "E004-D002-spatial-holdout-magnitude",
                "model": model_config,
                "data": {
                    "expected_shape": [16, 16],
                    "rms_epsilon": 1.0e-12,
                    "validation_region": {
                        "row_min": 0,
                        "row_max": 80,
                        "col_min": 0,
                        "col_max": 80,
                    },
                    "guard_region": {
                        "row_min": 0,
                        "row_max": 80,
                        "col_min": 0,
                        "col_max": 80,
                    },
                    "expected_split_counts": {
                        "train": 0,
                        "guard": 0,
                        "validation": 5,
                    },
                },
                "optimization": {"charbonnier_epsilon": 1.0e-3},
            },
        },
        checkpoint_path,
    )
    output_dir = tmp_path / "audit"
    args = argparse.Namespace(
        checkpoint=checkpoint_path,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        sample_count=3,
        selection="representative",
        weights="raw",
        device="cpu",
        dpi=40,
        contact_sheet_page_size=12,
    )

    result = run(args)

    assert result["checkpoint_step"] == 123
    assert result["sample_count"] == 3
    assert result["validation_sample_count"] == 5
    assert (output_dir / "audit_003_validation_samples.png").is_file()
    assert len(tuple((output_dir / "samples").glob("*.png"))) == 3
    written = json.loads(
        (output_dir / "audit_manifest.json").read_text(encoding="utf-8")
    )
    assert len(written["samples"]) == 3
    assert all("prediction_metrics" in sample for sample in written["samples"])
    assert all("echo_identity_metrics" in sample for sample in written["samples"])
