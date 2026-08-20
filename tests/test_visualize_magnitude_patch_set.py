from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from scripts.overfit_magnitude_patch_set import load_selected_samples
from scripts.overfit_patch_set import selection_manifest
from scripts.visualize_magnitude_patch_set import ordered_manifest_pairs, run
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair, discover_pairs


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


def _write_pair(echo_dir: Path, image_dir: Path, filename: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (1.5 + 0.1j) * echo
    savemat(echo_dir / filename, {"patch": echo})
    savemat(image_dir / filename, {"patch": image})


def test_manifest_pair_order_is_checkpoint_order(tmp_path: Path) -> None:
    pair_a = DiscoveredPair(0, 0, tmp_path / "a.mat", tmp_path / "a.mat")
    pair_b = DiscoveredPair(100, 100, tmp_path / "b.mat", tmp_path / "b.mat")
    manifest = {
        "sample_count": 2,
        "samples": [
            {"selection_index": 1, "filename": "b.mat", "row": 100, "col": 100},
            {"selection_index": 0, "filename": "a.mat", "row": 0, "col": 0},
        ],
    }

    selected = ordered_manifest_pairs((pair_b, pair_a), manifest)

    assert [pair.echo_path.name for pair in selected] == ["a.mat", "b.mat"]


def test_manifest_pair_order_rejects_coordinate_mismatch(tmp_path: Path) -> None:
    pair = DiscoveredPair(0, 0, tmp_path / "a.mat", tmp_path / "a.mat")
    manifest = {
        "sample_count": 1,
        "samples": [
            {"selection_index": 0, "filename": "a.mat", "row": 1, "col": 0}
        ],
    }

    with pytest.raises(RuntimeError, match="coordinate mismatch"):
        ordered_manifest_pairs((pair,), manifest)


def test_run_exports_contact_sheet_individual_figures_and_metrics(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    filenames = (
        "patch_row_0_col_0_2.mat",
        "patch_row_100_col_100_2.mat",
    )
    for seed, filename in enumerate(filenames, start=1):
        _write_pair(echo_dir, image_dir, filename, seed)

    pairs = discover_pairs(echo_dir, image_dir)
    samples = load_selected_samples(
        pairs,
        expected_shape=(16, 16),
        rms_epsilon=1.0e-12,
    )
    manifest = selection_manifest(
        samples,
        echo_dir=echo_dir,
        image_dir=image_dir,
        patch_shape=(16, 16),
        anchor_filename=filenames[0],
    )
    model_config = _model_config()
    model = SwinIR(**model_config)
    checkpoint = tmp_path / "final.pt"
    torch.save(
        {
            "schema_version": 1,
            "step": 123,
            "model": model.state_dict(),
            "ema_model": model.state_dict(),
            "resolved_config": {
                "experiment": "D002-B2-A-joint-magnitude-patch-set",
                "selection_manifest": manifest,
                "model": model_config,
                "data": {
                    "expected_shape": [16, 16],
                    "rms_epsilon": 1.0e-12,
                    "input": "log1p(abs(Echo) / rms(Echo))",
                    "target": "log1p(abs(Image) / rms(Echo))",
                },
                "optimization": {"charbonnier_epsilon": 1.0e-3},
            },
        },
        checkpoint,
    )
    output_dir = tmp_path / "visuals"
    args = argparse.Namespace(
        checkpoint=checkpoint,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        weights="raw",
        device="cpu",
        dpi=40,
    )

    result = run(args)

    assert result["checkpoint_step"] == 123
    assert result["sample_count"] == 2
    assert (output_dir / "all_02_echo_prediction_image.png").is_file()
    assert len(tuple((output_dir / "samples").glob("*.png"))) == 2
    written = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert [sample["filename"] for sample in written["samples"]] == list(filenames)
    assert all("normalized_log_rmse" in sample["metrics"] for sample in written["samples"])
