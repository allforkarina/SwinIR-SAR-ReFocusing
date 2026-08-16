from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from scripts.overfit_coordinate_conditioned_patch_set import (
    CoordinateBounds,
    CoordinateConditionedSwinIR,
    run,
    validate_args,
)
from scripts.overfit_patch_set import load_selected_samples, selection_manifest
from scripts.overfit_single_patch import SuccessCriteria
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair


def tiny_model_config() -> dict[str, object]:
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


def make_source_checkpoint(tmp_path: Path) -> Path:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    pairs = []
    for index, (row, col) in enumerate(((0, 0), (100, 100))):
        filename = f"patch_row_{row}_col_{col}_2.mat"
        echo_path = echo_dir / filename
        image_path = image_dir / filename
        rng = np.random.default_rng(index + 1)
        echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        image = (0.8 + 0.1j) * echo
        savemat(echo_path, {"patch": echo})
        savemat(image_path, {"patch": image})
        pairs.append(DiscoveredPair(row, col, echo_path, image_path))
    samples = load_selected_samples(pairs, expected_shape=(16, 16), rms_epsilon=1e-12)
    manifest = selection_manifest(
        samples,
        echo_dir=echo_dir,
        image_dir=image_dir,
        patch_shape=(16, 16),
        anchor_filename=pairs[0].echo_path.name,
    )
    criteria = SuccessCriteria(
        normalized_complex_rmse_max=0.0,
        complex_coherence_min=1.0,
        magnitude_correlation_min=1.0,
        rms_ratio_min=1.0,
        rms_ratio_max=1.0,
        log_magnitude_psnr_db_min=300.0,
        log_magnitude_ssim_min=1.0,
    )
    config = tiny_model_config()
    model = SwinIR(**config)
    source_config = {
        "experiment": "full_set_gradient_consolidation",
        "selection_manifest": manifest,
        "model": config,
        "data": {"expected_shape": [16, 16], "rms_epsilon": 1e-12},
        "optimization": {"charbonnier_epsilon": 1e-3},
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "success_criteria": asdict(criteria),
        },
    }
    path = tmp_path / "source.pt"
    torch.save(
        {
            "schema_version": 1,
            "resolved_config": source_config,
            "model": model.state_dict(),
        },
        path,
    )
    return path


def experiment_args(source: Path, output: Path, resume: Path | None) -> argparse.Namespace:
    return argparse.Namespace(
        source_checkpoint=source,
        output_dir=output,
        resume=resume,
        device="cpu",
        epochs=1,
        eval_every=1,
        save_every=1,
        required_consecutive_successes=1,
        seed=42,
        coordinate_hidden_dim=8,
        base_learning_rate=1e-5,
        condition_learning_rate=5e-4,
        ema_decay=0.99,
    )


def test_zero_initialized_conditioning_matches_base_for_all_coordinates() -> None:
    torch.manual_seed(9)
    config = tiny_model_config()
    base = SwinIR(**config).eval()
    conditioned = CoordinateConditionedSwinIR(
        **config, coordinate_hidden_dim=8
    ).eval()
    conditioned.load_unconditioned_state_dict(base.state_dict())
    inputs = torch.randn(1, 2, 16, 16)

    with torch.no_grad():
        expected = base(inputs)
        first = conditioned(inputs, torch.tensor([[-1.0, -1.0]]))
        second = conditioned(inputs, torch.tensor([[1.0, 1.0]]))

    torch.testing.assert_close(first, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(second, expected, rtol=0.0, atol=0.0)


def test_coordinate_encoding_maps_bounds_to_unit_square() -> None:
    bounds = CoordinateBounds(row_min=100, row_max=300, col_min=20, col_max=60)

    torch.testing.assert_close(bounds.encode(100, 20), torch.tensor([[-1.0, -1.0]]))
    torch.testing.assert_close(bounds.encode(300, 60), torch.tensor([[1.0, 1.0]]))
    torch.testing.assert_close(bounds.encode(200, 40), torch.tensor([[0.0, 0.0]]))


def test_tiny_coordinate_run_updates_conditioner_and_resumes(tmp_path: Path) -> None:
    source = make_source_checkpoint(tmp_path)
    output = tmp_path / "run"

    report = run(experiment_args(source, output, None))

    assert report["status"] == "failed"
    assert report["epoch"] == 1
    assert report["conditioning"]["final_layer_zero_initialized"] is True
    checkpoint = torch.load(
        output / "checkpoints" / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    final_weight = checkpoint["model"]["coordinate_mlp.2.weight"]
    assert bool((final_weight != 0).any())
    adam_steps = {
        int(state["step"].item())
        for state in checkpoint["optimizer"]["state"].values()
        if "step" in state
    }
    assert adam_steps == {1}

    resumed = run(
        experiment_args(source, output, output / "checkpoints" / "latest.pt")
    )
    assert resumed["epoch"] == 1


def test_validate_args_rejects_short_budget_before_source_check(tmp_path: Path) -> None:
    args = experiment_args(tmp_path / "missing.pt", tmp_path / "run", None)
    args.epochs = 2
    args.required_consecutive_successes = 3

    with pytest.raises(ValueError, match="required minimum=3"):
        validate_args(args)
