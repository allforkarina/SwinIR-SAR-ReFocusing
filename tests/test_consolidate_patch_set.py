from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from scripts.consolidate_patch_set import run, shuffled_epoch_order, validate_args
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


def write_pair(echo_path: Path, image_path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (0.8 + 0.1j) * echo
    savemat(echo_path, {"patch": echo})
    savemat(image_path, {"patch": image})


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
        write_pair(echo_path, image_path, index + 1)
        pairs.append(DiscoveredPair(row, col, echo_path, image_path))
    samples = load_selected_samples(pairs, expected_shape=(16, 16), rms_epsilon=1.0e-12)
    manifest = selection_manifest(
        samples,
        echo_dir=echo_dir,
        image_dir=image_dir,
        patch_shape=(16, 16),
        anchor_filename=pairs[0].echo_path.name,
    )
    impossible = SuccessCriteria(
        normalized_complex_rmse_max=0.0,
        complex_coherence_min=1.0,
        magnitude_correlation_min=1.0,
        rms_ratio_min=1.0,
        rms_ratio_max=1.0,
        log_magnitude_psnr_db_min=300.0,
        log_magnitude_ssim_min=1.0,
    )
    model_config = tiny_model_config()
    model = SwinIR(**model_config)
    source_config = {
        "experiment": "joint_patch_set_overfit",
        "selection_manifest": manifest,
        "model": model_config,
        "data": {"expected_shape": [16, 16], "rms_epsilon": 1.0e-12},
        "optimization": {"charbonnier_epsilon": 1.0e-3},
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "success_criteria": asdict(impossible),
        },
    }
    source_path = tmp_path / "source.pt"
    torch.save(
        {
            "schema_version": 1,
            "resolved_config": source_config,
            "ema_model": model.state_dict(),
        },
        source_path,
    )
    return source_path


def experiment_args(
    *,
    source_checkpoint: Path,
    output_dir: Path,
    resume: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        source_checkpoint=source_checkpoint,
        output_dir=output_dir,
        resume=resume,
        device="cpu",
        epochs=1,
        eval_every=1,
        save_every=1,
        required_consecutive_successes=1,
        seed=42,
        learning_rate=5.0e-5,
        ema_decay=0.99,
    )


def test_shuffled_epoch_order_contains_every_sample_once() -> None:
    order = shuffled_epoch_order(epoch=3, sample_count=16, seed=42)

    assert len(order) == 16
    assert sorted(order) == list(range(16))
    assert order == shuffled_epoch_order(epoch=3, sample_count=16, seed=42)


def test_validate_args_rejects_insufficient_evaluation_budget(tmp_path: Path) -> None:
    args = experiment_args(
        source_checkpoint=tmp_path / "missing.pt",
        output_dir=tmp_path / "run",
        resume=None,
    )
    args.epochs = 2
    args.eval_every = 1
    args.required_consecutive_successes = 3

    with pytest.raises(ValueError, match="required minimum=3"):
        validate_args(args)


def test_tiny_consolidation_uses_one_adam_step_and_resumes(tmp_path: Path) -> None:
    source_checkpoint = make_source_checkpoint(tmp_path)
    output_dir = tmp_path / "run"

    report = run(
        experiment_args(
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            resume=None,
        )
    )

    assert report["status"] == "failed"
    assert report["epoch"] == 1
    assert report["optimizer_steps"] == 1
    assert report["sample_presentations"] == 2
    assert report["source"]["initial_weights"] == "ema_model"
    checkpoint = torch.load(
        output_dir / "checkpoints" / "final.pt",
        map_location="cpu",
        weights_only=False,
    )
    adam_steps = {
        int(state["step"].item())
        for state in checkpoint["optimizer"]["state"].values()
        if "step" in state
    }
    assert adam_steps == {1}
    assert (output_dir / "selected_samples.json").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()

    resumed = run(
        experiment_args(
            source_checkpoint=source_checkpoint,
            output_dir=output_dir,
            resume=output_dir / "checkpoints" / "latest.pt",
        )
    )
    assert resumed["epoch"] == 1
