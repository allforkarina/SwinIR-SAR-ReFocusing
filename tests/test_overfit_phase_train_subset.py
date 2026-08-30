from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from scripts.overfit_phase_train_subset import EXPERIMENT, run, validate_profile
from scripts.overfit_single_patch import load_base_config
from scripts.visualize_phase_train_subset_checkpoint import run as visualize_run


def _phase_pair(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    row = np.fft.fftfreq(shape[0])[:, None]
    col = np.fft.fftfreq(shape[1])[None, :]
    correction = np.exp(1j * (0.2 * seed + row - 0.5 * col))
    image = np.fft.ifft2(correction * np.fft.fft2(echo, norm="ortho"), norm="ortho")
    return echo, image


def _write_config(path: Path) -> None:
    config = {
        "experiment": EXPERIMENT,
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
            "drop_path_rate": 0.0,
            "ape": False,
            "patch_norm": True,
            "use_checkpoint": False,
            "upscale": 1,
            "img_range": 1.0,
            "upsampler": "",
            "resi_connection": "1conv",
        },
        "data": {
            "expected_shape": [16, 16],
            "rms_epsilon": 1.0e-12,
            "fft_norm": "ortho",
            "representation": "test",
        },
        "selection": {
            "source_split": "train",
            "sample_count": 2,
            "anchor_filename": "patch_row_0_col_0_2.mat",
            "validation_region": {
                "row_min": 100,
                "row_max": 100,
                "col_min": 100,
                "col_max": 100,
            },
            "guard_region": {
                "row_min": 100,
                "row_max": 100,
                "col_min": 100,
                "col_max": 100,
            },
            "expected_split_counts": {"train": 2, "guard": 0, "validation": 1},
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": 2.0e-4,
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
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
            "steps": 2,
            "eval_every": 2,
            "save_every": 2,
            "required_consecutive_successes": 1,
        },
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "high_frequency_radius_fraction": 0.25,
            "success_criteria": {
                "weighted_phase_alignment_min": 1.0,
                "coherence_fraction_of_oracle_min": 1.0,
                "ssim_gain_fraction_of_oracle_min": 1.0,
                "edge_gain_fraction_of_oracle_min": 1.0,
                "rmse_excess_over_oracle_max": 0.0,
                "high_frequency_energy_ratio_min": 1.0,
                "high_frequency_energy_ratio_max": 1.0,
            },
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def test_e011b_uses_only_train_records_and_writes_all_audits(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    names = (
        "patch_row_0_col_0_2.mat",
        "patch_row_0_col_100_2.mat",
        "patch_row_100_col_100_2.mat",
    )
    for seed, name in enumerate(names, start=1):
        echo, image = _phase_pair((16, 16), seed)
        savemat(echo_dir / name, {"patch": echo})
        savemat(image_dir / name, {"patch": image})
    output_dir = tmp_path / "run"

    report = run(
        argparse.Namespace(
            config=config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            resume=None,
            device="cpu",
        )
    )

    selection = report["selection_manifest"]
    selected_names = {sample["filename"] for sample in selection["samples"]}
    assert report["experiment"] == EXPERIMENT
    assert selection["source_split"] == "train"
    assert selection["candidate_count"] == 2
    assert selection["spatial_split"]["split_counts"] == {
        "train": 2,
        "guard": 0,
        "validation": 1,
    }
    assert selected_names == set(names[:2])
    assert names[2] not in selected_names
    assert report["step"] == 2
    assert len(report["artifacts"]["final_audit_samples"]) == 2
    assert len(list((output_dir / "figures").rglob("step_000002.png"))) == 2
    assert (output_dir / "checkpoints" / "latest.pt").is_file()

    audit_dir = tmp_path / "audit"
    audit = visualize_run(
        argparse.Namespace(
            checkpoint=output_dir / "checkpoints" / "best.pt",
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=audit_dir,
            device="cpu",
            dpi=30,
            contact_sheet_page_size=1,
        )
    )

    assert audit["experiment"] == EXPERIMENT
    assert audit["checkpoint_step"] == audit["stored_metrics_step"]
    assert audit["sample_count"] == 2
    assert len(audit["samples"]) == 2
    assert len(audit["contact_sheets"]) == 2
    assert len(list((audit_dir / "samples").glob("*.png"))) == 2
    assert len(list((audit_dir / "contact_sheets").glob("*.png"))) == 2
    assert (audit_dir / "audit_manifest.json").is_file()


@pytest.mark.parametrize(
    ("path", "sample_count"),
    (
        (Path("configs/train_phase_train_subset_128.yaml"), 128),
        (Path("configs/train_phase_train_subset_512.yaml"), 512),
    ),
)
def test_curriculum_profiles_are_valid(path: Path, sample_count: int) -> None:
    config = load_base_config(path)
    validate_profile(config)
    assert config["selection"]["sample_count"] == sample_count
    assert config["runtime"]["stop_on_success"] is False
    assert set(config["evaluation"]["artifact_sample_indices"]).issubset(
        config["evaluation"]["probe_sample_indices"]
    )


def test_tiny_curriculum_run_restarts_local_step_from_nested_weights(
    tmp_path: Path,
) -> None:
    source_config = tmp_path / "source.yaml"
    _write_config(source_config)
    source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    source["selection"]["expected_split_counts"]["train"] = 3
    source_config.write_text(
        yaml.safe_dump(source, sort_keys=False), encoding="utf-8"
    )
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    names = (
        "patch_row_0_col_0_2.mat",
        "patch_row_0_col_100_2.mat",
        "patch_row_0_col_200_2.mat",
        "patch_row_100_col_100_2.mat",
    )
    for seed, name in enumerate(names, start=1):
        echo, image = _phase_pair((16, 16), seed)
        savemat(echo_dir / name, {"patch": echo})
        savemat(image_dir / name, {"patch": image})
    source_output = tmp_path / "source_run"
    source_report = run(
        argparse.Namespace(
            config=source_config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=source_output,
            resume=None,
            init_checkpoint=None,
            device="cpu",
        )
    )

    target = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    target["experiment"] = "E013-A-D001-curriculum-128-phase-subset"
    target["selection"]["sample_count"] = 3
    target["initialization"] = {
        "mode": "raw_and_ema_weights_only",
        "expected_source_experiment": EXPERIMENT,
        "expected_source_step": 2,
        "expected_source_sample_count": 2,
    }
    target["runtime"].update(
        {
            "steps": 3,
            "eval_every": 3,
            "save_every": 3,
            "stop_on_success": False,
        }
    )
    target["evaluation"]["probe_sample_indices"] = [0, 2]
    target["evaluation"]["artifact_sample_indices"] = [0, 2]
    target_config = tmp_path / "target.yaml"
    target_config.write_text(
        yaml.safe_dump(target, sort_keys=False), encoding="utf-8"
    )
    target_output = tmp_path / "target_run"
    target_report = run(
        argparse.Namespace(
            config=target_config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=target_output,
            resume=None,
            init_checkpoint=source_output / "checkpoints" / "final.pt",
            device="cpu",
        )
    )

    assert target_report["step"] == 3
    assert target_report["training_sample_count"] == 3
    assert target_report["evaluation_sample_count"] == 2
    assert target_report["initialization"]["source_step"] == 2
    assert target_report["initialization"]["optimizer_restored"] is False
    source_names = [
        sample["filename"] for sample in source_report["selection_manifest"]["samples"]
    ]
    target_names = [
        sample["filename"] for sample in target_report["selection_manifest"]["samples"]
    ]
    assert source_names == target_names[:2]
