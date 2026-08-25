from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.io import savemat

from scripts.overfit_phase_train_subset import EXPERIMENT, run


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
