from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from scripts.overfit_phase_correction_patch_set import (
    ema_decay_with_warmup,
    run,
    summarize_metric_map,
)
from scripts.overfit_single_phase_correction import PhaseSuccessCriteria


def _phase_pair(shape: tuple[int, int], seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    row = np.fft.fftfreq(shape[0])[:, None]
    col = np.fft.fftfreq(shape[1])[None, :]
    correction = np.exp(
        1j * (0.2 * seed + (seed + 1.0) * row - (seed + 0.5) * col)
    )
    image = np.fft.ifft2(correction * np.fft.fft2(echo, norm="ortho"), norm="ortho")
    return echo, image


def _write_config(path: Path) -> None:
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
            "drop_path_rate": 0.1,
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
        "evaluation": {
            "log_magnitude_floor_db": -60.0,
            "high_frequency_radius_fraction": 0.25,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _metric(phase: float) -> dict[str, float]:
    return {
        "weighted_phase_alignment": phase,
        "normalized_complex_rmse": 0.45,
        "rmse_excess_over_oracle": 0.02,
        "complex_coherence": 0.8,
        "coherence_fraction_of_oracle": 0.98,
        "log_magnitude_ssim": 0.35,
        "ssim_gain_fraction_of_oracle": 1.1,
        "edge_correlation": 0.2,
        "edge_gain_fraction_of_oracle": 1.0,
        "high_frequency_energy_ratio": 1.0,
    }


def test_patch_set_summary_requires_every_sample_to_pass() -> None:
    criteria = PhaseSuccessCriteria()
    summary = summarize_metric_map(
        {"first.mat": _metric(0.98), "second.mat": _metric(0.94)}, criteria
    )

    assert summary["pass_count"] == 1
    assert not summary["all_passed"]
    assert summary["worst_sample_by_phase_alignment"] == "second.mat"
    assert summary["aggregate"]["weighted_phase_alignment"]["min"] == pytest.approx(
        0.94
    )


def test_ema_decay_warmup_tracks_early_raw_weights() -> None:
    assert ema_decay_with_warmup(0, 0.999) == pytest.approx(0.0)
    assert ema_decay_with_warmup(1, 0.999) == pytest.approx(0.5)
    assert ema_decay_with_warmup(999, 0.999) == pytest.approx(0.999)
    assert ema_decay_with_warmup(9999, 0.99) == pytest.approx(0.99)


def _args(
    config: Path,
    echo_dir: Path,
    image_dir: Path,
    output_dir: Path,
    resume: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        anchor_filename="patch_row_17500_col_9400_2.mat",
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
        success_phase_alignment=1.0,
        success_coherence_fraction=1.0,
        success_ssim_gain_fraction=1.0,
        success_edge_gain_fraction=1.0,
        success_rmse_excess=0.0,
        success_hf_ratio_min=1.0,
        success_hf_ratio_max=1.0,
    )


def test_tiny_joint_phase_run_writes_artifacts_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_config(config)
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    names = (
        "patch_row_17500_col_9400_2.mat",
        "patch_row_10000_col_10000_2.mat",
    )
    for seed, name in enumerate(names, start=1):
        echo, image = _phase_pair((16, 16), seed)
        savemat(echo_dir / name, {"patch": echo})
        savemat(image_dir / name, {"patch": image})
    output_dir = tmp_path / "run"

    first = run(_args(config, echo_dir, image_dir, output_dir, None))

    assert first["status"] == "failed"
    assert first["selection_manifest"]["sample_count"] == 2
    assert first["inference_contract"].startswith("Echo spectrum")
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "selected_samples.json").is_file()
    assert list((output_dir / "figures").rglob("step_000002.png"))

    resumed = run(
        _args(
            config,
            echo_dir,
            image_dir,
            output_dir,
            output_dir / "checkpoints" / "latest.pt",
        )
    )

    assert resumed["step"] == 2
    assert resumed["status"] == "failed"
