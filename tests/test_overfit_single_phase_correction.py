from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat

from scripts.overfit_single_phase_correction import (
    PhaseSuccessCriteria,
    _gain_fraction,
    apply_phase_correction,
    normalize_phasor,
    phase_loss_components,
    prepare_phase_supervision,
    run,
)


def _phase_only_pair(
    shape: tuple[int, int], seed: int = 7
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    row = np.fft.fftfreq(shape[0])[:, None]
    col = np.fft.fftfreq(shape[1])[None, :]
    correction = np.exp(1j * (0.4 - 2.0 * row + 1.2 * col + 0.7 * row * col))
    spectrum = np.fft.fft2(echo, norm="ortho")
    image = np.fft.ifft2(correction * spectrum, norm="ortho")
    return echo, image, correction


def test_phase_supervision_and_oracle_reconstruct_known_mapping() -> None:
    echo, image, correction = _phase_only_pair((16, 16))

    inputs, target, weights, target_image, scale = prepare_phase_supervision(
        echo,
        image,
        rms_epsilon=1.0e-12,
        phasor_epsilon=1.0e-8,
        energy_weight_power=0.5,
        fft_norm="ortho",
    )
    expected = np.fft.fftshift(correction)
    np.testing.assert_allclose(
        target[0].numpy() + 1j * target[1].numpy(), expected, atol=1.0e-6
    )
    prediction = apply_phase_correction(
        inputs.unsqueeze(0), target.unsqueeze(0), fft_norm="ortho"
    )
    predicted_complex = prediction[0, 0].numpy() + 1j * prediction[0, 1].numpy()

    np.testing.assert_allclose(predicted_complex, image / scale, atol=2.0e-6)
    assert weights.shape == (1, 16, 16)
    assert float(weights.mean()) == pytest.approx(1.0, rel=1.0e-6)


def test_phase_loss_is_finite_and_backpropagates() -> None:
    echo, image, _ = _phase_only_pair((16, 16))
    inputs, target, weights, target_image, _ = prepare_phase_supervision(
        echo,
        image,
        rms_epsilon=1.0e-12,
        phasor_epsilon=1.0e-8,
        energy_weight_power=0.5,
        fft_norm="ortho",
    )
    raw = (target.unsqueeze(0) + 0.1 * torch.randn(1, 2, 16, 16)).requires_grad_(True)
    correction = normalize_phasor(raw, 1.0e-6)
    prediction = apply_phase_correction(
        inputs.unsqueeze(0), correction, fft_norm="ortho"
    )
    losses = phase_loss_components(
        correction,
        target.unsqueeze(0),
        weights.unsqueeze(0),
        prediction,
        target_image.unsqueeze(0),
        phase_loss_weight=1.0,
        complex_reconstruction_weight=0.25,
        log_magnitude_weight=0.25,
        charbonnier_epsilon=1.0e-3,
    )

    losses["total"].backward()

    assert all(torch.isfinite(value) for value in losses.values())
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())


def test_phase_success_criteria_require_every_metric() -> None:
    criteria = PhaseSuccessCriteria()
    passing = {
        "weighted_phase_alignment": 0.96,
        "coherence_fraction_of_oracle": 0.91,
        "ssim_gain_fraction_of_oracle": 0.81,
        "edge_gain_fraction_of_oracle": 0.76,
        "rmse_excess_over_oracle": 0.07,
        "high_frequency_energy_ratio": 1.0,
    }
    assert criteria.is_satisfied(passing)
    for name in passing:
        failing = dict(passing)
        if name == "rmse_excess_over_oracle":
            failing[name] = 1.0
        elif name == "high_frequency_energy_ratio":
            failing[name] = 0.1
        else:
            failing[name] = 0.0
        assert not criteria.is_satisfied(failing), name


def test_gain_fraction_handles_non_positive_oracle_gain() -> None:
    assert _gain_fraction(0.2, 0.3, 0.25) == pytest.approx(0.0)
    assert _gain_fraction(0.25, 0.3, 0.25) == pytest.approx(1.0)
    assert _gain_fraction(0.4, 0.3, 0.5) == pytest.approx(0.5)


def _write_tiny_config(path: Path) -> None:
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


def _args(
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
        success_phase_alignment=1.0,
        success_coherence_fraction=1.0,
        success_ssim_gain_fraction=1.0,
        success_edge_gain_fraction=1.0,
        success_rmse_excess=0.0,
        success_hf_ratio_min=1.0,
        success_hf_ratio_max=1.0,
    )


def test_tiny_cpu_run_writes_artifacts_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    _write_tiny_config(config)
    echo, image, _ = _phase_only_pair((16, 16), seed=19)
    name = "patch_row_17500_col_9400_2.mat"
    echo_file = tmp_path / "echo" / name
    image_file = tmp_path / "image" / name
    echo_file.parent.mkdir()
    image_file.parent.mkdir()
    savemat(echo_file, {"patch": echo})
    savemat(image_file, {"patch": image})
    output_dir = tmp_path / "run"

    first = run(_args(config, echo_file, image_file, output_dir, None))

    assert first["status"] == "failed"
    assert first["inference_contract"].startswith("Echo spectrum")
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "figures" / "step_000001.png").is_file()
    assert (output_dir / "predictions" / "step_000001.mat").is_file()

    resumed = run(
        _args(
            config,
            echo_file,
            image_file,
            output_dir,
            output_dir / "checkpoints" / "latest.pt",
        )
    )

    assert resumed["step"] == 1
    assert resumed["status"] == "failed"
