from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from scipy.io import savemat

from scripts.evaluate_phase_unseen_checkpoint import run, select_representatives
from scripts.overfit_single_patch import sample_fingerprint
from swinir import SwinIR
from swinir.sar_dataset import CoordinateRegion, SplitName, build_manifest


def _model_config() -> dict[str, object]:
    return {
        "img_size": 16, "patch_size": 1, "in_chans": 2, "embed_dim": 12,
        "depths": [1], "num_heads": [3], "window_size": 4, "mlp_ratio": 2.0,
        "qkv_bias": True, "qk_scale": None, "drop_rate": 0.0, "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0, "ape": False, "patch_norm": True,
        "use_checkpoint": False, "upscale": 1, "img_range": 1.0,
        "upsampler": "", "resi_connection": "1conv",
    }


def _pair(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    row = np.fft.fftfreq(16)[:, None]
    col = np.fft.fftfreq(16)[None, :]
    image = np.fft.ifft2(np.exp(1j * (0.1 * seed + row - col)) * np.fft.fft2(echo, norm="ortho"), norm="ortho")
    return echo, image


def _grid(echo_dir: Path, image_dir: Path) -> None:
    echo_dir.mkdir(); image_dir.mkdir()
    for seed, (row, col) in enumerate(((0, 0), (0, 20), (40, 40)), start=1):
        echo, image = _pair(seed)
        name = f"patch_row_{row}_col_{col}_2.mat"
        savemat(echo_dir / name, {"patch": echo})
        savemat(image_dir / name, {"patch": image})


def _config(path: Path) -> None:
    value = {
        "experiment": "E012-D001-frozen-e011b-unseen-spatial-evaluation",
        "checkpoint": {"expected_experiment": "E011-B-D001-controlled-64-train-overfit", "expected_step": 1, "expected_sample_count": 2, "require_dataset_manifest_fingerprint": True},
        "data": {"expected_shape": [16, 16], "rms_epsilon": 1e-12, "fft_norm": "ortho", "representation": "fftshifted_echo_complex_spectrum_to_unit_phase_correction"},
        "selection": {"source_split": "validation", "validation_region": {"row_min": 40, "row_max": 40, "col_min": 40, "col_max": 40}, "guard_region": {"row_min": 24, "row_max": 56, "col_min": 24, "col_max": 56}, "expected_split_counts": {"train": 2, "guard": 0, "validation": 1}},
        "evaluation": {"authority": "raw_frozen_e011b_checkpoint_on_unseen_spatial_holdout", "log_magnitude_floor_db": -60.0, "high_frequency_radius_fraction": 0.25, "success_criteria": {"mean_phase_alignment_min": 0.5, "median_phase_alignment_min": 0.5, "p05_phase_alignment_min": 0.2, "mean_rmse_oracle_gap_fraction_closed_min": 0.5, "median_rmse_oracle_gap_fraction_closed_min": 0.5, "rmse_win_fraction_vs_echo_min": 0.9, "mean_coherence_fraction_of_oracle_min": 0.5, "mean_ssim_gain_fraction_of_oracle_min": 0.5, "mean_edge_gain_fraction_of_oracle_min": 0.4, "median_high_frequency_energy_ratio_min": 0.75, "median_high_frequency_energy_ratio_max": 1.25}},
        "output": {"representative_sample_count": 1, "figure_dpi": 20},
    }
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _checkpoint(path: Path, echo_dir: Path, image_dir: Path) -> None:
    manifest = build_manifest(echo_dir, image_dir, CoordinateRegion(40, 40, 40, 40), CoordinateRegion(24, 56, 24, 56), expected_counts={"train": 2, "guard": 0, "validation": 1})
    samples = []
    for index, record in enumerate(manifest.records_for(SplitName.TRAIN)):
        samples.append({"selection_index": index, "filename": record.echo_path.name, "row": record.row, "col": record.col, **sample_fingerprint(record.echo_path, record.image_path)})
    model = SwinIR(**_model_config())
    resolved = {"experiment": "E011-B-D001-controlled-64-train-overfit", "model": _model_config(), "data": {"expected_shape": [16, 16], "rms_epsilon": 1e-12, "fft_norm": "ortho", "representation": "fftshifted_echo_complex_spectrum_to_unit_phase_correction"}, "optimization": {"phasor_epsilon": 1e-6, "phase_energy_weight_power": 0.5}, "selection_manifest": {"dataset_manifest_fingerprint": manifest.fingerprint, "samples": samples, "fingerprint": "test-selection"}}
    torch.save({"schema_version": 1, "step": 1, "model": model.state_dict(), "ema_model": model.state_dict(), "last_metrics": {"step": 1}, "resolved_config": resolved}, path)


def _args(config: Path, checkpoint: Path, echo_dir: Path, image_dir: Path, output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(config=config, checkpoint=checkpoint, echo_dir=echo_dir, image_dir=image_dir, output_dir=output_dir, device="cpu")


def test_frozen_checkpoint_audit_writes_report_and_figure(tmp_path: Path) -> None:
    echo_dir, image_dir = tmp_path / "echo", tmp_path / "image"
    _grid(echo_dir, image_dir)
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "best.pt"
    _config(config); _checkpoint(checkpoint, echo_dir, image_dir)
    report = run(_args(config, checkpoint, echo_dir, image_dir, tmp_path / "audit"))
    assert report["checkpoint"]["step"] == 1
    assert report["evaluation_sample_count"] == 1
    assert report["training"] == "forbidden: checkpoint is frozen and no optimizer is created"
    assert (tmp_path / "audit" / "report.json").is_file()
    assert len(tuple((tmp_path / "audit" / "representative_samples").glob("*.png"))) == 1


def test_rejects_checkpoint_at_an_unexpected_step(tmp_path: Path) -> None:
    echo_dir, image_dir = tmp_path / "echo", tmp_path / "image"
    _grid(echo_dir, image_dir)
    config, checkpoint = tmp_path / "config.yaml", tmp_path / "best.pt"
    _config(config); _checkpoint(checkpoint, echo_dir, image_dir)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["step"] = 2
    payload["last_metrics"]["step"] = 2
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="step does not match"):
        run(_args(config, checkpoint, echo_dir, image_dir, tmp_path / "audit"))


def test_representatives_include_metric_extrema() -> None:
    metrics = {f"sample_{index}.mat": {"weighted_phase_alignment": float(index), "rmse_oracle_gap_fraction_closed": float(index)} for index in range(4)}
    coordinates = {name: (index, index) for index, name in enumerate(metrics)}
    selected = select_representatives(metrics, coordinates, 4)
    assert "sample_0.mat" in selected
    assert "sample_3.mat" in selected
