"""E010: train supervised phase refocusing on a strict spatial holdout."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from main import (
    EarlyStoppingState,
    build_loader,
    configure_logger,
    set_seed,
    write_crash_report,
)
from scripts.overfit_phase_correction_patch_set import ema_decay_with_warmup
from scripts.overfit_single_patch import append_jsonl, utc_now, write_json
from scripts.overfit_single_phase_correction import (
    add_oracle_relative_metrics,
    evaluate_correction,
    predict_correction,
    prepare_phase_supervision,
    train_phase_step,
)
from scripts.train_magnitude_spatial_holdout import RunPaths, load_config, make_run_paths
from swinir import SwinIR
from swinir.sar_dataset import (
    CoordinateRegion,
    DatasetIntegrityError,
    PairRecord,
    ResumableEpochSampler,
    SplitName,
    build_manifest,
    load_complex_patch,
)
from swinir.training import (
    PrecisionPolicy,
    atomic_torch_save,
    capture_rng_state,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    restore_rng_state,
)


CHECKPOINT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
METRIC_NAMES = (
    "weighted_phase_alignment",
    "normalized_complex_rmse",
    "rmse_excess_over_oracle",
    "rmse_oracle_gap_fraction_closed",
    "complex_coherence",
    "coherence_fraction_of_oracle",
    "magnitude_correlation",
    "log_magnitude_psnr_db",
    "log_magnitude_ssim",
    "ssim_gain_fraction_of_oracle",
    "edge_correlation",
    "edge_gain_fraction_of_oracle",
    "gradient_energy_ratio",
    "high_frequency_energy_ratio",
)


@dataclass(frozen=True)
class ValidationBaselines:
    echo_identity: dict[str, Any]
    unrestricted_phase_oracle: dict[str, Any]


class PhasePatchDataset(Dataset[dict[str, Any]]):
    """Load Echo spectrum and Image-supervised phase targets for one split."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        expected_shape: tuple[int, int],
        data_config: dict[str, Any],
        optimization: dict[str, Any],
    ) -> None:
        self.records = tuple(records)
        self.expected_shape = expected_shape
        self.data_config = data_config
        self.optimization = optimization
        if not self.records:
            raise ValueError("PhasePatchDataset requires at least one record")
        if any(record.split is SplitName.GUARD for record in self.records):
            raise ValueError("guard records must not be exposed through a Dataset")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            echo = load_complex_patch(record.echo_path, self.expected_shape)
            image = load_complex_patch(record.image_path, self.expected_shape)
            inputs, target, weights, target_image, scale = prepare_phase_supervision(
                echo,
                image,
                rms_epsilon=float(self.data_config["rms_epsilon"]),
                phasor_epsilon=float(self.optimization["phasor_epsilon"]),
                energy_weight_power=float(
                    self.optimization["phase_energy_weight_power"]
                ),
                fft_norm=str(self.data_config["fft_norm"]),
            )
        except Exception as error:
            detail = str(error) if isinstance(error, DatasetIntegrityError) else repr(error)
            raise DatasetIntegrityError(
                f"dataset index={index}, key={record.key}, row={record.row}, "
                f"col={record.col}, echo={record.echo_path}, image={record.image_path}: "
                f"{detail}"
            ) from error
        return {
            "input_spectrum": inputs,
            "target_phasor": target,
            "phase_weights": weights,
            "target_image": target_image,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "key": record.key,
            "filename": record.echo_path.name,
            "row": record.row,
            "col": record.col,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Echo-only phase refocusing on a strict spatial holdout."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_phase_spatial_holdout.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    if int(model.get("in_chans", 0)) != 2:
        raise ValueError("phase correction requires model.in_chans=2")
    if float(model.get("drop_path_rate", 0.0)) != 0.0:
        raise ValueError("E010 requires drop_path_rate=0")
    if model.get("use_checkpoint"):
        raise ValueError("activation checkpointing is disabled for E010")
    if data.get("representation") != (
        "fftshifted_echo_complex_spectrum_to_unit_phase_correction"
    ):
        raise ValueError("unsupported E010 phase representation")
    if data.get("fft_norm") not in ("ortho", "backward", "forward"):
        raise ValueError("unsupported FFT normalization")
    if optimization.get("optimizer", "").lower() != "adam":
        raise ValueError("E010 uses Adam")
    if optimization.get("learning_rate_schedule") != "constant":
        raise ValueError("E010 uses a constant learning rate")
    for name in (
        "learning_rate",
        "phase_loss_weight",
        "complex_reconstruction_weight",
        "log_magnitude_weight",
        "phase_energy_weight_power",
        "phasor_epsilon",
        "charbonnier_epsilon",
        "ema_decay",
    ):
        value = float(optimization[name])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"optimization.{name} must be finite and non-negative")
    if float(optimization["learning_rate"]) <= 0:
        raise ValueError("learning rate must be positive")
    for name in ("phasor_epsilon", "charbonnier_epsilon"):
        if float(optimization[name]) <= 0:
            raise ValueError(f"optimization.{name} must be positive")
    if not any(
        float(optimization[name]) > 0
        for name in (
            "phase_loss_weight",
            "complex_reconstruction_weight",
            "log_magnitude_weight",
        )
    ):
        raise ValueError("at least one phase-reconstruction loss weight must be positive")
    if not 0.0 <= float(optimization["phase_energy_weight_power"]) <= 1.0:
        raise ValueError("phase energy weight power must be in [0, 1]")
    if not 0.0 <= float(optimization["ema_decay"]) < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    if int(optimization.get("total_steps", 0)) <= 0:
        raise ValueError("optimization.total_steps must be positive")
    if int(runtime.get("batch_size", 0)) != 1:
        raise ValueError("E010 fixes physical batch_size=1")
    for name in (
        "validation_interval_steps",
        "archive_interval_steps",
        "log_interval_steps",
        "max_consecutive_fp16_overflows",
        "required_consecutive_successes",
    ):
        if int(runtime.get(name, 0)) <= 0:
            raise ValueError(f"runtime.{name} must be positive")
    if int(runtime["archive_interval_steps"]) % int(
        runtime["validation_interval_steps"]
    ):
        raise ValueError("archive interval must be divisible by validation interval")
    early = runtime.get("early_stopping")
    if not isinstance(early, dict) or int(early.get("patience", 0)) <= 0:
        raise ValueError("runtime.early_stopping requires positive patience")
    if float(early.get("min_delta", -1.0)) < 0:
        raise ValueError("early-stopping min_delta must be non-negative")
    criteria = evaluation.get("success_criteria")
    required_criteria = (
        "mean_phase_alignment_min",
        "median_phase_alignment_min",
        "p05_phase_alignment_min",
        "mean_rmse_oracle_gap_fraction_closed_min",
        "median_rmse_oracle_gap_fraction_closed_min",
        "rmse_win_fraction_vs_echo_min",
        "mean_coherence_fraction_of_oracle_min",
        "mean_ssim_gain_fraction_of_oracle_min",
        "mean_edge_gain_fraction_of_oracle_min",
        "median_high_frequency_energy_ratio_min",
        "median_high_frequency_energy_ratio_max",
    )
    if not isinstance(criteria, dict):
        raise ValueError("evaluation.success_criteria must be a mapping")
    for name in required_criteria:
        if not math.isfinite(float(criteria[name])):
            raise ValueError(f"success criterion {name} must be finite")
    for name in required_criteria[:-2]:
        if not 0.0 <= float(criteria[name]) <= 1.0:
            raise ValueError(f"success criterion {name} must be in [0, 1]")
    if not 0.0 < float(criteria[required_criteria[-2]]) <= float(
        criteria[required_criteria[-1]]
    ):
        raise ValueError("high-frequency bounds must be positive and ordered")


def resolved_config(
    config: dict[str, Any], args: argparse.Namespace, precision: PrecisionPolicy
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["schema_version"] = 1
    result["experiment"] = "E010-D001-phase-spatial-holdout"
    result["config_file"] = str(args.config.resolve())
    result["data"].update(
        {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "input_available_at_inference": (
                "fftshift(FFT2(Echo / RMS(Echo))) only"
            ),
            "training_target": "Image-supervised unit phase correction",
            "label_available_at_inference": False,
        }
    )
    result["optimization"]["ema_schedule"] = (
        "min(target_decay, 1 - 1 / successful_update)"
    )
    result["runtime"]["precision"] = precision.as_dict()
    result["runtime"]["initialization"] = (
        "random_seeded_no_E009_checkpoint_to_prevent_validation_leakage"
    )
    return result


def aggregate_metrics(
    per_sample: dict[str, dict[str, float]],
) -> dict[str, Any]:
    if not per_sample:
        raise ValueError("cannot summarize empty validation metrics")
    aggregate: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(
            [metrics[name] for metrics in per_sample.values()], dtype=np.float64
        )
        aggregate[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    worst_gap = min(
        per_sample,
        key=lambda filename: (
            per_sample[filename]["rmse_oracle_gap_fraction_closed"], filename
        ),
    )
    worst_phase = min(
        per_sample,
        key=lambda filename: (
            per_sample[filename]["weighted_phase_alignment"], filename
        ),
    )
    return {
        "sample_count": len(per_sample),
        "aggregate": aggregate,
        "worst_sample_by_rmse_gap_closed": worst_gap,
        "worst_sample_by_phase_alignment": worst_phase,
        "per_sample": per_sample,
    }


def _rmse_gap_fraction_closed(
    candidate: dict[str, float],
    echo: dict[str, float],
    oracle: dict[str, float],
) -> float:
    echo_rmse = float(echo["normalized_complex_rmse"])
    oracle_rmse = float(oracle["normalized_complex_rmse"])
    denominator = echo_rmse - oracle_rmse
    candidate_rmse = float(candidate["normalized_complex_rmse"])
    if denominator > 1.0e-12:
        return (echo_rmse - candidate_rmse) / denominator
    return 1.0 if candidate_rmse <= oracle_rmse + 1.0e-12 else 0.0


def add_generalization_metrics(
    candidate: dict[str, float],
    echo: dict[str, float],
    oracle: dict[str, float],
) -> dict[str, float]:
    result = add_oracle_relative_metrics(candidate, echo, oracle)
    result["rmse_oracle_gap_fraction_closed"] = _rmse_gap_fraction_closed(
        candidate, echo, oracle
    )
    return result


@torch.no_grad()
def evaluate_baselines(
    loader: DataLoader[dict[str, Any]],
    *,
    data_config: dict[str, Any],
    evaluation: dict[str, Any],
) -> ValidationBaselines:
    echo_map: dict[str, dict[str, float]] = {}
    oracle_map: dict[str, dict[str, float]] = {}
    for batch in loader:
        if int(batch["input_spectrum"].shape[0]) != 1:
            raise RuntimeError("validation baseline requires batch_size=1")
        identity = torch.zeros_like(batch["target_phasor"])
        identity[:, 0] = 1.0
        _, echo_metrics = evaluate_correction(
            identity,
            batch["input_spectrum"],
            batch["target_phasor"],
            batch["phase_weights"],
            batch["target_image"],
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        _, oracle_metrics = evaluate_correction(
            batch["target_phasor"],
            batch["input_spectrum"],
            batch["target_phasor"],
            batch["phase_weights"],
            batch["target_image"],
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        filename = str(batch["filename"][0])
        echo_map[filename] = echo_metrics
        oracle_map[filename] = oracle_metrics
    baseline_names = tuple(next(iter(echo_map.values())))
    return ValidationBaselines(
        echo_identity={
            "sample_count": len(echo_map),
            "aggregate": _aggregate_named(echo_map, baseline_names),
            "per_sample": echo_map,
        },
        unrestricted_phase_oracle={
            "sample_count": len(oracle_map),
            "aggregate": _aggregate_named(oracle_map, baseline_names),
            "per_sample": oracle_map,
        },
    )


def _aggregate_named(
    per_sample: dict[str, dict[str, float]], names: Sequence[str]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name in names:
        values = np.asarray([item[name] for item in per_sample.values()])
        result[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
            "p05": float(np.percentile(values, 5)),
            "p95": float(np.percentile(values, 95)),
        }
    return result


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    baselines: ValidationBaselines,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    data_config: dict[str, Any],
    optimization: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    per_sample: dict[str, dict[str, float]] = {}
    for batch in loader:
        if int(batch["input_spectrum"].shape[0]) != 1:
            raise RuntimeError("validation requires batch_size=1")
        correction = predict_correction(
            model,
            batch["input_spectrum"],
            device=device,
            precision=precision,
            phasor_epsilon=float(optimization["phasor_epsilon"]),
        )
        _, metrics = evaluate_correction(
            correction,
            batch["input_spectrum"],
            batch["target_phasor"],
            batch["phase_weights"],
            batch["target_image"],
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        filename = str(batch["filename"][0])
        per_sample[filename] = add_generalization_metrics(
            metrics,
            baselines.echo_identity["per_sample"][filename],
            baselines.unrestricted_phase_oracle["per_sample"][filename],
        )
    return aggregate_metrics(per_sample)


def compare_to_baselines(
    summary: dict[str, Any],
    baselines: ValidationBaselines,
    criteria: dict[str, Any],
) -> dict[str, Any]:
    aggregate = summary["aggregate"]
    echo = baselines.echo_identity["per_sample"]
    model = summary["per_sample"]
    if model.keys() != echo.keys():
        raise RuntimeError("model and baseline validation samples differ")
    rmse_win_fraction = float(
        np.mean(
            [
                model[name]["normalized_complex_rmse"]
                < echo[name]["normalized_complex_rmse"]
                for name in model
            ]
        )
    )
    values = {
        "mean_phase_alignment": aggregate["weighted_phase_alignment"]["mean"],
        "median_phase_alignment": aggregate["weighted_phase_alignment"]["median"],
        "p05_phase_alignment": aggregate["weighted_phase_alignment"]["p05"],
        "mean_rmse_oracle_gap_fraction_closed": aggregate[
            "rmse_oracle_gap_fraction_closed"
        ]["mean"],
        "median_rmse_oracle_gap_fraction_closed": aggregate[
            "rmse_oracle_gap_fraction_closed"
        ]["median"],
        "rmse_win_fraction_vs_echo": rmse_win_fraction,
        "mean_coherence_fraction_of_oracle": aggregate[
            "coherence_fraction_of_oracle"
        ]["mean"],
        "mean_ssim_gain_fraction_of_oracle": aggregate[
            "ssim_gain_fraction_of_oracle"
        ]["mean"],
        "mean_edge_gain_fraction_of_oracle": aggregate[
            "edge_gain_fraction_of_oracle"
        ]["mean"],
        "median_high_frequency_energy_ratio": aggregate[
            "high_frequency_energy_ratio"
        ]["median"],
    }
    checks = {
        "mean_phase_alignment": values["mean_phase_alignment"]
        >= float(criteria["mean_phase_alignment_min"]),
        "median_phase_alignment": values["median_phase_alignment"]
        >= float(criteria["median_phase_alignment_min"]),
        "p05_phase_alignment": values["p05_phase_alignment"]
        >= float(criteria["p05_phase_alignment_min"]),
        "mean_rmse_oracle_gap_fraction_closed": values[
            "mean_rmse_oracle_gap_fraction_closed"
        ]
        >= float(criteria["mean_rmse_oracle_gap_fraction_closed_min"]),
        "median_rmse_oracle_gap_fraction_closed": values[
            "median_rmse_oracle_gap_fraction_closed"
        ]
        >= float(criteria["median_rmse_oracle_gap_fraction_closed_min"]),
        "rmse_win_fraction_vs_echo": values["rmse_win_fraction_vs_echo"]
        >= float(criteria["rmse_win_fraction_vs_echo_min"]),
        "mean_coherence_fraction_of_oracle": values[
            "mean_coherence_fraction_of_oracle"
        ]
        >= float(criteria["mean_coherence_fraction_of_oracle_min"]),
        "mean_ssim_gain_fraction_of_oracle": values[
            "mean_ssim_gain_fraction_of_oracle"
        ]
        >= float(criteria["mean_ssim_gain_fraction_of_oracle_min"]),
        "mean_edge_gain_fraction_of_oracle": values[
            "mean_edge_gain_fraction_of_oracle"
        ]
        >= float(criteria["mean_edge_gain_fraction_of_oracle_min"]),
        "median_high_frequency_energy_ratio": float(
            criteria["median_high_frequency_energy_ratio_min"]
        )
        <= values["median_high_frequency_energy_ratio"]
        <= float(criteria["median_high_frequency_energy_ratio_max"]),
    }
    return {**values, "checks": checks, "passed": all(checks.values())}


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
    epoch: int,
    sample_offset: int,
    early_best_phase_alignment: float,
    best_phase_alignment: float,
    best_passed: bool,
    best_validation: dict[str, Any],
    last_validation: dict[str, Any],
    last_validation_step: int,
    consecutive_successes: int,
    success_step: int | None,
    early_stopping: EarlyStoppingState,
    manifest_fingerprint: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "sample_offset": sample_offset,
        "sampler": {"seed": config["runtime"]["seed"], "epoch": epoch, "start_index": sample_offset},
        "early_best_phase_alignment": early_best_phase_alignment,
        "best_phase_alignment": best_phase_alignment,
        "best_passed": best_passed,
        "best_validation": best_validation,
        "last_validation": last_validation,
        "last_validation_step": last_validation_step,
        "consecutive_successes": consecutive_successes,
        "success_step": success_step,
        "early_stopping": early_stopping.as_dict(),
        "rng": capture_rng_state(),
        "manifest_fingerprint": manifest_fingerprint,
        "resolved_config": config,
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def restore_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    manifest_fingerprint: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported E010 checkpoint schema version")
    if checkpoint.get("manifest_fingerprint") != manifest_fingerprint:
        raise RuntimeError("checkpoint dataset manifest fingerprint does not match")
    if checkpoint.get("resolved_config") != config:
        raise RuntimeError("checkpoint configuration does not exactly match")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_config(config)
    if not args.echo_dir.is_dir() or not args.image_dir.is_dir():
        raise FileNotFoundError("Echo and Image directories must both exist")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    resolved = resolved_config(config, args, precision)
    paths: RunPaths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    logger = configure_logger(paths.train_log)
    data = resolved["data"]
    optimization = resolved["optimization"]
    runtime = resolved["runtime"]
    evaluation = resolved["evaluation"]
    criteria = evaluation["success_criteria"]
    set_seed(int(runtime["seed"]), bool(runtime["strict_reproducibility"]))
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**data["validation_region"]),
        CoordinateRegion(**data["guard_region"]),
        expected_counts=data["expected_split_counts"],
    )
    if args.resume is None:
        write_json(paths.resolved_config, resolved)
        manifest.write_json(paths.manifest)
    else:
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        existing_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError("output directory resolved configuration does not match")
        if existing_manifest.get("fingerprint") != manifest.fingerprint:
            raise RuntimeError("output directory manifest fingerprint does not match")
    expected_shape = tuple(int(value) for value in data["expected_shape"])
    train_dataset = PhasePatchDataset(
        manifest.records_for(SplitName.TRAIN),
        expected_shape=expected_shape,
        data_config=data,
        optimization=optimization,
    )
    validation_dataset = PhasePatchDataset(
        manifest.records_for(SplitName.VALIDATION),
        expected_shape=expected_shape,
        data_config=data,
        optimization=optimization,
    )
    sampler = ResumableEpochSampler(train_dataset, seed=int(runtime["seed"]))
    train_loader = build_loader(
        train_dataset,
        batch_size=1,
        workers=int(data["num_workers"]),
        prefetch_factor=int(data["prefetch_factor"]),
        pin_memory=device.type == "cuda",
        sampler=sampler,
    )
    validation_loader = build_loader(
        validation_dataset,
        batch_size=1,
        workers=int(data["num_workers"]),
        prefetch_factor=int(data["prefetch_factor"]),
        pin_memory=device.type == "cuda",
    )
    model = SwinIR(**resolved["model"]).to(device)
    ema_model = make_ema_model(model).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(float(value) for value in optimization["betas"]),
        eps=float(optimization["epsilon"]),
        weight_decay=float(optimization["weight_decay"]),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = make_grad_scaler(precision)
    loss_config = {
        name: float(optimization[name])
        for name in (
            "phase_loss_weight",
            "complex_reconstruction_weight",
            "log_magnitude_weight",
            "charbonnier_epsilon",
        )
    }
    logger.info(
        "computing Echo and unrestricted phase oracle baselines on %s validation patches",
        len(validation_dataset),
    )
    baselines = evaluate_baselines(
        validation_loader, data_config=data, evaluation=evaluation
    )
    write_json(paths.root / "validation_baselines.json", {
        "echo_identity": baselines.echo_identity,
        "unrestricted_phase_oracle": baselines.unrestricted_phase_oracle,
    })
    global_step = 0
    epoch = 0
    sample_offset = 0
    early_best_phase_alignment = -math.inf
    best_phase_alignment = -math.inf
    best_passed = False
    best_validation: dict[str, Any] = {}
    last_validation: dict[str, Any] = {}
    last_validation_step = -1
    consecutive_successes = 0
    success_step: int | None = None
    early_stopping = EarlyStoppingState()

    def checkpoint_kwargs() -> dict[str, Any]:
        return {
            "model": model,
            "ema_model": ema_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "global_step": global_step,
            "epoch": epoch,
            "sample_offset": sample_offset,
            "early_best_phase_alignment": early_best_phase_alignment,
            "best_phase_alignment": best_phase_alignment,
            "best_passed": best_passed,
            "best_validation": best_validation,
            "last_validation": last_validation,
            "last_validation_step": last_validation_step,
            "consecutive_successes": consecutive_successes,
            "success_step": success_step,
            "early_stopping": early_stopping,
            "manifest_fingerprint": manifest.fingerprint,
            "config": resolved,
        }

    if args.resume is not None:
        checkpoint = restore_checkpoint(
            args.resume,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            manifest_fingerprint=manifest.fingerprint,
            config=resolved,
            device=device,
        )
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint["epoch"])
        sample_offset = int(checkpoint["sample_offset"])
        early_best_phase_alignment = float(checkpoint["early_best_phase_alignment"])
        best_phase_alignment = float(checkpoint["best_phase_alignment"])
        best_passed = bool(checkpoint["best_passed"])
        best_validation = dict(checkpoint["best_validation"])
        last_validation = dict(checkpoint["last_validation"])
        last_validation_step = int(checkpoint["last_validation_step"])
        consecutive_successes = int(checkpoint["consecutive_successes"])
        success_step = checkpoint["success_step"]
        early_state = checkpoint["early_stopping"]
        early_stopping = EarlyStoppingState(
            bad_validation_count=int(early_state["bad_validation_count"]),
            stopped=bool(early_state["stopped"]),
        )
        logger.info(
            "resumed step=%s epoch=%s offset=%s from %s",
            global_step,
            epoch,
            sample_offset,
            args.resume,
        )

    def perform_validation() -> None:
        nonlocal early_best_phase_alignment
        nonlocal best_phase_alignment, best_passed, best_validation
        nonlocal last_validation, last_validation_step
        nonlocal consecutive_successes, success_step
        summary = evaluate_validation(
            model,
            validation_loader,
            baselines,
            device=device,
            precision=precision,
            data_config=data,
            optimization=optimization,
            evaluation=evaluation,
        )
        comparison = compare_to_baselines(summary, baselines, criteria)
        passed = bool(comparison["passed"])
        if global_step > 0:
            consecutive_successes = consecutive_successes + 1 if passed else 0
            if (
                consecutive_successes >= int(runtime["required_consecutive_successes"])
                and success_step is None
            ):
                success_step = global_step
        current_phase = float(comparison["mean_phase_alignment"])
        improved_for_early_stop = current_phase > early_best_phase_alignment + float(
            runtime["early_stopping"]["min_delta"]
        )
        if improved_for_early_stop:
            early_best_phase_alignment = current_phase
            early_stopping.bad_validation_count = 0
        else:
            early_stopping.bad_validation_count += 1
            early_stopping.stopped = early_stopping.bad_validation_count >= int(
                runtime["early_stopping"]["patience"]
            )
        best_improved = (passed and not best_passed) or (
            passed == best_passed and current_phase > best_phase_alignment
        )
        last_validation = {
            "step": global_step,
            "generated_at_utc": utc_now(),
            "summary": summary,
            "comparison": comparison,
            "consecutive_successes": consecutive_successes,
        }
        last_validation_step = global_step
        if best_improved:
            best_phase_alignment = current_phase
            best_passed = passed
            best_validation = last_validation
        append_jsonl(
            paths.metrics,
            {
                "timestamp_utc": utc_now(),
                "step": global_step,
                "split": "validation",
                **last_validation,
            },
        )
        logger.info(
            "validation step=%s phase_mean=%.4f phase_median=%.4f phase_p05=%.4f "
            "rmse_gap_closed_mean=%.4f rmse_win=%.3f coh_frac=%.4f "
            "ssim_gain=%.4f edge_gain=%.4f hf_median=%.4f pass=%s streak=%s",
            global_step,
            comparison["mean_phase_alignment"],
            comparison["median_phase_alignment"],
            comparison["p05_phase_alignment"],
            comparison["mean_rmse_oracle_gap_fraction_closed"],
            comparison["rmse_win_fraction_vs_echo"],
            comparison["mean_coherence_fraction_of_oracle"],
            comparison["mean_ssim_gain_fraction_of_oracle"],
            comparison["mean_edge_gain_fraction_of_oracle"],
            comparison["median_high_frequency_energy_ratio"],
            passed,
            consecutive_successes,
        )
        save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
        if best_improved:
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_kwargs())
        if success_step == global_step:
            save_checkpoint(paths.checkpoints / "passed.pt", **checkpoint_kwargs())
        if global_step > 0 and global_step % int(runtime["archive_interval_steps"]) == 0:
            save_checkpoint(
                paths.checkpoints / f"step_{global_step:06d}.pt", **checkpoint_kwargs()
            )

    if args.resume is None:
        perform_validation()
    interrupted = False
    overflow_streak = 0
    total_steps = int(optimization["total_steps"])
    try:
        while (
            global_step < total_steps
            and not early_stopping.stopped
            and success_step is None
        ):
            sampler.set_position(epoch, sample_offset)
            for batch in train_loader:
                while True:
                    result, losses = train_phase_step(
                        model,
                        ema_model,
                        optimizer,
                        scheduler,
                        scaler,
                        batch["input_spectrum"],
                        batch["target_phasor"],
                        batch["phase_weights"],
                        batch["target_image"],
                        device=device,
                        precision=precision,
                        fft_norm=str(data["fft_norm"]),
                        phasor_epsilon=float(optimization["phasor_epsilon"]),
                        loss_config=loss_config,
                        ema_decay=ema_decay_with_warmup(
                            global_step, float(optimization["ema_decay"])
                        ),
                    )
                    if result.did_optimizer_step:
                        overflow_streak = 0
                        break
                    overflow_streak += 1
                    logger.warning(
                        "mixed-precision overflow before step=%s retry=%s",
                        global_step + 1,
                        overflow_streak,
                    )
                    if overflow_streak >= int(runtime["max_consecutive_fp16_overflows"]):
                        raise FloatingPointError("too many consecutive FP16 overflows")
                sample_offset += int(batch["input_spectrum"].shape[0])
                global_step += 1
                if global_step % int(runtime["log_interval_steps"]) == 0:
                    append_jsonl(
                        paths.metrics,
                        {
                            "timestamp_utc": utc_now(),
                            "step": global_step,
                            "split": "train",
                            "loss": result.loss,
                            "loss_components": losses,
                            "gradient_norm": result.gradient_norm,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "epoch": epoch,
                            "sample_offset": sample_offset,
                        },
                    )
                    logger.info(
                        "train step=%s epoch=%s offset=%s loss=%.6f phase=%.6f "
                        "complex=%.6f logmag=%.6f grad_norm=%.4f",
                        global_step,
                        epoch,
                        sample_offset,
                        losses["total"],
                        losses["circular_phase"],
                        losses["complex_reconstruction"],
                        losses["log_magnitude"],
                        result.gradient_norm,
                    )
                if global_step % int(runtime["validation_interval_steps"]) == 0:
                    perform_validation()
                    if early_stopping.stopped or success_step is not None:
                        break
                if global_step >= total_steps:
                    break
            if sample_offset >= len(train_dataset):
                epoch += 1
                sample_offset = 0
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(paths.checkpoints / "interrupted.pt", **checkpoint_kwargs())
        logger.warning("interrupted checkpoint written at step=%s", global_step)
    except FloatingPointError as error:
        write_crash_report(paths.crash_report, global_step=global_step, error=error)
        save_checkpoint(paths.checkpoints / "crashed.pt", **checkpoint_kwargs())
        logger.exception("non-finite phase training state")
        raise
    if not interrupted and last_validation_step != global_step:
        perform_validation()
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_kwargs())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": resolved["experiment"],
        "status": status,
        "global_step": global_step,
        "epoch": epoch,
        "sample_offset": sample_offset,
        "stopped_early": early_stopping.stopped,
        "success_step": success_step,
        "consecutive_successes": consecutive_successes,
        "split_counts": manifest.split_counts,
        "manifest_fingerprint": manifest.fingerprint,
        "inference_contract": "Echo spectrum is the only model input; Image is unavailable",
        "authority": evaluation["authority"],
        "success_criteria": criteria,
        "baselines": {
            "echo_identity": baselines.echo_identity,
            "unrestricted_phase_oracle": baselines.unrestricted_phase_oracle,
        },
        "best_validation": best_validation,
        "last_validation": last_validation,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "manifest": str(paths.manifest.resolve()),
            "best_checkpoint": str((paths.checkpoints / "best.pt").resolve()),
            "latest_checkpoint": str((paths.checkpoints / "latest.pt").resolve()),
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
