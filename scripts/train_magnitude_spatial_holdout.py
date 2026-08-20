"""Train the D002 magnitude baseline with a spatially isolated holdout region."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from main import (
    EarlyStoppingState,
    build_loader,
    configure_logger,
    set_seed,
    update_early_stopping,
    write_crash_report,
)
from scripts.overfit_single_magnitude_patch import (
    evaluate_log_magnitude_prediction,
    prepare_log_magnitude_pair,
    train_magnitude_step,
)
from scripts.overfit_single_patch import append_jsonl, utc_now, write_json
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
    "normalized_log_rmse",
    "log_magnitude_correlation",
    "magnitude_rms_ratio_target",
    "log_magnitude_psnr_db",
    "log_magnitude_ssim",
    "magnitude_charbonnier",
)


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics: Path
    train_log: Path
    manifest: Path
    resolved_config: Path
    report: Path
    crash_report: Path


class MagnitudePatchDataset(Dataset[dict[str, Any]]):
    """Load one non-guard split using the E003 Echo-RMS log representation."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        *,
        expected_shape: tuple[int, int],
        rms_epsilon: float,
    ) -> None:
        self.records = tuple(records)
        self.expected_shape = expected_shape
        self.rms_epsilon = float(rms_epsilon)
        if not self.records:
            raise ValueError("MagnitudePatchDataset requires at least one record")
        if any(record.split is SplitName.GUARD for record in self.records):
            raise ValueError("guard records must not be exposed through a Dataset")
        if not math.isfinite(self.rms_epsilon) or self.rms_epsilon <= 0:
            raise ValueError("rms_epsilon must be finite and positive")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            echo = load_complex_patch(record.echo_path, self.expected_shape)
            image = load_complex_patch(record.image_path, self.expected_shape)
            inputs, targets, scale = prepare_log_magnitude_pair(
                echo,
                image,
                epsilon=self.rms_epsilon,
            )
        except Exception as error:
            detail = str(error) if isinstance(error, DatasetIntegrityError) else repr(error)
            raise DatasetIntegrityError(
                f"dataset index={index}, key={record.key}, row={record.row}, "
                f"col={record.col}, echo={record.echo_path}, image={record.image_path}: "
                f"{detail}"
            ) from error
        return {
            "input": inputs,
            "target": targets,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "key": record.key,
            "filename": record.echo_path.name,
            "row": record.row,
            "col": record.col,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train log-magnitude SwinIR on a strict spatial train/guard/validation split."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_magnitude_spatial_holdout.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be a mapping")
    for section in ("model", "data", "optimization", "runtime", "evaluation"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    data = config["data"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    evaluation = config["evaluation"]
    if model.get("in_chans") != 1:
        raise ValueError("magnitude restoration requires model.in_chans=1")
    if float(model.get("drop_path_rate", 0.0)) != 0.0:
        raise ValueError("E004 fixes drop_path_rate=0 to match E003")
    if model.get("use_checkpoint"):
        raise ValueError("activation checkpointing is disabled for E004")
    if data.get("representation") != "echo_rms_log1p_magnitude":
        raise ValueError("E004 requires the E003 Echo-RMS log-magnitude representation")
    if optimization.get("optimizer", "").lower() != "adam":
        raise ValueError("E004 uses Adam")
    if optimization.get("loss") != "magnitude_charbonnier":
        raise ValueError("E004 uses magnitude Charbonnier loss")
    if optimization.get("learning_rate_schedule") != "constant":
        raise ValueError("E004 keeps the E003 constant learning rate")
    if int(optimization.get("total_steps", 0)) <= 0:
        raise ValueError("optimization.total_steps must be positive")
    if runtime.get("batch_size") != 1:
        raise ValueError("E004 fixes physical batch_size=1")
    for name in (
        "validation_interval_steps",
        "archive_interval_steps",
        "log_interval_steps",
        "max_consecutive_fp16_overflows",
    ):
        if int(runtime.get(name, 0)) <= 0:
            raise ValueError(f"runtime.{name} must be positive")
    if runtime["archive_interval_steps"] % runtime["validation_interval_steps"] != 0:
        raise ValueError("archive interval must be divisible by validation interval")
    early_stopping = runtime.get("early_stopping")
    if not isinstance(early_stopping, dict):
        raise ValueError("runtime.early_stopping must be a mapping")
    if int(early_stopping.get("patience", 0)) <= 0:
        raise ValueError("early-stopping patience must be positive")
    if float(early_stopping.get("min_delta", -1.0)) < 0:
        raise ValueError("early-stopping min_delta must be non-negative")
    criteria = evaluation.get("success_criteria")
    if not isinstance(criteria, dict):
        raise ValueError("evaluation.success_criteria must be a mapping")
    for name in (
        "mean_rmse_relative_improvement_min",
        "median_rmse_relative_improvement_min",
        "rmse_win_fraction_min",
        "median_magnitude_rms_ratio_min",
        "median_magnitude_rms_ratio_max",
    ):
        if not math.isfinite(float(criteria[name])):
            raise ValueError(f"success criterion {name} must be finite")
    if not 0.0 <= float(criteria["rmse_win_fraction_min"]) <= 1.0:
        raise ValueError("rmse_win_fraction_min must be in [0, 1]")
    if not (
        0.0 < float(criteria["median_magnitude_rms_ratio_min"])
        <= float(criteria["median_magnitude_rms_ratio_max"])
    ):
        raise ValueError("median magnitude RMS ratio bounds must be positive and ordered")


def make_run_paths(root: Path, *, resuming: bool) -> RunPaths:
    if root.exists() and not resuming:
        raise FileExistsError(
            f"output directory already exists: {root}; choose a new directory or use --resume"
        )
    root.mkdir(parents=True, exist_ok=resuming)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    return RunPaths(
        root=root,
        checkpoints=checkpoints,
        metrics=root / "metrics.jsonl",
        train_log=root / "train.log",
        manifest=root / "split_manifest.json",
        resolved_config=root / "resolved_config.json",
        report=root / "report.json",
        crash_report=root / "crash_report.json",
    )


def resolved_config(
    config: dict[str, Any],
    args: argparse.Namespace,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["schema_version"] = 1
    result["experiment"] = "E004-D002-spatial-holdout-magnitude"
    result["config_file"] = str(args.config.resolve())
    result["data"]["echo_dir"] = str(args.echo_dir.resolve())
    result["data"]["image_dir"] = str(args.image_dir.resolve())
    result["runtime"]["precision"] = precision.as_dict()
    return result


def summarize_metrics(per_sample: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not per_sample:
        raise ValueError("cannot summarize empty validation metrics")
    aggregate = {}
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
    worst_filename = max(
        per_sample,
        key=lambda filename: (per_sample[filename]["normalized_log_rmse"], filename),
    )
    return {
        "sample_count": len(per_sample),
        "aggregate": aggregate,
        "worst_sample_by_rmse": worst_filename,
        "per_sample": per_sample,
    }


@torch.no_grad()
def evaluate_validation(
    model: torch.nn.Module | None,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    charbonnier_epsilon: float,
) -> dict[str, Any]:
    per_sample = {}
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    try:
        for batch in loader:
            inputs = batch["input"]
            targets = batch["target"]
            if int(inputs.shape[0]) != 1:
                raise RuntimeError("validation requires batch_size=1 for per-sample metrics")
            if model is None:
                prediction = inputs
            else:
                with precision.autocast():
                    device_prediction = model(
                        inputs.to(device, non_blocking=device.type == "cuda")
                    )
                prediction = device_prediction.detach().float().cpu()
            filename = str(batch["filename"][0])
            per_sample[filename] = evaluate_log_magnitude_prediction(
                prediction,
                targets,
                charbonnier_epsilon=charbonnier_epsilon,
            )
    finally:
        if model is not None and was_training:
            model.train()
    return summarize_metrics(per_sample)


def compare_to_identity(
    model_summary: dict[str, Any],
    identity_summary: dict[str, Any],
    criteria: dict[str, Any],
) -> dict[str, Any]:
    model_aggregate = model_summary["aggregate"]
    identity_aggregate = identity_summary["aggregate"]
    model_per_sample = model_summary["per_sample"]
    identity_per_sample = identity_summary["per_sample"]
    if model_per_sample.keys() != identity_per_sample.keys():
        raise RuntimeError("model and identity validation samples differ")

    mean_identity_rmse = float(identity_aggregate["normalized_log_rmse"]["mean"])
    median_identity_rmse = float(identity_aggregate["normalized_log_rmse"]["median"])
    mean_improvement = 1.0 - (
        float(model_aggregate["normalized_log_rmse"]["mean"]) / mean_identity_rmse
    )
    median_improvement = 1.0 - (
        float(model_aggregate["normalized_log_rmse"]["median"])
        / median_identity_rmse
    )
    win_fraction = float(
        np.mean(
            [
                model_per_sample[name]["normalized_log_rmse"]
                < identity_per_sample[name]["normalized_log_rmse"]
                for name in model_per_sample
            ]
        )
    )
    correlation_delta = float(
        model_aggregate["log_magnitude_correlation"]["mean"]
        - identity_aggregate["log_magnitude_correlation"]["mean"]
    )
    psnr_delta = float(
        model_aggregate["log_magnitude_psnr_db"]["mean"]
        - identity_aggregate["log_magnitude_psnr_db"]["mean"]
    )
    ssim_delta = float(
        model_aggregate["log_magnitude_ssim"]["mean"]
        - identity_aggregate["log_magnitude_ssim"]["mean"]
    )
    median_rms_ratio = float(
        model_aggregate["magnitude_rms_ratio_target"]["median"]
    )
    checks = {
        "mean_rmse_relative_improvement": mean_improvement
        >= float(criteria["mean_rmse_relative_improvement_min"]),
        "median_rmse_relative_improvement": median_improvement
        >= float(criteria["median_rmse_relative_improvement_min"]),
        "rmse_win_fraction": win_fraction >= float(criteria["rmse_win_fraction_min"]),
        "mean_correlation_improved": (
            correlation_delta > 0
            if criteria["require_mean_correlation_improvement"]
            else True
        ),
        "mean_psnr_improved": (
            psnr_delta > 0 if criteria["require_mean_psnr_improvement"] else True
        ),
        "mean_ssim_improved": (
            ssim_delta > 0 if criteria["require_mean_ssim_improvement"] else True
        ),
        "median_magnitude_rms_ratio": float(
            criteria["median_magnitude_rms_ratio_min"]
        )
        <= median_rms_ratio
        <= float(criteria["median_magnitude_rms_ratio_max"]),
    }
    return {
        "mean_rmse_relative_improvement": mean_improvement,
        "median_rmse_relative_improvement": median_improvement,
        "rmse_win_fraction": win_fraction,
        "mean_correlation_delta": correlation_delta,
        "mean_psnr_db_delta": psnr_delta,
        "mean_ssim_delta": ssim_delta,
        "median_magnitude_rms_ratio": median_rms_ratio,
        "checks": checks,
        "passed": all(checks.values()),
    }


def compact_validation(summary: dict[str, Any], comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": summary["sample_count"],
        "aggregate": summary["aggregate"],
        "worst_sample_by_rmse": summary["worst_sample_by_rmse"],
        "comparison_to_echo_identity": comparison,
    }


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
    seed: int,
    best_mean_rmse: float,
    best_validation: dict[str, Any],
    last_validation_step: int,
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
        "sampler": {"seed": seed, "epoch": epoch, "start_index": sample_offset},
        "best_mean_rmse": best_mean_rmse,
        "best_validation": best_validation,
        "last_validation_step": last_validation_step,
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
        raise RuntimeError("unsupported E004 checkpoint schema version")
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


def validate_and_record(
    *,
    model: torch.nn.Module,
    validation_loader: DataLoader[dict[str, Any]],
    identity_summary: dict[str, Any],
    criteria: dict[str, Any],
    device: torch.device,
    precision: PrecisionPolicy,
    charbonnier_epsilon: float,
    step: int,
    metrics_path: Path,
    logger: logging.Logger,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = evaluate_validation(
        model,
        validation_loader,
        device=device,
        precision=precision,
        charbonnier_epsilon=charbonnier_epsilon,
    )
    comparison = compare_to_identity(summary, identity_summary, criteria)
    compact = compact_validation(summary, comparison)
    append_jsonl(
        metrics_path,
        {"timestamp_utc": utc_now(), "step": step, "split": "validation", **compact},
    )
    aggregate = summary["aggregate"]
    logger.info(
        "validation step=%s mean_rmse=%.6f median_rmse=%.6f improvement=%.2f%% "
        "win_fraction=%.3f mean_corr=%.4f mean_psnr=%.2f mean_ssim=%.4f "
        "median_rms_ratio=%.4f passed=%s",
        step,
        aggregate["normalized_log_rmse"]["mean"],
        aggregate["normalized_log_rmse"]["median"],
        100.0 * comparison["mean_rmse_relative_improvement"],
        comparison["rmse_win_fraction"],
        aggregate["log_magnitude_correlation"]["mean"],
        aggregate["log_magnitude_psnr_db"]["mean"],
        aggregate["log_magnitude_ssim"]["mean"],
        comparison["median_magnitude_rms_ratio"],
        comparison["passed"],
    )
    return summary, comparison


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
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    logger = configure_logger(paths.train_log)
    runtime = resolved["runtime"]
    optimization = resolved["optimization"]
    data = resolved["data"]
    criteria = resolved["evaluation"]["success_criteria"]
    set_seed(int(runtime["seed"]), bool(runtime["strict_reproducibility"]))

    validation_region = CoordinateRegion(**data["validation_region"])
    guard_region = CoordinateRegion(**data["guard_region"])
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        validation_region,
        guard_region,
        expected_counts=data["expected_split_counts"],
    )
    if args.resume is None:
        write_json(paths.resolved_config, resolved)
        manifest.write_json(paths.manifest)
    else:
        if not paths.resolved_config.is_file() or not paths.manifest.is_file():
            raise FileNotFoundError("resume output directory is missing its run contract")
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError("output directory resolved configuration does not match")
        existing_manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        if existing_manifest.get("fingerprint") != manifest.fingerprint:
            raise RuntimeError("output directory manifest fingerprint does not match")

    expected_shape = tuple(int(value) for value in data["expected_shape"])
    train_dataset = MagnitudePatchDataset(
        manifest.records_for(SplitName.TRAIN),
        expected_shape=expected_shape,
        rms_epsilon=float(data["rms_epsilon"]),
    )
    validation_dataset = MagnitudePatchDataset(
        manifest.records_for(SplitName.VALIDATION),
        expected_shape=expected_shape,
        rms_epsilon=float(data["rms_epsilon"]),
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
    loss_epsilon = float(optimization["charbonnier_epsilon"])

    logger.info("computing Echo identity baseline on %s validation patches", len(validation_dataset))
    identity_summary = evaluate_validation(
        None,
        validation_loader,
        device=device,
        precision=precision,
        charbonnier_epsilon=loss_epsilon,
    )
    write_json(paths.root / "echo_identity_baseline.json", identity_summary)

    global_step = 0
    epoch = 0
    sample_offset = 0
    best_mean_rmse = math.inf
    best_validation: dict[str, Any] = {}
    early_stopping = EarlyStoppingState()
    last_validation_step = -1

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
            "seed": int(runtime["seed"]),
            "best_mean_rmse": best_mean_rmse,
            "best_validation": best_validation,
            "last_validation_step": last_validation_step,
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
        best_mean_rmse = float(checkpoint["best_mean_rmse"])
        best_validation = dict(checkpoint["best_validation"])
        last_validation_step = int(checkpoint["last_validation_step"])
        state = checkpoint["early_stopping"]
        early_stopping = EarlyStoppingState(
            bad_validation_count=int(state["bad_validation_count"]),
            stopped=bool(state["stopped"]),
        )
        logger.info(
            "resumed step=%s epoch=%s offset=%s from %s",
            global_step,
            epoch,
            sample_offset,
            args.resume,
        )

    def perform_validation() -> None:
        nonlocal best_mean_rmse, best_validation, last_validation_step
        summary, comparison = validate_and_record(
            model=model,
            validation_loader=validation_loader,
            identity_summary=identity_summary,
            criteria=criteria,
            device=device,
            precision=precision,
            charbonnier_epsilon=loss_epsilon,
            step=global_step,
            metrics_path=paths.metrics,
            logger=logger,
        )
        current = float(summary["aggregate"]["normalized_log_rmse"]["mean"])
        improved = update_early_stopping(
            early_stopping,
            current_value=current,
            best_value=best_mean_rmse,
            patience=int(runtime["early_stopping"]["patience"]),
            min_delta=float(runtime["early_stopping"]["min_delta"]),
        )
        if improved:
            best_mean_rmse = current
            best_validation = {
                "step": global_step,
                "generated_at_utc": utc_now(),
                "summary": summary,
                "comparison_to_echo_identity": comparison,
            }
        last_validation_step = global_step
        save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
        if improved:
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_kwargs())
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
        if early_stopping.stopped:
            logger.warning("checkpoint is already early-stopped; no updates will run")
        while global_step < total_steps and not early_stopping.stopped:
            sampler.set_position(epoch, sample_offset)
            for batch in train_loader:
                while True:
                    result = train_magnitude_step(
                        model,
                        ema_model,
                        optimizer,
                        scheduler,
                        scaler,
                        batch["input"],
                        batch["target"],
                        device=device,
                        precision=precision,
                        loss_epsilon=loss_epsilon,
                        ema_decay=float(optimization["ema_decay"]),
                    )
                    if result.did_optimizer_step:
                        overflow_streak = 0
                        break
                    overflow_streak += 1
                    logger.warning(
                        "FP16 overflow before step=%s, retry=%s",
                        global_step + 1,
                        overflow_streak,
                    )
                    if overflow_streak >= int(runtime["max_consecutive_fp16_overflows"]):
                        raise FloatingPointError("too many consecutive FP16 overflows")

                sample_offset += int(batch["input"].shape[0])
                global_step += 1
                if global_step % int(runtime["log_interval_steps"]) == 0:
                    append_jsonl(
                        paths.metrics,
                        {
                            "timestamp_utc": utc_now(),
                            "step": global_step,
                            "split": "train",
                            "loss": result.loss,
                            "gradient_norm": result.gradient_norm,
                            "learning_rate": float(optimizer.param_groups[0]["lr"]),
                            "epoch": epoch,
                            "sample_offset": sample_offset,
                        },
                    )
                    logger.info(
                        "train step=%s epoch=%s offset=%s loss=%.6f grad_norm=%.4f",
                        global_step,
                        epoch,
                        sample_offset,
                        result.loss,
                        result.gradient_norm,
                    )
                if global_step % int(runtime["validation_interval_steps"]) == 0:
                    perform_validation()
                    if early_stopping.stopped:
                        logger.info(
                            "early stopping at step=%s after %s non-improving validations",
                            global_step,
                            early_stopping.bad_validation_count,
                        )
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
        logger.exception("non-finite training state; crash checkpoint and report written")
        raise

    if not interrupted and last_validation_step != global_step:
        perform_validation()
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_kwargs())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())

    best_comparison = best_validation.get("comparison_to_echo_identity", {})
    status = (
        "interrupted"
        if interrupted
        else "passed"
        if bool(best_comparison.get("passed", False))
        else "failed"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": resolved["experiment"],
        "status": status,
        "global_step": global_step,
        "epoch": epoch,
        "sample_offset": sample_offset,
        "stopped_early": early_stopping.stopped,
        "split_counts": manifest.split_counts,
        "manifest_fingerprint": manifest.fingerprint,
        "representation": data["representation"],
        "authority": resolved["evaluation"]["authority"],
        "success_criteria": criteria,
        "echo_identity": identity_summary,
        "best_validation": best_validation,
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
