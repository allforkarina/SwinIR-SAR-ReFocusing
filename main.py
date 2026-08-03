"""Single-process training entry point for the SAR SwinIR baseline.

The intentionally unimplemented ``train_one_step`` is the final learner task.
Everything around it establishes the agreed data split, validation, logging, and
strict checkpoint contract before any long server run is started.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from swinir import SwinIR
from swinir.sar_dataset import (
    CoordinateRegion,
    ResumableEpochSampler,
    SARPatchDataset,
    SplitName,
    build_manifest,
)
from swinir.training import (
    PrecisionPolicy,
    TrainStepResult,
    atomic_torch_save,
    capture_rng_state,
    complex_charbonnier_loss,
    global_gradient_norm,
    make_ema_model,
    make_grad_scaler,
    normalized_complex_rmse,
    resolve_device,
    resolve_precision,
    restore_rng_state,
    update_ema,
)


CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    metrics_jsonl: Path
    train_log: Path
    manifest: Path
    resolved_config: Path
    crash_report: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SwinIR for SAR refocusing")
    parser.add_argument("--config", type=Path, default=Path("configs/train_sar.yaml"))
    parser.add_argument("--run-name", required=True)
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
    for section in ("model", "data", "optimization", "runtime"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    runtime = config["runtime"]
    optimization = config["optimization"]
    model = config["model"]
    if runtime["batch_size"] != 1:
        raise ValueError("this baseline fixes physical batch_size=1")
    if runtime["accumulation_steps"] != 1:
        raise ValueError("this baseline fixes accumulation_steps=1")
    if model["in_chans"] != 2:
        raise ValueError("SAR complex input requires exactly two real/imag channels")
    if model["use_checkpoint"]:
        raise ValueError("activation checkpointing is disabled for this baseline")
    if optimization["optimizer"].lower() != "adam":
        raise ValueError("this baseline uses Adam")
    if optimization["loss"] != "complex_charbonnier":
        raise ValueError("this baseline uses joint complex Charbonnier loss")
    if optimization["total_steps"] <= 0:
        raise ValueError("total_steps must be positive")
    if any(step <= 0 for step in optimization["milestones"]):
        raise ValueError("all scheduler milestones must be positive")
    if sorted(optimization["milestones"]) != optimization["milestones"]:
        raise ValueError("scheduler milestones must be sorted")


def make_run_paths(output_root: Path, run_name: str, resuming: bool) -> RunPaths:
    if not run_name or Path(run_name).name != run_name:
        raise ValueError("run_name must be one simple directory name")
    root = output_root / run_name
    if root.exists() and not resuming:
        raise FileExistsError(
            f"run directory already exists: {root}; choose a new --run-name or use --resume"
        )
    root.mkdir(parents=True, exist_ok=resuming)
    checkpoints = root / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    return RunPaths(
        root=root,
        checkpoints=checkpoints,
        metrics_jsonl=root / "metrics.jsonl",
        train_log=root / "train.log",
        manifest=root / "split_manifest.json",
        resolved_config=root / "resolved_config.yaml",
        crash_report=root / "crash_report.json",
    )


def configure_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("swinir_sar_train")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(path, encoding="utf-8")):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def set_seed(seed: int, strict_reproducibility: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not strict_reproducibility
    torch.use_deterministic_algorithms(strict_reproducibility)


def build_loader(
    dataset: SARPatchDataset,
    *,
    batch_size: int,
    workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    sampler: ResumableEpochSampler | None = None,
) -> DataLoader[dict[str, Any]]:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = prefetch_factor
    if sampler is None:
        return DataLoader(dataset, shuffle=False, **options)
    return DataLoader(dataset, sampler=sampler, **options)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
) -> dict[str, float]:
    """Evaluate the EMA model and the identity-Echo baseline on all validation data."""

    was_training = model.training
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_rmse = 0.0
    total_baseline_loss = 0.0
    total_baseline_rmse = 0.0
    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        batch_size = int(inputs.shape[0])
        with precision.autocast():
            predictions = model(inputs)
        loss = complex_charbonnier_loss(predictions, targets, loss_epsilon)
        rmse = normalized_complex_rmse(predictions, targets)
        baseline_loss = complex_charbonnier_loss(inputs, targets, loss_epsilon)
        baseline_rmse = normalized_complex_rmse(inputs, targets)
        total_examples += batch_size
        total_loss += float(loss.item()) * batch_size
        total_rmse += float(rmse.item()) * batch_size
        total_baseline_loss += float(baseline_loss.item()) * batch_size
        total_baseline_rmse += float(baseline_rmse.item()) * batch_size
    if was_training:
        model.train()
    if total_examples == 0:
        raise RuntimeError("validation loader is empty")
    return {
        "charbonnier": total_loss / total_examples,
        "complex_rmse": total_rmse / total_examples,
        "echo_baseline_charbonnier": total_baseline_loss / total_examples,
        "echo_baseline_complex_rmse": total_baseline_rmse / total_examples,
    }


def append_metrics(path: Path, metrics: dict[str, float | int | str]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics, ensure_ascii=False, sort_keys=True) + "\n")


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Adam,
    scheduler: MultiStepLR,
    scaler: torch.cuda.amp.GradScaler,
    global_step: int,
    epoch: int,
    sample_offset: int,
    sampler: ResumableEpochSampler,
    best_metrics: dict[str, float],
    manifest_fingerprint: str,
    config: dict[str, Any],
    precision: PrecisionPolicy,
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
        "sampler": {"seed": sampler.seed, "epoch": epoch, "start_index": sample_offset},
        "best_metrics": best_metrics,
        "rng": capture_rng_state(),
        "manifest_fingerprint": manifest_fingerprint,
        "config": config,
        "precision": precision.as_dict(),
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def load_checkpoint_strict(
    path: Path,
    *,
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Adam,
    scheduler: MultiStepLR,
    scaler: torch.cuda.amp.GradScaler,
    manifest_fingerprint: str,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported checkpoint schema version")
    if checkpoint.get("manifest_fingerprint") != manifest_fingerprint:
        raise RuntimeError("checkpoint dataset manifest fingerprint does not match")
    if checkpoint.get("config") != config:
        raise RuntimeError("checkpoint configuration does not exactly match current config")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint


def write_crash_report(path: Path, *, global_step: int, error: Exception) -> None:
    path.write_text(
        json.dumps(
            {
                "global_step": global_step,
                "error_type": type(error).__name__,
                "error": str(error),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def train_one_step(
    model: torch.nn.Module,
    ema_model: torch.nn.Module,
    optimizer: Adam,
    scheduler: MultiStepLR,
    scaler: torch.cuda.amp.GradScaler,
    batch: dict[str, Any],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
    ema_decay: float,
) -> TrainStepResult:
    """Run one physical batch.

    Learner task 3: implement this exact state transition:
    zero gradients -> move input/target -> autocast model forward -> FP32 joint
    Charbonnier loss -> reject non-finite loss -> backward -> record gradient
    norm (unscale first for FP16) -> optimizer step.  If and only if that step
    succeeds, advance the scheduler and EMA.  For an FP16 overflow, return a
    result with ``did_optimizer_step=False``; do not advance either state.
    """

    # currently, it's train process.
    model.train()
    # clear grad of last round, save space and time, set to None.
    optimizer.zero_grad(set_to_none=True)

    # move to device
    inputs = batch["input"].to(device, non_blocking=device.type == "cuda")
    targets = batch["target"].to(device, non_blocking=device.type == "cuda")

    # auto precision context
    with precision.autocast():
        # forward pass
        predictions = model(inputs)

    # compute loss in FP32, more precise.
    loss = complex_charbonnier_loss(predictions, targets, loss_epsilon)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite loss encountered during training")

    # from tensor to float, it is a count of loss, not a tensor.
    loss_value = float(loss.detach().item())

    # grad backward step
    # FP16 path here.
    if precision.uses_grad_scaler:
        # get the scale of last round before backward.
        scaler_scale_before = float(scaler.get_scale())

        # enlarge the loss's scale, then backward update the parameter.
        scaler.scale(loss).backward()
        # smaller the scale of the optimizer, then check if contains Inf/NaN.
        scaler.unscale_(optimizer)
        gradient_norm = global_gradient_norm(model.parameters())

        # Everythings fine, step the optimizer, if overflow, return False.
        scaler.step(optimizer)
        scaler.update() # update the scale for next round scale.

        scaler_scale_after = float(scaler.get_scale())
        # judge the optimizer step by comparing the scale before and after
        did_optimizer_step = scaler_scale_after >= scaler_scale_before

    # FP32 and BF16 path here.
    else:
        # backward update the parameter.
        loss.backward()
        # get the gradient norm.
        gradient_norm = global_gradient_norm(model.parameters())

        # update the optimizer.
        optimizer.step()

        scaler_scale_before = None
        scaler_scale_after = None
        did_optimizer_step = True

    # truely optimizer step.
    if did_optimizer_step:
        scheduler.step()
        update_ema(ema_model, model, ema_decay)

    return TrainStepResult(
        loss=loss_value,
        gradient_norm=gradient_norm,
        did_optimizer_step=did_optimizer_step,
        scaler_scale_before=scaler_scale_before,
        scaler_scale_after=scaler_scale_after,
    )


def run_training(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Build the deterministic data/model state and execute the step-driven loop."""

    validate_config(config)
    data_config = config["data"]
    optimization = config["optimization"]
    runtime = config["runtime"]
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    run_paths = make_run_paths(Path(runtime["output_root"]), args.run_name, args.resume is not None)
    logger = configure_logger(run_paths.train_log)
    set_seed(int(runtime["seed"]), bool(runtime["strict_reproducibility"]))
    run_paths.resolved_config.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    validation_region = CoordinateRegion(**data_config["validation_region"])
    guard_region = CoordinateRegion(**data_config["guard_region"])
    manifest = build_manifest(
        Path(data_config["echo_dir"]),
        Path(data_config["image_dir"]),
        validation_region,
        guard_region,
        expected_counts=data_config["expected_split_counts"],
    )
    manifest.write_json(run_paths.manifest)
    expected_shape = tuple(data_config["expected_shape"])
    train_dataset = SARPatchDataset(
        manifest.records_for(SplitName.TRAIN), expected_shape, data_config["rms_epsilon"]
    )
    validation_dataset = SARPatchDataset(
        manifest.records_for(SplitName.VALIDATION), expected_shape, data_config["rms_epsilon"]
    )
    sampler = ResumableEpochSampler(train_dataset, seed=int(runtime["seed"]))
    train_loader = build_loader(
        train_dataset,
        batch_size=runtime["batch_size"],
        workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=device.type == "cuda",
        sampler=sampler,
    )
    validation_loader = build_loader(
        validation_dataset,
        batch_size=runtime["batch_size"],
        workers=data_config["num_workers"],
        prefetch_factor=data_config["prefetch_factor"],
        pin_memory=device.type == "cuda",
    )

    model = SwinIR(**config["model"]).to(device)
    ema_model = make_ema_model(model).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=optimization["learning_rate"],
        betas=tuple(optimization["betas"]),
        eps=optimization["epsilon"],
        weight_decay=optimization["weight_decay"],
    )
    scheduler = MultiStepLR(
        optimizer, milestones=optimization["milestones"], gamma=optimization["gamma"]
    )
    scaler = make_grad_scaler(precision)
    global_step = 0
    epoch = 0
    sample_offset = 0
    best_metrics = {"charbonnier": math.inf, "complex_rmse": math.inf}

    if args.resume is not None:
        checkpoint = load_checkpoint_strict(
            args.resume,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            manifest_fingerprint=manifest.fingerprint,
            config=config,
            device=device,
        )
        global_step = int(checkpoint["global_step"])
        epoch = int(checkpoint["epoch"])
        sample_offset = int(checkpoint["sample_offset"])
        best_metrics = dict(checkpoint["best_metrics"])
        logger.info("resumed step=%s epoch=%s offset=%s", global_step, epoch, sample_offset)
    else:
        initial_metrics = validate(
            ema_model, validation_loader, device, precision, optimization["charbonnier_epsilon"]
        )
        append_metrics(run_paths.metrics_jsonl, {"step": 0, "split": "validation", **initial_metrics})
        logger.info("validation step=0 %s", initial_metrics)

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(log_dir=run_paths.root / "tensorboard")
    overflow_streak = 0
    total_steps = int(optimization["total_steps"])
    try:
        while global_step < total_steps:
            sampler.set_position(epoch, sample_offset)
            for batch in train_loader:
                result = train_one_step(
                    model,
                    ema_model,
                    optimizer,
                    scheduler,
                    scaler,
                    batch,
                    device=device,
                    precision=precision,
                    loss_epsilon=optimization["charbonnier_epsilon"],
                    ema_decay=optimization["ema_decay"],
                )
                sample_offset += int(batch["input"].shape[0])
                if not result.did_optimizer_step:
                    overflow_streak += 1
                    logger.warning("FP16 overflow at step=%s, streak=%s", global_step, overflow_streak)
                    if overflow_streak >= runtime["max_consecutive_fp16_overflows"]:
                        raise FloatingPointError("too many consecutive FP16 overflows")
                    continue
                overflow_streak = 0
                global_step += 1
                if global_step % runtime["log_interval_steps"] == 0:
                    lr = float(optimizer.param_groups[0]["lr"])
                    metrics = {
                        "step": global_step,
                        "split": "train",
                        "loss": result.loss,
                        "gradient_norm": result.gradient_norm,
                        "learning_rate": lr,
                    }
                    append_metrics(run_paths.metrics_jsonl, metrics)
                    for name, value in metrics.items():
                        if name not in {"step", "split"}:
                            writer.add_scalar(f"train/{name}", value, global_step)
                    logger.info("train %s", metrics)
                if global_step % runtime["validation_interval_steps"] == 0:
                    validation_metrics = validate(
                        ema_model, validation_loader, device, precision, optimization["charbonnier_epsilon"]
                    )
                    metrics = {"step": global_step, "split": "validation", **validation_metrics}
                    append_metrics(run_paths.metrics_jsonl, metrics)
                    for name, value in validation_metrics.items():
                        writer.add_scalar(f"validation/{name}", value, global_step)
                    logger.info("validation %s", metrics)
                    is_best = validation_metrics["charbonnier"] < best_metrics["charbonnier"]
                    if is_best:
                        best_metrics = validation_metrics
                    checkpoint_args = dict(
                        model=model,
                        ema_model=ema_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        global_step=global_step,
                        epoch=epoch,
                        sample_offset=sample_offset,
                        sampler=sampler,
                        best_metrics=best_metrics,
                        manifest_fingerprint=manifest.fingerprint,
                        config=config,
                        precision=precision,
                    )
                    save_checkpoint(run_paths.checkpoints / "latest.pt", **checkpoint_args)
                    if is_best:
                        save_checkpoint(run_paths.checkpoints / "best.pt", **checkpoint_args)
                    if global_step % runtime["archive_interval_steps"] == 0:
                        save_checkpoint(
                            run_paths.checkpoints / f"step_{global_step:06d}.pt", **checkpoint_args
                        )
                if global_step >= total_steps:
                    break
            if sample_offset >= len(train_dataset):
                epoch += 1
                sample_offset = 0
    except KeyboardInterrupt:
        save_checkpoint(
            run_paths.checkpoints / "interrupted.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
            epoch=epoch,
            sample_offset=sample_offset,
            sampler=sampler,
            best_metrics=best_metrics,
            manifest_fingerprint=manifest.fingerprint,
            config=config,
            precision=precision,
        )
        logger.warning("interrupted checkpoint written at step=%s", global_step)
        raise
    except FloatingPointError as error:
        write_crash_report(run_paths.crash_report, global_step=global_step, error=error)
        logger.exception("non-finite training state; crash report written")
        raise
    finally:
        writer.close()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_training(config, args)


if __name__ == "__main__":
    main()
