"""Consolidate a joint SAR patch-set model with one full-set update per epoch."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from scripts.overfit_patch_set import (
    CHECKPOINT_SCHEMA_VERSION as SOURCE_CHECKPOINT_SCHEMA_VERSION,
    LoadedSample,
    evaluate_patch_set,
    load_selected_samples,
    restore_checkpoint,
    save_checkpoint,
    save_representative_artifacts,
)
from scripts.overfit_single_patch import (
    SuccessCriteria,
    append_jsonl,
    file_sha256,
    make_run_paths,
    set_seed,
    utc_now,
    write_json,
)
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair
from swinir.training import (
    PrecisionPolicy,
    TrainStepResult,
    complex_charbonnier_loss,
    global_gradient_norm,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    update_ema,
)


REPORT_SCHEMA_VERSION = 1


def load_source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported source patch-set checkpoint schema version")
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") != "joint_patch_set_overfit":
        raise RuntimeError("source checkpoint is not a joint patch-set overfit run")
    if not isinstance(config.get("selection_manifest"), dict):
        raise RuntimeError("source checkpoint is missing its selection manifest")
    if not isinstance(checkpoint.get("ema_model"), dict):
        raise RuntimeError("source checkpoint is missing EMA model weights")
    return checkpoint


def source_pairs(manifest: dict[str, Any]) -> tuple[DiscoveredPair, ...]:
    records = manifest.get("samples")
    if not isinstance(records, list) or len(records) < 2:
        raise RuntimeError("source selection manifest must contain at least two samples")
    ordered = sorted(records, key=lambda record: int(record["selection_index"]))
    if [int(record["selection_index"]) for record in ordered] != list(range(len(ordered))):
        raise RuntimeError("source selection indices must be contiguous from zero")
    return tuple(
        DiscoveredPair(
            row=int(record["row"]),
            col=int(record["col"]),
            echo_path=Path(record["echo_path"]),
            image_path=Path(record["image_path"]),
        )
        for record in ordered
    )


def verify_loaded_samples(
    samples: Sequence[LoadedSample],
    manifest: dict[str, Any],
) -> None:
    records = sorted(manifest["samples"], key=lambda record: int(record["selection_index"]))
    if len(samples) != int(manifest["sample_count"]) or len(samples) != len(records):
        raise RuntimeError("loaded sample count does not match the source selection manifest")
    for sample, record in zip(samples, records, strict=True):
        if sample.filename != record["filename"]:
            raise RuntimeError("loaded sample order does not match the source selection manifest")
        current = sample.fingerprint
        for key in ("echo_sha256", "image_sha256", "echo_size_bytes", "image_size_bytes"):
            if current[key] != record[key]:
                raise RuntimeError(
                    f"source sample fingerprint changed for {sample.filename}: field={key}"
                )


def success_criteria_from_source(config: dict[str, Any]) -> SuccessCriteria:
    values = config.get("evaluation", {}).get("success_criteria")
    if not isinstance(values, dict):
        raise RuntimeError("source checkpoint is missing success criteria")
    criteria = SuccessCriteria(**values)
    criteria.validate()
    return criteria


def shuffled_epoch_order(epoch: int, sample_count: int, seed: int) -> tuple[int, ...]:
    if epoch < 0:
        raise ValueError("epoch must be non-negative")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return tuple(
        int(index)
        for index in np.random.default_rng(seed + epoch).permutation(sample_count)
    )


def train_full_set_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    samples: Sequence[LoadedSample],
    order: Sequence[int],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
    ema_decay: float,
) -> TrainStepResult:
    """Accumulate the mean gradient of every sample, then update exactly once."""

    if len(order) != len(samples) or sorted(order) != list(range(len(samples))):
        raise ValueError("order must contain every sample index exactly once")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    scaler_scale_before = float(scaler.get_scale()) if precision.uses_grad_scaler else None

    for index in order:
        sample = samples[index]
        inputs = sample.inputs.to(device, non_blocking=device.type == "cuda")
        targets = sample.targets.to(device, non_blocking=device.type == "cuda")
        with precision.autocast():
            predictions = model(inputs)
        loss = complex_charbonnier_loss(predictions, targets, loss_epsilon)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss for sample {sample.filename}")
        loss_sum += float(loss.detach().item())
        mean_loss_component = loss / len(samples)
        if precision.uses_grad_scaler:
            scaler.scale(mean_loss_component).backward()
        else:
            mean_loss_component.backward()

    if precision.uses_grad_scaler:
        scaler.unscale_(optimizer)
    gradient_norm = global_gradient_norm(model.parameters())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("non-finite full-set gradient norm")

    if precision.uses_grad_scaler:
        scaler.step(optimizer)
        scaler.update()
        scaler_scale_after = float(scaler.get_scale())
        did_optimizer_step = scaler_scale_after >= float(scaler_scale_before)
    else:
        optimizer.step()
        scaler_scale_after = None
        did_optimizer_step = True

    if did_optimizer_step:
        scheduler.step()
        update_ema(ema_model, model, ema_decay)

    return TrainStepResult(
        loss=loss_sum / len(samples),
        gradient_norm=gradient_norm,
        did_optimizer_step=did_optimizer_step,
        scaler_scale_before=scaler_scale_before,
        scaler_scale_after=scaler_scale_after,
    )


def make_resolved_config(
    args: argparse.Namespace,
    *,
    source_config: dict[str, Any],
    source_checkpoint_sha256: str,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "full_set_gradient_consolidation",
        "source": {
            "checkpoint": str(args.source_checkpoint.resolve()),
            "checkpoint_sha256": source_checkpoint_sha256,
            "initial_weights": "ema_model",
            "optimizer_state_reused": False,
        },
        "selection_manifest": source_config["selection_manifest"],
        "model": source_config["model"],
        "data": source_config["data"],
        "optimization": {
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate_schedule": "constant",
            "charbonnier_epsilon": float(
                source_config["optimization"]["charbonnier_epsilon"]
            ),
            "ema_decay": float(args.ema_decay),
            "max_epochs": int(args.epochs),
            "physical_batch_size": 1,
            "gradient_accumulation_samples": int(
                source_config["selection_manifest"]["sample_count"]
            ),
            "optimizer_steps_per_epoch": 1,
        },
        "evaluation": {
            "authority": "raw_model_all_samples",
            "eval_every_epochs": int(args.eval_every),
            "save_every_epochs": int(args.save_every),
            "required_consecutive_successes": int(args.required_consecutive_successes),
            "log_magnitude_floor_db": float(
                source_config["evaluation"]["log_magnitude_floor_db"]
            ),
            "success_criteria": source_config["evaluation"]["success_criteria"],
        },
        "runtime": {
            "seed": int(args.seed),
            "device": str(precision.device),
            "precision": precision.as_dict(),
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in ("epochs", "eval_every", "save_every", "required_consecutive_successes"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    minimum_epochs = args.eval_every * args.required_consecutive_successes
    if args.epochs < minimum_epochs:
        raise ValueError(
            "epochs must allow all required post-update success evaluations: "
            f"epochs={args.epochs}, required minimum={minimum_epochs}"
        )
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {args.source_checkpoint}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    source_checkpoint_sha256 = file_sha256(args.source_checkpoint)
    source_checkpoint = load_source_checkpoint(args.source_checkpoint)
    source_config = source_checkpoint["resolved_config"]
    source_manifest = source_config["selection_manifest"]
    criteria = success_criteria_from_source(source_config)
    expected_shape = tuple(int(value) for value in source_config["data"]["expected_shape"])
    samples = load_selected_samples(
        source_pairs(source_manifest),
        expected_shape=expected_shape,
        rms_epsilon=float(source_config["data"]["rms_epsilon"]),
    )
    verify_loaded_samples(samples, source_manifest)

    device = resolve_device(args.device)
    precision = resolve_precision(device)
    resolved_config = make_resolved_config(
        args,
        source_config=source_config,
        source_checkpoint_sha256=source_checkpoint_sha256,
        precision=precision,
    )
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    manifest_path = paths.root / "selected_samples.json"
    if args.resume is None:
        write_json(paths.resolved_config, resolved_config)
        write_json(manifest_path, source_manifest)
    else:
        if not paths.resolved_config.is_file():
            raise FileNotFoundError("resume output directory is missing resolved_config.json")
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing != resolved_config:
            raise RuntimeError("output directory resolved_config.json does not match this run")

    set_seed(int(args.seed))
    model = SwinIR(**source_config["model"]).to(device)
    model.load_state_dict(source_checkpoint["ema_model"], strict=True)
    ema_model = make_ema_model(model).to(device)
    del source_checkpoint
    gc.collect()

    optimizer = Adam(
        model.parameters(),
        lr=float(args.learning_rate),
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = make_grad_scaler(precision)
    loss_epsilon = float(source_config["optimization"]["charbonnier_epsilon"])
    floor_db = float(source_config["evaluation"]["log_magnitude_floor_db"])
    anchor_filename = str(source_manifest["anchor_filename"])

    epoch = 0
    best_worst_rmse = math.inf
    consecutive_successes = 0
    success_epoch: int | None = None
    last_metrics: dict[str, Any] = {}
    last_evaluation_epoch = -1
    if args.resume is not None:
        checkpoint = restore_checkpoint(
            args.resume,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            resolved_config=resolved_config,
            device=device,
        )
        epoch = int(checkpoint["step"])
        best_worst_rmse = float(checkpoint["best_worst_rmse"])
        consecutive_successes = int(checkpoint["consecutive_successes"])
        success_epoch = checkpoint["success_step"]
        last_metrics = dict(checkpoint["last_metrics"])
        print(f"resumed epoch={epoch} from {args.resume}", flush=True)

    def checkpoint_kwargs() -> dict[str, Any]:
        return {
            "model": model,
            "ema_model": ema_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "step": epoch,
            "resolved_config": resolved_config,
            "best_worst_rmse": best_worst_rmse,
            "consecutive_successes": consecutive_successes,
            "success_step": success_epoch,
            "last_metrics": last_metrics,
        }

    def evaluate_and_record(
        train_loss: float | None,
        gradient_norm: float | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        nonlocal best_worst_rmse, consecutive_successes, success_epoch
        nonlocal last_metrics, last_evaluation_epoch
        predictions, model_metrics = evaluate_patch_set(
            model,
            ema_model,
            samples,
            device=device,
            precision=precision,
            floor_db=floor_db,
            charbonnier_epsilon=loss_epsilon,
            criteria=criteria,
        )
        raw = model_metrics["raw"]
        all_passed = bool(raw["all_passed"])
        if epoch > 0:
            consecutive_successes = consecutive_successes + 1 if all_passed else 0
            if (
                consecutive_successes >= args.required_consecutive_successes
                and success_epoch is None
            ):
                success_epoch = epoch
        last_metrics = {
            "epoch": epoch,
            "optimizer_steps": epoch,
            "sample_presentations": epoch * len(samples),
            "timestamp_utc": utc_now(),
            "mean_full_set_train_loss": train_loss,
            "full_set_gradient_norm": gradient_norm,
            "raw": model_metrics["raw"],
            "ema": model_metrics["ema"],
            "raw_all_passed": all_passed,
            "consecutive_successes": consecutive_successes,
        }
        append_jsonl(paths.metrics, last_metrics)
        last_evaluation_epoch = epoch
        aggregate = raw["aggregate"]
        print(
            f"epoch={epoch} updates={epoch} "
            f"train_loss={train_loss if train_loss is not None else float('nan'):.6f} "
            f"grad_norm={gradient_norm if gradient_norm is not None else float('nan'):.4f} "
            f"raw_pass={raw['pass_count']}/{len(samples)} "
            f"worst_rmse={aggregate['normalized_complex_rmse']['max']:.6f} "
            f"min_coherence={aggregate['complex_coherence']['min']:.4f} "
            f"min_mag_corr={aggregate['magnitude_correlation']['min']:.4f} "
            f"rms_ratio=[{aggregate['rms_ratio_target']['min']:.4f},"
            f"{aggregate['rms_ratio_target']['max']:.4f}] "
            f"min_psnr={aggregate['log_magnitude_psnr_db']['min']:.2f} "
            f"min_ssim={aggregate['log_magnitude_ssim']['min']:.4f} "
            f"pass={all_passed} streak={consecutive_successes}",
            flush=True,
        )
        current_worst_rmse = float(aggregate["normalized_complex_rmse"]["max"])
        if current_worst_rmse < best_worst_rmse:
            best_worst_rmse = current_worst_rmse
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_kwargs())
        return predictions

    interrupted = False
    last_predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    last_train_loss: float | None = None
    last_gradient_norm: float | None = None
    try:
        if args.resume is None:
            last_predictions = evaluate_and_record(None, None)
            save_representative_artifacts(
                paths,
                step=epoch,
                samples=samples,
                predictions=last_predictions,
                metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                anchor_filename=anchor_filename,
                floor_db=floor_db,
            )

        overflow_streak = 0
        while epoch < args.epochs and success_epoch is None:
            train_result = train_full_set_epoch(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                samples,
                shuffled_epoch_order(epoch, len(samples), int(args.seed)),
                device=device,
                precision=precision,
                loss_epsilon=loss_epsilon,
                ema_decay=float(args.ema_decay),
            )
            if not train_result.did_optimizer_step:
                overflow_streak += 1
                if overflow_streak >= 8:
                    raise FloatingPointError("eight consecutive mixed-precision overflows")
                continue
            overflow_streak = 0
            epoch += 1
            last_train_loss = train_result.loss
            last_gradient_norm = train_result.gradient_norm

            if epoch % args.eval_every == 0:
                last_predictions = evaluate_and_record(last_train_loss, last_gradient_norm)
            if epoch % args.save_every == 0:
                if last_evaluation_epoch != epoch or last_predictions is None:
                    raise RuntimeError("save interval must coincide with an evaluation")
                save_representative_artifacts(
                    paths,
                    step=epoch,
                    samples=samples,
                    predictions=last_predictions,
                    metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                    anchor_filename=anchor_filename,
                    floor_db=floor_db,
                )
                save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(paths.checkpoints / "interrupted.pt", **checkpoint_kwargs())

    if last_evaluation_epoch != epoch:
        last_predictions = evaluate_and_record(last_train_loss, last_gradient_norm)
    if last_predictions is None:
        last_predictions, _ = evaluate_patch_set(
            model,
            ema_model,
            samples,
            device=device,
            precision=precision,
            floor_db=floor_db,
            charbonnier_epsilon=loss_epsilon,
            criteria=criteria,
        )
    artifact_samples = save_representative_artifacts(
        paths,
        step=epoch,
        samples=samples,
        predictions=last_predictions,
        metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
        anchor_filename=anchor_filename,
        floor_db=floor_db,
    )
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_kwargs())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())

    status = "interrupted" if interrupted else "passed" if success_epoch is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "epoch": epoch,
        "optimizer_steps": epoch,
        "sample_presentations": epoch * len(samples),
        "success_epoch": success_epoch,
        "required_consecutive_successes": args.required_consecutive_successes,
        "success_criteria": asdict(criteria),
        "source": resolved_config["source"],
        "selection_manifest": source_manifest,
        "precision": precision.as_dict(),
        "final": last_metrics,
        "best_raw_worst_sample_normalized_complex_rmse": best_worst_rmse,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "selection_manifest": str(manifest_path.resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "representative_samples": artifact_samples,
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize from a joint patch-set EMA model and consolidate it using one "
            "mean-gradient update over the complete selected set per epoch."
        )
    )
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--required-consecutive-successes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
