"""Test whether global patch coordinates resolve joint SAR mapping conflicts."""

from __future__ import annotations

import argparse
import gc
import json
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
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from scripts.consolidate_patch_set import (
    shuffled_epoch_order,
    source_pairs,
    success_criteria_from_source,
    verify_loaded_samples,
)
from scripts.overfit_patch_set import (
    CHECKPOINT_SCHEMA_VERSION as SOURCE_CHECKPOINT_SCHEMA_VERSION,
    LoadedSample,
    load_selected_samples,
    restore_checkpoint,
    save_checkpoint,
    save_representative_artifacts,
    summarize_metric_map,
)
from scripts.overfit_single_patch import (
    SuccessCriteria,
    append_jsonl,
    evaluate_prediction,
    file_sha256,
    make_run_paths,
    set_seed,
    utc_now,
    write_json,
)
from swinir import SwinIR
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


@dataclass(frozen=True)
class CoordinateBounds:
    row_min: int
    row_max: int
    col_min: int
    col_max: int

    def validate(self) -> None:
        if self.row_max <= self.row_min or self.col_max <= self.col_min:
            raise ValueError("coordinate bounds must have positive row and column spans")

    def encode(self, row: int, col: int) -> torch.Tensor:
        self.validate()
        row_value = 2.0 * (row - self.row_min) / (self.row_max - self.row_min) - 1.0
        col_value = 2.0 * (col - self.col_min) / (self.col_max - self.col_min) - 1.0
        values = torch.tensor([[row_value, col_value]], dtype=torch.float32)
        if not bool(torch.isfinite(values).all()) or bool((values.abs() > 1.0 + 1e-6).any()):
            raise ValueError(f"coordinate {(row, col)} falls outside normalization bounds")
        return values


def coordinate_bounds(samples: Sequence[LoadedSample]) -> CoordinateBounds:
    if len(samples) < 2:
        raise ValueError("coordinate conditioning requires at least two samples")
    bounds = CoordinateBounds(
        row_min=min(sample.row for sample in samples),
        row_max=max(sample.row for sample in samples),
        col_min=min(sample.col for sample in samples),
        col_max=max(sample.col for sample in samples),
    )
    bounds.validate()
    return bounds


class CoordinateConditionedSwinIR(SwinIR):
    """Same-size SwinIR with global row/column FiLM on shallow features."""

    def __init__(self, *args: Any, coordinate_hidden_dim: int = 64, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.upsampler != "":
            raise ValueError("coordinate diagnostic supports only same-size SAR restoration")
        if coordinate_hidden_dim <= 0:
            raise ValueError("coordinate_hidden_dim must be positive")
        self.coordinate_hidden_dim = int(coordinate_hidden_dim)
        self.coordinate_mlp = nn.Sequential(
            nn.Linear(2, self.coordinate_hidden_dim),
            nn.GELU(),
            nn.Linear(self.coordinate_hidden_dim, 2 * self.embed_dim),
        )
        final_layer = self.coordinate_mlp[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def load_unconditioned_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        expected_missing = {f"coordinate_mlp.{name}" for name in self.coordinate_mlp.state_dict()}
        if set(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                "source model weights do not match the coordinate-conditioned base: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )

    def forward(self, x: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        if coordinates.ndim != 2 or coordinates.shape != (x.shape[0], 2):
            raise ValueError(
                f"coordinates must have shape [B, 2], got {tuple(coordinates.shape)}"
            )
        if not bool(torch.isfinite(coordinates).all()):
            raise ValueError("coordinates contain non-finite values")
        original_height, original_width = x.shape[2:]
        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x_first = self.conv_first(x)
        modulation = self.coordinate_mlp(
            coordinates.to(device=x_first.device, dtype=x_first.dtype)
        )
        gamma, beta = modulation.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        x_first = x_first * (1.0 + gamma) + beta
        residual = self.conv_after_body(self.forward_features(x_first)) + x_first
        x = x + self.conv_last(residual)

        x = x / self.img_range + self.mean
        return x[:, :, :original_height, :original_width]


def load_source_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != SOURCE_CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported consolidation checkpoint schema version")
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") != "full_set_gradient_consolidation":
        raise RuntimeError("source checkpoint is not a full-set consolidation run")
    if not isinstance(config.get("selection_manifest"), dict):
        raise RuntimeError("source checkpoint is missing its selection manifest")
    if not isinstance(checkpoint.get("model"), dict):
        raise RuntimeError("source checkpoint is missing raw model weights")
    return checkpoint


@torch.no_grad()
def predict_conditioned(
    model: CoordinateConditionedSwinIR,
    inputs: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with precision.autocast():
        prediction = model(
            inputs.to(device, non_blocking=device.type == "cuda"),
            coordinates.to(device, non_blocking=device.type == "cuda"),
        )
    if was_training:
        model.train()
    return prediction.detach().float().cpu()


@torch.no_grad()
def evaluate_models(
    model: CoordinateConditionedSwinIR,
    ema_model: CoordinateConditionedSwinIR,
    samples: Sequence[LoadedSample],
    coordinates: Sequence[torch.Tensor],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    floor_db: float,
    charbonnier_epsilon: float,
    criteria: SuccessCriteria,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    raw_metrics: dict[str, dict[str, float]] = {}
    ema_metrics: dict[str, dict[str, float]] = {}
    for sample, coordinate in zip(samples, coordinates, strict=True):
        raw_prediction = predict_conditioned(
            model,
            sample.inputs,
            coordinate,
            device=device,
            precision=precision,
        )
        ema_prediction = predict_conditioned(
            ema_model,
            sample.inputs,
            coordinate,
            device=device,
            precision=precision,
        )
        predictions[sample.filename] = (raw_prediction, ema_prediction)
        raw_metrics[sample.filename] = evaluate_prediction(
            raw_prediction,
            sample.targets,
            target_peak=sample.target_peak,
            floor_db=floor_db,
            charbonnier_epsilon=charbonnier_epsilon,
        )
        ema_metrics[sample.filename] = evaluate_prediction(
            ema_prediction,
            sample.targets,
            target_peak=sample.target_peak,
            floor_db=floor_db,
            charbonnier_epsilon=charbonnier_epsilon,
        )
    return predictions, {
        "raw": summarize_metric_map(raw_metrics, criteria),
        "ema": summarize_metric_map(ema_metrics, criteria),
    }


def train_full_set_epoch(
    model: CoordinateConditionedSwinIR,
    ema_model: CoordinateConditionedSwinIR,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    samples: Sequence[LoadedSample],
    coordinates: Sequence[torch.Tensor],
    order: Sequence[int],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
    ema_decay: float,
) -> TrainStepResult:
    if len(order) != len(samples) or sorted(order) != list(range(len(samples))):
        raise ValueError("order must contain every sample index exactly once")
    if len(coordinates) != len(samples):
        raise ValueError("coordinates and samples must have equal lengths")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    scale_before = float(scaler.get_scale()) if precision.uses_grad_scaler else None
    for index in order:
        sample = samples[index]
        inputs = sample.inputs.to(device, non_blocking=device.type == "cuda")
        targets = sample.targets.to(device, non_blocking=device.type == "cuda")
        coordinate = coordinates[index].to(device, non_blocking=device.type == "cuda")
        with precision.autocast():
            prediction = model(inputs, coordinate)
        loss = complex_charbonnier_loss(prediction, targets, loss_epsilon)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss for sample {sample.filename}")
        loss_sum += float(loss.detach().item())
        component = loss / len(samples)
        if precision.uses_grad_scaler:
            scaler.scale(component).backward()
        else:
            component.backward()

    if precision.uses_grad_scaler:
        scaler.unscale_(optimizer)
    gradient_norm = global_gradient_norm(model.parameters())
    if not math.isfinite(gradient_norm):
        raise FloatingPointError("non-finite coordinate-conditioned gradient norm")
    if precision.uses_grad_scaler:
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        did_step = scale_after >= float(scale_before)
    else:
        optimizer.step()
        scale_after = None
        did_step = True
    if did_step:
        scheduler.step()
        update_ema(ema_model, model, ema_decay)
    return TrainStepResult(
        loss=loss_sum / len(samples),
        gradient_norm=gradient_norm,
        did_optimizer_step=did_step,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
    )


def make_resolved_config(
    args: argparse.Namespace,
    *,
    source_config: dict[str, Any],
    source_sha256: str,
    bounds: CoordinateBounds,
    precision: PrecisionPolicy,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "coordinate_conditioned_patch_set_overfit",
        "source": {
            "checkpoint": str(args.source_checkpoint.resolve()),
            "checkpoint_sha256": source_sha256,
            "initial_weights": "raw_model",
            "optimizer_state_reused": False,
        },
        "selection_manifest": source_config["selection_manifest"],
        "model": source_config["model"],
        "conditioning": {
            "kind": "global_row_col_shallow_feature_film",
            "bounds": asdict(bounds),
            "coordinate_range": [-1.0, 1.0],
            "hidden_dim": int(args.coordinate_hidden_dim),
            "final_layer_zero_initialized": True,
        },
        "data": source_config["data"],
        "optimization": {
            "optimizer": "adam",
            "base_learning_rate": float(args.base_learning_rate),
            "condition_learning_rate": float(args.condition_learning_rate),
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate_schedule": "constant",
            "charbonnier_epsilon": float(
                source_config["optimization"]["charbonnier_epsilon"]
            ),
            "ema_decay": float(args.ema_decay),
            "max_epochs": int(args.epochs),
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
    for name in (
        "epochs",
        "eval_every",
        "save_every",
        "required_consecutive_successes",
        "coordinate_hidden_dim",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    minimum_epochs = args.eval_every * args.required_consecutive_successes
    if args.epochs < minimum_epochs:
        raise ValueError(
            "epochs must allow all required post-update evaluations: "
            f"epochs={args.epochs}, required minimum={minimum_epochs}"
        )
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    for name in ("base_learning_rate", "condition_learning_rate"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not args.source_checkpoint.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {args.source_checkpoint}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    source_sha256 = file_sha256(args.source_checkpoint)
    source_checkpoint = load_source_checkpoint(args.source_checkpoint)
    source_config = source_checkpoint["resolved_config"]
    manifest = source_config["selection_manifest"]
    criteria = success_criteria_from_source(source_config)
    expected_shape = tuple(int(value) for value in source_config["data"]["expected_shape"])
    samples = load_selected_samples(
        source_pairs(manifest),
        expected_shape=expected_shape,
        rms_epsilon=float(source_config["data"]["rms_epsilon"]),
    )
    verify_loaded_samples(samples, manifest)
    bounds = coordinate_bounds(samples)
    coordinates = tuple(bounds.encode(sample.row, sample.col) for sample in samples)

    device = resolve_device(args.device)
    precision = resolve_precision(device)
    resolved_config = make_resolved_config(
        args,
        source_config=source_config,
        source_sha256=source_sha256,
        bounds=bounds,
        precision=precision,
    )
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    manifest_path = paths.root / "selected_samples.json"
    if args.resume is None:
        write_json(paths.resolved_config, resolved_config)
        write_json(manifest_path, manifest)
    else:
        if not paths.resolved_config.is_file():
            raise FileNotFoundError("resume output directory is missing resolved_config.json")
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing != resolved_config:
            raise RuntimeError("output directory resolved_config.json does not match this run")

    set_seed(int(args.seed))
    model = CoordinateConditionedSwinIR(
        **source_config["model"],
        coordinate_hidden_dim=int(args.coordinate_hidden_dim),
    ).to(device)
    model.load_unconditioned_state_dict(source_checkpoint["model"])
    ema_model = make_ema_model(model).to(device)
    del source_checkpoint
    gc.collect()

    condition_parameters = list(model.coordinate_mlp.parameters())
    condition_ids = {id(parameter) for parameter in condition_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in condition_ids
    ]
    optimizer = Adam(
        [
            {"params": base_parameters, "lr": float(args.base_learning_rate)},
            {"params": condition_parameters, "lr": float(args.condition_learning_rate)},
        ],
        betas=(0.9, 0.99),
        eps=1.0e-8,
        weight_decay=0.0,
    )
    scheduler = LambdaLR(optimizer, lr_lambda=[lambda _: 1.0, lambda _: 1.0])
    scaler = make_grad_scaler(precision)
    loss_epsilon = float(source_config["optimization"]["charbonnier_epsilon"])
    floor_db = float(source_config["evaluation"]["log_magnitude_floor_db"])
    anchor_filename = str(manifest["anchor_filename"])

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
        predictions, model_metrics = evaluate_models(
            model,
            ema_model,
            samples,
            coordinates,
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
            f"epoch={epoch} train_loss="
            f"{train_loss if train_loss is not None else float('nan'):.6f} "
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
        worst_rmse = float(aggregate["normalized_complex_rmse"]["max"])
        if worst_rmse < best_worst_rmse:
            best_worst_rmse = worst_rmse
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
            result = train_full_set_epoch(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                samples,
                coordinates,
                shuffled_epoch_order(epoch, len(samples), int(args.seed)),
                device=device,
                precision=precision,
                loss_epsilon=loss_epsilon,
                ema_decay=float(args.ema_decay),
            )
            if not result.did_optimizer_step:
                overflow_streak += 1
                if overflow_streak >= 8:
                    raise FloatingPointError("eight consecutive mixed-precision overflows")
                continue
            overflow_streak = 0
            epoch += 1
            last_train_loss = result.loss
            last_gradient_norm = result.gradient_norm
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
        last_predictions, _ = evaluate_models(
            model,
            ema_model,
            samples,
            coordinates,
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
        "success_epoch": success_epoch,
        "sample_presentations": epoch * len(samples),
        "success_criteria": asdict(criteria),
        "source": resolved_config["source"],
        "conditioning": resolved_config["conditioning"],
        "selection_manifest": manifest,
        "precision": precision.as_dict(),
        "final": last_metrics,
        "best_raw_worst_sample_normalized_complex_rmse": best_worst_rmse,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "representative_samples": artifact_samples,
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit the fixed patch set with normalized global row/col conditioning."
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
    parser.add_argument("--coordinate-hidden-dim", type=int, default=64)
    parser.add_argument("--base-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--condition-learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
