"""Jointly overfit a deterministic patch set in the log-magnitude domain."""

from __future__ import annotations

import argparse
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

from scripts.overfit_patch_set import (
    DEFAULT_ANCHOR,
    restore_checkpoint,
    sample_artifact_paths,
    save_checkpoint,
    select_spatially_distributed_pairs,
    selection_manifest,
    shuffled_sample_index,
)
from scripts.overfit_single_magnitude_patch import (
    MagnitudeSuccessCriteria,
    evaluate_log_magnitude_prediction,
    prepare_log_magnitude_pair,
    save_artifacts,
    train_magnitude_step,
)
from scripts.overfit_single_patch import (
    RunPaths,
    append_jsonl,
    load_base_config,
    make_run_paths,
    predict,
    sample_fingerprint,
    set_seed,
    utc_now,
    write_json,
)
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair, discover_pairs, load_complex_patch
from swinir.training import (
    PrecisionPolicy,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
)


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
class LoadedMagnitudeSample:
    filename: str
    row: int
    col: int
    inputs: torch.Tensor
    targets: torch.Tensor
    scale: float
    fingerprint: dict[str, Any]


def load_selected_samples(
    pairs: Sequence[DiscoveredPair],
    *,
    expected_shape: tuple[int, int],
    rms_epsilon: float,
) -> tuple[LoadedMagnitudeSample, ...]:
    loaded = []
    for pair in pairs:
        echo = load_complex_patch(pair.echo_path, expected_shape)
        image = load_complex_patch(pair.image_path, expected_shape)
        inputs, targets, scale = prepare_log_magnitude_pair(
            echo,
            image,
            epsilon=rms_epsilon,
        )
        if float(targets.max()) <= 0:
            raise ValueError(f"Image target has zero magnitude everywhere: {pair.image_path}")
        loaded.append(
            LoadedMagnitudeSample(
                filename=pair.echo_path.name,
                row=pair.row,
                col=pair.col,
                inputs=inputs.unsqueeze(0),
                targets=targets.unsqueeze(0),
                scale=scale,
                fingerprint=sample_fingerprint(pair.echo_path, pair.image_path),
            )
        )
    return tuple(loaded)


def summarize_metric_map(
    metrics_by_filename: dict[str, dict[str, float]],
    criteria: MagnitudeSuccessCriteria,
) -> dict[str, Any]:
    if not metrics_by_filename:
        raise ValueError("cannot summarize an empty metric set")
    aggregate: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        values = np.asarray(
            [metrics[name] for metrics in metrics_by_filename.values()],
            dtype=np.float64,
        )
        aggregate[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    passed = {
        filename: criteria.is_satisfied(metrics)
        for filename, metrics in metrics_by_filename.items()
    }
    worst_filename = max(
        metrics_by_filename,
        key=lambda filename: (
            metrics_by_filename[filename]["normalized_log_rmse"],
            filename,
        ),
    )
    return {
        "per_sample": metrics_by_filename,
        "per_sample_passed": passed,
        "pass_count": sum(passed.values()),
        "sample_count": len(passed),
        "all_passed": all(passed.values()),
        "aggregate": aggregate,
        "worst_sample_by_rmse": worst_filename,
    }


@torch.no_grad()
def evaluate_patch_set(
    model: nn.Module,
    ema_model: nn.Module,
    samples: Sequence[LoadedMagnitudeSample],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    charbonnier_epsilon: float,
    criteria: MagnitudeSuccessCriteria,
) -> tuple[
    dict[str, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[str, Any]],
]:
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    raw_metrics: dict[str, dict[str, float]] = {}
    ema_metrics: dict[str, dict[str, float]] = {}
    for sample in samples:
        raw_prediction = predict(
            model,
            sample.inputs,
            device=device,
            precision=precision,
        )
        ema_prediction = predict(
            ema_model,
            sample.inputs,
            device=device,
            precision=precision,
        )
        predictions[sample.filename] = (raw_prediction, ema_prediction)
        raw_metrics[sample.filename] = evaluate_log_magnitude_prediction(
            raw_prediction,
            sample.targets,
            charbonnier_epsilon=charbonnier_epsilon,
        )
        ema_metrics[sample.filename] = evaluate_log_magnitude_prediction(
            ema_prediction,
            sample.targets,
            charbonnier_epsilon=charbonnier_epsilon,
        )
    return predictions, {
        "raw": summarize_metric_map(raw_metrics, criteria),
        "ema": summarize_metric_map(ema_metrics, criteria),
    }


def baseline_metrics(
    samples: Sequence[LoadedMagnitudeSample],
    *,
    charbonnier_epsilon: float,
    criteria: MagnitudeSuccessCriteria,
) -> dict[str, Any]:
    zero: dict[str, dict[str, float]] = {}
    identity: dict[str, dict[str, float]] = {}
    for sample in samples:
        zero[sample.filename] = evaluate_log_magnitude_prediction(
            torch.zeros_like(sample.targets),
            sample.targets,
            charbonnier_epsilon=charbonnier_epsilon,
        )
        identity[sample.filename] = evaluate_log_magnitude_prediction(
            sample.inputs,
            sample.targets,
            charbonnier_epsilon=charbonnier_epsilon,
        )
    return {
        "zero": summarize_metric_map(zero, criteria),
        "echo_identity": summarize_metric_map(identity, criteria),
    }


def save_representative_artifacts(
    paths: RunPaths,
    *,
    step: int,
    samples: Sequence[LoadedMagnitudeSample],
    predictions: dict[str, tuple[torch.Tensor, torch.Tensor]],
    metrics: dict[str, dict[str, Any]],
    anchor_filename: str,
) -> list[str]:
    worst_filename = str(metrics["raw"]["worst_sample_by_rmse"])
    filenames = list(dict.fromkeys((anchor_filename, worst_filename)))
    samples_by_name = {sample.filename: sample for sample in samples}
    for filename in filenames:
        sample = samples_by_name[filename]
        raw_prediction, ema_prediction = predictions[filename]
        save_artifacts(
            sample_artifact_paths(paths, filename),
            step=step,
            inputs=sample.inputs,
            targets=sample.targets,
            raw_prediction=raw_prediction,
            ema_prediction=ema_prediction,
            echo_rms_scale=sample.scale,
            metrics={
                "raw": metrics["raw"]["per_sample"][filename],
                "ema": metrics["ema"]["per_sample"][filename],
            },
        )
    return filenames


def make_resolved_config(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    *,
    manifest: dict[str, Any],
    precision: PrecisionPolicy,
    criteria: MagnitudeSuccessCriteria,
) -> dict[str, Any]:
    model_config = dict(base_config["model"])
    model_config["in_chans"] = 1
    model_config["drop_path_rate"] = 0.0
    charbonnier_epsilon = (
        float(args.charbonnier_epsilon)
        if args.charbonnier_epsilon is not None
        else float(base_config["optimization"]["charbonnier_epsilon"])
    )
    rms_epsilon = (
        float(args.rms_epsilon)
        if args.rms_epsilon is not None
        else float(base_config["data"]["rms_epsilon"])
    )
    return {
        "schema_version": 1,
        "experiment": "D002-B2-A-joint-magnitude-patch-set",
        "base_config": str(args.config.resolve()),
        "selection_manifest": manifest,
        "model": model_config,
        "data": {
            "expected_shape": [int(value) for value in base_config["data"]["expected_shape"]],
            "rms_epsilon": rms_epsilon,
            "input": "log1p(abs(Echo) / rms(Echo))",
            "target": "log1p(abs(Image) / rms(Echo))",
            "normalization_source": "Echo only per pair",
            "physical_batch_size": 1,
            "sampling": "deterministic_shuffled_epochs",
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate_schedule": "constant",
            "loss": "magnitude_charbonnier",
            "charbonnier_epsilon": charbonnier_epsilon,
            "ema_decay": float(args.ema_decay),
            "max_steps": int(args.steps),
        },
        "evaluation": {
            "authority": "raw_model_all_samples",
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "required_consecutive_successes": int(args.required_consecutive_successes),
            "success_criteria": asdict(criteria),
        },
        "runtime": {
            "seed": int(args.seed),
            "device": str(precision.device),
            "precision": precision.as_dict(),
        },
    }


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "sample_count",
        "steps",
        "eval_every",
        "save_every",
        "required_consecutive_successes",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if args.eval_every % args.sample_count != 0:
        raise ValueError("eval_every must be divisible by sample_count (whole epochs)")
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    minimum_steps = args.eval_every * args.required_consecutive_successes
    if args.steps < minimum_steps:
        raise ValueError(
            "steps must allow all required post-update success evaluations: "
            f"steps={args.steps}, required minimum={minimum_steps}"
        )
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    for role, path in (("Echo", args.echo_dir), ("Image", args.image_dir)):
        if not path.is_dir():
            raise FileNotFoundError(f"{role} directory does not exist: {path}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    criteria = MagnitudeSuccessCriteria(
        normalized_log_rmse_max=args.success_rmse,
        log_magnitude_correlation_min=args.success_correlation,
        magnitude_rms_ratio_min=args.success_rms_ratio_min,
        magnitude_rms_ratio_max=args.success_rms_ratio_max,
        log_magnitude_psnr_db_min=args.success_psnr,
        log_magnitude_ssim_min=args.success_ssim,
    )
    criteria.validate()
    base_config = load_base_config(args.config)
    expected_shape = tuple(int(value) for value in base_config["data"]["expected_shape"])
    rms_epsilon = (
        float(args.rms_epsilon)
        if args.rms_epsilon is not None
        else float(base_config["data"]["rms_epsilon"])
    )
    pairs = discover_pairs(args.echo_dir, args.image_dir)
    selected_pairs = select_spatially_distributed_pairs(
        pairs,
        sample_count=int(args.sample_count),
        anchor_filename=args.anchor_filename,
        patch_shape=expected_shape,
    )
    samples = load_selected_samples(
        selected_pairs,
        expected_shape=expected_shape,
        rms_epsilon=rms_epsilon,
    )
    manifest = selection_manifest(
        samples,
        echo_dir=args.echo_dir,
        image_dir=args.image_dir,
        patch_shape=expected_shape,
        anchor_filename=args.anchor_filename,
    )
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    resolved_config = make_resolved_config(
        args,
        base_config,
        manifest=manifest,
        precision=precision,
        criteria=criteria,
    )

    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    manifest_path = paths.root / "selected_samples.json"
    if args.resume is None:
        write_json(paths.resolved_config, resolved_config)
        write_json(manifest_path, manifest)
    else:
        for required_path in (paths.resolved_config, manifest_path):
            if not required_path.is_file():
                raise FileNotFoundError(
                    f"resume output directory is missing {required_path.name}"
                )
        existing_config = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing_config != resolved_config:
            raise RuntimeError("output directory resolved_config.json does not match this run")

    print("selected magnitude patch set:", flush=True)
    for index, sample in enumerate(samples):
        marker = " anchor" if sample.filename == args.anchor_filename else ""
        print(
            f"  [{index:02d}] row={sample.row} col={sample.col} "
            f"file={sample.filename}{marker}",
            flush=True,
        )

    set_seed(int(args.seed))
    model = SwinIR(**resolved_config["model"]).to(device)
    ema_model = make_ema_model(model).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=float(resolved_config["optimization"]["learning_rate"]),
        betas=tuple(resolved_config["optimization"]["betas"]),
        eps=float(resolved_config["optimization"]["epsilon"]),
        weight_decay=float(resolved_config["optimization"]["weight_decay"]),
    )
    scheduler = LambdaLR(optimizer, lr_lambda=lambda _: 1.0)
    scaler = make_grad_scaler(precision)
    loss_epsilon = float(resolved_config["optimization"]["charbonnier_epsilon"])
    baselines = baseline_metrics(
        samples,
        charbonnier_epsilon=loss_epsilon,
        criteria=criteria,
    )

    step = 0
    best_worst_rmse = math.inf
    consecutive_successes = 0
    success_step: int | None = None
    last_metrics: dict[str, Any] = {}
    last_evaluation_step = -1
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
        step = int(checkpoint["step"])
        best_worst_rmse = float(checkpoint["best_worst_rmse"])
        consecutive_successes = int(checkpoint["consecutive_successes"])
        success_step = checkpoint["success_step"]
        last_metrics = dict(checkpoint["last_metrics"])
        print(f"resumed step={step} from {args.resume}", flush=True)

    def checkpoint_kwargs() -> dict[str, Any]:
        return {
            "model": model,
            "ema_model": ema_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "step": step,
            "resolved_config": resolved_config,
            "best_worst_rmse": best_worst_rmse,
            "consecutive_successes": consecutive_successes,
            "success_step": success_step,
            "last_metrics": last_metrics,
        }

    def evaluate_and_record(
        mean_train_loss: float | None,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        nonlocal best_worst_rmse, consecutive_successes, success_step
        nonlocal last_metrics, last_evaluation_step
        predictions, model_metrics = evaluate_patch_set(
            model,
            ema_model,
            samples,
            device=device,
            precision=precision,
            charbonnier_epsilon=loss_epsilon,
            criteria=criteria,
        )
        raw_summary = model_metrics["raw"]
        all_passed = bool(raw_summary["all_passed"])
        if step > 0:
            consecutive_successes = consecutive_successes + 1 if all_passed else 0
            if (
                consecutive_successes >= args.required_consecutive_successes
                and success_step is None
            ):
                success_step = step
        last_metrics = {
            "step": step,
            "completed_epochs": step / len(samples),
            "timestamp_utc": utc_now(),
            "mean_train_loss_since_last_evaluation": mean_train_loss,
            "raw": model_metrics["raw"],
            "ema": model_metrics["ema"],
            "raw_all_passed": all_passed,
            "consecutive_successes": consecutive_successes,
        }
        append_jsonl(paths.metrics, last_metrics)
        last_evaluation_step = step
        aggregate = raw_summary["aggregate"]
        print(
            f"step={step} epochs={step / len(samples):.1f} "
            f"train_loss={mean_train_loss if mean_train_loss is not None else float('nan'):.6f} "
            f"raw_pass={raw_summary['pass_count']}/{len(samples)} "
            f"worst_log_rmse={aggregate['normalized_log_rmse']['max']:.6f} "
            f"min_log_corr={aggregate['log_magnitude_correlation']['min']:.4f} "
            f"rms_ratio=[{aggregate['magnitude_rms_ratio_target']['min']:.4f},"
            f"{aggregate['magnitude_rms_ratio_target']['max']:.4f}] "
            f"min_psnr={aggregate['log_magnitude_psnr_db']['min']:.2f} "
            f"min_ssim={aggregate['log_magnitude_ssim']['min']:.4f} "
            f"pass={all_passed} streak={consecutive_successes}",
            flush=True,
        )
        current_worst_rmse = float(aggregate["normalized_log_rmse"]["max"])
        if current_worst_rmse < best_worst_rmse:
            best_worst_rmse = current_worst_rmse
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_kwargs())
        return predictions

    interrupted = False
    last_predictions: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None
    train_loss_sum = 0.0
    train_loss_count = 0
    try:
        if args.resume is None:
            last_predictions = evaluate_and_record(None)
            save_representative_artifacts(
                paths,
                step=step,
                samples=samples,
                predictions=last_predictions,
                metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                anchor_filename=args.anchor_filename,
            )

        overflow_streak = 0
        while step < args.steps and success_step is None:
            sample_index = shuffled_sample_index(step, len(samples), int(args.seed))
            sample = samples[sample_index]
            result = train_magnitude_step(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                sample.inputs,
                sample.targets,
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
            step += 1
            train_loss_sum += float(result.loss)
            train_loss_count += 1
            if step % args.eval_every == 0:
                last_predictions = evaluate_and_record(train_loss_sum / train_loss_count)
                train_loss_sum = 0.0
                train_loss_count = 0
            if step % args.save_every == 0:
                if last_evaluation_step != step:
                    raise RuntimeError("save interval must coincide with an evaluation")
                if last_predictions is None:
                    raise RuntimeError("artifact checkpoint is missing evaluated predictions")
                save_representative_artifacts(
                    paths,
                    step=step,
                    samples=samples,
                    predictions=last_predictions,
                    metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                    anchor_filename=args.anchor_filename,
                )
                save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(paths.checkpoints / "interrupted.pt", **checkpoint_kwargs())

    if last_evaluation_step != step:
        mean_train_loss = train_loss_sum / train_loss_count if train_loss_count else None
        last_predictions = evaluate_and_record(mean_train_loss)
    if last_predictions is None:
        last_predictions, _ = evaluate_patch_set(
            model,
            ema_model,
            samples,
            device=device,
            precision=precision,
            charbonnier_epsilon=loss_epsilon,
            criteria=criteria,
        )
    artifact_samples = save_representative_artifacts(
        paths,
        step=step,
        samples=samples,
        predictions=last_predictions,
        metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
        anchor_filename=args.anchor_filename,
    )
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_kwargs())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())

    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": "D002-B2-A-joint-magnitude-patch-set",
        "status": status,
        "step": step,
        "completed_epochs": step / len(samples),
        "success_step": success_step,
        "required_consecutive_successes": args.required_consecutive_successes,
        "success_criteria": asdict(criteria),
        "selection_manifest": manifest,
        "representation": resolved_config["data"],
        "precision": precision.as_dict(),
        "baselines": baselines,
        "final": last_metrics,
        "best_raw_worst_sample_normalized_log_rmse": best_worst_rmse,
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
            "Jointly overfit deterministic non-overlapping Echo/Image patches in "
            "the shared-scale log-magnitude domain."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_magnitude.yaml"))
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-filename", default=DEFAULT_ANCHOR)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=64000)
    parser.add_argument("--eval-every", type=int, default=1600)
    parser.add_argument("--save-every", type=int, default=8000)
    parser.add_argument("--required-consecutive-successes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--charbonnier-epsilon", type=float, default=None)
    parser.add_argument("--rms-epsilon", type=float, default=None)
    parser.add_argument("--success-rmse", type=float, default=0.10)
    parser.add_argument("--success-correlation", type=float, default=0.95)
    parser.add_argument("--success-rms-ratio-min", type=float, default=0.90)
    parser.add_argument("--success-rms-ratio-max", type=float, default=1.10)
    parser.add_argument("--success-psnr", type=float, default=30.0)
    parser.add_argument("--success-ssim", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
