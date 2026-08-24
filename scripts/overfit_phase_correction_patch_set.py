"""E009: jointly overfit supervised Echo-only phase correction on a patch set."""

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
    sample_artifact_paths,
    select_spatially_distributed_pairs,
    selection_manifest,
    shuffled_sample_index,
)
from scripts.overfit_single_patch import (
    RunPaths,
    append_jsonl,
    load_base_config,
    make_run_paths,
    sample_fingerprint,
    set_seed,
    utc_now,
    write_json,
)
from scripts.overfit_single_phase_correction import (
    PhaseSuccessCriteria,
    add_oracle_relative_metrics,
    evaluate_correction,
    predict_correction,
    prepare_phase_supervision,
    save_artifacts,
    train_phase_step,
)
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair, discover_pairs, load_complex_patch
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
    "complex_coherence",
    "coherence_fraction_of_oracle",
    "log_magnitude_ssim",
    "ssim_gain_fraction_of_oracle",
    "edge_correlation",
    "edge_gain_fraction_of_oracle",
    "high_frequency_energy_ratio",
)


@dataclass(frozen=True)
class LoadedPhaseSample:
    filename: str
    row: int
    col: int
    input_spectrum: torch.Tensor
    target_phasor: torch.Tensor
    phase_weights: torch.Tensor
    target_image: torch.Tensor
    echo_image: torch.Tensor
    oracle_prediction: torch.Tensor
    echo_metrics: dict[str, float]
    oracle_metrics: dict[str, float]
    scale: float
    fingerprint: dict[str, Any]


def ema_decay_with_warmup(completed_updates: int, target_decay: float) -> float:
    """Return a bias-reducing EMA decay for the next successful update."""

    if completed_updates < 0:
        raise ValueError("completed_updates must be non-negative")
    if not 0.0 <= target_decay < 1.0:
        raise ValueError("target_decay must be in [0, 1)")
    next_update = completed_updates + 1
    return min(target_decay, 1.0 - 1.0 / next_update)


def load_phase_samples(
    pairs: Sequence[DiscoveredPair],
    *,
    expected_shape: tuple[int, int],
    data_config: dict[str, Any],
    optimization: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[LoadedPhaseSample, ...]:
    loaded: list[LoadedPhaseSample] = []
    for pair in pairs:
        echo = load_complex_patch(pair.echo_path, expected_shape)
        image = load_complex_patch(pair.image_path, expected_shape)
        inputs, target, weights, target_image, scale = prepare_phase_supervision(
            echo,
            image,
            rms_epsilon=float(data_config["rms_epsilon"]),
            phasor_epsilon=float(optimization["phasor_epsilon"]),
            energy_weight_power=float(optimization["phase_energy_weight_power"]),
            fft_norm=str(data_config["fft_norm"]),
        )
        inputs = inputs.unsqueeze(0)
        target = target.unsqueeze(0)
        weights = weights.unsqueeze(0)
        target_image = target_image.unsqueeze(0)
        identity = torch.zeros_like(target)
        identity[:, 0] = 1.0
        echo_image, echo_metrics = evaluate_correction(
            identity,
            inputs,
            target,
            weights,
            target_image,
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        oracle_prediction, oracle_metrics = evaluate_correction(
            target,
            inputs,
            target,
            weights,
            target_image,
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        loaded.append(
            LoadedPhaseSample(
                filename=pair.echo_path.name,
                row=pair.row,
                col=pair.col,
                input_spectrum=inputs,
                target_phasor=target,
                phase_weights=weights,
                target_image=target_image,
                echo_image=echo_image,
                oracle_prediction=oracle_prediction,
                echo_metrics=echo_metrics,
                oracle_metrics=oracle_metrics,
                scale=scale,
                fingerprint=sample_fingerprint(pair.echo_path, pair.image_path),
            )
        )
    return tuple(loaded)


def aggregate_metric_map(
    metrics_by_filename: dict[str, dict[str, float]],
    metric_names: Sequence[str],
) -> dict[str, dict[str, float]]:
    if not metrics_by_filename:
        raise ValueError("cannot aggregate an empty metric map")
    result: dict[str, dict[str, float]] = {}
    for name in metric_names:
        values = np.asarray(
            [metrics[name] for metrics in metrics_by_filename.values()],
            dtype=np.float64,
        )
        result[name] = {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def summarize_metric_map(
    metrics_by_filename: dict[str, dict[str, float]],
    criteria: PhaseSuccessCriteria,
) -> dict[str, Any]:
    aggregate = aggregate_metric_map(metrics_by_filename, METRIC_NAMES)
    passed = {
        filename: criteria.is_satisfied(metrics)
        for filename, metrics in metrics_by_filename.items()
    }
    worst_rmse = max(
        metrics_by_filename,
        key=lambda filename: (
            metrics_by_filename[filename]["rmse_excess_over_oracle"],
            filename,
        ),
    )
    worst_phase = min(
        metrics_by_filename,
        key=lambda filename: (
            metrics_by_filename[filename]["weighted_phase_alignment"],
            filename,
        ),
    )
    return {
        "per_sample": metrics_by_filename,
        "per_sample_passed": passed,
        "pass_count": int(sum(passed.values())),
        "sample_count": len(passed),
        "all_passed": all(passed.values()),
        "aggregate": aggregate,
        "worst_sample_by_rmse_excess": worst_rmse,
        "worst_sample_by_phase_alignment": worst_phase,
    }


def baseline_metrics(samples: Sequence[LoadedPhaseSample]) -> dict[str, Any]:
    echo = {sample.filename: sample.echo_metrics for sample in samples}
    oracle = {sample.filename: sample.oracle_metrics for sample in samples}
    baseline_names = tuple(samples[0].echo_metrics)
    return {
        "echo_identity": {
            "per_sample": echo,
            "aggregate": aggregate_metric_map(echo, baseline_names),
        },
        "unrestricted_phase_oracle": {
            "per_sample": oracle,
            "aggregate": aggregate_metric_map(oracle, baseline_names),
        },
    }


@torch.no_grad()
def evaluate_patch_set(
    model: nn.Module,
    ema_model: nn.Module,
    samples: Sequence[LoadedPhaseSample],
    criteria: PhaseSuccessCriteria,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    data_config: dict[str, Any],
    optimization: dict[str, Any],
    evaluation: dict[str, Any],
) -> tuple[dict[str, tuple[torch.Tensor, ...]], dict[str, dict[str, Any]]]:
    predictions: dict[str, tuple[torch.Tensor, ...]] = {}
    raw_metrics: dict[str, dict[str, float]] = {}
    ema_metrics: dict[str, dict[str, float]] = {}
    for sample in samples:
        raw_correction = predict_correction(
            model,
            sample.input_spectrum,
            device=device,
            precision=precision,
            phasor_epsilon=float(optimization["phasor_epsilon"]),
        )
        ema_correction = predict_correction(
            ema_model,
            sample.input_spectrum,
            device=device,
            precision=precision,
            phasor_epsilon=float(optimization["phasor_epsilon"]),
        )
        raw_prediction, raw = evaluate_correction(
            raw_correction,
            sample.input_spectrum,
            sample.target_phasor,
            sample.phase_weights,
            sample.target_image,
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        ema_prediction, ema = evaluate_correction(
            ema_correction,
            sample.input_spectrum,
            sample.target_phasor,
            sample.phase_weights,
            sample.target_image,
            fft_norm=str(data_config["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        raw_metrics[sample.filename] = add_oracle_relative_metrics(
            raw, sample.echo_metrics, sample.oracle_metrics
        )
        ema_metrics[sample.filename] = add_oracle_relative_metrics(
            ema, sample.echo_metrics, sample.oracle_metrics
        )
        predictions[sample.filename] = (
            raw_correction,
            ema_correction,
            raw_prediction,
            ema_prediction,
        )
    return predictions, {
        "raw": summarize_metric_map(raw_metrics, criteria),
        "ema": summarize_metric_map(ema_metrics, criteria),
    }


def save_representative_artifacts(
    paths: RunPaths,
    *,
    step: int,
    samples: Sequence[LoadedPhaseSample],
    predictions: dict[str, tuple[torch.Tensor, ...]],
    metrics: dict[str, dict[str, Any]],
    anchor_filename: str,
    floor_db: float,
) -> list[str]:
    raw_summary = metrics["raw"]
    filenames = list(
        dict.fromkeys(
            (
                anchor_filename,
                str(raw_summary["worst_sample_by_rmse_excess"]),
                str(raw_summary["worst_sample_by_phase_alignment"]),
            )
        )
    )
    save_named_artifacts(
        paths,
        step=step,
        samples=samples,
        predictions=predictions,
        metrics=metrics,
        filenames=filenames,
        floor_db=floor_db,
    )
    return filenames


def save_named_artifacts(
    paths: RunPaths,
    *,
    step: int,
    samples: Sequence[LoadedPhaseSample],
    predictions: dict[str, tuple[torch.Tensor, ...]],
    metrics: dict[str, dict[str, Any]],
    filenames: Sequence[str],
    floor_db: float,
) -> None:
    by_name = {sample.filename: sample for sample in samples}
    for filename in filenames:
        sample = by_name[filename]
        raw_correction, ema_correction, raw_prediction, ema_prediction = predictions[
            filename
        ]
        save_artifacts(
            sample_artifact_paths(paths, filename),
            step=step,
            echo_image=sample.echo_image,
            target_image=sample.target_image,
            oracle_prediction=sample.oracle_prediction,
            raw_prediction=raw_prediction,
            ema_prediction=ema_prediction,
            raw_correction=raw_correction,
            ema_correction=ema_correction,
            scale=sample.scale,
            floor_db=floor_db,
            metrics={
                "raw": metrics["raw"]["per_sample"][filename],
                "ema": metrics["ema"]["per_sample"][filename],
            },
            experiment_label="E009",
        )


def checkpoint_payload(
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    step: int,
    resolved_config: dict[str, Any],
    best_pass_count: int,
    best_worst_rmse_excess: float,
    consecutive_successes: int,
    success_step: int | None,
    last_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": step,
        "model": model.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "rng": capture_rng_state(),
        "resolved_config": resolved_config,
        "best_pass_count": best_pass_count,
        "best_worst_rmse_excess": best_worst_rmse_excess,
        "consecutive_successes": consecutive_successes,
        "success_step": success_step,
        "last_metrics": last_metrics,
    }


def save_checkpoint(path: Path, **kwargs: Any) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def restore_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    resolved_config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported E009 checkpoint schema version")
    if checkpoint.get("resolved_config") != resolved_config:
        raise RuntimeError("checkpoint configuration or phase sample set does not match")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint


def make_resolved_config(
    args: argparse.Namespace,
    base: dict[str, Any],
    *,
    manifest: dict[str, Any],
    precision: PrecisionPolicy,
    criteria: PhaseSuccessCriteria,
) -> dict[str, Any]:
    model = dict(base["model"])
    model["in_chans"] = 2
    model["drop_path_rate"] = 0.0
    optimization = dict(base["optimization"])
    optimization.update(
        {
            "learning_rate": float(args.learning_rate),
            "ema_decay": float(args.ema_decay),
            "ema_schedule": "min(target_decay, 1 - 1 / successful_update)",
            "max_steps": int(args.steps),
        }
    )
    return {
        "schema_version": 1,
        "experiment": "E009-D002-joint-phase-correction-patch-set",
        "base_config": str(args.config.resolve()),
        "selection_manifest": manifest,
        "model": model,
        "data": {
            **base["data"],
            "physical_batch_size": 1,
            "sampling": "deterministic_shuffled_epochs",
            "input_available_at_inference": "fftshift(FFT2(Echo / RMS(Echo))) only",
            "training_target": "per-sample supervised unit phase correction",
            "label_available_at_inference": False,
        },
        "optimization": optimization,
        "evaluation": {
            **base["evaluation"],
            "authority": "raw_model_all_samples_relative_to_each_oracle",
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
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.sample_count < 2:
        raise ValueError("sample_count must be at least 2")
    if args.eval_every % args.sample_count != 0:
        raise ValueError("eval_every must be divisible by sample_count")
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    if args.steps < args.eval_every * args.required_consecutive_successes:
        raise ValueError("steps do not allow all required success evaluations")
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
    criteria = PhaseSuccessCriteria(
        weighted_phase_alignment_min=args.success_phase_alignment,
        coherence_fraction_of_oracle_min=args.success_coherence_fraction,
        ssim_gain_fraction_of_oracle_min=args.success_ssim_gain_fraction,
        edge_gain_fraction_of_oracle_min=args.success_edge_gain_fraction,
        rmse_excess_over_oracle_max=args.success_rmse_excess,
        high_frequency_energy_ratio_min=args.success_hf_ratio_min,
        high_frequency_energy_ratio_max=args.success_hf_ratio_max,
    )
    criteria.validate()
    base = load_base_config(args.config)
    expected_shape = tuple(int(value) for value in base["data"]["expected_shape"])
    pairs = discover_pairs(args.echo_dir, args.image_dir)
    selected_pairs = select_spatially_distributed_pairs(
        pairs,
        sample_count=int(args.sample_count),
        anchor_filename=args.anchor_filename,
        patch_shape=expected_shape,
    )
    samples = load_phase_samples(
        selected_pairs,
        expected_shape=expected_shape,
        data_config=base["data"],
        optimization=base["optimization"],
        evaluation=base["evaluation"],
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
    resolved = make_resolved_config(
        args,
        base,
        manifest=manifest,
        precision=precision,
        criteria=criteria,
    )
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    manifest_path = paths.root / "selected_samples.json"
    if args.resume is None:
        write_json(paths.resolved_config, resolved)
        write_json(manifest_path, manifest)
    else:
        for required in (paths.resolved_config, manifest_path):
            if not required.is_file():
                raise FileNotFoundError(f"resume output is missing {required.name}")
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError("output directory resolved_config.json does not match")

    print("selected E009 phase patch set:", flush=True)
    for index, sample in enumerate(samples):
        marker = " anchor" if sample.filename == args.anchor_filename else ""
        print(
            f"  [{index:02d}] row={sample.row} col={sample.col} "
            f"file={sample.filename}{marker}",
            flush=True,
        )

    set_seed(int(args.seed))
    model = SwinIR(**resolved["model"]).to(device)
    ema_model = make_ema_model(model).to(device)
    optimization = resolved["optimization"]
    evaluation = resolved["evaluation"]
    optimizer = Adam(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        betas=tuple(optimization["betas"]),
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
    baselines = baseline_metrics(samples)
    step = 0
    best_pass_count = -1
    best_worst_rmse_excess = math.inf
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
            resolved_config=resolved,
            device=device,
        )
        step = int(checkpoint["step"])
        best_pass_count = int(checkpoint["best_pass_count"])
        best_worst_rmse_excess = float(checkpoint["best_worst_rmse_excess"])
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
            "resolved_config": resolved,
            "best_pass_count": best_pass_count,
            "best_worst_rmse_excess": best_worst_rmse_excess,
            "consecutive_successes": consecutive_successes,
            "success_step": success_step,
            "last_metrics": last_metrics,
        }

    def evaluate_and_record(
        mean_train_loss: float | None,
    ) -> dict[str, tuple[torch.Tensor, ...]]:
        nonlocal best_pass_count, best_worst_rmse_excess
        nonlocal consecutive_successes, success_step, last_metrics
        nonlocal last_evaluation_step
        predictions, summaries = evaluate_patch_set(
            model,
            ema_model,
            samples,
            criteria,
            device=device,
            precision=precision,
            data_config=resolved["data"],
            optimization=optimization,
            evaluation=evaluation,
        )
        raw = summaries["raw"]
        all_passed = bool(raw["all_passed"])
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
            "raw": summaries["raw"],
            "ema": summaries["ema"],
            "raw_all_passed": all_passed,
            "consecutive_successes": consecutive_successes,
        }
        append_jsonl(paths.metrics, last_metrics)
        last_evaluation_step = step
        aggregate = raw["aggregate"]
        pass_count = int(raw["pass_count"])
        worst_gap = float(aggregate["rmse_excess_over_oracle"]["max"])
        print(
            f"step={step} epochs={step / len(samples):.1f} "
            f"train_loss={mean_train_loss if mean_train_loss is not None else float('nan'):.6f} "
            f"raw_pass={pass_count}/{len(samples)} "
            f"min_phase={aggregate['weighted_phase_alignment']['min']:.4f} "
            f"worst_oracle_gap={worst_gap:.4f} "
            f"min_coh_frac={aggregate['coherence_fraction_of_oracle']['min']:.4f} "
            f"min_ssim_gain={aggregate['ssim_gain_fraction_of_oracle']['min']:.4f} "
            f"min_edge_gain={aggregate['edge_gain_fraction_of_oracle']['min']:.4f} "
            f"hf=[{aggregate['high_frequency_energy_ratio']['min']:.4f},"
            f"{aggregate['high_frequency_energy_ratio']['max']:.4f}] "
            f"pass={all_passed} streak={consecutive_successes}",
            flush=True,
        )
        if pass_count > best_pass_count or (
            pass_count == best_pass_count and worst_gap < best_worst_rmse_excess
        ):
            best_pass_count = pass_count
            best_worst_rmse_excess = worst_gap
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_kwargs())
        return predictions

    interrupted = False
    last_predictions: dict[str, tuple[torch.Tensor, ...]] | None = None
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
                floor_db=float(evaluation["log_magnitude_floor_db"]),
            )
        overflow_streak = 0
        while step < args.steps and success_step is None:
            sample = samples[
                shuffled_sample_index(step, len(samples), int(args.seed))
            ]
            result, losses = train_phase_step(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                sample.input_spectrum,
                sample.target_phasor,
                sample.phase_weights,
                sample.target_image,
                device=device,
                precision=precision,
                fft_norm=str(resolved["data"]["fft_norm"]),
                phasor_epsilon=float(optimization["phasor_epsilon"]),
                loss_config=loss_config,
                ema_decay=ema_decay_with_warmup(step, float(args.ema_decay)),
            )
            if not result.did_optimizer_step:
                overflow_streak += 1
                if overflow_streak >= 8:
                    raise FloatingPointError("eight consecutive mixed-precision overflows")
                continue
            overflow_streak = 0
            step += 1
            train_loss_sum += float(losses["total"])
            train_loss_count += 1
            if step % args.eval_every == 0:
                last_predictions = evaluate_and_record(
                    train_loss_sum / train_loss_count
                )
                train_loss_sum = 0.0
                train_loss_count = 0
            if step % args.save_every == 0:
                if last_evaluation_step != step or last_predictions is None:
                    raise RuntimeError("save interval must coincide with an evaluation")
                save_representative_artifacts(
                    paths,
                    step=step,
                    samples=samples,
                    predictions=last_predictions,
                    metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                    anchor_filename=args.anchor_filename,
                    floor_db=float(evaluation["log_magnitude_floor_db"]),
                )
                save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(paths.checkpoints / "interrupted.pt", **checkpoint_kwargs())

    if last_evaluation_step != step:
        mean_loss = train_loss_sum / train_loss_count if train_loss_count else None
        last_predictions = evaluate_and_record(mean_loss)
    if last_predictions is None:
        last_predictions, _ = evaluate_patch_set(
            model,
            ema_model,
            samples,
            criteria,
            device=device,
            precision=precision,
            data_config=resolved["data"],
            optimization=optimization,
            evaluation=evaluation,
        )
    artifact_samples = [sample.filename for sample in samples]
    save_named_artifacts(
        paths,
        step=step,
        samples=samples,
        predictions=last_predictions,
        metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
        filenames=artifact_samples,
        floor_db=float(evaluation["log_magnitude_floor_db"]),
    )
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_kwargs())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_kwargs())
    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": resolved["experiment"],
        "status": status,
        "step": step,
        "completed_epochs": step / len(samples),
        "success_step": success_step,
        "inference_contract": "Echo spectrum is the only model input; Image is unavailable",
        "selection_manifest": manifest,
        "precision": precision.as_dict(),
        "baselines": baselines,
        "success_criteria": asdict(criteria),
        "final": last_metrics,
        "best": {
            "pass_count": best_pass_count,
            "worst_rmse_excess_over_oracle": best_worst_rmse_excess,
        },
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "selection_manifest": str(manifest_path.resolve()),
            "best_checkpoint": str((paths.checkpoints / "best.pt").resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "final_audit_samples": artifact_samples,
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Jointly overfit an Echo-only phase predictor on a patch set."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_phase_correction_patch_set.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-filename", default=DEFAULT_ANCHOR)
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=64000)
    parser.add_argument("--eval-every", type=int, default=1600)
    parser.add_argument("--save-every", type=int, default=8000)
    parser.add_argument("--required-consecutive-successes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--success-phase-alignment", type=float, default=0.95)
    parser.add_argument("--success-coherence-fraction", type=float, default=0.90)
    parser.add_argument("--success-ssim-gain-fraction", type=float, default=0.80)
    parser.add_argument("--success-edge-gain-fraction", type=float, default=0.75)
    parser.add_argument("--success-rmse-excess", type=float, default=0.08)
    parser.add_argument("--success-hf-ratio-min", type=float, default=0.75)
    parser.add_argument("--success-hf-ratio-max", type=float, default=1.25)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
