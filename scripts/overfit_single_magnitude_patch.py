"""Overfit one Echo/Image pair in the shared-scale log-magnitude domain.

This is decision 002 experiment B2-A.  It deliberately ignores complex phase:
the model receives ``log1p(abs(Echo) / rms(Echo))`` and predicts
``log1p(abs(Image) / rms(Echo))``.  The target never supplies an independent
normalization statistic, so relative Echo/Image amplitude remains observable.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from scripts.overfit_single_patch import (
    RunPaths,
    append_jsonl,
    load_base_config,
    make_run_paths,
    predict,
    restore_checkpoint,
    sample_fingerprint,
    save_checkpoint,
    set_seed,
    utc_now,
    write_json,
)
from swinir import SwinIR
from swinir.sar_dataset import load_complex_patch
from swinir.sar_metrics import peak_signal_to_noise_ratio, structural_similarity
from swinir.training import (
    PrecisionPolicy,
    TrainStepResult,
    global_gradient_norm,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    update_ema,
)


REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class MagnitudeSuccessCriteria:
    normalized_log_rmse_max: float = 0.10
    log_magnitude_correlation_min: float = 0.95
    magnitude_rms_ratio_min: float = 0.90
    magnitude_rms_ratio_max: float = 1.10
    log_magnitude_psnr_db_min: float = 30.0
    log_magnitude_ssim_min: float = 0.95

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all success criteria must be finite")
        if self.normalized_log_rmse_max < 0:
            raise ValueError("normalized_log_rmse_max must be non-negative")
        for name in ("log_magnitude_correlation_min", "log_magnitude_ssim_min"):
            if not 0.0 <= float(values[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if (
            self.magnitude_rms_ratio_min <= 0
            or self.magnitude_rms_ratio_max < self.magnitude_rms_ratio_min
        ):
            raise ValueError("magnitude RMS ratio bounds must be positive and ordered")

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["normalized_log_rmse"] <= self.normalized_log_rmse_max
            and metrics["log_magnitude_correlation"]
            >= self.log_magnitude_correlation_min
            and self.magnitude_rms_ratio_min
            <= metrics["magnitude_rms_ratio_target"]
            <= self.magnitude_rms_ratio_max
            and metrics["log_magnitude_psnr_db"]
            >= self.log_magnitude_psnr_db_min
            and metrics["log_magnitude_ssim"] >= self.log_magnitude_ssim_min
        )


def prepare_log_magnitude_pair(
    echo: np.ndarray,
    image: np.ndarray,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Create one-channel log-magnitude tensors using Echo RMS for both arrays."""

    echo_array = np.asarray(echo)
    image_array = np.asarray(image)
    if echo_array.shape != image_array.shape or echo_array.ndim != 2:
        raise ValueError(
            f"Echo and Image must be matching 2-D matrices, got "
            f"{echo_array.shape} and {image_array.shape}"
        )
    if not np.iscomplexobj(echo_array) or not np.iscomplexobj(image_array):
        raise ValueError("Echo and Image must contain complex values")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not bool(
        (np.isfinite(echo_array.real) & np.isfinite(echo_array.imag)).all()
        and (np.isfinite(image_array.real) & np.isfinite(image_array.imag)).all()
    ):
        raise ValueError("Echo or Image contains non-finite values")

    scale = math.sqrt(float(np.mean(np.abs(echo_array) ** 2)) + epsilon)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Echo RMS normalization scale is invalid")
    echo_log = np.log1p(np.abs(echo_array) / scale).astype(np.float32)
    image_log = np.log1p(np.abs(image_array) / scale).astype(np.float32)
    return torch.from_numpy(echo_log).unsqueeze(0), torch.from_numpy(image_log).unsqueeze(0), scale


def magnitude_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if prediction.ndim != 4 or prediction.shape[1] != 1:
        raise ValueError(
            f"magnitude tensors must have shape [B, 1, H, W], got {tuple(prediction.shape)}"
        )
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    return torch.sqrt((prediction.float() - target.float()).square() + epsilon**2).mean()


def _tensor_to_image(values: torch.Tensor) -> np.ndarray:
    tensor = values.detach().float().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("only batch size one is supported")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 1:
        raise ValueError(f"expected [1, H, W], got {tuple(tensor.shape)}")
    return tensor[0].numpy().astype(np.float64)


def _pearson_correlation(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction_centered = prediction.ravel() - float(prediction.mean())
    target_centered = target.ravel() - float(target.mean())
    denominator = float(
        np.linalg.norm(prediction_centered) * np.linalg.norm(target_centered)
    )
    if denominator == 0:
        return 0.0
    return float(np.dot(prediction_centered, target_centered) / denominator)


def evaluate_log_magnitude_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    charbonnier_epsilon: float,
) -> dict[str, float]:
    prediction_log = _tensor_to_image(prediction)
    target_log = _tensor_to_image(target)
    target_log_rms = math.sqrt(float(np.mean(target_log**2)))
    if target_log_rms <= 0:
        raise ValueError("Image target has zero log magnitude everywhere")

    normalized_log_rmse = (
        math.sqrt(float(np.mean((prediction_log - target_log) ** 2))) / target_log_rms
    )
    prediction_magnitude = np.expm1(np.maximum(prediction_log, 0.0))
    target_magnitude = np.expm1(np.maximum(target_log, 0.0))
    target_magnitude_rms = math.sqrt(float(np.mean(target_magnitude**2)))
    prediction_magnitude_rms = math.sqrt(float(np.mean(prediction_magnitude**2)))
    magnitude_rms_ratio = (
        prediction_magnitude_rms / target_magnitude_rms
        if target_magnitude_rms > 0
        else math.inf
    )

    target_peak = float(target_log.max())
    if target_peak <= 0:
        raise ValueError("Image target has no positive log magnitude")
    prediction_display = np.clip(prediction_log / target_peak, 0.0, 1.0)
    target_display = np.clip(target_log / target_peak, 0.0, 1.0)
    return {
        "normalized_log_rmse": normalized_log_rmse,
        "log_magnitude_correlation": _pearson_correlation(prediction_log, target_log),
        "magnitude_rms_ratio_target": magnitude_rms_ratio,
        "log_magnitude_psnr_db": peak_signal_to_noise_ratio(
            prediction_display, target_display
        ),
        "log_magnitude_ssim": structural_similarity(
            prediction_display, target_display
        ),
        "magnitude_charbonnier": float(
            magnitude_charbonnier_loss(
                prediction, target, charbonnier_epsilon
            ).item()
        ),
    }


def evaluate_models(
    model: nn.Module,
    ema_model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    charbonnier_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    raw_prediction = predict(model, inputs, device=device, precision=precision)
    ema_prediction = predict(ema_model, inputs, device=device, precision=precision)
    return raw_prediction, ema_prediction, {
        "raw": evaluate_log_magnitude_prediction(
            raw_prediction, targets, charbonnier_epsilon=charbonnier_epsilon
        ),
        "ema": evaluate_log_magnitude_prediction(
            ema_prediction, targets, charbonnier_epsilon=charbonnier_epsilon
        ),
    }


def train_magnitude_step(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
    ema_decay: float,
) -> TrainStepResult:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_inputs = inputs.to(device, non_blocking=device.type == "cuda")
    device_targets = targets.to(device, non_blocking=device.type == "cuda")
    with precision.autocast():
        predictions = model(device_inputs)
    loss = magnitude_charbonnier_loss(predictions, device_targets, loss_epsilon)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite loss encountered during training")
    loss_value = float(loss.detach().item())

    if precision.uses_grad_scaler:
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = global_gradient_norm(model.parameters())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        did_optimizer_step = scale_after >= scale_before
    else:
        loss.backward()
        gradient_norm = global_gradient_norm(model.parameters())
        optimizer.step()
        scale_before = None
        scale_after = None
        did_optimizer_step = True

    if did_optimizer_step:
        scheduler.step()
        update_ema(ema_model, model, ema_decay)
    return TrainStepResult(
        loss=loss_value,
        gradient_norm=gradient_norm,
        did_optimizer_step=did_optimizer_step,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
    )


def save_artifacts(
    paths: RunPaths,
    *,
    step: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    raw_prediction: torch.Tensor,
    ema_prediction: torch.Tensor,
    echo_rms_scale: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    arrays = [
        _tensor_to_image(inputs),
        _tensor_to_image(raw_prediction),
        _tensor_to_image(ema_prediction),
        _tensor_to_image(targets),
    ]
    target_peak = float(arrays[-1].max())
    displays = [np.clip(array / target_peak, 0.0, 1.0) for array in arrays]
    titles = ("Echo input", "Raw prediction", "EMA prediction", "Image target")
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    image_handle = None
    difference_handle = None
    for column, (display, title) in enumerate(zip(displays, titles, strict=True)):
        image_handle = axes[0, column].imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
        axes[0, column].set_title(title)
        difference = np.abs(display - displays[-1])
        difference_handle = axes[1, column].imshow(
            difference, cmap="magma", vmin=0.0, vmax=1.0
        )
        axes[1, column].set_title(f"|{title} - target|")
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    raw_metrics = metrics["raw"]
    figure.suptitle(
        f"B2-A step={step}  log-RMSE={raw_metrics['normalized_log_rmse']:.4f}  "
        f"corr={raw_metrics['log_magnitude_correlation']:.4f}  "
        f"PSNR={raw_metrics['log_magnitude_psnr_db']:.2f} dB  "
        f"SSIM={raw_metrics['log_magnitude_ssim']:.4f}"
    )
    if image_handle is not None:
        figure.colorbar(image_handle, ax=axes[0, :], shrink=0.8, label="target-peak log scale")
    if difference_handle is not None:
        figure.colorbar(difference_handle, ax=axes[1, :], shrink=0.8, label="absolute difference")
    figure.savefig(paths.figures / f"step_{step:06d}.png", dpi=160)
    plt.close(figure)

    raw_log = arrays[1]
    ema_log = arrays[2]
    savemat(
        paths.predictions / f"step_{step:06d}.mat",
        {
            "raw_log_magnitude": raw_log,
            "ema_log_magnitude": ema_log,
            "raw_magnitude": np.expm1(np.maximum(raw_log, 0.0)) * echo_rms_scale,
            "ema_magnitude": np.expm1(np.maximum(ema_log, 0.0)) * echo_rms_scale,
            "echo_rms_scale": np.asarray(echo_rms_scale, dtype=np.float64),
            "step": np.asarray(step, dtype=np.int64),
        },
        do_compression=True,
    )


def make_resolved_config(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    *,
    fingerprint: dict[str, Any],
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
        "experiment": "D002-B2-A-single-patch",
        "base_config": str(args.config.resolve()),
        "sample": fingerprint,
        "model": model_config,
        "data": {
            "expected_shape": [int(value) for value in base_config["data"]["expected_shape"]],
            "rms_epsilon": rms_epsilon,
            "input": "log1p(abs(Echo) / rms(Echo))",
            "target": "log1p(abs(Image) / rms(Echo))",
            "normalization_source": "Echo only",
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
    for name in ("steps", "eval_every", "save_every", "required_consecutive_successes"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    minimum_steps = args.eval_every * args.required_consecutive_successes
    if args.steps < minimum_steps:
        raise ValueError(
            "steps must allow all required post-update success evaluations: "
            f"steps={args.steps}, required minimum={minimum_steps}"
        )
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if args.echo_file.name != args.image_file.name:
        raise ValueError("Echo and Image filenames must match exactly")
    for role, path in (("Echo", args.echo_file), ("Image", args.image_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{role} file does not exist: {path}")
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
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    fingerprint = sample_fingerprint(args.echo_file, args.image_file)
    resolved_config = make_resolved_config(
        args,
        base_config,
        fingerprint=fingerprint,
        precision=precision,
        criteria=criteria,
    )
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    if args.resume is None:
        write_json(paths.resolved_config, resolved_config)
    else:
        if not paths.resolved_config.is_file():
            raise FileNotFoundError("resume output directory is missing resolved_config.json")
        import json

        existing_config = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing_config != resolved_config:
            raise RuntimeError("output directory resolved_config.json does not match this run")

    set_seed(int(args.seed))
    expected_shape = tuple(resolved_config["data"]["expected_shape"])
    echo = load_complex_patch(args.echo_file, expected_shape)
    image = load_complex_patch(args.image_file, expected_shape)
    input_tensor, target_tensor, echo_rms_scale = prepare_log_magnitude_pair(
        echo,
        image,
        epsilon=float(resolved_config["data"]["rms_epsilon"]),
    )
    inputs = input_tensor.unsqueeze(0)
    targets = target_tensor.unsqueeze(0)
    if float(targets.max()) <= 0:
        raise ValueError("Image target has zero magnitude everywhere")

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
    baseline_metrics = {
        "zero": evaluate_log_magnitude_prediction(
            torch.zeros_like(targets), targets, charbonnier_epsilon=loss_epsilon
        ),
        "echo_identity": evaluate_log_magnitude_prediction(
            inputs, targets, charbonnier_epsilon=loss_epsilon
        ),
    }

    step = 0
    best_rmse = math.inf
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
        best_rmse = float(checkpoint["best_rmse"])
        consecutive_successes = int(checkpoint["consecutive_successes"])
        success_step = checkpoint["success_step"]
        last_metrics = dict(checkpoint["last_metrics"])
        print(f"resumed step={step} from {args.resume}", flush=True)

    def evaluate_and_record(train_loss: float | None) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal best_rmse, consecutive_successes, success_step, last_metrics
        nonlocal last_evaluation_step
        raw_prediction, ema_prediction, model_metrics = evaluate_models(
            model,
            ema_model,
            inputs,
            targets,
            device=device,
            precision=precision,
            charbonnier_epsilon=loss_epsilon,
        )
        passed = criteria.is_satisfied(model_metrics["raw"])
        consecutive_successes = consecutive_successes + 1 if passed else 0
        if consecutive_successes >= args.required_consecutive_successes and success_step is None:
            success_step = step
        last_metrics = {
            "step": step,
            "timestamp_utc": utc_now(),
            "train_loss": train_loss,
            "raw": model_metrics["raw"],
            "ema": model_metrics["ema"],
            "raw_passed": passed,
            "consecutive_successes": consecutive_successes,
        }
        append_jsonl(paths.metrics, last_metrics)
        last_evaluation_step = step
        raw = model_metrics["raw"]
        print(
            f"step={step} loss={raw['magnitude_charbonnier']:.6f} "
            f"log_rmse={raw['normalized_log_rmse']:.6f} "
            f"log_corr={raw['log_magnitude_correlation']:.4f} "
            f"rms_ratio={raw['magnitude_rms_ratio_target']:.4f} "
            f"psnr={raw['log_magnitude_psnr_db']:.2f} "
            f"ssim={raw['log_magnitude_ssim']:.4f} "
            f"pass={passed} streak={consecutive_successes}",
            flush=True,
        )
        return raw_prediction, ema_prediction

    def checkpoint_args() -> dict[str, Any]:
        return {
            "model": model,
            "ema_model": ema_model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "step": step,
            "resolved_config": resolved_config,
            "best_rmse": best_rmse,
            "consecutive_successes": consecutive_successes,
            "success_step": success_step,
            "last_metrics": last_metrics,
        }

    def save_best_if_improved() -> None:
        nonlocal best_rmse
        current_rmse = float(last_metrics["raw"]["normalized_log_rmse"])
        if current_rmse < best_rmse:
            best_rmse = current_rmse
            save_checkpoint(paths.checkpoints / "best.pt", **checkpoint_args())

    interrupted = False
    last_raw_prediction: torch.Tensor | None = None
    last_ema_prediction: torch.Tensor | None = None
    try:
        if args.resume is None:
            last_raw_prediction, last_ema_prediction = evaluate_and_record(None)
            save_best_if_improved()
            save_artifacts(
                paths,
                step=step,
                inputs=inputs,
                targets=targets,
                raw_prediction=last_raw_prediction,
                ema_prediction=last_ema_prediction,
                echo_rms_scale=echo_rms_scale,
                metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
            )

        overflow_streak = 0
        last_train_loss: float | None = None
        while step < args.steps and success_step is None:
            result = train_magnitude_step(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                inputs,
                targets,
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
            last_train_loss = result.loss
            if step % args.eval_every == 0:
                last_raw_prediction, last_ema_prediction = evaluate_and_record(last_train_loss)
            if step % args.save_every == 0:
                if last_evaluation_step != step:
                    last_raw_prediction, last_ema_prediction = evaluate_and_record(last_train_loss)
                save_best_if_improved()
                save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_args())
                save_artifacts(
                    paths,
                    step=step,
                    inputs=inputs,
                    targets=targets,
                    raw_prediction=last_raw_prediction,
                    ema_prediction=last_ema_prediction,
                    echo_rms_scale=echo_rms_scale,
                    metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                )
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(paths.checkpoints / "interrupted.pt", **checkpoint_args())

    if last_evaluation_step != step:
        last_raw_prediction, last_ema_prediction = evaluate_and_record(None)
    if last_raw_prediction is None or last_ema_prediction is None:
        last_raw_prediction, last_ema_prediction, _ = evaluate_models(
            model,
            ema_model,
            inputs,
            targets,
            device=device,
            precision=precision,
            charbonnier_epsilon=loss_epsilon,
        )
    save_best_if_improved()
    save_artifacts(
        paths,
        step=step,
        inputs=inputs,
        targets=targets,
        raw_prediction=last_raw_prediction,
        ema_prediction=last_ema_prediction,
        echo_rms_scale=echo_rms_scale,
        metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
    )
    save_checkpoint(paths.checkpoints / "final.pt", **checkpoint_args())
    save_checkpoint(paths.checkpoints / "latest.pt", **checkpoint_args())

    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": "D002-B2-A-single-patch",
        "status": status,
        "step": step,
        "success_step": success_step,
        "required_consecutive_successes": args.required_consecutive_successes,
        "success_criteria": asdict(criteria),
        "sample": fingerprint,
        "representation": resolved_config["data"],
        "echo_rms_scale": echo_rms_scale,
        "precision": precision.as_dict(),
        "baselines": baseline_metrics,
        "final": last_metrics,
        "best_raw_normalized_log_rmse": best_rmse,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "final_figure": str((paths.figures / f"step_{step:06d}.png").resolve()),
            "final_prediction": str((paths.predictions / f"step_{step:06d}.mat").resolve()),
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit one paired patch as Echo-log-magnitude to Image-log-magnitude."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_magnitude.yaml"))
    parser.add_argument("--echo-file", type=Path, required=True)
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
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
