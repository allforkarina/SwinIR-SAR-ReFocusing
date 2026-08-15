"""Overfit one paired complex SAR patch as a controlled diagnostic experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
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
import yaml
from scipy.io import savemat
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from main import train_one_step
from swinir import SwinIR
from swinir.sar_dataset import load_complex_patch, normalize_complex_pair
from swinir.sar_metrics import evaluate_complex_prediction, log_magnitude_image
from swinir.training import (
    PrecisionPolicy,
    atomic_torch_save,
    capture_rng_state,
    complex_charbonnier_loss,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    restore_rng_state,
)


CHECKPOINT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SuccessCriteria:
    normalized_complex_rmse_max: float = 0.10
    complex_coherence_min: float = 0.95
    magnitude_correlation_min: float = 0.95
    rms_ratio_min: float = 0.90
    rms_ratio_max: float = 1.10
    log_magnitude_psnr_db_min: float = 30.0
    log_magnitude_ssim_min: float = 0.95

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all success criteria must be finite")
        if self.normalized_complex_rmse_max < 0:
            raise ValueError("normalized_complex_rmse_max must be non-negative")
        for name in (
            "complex_coherence_min",
            "magnitude_correlation_min",
            "log_magnitude_ssim_min",
        ):
            value = float(values[name])
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.rms_ratio_min <= 0 or self.rms_ratio_max < self.rms_ratio_min:
            raise ValueError("RMS ratio bounds must be positive and ordered")

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["normalized_complex_rmse"] <= self.normalized_complex_rmse_max
            and metrics["complex_coherence"] >= self.complex_coherence_min
            and metrics["magnitude_correlation"] >= self.magnitude_correlation_min
            and self.rms_ratio_min
            <= metrics["rms_ratio_target"]
            <= self.rms_ratio_max
            and metrics["log_magnitude_psnr_db"]
            >= self.log_magnitude_psnr_db_min
            and metrics["log_magnitude_ssim"] >= self.log_magnitude_ssim_min
        )


@dataclass(frozen=True)
class RunPaths:
    root: Path
    checkpoints: Path
    figures: Path
    predictions: Path
    resolved_config: Path
    metrics: Path
    report: Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_fingerprint(echo_path: Path, image_path: Path) -> dict[str, Any]:
    return {
        "echo_path": str(echo_path.resolve()),
        "image_path": str(image_path.resolve()),
        "echo_sha256": file_sha256(echo_path),
        "image_sha256": file_sha256(image_path),
        "echo_size_bytes": echo_path.stat().st_size,
        "image_size_bytes": image_path.stat().st_size,
    }


def load_base_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be a mapping")
    for section in ("model", "data", "optimization"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"configuration section {section!r} must be a mapping")
    return config


def make_run_paths(root: Path, *, resuming: bool) -> RunPaths:
    if root.exists() and not resuming:
        raise FileExistsError(
            f"output directory already exists: {root}; choose another path or use --resume"
        )
    root.mkdir(parents=True, exist_ok=resuming)
    checkpoints = root / "checkpoints"
    figures = root / "figures"
    predictions = root / "predictions"
    for directory in (checkpoints, figures, predictions):
        directory.mkdir(exist_ok=True)
    return RunPaths(
        root=root,
        checkpoints=checkpoints,
        figures=figures,
        predictions=predictions,
        resolved_config=root / "resolved_config.json",
        metrics=root / "metrics.jsonl",
        report=root / "report.json",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_to_complex(values: torch.Tensor) -> np.ndarray:
    tensor = values.detach().float().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("only batch size one is supported")
        tensor = tensor[0]
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError(f"expected [2, H, W] complex channels, got {tuple(tensor.shape)}")
    return tensor[0].numpy().astype(np.float64) + 1j * tensor[1].numpy().astype(np.float64)


@torch.no_grad()
def predict(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with precision.autocast():
        prediction = model(inputs.to(device, non_blocking=device.type == "cuda"))
    if was_training:
        model.train()
    return prediction.detach().float().cpu()


def evaluate_prediction(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    target_peak: float,
    floor_db: float,
    charbonnier_epsilon: float,
) -> dict[str, float]:
    metrics = evaluate_complex_prediction(
        tensor_to_complex(prediction),
        tensor_to_complex(target),
        target_peak=target_peak,
        floor_db=floor_db,
    )
    metrics["complex_charbonnier"] = float(
        complex_charbonnier_loss(prediction, target, charbonnier_epsilon).item()
    )
    return metrics


def evaluate_models(
    model: nn.Module,
    ema_model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    target_peak: float,
    floor_db: float,
    charbonnier_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    raw_prediction = predict(model, inputs, device=device, precision=precision)
    ema_prediction = predict(ema_model, inputs, device=device, precision=precision)
    metrics = {
        "raw": evaluate_prediction(
            raw_prediction,
            targets,
            target_peak=target_peak,
            floor_db=floor_db,
            charbonnier_epsilon=charbonnier_epsilon,
        ),
        "ema": evaluate_prediction(
            ema_prediction,
            targets,
            target_peak=target_peak,
            floor_db=floor_db,
            charbonnier_epsilon=charbonnier_epsilon,
        ),
    }
    return raw_prediction, ema_prediction, metrics


def save_artifacts(
    paths: RunPaths,
    *,
    step: int,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    raw_prediction: torch.Tensor,
    ema_prediction: torch.Tensor,
    scale: float,
    target_peak: float,
    floor_db: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    arrays = [
        tensor_to_complex(inputs),
        tensor_to_complex(raw_prediction),
        tensor_to_complex(ema_prediction),
        tensor_to_complex(targets),
    ]
    titles = ("Echo", "Raw prediction", "EMA prediction", "Image target")
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    magnitude_handle = None
    phase_handle = None
    for column, (array, title) in enumerate(zip(arrays, titles, strict=True)):
        normalized_log = log_magnitude_image(
            array, reference_peak=target_peak, floor_db=floor_db
        )
        magnitude_db = floor_db + (-floor_db) * normalized_log
        magnitude_handle = axes[0, column].imshow(
            magnitude_db, cmap="gray", vmin=floor_db, vmax=0.0
        )
        phase_handle = axes[1, column].imshow(
            np.angle(array), cmap="twilight", vmin=-math.pi, vmax=math.pi
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Log magnitude")
    axes[1, 0].set_ylabel("Phase")
    raw_metrics = metrics["raw"]
    figure.suptitle(
        f"step={step}  raw RMSE={raw_metrics['normalized_complex_rmse']:.4f}  "
        f"coherence={raw_metrics['complex_coherence']:.4f}  "
        f"PSNR={raw_metrics['log_magnitude_psnr_db']:.2f} dB  "
        f"SSIM={raw_metrics['log_magnitude_ssim']:.4f}"
    )
    if magnitude_handle is not None:
        figure.colorbar(magnitude_handle, ax=axes[0, :], shrink=0.8, label="dB")
    if phase_handle is not None:
        figure.colorbar(phase_handle, ax=axes[1, :], shrink=0.8, label="radian")
    figure.savefig(paths.figures / f"step_{step:06d}.png", dpi=160)
    plt.close(figure)

    savemat(
        paths.predictions / f"step_{step:06d}.mat",
        {
            "raw_prediction": tensor_to_complex(raw_prediction) * scale,
            "ema_prediction": tensor_to_complex(ema_prediction) * scale,
            "normalization_scale": np.asarray(scale, dtype=np.float64),
            "step": np.asarray(step, dtype=np.int64),
        },
        do_compression=True,
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
    best_rmse: float,
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
        "best_rmse": best_rmse,
        "consecutive_successes": consecutive_successes,
        "success_step": success_step,
        "last_metrics": last_metrics,
    }


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
        raise RuntimeError("unsupported single-patch checkpoint schema version")
    if checkpoint.get("resolved_config") != resolved_config:
        raise RuntimeError("checkpoint configuration or sample fingerprint does not match")
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng"])
    return checkpoint


def save_checkpoint(
    path: Path,
    **kwargs: Any,
) -> None:
    atomic_torch_save(checkpoint_payload(**kwargs), path)


def make_resolved_config(
    args: argparse.Namespace,
    base_config: dict[str, Any],
    *,
    fingerprint: dict[str, Any],
    precision: PrecisionPolicy,
    criteria: SuccessCriteria,
) -> dict[str, Any]:
    model_config = dict(base_config["model"])
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
    expected_shape = [int(value) for value in base_config["data"]["expected_shape"]]
    return {
        "schema_version": 1,
        "base_config": str(args.config.resolve()),
        "sample": fingerprint,
        "model": model_config,
        "data": {
            "expected_shape": expected_shape,
            "rms_epsilon": rms_epsilon,
        },
        "optimization": {
            "optimizer": "adam",
            "learning_rate": float(args.learning_rate),
            "betas": [0.9, 0.99],
            "epsilon": 1.0e-8,
            "weight_decay": 0.0,
            "learning_rate_schedule": "constant",
            "charbonnier_epsilon": charbonnier_epsilon,
            "ema_decay": float(args.ema_decay),
            "max_steps": int(args.steps),
        },
        "evaluation": {
            "eval_every": int(args.eval_every),
            "save_every": int(args.save_every),
            "required_consecutive_successes": int(args.required_consecutive_successes),
            "log_magnitude_floor_db": float(args.db_floor),
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
    minimum_evaluated_steps = args.eval_every * args.required_consecutive_successes
    if args.steps < minimum_evaluated_steps:
        raise ValueError(
            "steps must allow all required post-update success evaluations: "
            f"steps={args.steps}, required minimum={minimum_evaluated_steps}"
        )
    if args.save_every % args.eval_every != 0:
        raise ValueError("save_every must be divisible by eval_every")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("ema_decay must be in [0, 1)")
    if not math.isfinite(args.db_floor) or args.db_floor >= 0:
        raise ValueError("db_floor must be finite and negative")
    if args.echo_file.name != args.image_file.name:
        raise ValueError("Echo and Image filenames must match exactly")
    for role, path in (("Echo", args.echo_file), ("Image", args.image_file)):
        if not path.is_file():
            raise FileNotFoundError(f"{role} file does not exist: {path}")
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint does not exist: {args.resume}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    criteria = SuccessCriteria(
        normalized_complex_rmse_max=args.success_rmse,
        complex_coherence_min=args.success_coherence,
        magnitude_correlation_min=args.success_magnitude_correlation,
        rms_ratio_min=args.success_rms_ratio_min,
        rms_ratio_max=args.success_rms_ratio_max,
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
            raise FileNotFoundError(
                f"resume output directory is missing {paths.resolved_config.name}"
            )
        existing_config = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing_config != resolved_config:
            raise RuntimeError("output directory resolved_config.json does not match this run")

    set_seed(int(args.seed))
    expected_shape = tuple(resolved_config["data"]["expected_shape"])
    echo = load_complex_patch(args.echo_file, expected_shape)
    image = load_complex_patch(args.image_file, expected_shape)
    input_tensor, target_tensor, scale = normalize_complex_pair(
        echo,
        image,
        epsilon=float(resolved_config["data"]["rms_epsilon"]),
    )
    inputs = input_tensor.unsqueeze(0)
    targets = target_tensor.unsqueeze(0)
    target_complex = tensor_to_complex(targets)
    target_peak = float(np.abs(target_complex).max())
    if target_peak <= 0:
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
    charbonnier_epsilon = float(resolved_config["optimization"]["charbonnier_epsilon"])

    baseline_metrics = {
        "zero": evaluate_prediction(
            torch.zeros_like(targets),
            targets,
            target_peak=target_peak,
            floor_db=float(args.db_floor),
            charbonnier_epsilon=charbonnier_epsilon,
        ),
        "echo_identity": evaluate_prediction(
            inputs,
            targets,
            target_peak=target_peak,
            floor_db=float(args.db_floor),
            charbonnier_epsilon=charbonnier_epsilon,
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
            target_peak=target_peak,
            floor_db=float(args.db_floor),
            charbonnier_epsilon=charbonnier_epsilon,
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
        raw_rmse = float(model_metrics["raw"]["normalized_complex_rmse"])
        print(
            f"step={step} loss={model_metrics['raw']['complex_charbonnier']:.6f} "
            f"rmse={raw_rmse:.6f} coherence={model_metrics['raw']['complex_coherence']:.4f} "
            f"mag_corr={model_metrics['raw']['magnitude_correlation']:.4f} "
            f"psnr={model_metrics['raw']['log_magnitude_psnr_db']:.2f} "
            f"ssim={model_metrics['raw']['log_magnitude_ssim']:.4f} "
            f"pass={passed} streak={consecutive_successes}",
            flush=True,
        )
        return raw_prediction, ema_prediction

    def save_best_if_improved() -> bool:
        nonlocal best_rmse
        current_rmse = float(last_metrics["raw"]["normalized_complex_rmse"])
        if current_rmse >= best_rmse:
            return False
        best_rmse = current_rmse
        save_checkpoint(
            paths.checkpoints / "best.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            resolved_config=resolved_config,
            best_rmse=best_rmse,
            consecutive_successes=consecutive_successes,
            success_step=success_step,
            last_metrics=last_metrics,
        )
        return True

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
                scale=scale,
                target_peak=target_peak,
                floor_db=float(args.db_floor),
                metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
            )

        overflow_streak = 0
        last_train_loss: float | None = None
        while step < args.steps and success_step is None:
            model.train()
            train_result = train_one_step(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                {"input": inputs, "target": targets},
                device=device,
                precision=precision,
                loss_epsilon=charbonnier_epsilon,
                ema_decay=float(args.ema_decay),
            )
            if not train_result.did_optimizer_step:
                overflow_streak += 1
                if overflow_streak >= 8:
                    raise FloatingPointError("eight consecutive mixed-precision overflows")
                continue
            overflow_streak = 0
            step += 1
            last_train_loss = train_result.loss

            if step % args.eval_every == 0:
                last_raw_prediction, last_ema_prediction = evaluate_and_record(last_train_loss)
            if step % args.save_every == 0:
                if last_evaluation_step != step:
                    last_raw_prediction, last_ema_prediction = evaluate_and_record(last_train_loss)
                if last_raw_prediction is None or last_ema_prediction is None:
                    raise RuntimeError("artifact checkpoint is missing evaluated predictions")
                save_artifacts(
                    paths,
                    step=step,
                    inputs=inputs,
                    targets=targets,
                    raw_prediction=last_raw_prediction,
                    ema_prediction=last_ema_prediction,
                    scale=scale,
                    target_peak=target_peak,
                    floor_db=float(args.db_floor),
                    metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
                )
                save_best_if_improved()
                save_checkpoint(
                    paths.checkpoints / "latest.pt",
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    resolved_config=resolved_config,
                    best_rmse=best_rmse,
                    consecutive_successes=consecutive_successes,
                    success_step=success_step,
                    last_metrics=last_metrics,
                )
    except KeyboardInterrupt:
        interrupted = True
        save_checkpoint(
            paths.checkpoints / "interrupted.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            resolved_config=resolved_config,
            best_rmse=best_rmse,
            consecutive_successes=consecutive_successes,
            success_step=success_step,
            last_metrics=last_metrics,
        )

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
            target_peak=target_peak,
            floor_db=float(args.db_floor),
            charbonnier_epsilon=charbonnier_epsilon,
        )
    save_best_if_improved()
    save_artifacts(
        paths,
        step=step,
        inputs=inputs,
        targets=targets,
        raw_prediction=last_raw_prediction,
        ema_prediction=last_ema_prediction,
        scale=scale,
        target_peak=target_peak,
        floor_db=float(args.db_floor),
        metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
    )
    save_checkpoint(
        paths.checkpoints / "final.pt",
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        resolved_config=resolved_config,
        best_rmse=best_rmse,
        consecutive_successes=consecutive_successes,
        success_step=success_step,
        last_metrics=last_metrics,
    )
    save_checkpoint(
        paths.checkpoints / "latest.pt",
        model=model,
        ema_model=ema_model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=step,
        resolved_config=resolved_config,
        best_rmse=best_rmse,
        consecutive_successes=consecutive_successes,
        success_step=success_step,
        last_metrics=last_metrics,
    )

    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "status": status,
        "step": step,
        "success_step": success_step,
        "required_consecutive_successes": args.required_consecutive_successes,
        "success_criteria": asdict(criteria),
        "sample": fingerprint,
        "normalization_scale": scale,
        "precision": precision.as_dict(),
        "baselines": baseline_metrics,
        "final": last_metrics,
        "best_raw_normalized_complex_rmse": best_rmse,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "metrics": str(paths.metrics.resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "final_figure": str((paths.figures / f"step_{step:06d}.png").resolve()),
            "final_prediction": str(
                (paths.predictions / f"step_{step:06d}.mat").resolve()
            ),
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit one paired 512x512 complex SAR patch with SwinIR."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/train_sar.yaml"))
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
    parser.add_argument("--db-floor", type=float, default=-60.0)
    parser.add_argument("--success-rmse", type=float, default=0.10)
    parser.add_argument("--success-coherence", type=float, default=0.95)
    parser.add_argument("--success-magnitude-correlation", type=float, default=0.95)
    parser.add_argument("--success-rms-ratio-min", type=float, default=0.90)
    parser.add_argument("--success-rms-ratio-max", type=float, default=1.10)
    parser.add_argument("--success-psnr", type=float, default=30.0)
    parser.add_argument("--success-ssim", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
