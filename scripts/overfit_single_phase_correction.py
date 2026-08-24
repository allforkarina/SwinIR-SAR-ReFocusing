"""E008: overfit an Echo-only predictor of a supervised phase correction."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import torch
from scipy.io import savemat
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.diagnose_shared_complex_filter import evaluate_focus_prediction
from scripts.overfit_single_patch import (
    RunPaths,
    append_jsonl,
    load_base_config,
    make_run_paths,
    sample_fingerprint,
    set_seed,
    tensor_to_complex,
    utc_now,
    write_json,
)
from swinir import SwinIR
from swinir.sar_dataset import load_complex_patch
from swinir.sar_metrics import log_magnitude_image
from swinir.training import (
    PrecisionPolicy,
    TrainStepResult,
    atomic_torch_save,
    capture_rng_state,
    complex_charbonnier_loss,
    global_gradient_norm,
    make_ema_model,
    make_grad_scaler,
    resolve_device,
    resolve_precision,
    restore_rng_state,
    update_ema,
)


CHECKPOINT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PhaseSuccessCriteria:
    weighted_phase_alignment_min: float = 0.95
    coherence_fraction_of_oracle_min: float = 0.90
    ssim_gain_fraction_of_oracle_min: float = 0.80
    edge_gain_fraction_of_oracle_min: float = 0.75
    rmse_excess_over_oracle_max: float = 0.08
    high_frequency_energy_ratio_min: float = 0.75
    high_frequency_energy_ratio_max: float = 1.25

    def validate(self) -> None:
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all phase success criteria must be finite")
        for name in (
            "weighted_phase_alignment_min",
            "coherence_fraction_of_oracle_min",
            "ssim_gain_fraction_of_oracle_min",
            "edge_gain_fraction_of_oracle_min",
        ):
            if not 0.0 <= float(values[name]) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.rmse_excess_over_oracle_max < 0:
            raise ValueError("rmse_excess_over_oracle_max must be non-negative")
        if (
            self.high_frequency_energy_ratio_min <= 0
            or self.high_frequency_energy_ratio_max
            < self.high_frequency_energy_ratio_min
        ):
            raise ValueError("high-frequency ratio bounds must be positive and ordered")

    def is_satisfied(self, metrics: dict[str, float]) -> bool:
        return (
            metrics["weighted_phase_alignment"]
            >= self.weighted_phase_alignment_min
            and metrics["coherence_fraction_of_oracle"]
            >= self.coherence_fraction_of_oracle_min
            and metrics["ssim_gain_fraction_of_oracle"]
            >= self.ssim_gain_fraction_of_oracle_min
            and metrics["edge_gain_fraction_of_oracle"]
            >= self.edge_gain_fraction_of_oracle_min
            and metrics["rmse_excess_over_oracle"]
            <= self.rmse_excess_over_oracle_max
            and self.high_frequency_energy_ratio_min
            <= metrics["high_frequency_energy_ratio"]
            <= self.high_frequency_energy_ratio_max
        )


def prepare_phase_supervision(
    echo: np.ndarray,
    image: np.ndarray,
    *,
    rms_epsilon: float,
    phasor_epsilon: float,
    energy_weight_power: float,
    fft_norm: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if echo.shape != image.shape or echo.ndim != 2:
        raise ValueError("Echo and Image must be same-shape two-dimensional arrays")
    if not np.iscomplexobj(echo) or not np.iscomplexobj(image):
        raise ValueError("Echo and Image must both be complex")
    if not 0.0 <= energy_weight_power <= 1.0:
        raise ValueError("energy_weight_power must be in [0, 1]")
    scale = math.sqrt(float(np.mean(np.abs(echo) ** 2)) + rms_epsilon)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Echo RMS normalization scale is invalid")
    normalized_echo = echo / scale
    normalized_image = image / scale
    echo_spectrum = np.fft.fftshift(
        np.fft.fft2(normalized_echo, norm=fft_norm)
    )
    image_spectrum = np.fft.fftshift(
        np.fft.fft2(normalized_image, norm=fft_norm)
    )
    cross = image_spectrum * np.conj(echo_spectrum)
    cross_magnitude = np.abs(cross)
    correction = np.ones_like(cross, dtype=np.complex128)
    reliable = cross_magnitude > phasor_epsilon
    correction[reliable] = cross[reliable] / cross_magnitude[reliable]
    base_weight = np.abs(echo_spectrum) * np.abs(image_spectrum)
    weights = np.power(base_weight, energy_weight_power)
    mean_weight = float(weights.mean())
    if not math.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("phase supervision has no positive cross-spectral energy")
    weights /= mean_weight

    input_spectrum = torch.from_numpy(
        np.stack((echo_spectrum.real, echo_spectrum.imag)).astype(np.float32)
    )
    target_phasor = torch.from_numpy(
        np.stack((correction.real, correction.imag)).astype(np.float32)
    )
    phase_weights = torch.from_numpy(weights[None].astype(np.float32))
    target_image = torch.from_numpy(
        np.stack((normalized_image.real, normalized_image.imag)).astype(np.float32)
    )
    return input_spectrum, target_phasor, phase_weights, target_image, scale


def normalize_phasor(values: torch.Tensor, epsilon: float) -> torch.Tensor:
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("phasor tensor must have shape [B, 2, H, W]")
    norm = torch.sqrt(values.float().square().sum(dim=1, keepdim=True) + epsilon**2)
    return values.float() / norm


def apply_phase_correction(
    echo_spectrum: torch.Tensor,
    correction_phasor: torch.Tensor,
    *,
    fft_norm: str,
) -> torch.Tensor:
    if echo_spectrum.shape != correction_phasor.shape:
        raise ValueError("spectrum and correction phasor shapes must match")
    spectrum = torch.complex(
        echo_spectrum[:, 0].float(), echo_spectrum[:, 1].float()
    )
    correction = torch.complex(
        correction_phasor[:, 0].float(), correction_phasor[:, 1].float()
    )
    focused_spectrum = torch.fft.ifftshift(
        correction * spectrum, dim=(-2, -1)
    )
    focused = torch.fft.ifft2(focused_spectrum, norm=fft_norm)
    return torch.stack((focused.real, focused.imag), dim=1)


def phase_loss_components(
    correction_phasor: torch.Tensor,
    target_phasor: torch.Tensor,
    phase_weights: torch.Tensor,
    prediction_image: torch.Tensor,
    target_image: torch.Tensor,
    *,
    phase_loss_weight: float,
    complex_reconstruction_weight: float,
    log_magnitude_weight: float,
    charbonnier_epsilon: float,
) -> dict[str, torch.Tensor]:
    dot = (correction_phasor * target_phasor.float()).sum(dim=1)
    weights = phase_weights[:, 0].float()
    circular = (weights * (1.0 - dot)).sum() / weights.sum()
    complex_reconstruction = complex_charbonnier_loss(
        prediction_image, target_image, charbonnier_epsilon
    )
    prediction_magnitude = torch.sqrt(
        prediction_image[:, 0].float().square()
        + prediction_image[:, 1].float().square()
        + 1.0e-12
    )
    target_magnitude = torch.sqrt(
        target_image[:, 0].float().square()
        + target_image[:, 1].float().square()
        + 1.0e-12
    )
    log_magnitude = torch.mean(
        torch.abs(torch.log1p(prediction_magnitude) - torch.log1p(target_magnitude))
    )
    total = (
        phase_loss_weight * circular
        + complex_reconstruction_weight * complex_reconstruction
        + log_magnitude_weight * log_magnitude
    )
    return {
        "total": total,
        "circular_phase": circular,
        "complex_reconstruction": complex_reconstruction,
        "log_magnitude": log_magnitude,
    }


def _weighted_phase_alignment(
    correction: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
) -> float:
    dot = (correction.float() * target.float()).sum(dim=1)
    weight_values = weights[:, 0].float()
    return float((weight_values * dot).sum().item() / weight_values.sum().item())


def _gain_fraction(candidate: float, baseline: float, oracle: float) -> float:
    oracle_gain = oracle - baseline
    return (candidate - baseline) / oracle_gain if oracle_gain > 0 else 0.0


def evaluate_correction(
    correction: torch.Tensor,
    input_spectrum: torch.Tensor,
    target_phasor: torch.Tensor,
    phase_weights: torch.Tensor,
    target_image: torch.Tensor,
    *,
    fft_norm: str,
    floor_db: float,
    high_frequency_radius_fraction: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = apply_phase_correction(input_spectrum, correction, fft_norm=fft_norm)
    prediction_complex = tensor_to_complex(prediction)
    target_complex = tensor_to_complex(target_image)
    metrics = evaluate_focus_prediction(
        prediction_complex,
        target_complex,
        floor_db=floor_db,
        high_frequency_radius_fraction=high_frequency_radius_fraction,
    )
    metrics["weighted_phase_alignment"] = _weighted_phase_alignment(
        correction, target_phasor, phase_weights
    )
    return prediction, metrics


def add_oracle_relative_metrics(
    candidate: dict[str, float],
    echo: dict[str, float],
    oracle: dict[str, float],
) -> dict[str, float]:
    result = dict(candidate)
    result["coherence_fraction_of_oracle"] = (
        candidate["complex_coherence"] / oracle["complex_coherence"]
        if oracle["complex_coherence"] > 0
        else 0.0
    )
    result["ssim_gain_fraction_of_oracle"] = _gain_fraction(
        candidate["log_magnitude_ssim"],
        echo["log_magnitude_ssim"],
        oracle["log_magnitude_ssim"],
    )
    result["edge_gain_fraction_of_oracle"] = _gain_fraction(
        candidate["edge_correlation"],
        echo["edge_correlation"],
        oracle["edge_correlation"],
    )
    result["rmse_excess_over_oracle"] = max(
        0.0,
        candidate["normalized_complex_rmse"]
        - oracle["normalized_complex_rmse"],
    )
    return result


@torch.no_grad()
def predict_correction(
    model: nn.Module,
    inputs: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    phasor_epsilon: float,
) -> torch.Tensor:
    was_training = model.training
    model.eval()
    with precision.autocast():
        raw = model(inputs.to(device, non_blocking=device.type == "cuda"))
    correction = normalize_phasor(raw, phasor_epsilon).detach().cpu()
    if was_training:
        model.train()
    return correction


def evaluate_models(
    model: nn.Module,
    ema_model: nn.Module,
    inputs: torch.Tensor,
    target_phasor: torch.Tensor,
    phase_weights: torch.Tensor,
    target_image: torch.Tensor,
    echo_metrics: dict[str, float],
    oracle_metrics: dict[str, float],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    phasor_epsilon: float,
    fft_norm: str,
    floor_db: float,
    high_frequency_radius_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, dict[str, float]]]:
    raw_correction = predict_correction(
        model,
        inputs,
        device=device,
        precision=precision,
        phasor_epsilon=phasor_epsilon,
    )
    ema_correction = predict_correction(
        ema_model,
        inputs,
        device=device,
        precision=precision,
        phasor_epsilon=phasor_epsilon,
    )
    raw_prediction, raw_metrics = evaluate_correction(
        raw_correction,
        inputs,
        target_phasor,
        phase_weights,
        target_image,
        fft_norm=fft_norm,
        floor_db=floor_db,
        high_frequency_radius_fraction=high_frequency_radius_fraction,
    )
    ema_prediction, ema_metrics = evaluate_correction(
        ema_correction,
        inputs,
        target_phasor,
        phase_weights,
        target_image,
        fft_norm=fft_norm,
        floor_db=floor_db,
        high_frequency_radius_fraction=high_frequency_radius_fraction,
    )
    return (
        raw_correction,
        ema_correction,
        raw_prediction,
        ema_prediction,
        {
            "raw": add_oracle_relative_metrics(raw_metrics, echo_metrics, oracle_metrics),
            "ema": add_oracle_relative_metrics(ema_metrics, echo_metrics, oracle_metrics),
        },
    )


def train_phase_step(
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: Adam,
    scheduler: LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    inputs: torch.Tensor,
    target_phasor: torch.Tensor,
    phase_weights: torch.Tensor,
    target_image: torch.Tensor,
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    fft_norm: str,
    phasor_epsilon: float,
    loss_config: dict[str, float],
    ema_decay: float,
) -> tuple[TrainStepResult, dict[str, float]]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    device_inputs = inputs.to(device, non_blocking=device.type == "cuda")
    device_target_phasor = target_phasor.to(
        device, non_blocking=device.type == "cuda"
    )
    device_weights = phase_weights.to(device, non_blocking=device.type == "cuda")
    device_target_image = target_image.to(
        device, non_blocking=device.type == "cuda"
    )
    with precision.autocast():
        raw = model(device_inputs)
    correction = normalize_phasor(raw, phasor_epsilon)
    prediction = apply_phase_correction(
        device_inputs, correction, fft_norm=fft_norm
    )
    losses = phase_loss_components(
        correction,
        device_target_phasor,
        device_weights,
        prediction,
        device_target_image,
        phase_loss_weight=loss_config["phase_loss_weight"],
        complex_reconstruction_weight=loss_config[
            "complex_reconstruction_weight"
        ],
        log_magnitude_weight=loss_config["log_magnitude_weight"],
        charbonnier_epsilon=loss_config["charbonnier_epsilon"],
    )
    loss = losses["total"]
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("non-finite phase-correction loss")
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
    result = TrainStepResult(
        loss=float(loss.detach().item()),
        gradient_norm=gradient_norm,
        did_optimizer_step=did_optimizer_step,
        scaler_scale_before=scale_before,
        scaler_scale_after=scale_after,
    )
    return result, {
        name: float(value.detach().item()) for name, value in losses.items()
    }


def save_artifacts(
    paths: RunPaths,
    *,
    step: int,
    echo_image: torch.Tensor,
    target_image: torch.Tensor,
    oracle_prediction: torch.Tensor,
    raw_prediction: torch.Tensor,
    ema_prediction: torch.Tensor,
    raw_correction: torch.Tensor,
    ema_correction: torch.Tensor,
    scale: float,
    floor_db: float,
    metrics: dict[str, dict[str, float]],
) -> None:
    arrays = tuple(
        tensor_to_complex(values)
        for values in (
            echo_image,
            raw_prediction,
            ema_prediction,
            oracle_prediction,
            target_image,
        )
    )
    titles = ("Echo", "Raw predicted phase", "EMA predicted phase", "Oracle phase", "Image")
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    target_peak = max(float(np.abs(arrays[-1]).max()), np.finfo(np.float64).tiny)
    for column, (array, title) in enumerate(zip(arrays, titles, strict=True)):
        own_peak = max(float(np.abs(array).max()), np.finfo(np.float64).tiny)
        axes[0, column].imshow(
            log_magnitude_image(array, reference_peak=own_peak, floor_db=floor_db),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[1, column].imshow(
            log_magnitude_image(array, reference_peak=target_peak, floor_db=floor_db),
            cmap="gray",
            vmin=0,
            vmax=1,
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Independent peak")
    axes[1, 0].set_ylabel("Shared Image peak")
    raw = metrics["raw"]
    figure.suptitle(
        f"E008 step={step} phase alignment={raw['weighted_phase_alignment']:.4f} "
        f"coherence={raw['complex_coherence']:.4f} SSIM={raw['log_magnitude_ssim']:.4f} "
        f"edge={raw['edge_correlation']:.4f} oracle RMSE gap={raw['rmse_excess_over_oracle']:.4f}",
        fontsize=10,
    )
    figure.savefig(paths.figures / f"step_{step:06d}.png", dpi=160)
    plt.close(figure)
    raw_complex = torch.complex(raw_correction[0, 0], raw_correction[0, 1]).numpy()
    ema_complex = torch.complex(ema_correction[0, 0], ema_correction[0, 1]).numpy()
    savemat(
        paths.predictions / f"step_{step:06d}.mat",
        {
            "raw_prediction": arrays[1] * scale,
            "ema_prediction": arrays[2] * scale,
            "oracle_prediction": arrays[3] * scale,
            "raw_phase_correction": raw_complex,
            "ema_phase_correction": ema_complex,
            "normalization_scale": np.asarray(scale),
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
        raise RuntimeError("unsupported E008 checkpoint schema version")
    if checkpoint.get("resolved_config") != resolved_config:
        raise RuntimeError("checkpoint configuration or sample fingerprint does not match")
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
    fingerprint: dict[str, Any],
    precision: PrecisionPolicy,
    criteria: PhaseSuccessCriteria,
) -> dict[str, Any]:
    model = dict(base["model"])
    model["in_chans"] = 2
    model["drop_path_rate"] = 0.0
    optimization = dict(base["optimization"])
    optimization["learning_rate"] = float(args.learning_rate)
    optimization["ema_decay"] = float(args.ema_decay)
    optimization["max_steps"] = int(args.steps)
    return {
        "schema_version": 1,
        "experiment": "E008-D002-supervised-single-phase-correction",
        "base_config": str(args.config.resolve()),
        "sample": fingerprint,
        "model": model,
        "data": {
            **base["data"],
            "input_available_at_inference": "fftshift(FFT2(Echo / RMS(Echo))) only",
            "training_target": "unit phasor of shifted FFT2(Image) * conj(shifted FFT2(Echo))",
            "label_available_at_inference": False,
        },
        "optimization": optimization,
        "evaluation": {
            **base["evaluation"],
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
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.steps < args.eval_every * args.required_consecutive_successes:
        raise ValueError("steps do not allow all required success evaluations")
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
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    fingerprint = sample_fingerprint(args.echo_file, args.image_file)
    resolved = make_resolved_config(
        args, base, fingerprint=fingerprint, precision=precision, criteria=criteria
    )
    paths = make_run_paths(args.output_dir, resuming=args.resume is not None)
    if args.resume is None:
        write_json(paths.resolved_config, resolved)
    else:
        existing = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
        if existing != resolved:
            raise RuntimeError("output directory resolved_config.json does not match")

    set_seed(int(args.seed))
    expected_shape = tuple(int(value) for value in resolved["data"]["expected_shape"])
    echo = load_complex_patch(args.echo_file, expected_shape)
    image = load_complex_patch(args.image_file, expected_shape)
    optimization = resolved["optimization"]
    evaluation = resolved["evaluation"]
    inputs, target_phasor, phase_weights, target_image, scale = prepare_phase_supervision(
        echo,
        image,
        rms_epsilon=float(resolved["data"]["rms_epsilon"]),
        phasor_epsilon=float(optimization["phasor_epsilon"]),
        energy_weight_power=float(optimization["phase_energy_weight_power"]),
        fft_norm=str(resolved["data"]["fft_norm"]),
    )
    inputs = inputs.unsqueeze(0)
    target_phasor = target_phasor.unsqueeze(0)
    phase_weights = phase_weights.unsqueeze(0)
    target_image = target_image.unsqueeze(0)
    identity = torch.zeros_like(target_phasor)
    identity[:, 0] = 1.0
    echo_image, echo_metrics = evaluate_correction(
        identity,
        inputs,
        target_phasor,
        phase_weights,
        target_image,
        fft_norm=str(resolved["data"]["fft_norm"]),
        floor_db=float(evaluation["log_magnitude_floor_db"]),
        high_frequency_radius_fraction=float(
            evaluation["high_frequency_radius_fraction"]
        ),
    )
    oracle_prediction, oracle_metrics = evaluate_correction(
        target_phasor,
        inputs,
        target_phasor,
        phase_weights,
        target_image,
        fft_norm=str(resolved["data"]["fft_norm"]),
        floor_db=float(evaluation["log_magnitude_floor_db"]),
        high_frequency_radius_fraction=float(
            evaluation["high_frequency_radius_fraction"]
        ),
    )

    model = SwinIR(**resolved["model"]).to(device)
    ema_model = make_ema_model(model).to(device)
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
            resolved_config=resolved,
            device=device,
        )
        step = int(checkpoint["step"])
        best_rmse = float(checkpoint["best_rmse"])
        consecutive_successes = int(checkpoint["consecutive_successes"])
        success_step = checkpoint["success_step"]
        last_metrics = dict(checkpoint["last_metrics"])
        print(f"resumed step={step} from {args.resume}", flush=True)

    def evaluate_and_record(train_losses: dict[str, float] | None) -> tuple[torch.Tensor, ...]:
        nonlocal best_rmse, consecutive_successes, success_step, last_metrics
        nonlocal last_evaluation_step
        outputs = evaluate_models(
            model,
            ema_model,
            inputs,
            target_phasor,
            phase_weights,
            target_image,
            echo_metrics,
            oracle_metrics,
            device=device,
            precision=precision,
            phasor_epsilon=float(optimization["phasor_epsilon"]),
            fft_norm=str(resolved["data"]["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        raw_correction, ema_correction, raw_prediction, ema_prediction, metrics = outputs
        passed = criteria.is_satisfied(metrics["raw"])
        consecutive_successes = consecutive_successes + 1 if passed else 0
        if consecutive_successes >= args.required_consecutive_successes and success_step is None:
            success_step = step
        last_metrics = {
            "step": step,
            "timestamp_utc": utc_now(),
            "train_losses": train_losses,
            "raw": metrics["raw"],
            "ema": metrics["ema"],
            "raw_passed": passed,
            "consecutive_successes": consecutive_successes,
        }
        append_jsonl(paths.metrics, last_metrics)
        last_evaluation_step = step
        raw = metrics["raw"]
        print(
            f"step={step} loss={None if train_losses is None else train_losses['total']} "
            f"phase_align={raw['weighted_phase_alignment']:.4f} "
            f"rmse={raw['normalized_complex_rmse']:.4f} "
            f"oracle_gap={raw['rmse_excess_over_oracle']:.4f} "
            f"coherence={raw['complex_coherence']:.4f} "
            f"ssim={raw['log_magnitude_ssim']:.4f} edge={raw['edge_correlation']:.4f} "
            f"hf={raw['high_frequency_energy_ratio']:.4f} "
            f"pass={passed} streak={consecutive_successes}",
            flush=True,
        )
        return raw_correction, ema_correction, raw_prediction, ema_prediction

    def save_best_if_improved() -> None:
        nonlocal best_rmse
        current = float(last_metrics["raw"]["normalized_complex_rmse"])
        if current >= best_rmse:
            return
        best_rmse = current
        save_checkpoint(
            paths.checkpoints / "best.pt",
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            resolved_config=resolved,
            best_rmse=best_rmse,
            consecutive_successes=consecutive_successes,
            success_step=success_step,
            last_metrics=last_metrics,
        )

    def save_current_artifacts(outputs: tuple[torch.Tensor, ...]) -> None:
        save_artifacts(
            paths,
            step=step,
            echo_image=echo_image,
            target_image=target_image,
            oracle_prediction=oracle_prediction,
            raw_prediction=outputs[2],
            ema_prediction=outputs[3],
            raw_correction=outputs[0],
            ema_correction=outputs[1],
            scale=scale,
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            metrics={"raw": last_metrics["raw"], "ema": last_metrics["ema"]},
        )

    interrupted = False
    last_outputs: tuple[torch.Tensor, ...] | None = None
    try:
        if args.resume is None:
            last_outputs = evaluate_and_record(None)
            save_best_if_improved()
            save_current_artifacts(last_outputs)
        overflow_streak = 0
        last_losses: dict[str, float] | None = None
        while step < args.steps and success_step is None:
            result, last_losses = train_phase_step(
                model,
                ema_model,
                optimizer,
                scheduler,
                scaler,
                inputs,
                target_phasor,
                phase_weights,
                target_image,
                device=device,
                precision=precision,
                fft_norm=str(resolved["data"]["fft_norm"]),
                phasor_epsilon=float(optimization["phasor_epsilon"]),
                loss_config=loss_config,
                ema_decay=float(optimization["ema_decay"]),
            )
            if not result.did_optimizer_step:
                overflow_streak += 1
                if overflow_streak >= 8:
                    raise FloatingPointError("eight consecutive mixed-precision overflows")
                continue
            overflow_streak = 0
            step += 1
            if step % args.eval_every == 0:
                last_outputs = evaluate_and_record(last_losses)
            if step % args.save_every == 0:
                if last_evaluation_step != step:
                    last_outputs = evaluate_and_record(last_losses)
                if last_outputs is None:
                    raise RuntimeError("evaluated phase outputs are missing")
                save_current_artifacts(last_outputs)
                save_best_if_improved()
                save_checkpoint(
                    paths.checkpoints / "latest.pt",
                    model=model,
                    ema_model=ema_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    step=step,
                    resolved_config=resolved,
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
            resolved_config=resolved,
            best_rmse=best_rmse,
            consecutive_successes=consecutive_successes,
            success_step=success_step,
            last_metrics=last_metrics,
        )
    if last_evaluation_step != step:
        last_outputs = evaluate_and_record(None)
    if last_outputs is None:
        last_outputs = evaluate_and_record(None)
    save_best_if_improved()
    save_current_artifacts(last_outputs)
    for name in ("latest.pt", "final.pt"):
        save_checkpoint(
            paths.checkpoints / name,
            model=model,
            ema_model=ema_model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            resolved_config=resolved,
            best_rmse=best_rmse,
            consecutive_successes=consecutive_successes,
            success_step=success_step,
            last_metrics=last_metrics,
        )
    status = "interrupted" if interrupted else "passed" if success_step is not None else "failed"
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": resolved["experiment"],
        "status": status,
        "step": step,
        "success_step": success_step,
        "sample": fingerprint,
        "precision": precision.as_dict(),
        "inference_contract": "Echo spectrum is the only model input; Image is unavailable",
        "baselines": {"echo_identity": echo_metrics, "unrestricted_phase_oracle": oracle_metrics},
        "success_criteria": asdict(criteria),
        "final": last_metrics,
        "best_raw_normalized_complex_rmse": best_rmse,
        "artifacts": {
            "root": str(paths.root.resolve()),
            "best_checkpoint": str((paths.checkpoints / "best.pt").resolve()),
            "final_checkpoint": str((paths.checkpoints / "final.pt").resolve()),
            "final_figure": str((paths.figures / f"step_{step:06d}.png").resolve()),
        },
    }
    write_json(paths.report, report)
    print(f"status={status} report={paths.report.resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit one Echo-only SwinIR phase-correction predictor."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_single_phase_correction.yaml"),
    )
    parser.add_argument("--echo-file", type=Path, required=True)
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
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
