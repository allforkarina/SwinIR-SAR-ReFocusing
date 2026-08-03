"""Small, explicit training primitives for complex two-channel SAR patches."""

from __future__ import annotations

import copy
import math
import os
import random
import tempfile
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class PrecisionPolicy:
    """The resolved, recorded mixed-precision policy for one process."""

    device: torch.device
    name: str
    autocast_dtype: torch.dtype | None
    uses_grad_scaler: bool

    def autocast(self) -> AbstractContextManager[Any]:
        if self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(
            device_type=self.device.type,
            dtype=self.autocast_dtype,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": str(self.device),
            "name": self.name,
            "autocast_dtype": (
                str(self.autocast_dtype).replace("torch.", "")
                if self.autocast_dtype is not None
                else None
            ),
            "uses_grad_scaler": self.uses_grad_scaler,
        }


@dataclass(frozen=True)
class TrainStepResult:
    """Information the outer loop needs after consuming one physical batch."""

    loss: float
    gradient_norm: float
    did_optimizer_step: bool
    scaler_scale_before: float | None
    scaler_scale_after: float | None


def resolve_device(requested: str | None = None) -> torch.device:
    """Resolve ``auto`` to cuda:0 when available, otherwise CPU."""

    request = "auto" if requested is None else requested.strip().lower()
    if request in {"", "auto"}:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
    if device.type not in {"cuda", "cpu"}:
        raise ValueError("only CUDA and CPU devices are supported by this trainer")
    return device


def resolve_precision(device: torch.device) -> PrecisionPolicy:
    """Use CUDA BF16 when available, FP16 plus scaling otherwise, CPU FP32."""

    if device.type == "cuda":
        if torch.cuda.is_bf16_supported():
            return PrecisionPolicy(device, "cuda_bf16", torch.bfloat16, False)
        return PrecisionPolicy(device, "cuda_fp16", torch.float16, True)
    return PrecisionPolicy(device, "cpu_fp32", None, False)


def make_grad_scaler(policy: PrecisionPolicy) -> torch.cuda.amp.GradScaler:
    """Create a scaler that is enabled only for the FP16 fallback path."""

    if hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=policy.uses_grad_scaler)
    return torch.cuda.amp.GradScaler(enabled=policy.uses_grad_scaler)


def _validate_complex_tensor_pair(prediction: torch.Tensor, target: torch.Tensor) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target shapes differ: {tuple(prediction.shape)} vs "
            f"{tuple(target.shape)}"
        )
    if prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError(
            "complex SAR tensors must have shape [B, 2, H, W], got "
            f"{tuple(prediction.shape)}"
        )


def complex_charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    epsilon: float = 1e-3,
) -> torch.Tensor:
    """Joint real/imaginary Charbonnier loss, always accumulated in FP32."""

    _validate_complex_tensor_pair(prediction, target)

    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be a finite positive number")

    difference = prediction.float() - target.float()
    squared_magnitude = difference[:, 0].square() + difference[:, 1].square()

    # get the predict and label difference,
    return torch.sqrt(squared_magnitude + epsilon**2).mean()


def normalized_complex_rmse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Root mean square error across both normalized complex channels."""

    _validate_complex_tensor_pair(prediction, target)
    return torch.sqrt((prediction.float() - target.float()).square().mean())


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float) -> None:
    """Update FP32 EMA weights after a successful optimizer step only."""

    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    ema_parameters = dict(ema_model.named_parameters())
    model_parameters = dict(model.named_parameters())
    if ema_parameters.keys() != model_parameters.keys():
        raise ValueError("EMA and training model parameters do not match")
    for name, ema_parameter in ema_parameters.items():
        source = model_parameters[name].detach().float()
        ema_parameter.mul_(decay).add_(source, alpha=1.0 - decay)

    ema_buffers = dict(ema_model.named_buffers())
    model_buffers = dict(model.named_buffers())
    if ema_buffers.keys() != model_buffers.keys():
        raise ValueError("EMA and training model buffers do not match")
    for name, ema_buffer in ema_buffers.items():
        ema_buffer.copy_(model_buffers[name].detach())


def make_ema_model(model: nn.Module) -> nn.Module:
    """Create a frozen FP32 copy used for validation and best checkpoints."""

    ema_model = copy.deepcopy(model).float().eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


def global_gradient_norm(parameters: Any) -> float:
    """Return the L2 norm of all currently populated gradients."""

    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().float().square().sum().item())
    return math.sqrt(squared_norm)


def capture_rng_state() -> dict[str, Any]:
    """Capture all random sources relevant to this single-process trainer."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore a state captured by :func:`capture_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint atomically, preserving the previous file on failure."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def clone_state_dict_as_fp32(model: nn.Module) -> dict[str, torch.Tensor]:
    """A CPU-independent state snapshot useful for small unit tests."""

    return {
        name: value.detach().float().clone()
        for name, value in model.state_dict().items()
    }
