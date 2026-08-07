"""Evaluate a trained SAR SwinIR checkpoint on an independent paired MAT set.

The script evaluates the EMA weights saved in a checkpoint.  It writes one JSON
report by default and only writes denormalized complex prediction MAT files
when ``--save-predictions`` is explicitly requested.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from swinir import SwinIR
from swinir.sar_dataset import (
    PairRecord,
    SARPatchDataset,
    SplitName,
    discover_pairs,
)
from swinir.training import (
    PrecisionPolicy,
    complex_charbonnier_loss,
    normalized_complex_rmse,
    resolve_device,
    resolve_precision,
)


SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = frozenset({1, 2})
TEST_REPORT_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT = Path("runs/sar_baseline_v1/checkpoints/best.pt")


@dataclass
class MetricTotals:
    """Example-weighted totals for model and identity-Echo metrics."""

    count: int = 0
    loss: float = 0.0
    rmse: float = 0.0
    baseline_loss: float = 0.0
    baseline_rmse: float = 0.0

    def update(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        echo: torch.Tensor,
        loss_epsilon: float,
    ) -> None:
        batch_size = int(prediction.shape[0])
        if batch_size <= 0:
            raise ValueError("metrics cannot be updated from an empty batch")
        self.count += batch_size
        self.loss += float(
            complex_charbonnier_loss(prediction, target, loss_epsilon).item()
        ) * batch_size
        self.rmse += float(normalized_complex_rmse(prediction, target).item()) * batch_size
        self.baseline_loss += float(
            complex_charbonnier_loss(echo, target, loss_epsilon).item()
        ) * batch_size
        self.baseline_rmse += float(normalized_complex_rmse(echo, target).item()) * batch_size

    def as_dict(self) -> dict[str, float | int]:
        if self.count <= 0:
            raise RuntimeError("metric totals contain no examples")
        charbonnier = self.loss / self.count
        complex_rmse = self.rmse / self.count
        baseline_charbonnier = self.baseline_loss / self.count
        baseline_rmse = self.baseline_rmse / self.count
        return {
            "patch_count": self.count,
            "charbonnier": charbonnier,
            "complex_rmse": complex_rmse,
            "echo_baseline_charbonnier": baseline_charbonnier,
            "echo_baseline_complex_rmse": baseline_rmse,
            "charbonnier_absolute_improvement": baseline_charbonnier - charbonnier,
            "charbonnier_relative_improvement_percent": (
                (baseline_charbonnier - charbonnier) / baseline_charbonnier * 100.0
            ),
            "complex_rmse_absolute_improvement": baseline_rmse - complex_rmse,
            "complex_rmse_relative_improvement_percent": (
                (baseline_rmse - complex_rmse) / baseline_rmse * 100.0
            ),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test SAR SwinIR on paired MAT patches")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--independent-stride",
        type=int,
        default=600,
        help="minimum coordinate spacing in pixels for the independent-patch report",
    )
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def _checkpoint_config(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if checkpoint.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise RuntimeError(
            "unsupported checkpoint schema version: "
            f"{checkpoint.get('schema_version')!r}"
        )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("checkpoint is missing its training configuration")
    for section in ("model", "data", "optimization"):
        if not isinstance(config.get(section), dict):
            raise RuntimeError(f"checkpoint config section {section!r} is missing")
    if not isinstance(checkpoint.get("ema_model"), dict):
        raise RuntimeError("checkpoint is missing EMA model weights")
    return config


def load_ema_model(
    checkpoint_path: Path, device: torch.device
) -> tuple[dict[str, Any], torch.nn.Module]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    config = _checkpoint_config(checkpoint)
    model = SwinIR(**config["model"]).to(device)
    model.load_state_dict(checkpoint["ema_model"], strict=True)
    model.eval()
    return checkpoint, model


def expected_shape_from_config(config: dict[str, Any]) -> tuple[int, int]:
    raw_shape = config["data"].get("expected_shape")
    if (
        not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) != 2
        or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in raw_shape)
    ):
        raise RuntimeError("checkpoint data.expected_shape must be two positive integers")
    return int(raw_shape[0]), int(raw_shape[1])


def test_records(echo_dir: Path, image_dir: Path) -> tuple[PairRecord, ...]:
    """Discover strict pairs without applying the training split regions."""

    return tuple(
        PairRecord(
            key=pair.key,
            row=pair.row,
            col=pair.col,
            split=SplitName.TRAIN,
            echo_path=pair.echo_path,
            image_path=pair.image_path,
        )
        for pair in discover_pairs(echo_dir, image_dir)
    )


def is_independent_coordinate(
    row: int,
    col: int,
    *,
    origin_row: int,
    origin_col: int,
    stride: int,
) -> bool:
    return (row - origin_row) % stride == 0 and (col - origin_col) % stride == 0


def _batch_coordinate_values(value: Any) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.tolist()]
    return [int(item) for item in value]


def save_denormalized_predictions(
    prediction: torch.Tensor,
    scales: torch.Tensor,
    filenames: Sequence[str],
    output_dir: Path,
) -> None:
    """Write one complex ``patch`` MAT file per prediction in original units."""

    values = prediction.detach().float().cpu().numpy()
    scale_values = scales.detach().float().cpu().numpy()
    if values.shape[0] != len(filenames) or scale_values.shape[0] != len(filenames):
        raise ValueError("prediction batch, scales, and filenames must have equal lengths")
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, filename in enumerate(filenames):
        complex_patch = (values[index, 0] + 1j * values[index, 1]) * scale_values[index]
        savemat(output_dir / filename, {"patch": complex_patch})


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Any]],
    *,
    device: torch.device,
    precision: PrecisionPolicy,
    loss_epsilon: float,
    independent_origin: tuple[int, int],
    independent_stride: int,
    prediction_dir: Path | None,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Evaluate all patches and the deterministic non-overlapping subset."""

    all_totals = MetricTotals()
    independent_totals = MetricTotals()
    origin_row, origin_col = independent_origin
    model.eval()

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=device.type == "cuda")
        targets = batch["target"].to(device, non_blocking=device.type == "cuda")
        with precision.autocast():
            predictions = model(inputs)

        all_totals.update(predictions, targets, inputs, loss_epsilon)
        rows = _batch_coordinate_values(batch["row"])
        cols = _batch_coordinate_values(batch["col"])
        selected = [
            index
            for index, (row, col) in enumerate(zip(rows, cols, strict=True))
            if is_independent_coordinate(
                row,
                col,
                origin_row=origin_row,
                origin_col=origin_col,
                stride=independent_stride,
            )
        ]
        if selected:
            independent_totals.update(
                predictions[selected], targets[selected], inputs[selected], loss_epsilon
            )
        if prediction_dir is not None:
            filenames = [Path(path).name for path in batch["echo_path"]]
            save_denormalized_predictions(
                predictions, batch["scale"], filenames, prediction_dir
            )

    return all_totals.as_dict(), independent_totals.as_dict()


def make_loader(dataset: SARPatchDataset, workers: int, pin_memory: bool) -> DataLoader[dict[str, Any]]:
    if workers < 0:
        raise ValueError("num_workers must be non-negative")
    options: dict[str, Any] = {
        "batch_size": 1,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": workers > 0,
    }
    return DataLoader(dataset, **options)


def default_output_dir(checkpoint: Path, echo_dir: Path) -> Path:
    return checkpoint.parent.parent / f"test_{echo_dir.name}"


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint, model = load_ema_model(args.checkpoint, device)
    config = _checkpoint_config(checkpoint)
    expected_shape = expected_shape_from_config(config)
    rms_epsilon = float(config["data"].get("rms_epsilon", 1e-12))
    loss_epsilon = float(config["optimization"].get("charbonnier_epsilon", 1e-3))
    if not math.isfinite(rms_epsilon) or rms_epsilon <= 0:
        raise RuntimeError("checkpoint data.rms_epsilon must be finite and positive")
    if not math.isfinite(loss_epsilon) or loss_epsilon <= 0:
        raise RuntimeError("checkpoint optimization.charbonnier_epsilon must be finite and positive")
    if args.independent_stride < max(expected_shape):
        raise ValueError(
            "independent_stride must be at least the patch size to avoid raw-window overlap"
        )

    records = test_records(args.echo_dir, args.image_dir)
    dataset = SARPatchDataset(records, expected_shape=expected_shape, epsilon=rms_epsilon)
    origin = (min(record.row for record in records), min(record.col for record in records))
    output_dir = args.output_dir or default_output_dir(args.checkpoint, args.echo_dir)
    prediction_dir = output_dir / "predictions" if args.save_predictions else None
    precision = resolve_precision(device)
    loader = make_loader(dataset, args.num_workers, pin_memory=device.type == "cuda")
    all_metrics, independent_metrics = evaluate(
        model,
        loader,
        device=device,
        precision=precision,
        loss_epsilon=loss_epsilon,
        independent_origin=origin,
        independent_stride=args.independent_stride,
        prediction_dir=prediction_dir,
    )
    report = {
        "schema_version": TEST_REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_global_step": checkpoint.get("global_step"),
        "weights": "ema_model",
        "device": str(device),
        "precision": precision.as_dict(),
        "dataset": {
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "patch_count": len(records),
            "expected_shape": list(expected_shape),
            "normalization": "per_patch_echo_rms",
            "rms_epsilon": rms_epsilon,
        },
        "independent_subset": {
            "origin_row": origin[0],
            "origin_col": origin[1],
            "stride_pixels": args.independent_stride,
        },
        "metrics": {
            "all_patches_patch_weighted": all_metrics,
            "independent_patches": independent_metrics,
        },
        "prediction_export": {
            "enabled": args.save_predictions,
            "directory": str(prediction_dir.resolve()) if prediction_dir else None,
        },
    }
    report_path = output_dir / "test_report.json"
    write_report(report_path, report)
    print(json.dumps(report["metrics"], indent=2, ensure_ascii=False))
    print(f"JSON report: {report_path.resolve()}")


if __name__ == "__main__":
    main()
