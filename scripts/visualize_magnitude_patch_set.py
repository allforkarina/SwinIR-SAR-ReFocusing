"""Visualize every sample from a joint log-magnitude overfit checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.overfit_magnitude_patch_set import (
    LoadedMagnitudeSample,
    load_selected_samples,
)
from scripts.overfit_single_magnitude_patch import evaluate_log_magnitude_prediction
from scripts.overfit_single_patch import predict, write_json
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair, discover_pairs
from swinir.training import resolve_device, resolve_precision


MANIFEST_SCHEMA_VERSION = 1
EXPECTED_EXPERIMENT = "D002-B2-A-joint-magnitude-patch-set"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Echo input, checkpoint prediction, and Image target for every "
            "sample recorded by a joint magnitude-overfit checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--dpi", type=int, default=160)
    return parser.parse_args()


def load_magnitude_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") != EXPECTED_EXPERIMENT:
        raise RuntimeError(
            "checkpoint is not a D002 joint log-magnitude patch-set checkpoint"
        )
    manifest = config.get("selection_manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("samples"), list):
        raise RuntimeError("checkpoint is missing its selection manifest")
    for key in ("model", "ema_model"):
        if not isinstance(checkpoint.get(key), dict):
            raise RuntimeError(f"checkpoint is missing {key} weights")
    return checkpoint


def ordered_manifest_pairs(
    pairs: Sequence[DiscoveredPair], manifest: dict[str, Any]
) -> tuple[DiscoveredPair, ...]:
    records = manifest["samples"]
    expected_count = int(manifest.get("sample_count", len(records)))
    if expected_count != len(records) or expected_count <= 0:
        raise RuntimeError("selection manifest has an inconsistent sample count")
    ordered_records = sorted(records, key=lambda record: int(record["selection_index"]))
    indices = [int(record["selection_index"]) for record in ordered_records]
    if indices != list(range(expected_count)):
        raise RuntimeError("selection manifest indices must be contiguous from zero")

    by_filename = {pair.echo_path.name: pair for pair in pairs}
    if len(by_filename) != len(pairs):
        raise RuntimeError("discovered pairs contain duplicate filenames")
    selected = []
    for record in ordered_records:
        filename = str(record["filename"])
        if filename not in by_filename:
            raise FileNotFoundError(f"checkpoint sample is missing from dataset: {filename}")
        pair = by_filename[filename]
        if (pair.row, pair.col) != (int(record["row"]), int(record["col"])):
            raise RuntimeError(f"coordinate mismatch for checkpoint sample: {filename}")
        selected.append(pair)
    return tuple(selected)


def validate_sample_fingerprints(
    samples: Sequence[LoadedMagnitudeSample], manifest: dict[str, Any]
) -> None:
    records = {
        str(record["filename"]): record for record in manifest["samples"]
    }
    for sample in samples:
        record = records[sample.filename]
        for key in ("echo_sha256", "image_sha256", "echo_size_bytes", "image_size_bytes"):
            if sample.fingerprint[key] != record[key]:
                raise RuntimeError(
                    f"dataset fingerprint mismatch for {sample.filename}: {key}"
                )


def tensor_image(value: torch.Tensor) -> np.ndarray:
    array = value.detach().float().cpu().numpy()
    if array.shape[:2] != (1, 1) or array.ndim != 4:
        raise ValueError(f"expected [1, 1, H, W], got {array.shape}")
    return np.asarray(array[0, 0], dtype=np.float64)


def shared_target_peak_displays(
    sample: LoadedMagnitudeSample, prediction: torch.Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = (
        tensor_image(sample.inputs),
        tensor_image(prediction),
        tensor_image(sample.targets),
    )
    target_peak = float(arrays[-1].max())
    if not np.isfinite(target_peak) or target_peak <= 0:
        raise ValueError(f"target has no finite positive peak: {sample.filename}")
    return tuple(np.clip(array / target_peak, 0.0, 1.0) for array in arrays)


def export_sample_figure(
    displays: tuple[np.ndarray, np.ndarray, np.ndarray],
    path: Path,
    *,
    sample: LoadedMagnitudeSample,
    metrics: dict[str, float],
    weights: str,
    step: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, display, title in zip(
        axes,
        displays,
        ("Echo input", f"{weights.upper()} prediction", "Image target"),
        strict=True,
    ):
        axis.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"[{sample.filename}] step={step}  RMSE={metrics['normalized_log_rmse']:.4f}  "
        f"corr={metrics['log_magnitude_correlation']:.4f}  "
        f"PSNR={metrics['log_magnitude_psnr_db']:.2f} dB  "
        f"SSIM={metrics['log_magnitude_ssim']:.4f}"
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[
        tuple[
            LoadedMagnitudeSample,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            dict[str, float],
        ]
    ],
    path: Path,
    *,
    weights: str,
    step: int,
    checkpoint_name: str,
    dpi: int,
) -> None:
    if not rows:
        raise ValueError("cannot export an empty contact sheet")
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, 4 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = ("Echo input", f"{weights.upper()} prediction", "Image target")
    for row_index, (sample, displays, metrics) in enumerate(rows):
        for column, display in enumerate(displays):
            axis = axes[row_index, column]
            axis.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(column_titles[column], fontsize=12)
        axes[row_index, 0].set_ylabel(
            f"[{row_index:02d}] row={sample.row} col={sample.col}\n"
            f"RMSE={metrics['normalized_log_rmse']:.4f}  "
            f"SSIM={metrics['log_magnitude_ssim']:.4f}",
            fontsize=8,
        )
    figure.suptitle(
        f"D002 magnitude visual audit: {checkpoint_name}, step={step}, weights={weights}\n"
        "Each row uses one target-peak-normalized shared log-magnitude scale",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    for role, directory in (("Echo", args.echo_dir), ("Image", args.image_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} directory does not exist: {directory}")
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new directory"
        )

    checkpoint = load_magnitude_checkpoint(args.checkpoint)
    config = checkpoint["resolved_config"]
    manifest = config["selection_manifest"]
    expected_shape = tuple(int(value) for value in config["data"]["expected_shape"])
    rms_epsilon = float(config["data"]["rms_epsilon"])
    pairs = ordered_manifest_pairs(discover_pairs(args.echo_dir, args.image_dir), manifest)
    samples = load_selected_samples(
        pairs,
        expected_shape=expected_shape,
        rms_epsilon=rms_epsilon,
    )
    validate_sample_fingerprints(samples, manifest)

    device = resolve_device(args.device)
    precision = resolve_precision(device)
    model = SwinIR(**config["model"])
    state_key = "model" if args.weights == "raw" else "ema_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval()
    step = int(checkpoint["step"])
    loss_epsilon = float(config["optimization"]["charbonnier_epsilon"])

    args.output_dir.mkdir(parents=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir()
    contact_rows = []
    manifest_entries = []
    for index, sample in enumerate(samples):
        prediction = predict(
            model,
            sample.inputs,
            device=device,
            precision=precision,
        )
        metrics = evaluate_log_magnitude_prediction(
            prediction,
            sample.targets,
            charbonnier_epsilon=loss_epsilon,
        )
        displays = shared_target_peak_displays(sample, prediction)
        figure_name = f"{index:02d}_{Path(sample.filename).stem}.png"
        export_sample_figure(
            displays,
            samples_dir / figure_name,
            sample=sample,
            metrics=metrics,
            weights=args.weights,
            step=step,
            dpi=args.dpi,
        )
        contact_rows.append((sample, displays, metrics))
        manifest_entries.append(
            {
                "selection_index": index,
                "filename": sample.filename,
                "row": sample.row,
                "col": sample.col,
                "figure": str(Path("samples") / figure_name),
                "metrics": metrics,
            }
        )
        print(
            f"[{index:02d}] {sample.filename} "
            f"rmse={metrics['normalized_log_rmse']:.4f} "
            f"corr={metrics['log_magnitude_correlation']:.4f} "
            f"psnr={metrics['log_magnitude_psnr_db']:.2f} "
            f"ssim={metrics['log_magnitude_ssim']:.4f}",
            flush=True,
        )

    contact_sheet = args.output_dir / f"all_{len(samples):02d}_echo_prediction_image.png"
    export_contact_sheet(
        contact_rows,
        contact_sheet,
        weights=args.weights,
        step=step,
        checkpoint_name=args.checkpoint.name,
        dpi=args.dpi,
    )
    result = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "weights": args.weights,
        "device": str(device),
        "sample_count": len(samples),
        "representation": config["data"],
        "display_normalization": (
            "per sample, Echo/prediction/Image share Image target peak in log-magnitude domain"
        ),
        "contact_sheet": contact_sheet.name,
        "samples": manifest_entries,
    }
    write_json(args.output_dir / "manifest.json", result)
    print(f"contact_sheet={contact_sheet.resolve()}", flush=True)
    print(f"manifest={(args.output_dir / 'manifest.json').resolve()}", flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
