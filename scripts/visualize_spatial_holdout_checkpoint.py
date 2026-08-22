"""Export a representative visual audit of an E004/E005 spatial holdout checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
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

from scripts.overfit_magnitude_patch_set import LoadedMagnitudeSample, load_selected_samples
from scripts.overfit_single_magnitude_patch import evaluate_log_magnitude_prediction
from scripts.overfit_single_patch import predict, write_json
from scripts.visualize_magnitude_patch_set import tensor_image
from swinir import SwinIR
from swinir.sar_dataset import (
    CoordinateRegion,
    DiscoveredPair,
    SplitName,
    build_manifest,
)
from swinir.training import resolve_device, resolve_precision


REPORT_SCHEMA_VERSION = 1
SUPPORTED_EXPERIMENTS = {
    "E004-D002-spatial-holdout-magnitude",
    "E005-D002-spatial-holdout-energy-preserving",
}


@dataclass(frozen=True)
class SelectedValidationSample:
    filename: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Echo, raw/EMA prediction, and Image for representative "
            "spatial-validation patches from an E004/E005 best checkpoint."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument(
        "--selection", choices=("representative", "all"), default="representative"
    )
    parser.add_argument("--weights", choices=("raw", "ema"), default="raw")
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    return parser.parse_args()


def load_spatial_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") not in SUPPORTED_EXPERIMENTS:
        raise RuntimeError("checkpoint is not a supported E004/E005 spatial holdout run")
    for key in ("model", "ema_model"):
        if not isinstance(checkpoint.get(key), dict):
            raise RuntimeError(f"checkpoint is missing {key} weights")
    best_validation = checkpoint.get("best_validation")
    if not isinstance(best_validation, dict):
        raise RuntimeError("checkpoint is missing best_validation metadata")
    summary = best_validation.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("per_sample"), dict):
        raise RuntimeError("checkpoint is missing per-sample validation metrics")
    return checkpoint


def _metric_extreme(
    metrics: dict[str, dict[str, float]], metric: str, *, highest: bool
) -> str:
    return sorted(
        metrics,
        key=lambda name: (float(metrics[name][metric]), name),
        reverse=highest,
    )[0]


def select_representative_samples(
    metrics: dict[str, dict[str, float]],
    coordinates: dict[str, tuple[int, int]],
    sample_count: int,
) -> tuple[SelectedValidationSample, ...]:
    """Select deterministic metric extremes plus spatially distributed samples."""

    if metrics.keys() != coordinates.keys():
        raise RuntimeError("validation metric filenames do not match validation manifest")
    if not 0 < sample_count <= len(metrics):
        raise ValueError(f"sample_count must be in [1, {len(metrics)}]")

    reasons_by_filename: dict[str, list[str]] = {}
    ordered_filenames: list[str] = []

    def add(filename: str, reason: str) -> None:
        if filename not in reasons_by_filename:
            reasons_by_filename[filename] = []
            ordered_filenames.append(filename)
        reasons_by_filename[filename].append(reason)

    rmse_order = sorted(
        metrics,
        key=lambda name: (float(metrics[name]["normalized_log_rmse"]), name),
    )
    add(rmse_order[0], "best_rmse")
    add(rmse_order[len(rmse_order) // 2], "median_rmse")
    add(rmse_order[-1], "worst_rmse")
    for metric, label in (
        ("magnitude_rms_ratio_target", "rms_ratio"),
        ("log_magnitude_correlation", "correlation"),
        ("log_magnitude_ssim", "ssim"),
    ):
        add(_metric_extreme(metrics, metric, highest=False), f"lowest_{label}")
        add(_metric_extreme(metrics, metric, highest=True), f"highest_{label}")

    spatial_order = sorted(coordinates, key=lambda name: (*coordinates[name], name))
    spatial_indices = np.linspace(
        0, len(spatial_order) - 1, num=min(len(spatial_order), sample_count * 3)
    ).round().astype(int)
    for quantile_index, spatial_index in enumerate(spatial_indices):
        add(spatial_order[int(spatial_index)], f"spatial_quantile_{quantile_index:02d}")
        if len(ordered_filenames) >= sample_count:
            break
    for filename in spatial_order:
        if len(ordered_filenames) >= sample_count:
            break
        add(filename, "spatial_fill")

    return tuple(
        SelectedValidationSample(
            filename=filename,
            reasons=tuple(reasons_by_filename[filename]),
        )
        for filename in ordered_filenames[:sample_count]
    )


def select_all_samples(
    coordinates: dict[str, tuple[int, int]],
) -> tuple[SelectedValidationSample, ...]:
    return tuple(
        SelectedValidationSample(filename, ("all_validation",))
        for filename in sorted(coordinates, key=lambda name: (*coordinates[name], name))
    )


def _independent_displays(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    displays = []
    for array in arrays:
        peak = float(np.max(array))
        if not np.isfinite(peak) or peak <= 0:
            displays.append(np.zeros_like(array))
        else:
            displays.append(np.clip(array / peak, 0.0, 1.0))
    return tuple(displays)


def _shared_target_displays(arrays: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    target_peak = float(np.max(arrays[-1]))
    if not np.isfinite(target_peak) or target_peak <= 0:
        raise ValueError("Image target has no finite positive log-magnitude peak")
    return tuple(np.clip(array / target_peak, 0.0, 1.0) for array in arrays)


def export_sample_figure(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    path: Path,
    *,
    sample: LoadedMagnitudeSample,
    prediction_metrics: dict[str, float],
    echo_metrics: dict[str, float],
    reasons: Sequence[str],
    weights: str,
    step: int,
    dpi: int,
) -> None:
    rows = (_independent_displays(arrays), _shared_target_displays(arrays))
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    titles = ("Echo input", f"{weights.upper()} prediction", "Image target")
    for row_index, displays in enumerate(rows):
        for column_index, (display, title) in enumerate(zip(displays, titles, strict=True)):
            axis = axes[row_index, column_index]
            axis.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(title)
        axes[row_index, 0].set_ylabel(
            "Independent peak" if row_index == 0 else "Shared Image peak",
            fontsize=10,
        )
    figure.suptitle(
        f"{sample.filename}  step={step}  selection={','.join(reasons)}\n"
        f"prediction: RMSE={prediction_metrics['normalized_log_rmse']:.4f}, "
        f"corr={prediction_metrics['log_magnitude_correlation']:.4f}, "
        f"RMS ratio={prediction_metrics['magnitude_rms_ratio_target']:.4f}, "
        f"PSNR={prediction_metrics['log_magnitude_psnr_db']:.2f} dB, "
        f"SSIM={prediction_metrics['log_magnitude_ssim']:.4f}\n"
        f"Echo identity: RMSE={echo_metrics['normalized_log_rmse']:.4f}, "
        f"corr={echo_metrics['log_magnitude_correlation']:.4f}, "
        f"RMS ratio={echo_metrics['magnitude_rms_ratio_target']:.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[
        tuple[
            LoadedMagnitudeSample,
            tuple[np.ndarray, np.ndarray, np.ndarray],
            dict[str, float],
            Sequence[str],
        ]
    ],
    path: Path,
    *,
    weights: str,
    step: int,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(rows),
        3,
        figsize=(12, 3.2 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    titles = ("Echo input", f"{weights.upper()} prediction", "Image target")
    for row_index, (sample, arrays, metrics, reasons) in enumerate(rows):
        for column_index, display in enumerate(_shared_target_displays(arrays)):
            axis = axes[row_index, column_index]
            axis.imshow(display, cmap="gray", vmin=0.0, vmax=1.0)
            axis.axis("off")
            if row_index == 0:
                axis.set_title(titles[column_index])
        axes[row_index, 0].set_ylabel(
            f"row={sample.row} col={sample.col}\n"
            f"RMSE={metrics['normalized_log_rmse']:.3f} "
            f"ratio={metrics['magnitude_rms_ratio_target']:.3f}\n"
            f"{','.join(reasons)}",
            fontsize=7,
        )
    figure.suptitle(
        f"Spatial holdout visual audit, step={step}, weights={weights}\n"
        "All three columns in each row share the Image target peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    if args.sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if args.contact_sheet_page_size <= 0:
        raise ValueError("contact_sheet_page_size must be positive")
    for role, directory in (("Echo", args.echo_dir), ("Image", args.image_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} directory does not exist: {directory}")
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new directory"
        )

    checkpoint = load_spatial_checkpoint(args.checkpoint)
    config = checkpoint["resolved_config"]
    data = config["data"]
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**data["validation_region"]),
        CoordinateRegion(**data["guard_region"]),
        expected_counts=data["expected_split_counts"],
    )
    if manifest.fingerprint != checkpoint.get("manifest_fingerprint"):
        raise RuntimeError("dataset manifest fingerprint does not match checkpoint")
    validation_records = manifest.records_for(SplitName.VALIDATION)
    coordinates = {
        record.echo_path.name: (record.row, record.col) for record in validation_records
    }
    stored_metrics = checkpoint["best_validation"]["summary"]["per_sample"]
    if args.selection == "all":
        selected = select_all_samples(coordinates)
    else:
        selected = select_representative_samples(
            stored_metrics, coordinates, args.sample_count
        )
    records_by_filename = {
        record.echo_path.name: record for record in validation_records
    }
    pairs = tuple(
        DiscoveredPair(
            row=records_by_filename[item.filename].row,
            col=records_by_filename[item.filename].col,
            echo_path=records_by_filename[item.filename].echo_path,
            image_path=records_by_filename[item.filename].image_path,
        )
        for item in selected
    )
    samples = load_selected_samples(
        pairs,
        expected_shape=tuple(int(value) for value in data["expected_shape"]),
        rms_epsilon=float(data["rms_epsilon"]),
    )

    device = resolve_device(args.device)
    precision = resolve_precision(device)
    model = SwinIR(**config["model"])
    state_key = "model" if args.weights == "raw" else "ema_model"
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device).eval()
    step = int(checkpoint["best_validation"]["step"])
    loss_epsilon = float(config["optimization"]["charbonnier_epsilon"])

    args.output_dir.mkdir(parents=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir()
    contact_rows = []
    manifest_entries = []
    for index, (sample, selection) in enumerate(zip(samples, selected, strict=True)):
        prediction = predict(model, sample.inputs, device=device, precision=precision)
        prediction_metrics = evaluate_log_magnitude_prediction(
            prediction,
            sample.targets,
            charbonnier_epsilon=loss_epsilon,
        )
        echo_metrics = evaluate_log_magnitude_prediction(
            sample.inputs,
            sample.targets,
            charbonnier_epsilon=loss_epsilon,
        )
        arrays = (
            tensor_image(sample.inputs),
            tensor_image(prediction),
            tensor_image(sample.targets),
        )
        figure_name = f"{index:02d}_{Path(sample.filename).stem}.png"
        export_sample_figure(
            arrays,
            samples_dir / figure_name,
            sample=sample,
            prediction_metrics=prediction_metrics,
            echo_metrics=echo_metrics,
            reasons=selection.reasons,
            weights=args.weights,
            step=step,
            dpi=args.dpi,
        )
        contact_rows.append((sample, arrays, prediction_metrics, selection.reasons))
        manifest_entries.append(
            {
                "selection_index": index,
                "filename": sample.filename,
                "row": sample.row,
                "col": sample.col,
                "selection_reasons": list(selection.reasons),
                "figure": str(Path("samples") / figure_name),
                "prediction_metrics": prediction_metrics,
                "echo_identity_metrics": echo_metrics,
                "stored_best_raw_metrics": stored_metrics[sample.filename],
            }
        )
        print(
            f"[{index:02d}] {sample.filename} "
            f"rmse={prediction_metrics['normalized_log_rmse']:.4f} "
            f"corr={prediction_metrics['log_magnitude_correlation']:.4f} "
            f"rms_ratio={prediction_metrics['magnitude_rms_ratio_target']:.4f} "
            f"psnr={prediction_metrics['log_magnitude_psnr_db']:.2f} "
            f"ssim={prediction_metrics['log_magnitude_ssim']:.4f}",
            flush=True,
        )

    contact_sheets = []
    page_size = int(args.contact_sheet_page_size)
    for page_index, start in enumerate(range(0, len(contact_rows), page_size), start=1):
        page_rows = contact_rows[start : start + page_size]
        if len(contact_rows) <= page_size:
            name = f"audit_{len(samples):03d}_validation_samples.png"
        else:
            name = f"audit_page_{page_index:03d}.png"
        contact_sheet = args.output_dir / name
        export_contact_sheet(
            page_rows,
            contact_sheet,
            weights=args.weights,
            step=step,
            dpi=args.dpi,
        )
        contact_sheets.append(contact_sheet)
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "weights": args.weights,
        "device": str(device),
        "selection": args.selection,
        "sample_count": len(samples),
        "validation_sample_count": len(validation_records),
        "manifest_fingerprint": manifest.fingerprint,
        "display_normalization": {
            "individual_top_row": "each panel uses its own log-magnitude peak",
            "individual_bottom_row": "Echo/prediction/Image share Image target peak",
            "contact_sheet": "Echo/prediction/Image share Image target peak",
        },
        "contact_sheets": [path.name for path in contact_sheets],
        "samples": manifest_entries,
    }
    write_json(args.output_dir / "audit_manifest.json", result)
    for contact_sheet in contact_sheets:
        print(f"contact_sheet={contact_sheet.resolve()}", flush=True)
    print(f"manifest={(args.output_dir / 'audit_manifest.json').resolve()}", flush=True)
    return result


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
