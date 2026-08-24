"""Export E010 unseen-patch Echo/prediction/oracle/Image visual audits."""

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

from scripts.overfit_phase_correction_patch_set import load_phase_samples
from scripts.overfit_single_patch import tensor_to_complex, write_json
from scripts.overfit_single_phase_correction import (
    evaluate_correction,
    predict_correction,
)
from scripts.train_phase_spatial_holdout import add_generalization_metrics
from swinir import SwinIR
from swinir.sar_dataset import (
    CoordinateRegion,
    DiscoveredPair,
    SplitName,
    build_manifest,
)
from swinir.sar_metrics import log_magnitude_image
from swinir.training import resolve_device, resolve_precision


REPORT_SCHEMA_VERSION = 1
SUPPORTED_EXPERIMENT = "E010-D001-phase-spatial-holdout"


@dataclass(frozen=True)
class SelectedSample:
    filename: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize unseen validation patches from an E010 checkpoint."
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") != SUPPORTED_EXPERIMENT:
        raise RuntimeError("checkpoint is not an E010 spatial phase run")
    for key in ("model", "ema_model"):
        if not isinstance(checkpoint.get(key), dict):
            raise RuntimeError(f"checkpoint is missing {key} weights")
    best = checkpoint.get("best_validation")
    if not isinstance(best, dict):
        raise RuntimeError("checkpoint is missing best_validation")
    summary = best.get("summary")
    if not isinstance(summary, dict) or not isinstance(summary.get("per_sample"), dict):
        raise RuntimeError("checkpoint is missing per-sample best validation metrics")
    return checkpoint


def select_representative(
    metrics: dict[str, dict[str, float]],
    coordinates: dict[str, tuple[int, int]],
    count: int,
) -> tuple[SelectedSample, ...]:
    if metrics.keys() != coordinates.keys():
        raise RuntimeError("validation metrics and manifest samples differ")
    if not 0 < count <= len(metrics):
        raise ValueError(f"sample_count must be in [1, {len(metrics)}]")
    reasons: dict[str, list[str]] = {}
    order: list[str] = []

    def add(filename: str, reason: str) -> None:
        if filename not in reasons:
            reasons[filename] = []
            order.append(filename)
        reasons[filename].append(reason)

    for metric, low_label, high_label in (
        ("weighted_phase_alignment", "lowest_phase", "highest_phase"),
        (
            "rmse_oracle_gap_fraction_closed",
            "lowest_rmse_gap_closed",
            "highest_rmse_gap_closed",
        ),
        (
            "coherence_fraction_of_oracle",
            "lowest_coherence_fraction",
            "highest_coherence_fraction",
        ),
        ("ssim_gain_fraction_of_oracle", "lowest_ssim_gain", "highest_ssim_gain"),
        ("edge_gain_fraction_of_oracle", "lowest_edge_gain", "highest_edge_gain"),
    ):
        ranked = sorted(metrics, key=lambda name: (float(metrics[name][metric]), name))
        add(ranked[0], low_label)
        add(ranked[-1], high_label)
    gap_ranked = sorted(
        metrics,
        key=lambda name: (float(metrics[name]["rmse_oracle_gap_fraction_closed"]), name),
    )
    add(gap_ranked[len(gap_ranked) // 2], "median_rmse_gap_closed")
    spatial = sorted(coordinates, key=lambda name: (*coordinates[name], name))
    indices = np.linspace(0, len(spatial) - 1, num=min(len(spatial), count * 3))
    for index, spatial_index in enumerate(indices.round().astype(int)):
        add(spatial[int(spatial_index)], f"spatial_quantile_{index:02d}")
        if len(order) >= count:
            break
    for filename in spatial:
        if len(order) >= count:
            break
        add(filename, "spatial_fill")
    return tuple(
        SelectedSample(filename, tuple(reasons[filename]))
        for filename in order[:count]
    )


def export_sample_figure(
    arrays: tuple[np.ndarray, ...],
    path: Path,
    *,
    filename: str,
    metrics: dict[str, float],
    reasons: Sequence[str],
    weights: str,
    step: int,
    floor_db: float,
    dpi: int,
) -> None:
    titles = ("Echo", f"{weights.upper()} prediction", "Oracle phase", "Image")
    target_peak = max(float(np.abs(arrays[-1]).max()), np.finfo(np.float64).tiny)
    figure, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
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
    figure.suptitle(
        f"{filename} step={step} selection={','.join(reasons)}\n"
        f"phase={metrics['weighted_phase_alignment']:.4f} "
        f"gap_closed={metrics['rmse_oracle_gap_fraction_closed']:.4f} "
        f"coh_frac={metrics['coherence_fraction_of_oracle']:.4f} "
        f"SSIM_gain={metrics['ssim_gain_fraction_of_oracle']:.4f} "
        f"edge_gain={metrics['edge_gain_fraction_of_oracle']:.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[tuple[str, tuple[np.ndarray, ...], dict[str, float], Sequence[str]]],
    path: Path,
    *,
    weights: str,
    step: int,
    floor_db: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 4, figsize=(16, 3.2 * len(rows)), constrained_layout=True, squeeze=False
    )
    titles = ("Echo", f"{weights.upper()} prediction", "Oracle phase", "Image")
    for row_index, (filename, arrays, metrics, reasons) in enumerate(rows):
        target_peak = max(float(np.abs(arrays[-1]).max()), np.finfo(np.float64).tiny)
        for column, array in enumerate(arrays):
            axes[row_index, column].imshow(
                log_magnitude_image(array, reference_peak=target_peak, floor_db=floor_db),
                cmap="gray",
                vmin=0,
                vmax=1,
            )
            axes[row_index, column].axis("off")
            if row_index == 0:
                axes[row_index, column].set_title(titles[column])
        axes[row_index, 0].set_ylabel(
            f"{Path(filename).stem}\nphase={metrics['weighted_phase_alignment']:.3f} "
            f"gap={metrics['rmse_oracle_gap_fraction_closed']:.3f}\n"
            f"{','.join(reasons)}",
            fontsize=7,
        )
    figure.suptitle(
        f"E010 unseen spatial holdout audit, step={step}, weights={weights}\n"
        "Every row shares its Image target peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dpi <= 0 or args.sample_count <= 0 or args.contact_sheet_page_size <= 0:
        raise ValueError("dpi and sample counts must be positive")
    for role, directory in (("Echo", args.echo_dir), ("Image", args.image_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} directory does not exist: {directory}")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["resolved_config"]
    data = config["data"]
    optimization = config["optimization"]
    evaluation = config["evaluation"]
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
    stored = checkpoint["best_validation"]["summary"]["per_sample"]
    if args.selection == "all":
        selected = tuple(
            SelectedSample(name, ("all_validation",))
            for name in sorted(coordinates, key=lambda name: (*coordinates[name], name))
        )
    else:
        selected = select_representative(stored, coordinates, args.sample_count)
    records = {record.echo_path.name: record for record in validation_records}
    pairs = tuple(
        DiscoveredPair(
            row=records[item.filename].row,
            col=records[item.filename].col,
            echo_path=records[item.filename].echo_path,
            image_path=records[item.filename].image_path,
        )
        for item in selected
    )
    samples = load_phase_samples(
        pairs,
        expected_shape=tuple(int(value) for value in data["expected_shape"]),
        data_config=data,
        optimization=optimization,
        evaluation=evaluation,
    )
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    model = SwinIR(**config["model"])
    model.load_state_dict(checkpoint["model" if args.weights == "raw" else "ema_model"])
    model.to(device).eval()
    step = int(checkpoint["best_validation"]["step"])
    args.output_dir.mkdir(parents=True)
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir()
    contact_rows = []
    entries = []
    for index, (sample, selection) in enumerate(zip(samples, selected, strict=True)):
        correction = predict_correction(
            model,
            sample.input_spectrum,
            device=device,
            precision=precision,
            phasor_epsilon=float(optimization["phasor_epsilon"]),
        )
        prediction, base_metrics = evaluate_correction(
            correction,
            sample.input_spectrum,
            sample.target_phasor,
            sample.phase_weights,
            sample.target_image,
            fft_norm=str(data["fft_norm"]),
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                evaluation["high_frequency_radius_fraction"]
            ),
        )
        metrics = add_generalization_metrics(
            base_metrics, sample.echo_metrics, sample.oracle_metrics
        )
        arrays = tuple(
            tensor_to_complex(tensor)
            for tensor in (
                sample.echo_image,
                prediction,
                sample.oracle_prediction,
                sample.target_image,
            )
        )
        name = f"{index:03d}_{Path(sample.filename).stem}.png"
        export_sample_figure(
            arrays,
            samples_dir / name,
            filename=sample.filename,
            metrics=metrics,
            reasons=selection.reasons,
            weights=args.weights,
            step=step,
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            dpi=args.dpi,
        )
        contact_rows.append((sample.filename, arrays, metrics, selection.reasons))
        entries.append(
            {
                "selection_index": index,
                "filename": sample.filename,
                "row": sample.row,
                "col": sample.col,
                "selection_reasons": list(selection.reasons),
                "figure": str(Path("samples") / name),
                "metrics": metrics,
                "stored_best_raw_metrics": stored[sample.filename],
            }
        )
        print(
            f"[{index:03d}] {sample.filename} phase={metrics['weighted_phase_alignment']:.4f} "
            f"gap_closed={metrics['rmse_oracle_gap_fraction_closed']:.4f} "
            f"coh_frac={metrics['coherence_fraction_of_oracle']:.4f}",
            flush=True,
        )
    contact_sheets = []
    page_size = int(args.contact_sheet_page_size)
    for page, start in enumerate(range(0, len(contact_rows), page_size), start=1):
        name = (
            f"audit_{len(contact_rows):03d}_validation_samples.png"
            if len(contact_rows) <= page_size
            else f"audit_page_{page:03d}.png"
        )
        export_contact_sheet(
            contact_rows[start : start + page_size],
            args.output_dir / name,
            weights=args.weights,
            step=step,
            floor_db=float(evaluation["log_magnitude_floor_db"]),
            dpi=args.dpi,
        )
        contact_sheets.append(name)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "weights": args.weights,
        "selection": args.selection,
        "sample_count": len(samples),
        "validation_sample_count": len(validation_records),
        "manifest_fingerprint": manifest.fingerprint,
        "contact_sheets": contact_sheets,
        "samples": entries,
    }
    write_json(args.output_dir / "audit_manifest.json", report)
    for name in contact_sheets:
        print(f"contact_sheet={(args.output_dir / name).resolve()}", flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
