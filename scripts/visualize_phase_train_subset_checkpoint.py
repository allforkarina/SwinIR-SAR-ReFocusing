"""Export a complete E011-B phase-overfit audit from a saved checkpoint."""

from __future__ import annotations

import argparse
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

from scripts.overfit_phase_correction_patch_set import (
    evaluate_patch_set,
    load_phase_samples,
)
from scripts.overfit_phase_train_subset import EXPERIMENT
from scripts.overfit_single_patch import tensor_to_complex, write_json
from scripts.overfit_single_phase_correction import PhaseSuccessCriteria
from swinir import SwinIR
from swinir.sar_dataset import DiscoveredPair
from swinir.sar_metrics import log_magnitude_image
from swinir.training import resolve_device, resolve_precision


REPORT_SCHEMA_VERSION = 1
FINGERPRINT_KEYS = (
    "echo_sha256",
    "image_sha256",
    "echo_size_bytes",
    "image_size_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize all fixed training samples from an E011-B checkpoint using "
            "RAW and EMA weights."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contact-sheet-page-size", type=int, default=8)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("schema_version") != 1:
        raise RuntimeError("unsupported phase-overfit checkpoint schema")
    config = checkpoint.get("resolved_config")
    if not isinstance(config, dict) or config.get("experiment") != EXPERIMENT:
        raise RuntimeError("checkpoint is not an E011-B phase train-subset run")
    for key in ("model", "ema_model"):
        if not isinstance(checkpoint.get(key), dict):
            raise RuntimeError(f"checkpoint is missing {key} weights")
    step = checkpoint.get("step")
    if not isinstance(step, int) or step < 0:
        raise RuntimeError("checkpoint has an invalid step")
    metrics = checkpoint.get("last_metrics")
    if not isinstance(metrics, dict) or metrics.get("step") != step:
        raise RuntimeError("checkpoint weights and last_metrics are from different steps")
    manifest = config.get("selection_manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("checkpoint is missing selection_manifest")
    records = manifest.get("samples")
    sample_count = manifest.get("sample_count")
    if (
        not isinstance(records, list)
        or not isinstance(sample_count, int)
        or sample_count <= 0
        or len(records) != sample_count
    ):
        raise RuntimeError("checkpoint has an invalid selected sample manifest")
    for weights in ("raw", "ema"):
        summary = metrics.get(weights)
        if not isinstance(summary, dict) or not isinstance(
            summary.get("per_sample"), dict
        ):
            raise RuntimeError(f"checkpoint is missing {weights} per-sample metrics")
    return checkpoint


def manifest_pairs(
    manifest: dict[str, Any], echo_dir: Path, image_dir: Path
) -> tuple[DiscoveredPair, ...]:
    records = manifest["samples"]
    indices = [record.get("selection_index") for record in records]
    if indices != list(range(len(records))):
        raise RuntimeError("selection manifest indices are not contiguous and ordered")
    names = [record.get("filename") for record in records]
    if not all(isinstance(name, str) and name for name in names):
        raise RuntimeError("selection manifest contains an invalid filename")
    if len(set(names)) != len(names):
        raise RuntimeError("selection manifest contains duplicate filenames")
    return tuple(
        DiscoveredPair(
            row=int(record["row"]),
            col=int(record["col"]),
            echo_path=echo_dir / str(record["filename"]),
            image_path=image_dir / str(record["filename"]),
        )
        for record in records
    )


def validate_fingerprints(
    samples: Sequence[Any], manifest: dict[str, Any]
) -> None:
    records = manifest["samples"]
    if len(samples) != len(records):
        raise RuntimeError("loaded samples and selection manifest differ")
    for sample, record in zip(samples, records, strict=True):
        if (
            sample.filename != record["filename"]
            or sample.row != int(record["row"])
            or sample.col != int(record["col"])
        ):
            raise RuntimeError(f"sample identity mismatch: {sample.filename}")
        for key in FINGERPRINT_KEYS:
            if sample.fingerprint.get(key) != record.get(key):
                raise RuntimeError(
                    f"dataset fingerprint mismatch for {sample.filename}: {key}"
                )


def metric_consistency(
    recomputed: dict[str, dict[str, Any]], stored: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for weights in ("raw", "ema"):
        current = recomputed[weights]["per_sample"]
        previous = stored[weights]["per_sample"]
        if current.keys() != previous.keys():
            raise RuntimeError(f"{weights} recomputed and stored sample sets differ")
        largest = 0.0
        largest_location: dict[str, Any] | None = None
        compared = 0
        for filename in current:
            common = current[filename].keys() & previous[filename].keys()
            for metric in common:
                before = previous[filename][metric]
                after = current[filename][metric]
                if not isinstance(before, (int, float)) or not isinstance(
                    after, (int, float)
                ):
                    continue
                difference = abs(float(after) - float(before))
                compared += 1
                if difference > largest:
                    largest = difference
                    largest_location = {
                        "filename": filename,
                        "metric": metric,
                        "stored": float(before),
                        "recomputed": float(after),
                    }
        result[weights] = {
            "compared_values": compared,
            "maximum_absolute_difference": largest,
            "maximum_difference_location": largest_location,
        }
    return result


def export_sample_figure(
    arrays: tuple[np.ndarray, ...],
    path: Path,
    *,
    filename: str,
    raw_metrics: dict[str, float],
    ema_metrics: dict[str, float],
    step: int,
    floor_db: float,
    dpi: int,
) -> None:
    titles = (
        "Echo",
        "RAW prediction",
        "EMA prediction",
        "Oracle phase",
        "Image",
    )
    target_peak = max(float(np.abs(arrays[-1]).max()), np.finfo(np.float64).tiny)
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
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
        f"E011-B checkpoint step={step}: {filename}\n"
        f"RAW phase={raw_metrics['weighted_phase_alignment']:.4f} "
        f"gap={raw_metrics['rmse_excess_over_oracle']:.4f} "
        f"coh_frac={raw_metrics['coherence_fraction_of_oracle']:.4f} "
        f"SSIM_gain={raw_metrics['ssim_gain_fraction_of_oracle']:.4f} "
        f"edge_gain={raw_metrics['edge_gain_fraction_of_oracle']:.4f}\n"
        f"EMA phase={ema_metrics['weighted_phase_alignment']:.4f} "
        f"gap={ema_metrics['rmse_excess_over_oracle']:.4f} "
        f"coh_frac={ema_metrics['coherence_fraction_of_oracle']:.4f} "
        f"SSIM_gain={ema_metrics['ssim_gain_fraction_of_oracle']:.4f} "
        f"edge_gain={ema_metrics['edge_gain_fraction_of_oracle']:.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[
        tuple[
            str,
            tuple[np.ndarray, ...],
            dict[str, float],
            dict[str, float],
        ]
    ],
    path: Path,
    *,
    step: int,
    floor_db: float,
    dpi: int,
) -> None:
    titles = (
        "Echo",
        "RAW prediction",
        "EMA prediction",
        "Oracle phase",
        "Image",
    )
    figure, axes = plt.subplots(
        len(rows),
        5,
        figsize=(20, 3.2 * len(rows)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, (filename, arrays, raw, ema) in enumerate(rows):
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
            f"{Path(filename).stem}\n"
            f"RAW phase={raw['weighted_phase_alignment']:.3f} "
            f"gap={raw['rmse_excess_over_oracle']:.3f}\n"
            f"EMA phase={ema['weighted_phase_alignment']:.3f} "
            f"gap={ema['rmse_excess_over_oracle']:.3f}",
            fontsize=7,
        )
    figure.suptitle(
        f"E011-B fixed training-subset audit, checkpoint step={step}\n"
        "Every row uses the corresponding Image peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.dpi <= 0 or args.contact_sheet_page_size <= 0:
        raise ValueError("dpi and contact_sheet_page_size must be positive")
    for role, directory in (("Echo", args.echo_dir), ("Image", args.image_dir)):
        if not directory.is_dir():
            raise FileNotFoundError(f"{role} directory does not exist: {directory}")
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")

    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["resolved_config"]
    manifest = config["selection_manifest"]
    samples = load_phase_samples(
        manifest_pairs(manifest, args.echo_dir, args.image_dir),
        expected_shape=tuple(int(value) for value in config["data"]["expected_shape"]),
        data_config=config["data"],
        optimization=config["optimization"],
        evaluation=config["evaluation"],
    )
    validate_fingerprints(samples, manifest)

    device = resolve_device(args.device)
    precision = resolve_precision(device)
    model = SwinIR(**config["model"]).to(device)
    ema_model = SwinIR(**config["model"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    model.eval()
    ema_model.eval()
    criteria = PhaseSuccessCriteria(**config["evaluation"]["success_criteria"])
    predictions, summaries = evaluate_patch_set(
        model,
        ema_model,
        samples,
        criteria,
        device=device,
        precision=precision,
        data_config=config["data"],
        optimization=config["optimization"],
        evaluation=config["evaluation"],
    )
    consistency = metric_consistency(summaries, checkpoint["last_metrics"])

    args.output_dir.mkdir(parents=True)
    samples_dir = args.output_dir / "samples"
    contact_dir = args.output_dir / "contact_sheets"
    samples_dir.mkdir()
    contact_dir.mkdir()
    floor_db = float(config["evaluation"]["log_magnitude_floor_db"])
    step = int(checkpoint["step"])
    contact_rows = []
    entries = []
    for index, sample in enumerate(samples):
        _, _, raw_prediction, ema_prediction = predictions[sample.filename]
        arrays = tuple(
            tensor_to_complex(tensor)
            for tensor in (
                sample.echo_image,
                raw_prediction,
                ema_prediction,
                sample.oracle_prediction,
                sample.target_image,
            )
        )
        raw = summaries["raw"]["per_sample"][sample.filename]
        ema = summaries["ema"]["per_sample"][sample.filename]
        figure_name = f"{index:03d}_{Path(sample.filename).stem}.png"
        export_sample_figure(
            arrays,
            samples_dir / figure_name,
            filename=sample.filename,
            raw_metrics=raw,
            ema_metrics=ema,
            step=step,
            floor_db=floor_db,
            dpi=args.dpi,
        )
        contact_rows.append((sample.filename, arrays, raw, ema))
        entries.append(
            {
                "selection_index": index,
                "filename": sample.filename,
                "row": sample.row,
                "col": sample.col,
                "figure": str(Path("samples") / figure_name),
                "raw_metrics": raw,
                "ema_metrics": ema,
                "stored_raw_metrics": checkpoint["last_metrics"]["raw"][
                    "per_sample"
                ][sample.filename],
                "stored_ema_metrics": checkpoint["last_metrics"]["ema"][
                    "per_sample"
                ][sample.filename],
            }
        )
        print(
            f"[{index + 1:02d}/{len(samples):02d}] {sample.filename} "
            f"RAW phase={raw['weighted_phase_alignment']:.4f} "
            f"gap={raw['rmse_excess_over_oracle']:.4f} "
            f"EMA phase={ema['weighted_phase_alignment']:.4f}",
            flush=True,
        )

    contact_sheets = []
    page_size = int(args.contact_sheet_page_size)
    for page, start in enumerate(range(0, len(contact_rows), page_size), start=1):
        name = f"audit_page_{page:03d}.png"
        export_contact_sheet(
            contact_rows[start : start + page_size],
            contact_dir / name,
            step=step,
            floor_db=floor_db,
            dpi=args.dpi,
        )
        contact_sheets.append(str(Path("contact_sheets") / name))

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_step": step,
        "stored_metrics_step": int(checkpoint["last_metrics"]["step"]),
        "sample_count": len(samples),
        "selection_manifest_fingerprint": manifest.get("fingerprint"),
        "device": str(device),
        "precision": precision.as_dict(),
        "success_criteria": config["evaluation"]["success_criteria"],
        "metric_consistency": consistency,
        "raw_summary": summaries["raw"],
        "ema_summary": summaries["ema"],
        "contact_sheets": contact_sheets,
        "samples": entries,
    }
    write_json(args.output_dir / "audit_manifest.json", report)
    print(f"audit_manifest={(args.output_dir / 'audit_manifest.json').resolve()}")
    for name in contact_sheets:
        print(f"contact_sheet={(args.output_dir / name).resolve()}")
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
