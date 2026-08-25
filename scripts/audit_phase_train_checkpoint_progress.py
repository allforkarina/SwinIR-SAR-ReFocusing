"""E011-A: compare E010 initial and final checkpoints on seen training patches."""

from __future__ import annotations

import argparse
import gc
import hashlib
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
from scripts.overfit_single_phase_correction import evaluate_correction, predict_correction
from scripts.train_phase_spatial_holdout import (
    METRIC_NAMES,
    add_generalization_metrics,
    aggregate_metrics,
)
from scripts.visualize_phase_spatial_holdout_checkpoint import load_checkpoint
from swinir import SwinIR
from swinir.sar_dataset import (
    CoordinateRegion,
    DiscoveredPair,
    PairRecord,
    SplitName,
    build_manifest,
)
from swinir.sar_metrics import log_magnitude_image
from swinir.training import PrecisionPolicy, resolve_device, resolve_precision


REPORT_SCHEMA_VERSION = 1
SELECTION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointIdentity:
    step: int
    manifest_fingerprint: str
    resolved_config: dict[str, Any]


@dataclass(frozen=True)
class VisualSelection:
    filename: str
    reasons: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare an E010 initial checkpoint with its final checkpoint on a "
            "deterministic spatial probe drawn only from the training split."
        )
    )
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=441)
    parser.add_argument("--visual-count", type=int, default=12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--contact-sheet-page-size", type=int, default=12)
    parser.add_argument("--phase-delta-min", type=float, default=0.05)
    parser.add_argument("--rmse-gap-delta-min", type=float, default=0.05)
    parser.add_argument("--coherence-delta-min", type=float, default=0.05)
    parser.add_argument("--rmse-win-fraction-min", type=float, default=0.75)
    return parser.parse_args()


def pair_from_record(record: PairRecord) -> DiscoveredPair:
    return DiscoveredPair(
        row=record.row,
        col=record.col,
        echo_path=record.echo_path,
        image_path=record.image_path,
    )


def select_spatial_probe(
    records: Sequence[PairRecord], sample_count: int
) -> tuple[PairRecord, ...]:
    """Deterministic normalized farthest-point sampling without loading pixels."""

    ordered = tuple(
        sorted(records, key=lambda item: (item.row, item.col, item.echo_path.name))
    )
    if not 0 < sample_count <= len(ordered):
        raise ValueError(f"sample_count must be in [1, {len(ordered)}]")
    rows = np.asarray([record.row for record in ordered], dtype=np.float64)
    cols = np.asarray([record.col for record in ordered], dtype=np.float64)
    row_span = max(float(rows.max() - rows.min()), 1.0)
    col_span = max(float(cols.max() - cols.min()), 1.0)
    coordinates = np.stack(
        ((rows - rows.min()) / row_span, (cols - cols.min()) / col_span), axis=1
    )
    selected_indices = [0]
    selected_mask = np.zeros(len(ordered), dtype=bool)
    selected_mask[0] = True
    minimum_distance = np.full(len(ordered), np.inf, dtype=np.float64)
    while len(selected_indices) < sample_count:
        latest = coordinates[selected_indices[-1]]
        squared_distance = np.square(coordinates - latest).sum(axis=1)
        minimum_distance = np.minimum(minimum_distance, squared_distance)
        minimum_distance[selected_mask] = -np.inf
        next_index = int(np.argmax(minimum_distance))
        if not np.isfinite(minimum_distance[next_index]):
            raise RuntimeError("spatial probe selection exhausted eligible records")
        selected_indices.append(next_index)
        selected_mask[next_index] = True
    return tuple(ordered[index] for index in selected_indices)


def selection_manifest(
    records: Sequence[PairRecord], *, manifest_fingerprint: str
) -> dict[str, Any]:
    samples = [
        {
            "selection_index": index,
            "filename": record.echo_path.name,
            "row": record.row,
            "col": record.col,
            "split": record.split.value,
        }
        for index, record in enumerate(records)
    ]
    fingerprint = hashlib.sha256(
        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "selection_method": "lexicographic_anchor_normalized_farthest_point",
        "source_split": SplitName.TRAIN.value,
        "sample_count": len(samples),
        "dataset_manifest_fingerprint": manifest_fingerprint,
        "selection_fingerprint": fingerprint,
        "samples": samples,
    }


def checkpoint_identity(checkpoint: dict[str, Any]) -> CheckpointIdentity:
    return CheckpointIdentity(
        step=int(checkpoint["global_step"]),
        manifest_fingerprint=str(checkpoint["manifest_fingerprint"]),
        resolved_config=checkpoint["resolved_config"],
    )


def validate_checkpoint_pair(
    initial: CheckpointIdentity, final: CheckpointIdentity
) -> None:
    if initial.manifest_fingerprint != final.manifest_fingerprint:
        raise RuntimeError("initial and final checkpoints use different manifests")
    if initial.resolved_config != final.resolved_config:
        raise RuntimeError("initial and final checkpoints use different configurations")
    if initial.step >= final.step:
        raise RuntimeError("initial checkpoint step must precede final checkpoint step")


def load_identity(path: Path) -> CheckpointIdentity:
    checkpoint = load_checkpoint(path)
    identity = checkpoint_identity(checkpoint)
    del checkpoint
    gc.collect()
    return identity


def load_one_sample(
    record: PairRecord,
    *,
    config: dict[str, Any],
):
    data = config["data"]
    return load_phase_samples(
        (pair_from_record(record),),
        expected_shape=tuple(int(value) for value in data["expected_shape"]),
        data_config=data,
        optimization=config["optimization"],
        evaluation=config["evaluation"],
    )[0]


def evaluate_checkpoint_on_probe(
    checkpoint_path: Path,
    records: Sequence[PairRecord],
    *,
    expected_identity: CheckpointIdentity,
    device: torch.device,
    precision: PrecisionPolicy,
    label: str,
) -> dict[str, dict[str, float]]:
    checkpoint = load_checkpoint(checkpoint_path)
    identity = checkpoint_identity(checkpoint)
    if identity != expected_identity:
        raise RuntimeError(f"{label} checkpoint identity changed while auditing")
    config = identity.resolved_config
    model = SwinIR(**config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    del checkpoint
    metrics_by_filename: dict[str, dict[str, float]] = {}
    for index, record in enumerate(records, start=1):
        sample = load_one_sample(record, config=config)
        correction = predict_correction(
            model,
            sample.input_spectrum,
            device=device,
            precision=precision,
            phasor_epsilon=float(config["optimization"]["phasor_epsilon"]),
        )
        _, base_metrics = evaluate_correction(
            correction,
            sample.input_spectrum,
            sample.target_phasor,
            sample.phase_weights,
            sample.target_image,
            fft_norm=str(config["data"]["fft_norm"]),
            floor_db=float(config["evaluation"]["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                config["evaluation"]["high_frequency_radius_fraction"]
            ),
        )
        metrics_by_filename[sample.filename] = add_generalization_metrics(
            base_metrics, sample.echo_metrics, sample.oracle_metrics
        )
        if index % 32 == 0 or index == len(records):
            print(f"{label} {index}/{len(records)}", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return metrics_by_filename


def compare_checkpoints(
    initial_metrics: dict[str, dict[str, float]],
    final_metrics: dict[str, dict[str, float]],
    *,
    phase_delta_min: float,
    rmse_gap_delta_min: float,
    coherence_delta_min: float,
    rmse_win_fraction_min: float,
) -> dict[str, Any]:
    if initial_metrics.keys() != final_metrics.keys():
        raise RuntimeError("initial and final metric samples differ")
    initial_summary = aggregate_metrics(initial_metrics)
    final_summary = aggregate_metrics(final_metrics)
    aggregate_deltas: dict[str, dict[str, float]] = {}
    for name in METRIC_NAMES:
        aggregate_deltas[name] = {
            statistic: float(
                final_summary["aggregate"][name][statistic]
                - initial_summary["aggregate"][name][statistic]
            )
            for statistic in ("mean", "median", "p05", "p95")
        }
    rmse_win_fraction = float(
        np.mean(
            [
                final_metrics[name]["normalized_complex_rmse"]
                < initial_metrics[name]["normalized_complex_rmse"]
                for name in initial_metrics
            ]
        )
    )
    values = {
        "mean_phase_alignment_delta": aggregate_deltas[
            "weighted_phase_alignment"
        ]["mean"],
        "median_phase_alignment_delta": aggregate_deltas[
            "weighted_phase_alignment"
        ]["median"],
        "mean_rmse_oracle_gap_fraction_closed_delta": aggregate_deltas[
            "rmse_oracle_gap_fraction_closed"
        ]["mean"],
        "mean_coherence_fraction_of_oracle_delta": aggregate_deltas[
            "coherence_fraction_of_oracle"
        ]["mean"],
        "mean_ssim_gain_fraction_of_oracle_delta": aggregate_deltas[
            "ssim_gain_fraction_of_oracle"
        ]["mean"],
        "mean_edge_gain_fraction_of_oracle_delta": aggregate_deltas[
            "edge_gain_fraction_of_oracle"
        ]["mean"],
        "final_rmse_win_fraction_vs_initial": rmse_win_fraction,
    }
    checks = {
        "mean_phase_alignment_delta": values["mean_phase_alignment_delta"]
        >= phase_delta_min,
        "median_phase_alignment_delta": values["median_phase_alignment_delta"]
        >= phase_delta_min,
        "mean_rmse_oracle_gap_fraction_closed_delta": values[
            "mean_rmse_oracle_gap_fraction_closed_delta"
        ]
        >= rmse_gap_delta_min,
        "mean_coherence_fraction_of_oracle_delta": values[
            "mean_coherence_fraction_of_oracle_delta"
        ]
        >= coherence_delta_min,
        "final_rmse_win_fraction_vs_initial": rmse_win_fraction
        >= rmse_win_fraction_min,
    }
    initial_aggregate = {
        key: value for key, value in initial_summary.items() if key != "per_sample"
    }
    final_aggregate = {
        key: value for key, value in final_summary.items() if key != "per_sample"
    }
    return {
        "thresholds": {
            "phase_delta_min": phase_delta_min,
            "rmse_gap_delta_min": rmse_gap_delta_min,
            "coherence_delta_min": coherence_delta_min,
            "rmse_win_fraction_min": rmse_win_fraction_min,
        },
        **values,
        "checks": checks,
        "metric_training_signal_supported": all(checks.values()),
        "aggregate_deltas": aggregate_deltas,
        "initial_summary": initial_aggregate,
        "final_summary": final_aggregate,
    }


def select_visual_samples(
    initial_metrics: dict[str, dict[str, float]],
    final_metrics: dict[str, dict[str, float]],
    coordinates: dict[str, tuple[int, int]],
    count: int,
) -> tuple[VisualSelection, ...]:
    if not (
        initial_metrics.keys() == final_metrics.keys() == coordinates.keys()
    ):
        raise RuntimeError("visual selection inputs contain different samples")
    if not 0 < count <= len(final_metrics):
        raise ValueError(f"visual_count must be in [1, {len(final_metrics)}]")
    reasons: dict[str, list[str]] = {}
    order: list[str] = []

    def add(filename: str, reason: str) -> None:
        if filename not in reasons:
            reasons[filename] = []
            order.append(filename)
        reasons[filename].append(reason)

    def delta(filename: str, metric: str) -> float:
        return final_metrics[filename][metric] - initial_metrics[filename][metric]

    for metric, label in (
        ("weighted_phase_alignment", "phase_delta"),
        ("rmse_oracle_gap_fraction_closed", "rmse_gap_delta"),
        ("coherence_fraction_of_oracle", "coherence_delta"),
    ):
        ranked = sorted(initial_metrics, key=lambda name: (delta(name, metric), name))
        add(ranked[0], f"lowest_{label}")
        add(ranked[-1], f"highest_{label}")
    final_phase = sorted(
        final_metrics,
        key=lambda name: (final_metrics[name]["weighted_phase_alignment"], name),
    )
    add(final_phase[0], "lowest_final_phase")
    add(final_phase[-1], "highest_final_phase")
    spatial = sorted(coordinates, key=lambda name: (*coordinates[name], name))
    spatial_indices = np.linspace(0, len(spatial) - 1, num=min(len(spatial), count * 3))
    for index, spatial_index in enumerate(spatial_indices.round().astype(int)):
        add(spatial[int(spatial_index)], f"spatial_quantile_{index:02d}")
        if len(order) >= count:
            break
    for filename in spatial:
        if len(order) >= count:
            break
        add(filename, "spatial_fill")
    return tuple(
        VisualSelection(filename, tuple(reasons[filename]))
        for filename in order[:count]
    )


def predict_selected(
    checkpoint_path: Path,
    records: Sequence[PairRecord],
    *,
    expected_identity: CheckpointIdentity,
    device: torch.device,
    precision: PrecisionPolicy,
) -> dict[str, np.ndarray]:
    checkpoint = load_checkpoint(checkpoint_path)
    identity = checkpoint_identity(checkpoint)
    if identity != expected_identity:
        raise RuntimeError("checkpoint identity changed while exporting figures")
    config = identity.resolved_config
    model = SwinIR(**config["model"])
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    del checkpoint
    predictions: dict[str, np.ndarray] = {}
    for record in records:
        sample = load_one_sample(record, config=config)
        correction = predict_correction(
            model,
            sample.input_spectrum,
            device=device,
            precision=precision,
            phasor_epsilon=float(config["optimization"]["phasor_epsilon"]),
        )
        prediction, _ = evaluate_correction(
            correction,
            sample.input_spectrum,
            sample.target_phasor,
            sample.phase_weights,
            sample.target_image,
            fft_norm=str(config["data"]["fft_norm"]),
            floor_db=float(config["evaluation"]["log_magnitude_floor_db"]),
            high_frequency_radius_fraction=float(
                config["evaluation"]["high_frequency_radius_fraction"]
            ),
        )
        predictions[sample.filename] = tensor_to_complex(prediction)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return predictions


def export_sample_figure(
    arrays: tuple[np.ndarray, ...],
    path: Path,
    *,
    filename: str,
    reasons: Sequence[str],
    initial_step: int,
    final_step: int,
    initial_metrics: dict[str, float],
    final_metrics: dict[str, float],
    floor_db: float,
    dpi: int,
) -> None:
    titles = (
        "Echo",
        f"Initial prediction (step {initial_step})",
        f"Final prediction (step {final_step})",
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
    phase_delta = (
        final_metrics["weighted_phase_alignment"]
        - initial_metrics["weighted_phase_alignment"]
    )
    gap_delta = (
        final_metrics["rmse_oracle_gap_fraction_closed"]
        - initial_metrics["rmse_oracle_gap_fraction_closed"]
    )
    coherence_delta = (
        final_metrics["coherence_fraction_of_oracle"]
        - initial_metrics["coherence_fraction_of_oracle"]
    )
    figure.suptitle(
        f"{filename} selection={','.join(reasons)}\n"
        f"final phase={final_metrics['weighted_phase_alignment']:.4f} "
        f"delta phase={phase_delta:+.4f} delta gap={gap_delta:+.4f} "
        f"delta coh={coherence_delta:+.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def export_contact_sheet(
    rows: Sequence[tuple[str, tuple[np.ndarray, ...], dict[str, float]]],
    path: Path,
    *,
    initial_step: int,
    final_step: int,
    floor_db: float,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        len(rows), 5, figsize=(20, 3.2 * len(rows)), constrained_layout=True, squeeze=False
    )
    titles = (
        "Echo",
        f"Initial step {initial_step}",
        f"Final step {final_step}",
        "Oracle phase",
        "Image",
    )
    for row_index, (filename, arrays, deltas) in enumerate(rows):
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
            f"dphase={deltas['weighted_phase_alignment']:+.3f} "
            f"dgap={deltas['rmse_oracle_gap_fraction_closed']:+.3f}",
            fontsize=7,
        )
    figure.suptitle(
        "E011-A seen-training checkpoint progress audit\n"
        "Every row shares its Image target peak",
        fontsize=14,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if args.sample_count <= 0 or args.visual_count <= 0:
        raise ValueError("sample counts must be positive")
    if args.visual_count > args.sample_count:
        raise ValueError("visual_count cannot exceed sample_count")
    if args.dpi <= 0 or args.contact_sheet_page_size <= 0:
        raise ValueError("dpi and contact sheet page size must be positive")
    for name in (
        "phase_delta_min",
        "rmse_gap_delta_min",
        "coherence_delta_min",
        "rmse_win_fraction_min",
    ):
        value = float(getattr(args, name))
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    initial_identity = load_identity(args.initial_checkpoint)
    final_identity = load_identity(args.final_checkpoint)
    validate_checkpoint_pair(initial_identity, final_identity)
    config = initial_identity.resolved_config
    data = config["data"]
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**data["validation_region"]),
        CoordinateRegion(**data["guard_region"]),
        expected_counts=data["expected_split_counts"],
    )
    if manifest.fingerprint != initial_identity.manifest_fingerprint:
        raise RuntimeError("dataset manifest fingerprint does not match checkpoints")
    train_records = manifest.records_for(SplitName.TRAIN)
    probe_records = select_spatial_probe(train_records, int(args.sample_count))
    selected_manifest = selection_manifest(
        probe_records, manifest_fingerprint=manifest.fingerprint
    )
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "selected_train_samples.json", selected_manifest)
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    initial_metrics = evaluate_checkpoint_on_probe(
        args.initial_checkpoint,
        probe_records,
        expected_identity=initial_identity,
        device=device,
        precision=precision,
        label=f"initial_step_{initial_identity.step}",
    )
    final_metrics = evaluate_checkpoint_on_probe(
        args.final_checkpoint,
        probe_records,
        expected_identity=final_identity,
        device=device,
        precision=precision,
        label=f"final_step_{final_identity.step}",
    )
    comparison = compare_checkpoints(
        initial_metrics,
        final_metrics,
        phase_delta_min=float(args.phase_delta_min),
        rmse_gap_delta_min=float(args.rmse_gap_delta_min),
        coherence_delta_min=float(args.coherence_delta_min),
        rmse_win_fraction_min=float(args.rmse_win_fraction_min),
    )
    coordinates = {
        record.echo_path.name: (record.row, record.col) for record in probe_records
    }
    visual_selections = select_visual_samples(
        initial_metrics,
        final_metrics,
        coordinates,
        int(args.visual_count),
    )
    record_by_filename = {
        record.echo_path.name: record for record in probe_records
    }
    visual_records = tuple(
        record_by_filename[selection.filename] for selection in visual_selections
    )
    initial_predictions = predict_selected(
        args.initial_checkpoint,
        visual_records,
        expected_identity=initial_identity,
        device=device,
        precision=precision,
    )
    final_predictions = predict_selected(
        args.final_checkpoint,
        visual_records,
        expected_identity=final_identity,
        device=device,
        precision=precision,
    )
    samples_dir = args.output_dir / "samples"
    samples_dir.mkdir()
    contact_rows = []
    sample_entries = []
    floor_db = float(config["evaluation"]["log_magnitude_floor_db"])
    for index, (record, selection) in enumerate(
        zip(visual_records, visual_selections, strict=True)
    ):
        sample = load_one_sample(record, config=config)
        arrays = (
            tensor_to_complex(sample.echo_image),
            initial_predictions[sample.filename],
            final_predictions[sample.filename],
            tensor_to_complex(sample.oracle_prediction),
            tensor_to_complex(sample.target_image),
        )
        metric_deltas = {
            name: float(
                final_metrics[sample.filename][name]
                - initial_metrics[sample.filename][name]
            )
            for name in METRIC_NAMES
        }
        figure_name = f"{index:03d}_{Path(sample.filename).stem}.png"
        export_sample_figure(
            arrays,
            samples_dir / figure_name,
            filename=sample.filename,
            reasons=selection.reasons,
            initial_step=initial_identity.step,
            final_step=final_identity.step,
            initial_metrics=initial_metrics[sample.filename],
            final_metrics=final_metrics[sample.filename],
            floor_db=floor_db,
            dpi=int(args.dpi),
        )
        contact_rows.append((sample.filename, arrays, metric_deltas))
        sample_entries.append(
            {
                "selection_index": index,
                "filename": sample.filename,
                "row": sample.row,
                "col": sample.col,
                "selection_reasons": list(selection.reasons),
                "figure": str(Path("samples") / figure_name),
                "initial_metrics": initial_metrics[sample.filename],
                "final_metrics": final_metrics[sample.filename],
                "metric_deltas": metric_deltas,
            }
        )
    contact_sheets = []
    page_size = int(args.contact_sheet_page_size)
    for page, start in enumerate(range(0, len(contact_rows), page_size), start=1):
        name = (
            f"audit_{len(contact_rows):03d}_training_samples.png"
            if len(contact_rows) <= page_size
            else f"audit_page_{page:03d}.png"
        )
        export_contact_sheet(
            contact_rows[start : start + page_size],
            args.output_dir / name,
            initial_step=initial_identity.step,
            final_step=final_identity.step,
            floor_db=floor_db,
            dpi=int(args.dpi),
        )
        contact_sheets.append(name)
    status = (
        "training_signal_supported"
        if comparison["metric_training_signal_supported"]
        else "training_signal_not_supported"
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "E011-A-D001-seen-training-checkpoint-progress",
        "status": status,
        "interpretation_contract": (
            "Metric status is diagnostic; final acceptance also requires manual review."
        ),
        "initial_checkpoint": str(args.initial_checkpoint.resolve()),
        "initial_step": initial_identity.step,
        "final_checkpoint": str(args.final_checkpoint.resolve()),
        "final_step": final_identity.step,
        "source_split": SplitName.TRAIN.value,
        "sample_count": len(probe_records),
        "visual_count": len(visual_records),
        "selection_fingerprint": selected_manifest["selection_fingerprint"],
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "precision": precision.as_dict(),
        "comparison": comparison,
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "contact_sheets": contact_sheets,
        "samples": sample_entries,
    }
    write_json(args.output_dir / "report.json", report)
    print(
        f"status={status} report={(args.output_dir / 'report.json').resolve()}",
        flush=True,
    )
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
