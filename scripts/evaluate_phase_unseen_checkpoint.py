"""E012: evaluate a frozen E011-B checkpoint on unseen spatial holdout patches.

The checkpoint is an immutable input.  The script neither updates model weights
nor uses holdout metrics to choose a checkpoint, so its output is a zero-training
same-scene generalization audit rather than a training run.
"""

from __future__ import annotations

import argparse
import copy
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
import yaml
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.overfit_single_patch import sample_fingerprint, tensor_to_complex, write_json
from scripts.overfit_single_phase_correction import (
    apply_phase_correction,
    evaluate_correction,
    predict_correction,
)
from scripts.overfit_phase_train_subset import EXPERIMENT as E011B_EXPERIMENT
from scripts.train_phase_spatial_holdout import (
    PhasePatchDataset,
    compare_to_baselines,
    evaluate_baselines,
    evaluate_validation,
)
from swinir import SwinIR
from swinir.sar_dataset import CoordinateRegion, DiscoveredPair, SplitName, build_manifest
from swinir.sar_metrics import log_magnitude_image
from swinir.training import resolve_device, resolve_precision


CHECKPOINT_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
FINGERPRINT_KEYS = (
    "echo_sha256",
    "image_sha256",
    "echo_size_bytes",
    "image_size_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen E011-B checkpoint on unseen spatial holdout patches."
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/evaluate_phase_unseen_checkpoint.yaml")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("top-level configuration must be a mapping")
    for name in ("checkpoint", "data", "selection", "evaluation", "output"):
        if not isinstance(config.get(name), dict):
            raise ValueError(f"config.{name} must be a mapping")
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("experiment") != "E012-D001-frozen-e011b-unseen-spatial-evaluation":
        raise ValueError("config.experiment must identify E012")
    checkpoint = config["checkpoint"]
    if checkpoint.get("expected_experiment") != E011B_EXPERIMENT:
        raise ValueError("E012 requires an E011-B checkpoint")
    if int(checkpoint.get("expected_step", -1)) < 0:
        raise ValueError("checkpoint.expected_step must be non-negative")
    if int(checkpoint.get("expected_sample_count", 0)) <= 0:
        raise ValueError("checkpoint.expected_sample_count must be positive")
    if not isinstance(checkpoint.get("require_dataset_manifest_fingerprint"), bool):
        raise ValueError("checkpoint.require_dataset_manifest_fingerprint must be boolean")
    data = config["data"]
    if data.get("representation") != "fftshifted_echo_complex_spectrum_to_unit_phase_correction":
        raise ValueError("E012 supports only the unit-phasor phase representation")
    shape = tuple(int(value) for value in data.get("expected_shape", ()))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("data.expected_shape must contain two positive dimensions")
    if data.get("fft_norm") not in ("ortho", "backward", "forward"):
        raise ValueError("data.fft_norm is unsupported")
    selection = config["selection"]
    if selection.get("source_split") != SplitName.VALIDATION.value:
        raise ValueError("E012 must evaluate only the validation split")
    for name in ("validation_region", "guard_region", "expected_split_counts"):
        if not isinstance(selection.get(name), dict):
            raise ValueError(f"selection.{name} must be a mapping")
    if int(config["output"].get("representative_sample_count", 0)) <= 0:
        raise ValueError("output.representative_sample_count must be positive")
    if int(config["output"].get("figure_dpi", 0)) <= 0:
        raise ValueError("output.figure_dpi must be positive")


def load_checkpoint(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected = config["checkpoint"]
    if checkpoint.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError("unsupported E011-B checkpoint schema")
    resolved = checkpoint.get("resolved_config")
    if not isinstance(resolved, dict) or resolved.get("experiment") != expected["expected_experiment"]:
        raise RuntimeError("checkpoint is not the expected E011-B experiment")
    checkpoint_data = resolved.get("data")
    if not isinstance(checkpoint_data, dict):
        raise RuntimeError("checkpoint is missing its phase data contract")
    expected_data = config["data"]
    comparable_data = ("expected_shape", "rms_epsilon", "fft_norm", "representation")
    for name in comparable_data:
        if checkpoint_data.get(name) != expected_data.get(name):
            raise RuntimeError(
                f"checkpoint and E012 data contracts differ for {name}: "
                f"checkpoint={checkpoint_data.get(name)!r}, config={expected_data.get(name)!r}"
            )
    if checkpoint.get("step") != int(expected["expected_step"]):
        raise RuntimeError(
            f"checkpoint step does not match E012 contract: expected={expected['expected_step']}, "
            f"actual={checkpoint.get('step')}"
        )
    for name in ("model", "ema_model"):
        if not isinstance(checkpoint.get(name), dict):
            raise RuntimeError(f"checkpoint is missing {name} weights")
    metrics = checkpoint.get("last_metrics")
    if not isinstance(metrics, dict) or metrics.get("step") != checkpoint["step"]:
        raise RuntimeError("checkpoint weights and stored metrics are from different steps")
    selection_manifest = resolved.get("selection_manifest")
    if not isinstance(selection_manifest, dict):
        raise RuntimeError("checkpoint is missing its selected training-sample manifest")
    samples = selection_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != int(expected["expected_sample_count"]):
        raise RuntimeError("checkpoint selected training-sample count does not match E012 contract")
    return checkpoint


def validate_checkpoint_data(
    checkpoint: dict[str, Any],
    *,
    manifest: Any,
    echo_dir: Path,
    image_dir: Path,
    patch_shape: tuple[int, int],
    require_dataset_manifest_fingerprint: bool,
) -> list[dict[str, Any]]:
    selection = checkpoint["resolved_config"]["selection_manifest"]
    expected_manifest = selection.get("dataset_manifest_fingerprint")
    if require_dataset_manifest_fingerprint and expected_manifest != manifest.fingerprint:
        raise RuntimeError(
            "current dataset manifest differs from the dataset used by the checkpoint"
        )
    records_by_name = {record.echo_path.name: record for record in manifest.records}
    validated: list[dict[str, Any]] = []
    for item in selection["samples"]:
        filename = item.get("filename")
        if not isinstance(filename, str) or filename not in records_by_name:
            raise RuntimeError(f"checkpoint training sample is absent from current dataset: {filename!r}")
        record = records_by_name[filename]
        if record.split is not SplitName.TRAIN:
            raise RuntimeError(f"checkpoint training sample is no longer in train split: {filename}")
        if record.row != int(item["row"]) or record.col != int(item["col"]):
            raise RuntimeError(f"checkpoint training coordinate differs: {filename}")
        current = sample_fingerprint(echo_dir / filename, image_dir / filename)
        mismatched = [key for key in FINGERPRINT_KEYS if current[key] != item.get(key)]
        if mismatched:
            raise RuntimeError(
                f"checkpoint training sample content differs: {filename}, fields={mismatched}"
            )
        validated.append({"filename": filename, "row": record.row, "col": record.col})
    for validation in manifest.records_for(SplitName.VALIDATION):
        for training in validated:
            if (
                abs(validation.row - int(training["row"])) < patch_shape[0]
                and abs(validation.col - int(training["col"])) < patch_shape[1]
            ):
                raise RuntimeError(
                    "validation patch overlaps a checkpoint training patch: "
                    f"validation={validation.echo_path.name}, training={training['filename']}"
                )
    return validated


def select_representatives(
    metrics: dict[str, dict[str, float]],
    coordinates: dict[str, tuple[int, int]],
    count: int,
) -> list[str]:
    if count <= 0:
        raise ValueError("representative count must be positive")
    if metrics.keys() != coordinates.keys():
        raise ValueError("metrics and coordinates must describe the same samples")
    ordered = sorted(metrics)
    candidates = [
        min(ordered, key=lambda name: (metrics[name]["weighted_phase_alignment"], name)),
        max(ordered, key=lambda name: (metrics[name]["weighted_phase_alignment"], name)),
        min(ordered, key=lambda name: (metrics[name]["rmse_oracle_gap_fraction_closed"], name)),
        max(ordered, key=lambda name: (metrics[name]["rmse_oracle_gap_fraction_closed"], name)),
    ]
    spatial = sorted(ordered, key=lambda name: (*coordinates[name], name))
    if count >= len(spatial):
        candidates.extend(spatial)
    else:
        positions = np.linspace(0, len(spatial) - 1, num=count, dtype=int)
        candidates.extend(spatial[position] for position in positions)
    selected: list[str] = []
    for name in candidates:
        if name not in selected:
            selected.append(name)
        if len(selected) == min(count, len(ordered)):
            break
    return selected


def _identity_like(target: torch.Tensor) -> torch.Tensor:
    identity = torch.zeros_like(target)
    identity[:, 0] = 1.0
    return identity


def export_figure(
    path: Path,
    *,
    filename: str,
    echo: torch.Tensor,
    raw: torch.Tensor,
    ema: torch.Tensor,
    oracle: torch.Tensor,
    image: torch.Tensor,
    raw_metrics: dict[str, float],
    ema_metrics: dict[str, float],
    floor_db: float,
    dpi: int,
) -> None:
    arrays = tuple(tensor_to_complex(value) for value in (echo, raw, ema, oracle, image))
    titles = ("Echo", "RAW prediction", "EMA prediction", "Oracle phase", "Image")
    target_peak = max(float(np.abs(arrays[-1]).max()), np.finfo(np.float64).tiny)
    figure, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    for column, (array, title) in enumerate(zip(arrays, titles, strict=True)):
        own_peak = max(float(np.abs(array).max()), np.finfo(np.float64).tiny)
        axes[0, column].imshow(
            log_magnitude_image(array, reference_peak=own_peak, floor_db=floor_db),
            cmap="gray", vmin=0, vmax=1,
        )
        axes[1, column].imshow(
            log_magnitude_image(array, reference_peak=target_peak, floor_db=floor_db),
            cmap="gray", vmin=0, vmax=1,
        )
        axes[0, column].set_title(title)
        axes[0, column].axis("off")
        axes[1, column].axis("off")
    axes[0, 0].set_ylabel("Independent peak")
    axes[1, 0].set_ylabel("Shared Image peak")
    figure.suptitle(
        f"E012 frozen checkpoint: {filename}\n"
        f"RAW phase={raw_metrics['weighted_phase_alignment']:.4f} "
        f"gap_closed={raw_metrics['rmse_oracle_gap_fraction_closed']:.4f}; "
        f"EMA phase={ema_metrics['weighted_phase_alignment']:.4f} "
        f"gap_closed={ema_metrics['rmse_oracle_gap_fraction_closed']:.4f}",
        fontsize=10,
    )
    figure.savefig(path, dpi=dpi)
    plt.close(figure)


def _resolved_config(config: dict[str, Any], args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(config)
    result["config_file"] = str(args.config.resolve())
    result["checkpoint_path"] = str(args.checkpoint.resolve())
    result["echo_dir"] = str(args.echo_dir.resolve())
    result["image_dir"] = str(args.image_dir.resolve())
    result["checkpoint_identity"] = {
        "experiment": checkpoint["resolved_config"]["experiment"],
        "step": checkpoint["step"],
        "selection_manifest_fingerprint": checkpoint["resolved_config"]["selection_manifest"].get("fingerprint"),
    }
    result["inference_contract"] = "Echo spectrum is the only model input; Image is unavailable to the checkpoint forward pass"
    result["training"] = "forbidden: checkpoint is frozen and no optimizer is created"
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    validate_config(config)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    if not args.echo_dir.is_dir() or not args.image_dir.is_dir():
        raise FileNotFoundError("Echo and Image directories must both exist")
    checkpoint = load_checkpoint(args.checkpoint, config)
    data = config["data"]
    selection = config["selection"]
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**selection["validation_region"]),
        CoordinateRegion(**selection["guard_region"]),
        expected_counts={name: int(value) for name, value in selection["expected_split_counts"].items()},
    )
    training_samples = validate_checkpoint_data(
        checkpoint,
        manifest=manifest,
        echo_dir=args.echo_dir,
        image_dir=args.image_dir,
        patch_shape=tuple(int(value) for value in data["expected_shape"]),
        require_dataset_manifest_fingerprint=bool(
            config["checkpoint"]["require_dataset_manifest_fingerprint"]
        ),
    )
    records = manifest.records_for(SplitName.VALIDATION)
    dataset = PhasePatchDataset(
        records,
        expected_shape=tuple(int(value) for value in data["expected_shape"]),
        data_config=data,
        optimization=checkpoint["resolved_config"]["optimization"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    device = resolve_device(args.device)
    precision = resolve_precision(device)
    model_config = checkpoint["resolved_config"]["model"]
    raw_model = SwinIR(**model_config).to(device)
    ema_model = SwinIR(**model_config).to(device)
    raw_model.load_state_dict(checkpoint["model"], strict=True)
    ema_model.load_state_dict(checkpoint["ema_model"], strict=True)
    raw_model.eval()
    ema_model.eval()
    baselines = evaluate_baselines(loader, data_config=data, evaluation=config["evaluation"])
    raw_summary = evaluate_validation(
        raw_model, loader, baselines, device=device, precision=precision,
        data_config=data, optimization=checkpoint["resolved_config"]["optimization"],
        evaluation=config["evaluation"],
    )
    ema_summary = evaluate_validation(
        ema_model, loader, baselines, device=device, precision=precision,
        data_config=data, optimization=checkpoint["resolved_config"]["optimization"],
        evaluation=config["evaluation"],
    )
    raw_comparison = compare_to_baselines(raw_summary, baselines, config["evaluation"]["success_criteria"])
    ema_comparison = compare_to_baselines(ema_summary, baselines, config["evaluation"]["success_criteria"])

    args.output_dir.mkdir(parents=True)
    figures = args.output_dir / "representative_samples"
    figures.mkdir()
    coordinates = {record.echo_path.name: (record.row, record.col) for record in records}
    names = select_representatives(
        raw_summary["per_sample"], coordinates, int(config["output"]["representative_sample_count"])
    )
    record_index = {record.echo_path.name: index for index, record in enumerate(records)}
    figure_entries = []
    optimization = checkpoint["resolved_config"]["optimization"]
    for index, filename in enumerate(names):
        sample = dataset[record_index[filename]]
        inputs = sample["input_spectrum"].unsqueeze(0)
        target = sample["target_phasor"].unsqueeze(0)
        weights = sample["phase_weights"].unsqueeze(0)
        image = sample["target_image"].unsqueeze(0)
        raw_correction = predict_correction(raw_model, inputs, device=device, precision=precision, phasor_epsilon=float(optimization["phasor_epsilon"]))
        ema_correction = predict_correction(ema_model, inputs, device=device, precision=precision, phasor_epsilon=float(optimization["phasor_epsilon"]))
        raw_prediction, _ = evaluate_correction(raw_correction, inputs, target, weights, image, fft_norm=str(data["fft_norm"]), floor_db=float(config["evaluation"]["log_magnitude_floor_db"]), high_frequency_radius_fraction=float(config["evaluation"]["high_frequency_radius_fraction"]))
        ema_prediction, _ = evaluate_correction(ema_correction, inputs, target, weights, image, fft_norm=str(data["fft_norm"]), floor_db=float(config["evaluation"]["log_magnitude_floor_db"]), high_frequency_radius_fraction=float(config["evaluation"]["high_frequency_radius_fraction"]))
        oracle_prediction, _ = evaluate_correction(target, inputs, target, weights, image, fft_norm=str(data["fft_norm"]), floor_db=float(config["evaluation"]["log_magnitude_floor_db"]), high_frequency_radius_fraction=float(config["evaluation"]["high_frequency_radius_fraction"]))
        echo_prediction = apply_phase_correction(inputs, _identity_like(target), fft_norm=str(data["fft_norm"]))
        figure_name = f"{index:03d}_{Path(filename).stem}.png"
        export_figure(
            figures / figure_name, filename=filename, echo=echo_prediction,
            raw=raw_prediction, ema=ema_prediction, oracle=oracle_prediction, image=image,
            raw_metrics=raw_summary["per_sample"][filename],
            ema_metrics=ema_summary["per_sample"][filename],
            floor_db=float(config["evaluation"]["log_magnitude_floor_db"]),
            dpi=int(config["output"]["figure_dpi"]),
        )
        figure_entries.append({"filename": filename, "row": sample["row"], "col": sample["col"], "figure": str(Path("representative_samples") / figure_name)})

    resolved = _resolved_config(config, args, checkpoint)
    write_json(args.output_dir / "resolved_config.json", resolved)
    manifest.write_json(args.output_dir / "split_manifest.json")
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": config["experiment"],
        "status": "completed",
        "checkpoint": {"path": str(args.checkpoint.resolve()), "experiment": checkpoint["resolved_config"]["experiment"], "step": checkpoint["step"]},
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "validated_checkpoint_training_samples": training_samples,
        "evaluation_split": SplitName.VALIDATION.value,
        "evaluation_sample_count": len(dataset),
        "inference_contract": resolved["inference_contract"],
        "training": resolved["training"],
        "authority": config["evaluation"]["authority"],
        "raw_authoritative": {"summary": raw_summary, "comparison": raw_comparison},
        "ema_auxiliary": {"summary": ema_summary, "comparison": ema_comparison},
        "baselines": {"echo_identity": baselines.echo_identity, "unrestricted_phase_oracle": baselines.unrestricted_phase_oracle},
        "representative_samples": figure_entries,
    }
    write_json(args.output_dir / "report.json", report)
    print(f"status=completed report={(args.output_dir / 'report.json').resolve()}", flush=True)
    print(f"raw_pass={raw_comparison['passed']} raw_phase_mean={raw_comparison['mean_phase_alignment']:.4f}", flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
