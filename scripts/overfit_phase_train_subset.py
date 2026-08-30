"""E011-B: controlled phase-correction overfit on a fixed training subset."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.overfit_phase_correction_patch_set import run as run_joint_overfit
from scripts.overfit_single_patch import load_base_config
from swinir.sar_dataset import (
    CoordinateRegion,
    DiscoveredPair,
    SplitName,
    build_manifest,
)


EXPERIMENT = "E011-B-D001-controlled-64-train-overfit"
EXPERIMENT_LABEL = "E011-B/64"
CURRICULUM_EXPERIMENTS = {
    "E013-A-D001-curriculum-128-phase-subset": "E013-A/128",
    "E013-B-D001-curriculum-512-phase-subset": "E013-B/512",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overfit supervised phase correction on a fixed, spatially distributed "
            "subset drawn only from the E010 training split."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_phase_train_subset_64.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Load RAW/EMA weights only; optimizer, RNG, and local step start fresh.",
    )
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def _require_positive_int(mapping: dict[str, Any], name: str) -> int:
    value = int(mapping.get(name, 0))
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_profile(config: dict[str, Any]) -> None:
    experiment = config.get("experiment")
    if experiment != EXPERIMENT and experiment not in CURRICULUM_EXPERIMENTS:
        raise ValueError("config.experiment is not a supported phase-subset profile")
    selection = config.get("selection")
    runtime = config.get("runtime")
    evaluation = config.get("evaluation")
    if not isinstance(selection, dict):
        raise ValueError("config.selection must be a mapping")
    if not isinstance(runtime, dict):
        raise ValueError("config.runtime must be a mapping")
    if not isinstance(evaluation, dict):
        raise ValueError("config.evaluation must be a mapping")
    if selection.get("source_split") != SplitName.TRAIN.value:
        raise ValueError("E011-B selection.source_split must be 'train'")
    sample_count = _require_positive_int(selection, "sample_count")
    if sample_count < 2:
        raise ValueError("selection.sample_count must be at least 2")
    if not isinstance(selection.get("anchor_filename"), str):
        raise ValueError("selection.anchor_filename must be a filename")
    for name in ("validation_region", "guard_region", "expected_split_counts"):
        if not isinstance(selection.get(name), dict):
            raise ValueError(f"selection.{name} must be a mapping")
    steps = _require_positive_int(runtime, "steps")
    eval_every = _require_positive_int(runtime, "eval_every")
    save_every = _require_positive_int(runtime, "save_every")
    _require_positive_int(runtime, "required_consecutive_successes")
    if eval_every % sample_count:
        raise ValueError("runtime.eval_every must cover a whole number of subset epochs")
    if save_every % eval_every:
        raise ValueError("runtime.save_every must be divisible by eval_every")
    if steps % sample_count:
        raise ValueError("runtime.steps must cover a whole number of subset epochs")
    criteria = evaluation.get("success_criteria")
    if not isinstance(criteria, dict):
        raise ValueError("evaluation.success_criteria must be a mapping")
    required = (
        "weighted_phase_alignment_min",
        "coherence_fraction_of_oracle_min",
        "ssim_gain_fraction_of_oracle_min",
        "edge_gain_fraction_of_oracle_min",
        "rmse_excess_over_oracle_max",
        "high_frequency_energy_ratio_min",
        "high_frequency_energy_ratio_max",
    )
    for name in required:
        if not math.isfinite(float(criteria[name])):
            raise ValueError(f"evaluation.success_criteria.{name} must be finite")
    if experiment in CURRICULUM_EXPERIMENTS:
        initialization = config.get("initialization")
        if not isinstance(initialization, dict):
            raise ValueError("curriculum config.initialization must be a mapping")
        if initialization.get("mode") != "raw_and_ema_weights_only":
            raise ValueError("curriculum initialization must load RAW/EMA weights only")
        if not isinstance(initialization.get("expected_source_experiment"), str):
            raise ValueError("initialization.expected_source_experiment is required")
        _require_positive_int(initialization, "expected_source_sample_count")
        probes = evaluation.get("probe_sample_indices")
        artifacts = evaluation.get("artifact_sample_indices")
        for name, values in (("probe_sample_indices", probes), ("artifact_sample_indices", artifacts)):
            if not isinstance(values, list) or not values:
                raise ValueError(f"evaluation.{name} must be a non-empty list")
            indices = [int(value) for value in values]
            if len(indices) != len(set(indices)):
                raise ValueError(f"evaluation.{name} must contain unique indices")
            if min(indices) < 0 or max(indices) >= sample_count:
                raise ValueError(f"evaluation.{name} contains an out-of-range index")
        if not set(int(value) for value in artifacts).issubset(
            int(value) for value in probes
        ):
            raise ValueError("artifact probes must be included in evaluation probes")
        if not isinstance(runtime.get("stop_on_success"), bool):
            raise ValueError("runtime.stop_on_success must be boolean")


def _joint_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    selection = config["selection"]
    runtime = config["runtime"]
    optimization = config["optimization"]
    criteria = config["evaluation"]["success_criteria"]
    return argparse.Namespace(
        config=args.config,
        echo_dir=args.echo_dir,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        anchor_filename=str(selection["anchor_filename"]),
        sample_count=int(selection["sample_count"]),
        resume=args.resume,
        device=args.device,
        steps=int(runtime["steps"]),
        eval_every=int(runtime["eval_every"]),
        save_every=int(runtime["save_every"]),
        required_consecutive_successes=int(
            runtime["required_consecutive_successes"]
        ),
        seed=int(runtime["seed"]),
        learning_rate=float(optimization["learning_rate"]),
        ema_decay=float(optimization["ema_decay"]),
        success_phase_alignment=float(criteria["weighted_phase_alignment_min"]),
        success_coherence_fraction=float(
            criteria["coherence_fraction_of_oracle_min"]
        ),
        success_ssim_gain_fraction=float(
            criteria["ssim_gain_fraction_of_oracle_min"]
        ),
        success_edge_gain_fraction=float(
            criteria["edge_gain_fraction_of_oracle_min"]
        ),
        success_rmse_excess=float(criteria["rmse_excess_over_oracle_max"]),
        success_hf_ratio_min=float(criteria["high_frequency_energy_ratio_min"]),
        success_hf_ratio_max=float(criteria["high_frequency_energy_ratio_max"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_base_config(args.config)
    validate_profile(config)
    experiment = str(config["experiment"])
    is_curriculum = experiment in CURRICULUM_EXPERIMENTS
    init_checkpoint = getattr(args, "init_checkpoint", None)
    if is_curriculum:
        if init_checkpoint is None:
            raise ValueError("curriculum training requires --init-checkpoint")
        if not init_checkpoint.is_file():
            raise FileNotFoundError(
                f"initialization checkpoint does not exist: {init_checkpoint}"
            )
    elif init_checkpoint is not None:
        raise ValueError("E011-B does not accept --init-checkpoint")
    selection = config["selection"]
    manifest = build_manifest(
        args.echo_dir,
        args.image_dir,
        CoordinateRegion(**selection["validation_region"]),
        CoordinateRegion(**selection["guard_region"]),
        expected_counts={
            str(name): int(value)
            for name, value in selection["expected_split_counts"].items()
        },
    )
    train_records = manifest.records_for(SplitName.TRAIN)
    candidates = tuple(
        DiscoveredPair(
            row=record.row,
            col=record.col,
            echo_path=record.echo_path,
            image_path=record.image_path,
        )
        for record in train_records
    )
    anchor = str(selection["anchor_filename"])
    if sum(pair.echo_path.name == anchor for pair in candidates) != 1:
        raise ValueError(
            "selection anchor must identify exactly one E010 training record: "
            f"{anchor!r}"
        )
    metadata = {
        "source_split": SplitName.TRAIN.value,
        "candidate_count": len(candidates),
        "dataset_manifest_fingerprint": manifest.fingerprint,
        "spatial_split": {
            "validation_region": selection["validation_region"],
            "guard_region": selection["guard_region"],
            "split_counts": manifest.split_counts,
        },
    }
    return run_joint_overfit(
        _joint_args(args, config),
        candidate_pairs=candidates,
        selection_metadata=metadata,
        experiment=experiment,
        experiment_label=(
            CURRICULUM_EXPERIMENTS[experiment]
            if is_curriculum
            else EXPERIMENT_LABEL
        ),
        initialization_checkpoint=init_checkpoint,
        expected_initialization=(config["initialization"] if is_curriculum else None),
        evaluation_sample_indices=(
            config["evaluation"]["probe_sample_indices"]
            if is_curriculum
            else None
        ),
        artifact_sample_indices=(
            config["evaluation"]["artifact_sample_indices"]
            if is_curriculum
            else None
        ),
        stop_on_success=(
            bool(config["runtime"]["stop_on_success"])
            if is_curriculum
            else True
        ),
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
