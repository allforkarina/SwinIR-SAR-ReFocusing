"""E015: controlled 64-patch phase-supervision target ablation.

All arms use the same deterministic patch set, random seed, model, optimizer
schedule, and 19,200 optimizer updates.  Only the auxiliary loss target and,
for the phase-only arm, its two zero-valued auxiliary weights may differ.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.overfit_phase_correction_patch_set import run as run_patch_set
from scripts.overfit_single_patch import load_base_config


EXPECTED_EXPERIMENT = "E015-D001-phase-supervision-target-ablation-64"
EXPECTED_SAMPLE_COUNT = 64
EXPECTED_UPDATES_PER_SAMPLE = 300
EXPECTED_STEPS = EXPECTED_SAMPLE_COUNT * EXPECTED_UPDATES_PER_SAMPLE
EXPECTED_EVAL_EVERY = 3200
EXPECTED_SAVE_EVERY = 3200


def validate_profile(config: dict[str, Any], arm: str) -> dict[str, Any]:
    if config.get("experiment") != EXPECTED_EXPERIMENT:
        raise ValueError(f"unexpected E015 experiment: {config.get('experiment')!r}")
    selection = config.get("selection", {})
    runtime = config.get("runtime", {})
    if int(selection.get("sample_count", 0)) != EXPECTED_SAMPLE_COUNT:
        raise ValueError("E015 requires exactly 64 selected patches")
    if selection.get("sample_weighting") != "uniform":
        raise ValueError("E015 requires equal patch-level sample weighting")
    if int(runtime.get("steps", 0)) != EXPECTED_STEPS:
        raise ValueError("E015 requires exactly 19,200 optimizer steps per arm")
    if int(runtime.get("updates_per_sample", 0)) != EXPECTED_UPDATES_PER_SAMPLE:
        raise ValueError("E015 requires 300 average updates per selected patch")
    if int(runtime.get("eval_every", 0)) != EXPECTED_EVAL_EVERY:
        raise ValueError("E015 evaluation interval must be 3,200 steps")
    if int(runtime.get("save_every", 0)) != EXPECTED_SAVE_EVERY:
        raise ValueError("E015 save interval must be 3,200 steps")
    arms = config.get("ablation_arms")
    if not isinstance(arms, dict) or arm not in arms:
        raise ValueError(f"E015 configuration does not define arm {arm}")
    profile = arms[arm]
    expected = {
        "A": ("image", 0.25, 0.25),
        "B": ("phase_oracle", 0.25, 0.25),
        "C": ("phase_oracle", 0.0, 0.0),
    }[arm]
    actual = (
        profile.get("auxiliary_reconstruction_target"),
        float(profile.get("complex_reconstruction_weight", math.nan)),
        float(profile.get("log_magnitude_weight", math.nan)),
    )
    if actual != expected:
        raise ValueError(f"E015 arm {arm} contract differs: {actual!r} != {expected!r}")
    if not isinstance(profile.get("experiment"), str):
        raise ValueError(f"E015 arm {arm} must define an experiment name")
    return profile


def make_patch_set_args(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    selection = config["selection"]
    optimization = config["optimization"]
    runtime = config["runtime"]
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
        required_consecutive_successes=int(runtime["required_consecutive_successes"]),
        seed=int(runtime["seed"]),
        learning_rate=float(optimization["learning_rate"]),
        ema_decay=float(optimization["ema_decay"]),
        success_phase_alignment=float(criteria["weighted_phase_alignment_min"]),
        success_coherence_fraction=float(criteria["coherence_fraction_of_oracle_min"]),
        success_ssim_gain_fraction=float(criteria["ssim_gain_fraction_of_oracle_min"]),
        success_edge_gain_fraction=float(criteria["edge_gain_fraction_of_oracle_min"]),
        success_rmse_excess=float(criteria["rmse_excess_over_oracle_max"]),
        success_hf_ratio_min=float(criteria["high_frequency_energy_ratio_min"]),
        success_hf_ratio_max=float(criteria["high_frequency_energy_ratio_max"]),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = load_base_config(args.config)
    profile = validate_profile(config, args.arm)
    patch_args = make_patch_set_args(args, config)
    return run_patch_set(
        patch_args,
        experiment=str(profile["experiment"]),
        experiment_label=f"E015-{args.arm}",
        stop_on_success=False,
        auxiliary_target=str(profile["auxiliary_reconstruction_target"]),
        optimization_overrides={
            "complex_reconstruction_weight": float(
                profile["complex_reconstruction_weight"]
            ),
            "log_magnitude_weight": float(profile["log_magnitude_weight"]),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("A", "B", "C"), required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_phase_supervision_ablation_64.yaml"),
    )
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
