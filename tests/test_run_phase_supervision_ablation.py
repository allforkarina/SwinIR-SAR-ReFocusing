from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from scripts.run_phase_supervision_ablation import (
    EXPECTED_STEPS,
    validate_profile,
)


def load_config() -> dict:
    return yaml.safe_load(
        Path("configs/train_phase_supervision_ablation_64.yaml").read_text(
            encoding="utf-8"
        )
    )


@pytest.mark.parametrize(
    ("arm", "target", "complex_weight", "log_weight"),
    (
        ("A", "image", 0.25, 0.25),
        ("B", "phase_oracle", 0.25, 0.25),
        ("C", "phase_oracle", 0.0, 0.0),
    ),
)
def test_e015_profiles_are_controlled(
    arm: str, target: str, complex_weight: float, log_weight: float
) -> None:
    config = load_config()
    profile = validate_profile(config, arm)

    assert config["runtime"]["steps"] == EXPECTED_STEPS
    assert config["runtime"]["steps"] // config["selection"]["sample_count"] == 300
    assert profile["auxiliary_reconstruction_target"] == target
    assert profile["complex_reconstruction_weight"] == complex_weight
    assert profile["log_magnitude_weight"] == log_weight


def test_e015_rejects_nonuniform_patch_weighting() -> None:
    config = deepcopy(load_config())
    config["selection"]["sample_weighting"] = "brightness_weighted"

    with pytest.raises(ValueError, match="equal patch-level"):
        validate_profile(config, "A")
