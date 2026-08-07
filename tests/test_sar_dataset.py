from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.io import savemat

from swinir.sar_dataset import (
    CoordinateRegion,
    DatasetIntegrityError,
    ResumableEpochSampler,
    SplitName,
    classify_coordinate,
    discover_pairs,
    load_complex_patch,
    normalize_complex_pair,
    parse_patch_coordinate,
)


VALIDATION = CoordinateRegion(3301, 6201, 8200, 13500)
GUARD = CoordinateRegion(2801, 6701, 7700, 14000)


def test_coordinate_region_and_parser_have_inclusive_boundaries() -> None:
    assert VALIDATION.contains(3301, 8200)
    assert VALIDATION.contains(6201, 13500)
    assert not VALIDATION.contains(3300, 8200)
    assert parse_patch_coordinate(Path("patch_row_3301_col_8200.mat")) == (
        3301,
        8200,
    )
    assert parse_patch_coordinate(Path("patch_row_17200_col_4000_2.mat")) == (
        17200,
        4000,
    )


def test_discover_pairs_is_coordinate_strict(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    for row, col in ((1, 2300), (101, 2400)):
        name = f"patch_row_{row}_col_{col}.mat"
        (echo_dir / name).touch()
        (image_dir / name).touch()

    pairs = discover_pairs(echo_dir, image_dir)

    assert [(pair.row, pair.col) for pair in pairs] == [(1, 2300), (101, 2400)]


def test_discover_pairs_rejects_unmatched_coordinates(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    (echo_dir / "patch_row_1_col_2300.mat").touch()
    (image_dir / "patch_row_1_col_2400.mat").touch()

    with pytest.raises(DatasetIntegrityError, match="coordinates do not match"):
        discover_pairs(echo_dir, image_dir)


def test_load_complex_patch_is_strict(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.mat"
    real_path = tmp_path / "real.mat"
    expected = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]], dtype=np.complex128
    )
    savemat(valid_path, {"patch": expected})
    savemat(real_path, {"patch": expected.real})

    np.testing.assert_array_equal(
        load_complex_patch(valid_path, expected_shape=(2, 2)), expected
    )
    with pytest.raises(DatasetIntegrityError, match="is not complex"):
        load_complex_patch(real_path, expected_shape=(2, 2))


def test_resumable_sampler_reconstructs_epoch_order() -> None:
    data = list(range(8))
    sampler = ResumableEpochSampler(data, seed=42)
    full_order = list(sampler)

    sampler.set_position(epoch=0, start_index=3)
    assert list(sampler) == full_order[3:]

    state = sampler.state_dict()
    restored = ResumableEpochSampler(data, seed=42)
    restored.load_state_dict(state)
    assert list(restored) == full_order[3:]


def test_learner_task_classifies_validation_guard_and_train() -> None:
    assert classify_coordinate(3301, 8200, VALIDATION, GUARD) is SplitName.VALIDATION
    assert classify_coordinate(2801, 7700, VALIDATION, GUARD) is SplitName.GUARD
    assert classify_coordinate(2701, 8200, VALIDATION, GUARD) is SplitName.TRAIN


def test_learner_task_uses_echo_rms_for_both_complex_tensors() -> None:
    echo = np.array([[3 + 4j, 0 + 0j], [0 + 0j, 0 + 0j]], dtype=np.complex128)
    image = echo * (2 - 1j)

    input_tensor, target_tensor, scale = normalize_complex_pair(echo, image)

    expected_scale = float(np.sqrt(np.mean(np.abs(echo) ** 2) + 1e-12))
    assert scale == pytest.approx(expected_scale)
    assert input_tensor.dtype is torch.float32
    assert target_tensor.dtype is torch.float32
    assert input_tensor.shape == (2, 2, 2)
    assert target_tensor.shape == (2, 2, 2)
    input_complex = input_tensor[0].numpy() + 1j * input_tensor[1].numpy()
    target_complex = target_tensor[0].numpy() + 1j * target_tensor[1].numpy()
    np.testing.assert_allclose(input_complex, echo / scale, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(target_complex, image / scale, rtol=1e-6, atol=1e-7)
