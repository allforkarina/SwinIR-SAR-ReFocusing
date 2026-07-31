from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from scripts.analyze_sar_dataset import (
    canonical_pair_key,
    discover_pairs,
    evenly_spaced_indices,
    read_complex,
)


def test_canonical_pair_key_handles_server_filename_roles() -> None:
    echo = Path("x_+00000_y_+04250_echo_0.mat")
    typo_image = Path("x_+00000_y_+04250_iamge_0.mat")
    image = Path("x_+00000_y_+04250_image_0.mat")

    assert canonical_pair_key(echo) == canonical_pair_key(typo_image)
    assert canonical_pair_key(echo) == canonical_pair_key(image)


def test_discover_pairs_does_not_pair_by_position(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    (echo_dir / "a_echo_0.mat").touch()
    (echo_dir / "unmatched_echo_0.mat").touch()
    (image_dir / "a_iamge_0.mat").touch()
    (image_dir / "other_iamge_0.mat").touch()

    pairs, summary = discover_pairs(echo_dir, image_dir)

    assert [pair.key for pair in pairs] == ["a__sample_0"]
    assert summary["echo_only_count"] == 1
    assert summary["image_only_count"] == 1


def test_evenly_spaced_indices_are_deterministic() -> None:
    assert evenly_spaced_indices(10, 3) == [0, 4, 9]
    assert evenly_spaced_indices(3, 0) == [0, 1, 2]


def test_read_complex_supports_matlab_compound_dtype(tmp_path: Path) -> None:
    path = tmp_path / "sample.mat"
    dtype = np.dtype([("real", "<f4"), ("imag", "<f4")])
    raw = np.zeros((2, 2), dtype=dtype)
    raw["real"] = [[1, 2], [3, 4]]
    raw["imag"] = [[-1, -2], [-3, -4]]
    with h5py.File(path, "w") as file:
        file.create_dataset("coarse_patch", data=raw)

    with h5py.File(path, "r") as file:
        result = read_complex(file["coarse_patch"])

    expected = np.array([[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]], dtype=np.complex64)
    np.testing.assert_array_equal(result, expected)
