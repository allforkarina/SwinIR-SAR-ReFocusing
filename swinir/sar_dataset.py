"""Strict paired MATLAB dataset support for SAR refocusing training.

The server dataset stores matching complex ``patch`` matrices in separate
``echo`` and ``image`` directories. This module keeps discovery, spatial split
metadata, numeric validation, and tensor conversion explicit so a training run
can prove exactly which samples it consumed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset, Sampler


COORDINATE_PATTERN = re.compile(
    r"^patch_row_([+-]?\d+)_col_([+-]?\d+)(?:_\d+)?\.mat$", re.IGNORECASE
)
MANIFEST_SCHEMA_VERSION = 1


class DatasetIntegrityError(RuntimeError):
    """Raised when on-disk data violates the configured training contract."""


class SplitName(str, Enum):
    TRAIN = "train"             # 训练集 最外围的数据
    GUARD = "guard"             # 由于重叠 加入保护带隔离验证集
    VALIDATION = "validation"   # 验证集


@dataclass(frozen=True)
class CoordinateRegion:
    """
        data region in certain square area.
        self.row_min <= row <= self.row_max
        self.col_min <= col <= self.col_max
    """
    row_min: int
    row_max: int
    col_min: int
    col_max: int

    def __post_init__(self) -> None:
        if self.row_min > self.row_max:
            raise ValueError("row_min must not exceed row_max")
        if self.col_min > self.col_max:
            raise ValueError("col_min must not exceed col_max")

    def contains(self, row: int, col: int) -> bool:
        """Check if the given coordinate is within the region."""
        return (
            self.row_min <= row <= self.row_max
            and self.col_min <= col <= self.col_max
        )


@dataclass(frozen=True)
class DiscoveredPair:
    row: int
    col: int
    echo_path: Path
    image_path: Path

    @property
    def key(self) -> str:
        return f"row_{self.row}_col_{self.col}"


@dataclass(frozen=True)
class PairRecord:
    key: str
    row: int
    col: int
    split: SplitName
    echo_path: Path
    image_path: Path


@dataclass(frozen=True)
class DatasetManifest:
    echo_dir: Path                          # echo data path
    image_dir: Path                         # image data path
    validation_region: CoordinateRegion     # 验证集区域
    guard_region: CoordinateRegion          # 保护带区域
    records: tuple[PairRecord, ...]         # all the records of echo-image pair
    fingerprint: str

    @property
    def split_counts(self) -> dict[str, int]:
        # count all the records for train, guard, validation splits.
        counts = Counter(record.split.value for record in self.records)
        # return a dict, train, guard, validation splits' count, in a format: train.value: counts[train.value], etc.
        return {split.value: counts[split.value] for split in SplitName}

    def records_for(self, split: SplitName) -> tuple[PairRecord, ...]:
        return tuple(record for record in self.records if record.split is split)

    def as_dict(self) -> dict[str, Any]:
        """
            Write the manifest info into a dict, which can be serialized to json file.
        Returns:
            dict[str, Any]: info of the dataset, include split, each record's info, etc.
        """
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "echo_dir": str(self.echo_dir.resolve()),
            "image_dir": str(self.image_dir.resolve()),
            "validation_region": asdict(self.validation_region),
            "guard_region": asdict(self.guard_region),
            "split_counts": self.split_counts,
            "fingerprint": self.fingerprint,
            "records": [
                {
                    "key": record.key,
                    "row": record.row,
                    "col": record.col,
                    "split": record.split.value,
                    "echo_file": record.echo_path.name,
                    "image_file": record.image_path.name,
                    "echo_size_bytes": record.echo_path.stat().st_size,
                    "image_size_bytes": record.image_path.stat().st_size,
                }
                for record in self.records
            ],
        }

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.as_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def parse_patch_coordinate(path: Path) -> tuple[int, int]:
    match = COORDINATE_PATTERN.fullmatch(path.name)
    if match is None:
        raise DatasetIntegrityError(
            f"cannot parse row/col from MAT filename: {path.name}"
        )
    # parsing path and return the row and col in path's name.
    return int(match.group(1)), int(match.group(2))


def _index_directory(directory: Path, role: str) -> dict[tuple[int, int], Path]:
    """
        build up a directory for each records coordinate to their path.
    """
    if not directory.is_dir():
        raise DatasetIntegrityError(f"{role} directory does not exist: {directory}")

    # dict: coordinate (row, col) -> path
    grouped: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in sorted(directory.iterdir()):
        # if the file is .mat, then append this coordinate-path tuple into the grouped dict.
        if path.is_file() and path.suffix.lower() == ".mat":
            grouped[parse_patch_coordinate(path)].append(path)

    if not grouped:
        raise DatasetIntegrityError(f"{role} directory contains no MAT files: {directory}")

    # find false cases: if there is multiple file path refer to the same coordinate.
    collisions = {coordinate: paths for coordinate, paths in grouped.items() if len(paths) != 1}
    if collisions:
        coordinate, paths = next(iter(collisions.items()))
        names = [path.name for path in paths]
        raise DatasetIntegrityError(
            f"duplicate {role} coordinate {coordinate}: files={names}"
        )
    return {coordinate: paths[0] for coordinate, paths in grouped.items()}


def discover_pairs(echo_dir: Path, image_dir: Path) -> tuple[DiscoveredPair, ...]:
    # coordinate (row, col) -> path [xxx.mat]
    echo_index = _index_directory(echo_dir, "echo")
    image_index = _index_directory(image_dir, "image")

    # Do minus to find that only echo no image, or only image no echo.
    echo_only = sorted(echo_index.keys() - image_index.keys())
    image_only = sorted(image_index.keys() - echo_index.keys())
    if echo_only or image_only:
        raise DatasetIntegrityError(
            "echo/image coordinates do not match: "
            f"echo_only={echo_only[:10]}, image_only={image_only[:10]}"
        )

    # define empty pair list
    pairs = []
    # get row and col from echo.
    for row, col in sorted(echo_index):
        # get the path of echo from the coordinate.
        echo_path = echo_index[(row, col)]
        image_path = image_index[(row, col)]    # pair

        if echo_path.name != image_path.name:
            raise DatasetIntegrityError(
                f"paired coordinate {(row, col)} has different filenames: "
                f"echo={echo_path.name}, image={image_path.name}"
            )
        pairs.append(
            DiscoveredPair(
                row=row,
                col=col,
                echo_path=echo_path,
                image_path=image_path,
            )
        )
    return tuple(pairs)


def classify_coordinate(
    row: int,
    col: int,
    validation_region: CoordinateRegion,    # well defined validation region
    guard_region: CoordinateRegion,         # well defined guard region, with row_min, row_max, col_min, col_max
) -> SplitName:
    """Classify one patch coordinate into train, guard, or validation.

    Learner task 1: implement the accepted spatial split. Validation is the
    inner rectangle, guard is the surrounding rectangle excluding validation,
    and everything outside the guard rectangle is training data.
    """

    if validation_region.contains(row, col):
        return SplitName.VALIDATION
    elif guard_region.contains(row, col):
        return SplitName.GUARD
    else:
        return SplitName.TRAIN


def _manifest_fingerprint(records: Sequence[PairRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        row = (
            f"{record.row}|{record.col}|{record.split.value}|"
            f"{record.echo_path.name}|{record.echo_path.stat().st_size}|"
            f"{record.image_path.name}|{record.image_path.stat().st_size}\n"
        )
        digest.update(row.encode("utf-8"))
    return digest.hexdigest()


def build_manifest(
    echo_dir: Path,
    image_dir: Path,
    validation_region: CoordinateRegion,
    guard_region: CoordinateRegion,
    expected_counts: dict[str, int] | None = None,
) -> DatasetManifest:

    # guard region must contain the full validation region, outside the validation region.
    if not guard_region.contains(
        validation_region.row_min, validation_region.col_min
    ) or not guard_region.contains(
        validation_region.row_max, validation_region.col_max
    ):
        raise ValueError("guard_region must contain the full validation_region")

    records = tuple(
        PairRecord(
            key=pair.key,
            row=pair.row,
            col=pair.col,
            split=classify_coordinate(
                pair.row, pair.col, validation_region, guard_region
            ),
            echo_path=pair.echo_path,
            image_path=pair.image_path,
        )
        for pair in discover_pairs(echo_dir, image_dir)
    )
    manifest = DatasetManifest(
        echo_dir=echo_dir,
        image_dir=image_dir,
        validation_region=validation_region,
        guard_region=guard_region,
        records=records,
        fingerprint=_manifest_fingerprint(records),
    )
    if expected_counts is not None:
        unknown = set(expected_counts) - {split.value for split in SplitName}
        if unknown:
            raise ValueError(f"unknown expected split names: {sorted(unknown)}")
        actual = manifest.split_counts
        mismatches = {
            name: {"expected": expected, "actual": actual[name]}
            for name, expected in expected_counts.items()
            if actual[name] != expected
        }
        if mismatches:
            raise DatasetIntegrityError(f"split counts do not match: {mismatches}")
    return manifest


def load_complex_patch(
    path: Path,
    expected_shape: tuple[int, int] = (512, 512),
    variable_name: str = "patch",
) -> np.ndarray:
    try:
        variables = loadmat(path, variable_names=[variable_name])
    except Exception as error:
        raise DatasetIntegrityError(f"failed to read MAT file {path}: {error}") from error
    if variable_name not in variables:
        raise DatasetIntegrityError(
            f"MAT file {path} is missing complex variable {variable_name!r}"
        )

    values = np.asarray(variables[variable_name])
    if values.shape != expected_shape:
        raise DatasetIntegrityError(
            f"MAT file {path} has shape {values.shape}, expected {expected_shape}"
        )
    if not np.iscomplexobj(values):
        raise DatasetIntegrityError(
            f"MAT file {path} variable {variable_name!r} is not complex: {values.dtype}"
        )
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not bool(finite.all()):
        count = int(values.size - finite.sum())
        raise DatasetIntegrityError(
            f"MAT file {path} contains {count} non-finite complex values"
        )
    return values


def normalize_complex_pair(
    echo: np.ndarray,
    image: np.ndarray,
    epsilon: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Normalize a complex pair using only the current Echo patch RMS.

    Learner task 2: compute one positive scale from ``echo``, divide both
    arrays by it, and return real/imaginary float32 tensors with shape
    ``[2, H, W]`` plus the Python float scale. Do not derive any statistic from
    ``image``.
    """
    # get the RMS of the echo in a pair.
    # add a epsilon to avoid zero.
    rms = np.sqrt( np.mean( np.abs(echo) ** 2 ) + epsilon )
    # rms = np.sqrt( np.mean( echo.real**2 + echo.imag**2 ) + epsilon )

    normalized_echo = echo / rms
    normalized_image = image / rms

    # convert to torch tensor
    echo_real = normalized_echo.real.astype(np.float32)
    echo_imag = normalized_echo.imag.astype(np.float32)
    image_real = normalized_image.real.astype(np.float32)
    image_imag = normalized_image.imag.astype(np.float32)

    echo_tensor = torch.from_numpy(np.stack([echo_real, echo_imag]))
    image_tensor = torch.from_numpy(np.stack([image_real, image_imag]))

    return echo_tensor, image_tensor, float(rms)


class SARPatchDataset(Dataset[dict[str, Any]]):
    """Strict lazy loader for one non-guard manifest split."""

    def __init__(
        self,
        records: Sequence[PairRecord],
        expected_shape: tuple[int, int] = (512, 512),
        epsilon: float = 1e-12,
    ) -> None:
        self.records = tuple(records)
        self.expected_shape = expected_shape
        self.epsilon = float(epsilon)
        if not self.records:
            raise ValueError("SARPatchDataset requires at least one record")
        if any(record.split is SplitName.GUARD for record in self.records):
            raise ValueError("guard records must not be exposed through a Dataset")
        if not math.isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("epsilon must be a finite positive number")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        try:
            echo = load_complex_patch(record.echo_path, self.expected_shape)
            image = load_complex_patch(record.image_path, self.expected_shape)
            input_tensor, target_tensor, scale = normalize_complex_pair(
                echo, image, self.epsilon
            )
            expected_tensor_shape = (2, *self.expected_shape)
            for name, tensor in (
                ("input", input_tensor),
                ("target", target_tensor),
            ):
                if tensor.shape != expected_tensor_shape:
                    raise DatasetIntegrityError(
                        f"{name} tensor has shape {tuple(tensor.shape)}, "
                        f"expected {expected_tensor_shape}"
                    )
                if tensor.dtype is not torch.float32:
                    raise DatasetIntegrityError(
                        f"{name} tensor has dtype {tensor.dtype}, expected float32"
                    )
                if not bool(torch.isfinite(tensor).all()):
                    raise DatasetIntegrityError(f"{name} tensor contains non-finite values")
            if not math.isfinite(scale) or scale <= 0:
                raise DatasetIntegrityError(
                    f"normalization scale must be finite and positive, got {scale}"
                )
        except Exception as error:
            if isinstance(error, DatasetIntegrityError):
                detail = str(error)
            else:
                detail = repr(error)
            raise DatasetIntegrityError(
                f"dataset index={index}, key={record.key}, row={record.row}, "
                f"col={record.col}, echo={record.echo_path}, "
                f"image={record.image_path}: {detail}"
            ) from error

        return {
            "input": input_tensor,
            "target": target_tensor,
            "scale": torch.tensor(scale, dtype=torch.float32),
            "key": record.key,
            "row": record.row,
            "col": record.col,
            "echo_path": str(record.echo_path),
            "image_path": str(record.image_path),
        }


class ResumableEpochSampler(Sampler[int]):
    """Deterministic epoch permutation that can restart at a processed offset."""

    def __init__(self, data_source: Sequence[Any], seed: int = 42) -> None:
        if len(data_source) <= 0:
            raise ValueError("sampler data source must not be empty")
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_position(self, epoch: int, start_index: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not 0 <= start_index <= len(self.data_source):
            raise ValueError(
                f"start_index must be in [0, {len(self.data_source)}], got {start_index}"
            )
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def state_dict(self) -> dict[str, int]:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "start_index": self.start_index,
        }

    def load_state_dict(self, state: dict[str, int]) -> None:
        if int(state["seed"]) != self.seed:
            raise ValueError(
                f"sampler seed mismatch: checkpoint={state['seed']}, current={self.seed}"
            )
        self.set_position(int(state["epoch"]), int(state["start_index"]))

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(len(self.data_source), generator=generator).tolist()
        yield from permutation[self.start_index :]

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index
