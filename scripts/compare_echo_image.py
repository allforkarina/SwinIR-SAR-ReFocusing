"""Compare one paired complex Echo/Image MAT patch in the magnitude domain."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPORT_SCHEMA_VERSION = 1


def load_complex_mat(path: Path, variable_name: str = "patch") -> np.ndarray:
    """Load one finite, two-dimensional complex array from a MATLAB v5 file."""

    try:
        variables = loadmat(path, variable_names=[variable_name])
    except Exception as error:
        raise ValueError(f"failed to read MAT file {path}: {error}") from error
    if variable_name not in variables:
        raise ValueError(f"MAT file {path} is missing variable {variable_name!r}")

    values = np.asarray(variables[variable_name])
    if values.ndim != 2 or not values.size:
        raise ValueError(f"MAT variable must be a non-empty 2D array, got {values.shape}")
    if not np.iscomplexobj(values):
        raise ValueError(f"MAT variable {variable_name!r} is not complex: {values.dtype}")
    finite = np.isfinite(values.real) & np.isfinite(values.imag)
    if not bool(finite.all()):
        raise ValueError(f"MAT variable {variable_name!r} contains non-finite values")
    return values


def log_magnitude(
    values: np.ndarray,
    *,
    floor_db: float,
    reference_peak: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return clipped dB and [0, 1] log magnitude using the requested peak."""

    if floor_db >= 0:
        raise ValueError("floor_db must be negative")
    magnitude = np.abs(values).astype(np.float64, copy=False)
    own_peak = float(magnitude.max())
    peak = own_peak if reference_peak is None else float(reference_peak)
    if peak <= 0 or not np.isfinite(peak):
        raise ValueError("normalization peak must be finite and positive")

    relative_floor = 10.0 ** (floor_db / 20.0)
    decibels = 20.0 * np.log10(np.maximum(magnitude / peak, relative_floor))
    decibels = np.clip(decibels, floor_db, 0.0)
    normalized = (decibels - floor_db) / -floor_db
    return decibels, normalized, own_peak


def magnitude_statistics(values: np.ndarray) -> dict[str, float]:
    magnitude = np.abs(values).astype(np.float64, copy=False)
    return {
        "rms": float(np.sqrt(np.mean(magnitude**2))),
        "peak": float(magnitude.max()),
        "mean": float(magnitude.mean()),
        "median": float(np.median(magnitude)),
        "p99": float(np.percentile(magnitude, 99.0)),
    }


def pearson_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_centered = first.ravel() - float(first.mean())
    second_centered = second.ravel() - float(second.mean())
    denominator = float(np.linalg.norm(first_centered) * np.linalg.norm(second_centered))
    if denominator == 0:
        return 0.0
    return float(np.dot(first_centered, second_centered) / denominator)


def compare_arrays(
    echo: np.ndarray, image: np.ndarray, *, floor_db: float
) -> dict[str, Any]:
    if echo.shape != image.shape:
        raise ValueError(f"shape mismatch: echo={echo.shape}, image={image.shape}")

    echo_stats = magnitude_statistics(echo)
    image_stats = magnitude_statistics(image)
    if echo_stats["peak"] <= 0 or image_stats["peak"] <= 0:
        raise ValueError("echo and image must both contain non-zero magnitude")

    echo_independent_db, echo_independent, _ = log_magnitude(
        echo, floor_db=floor_db
    )
    image_independent_db, image_independent, _ = log_magnitude(
        image, floor_db=floor_db
    )
    shared_peak = max(echo_stats["peak"], image_stats["peak"])
    echo_shared_db, echo_shared, _ = log_magnitude(
        echo, floor_db=floor_db, reference_peak=shared_peak
    )
    image_shared_db, image_shared, _ = log_magnitude(
        image, floor_db=floor_db, reference_peak=shared_peak
    )

    return {
        "echo_statistics": echo_stats,
        "image_statistics": image_stats,
        "comparison": {
            "image_to_echo_rms_ratio": image_stats["rms"] / echo_stats["rms"],
            "image_to_echo_peak_ratio": image_stats["peak"] / echo_stats["peak"],
            "independent_log_magnitude_correlation": pearson_correlation(
                echo_independent, image_independent
            ),
            "independent_log_magnitude_mean_absolute_difference": float(
                np.mean(np.abs(image_independent - echo_independent))
            ),
        },
        "arrays": {
            "echo_independent_db": echo_independent_db,
            "image_independent_db": image_independent_db,
            "independent_absolute_difference": np.abs(
                image_independent - echo_independent
            ),
            "echo_shared_db": echo_shared_db,
            "image_shared_db": image_shared_db,
            "shared_signed_difference": image_shared - echo_shared,
        },
        "shared_peak": shared_peak,
    }


def export_figure(
    result: dict[str, Any], path: Path, *, floor_db: float, title: str
) -> None:
    arrays = result["arrays"]
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)

    independent_echo = axes[0, 0].imshow(
        arrays["echo_independent_db"], cmap="gray", vmin=floor_db, vmax=0.0
    )
    axes[0, 1].imshow(
        arrays["image_independent_db"], cmap="gray", vmin=floor_db, vmax=0.0
    )
    independent_difference = axes[0, 2].imshow(
        arrays["independent_absolute_difference"], cmap="magma", vmin=0.0, vmax=1.0
    )
    axes[0, 0].set_title("Echo: independently normalized")
    axes[0, 1].set_title("Image: independently normalized")
    axes[0, 2].set_title("Absolute structure difference")

    shared_echo = axes[1, 0].imshow(
        arrays["echo_shared_db"], cmap="gray", vmin=floor_db, vmax=0.0
    )
    axes[1, 1].imshow(
        arrays["image_shared_db"], cmap="gray", vmin=floor_db, vmax=0.0
    )
    shared_difference = axes[1, 2].imshow(
        arrays["shared_signed_difference"], cmap="coolwarm", vmin=-1.0, vmax=1.0
    )
    axes[1, 0].set_title("Echo: shared peak scale")
    axes[1, 1].set_title("Image: shared peak scale")
    axes[1, 2].set_title("Image - Echo on shared log scale")

    for axis in axes.ravel():
        axis.axis("off")
    figure.colorbar(
        independent_echo, ax=axes[0, :2], shrink=0.82, label="Relative magnitude (dB)"
    )
    figure.colorbar(
        independent_difference,
        ax=axes[0, 2],
        shrink=0.82,
        label="Normalized absolute difference",
    )
    figure.colorbar(
        shared_echo, ax=axes[1, :2], shrink=0.82, label="Shared-peak magnitude (dB)"
    )
    figure.colorbar(
        shared_difference,
        ax=axes[1, 2],
        shrink=0.82,
        label="Signed normalized difference",
    )
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new directory"
        )
    echo = load_complex_mat(args.echo_file, args.variable_name)
    image = load_complex_mat(args.image_file, args.variable_name)
    result = compare_arrays(echo, image, floor_db=args.db_floor)

    args.output_dir.mkdir(parents=True)
    figure_path = args.output_dir / "comparison.png"
    report_path = args.output_dir / "report.json"
    export_figure(
        result,
        figure_path,
        floor_db=args.db_floor,
        title=f"Echo vs Image magnitude: {args.echo_file.name}",
    )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "echo_file": str(args.echo_file.resolve()),
        "image_file": str(args.image_file.resolve()),
        "variable_name": args.variable_name,
        "shape": list(echo.shape),
        "db_floor": float(args.db_floor),
        "normalization": {
            "independent": "each magnitude uses its own peak",
            "shared": "both magnitudes use the larger of the two peaks",
        },
        "echo_statistics": result["echo_statistics"],
        "image_statistics": result["image_statistics"],
        "comparison": result["comparison"],
        "figure": str(figure_path.resolve()),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    comparison = report["comparison"]
    print(
        f"echo: rms={report['echo_statistics']['rms']:.6g} "
        f"peak={report['echo_statistics']['peak']:.6g}"
    )
    print(
        f"image: rms={report['image_statistics']['rms']:.6g} "
        f"peak={report['image_statistics']['peak']:.6g}"
    )
    print(
        f"image/echo: rms_ratio={comparison['image_to_echo_rms_ratio']:.6g} "
        f"peak_ratio={comparison['image_to_echo_peak_ratio']:.6g}"
    )
    print(
        "independently normalized log magnitudes: "
        f"correlation={comparison['independent_log_magnitude_correlation']:.6g} "
        "mean_abs_difference="
        f"{comparison['independent_log_magnitude_mean_absolute_difference']:.6g}"
    )
    print(f"figure={figure_path.resolve()}")
    print(f"report={report_path.resolve()}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one paired complex Echo/Image MAT patch with independent and shared "
            "magnitude normalization."
        )
    )
    parser.add_argument("--echo-file", type=Path, required=True)
    parser.add_argument("--image-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variable-name", default="patch")
    parser.add_argument("--db-floor", type=float, default=-60.0)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
