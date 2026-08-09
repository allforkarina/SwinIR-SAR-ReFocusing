"""Export side-by-side complex SAR refocusing visualizations.

Each selected paired MAT patch produces a 2-by-3 PNG: logarithmic magnitude
and phase for the Echo input, the EMA checkpoint prediction, and the Image
label.  The generated manifest records the deterministic random selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from test import _checkpoint_config, expected_shape_from_config, load_ema_model
from swinir.sar_dataset import DiscoveredPair, load_complex_patch, normalize_complex_pair
from swinir.training import resolve_device, resolve_precision


MANIFEST_SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT = Path("runs/sar_baseline_v1/checkpoints/best.pt")
DEFAULT_OUTPUT_DIR = Path("outputs/refocusing_visualizations")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Echo, EMA prediction, and Image-label comparison figures."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cuda:0, or cpu")
    return parser.parse_args()


def discover_pairs(echo_dir: Path, image_dir: Path) -> tuple[DiscoveredPair, ...]:
    """Import lazily so this script remains easy to test as a standalone module."""

    from swinir.sar_dataset import discover_pairs as discover

    return discover(echo_dir, image_dir)


def select_pairs(
    pairs: Sequence[DiscoveredPair], *, count: int, seed: int
) -> tuple[DiscoveredPair, ...]:
    """Return a deterministic, coordinate-ordered random sample without replacement."""

    if count <= 0:
        raise ValueError("count must be positive")
    if count > len(pairs):
        raise ValueError(f"count={count} exceeds available paired patches ({len(pairs)})")
    indices = np.random.default_rng(seed).choice(len(pairs), size=count, replace=False)
    return tuple(pairs[index] for index in sorted(indices.tolist()))


def logarithmic_magnitude(values: np.ndarray) -> np.ndarray:
    """Return finite magnitude values in dB, preserving the raw data scale."""

    magnitude = np.abs(values)
    floor = max(float(magnitude.max()) * 1e-12, np.finfo(np.float64).tiny)
    return 20.0 * np.log10(np.maximum(magnitude, floor))


def export_figure(
    echo: np.ndarray,
    prediction: np.ndarray,
    image: np.ndarray,
    path: Path,
    *,
    row: int,
    col: int,
) -> None:
    """Save the agreed 2-by-3 magnitude/phase comparison for one patch."""

    values = (echo, prediction, image)
    magnitude_values = tuple(logarithmic_magnitude(value) for value in values)
    magnitude_min = min(float(value.min()) for value in magnitude_values)
    magnitude_max = max(float(value.max()) for value in magnitude_values)
    if magnitude_min == magnitude_max:
        magnitude_max = magnitude_min + 1.0

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    titles = ("Echo input", "EMA prediction", "Image label")
    magnitude_images = []
    phase_images = []
    for index, (title, magnitude, value) in enumerate(
        zip(titles, magnitude_values, values, strict=True)
    ):
        magnitude_images.append(
            axes[0, index].imshow(
                magnitude, cmap="magma", vmin=magnitude_min, vmax=magnitude_max
            )
        )
        phase_images.append(
            axes[1, index].imshow(
                np.angle(value), cmap="twilight", vmin=-np.pi, vmax=np.pi
            )
        )
        axes[0, index].set_title(title)
        axes[0, index].axis("off")
        axes[1, index].axis("off")

    axes[0, 0].set_ylabel("Log magnitude")
    axes[1, 0].set_ylabel("Phase")
    figure.colorbar(magnitude_images[0], ax=axes[0, :], label="Magnitude (dB)")
    figure.colorbar(phase_images[0], ax=axes[1, :], label="Phase (radians)")
    figure.suptitle(f"SAR refocusing comparison: row={row}, col={col}")
    figure.savefig(path, dpi=160)
    plt.close(figure)


@torch.no_grad()
def predict_patch(
    model: torch.nn.Module,
    echo: np.ndarray,
    image: np.ndarray,
    *,
    device: torch.device,
    precision: Any,
    rms_epsilon: float,
) -> np.ndarray:
    """Run one normalized complex patch and return a denormalized prediction."""

    input_tensor, _, scale = normalize_complex_pair(echo, image, rms_epsilon)
    with precision.autocast():
        prediction = model(input_tensor.unsqueeze(0).to(device))
    output = prediction.detach().float().cpu().numpy()[0]
    return (output[0] + 1j * output[1]) * scale


def write_manifest(path: Path, content: dict[str, Any]) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {args.output_dir}; choose a new --output-dir"
        )

    device = resolve_device(args.device)
    checkpoint, model = load_ema_model(args.checkpoint, device)
    config = _checkpoint_config(checkpoint)
    expected_shape = expected_shape_from_config(config)
    rms_epsilon = float(config["data"].get("rms_epsilon", 1e-12))
    pairs = discover_pairs(args.echo_dir, args.image_dir)
    selected_pairs = select_pairs(pairs, count=args.count, seed=args.seed)
    args.output_dir.mkdir(parents=True)
    precision = resolve_precision(device)

    entries = []
    for rank, pair in enumerate(selected_pairs, start=1):
        echo = load_complex_patch(pair.echo_path, expected_shape)
        image = load_complex_patch(pair.image_path, expected_shape)
        prediction = predict_patch(
            model,
            echo,
            image,
            device=device,
            precision=precision,
            rms_epsilon=rms_epsilon,
        )
        filename = f"{rank:03d}_row_{pair.row}_col_{pair.col}.png"
        export_figure(echo, prediction, image, args.output_dir / filename, row=pair.row, col=pair.col)
        entries.append(
            {
                "rank": rank,
                "row": pair.row,
                "col": pair.col,
                "echo_file": pair.echo_path.name,
                "image_file": pair.image_path.name,
                "figure": filename,
            }
        )

    write_manifest(
        args.output_dir / "manifest.json",
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "weights": "ema_model",
            "device": str(device),
            "echo_dir": str(args.echo_dir.resolve()),
            "image_dir": str(args.image_dir.resolve()),
            "count": args.count,
            "seed": args.seed,
            "normalization": "per_patch_echo_rms",
            "samples": entries,
        },
    )
    print(f"Exported {len(entries)} figures to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
