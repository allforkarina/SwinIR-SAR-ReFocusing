"""E014: read-only phase-neighborhood and frequency-reliability audit.

The audit compares Oracle unit phase corrections at controlled row/column
distances. Every selected patch pair contributes one equally weighted
observation. Echo brightness is reported only as a stratum and never changes
pair weights. Relative-energy masks and soft weights are diagnostic ablations;
the existing absolute reliability rule remains the named baseline.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_phase_dataset import (
    CoordinatePair,
    coordinate_records,
    load_pair,
    normalized_phase_oracle,
    utc_now,
    validate_source_output_separation,
    write_csv,
    write_json,
)
from scripts.analyze_sar_dataset import discover_pairs, distribution, evenly_spaced_indices
from scripts.diagnose_shared_complex_filter import evaluate_focus_prediction


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SpatialPhasePair:
    axis: str
    distance: int
    first: CoordinatePair
    second: CoordinatePair


@dataclass(frozen=True)
class PhaseData:
    correction: np.ndarray
    cross_energy: np.ndarray
    echo_rms: float
    normalized_echo: np.ndarray
    normalized_image: np.ndarray


def build_pair_candidates(
    records: Sequence[CoordinatePair], axis: str, distance: int
) -> list[SpatialPhasePair]:
    if axis not in {"row", "col"}:
        raise ValueError(f"unsupported axis: {axis!r}")
    if distance <= 0:
        raise ValueError("distance must be positive")
    index: dict[tuple[str, int, int], CoordinatePair] = {}
    for record in records:
        key = (record.parent_id, record.row, record.col)
        if key in index:
            raise ValueError(f"duplicate coordinate within parent group: {key}")
        index[key] = record
    result: list[SpatialPhasePair] = []
    for key in sorted(index):
        parent_id, row, col = key
        target = (
            (parent_id, row + distance, col)
            if axis == "row"
            else (parent_id, row, col + distance)
        )
        if target in index:
            result.append(
                SpatialPhasePair(axis, distance, index[key], index[target])
            )
    return result


def select_pairs(
    records: Sequence[CoordinatePair],
    distances: Sequence[int],
    pairs_per_distance_axis: int,
) -> tuple[list[SpatialPhasePair], list[dict[str, Any]]]:
    selected: list[SpatialPhasePair] = []
    inventory: list[dict[str, Any]] = []
    for axis in ("row", "col"):
        for distance in distances:
            candidates = build_pair_candidates(records, axis, int(distance))
            indices = evenly_spaced_indices(len(candidates), pairs_per_distance_axis)
            chosen = [candidates[index] for index in indices]
            selected.extend(chosen)
            inventory.append(
                {
                    "axis": axis,
                    "distance": int(distance),
                    "candidate_pair_count": len(candidates),
                    "selected_pair_count": len(chosen),
                }
            )
    return selected, inventory


def load_phase_data(
    record: CoordinatePair, *, fft_norm: str, phasor_epsilon: float
) -> PhaseData:
    echo, image = load_pair(record.pair)
    normalized_echo, normalized_image, _, correction, weights = normalized_phase_oracle(
        echo, image, fft_norm=fft_norm, phasor_epsilon=phasor_epsilon
    )
    return PhaseData(
        correction=correction,
        cross_energy=np.square(weights),
        echo_rms=math.sqrt(float(np.mean(np.abs(echo) ** 2))),
        normalized_echo=normalized_echo,
        normalized_image=normalized_image,
    )


def relative_energy(energy: np.ndarray) -> np.ndarray:
    reference = float(np.quantile(energy, 0.99))
    if not math.isfinite(reference) or reference <= 0:
        return np.zeros_like(energy, dtype=np.float64)
    return np.asarray(energy / reference, dtype=np.float64)


def weighted_phase_similarity(
    first: np.ndarray, second: np.ndarray, weights: np.ndarray
) -> float:
    denominator = float(weights.sum())
    if denominator <= 0:
        return 0.0
    similarity = np.real(second * np.conj(first))
    return float(np.sum(similarity * weights) / denominator)


def pair_profile_rows(
    pair: SpatialPhasePair,
    first: PhaseData,
    second: PhaseData,
    *,
    phasor_epsilon: float,
    relative_thresholds_db: Sequence[float],
    soft_weight_powers: Sequence[float],
) -> list[dict[str, Any]]:
    base = {
        "axis": pair.axis,
        "distance": pair.distance,
        "first_file": pair.first.pair.echo.name,
        "second_file": pair.second.pair.echo.name,
        "first_row": pair.first.row,
        "first_col": pair.first.col,
        "second_row": pair.second.row,
        "second_col": pair.second.col,
        "pair_log10_echo_rms": math.log10(
            max(math.sqrt(first.echo_rms * second.echo_rms), 1.0e-30)
        ),
    }
    reliable = (first.cross_energy > phasor_epsilon) & (
        second.cross_energy > phasor_epsilon
    )
    baseline_weights = (
        np.power(first.cross_energy * second.cross_energy, 0.25) * reliable
    )
    rows = [
        {
            **base,
            "profile": "current_absolute_reliability",
            "relative_threshold_db": None,
            "soft_weight_power": 0.5,
            "retained_frequency_fraction": float(reliable.mean()),
            "phase_similarity": weighted_phase_similarity(
                first.correction, second.correction, baseline_weights
            ),
        }
    ]
    first_relative = relative_energy(first.cross_energy)
    second_relative = relative_energy(second.cross_energy)
    for threshold_db in relative_thresholds_db:
        threshold = 10.0 ** (float(threshold_db) / 10.0)
        mask = (first_relative >= threshold) & (second_relative >= threshold)
        for power in soft_weight_powers:
            weights = (
                np.power(first_relative * second_relative, float(power) / 2.0)
                * mask
            )
            rows.append(
                {
                    **base,
                    "profile": "relative_energy_ablation",
                    "relative_threshold_db": float(threshold_db),
                    "soft_weight_power": float(power),
                    "retained_frequency_fraction": float(mask.mean()),
                    "phase_similarity": weighted_phase_similarity(
                        first.correction, second.correction, weights
                    ),
                }
            )
    return rows


def assign_energy_strata(rows: list[dict[str, Any]]) -> dict[str, float]:
    baseline = [
        float(row["pair_log10_echo_rms"])
        for row in rows
        if row["profile"] == "current_absolute_reliability"
    ]
    if not baseline:
        return {"low_mid_boundary": 0.0, "mid_high_boundary": 0.0}
    low, high = np.quantile(np.asarray(baseline), (1.0 / 3.0, 2.0 / 3.0))
    for row in rows:
        value = float(row["pair_log10_echo_rms"])
        row["echo_energy_stratum"] = "low" if value <= low else "mid" if value <= high else "high"
    return {"low_mid_boundary": float(low), "mid_high_boundary": float(high)}


def summarize_pair_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["profile"],
            row["relative_threshold_db"],
            row["soft_weight_power"],
            row["axis"],
            row["distance"],
            row["echo_energy_stratum"],
        )
        grouped[key].append(row)
    result: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items(), key=lambda item: str(item[0])):
        similarity = distribution([row["phase_similarity"] for row in group])
        retained = distribution([row["retained_frequency_fraction"] for row in group])
        result.append(
            {
                "profile": key[0],
                "relative_threshold_db": key[1],
                "soft_weight_power": key[2],
                "axis": key[3],
                "distance": key[4],
                "echo_energy_stratum": key[5],
                "pair_count": len(group),
                "phase_similarity_mean": similarity["mean"],
                "phase_similarity_median": similarity["median"],
                "phase_similarity_p05": similarity["p05"],
                "retained_frequency_fraction_mean": retained["mean"],
                "retained_frequency_fraction_median": retained["median"],
            }
        )
    return result


def masked_oracle_rows(
    records: Sequence[CoordinatePair],
    *,
    fft_norm: str,
    phasor_epsilon: float,
    relative_thresholds_db: Sequence[float],
    floor_db: float,
    high_frequency_radius_fraction: float,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for record in records:
        data = load_phase_data(record, fft_norm=fft_norm, phasor_epsilon=phasor_epsilon)
        echo_metrics = evaluate_focus_prediction(
            data.normalized_echo,
            data.normalized_image,
            floor_db=floor_db,
            high_frequency_radius_fraction=high_frequency_radius_fraction,
        )
        echo_rmse = float(echo_metrics["normalized_complex_rmse"])
        spectrum = np.fft.fftshift(np.fft.fft2(data.normalized_echo, norm=fft_norm))
        relative = relative_energy(data.cross_energy)
        profiles: list[tuple[str, float | None, np.ndarray]] = [
            (
                "current_absolute_reliability",
                None,
                data.cross_energy > phasor_epsilon,
            )
        ]
        profiles.extend(
            (
                "relative_energy_ablation",
                float(threshold_db),
                relative >= 10.0 ** (float(threshold_db) / 10.0),
            )
            for threshold_db in relative_thresholds_db
        )
        for profile, threshold_db, mask in profiles:
            correction = np.ones_like(data.correction)
            correction[mask] = data.correction[mask]
            prediction = np.fft.ifft2(
                np.fft.ifftshift(spectrum * correction), norm=fft_norm
            )
            metrics = evaluate_focus_prediction(
                prediction,
                data.normalized_image,
                floor_db=floor_db,
                high_frequency_radius_fraction=high_frequency_radius_fraction,
            )
            rmse = float(metrics["normalized_complex_rmse"])
            result.append(
                {
                    "file": record.pair.echo.name,
                    "row": record.row,
                    "col": record.col,
                    "profile": profile,
                    "relative_threshold_db": threshold_db,
                    "retained_frequency_fraction": float(mask.mean()),
                    "normalized_complex_rmse": rmse,
                    "rmse_gap_fraction_closed": (
                        (echo_rmse - rmse) / echo_rmse if echo_rmse > 0 else 0.0
                    ),
                    "complex_coherence": float(metrics["complex_coherence"]),
                    "log_magnitude_ssim": float(metrics["log_magnitude_ssim"]),
                    "edge_correlation": float(metrics["edge_correlation"]),
                }
            )
    return result


def save_figures(
    output_dir: Path,
    pair_rows: Sequence[dict[str, Any]],
    oracle_rows: Sequence[dict[str, Any]],
) -> None:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    baseline = [row for row in pair_rows if row["profile"] == "current_absolute_reliability"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for axis_name, axis in zip(("row", "col"), axes):
        selected = [row for row in baseline if row["axis"] == axis_name]
        distances = sorted({int(row["distance"]) for row in selected})
        medians = [
            float(np.median([row["phase_similarity"] for row in selected if row["distance"] == distance]))
            for distance in distances
        ]
        axis.plot(distances, medians, marker="o")
        axis.set(title=f"{axis_name} direction", xlabel="coordinate distance", ylabel="median phase similarity")
        axis.grid(alpha=0.3)
    fig.savefig(figure_dir / "baseline_similarity_vs_distance.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    grouped: dict[float | None, list[dict[str, Any]]] = defaultdict(list)
    for row in oracle_rows:
        grouped[row["relative_threshold_db"]].append(row)
    points = []
    for threshold, rows in grouped.items():
        points.append(
            (
                float(np.mean([row["retained_frequency_fraction"] for row in rows])),
                float(np.mean([row["rmse_gap_fraction_closed"] for row in rows])),
                "current" if threshold is None else f"{threshold:g} dB",
            )
        )
    for retained, closed, label in points:
        axis.scatter(retained, closed)
        axis.annotate(label, (retained, closed), xytext=(4, 4), textcoords="offset points")
    axis.set(xlabel="mean retained frequency fraction", ylabel="mean Oracle RMSE gap closed", title="Reliability mask trade-off")
    axis.grid(alpha=0.3)
    fig.savefig(figure_dir / "mask_recoverability_tradeoff.png", dpi=160)
    plt.close(fig)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    validate_source_output_separation(args.echo_dir, args.image_dir, args.output_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    pairs, pairing = discover_pairs(args.echo_dir, args.image_dir)
    records, grouping = coordinate_records(pairs)
    selected, inventory = select_pairs(
        records, args.distances, args.pairs_per_distance_axis
    )
    pair_rows: list[dict[str, Any]] = []
    for index, pair in enumerate(selected, start=1):
        first = load_phase_data(pair.first, fft_norm=args.fft_norm, phasor_epsilon=args.phasor_epsilon)
        second = load_phase_data(pair.second, fft_norm=args.fft_norm, phasor_epsilon=args.phasor_epsilon)
        pair_rows.extend(
            pair_profile_rows(
                pair,
                first,
                second,
                phasor_epsilon=args.phasor_epsilon,
                relative_thresholds_db=args.relative_thresholds_db,
                soft_weight_powers=args.soft_weight_powers,
            )
        )
        if args.progress_every and index % args.progress_every == 0:
            print(f"neighborhood pairs {index}/{len(selected)}", flush=True)
    strata = assign_energy_strata(pair_rows)
    summary_rows = summarize_pair_rows(pair_rows)
    unique_records = {
        (record.parent_id, record.row, record.col): record
        for pair in selected
        for record in (pair.first, pair.second)
    }
    ordered_records = [unique_records[key] for key in sorted(unique_records)]
    oracle_indices = evenly_spaced_indices(len(ordered_records), args.oracle_sample_count)
    oracle_records = [ordered_records[index] for index in oracle_indices]
    oracle_rows = masked_oracle_rows(
        oracle_records,
        fft_norm=args.fft_norm,
        phasor_epsilon=args.phasor_epsilon,
        relative_thresholds_db=args.relative_thresholds_db,
        floor_db=args.floor_db,
        high_frequency_radius_fraction=args.high_frequency_radius_fraction,
    )
    write_csv(args.output_dir / "pair_metrics.csv", pair_rows)
    write_csv(args.output_dir / "profile_summary.csv", summary_rows)
    write_csv(args.output_dir / "oracle_mask_tradeoff.csv", oracle_rows)
    save_figures(args.output_dir, pair_rows, oracle_rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "experiment": "E014-D001-refined-phase-neighborhood-reliability",
        "read_only_contract": {
            "source_mutation": "forbidden",
            "dataset_splitting": "not_implemented_by_design",
        },
        "pairing": pairing,
        "coordinate_grouping": grouping,
        "parameters": {
            "distances": [int(value) for value in args.distances],
            "pairs_per_distance_axis": int(args.pairs_per_distance_axis),
            "relative_thresholds_db": [float(value) for value in args.relative_thresholds_db],
            "soft_weight_powers": [float(value) for value in args.soft_weight_powers],
            "relative_energy_reference": "per-patch 99th percentile cross-spectrum energy",
            "current_baseline": "cross energy > phasor_epsilon; sqrt cross-energy phase weights",
            "patch_pair_weighting": "uniform; brightness is reporting-only",
            "oracle_sample_count": len(oracle_records),
        },
        "pair_inventory": inventory,
        "selected_pair_count": len(selected),
        "pair_profile_row_count": len(pair_rows),
        "echo_energy_strata_log10_rms_boundaries": strata,
        "outputs": {
            "pair_metrics": "pair_metrics.csv",
            "profile_summary": "profile_summary.csv",
            "oracle_mask_tradeoff": "oracle_mask_tradeoff.csv",
            "figures": "figures",
        },
    }
    write_json(args.output_dir / "summary.json", report)
    print(f"status=completed report={(args.output_dir / 'summary.json').resolve()}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--echo-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--distances", type=int, nargs="+", default=(100, 200, 400, 800, 1600))
    parser.add_argument("--pairs-per-distance-axis", type=int, default=100)
    parser.add_argument("--relative-thresholds-db", type=float, nargs="+", default=(-20.0, -30.0, -40.0, -50.0))
    parser.add_argument("--soft-weight-powers", type=float, nargs="+", default=(0.0, 0.25, 0.5, 1.0))
    parser.add_argument("--oracle-sample-count", type=int, default=128)
    parser.add_argument("--fft-norm", choices=("ortho", "backward", "forward"), default="ortho")
    parser.add_argument("--phasor-epsilon", type=float, default=1.0e-6)
    parser.add_argument("--floor-db", type=float, default=-60.0)
    parser.add_argument("--high-frequency-radius-fraction", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if any(value <= 0 for value in args.distances):
        parser.error("--distances values must be positive")
    if args.pairs_per_distance_axis <= 0:
        parser.error("--pairs-per-distance-axis must be positive")
    if args.oracle_sample_count <= 0:
        parser.error("--oracle-sample-count must be positive")
    if any(value >= 0 for value in args.relative_thresholds_db):
        parser.error("--relative-thresholds-db values must be negative")
    if any(value < 0 for value in args.soft_weight_powers):
        parser.error("--soft-weight-powers values must be non-negative")
    return args


def main() -> None:
    audit(parse_args())


if __name__ == "__main__":
    main()
