from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.analyze_phase_neighborhood import audit
from tests.test_analyze_phase_dataset import write_paired_grid


def test_refined_neighborhood_audit_is_read_only_and_writes_profiles(
    tmp_path: Path,
) -> None:
    write_paired_grid(tmp_path, patch_size=16, step=4)
    source_bytes = {
        path: path.read_bytes()
        for directory in (tmp_path / "echo", tmp_path / "image")
        for path in directory.glob("*.mat")
    }
    output = tmp_path / "neighborhood"
    report = audit(
        Namespace(
            echo_dir=tmp_path / "echo",
            image_dir=tmp_path / "image",
            output_dir=output,
            distances=(4,),
            pairs_per_distance_axis=2,
            relative_thresholds_db=(-20.0, -40.0),
            soft_weight_powers=(0.0, 0.5),
            oracle_sample_count=2,
            fft_norm="ortho",
            phasor_epsilon=1.0e-6,
            floor_db=-60.0,
            high_frequency_radius_fraction=0.25,
            progress_every=0,
        )
    )

    assert report["selected_pair_count"] == 4
    assert report["pair_profile_row_count"] == 20
    assert report["parameters"]["patch_pair_weighting"].startswith("uniform")
    assert report["read_only_contract"]["dataset_splitting"].startswith(
        "not_implemented"
    )
    for relative in (
        "summary.json",
        "pair_metrics.csv",
        "profile_summary.csv",
        "oracle_mask_tradeoff.csv",
        "figures/baseline_similarity_vs_distance.png",
        "figures/mask_recoverability_tradeoff.png",
    ):
        assert (output / relative).is_file(), relative
    for path, expected in source_bytes.items():
        assert path.read_bytes() == expected
