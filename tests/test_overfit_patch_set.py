from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.io import savemat

from scripts.overfit_patch_set import (
    patches_overlap,
    run,
    select_spatially_distributed_pairs,
    shuffled_sample_index,
    validate_args,
)
from swinir.sar_dataset import discover_pairs


def write_tiny_config(path: Path) -> None:
    config = {
        "model": {
            "img_size": 16,
            "patch_size": 1,
            "in_chans": 2,
            "embed_dim": 12,
            "depths": [1],
            "num_heads": [3],
            "window_size": 4,
            "mlp_ratio": 2.0,
            "qkv_bias": True,
            "qk_scale": None,
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "drop_path_rate": 0.5,
            "ape": False,
            "patch_norm": True,
            "use_checkpoint": False,
            "upscale": 1,
            "img_range": 1.0,
            "upsampler": "",
            "resi_connection": "1conv",
        },
        "data": {"expected_shape": [16, 16], "rms_epsilon": 1.0e-12},
        "optimization": {"charbonnier_epsilon": 1.0e-3},
    }
    path.write_text(yaml.safe_dump(config), encoding="utf-8")


def write_pair(echo_dir: Path, image_dir: Path, filename: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    echo = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
    image = (0.8 + 0.1j) * echo
    savemat(echo_dir / filename, {"patch": echo})
    savemat(image_dir / filename, {"patch": image})


def experiment_args(
    *,
    config: Path,
    echo_dir: Path,
    image_dir: Path,
    output_dir: Path,
    anchor_filename: str,
    resume: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        config=config,
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        anchor_filename=anchor_filename,
        sample_count=2,
        resume=resume,
        device="cpu",
        steps=2,
        eval_every=2,
        save_every=2,
        required_consecutive_successes=1,
        seed=42,
        learning_rate=2.0e-4,
        ema_decay=0.999,
        charbonnier_epsilon=None,
        rms_epsilon=None,
        db_floor=-60.0,
        success_rmse=0.0,
        success_coherence=1.0,
        success_magnitude_correlation=1.0,
        success_rms_ratio_min=1.0,
        success_rms_ratio_max=1.0,
        success_psnr=300.0,
        success_ssim=1.0,
    )


def test_farthest_selection_is_deterministic_and_non_overlapping(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    for index, (row, col) in enumerate(
        ((0, 0), (0, 20), (0, 40), (20, 0), (20, 20), (20, 40), (40, 0), (40, 20), (40, 40))
    ):
        write_pair(echo_dir, image_dir, f"patch_row_{row}_col_{col}_2.mat", index)
    pairs = discover_pairs(echo_dir, image_dir)
    anchor = "patch_row_20_col_20_2.mat"

    first = select_spatially_distributed_pairs(
        pairs, sample_count=5, anchor_filename=anchor, patch_shape=(16, 16)
    )
    second = select_spatially_distributed_pairs(
        pairs, sample_count=5, anchor_filename=anchor, patch_shape=(16, 16)
    )
    expanded = select_spatially_distributed_pairs(
        pairs, sample_count=7, anchor_filename=anchor, patch_shape=(16, 16)
    )
    row_span = max(pair.row for pair in pairs) - min(pair.row for pair in pairs)
    col_span = max(pair.col for pair in pairs) - min(pair.col for pair in pairs)
    reference = [next(pair for pair in pairs if pair.echo_path.name == anchor)]
    while len(reference) < 7:
        eligible = [
            pair
            for pair in pairs
            if pair not in reference
            and all(not patches_overlap(pair, chosen, (16, 16)) for chosen in reference)
        ]
        reference.append(
            max(
                eligible,
                key=lambda pair: (
                    min(
                        ((pair.row - chosen.row) / row_span) ** 2
                        + ((pair.col - chosen.col) / col_span) ** 2
                        for chosen in reference
                    ),
                    -pair.row,
                    -pair.col,
                ),
            )
        )

    assert [pair.echo_path.name for pair in first] == [pair.echo_path.name for pair in second]
    assert [pair.echo_path.name for pair in first] == [
        pair.echo_path.name for pair in expanded[:5]
    ]
    assert [pair.echo_path.name for pair in expanded] == [
        pair.echo_path.name for pair in reference
    ]
    assert first[0].echo_path.name == anchor
    for index, pair in enumerate(first):
        assert all(
            not patches_overlap(pair, other, (16, 16)) for other in first[index + 1 :]
        )


def test_shuffled_order_covers_each_sample_once_per_epoch() -> None:
    first_epoch = [shuffled_sample_index(step, 4, 42) for step in range(4)]
    second_epoch = [shuffled_sample_index(step, 4, 42) for step in range(4, 8)]

    assert sorted(first_epoch) == [0, 1, 2, 3]
    assert sorted(second_epoch) == [0, 1, 2, 3]
    assert first_epoch != second_epoch


def test_validate_args_requires_whole_epoch_evaluations(tmp_path: Path) -> None:
    args = experiment_args(
        config=tmp_path / "config.yaml",
        echo_dir=tmp_path,
        image_dir=tmp_path,
        output_dir=tmp_path / "run",
        anchor_filename="anchor.mat",
        resume=None,
    )
    args.eval_every = 3

    with pytest.raises(ValueError, match="whole epochs"):
        validate_args(args)


def test_tiny_joint_run_writes_manifest_metrics_and_resumes(tmp_path: Path) -> None:
    config = tmp_path / "tiny.yaml"
    write_tiny_config(config)
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    anchor = "patch_row_0_col_0_2.mat"
    write_pair(echo_dir, image_dir, anchor, 1)
    write_pair(echo_dir, image_dir, "patch_row_100_col_100_2.mat", 2)
    output_dir = tmp_path / "run"

    report = run(
        experiment_args(
            config=config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            anchor_filename=anchor,
            resume=None,
        )
    )

    assert report["status"] == "failed"
    assert report["selection_manifest"]["sample_count"] == 2
    assert report["final"]["raw"]["sample_count"] == 2
    assert (output_dir / "selected_samples.json").is_file()
    assert (output_dir / "metrics.jsonl").is_file()
    assert (output_dir / "checkpoints" / "best.pt").is_file()
    assert (output_dir / "checkpoints" / "latest.pt").is_file()
    assert (output_dir / "checkpoints" / "final.pt").is_file()
    selected = json.loads(
        (output_dir / "selected_samples.json").read_text(encoding="utf-8")
    )
    assert selected["samples"][0]["filename"] == anchor

    resumed = run(
        experiment_args(
            config=config,
            echo_dir=echo_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            anchor_filename=anchor,
            resume=output_dir / "checkpoints" / "latest.pt",
        )
    )
    assert resumed["step"] == 2
