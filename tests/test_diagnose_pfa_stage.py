from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import savemat

from scripts.diagnose_pfa_stage import (
    TransformSpec,
    diagnose,
    infer_conclusion,
    load_pair,
    optimize_orientation,
    polar_to_cartesian,
    rank_specs,
    resize_coordinate_grid,
    resize_image_complex,
    simple_specs,
    split_fit_holdout,
)
from scripts.analyze_sar_dataset import FilePair


def complex_noise(seed: int, shape: tuple[int, int] = (24, 24)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def repeated_samples(
    echo: np.ndarray, image: np.ndarray, count: int = 3
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return [
        (f"sample_{index}", echo * np.exp(0.1j * index), image)
        for index in range(count)
    ]


def test_simple_stage_ranking_recovers_known_transforms() -> None:
    image = complex_noise(10)
    cases = (
        ("image_domain", image),
        ("one_dimensional_fft", np.fft.fft(image, axis=1, norm="ortho")),
        ("cartesian_2d_spectrum", np.fft.fft2(image, norm="ortho")),
    )

    for expected_stage, echo in cases:
        result = rank_specs(simple_specs(), repeated_samples(echo, image))[0]
        assert result.spec.stage == expected_stage
        assert result.metrics["complex_coherence_median"] > 0.999999
        assert result.score > 0.999999


def test_polar_hypothesis_beats_simple_fft_for_consistent_sector_data() -> None:
    polar_echoes = [complex_noise(seed, (28, 28)) for seed in range(3)]
    polar_spec = TransformSpec(
        name="known_polar",
        stage="polar_frequency",
        family="polar",
        axes=(0, 1),
        direction="ifft",
        theta_axis=0,
        theta_span_deg=20.0,
        radius_center_ratio=2.0,
    )
    samples = []
    for index, echo in enumerate(polar_echoes):
        cartesian = polar_to_cartesian(
            echo,
            theta_axis=0,
            theta_reversed=False,
            radial_reversed=False,
            theta_span_deg=20.0,
            radius_center_ratio=2.0,
        )
        image = np.fft.ifft2(cartesian, norm="ortho")
        samples.append((f"sample_{index}", echo, image))

    polar_result = optimize_orientation(polar_spec, samples)
    simple_result = rank_specs(simple_specs(), samples)[0]

    assert polar_result.score > 0.999999
    assert polar_result.score > simple_result.score + 0.20
    conclusion = infer_conclusion([polar_result, simple_result])
    assert conclusion["selected_stage"] == "polar_frequency"
    assert conclusion["evidence_strength"] == "strong_support"


def test_uninformative_candidates_are_reported_as_unknown() -> None:
    echo = complex_noise(1)
    target = complex_noise(2)
    result = optimize_orientation(simple_specs()[0], repeated_samples(echo, target))

    conclusion = infer_conclusion([result])

    assert conclusion["selected_stage"] == "unknown"
    assert conclusion["evidence_strength"] == "unidentified"


def test_matlab_v5_complex_patch_pair_is_loaded(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    echo_dir.mkdir()
    image_dir.mkdir()
    filename = "patch_row_100_col_200.mat"
    echo = complex_noise(20, (8, 8))
    image = complex_noise(21, (8, 8))
    savemat(echo_dir / filename, {"patch": echo})
    savemat(image_dir / filename, {"patch": image})
    pair = FilePair("sample", echo_dir / filename, image_dir / filename)

    loaded_echo, loaded_image = load_pair(pair)

    np.testing.assert_allclose(loaded_echo, echo)
    np.testing.assert_allclose(loaded_image, image)


def test_fit_holdout_split_is_disjoint_and_spans_sorted_pairs() -> None:
    pairs = [
        FilePair(f"pair_{index}", Path(f"echo_{index}.mat"), Path(f"image_{index}.mat"))
        for index in range(8)
    ]

    fit, holdout = split_fit_holdout(pairs, 4)

    assert len(fit) == 4
    assert len(holdout) == 4
    assert {pair.key for pair in fit}.isdisjoint(pair.key for pair in holdout)
    assert fit[0].key == "pair_0"
    assert fit[-1].key == "pair_7"


def test_coarse_resizing_distinguishes_coordinate_grid_from_image() -> None:
    values = complex_noise(30, (32, 32))

    coordinate_grid = resize_coordinate_grid(values, (16, 16))
    image = resize_image_complex(values, (16, 16))

    assert coordinate_grid.shape == (16, 16)
    assert image.shape == (16, 16)
    assert np.iscomplexobj(coordinate_grid)
    assert np.iscomplexobj(image)
    assert not np.allclose(coordinate_grid, image)


def test_diagnose_runs_from_mat_directories_and_writes_reports(tmp_path: Path) -> None:
    echo_dir = tmp_path / "echo"
    image_dir = tmp_path / "image"
    output_dir = tmp_path / "report"
    echo_dir.mkdir()
    image_dir.mkdir()
    for index in range(4):
        image = complex_noise(100 + index, (16, 16))
        echo = np.fft.fft2(image, norm="ortho")
        filename = f"patch_row_{index * 100}_col_0.mat"
        savemat(echo_dir / filename, {"patch": echo})
        savemat(image_dir / filename, {"patch": image})
    args = SimpleNamespace(
        echo_dir=echo_dir,
        image_dir=image_dir,
        output_dir=output_dir,
        sample_count=4,
        fit_count=2,
        search_size=16,
        coarse_fit_count=2,
        simple_top_per_stage=1,
        polar_refine_count=1,
        polar_full_count=1,
        progress_every=0,
        theta_spans=(20.0,),
        radius_center_ratios=(2.0,),
    )

    report = diagnose(args)

    assert report["conclusion"]["selected_stage"] == "cartesian_2d_spectrum"
    assert report["conclusion"]["evidence_strength"] == "strong_support"
    assert (output_dir / "diagnosis_report.json").is_file()
    assert (output_dir / "holdout_ranking.csv").is_file()
    assert (output_dir / "summary_zh.txt").is_file()
    assert (output_dir / "representative_best_candidate.png").is_file()
