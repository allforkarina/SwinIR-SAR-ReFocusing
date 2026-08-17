from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from scripts.compare_echo_image import compare_arrays, log_magnitude, run


def structured_complex(size: int = 16) -> np.ndarray:
    rows, cols = np.mgrid[:size, :size]
    magnitude = 0.1 + rows / size + 0.4 * cols / size
    phase = 0.05 * rows - 0.08 * cols
    return magnitude * np.exp(1j * phase)


def test_independent_log_normalization_is_scale_invariant() -> None:
    echo = structured_complex()

    echo_db, echo_normalized, echo_peak = log_magnitude(echo, floor_db=-60.0)
    image_db, image_normalized, image_peak = log_magnitude(
        0.25 * echo, floor_db=-60.0
    )

    assert echo_peak == pytest.approx(4.0 * image_peak)
    np.testing.assert_allclose(echo_db, image_db, atol=1e-12)
    np.testing.assert_allclose(echo_normalized, image_normalized, atol=1e-12)
    assert float(echo_db.max()) == pytest.approx(0.0)
    assert float(echo_db.min()) >= -60.0


def test_shared_peak_scale_preserves_amplitude_difference() -> None:
    echo = structured_complex()
    image = 0.5 * echo

    result = compare_arrays(echo, image, floor_db=-60.0)

    np.testing.assert_allclose(
        result["arrays"]["echo_independent_db"],
        result["arrays"]["image_independent_db"],
        atol=1e-12,
    )
    assert float(result["arrays"]["echo_shared_db"].max()) == pytest.approx(0.0)
    assert float(result["arrays"]["image_shared_db"].max()) == pytest.approx(
        20.0 * np.log10(0.5)
    )
    assert result["comparison"]["image_to_echo_rms_ratio"] == pytest.approx(0.5)
    assert result["comparison"]["independent_log_magnitude_correlation"] == pytest.approx(
        1.0
    )


def test_compare_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_arrays(structured_complex(16), structured_complex(12), floor_db=-60.0)


def test_run_writes_png_and_json_report(tmp_path: Path) -> None:
    echo = structured_complex()
    image = 0.75 * echo
    echo_file = tmp_path / "echo.mat"
    image_file = tmp_path / "image.mat"
    output_dir = tmp_path / "comparison"
    savemat(echo_file, {"patch": echo})
    savemat(image_file, {"patch": image})
    args = argparse.Namespace(
        echo_file=echo_file,
        image_file=image_file,
        output_dir=output_dir,
        variable_name="patch",
        db_floor=-60.0,
    )

    report = run(args)

    assert (output_dir / "comparison.png").stat().st_size > 0
    saved_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
    assert saved_report == report
    assert report["shape"] == [16, 16]
    assert report["comparison"]["image_to_echo_peak_ratio"] == pytest.approx(0.75)
