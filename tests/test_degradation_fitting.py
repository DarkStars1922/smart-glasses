from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from analysis.degradation.fitting import (
    fit_burst_noise,
    fit_color_response,
    fit_manifest,
    fit_point_grid,
)
from analysis.degradation.reporting import _params_for_comparison, write_results
from analysis.degradation.schema import load_manifest


SOURCE_BARS = np.asarray(
    [
        (255, 255, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
    ],
    dtype=np.float32,
) / 255.0


@pytest.fixture(scope="module")
def current_fit() -> tuple[object, dict[str, object]]:
    manifest = load_manifest(Path("analysis/calibration_b_v1.json"))
    return manifest, fit_manifest(manifest, seed=20260721)


def _synthetic_point_burst() -> list[np.ndarray]:
    height, width = 104, 124
    base = np.zeros((height, width), dtype=np.float32)
    for y in range(16, height - 10, 16):
        for x in range(20, width - 10, 20):
            base[y, x] = 1.0
    blurred = cv2.GaussianBlur(base, (0, 0), sigmaX=3.0 / 2.355, sigmaY=5.0 / 2.355)
    blurred /= blurred.max()
    rgb = np.stack((0.15 * blurred, 0.9 * blurred, blurred), axis=-1)
    generator = np.random.default_rng(412)
    return [
        np.clip(rgb + generator.normal(0.0, 0.002, rgb.shape), 0.0, 1.0).astype(
            np.float32
        )
        for _ in range(10)
    ]


def _synthetic_heteroscedastic_burst() -> list[np.ndarray]:
    x = np.linspace(0.08, 0.92, 64, dtype=np.float32)
    base = np.broadcast_to(x[None, :, None], (64, 64, 3)).copy()
    base[:, 31:33] = 0.02
    slope = np.asarray((0.02, 0.01, 0.03), dtype=np.float32)
    intercept = np.asarray((0.0004, 0.0004, 0.0004), dtype=np.float32)
    generator = np.random.default_rng(905)
    return [
        np.clip(
            base + generator.normal(0.0, np.sqrt(base * slope + intercept), base.shape),
            0.0,
            1.0,
        ).astype(np.float32)
        for _ in range(48)
    ]


def _diagonal_bar_observations() -> list[np.ndarray]:
    gains = np.asarray((0.5, 0.8, 1.1), dtype=np.float32)
    colors = np.clip(SOURCE_BARS * gains + 0.03, 0.0, 1.0)
    image = np.zeros((64, 160, 3), dtype=np.float32)
    for index, color in enumerate(colors):
        image[:, index * 20 : (index + 1) * 20] = color
    return [image.copy() for _ in range(6)]


def test_point_grid_recovers_scale_and_fwhm() -> None:
    result = fit_point_grid(
        _synthetic_point_burst(),
        {"point_spacing_px": 40, "point_size_px": 0},
    )

    assert result.status == "estimated"
    assert result.value["scale_camera_per_source"] == pytest.approx(
        (0.5, 0.4), abs=0.03
    )
    assert result.value["fwhm_camera_px"] == pytest.approx((3.0, 5.0), abs=0.9)


def test_noise_fit_ignores_high_gradient_pixels() -> None:
    result = fit_burst_noise(_synthetic_heteroscedastic_burst(), {})

    assert result.status == "estimated"
    assert result.value["slope_rgb"] == pytest.approx(
        (0.02, 0.01, 0.03), abs=0.007
    )


def test_color_fit_prefers_diagonal_when_full_matrix_does_not_generalize() -> None:
    result = fit_color_response(
        _diagonal_bar_observations(),
        {
            "colors_rgb": (SOURCE_BARS * 255.0).astype(int).tolist(),
            "bar_axis": "x",
            "bar_order": "forward",
        },
    )

    assert result.status == "estimated"
    assert result.value["selected_model"] == "diagonal"
    assert result.value["leave_one_out_mae"] < 0.02


def test_default_synthesis_does_not_apply_provisional_color_or_noise() -> None:
    source = np.zeros((8, 8, 3), dtype=np.float32)
    real = np.zeros((8, 8, 3), dtype=np.float32)
    effective = {
        "scale_camera_per_source": [1.0, 1.0],
        "blur_fwhm_camera_px": [0.0, 0.0],
        "blur_angle_deg": 0.0,
        "color_matrix": [[0.0, 0.0, 0.0]] * 3,
        "color_bias_rgb": [0.3, 0.4, 0.5],
        "noise_slope_rgb": [0.1, 0.1, 0.1],
        "noise_intercept_rgb": [0.1, 0.1, 0.1],
        "jpeg_quality": 96,
        "status": {"color": "provisional", "noise": "provisional"},
    }

    params = _params_for_comparison(source, effective, real)

    assert params.color_matrix == (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    assert params.bias_rgb == (0.0, 0.0, 0.0)
    assert params.noise_slope == (0.0, 0.0, 0.0)
    assert params.noise_intercept == (0.0, 0.0, 0.0)


def test_current_dataset_reports_effective_identifiability(
    current_fit: tuple[object, dict[str, object]],
) -> None:
    _, result = current_fit

    assert result["dataset"]["frame_count"] == 120
    assert result["dataset"]["jpeg"]["equivalent_quality"] == 96
    assert result["dataset"]["jpeg"]["subsampling"] == "4:2:0"
    assert set(result["domains"]) == {
        "supermacro_47_primary",
        "supermacro_69_primary",
    }
    for domain in result["domains"].values():
        assert domain["identifiability"]["display_camera_response"]["status"] == (
            "not_identifiable"
        )
        assert domain["identifiability"]["independent_sensor_noise"]["status"] == (
            "not_identifiable"
        )
    assert result["domains"]["supermacro_47_primary"]["groups"]["23"]["psf"][
        "status"
    ] == "estimated"
    for domain in result["domains"].values():
        assert "effective_parameters" in domain
        assert domain["effective_parameters"]["jpeg_quality"] == 96
        assert len(domain["effective_parameters"]["scale_camera_per_source"]) == 2


def test_writes_deterministic_results_and_diagnostics(
    tmp_path: Path,
    current_fit: tuple[object, dict[str, object]],
    caplog: pytest.LogCaptureFixture,
) -> None:
    manifest, result = current_fit
    first = tmp_path / "first"
    second = tmp_path / "second"

    write_results(result, manifest, first, seed=20260721)
    write_results(result, manifest, second, seed=20260721)

    first_files = {
        path.relative_to(first): path.read_bytes()
        for path in sorted(first.rglob("*"))
        if path.is_file()
    }
    second_files = {
        path.relative_to(second): path.read_bytes()
        for path in sorted(second.rglob("*"))
        if path.is_file()
    }
    assert first_files == second_files
    parameters = json.loads((first / "v1_parameters.json").read_text(encoding="utf-8"))
    assert parameters["dataset"]["jpeg"]["equivalent_quality"] == 96
    report = (first / "v1_report.md").read_text(encoding="utf-8")
    assert "Identifiability" in report
    assert "47 mm" in report
    assert "69 mm" in report
    assert "JPEG quality 96" in report
    assert "下一轮拍摄" in report
    diagnostics = list((first / "v1_diagnostics").glob("*.png"))
    assert len([path for path in diagnostics if path.name.startswith("roi_")]) == 12
    assert len([path for path in diagnostics if path.name.startswith("comparison_")]) >= 2
    assert not [
        record for record in caplog.records if "Clipping input data" in record.message
    ]
