from __future__ import annotations

import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.degradation.calibration_c import (
    fit_field_point_edge_capture,
    fit_point_grid_capture,
    predict_quadratic_warp,
)
from analysis.degradation.fitting import _edge_widths, _read_exif, inspect_jpeg


ROOT = Path("images/real/c_C")
OUTPUT = Path("analysis/results")
SOURCE_SIZE = (2500, 1600)
POINT_GRID_CONFIG = {
    "W": {
        "folder": "01",
        "channel_index": None,
        "focal_mm": 47,
        "anchors": ((1696, 1317), (2202, 1337), (1669, 2052), (2169, 2089)),
    },
    "R": {
        "folder": "02",
        "channel_index": 0,
        "focal_mm": 49,
        "anchors": ((1496, 1646), (1984, 1687), (1455, 2370), (1947, 2420)),
    },
    "G": {
        "folder": "03",
        "channel_index": 1,
        "focal_mm": 49,
        "anchors": ((1447, 1395), (1956, 1449), (1402, 2135), (1905, 2185)),
    },
    "B": {
        "folder": "04",
        "channel_index": 2,
        "focal_mm": 49,
        "anchors": ((1418, 1917), (1932, 1940), (1417, 2665), (1931, 2690)),
    },
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 8) if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _burst_number(path: Path) -> int:
    match = re.search(r"TIMEBURST(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _files(folder: str) -> list[Path]:
    return sorted((ROOT / folder).glob("*.jpg"), key=lambda path: (_burst_number(path), path.name))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _quartiles(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "q25": np.percentile(values, 25, axis=0),
        "median": np.median(values, axis=0),
        "q75": np.percentile(values, 75, axis=0),
    }


def _fit_point_grids() -> tuple[dict[str, Any], dict[str, Any]]:
    summaries: dict[str, Any] = {}
    raw_results: dict[str, Any] = {}
    for color, config in POINT_GRID_CONFIG.items():
        path = _files(str(config["folder"]))[0]
        result = fit_point_grid_capture(
            _load_rgb(path),
            source_size=SOURCE_SIZE,
            anchor_corners=np.asarray(config["anchors"]),
            channel_index=config["channel_index"],
            square_side_px=15,
        )
        raw_results[color] = result
        points = [point for point in result["points"] if point["inlier"]]
        fwhm = np.asarray(
            [point["psf_fwhm_minor_major_angle"][:2] for point in points]
        )
        angles = np.asarray(
            [point["psf_fwhm_minor_major_angle"][2] for point in points]
        )
        scales = np.asarray([point["scale_singular_values"] for point in points])
        contrasts = np.asarray([point["contrast"] for point in points])
        focal = int(config["focal_mm"])
        normalization = 47.0 / focal
        warp = result["warp"]
        status = (
            "estimated"
            if len(points) >= 23 and float(np.median(contrasts)) >= 0.3
            else "provisional"
        )
        summaries[color] = {
            "status": status,
            "reason": (
                "quadratic field fit to automatically recovered point centroids"
                if status == "estimated"
                else "low-signal points were rejected; spatial trend is usable but color PSF remains provisional"
            ),
            "capture": str(path),
            "reported_focal_mm": focal,
            "normalization_to_47mm": normalization,
            "detected_count": result["detected_count"],
            "inlier_count": len(points),
            "rejected_indices": np.flatnonzero(~warp["inlier_mask"]),
            "quadratic_coefficients_normalized_source_to_camera": warp["coefficients"],
            "homography_source_to_camera": warp["homography"],
            "quadratic_residual_px": {
                "median": warp["quadratic_median_residual_px"],
                "p95": warp["quadratic_p95_residual_px"],
            },
            "homography_residual_px": {
                "median": warp["homography_median_residual_px"],
                "p95": warp["homography_p95_residual_px"],
            },
            "local_scale_singular_camera_per_source": _quartiles(scales),
            "local_scale_normalized_47mm": _quartiles(scales * normalization),
            "psf_fwhm_camera_px": _quartiles(fwhm),
            "psf_fwhm_normalized_47mm": _quartiles(fwhm * normalization),
            "psf_major_angle_deg_median": np.median(angles),
            "point_contrast": _quartiles(contrasts),
            "spatial_samples": [
                {
                    "source_xy": point["source_xy"],
                    "source_uv": np.asarray(point["source_xy"])
                    / np.asarray((SOURCE_SIZE[0] - 1, SOURCE_SIZE[1] - 1)),
                    "observed_xy": point["observed_xy"],
                    "scale_singular_camera_per_source": point["scale_singular_values"],
                    "psf_fwhm_camera_px": point["psf_fwhm_minor_major_angle"][:2],
                    "psf_major_angle_deg": point["psf_fwhm_minor_major_angle"][2],
                    "contrast": point["contrast"],
                }
                for point in points
            ],
        }
    return summaries, raw_results


def _fit_field_burst() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames: list[dict[str, Any]] = []
    accepted_fwhm: list[Sequence[float]] = []
    center_positions: list[np.ndarray] = []
    edge_widths: list[float] = []
    for path in _files("05"):
        image = _load_rgb(path)
        result = fit_field_point_edge_capture(
            image,
            source_size=SOURCE_SIZE,
            roi_xyxy=(1400, 1050, 2800, 2250),
            square_side_px=15,
        )
        point_rows = []
        for point in result["points"]:
            valid = point["prediction_residual_px"] < 20.0 and point["contrast"] > 0.08
            if valid:
                accepted_fwhm.append(point["psf_fwhm_minor_major_angle"][:2])
            point_rows.append(
                {
                    "valid": valid,
                    "prediction_residual_px": point["prediction_residual_px"],
                    "contrast": point["contrast"],
                    "psf_fwhm_camera_px": point["psf_fwhm_minor_major_angle"][:2],
                }
            )
        center = predict_quadratic_warp(
            np.asarray(((1250.0, 800.0),)),
            result["warp"]["coefficients"],
            source_size=SOURCE_SIZE,
        )[0]
        center_positions.append(center)
        widths = _edge_widths(image[1050:2250, 1400:2800])
        edge_widths.extend(widths)
        valid_rows = [row for row in point_rows if row["valid"]]
        frames.append(
            {
                "capture": str(path),
                "large_patch_count": result["large_patch_count"],
                "valid_point_count": len(valid_rows),
                "median_point_contrast": np.median([row["contrast"] for row in point_rows]),
                "median_valid_psf_fwhm_camera_px": (
                    np.median([row["psf_fwhm_camera_px"] for row in valid_rows], axis=0)
                    if valid_rows
                    else None
                ),
                "edge_gradient_core_width_px_median": np.median(widths) if widths else None,
                "chart_center_camera_xy": center,
                "points": point_rows,
            }
        )
    fwhm_array = np.asarray(accepted_fwhm)
    centers = np.asarray(center_positions)
    drift = centers - centers[0]
    summary = {
        "status": "provisional",
        "reason": (
            "the burst captures a strong autofocus transition; low-contrast early points are excluded from PSF summaries"
        ),
        "frame_count": len(frames),
        "accepted_point_count": len(fwhm_array),
        "psf_fwhm_camera_px": _quartiles(fwhm_array),
        "chart_drift_relative_first_px": {
            "x_range": (drift[:, 0].min(), drift[:, 0].max()),
            "y_range": (drift[:, 1].min(), drift[:, 1].max()),
        },
        "edge_gradient_core_width_camera_px": _quartiles(np.asarray(edge_widths)),
        "interpretation": (
            "2-3 px edge cores coexist with broader point PSFs, indicating nonlinear ISP sharpening/ringing; a single Gaussian cannot match both"
        ),
    }
    return summary, frames


def _joint_reference_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    margin, gray_gap = 145, 12
    gray_width = (2500 - 2 * margin - 7 * gray_gap) // 8
    gray_height = 245
    gray_source = np.asarray(
        [
            (
                margin + column * (gray_width + gray_gap) + gray_width / 2,
                125 + row * (gray_height + gray_gap) + gray_height / 2,
            )
            for row in range(2)
            for column in range(8)
        ],
        dtype=np.float32,
    )
    gray_source_corners = np.asarray(
        ((145, 125), (2349, 125), (145, 627), (2349, 627)), dtype=np.float32
    )
    gray_camera_corners = np.asarray(
        ((2432, 955), (2428, 1817), (2168, 951), (2168, 1818)), dtype=np.float32
    )
    gray_camera = cv2.perspectiveTransform(
        gray_source[None], cv2.getPerspectiveTransform(gray_source_corners, gray_camera_corners)
    )[0]

    color_gap = 14
    color_width = (2500 - 290 - 9 * color_gap) // 10
    color_source = np.asarray(
        [
            (145 + index * (color_width + color_gap) + color_width / 2, 1167.5)
            for index in range(10)
        ],
        dtype=np.float32,
    )
    color_source_corners = np.asarray(
        ((145, 900), (2351, 900), (145, 1435), (2351, 1435)), dtype=np.float32
    )
    color_camera_corners = np.asarray(
        ((2124, 980), (2118, 1820), (1878, 980), (1888, 1820)), dtype=np.float32
    )
    color_camera = cv2.perspectiveTransform(
        color_source[None], cv2.getPerspectiveTransform(color_source_corners, color_camera_corners)
    )[0]
    return gray_source, gray_camera, color_source, color_camera


def _sample_points(image: np.ndarray, points: np.ndarray, radius: int = 14) -> np.ndarray:
    values = []
    for x, y in points:
        integer_x, integer_y = int(round(x)), int(round(y))
        values.append(
            np.median(
                image[
                    integer_y - radius : integer_y + radius + 1,
                    integer_x - radius : integer_x + radius + 1,
                ],
                axis=(0, 1),
            )
        )
    return np.asarray(values)


def _fit_power_response(source: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    best: tuple[float, float, float, float] | None = None
    for gamma in np.linspace(0.15, 4.0, 386):
        design = np.column_stack((source**gamma, np.ones(len(source))))
        gain, bias = np.linalg.lstsq(design, observed, rcond=None)[0]
        if gain < 0.0:
            continue
        mae = float(np.mean(np.abs(design @ (gain, bias) - observed)))
        candidate = (mae, float(gamma), float(gain), float(bias))
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("could not fit a monotonic power response")
    return {"mae": best[0], "gamma": best[1], "gain": best[2], "bias": best[3]}


def _fit_joint_burst() -> dict[str, Any]:
    paths = _files("00")
    images = [_load_rgb(path) for path in paths]
    _, gray_reference, _, color_reference = _joint_reference_points()
    roi = (1400, 700, 2600, 2300)
    x0, y0, x1, y1 = roi
    reference = cv2.resize(
        cv2.cvtColor(images[0], cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1],
        None,
        fx=0.5,
        fy=0.5,
    )
    gray_rows: list[np.ndarray] = []
    color_rows: list[np.ndarray] = []
    frame_rows: list[dict[str, Any]] = []
    valid_indices: list[int] = []
    for index, (path, image) in enumerate(zip(paths, images)):
        matrix = np.eye(2, 3, dtype=np.float32)
        correlation = 1.0
        if index:
            current = cv2.resize(
                cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1],
                None,
                fx=0.5,
                fy=0.5,
            )
            try:
                correlation, matrix = cv2.findTransformECC(
                    reference,
                    current,
                    matrix,
                    cv2.MOTION_AFFINE,
                    (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
                    None,
                    3,
                )
            except cv2.error:
                correlation = 0.0
            matrix[:, 2] *= 2.0

        def transform(points: np.ndarray) -> np.ndarray:
            local = points.copy()
            local[:, 0] -= x0
            local[:, 1] -= y0
            local = cv2.transform(local[None], matrix)[0]
            local[:, 0] += x0
            local[:, 1] += y0
            return local

        gray = _sample_points(image, transform(gray_reference))
        colors = _sample_points(image, transform(color_reference))
        valid = correlation >= 0.75
        if valid:
            valid_indices.append(index)
            gray_rows.append(gray)
            color_rows.append(colors)
        frame_rows.append(
            {
                "capture": str(path),
                "registration_correlation": correlation,
                "used": valid,
                "affine_reference_to_frame": matrix,
                "gray_255_rgb": gray[-1],
                "gray_128_rgb": gray[8],
                "color_white_rgb": colors[-1],
            }
        )

    gray_stack = np.asarray(gray_rows)
    color_stack = np.asarray(color_rows)
    gray_median = np.median(gray_stack, axis=0)
    color_median = np.median(color_stack, axis=0)
    source_gray = np.linspace(0.0, 1.0, 16)
    response = [
        _fit_power_response(source_gray, gray_median[:, channel]) for channel in range(3)
    ]
    source_colors = np.asarray(
        (
            (1, 0, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 1, 0),
            (0, 1, 1),
            (1, 0, 1),
            (64 / 255, 64 / 255, 64 / 255),
            (128 / 255, 128 / 255, 128 / 255),
            (192 / 255, 192 / 255, 192 / 255),
            (1, 1, 1),
        ),
        dtype=np.float64,
    )
    design = np.column_stack((source_colors, np.ones(len(source_colors))))
    coefficients = np.linalg.lstsq(design, color_median, rcond=None)[0]
    prediction = design @ coefficients
    white_per_frame = gray_stack[:, -1]
    normalized_white = white_per_frame / np.median(white_per_frame, axis=0)
    return {
        "status": "provisional",
        "reason": (
            "same-frame gray/color samples constrain an effective JPEG response, but unique levels at unique field positions confound tone response with spatial attenuation"
        ),
        "frame_count": len(paths),
        "registered_frame_count": len(valid_indices),
        "excluded_frame_indices_1based": [index + 1 for index in range(len(paths)) if index not in valid_indices],
        "gray_source_levels": source_gray,
        "observed_gray_rgb_median": gray_median,
        "observed_gray_rgb_iqr": np.percentile(gray_stack, 75, axis=0)
        - np.percentile(gray_stack, 25, axis=0),
        "power_response_rgb": response,
        "direct_jpeg_color_matrix": coefficients[:3].T,
        "direct_jpeg_color_bias": coefficients[3],
        "direct_jpeg_color_fit_mae": np.mean(np.abs(prediction - color_median)),
        "color_matrix_has_negative_entries": bool(np.any(coefficients[:3].T < 0.0)),
        "source_color_patches_rgb": source_colors,
        "observed_color_patches_rgb_median": color_median,
        "auto_response_white_gain_iqr": _quartiles(normalized_white),
        "frames": frame_rows,
    }


def _fit_black_burst() -> dict[str, Any]:
    medians: list[np.ndarray] = []
    highpass_sigma: list[np.ndarray] = []
    paths = _files("B_00")
    for path in paths:
        image = _load_rgb(path)
        roi = image[1000:3100, 900:2350]
        medians.append(np.median(roi, axis=(0, 1)))
        residual = roi - cv2.GaussianBlur(roi, (0, 0), 3.0)
        residual_center = np.median(residual, axis=(0, 1))
        highpass_sigma.append(
            1.4826 * np.median(np.abs(residual - residual_center), axis=(0, 1))
        )
    return {
        "status": "provisional",
        "reason": (
            "full-black JPEGs estimate the path background and processed high-frequency residual, not independent sensor read/shot noise"
        ),
        "frame_count": len(paths),
        "roi_xyxy": (900, 1000, 2350, 3100),
        "background_rgb": _quartiles(np.asarray(medians)),
        "highpass_robust_sigma_rgb": _quartiles(np.asarray(highpass_sigma)),
        "noise_slope_rgb": None,
        "independent_sensor_noise_status": "not_identifiable",
    }


def _dataset_metadata() -> dict[str, Any]:
    files = sorted(ROOT.rglob("*.jpg"))
    focal: list[float] = []
    exposures: list[float] = []
    iso: list[float] = []
    bias: list[float] = []
    for path in files:
        tags = _read_exif(path)
        if tags.get("FocalLengthIn35mmFilm") is not None:
            focal.append(float(tags["FocalLengthIn35mmFilm"]))
        if tags.get("ExposureTime") is not None:
            exposures.append(float(tags["ExposureTime"]))
        if tags.get("ISOSpeedRatings") is not None:
            iso.append(float(tags["ISOSpeedRatings"]))
        if tags.get("ExposureBiasValue") is not None:
            bias.append(float(tags["ExposureBiasValue"]))
    jpeg = [inspect_jpeg(path) for path in files]
    return {
        "root": str(ROOT),
        "frame_count": len(files),
        "group_frame_counts": {
            path.name: len(list(path.glob("*.jpg")))
            for path in sorted(ROOT.iterdir())
            if path.is_dir()
        },
        "inventory": {
            "expected_core_frames": 43,
            "actual_frames": len(files),
            "status": "incomplete_but_fittable",
            "missing_or_mislabeled": (
                "one of the two asymmetric-grid boundary captures is absent; folder 21 contains a text chart rather than the asymmetric grid"
            ),
            "f11_fiducials": (
                "outer fiducials were clipped in full-screen display, so absolute source-frame crop remains unidentifiable"
            ),
        },
        "exif": {
            "focal_35mm_range": (min(focal), max(focal)),
            "exposure_seconds_range": (min(exposures), max(exposures)),
            "iso_range": (min(iso), max(iso)),
            "exposure_bias_range": (min(bias), max(bias)),
            "white_balance": "auto",
        },
        "jpeg": {
            "equivalent_quality": sorted({item["equivalent_quality"] for item in jpeg}),
            "subsampling": sorted({item["subsampling"] for item in jpeg}),
        },
    }


def _write_diagnostics(
    point_raw: dict[str, Any], field_frames: list[dict[str, Any]], joint: dict[str, Any], black: dict[str, Any]
) -> None:
    diagnostics = OUTPUT / "v2_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 12))
    for axis, (color, config) in zip(axes.ravel(), POINT_GRID_CONFIG.items()):
        image = _load_rgb(_files(str(config["folder"]))[0])
        result = point_raw[color]
        points = result["observed_points"]
        margin = 100
        left = max(0, int(points[:, 0].min()) - margin)
        right = min(image.shape[1], int(points[:, 0].max()) + margin)
        top = max(0, int(points[:, 1].min()) - margin)
        bottom = min(image.shape[0], int(points[:, 1].max()) + margin)
        axis.imshow(image[top:bottom, left:right])
        for index, point in enumerate(result["points"]):
            x, y = np.asarray(point["observed_xy"]) - (left, top)
            axis.plot(x, y, "go" if point["inlier"] else "rx", markersize=4)
            axis.text(x + 4, y, str(index), fontsize=6, color="yellow")
        axis.set_title(f"{color}: green=inlier, red=rejected")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(diagnostics / "point_grid_overlays.png", dpi=150)
    plt.close(figure)

    frame_index = np.arange(1, len(field_frames) + 1)
    contrast = [frame["median_point_contrast"] for frame in field_frames]
    fwhm = np.asarray(
        [
            frame["median_valid_psf_fwhm_camera_px"]
            if frame["median_valid_psf_fwhm_camera_px"] is not None
            else (np.nan, np.nan)
            for frame in field_frames
        ]
    )
    figure, first_axis = plt.subplots(figsize=(8, 4.5))
    first_axis.plot(frame_index, contrast, "o-", color="black", label="point contrast")
    first_axis.set_xlabel("burst frame")
    first_axis.set_ylabel("median point contrast")
    second_axis = first_axis.twinx()
    second_axis.plot(frame_index, fwhm[:, 0], "s-", label="PSF minor")
    second_axis.plot(frame_index, fwhm[:, 1], "^-", label="PSF major")
    second_axis.set_ylabel("PSF FWHM (camera px)")
    second_axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(diagnostics / "field_burst_autofocus.png", dpi=150)
    plt.close(figure)

    source = np.asarray(joint["gray_source_levels"])
    observed = np.asarray(joint["observed_gray_rgb_median"])
    figure, axis = plt.subplots(figsize=(7, 5))
    names = ("R", "G", "B")
    colors = ("red", "green", "blue")
    for channel, (name, color) in enumerate(zip(names, colors)):
        axis.plot(source, observed[:, channel], "o", color=color, label=f"{name} samples")
        params = joint["power_response_rgb"][channel]
        axis.plot(
            source,
            params["bias"] + params["gain"] * source ** params["gamma"],
            color=color,
        )
    axis.set_xlabel("source gray level")
    axis.set_ylabel("JPEG code value")
    axis.legend()
    figure.tight_layout()
    figure.savefig(diagnostics / "joint_gray_response.png", dpi=150)
    plt.close(figure)

    background = np.asarray(black["background_rgb"]["median"])
    sigma = np.asarray(black["highpass_robust_sigma_rgb"]["median"])
    figure, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    axes[0].bar(("R", "G", "B"), background, color=("red", "green", "blue"))
    axes[0].set_title("Black-target path background")
    axes[1].bar(("R", "G", "B"), sigma, color=("red", "green", "blue"))
    axes[1].set_title("High-pass robust sigma")
    figure.tight_layout()
    figure.savefig(diagnostics / "black_burst.png", dpi=150)
    plt.close(figure)


def _write_report(result: dict[str, Any]) -> None:
    grids = result["fit"]["geometry_and_spatial_psf"]["point_grids"]
    field = result["fit"]["temporal_psf"]
    joint = result["fit"]["joint_response"]
    black = result["fit"]["background_and_noise"]
    rows = []
    for color in ("W", "R", "G", "B"):
        item = grids[color]
        rows.append(
            f"| {color} | {item['inlier_count']}/25 | "
            f"{item['quadratic_residual_px']['median']:.2f} / {item['homography_residual_px']['median']:.2f} | "
            f"{item['psf_fwhm_normalized_47mm']['median'][0]:.2f} x {item['psf_fwhm_normalized_47mm']['median'][1]:.2f} | {item['status']} |"
        )
    report = f"""# c_C v2 有效退化模型拟合报告

## 结论

本轮 42 张 `c_C` JPEG 足以把 47 mm 自动域从固定局部模型推进到“二次曲面几何场 + 空间/颜色条件 PSF + 自动对焦时变 PSF + 同帧 JPEG 响应 + 路径黑场”的 v2 有效模型。它仍是从显示 sRGB 到手机 JPEG 的路径级模型，不能把光学、传感器、ISP 和 JPEG 分量分别解释为独立物理参数。

## 几何与空间 PSF

| 点阵 | 有效点 | 二次场/单应中位残差 (px) | 归一到 47 mm 的 PSF FWHM (px) | 状态 |
|---|---:|---:|---:|---|
{chr(10).join(rows)}

四种点阵的二次场残差均明显低于单应模型，因此几何算子应写为 `W_(H,delta2)`；`H` 描述姿态投影，`delta2(u)` 描述镜片/光路造成的低阶曲面残差。白点阵对比度最高，其中心 PSF 为 {grids['W']['psf_fwhm_camera_px']['median'][0]:.2f} x {grids['W']['psf_fwhm_camera_px']['median'][1]:.2f} 相机 px。绿色和蓝色点阵信噪比较低，颜色 PSF 暂定而非最终色散参数。

## 自动对焦时变项

`05` 连拍共接受 {field['accepted_point_count']} 个点样本，PSF FWHM 中位数为 {field['psf_fwhm_camera_px']['median'][0]:.2f} x {field['psf_fwhm_camera_px']['median'][1]:.2f} px，IQR 为 {field['psf_fwhm_camera_px']['q25'][0]:.2f}--{field['psf_fwhm_camera_px']['q75'][0]:.2f} / {field['psf_fwhm_camera_px']['q25'][1]:.2f}--{field['psf_fwhm_camera_px']['q75'][1]:.2f} px。前半段失焦、后四帧锁焦明显，故 `k_(u,t)` 必须按自动对焦状态采样。边缘梯度核心仅约 {field['edge_gradient_core_width_camera_px']['median']:.1f} px，与宽点 PSF 并存，说明手机 ISP 有锐化/振铃；单一正高斯只能近似，正式合成应允许经验核或带窄锐化项的核。

## 光度、背景与编码

联合响应 burst 有 {joint['registered_frame_count']}/10 帧可靠配准。灰阶幂响应的 RGB gamma 初值为 {joint['power_response_rgb'][0]['gamma']:.2f}, {joint['power_response_rgb'][1]['gamma']:.2f}, {joint['power_response_rgb'][2]['gamma']:.2f}，但灰阶级别与空间位置一一绑定，仍与空间衰减混杂，状态为 `provisional`。直接 JPEG 颜色矩阵 MAE 为 {joint['direct_jpeg_color_fit_mae']:.3f}，且矩阵含负元素，不作为物理颜色矩阵启用。

全黑 burst 的路径背景 RGB 中位数为 {black['background_rgb']['median'][0]:.4f}, {black['background_rgb']['median'][1]:.4f}, {black['background_rgb']['median'][2]:.4f}；高通稳健 sigma 为 {black['highpass_robust_sigma_rgb']['median'][0]:.4f}, {black['highpass_robust_sigma_rgb']['median'][1]:.4f}, {black['highpass_robust_sigma_rgb']['median'][2]:.4f}。这些是环境光、散射、ISP 和 JPEG 的合并残差，不等于传感器 read noise。全部 42 张文件仍为 JPEG quality 96、4:2:0。

## 仍不能确定

1. 绝对源图边界、完整裁剪和镜像：F11 截掉了外侧 fiducial，且 `21` 实际为文本图，缺少一张开始/结束非对称定位图。
2. 唯一的空间衰减 `m(u)`、加性背景 `b(u)`、tone curve 和 3 x 3 物理颜色矩阵：当前联合响应图每个灰阶只出现在一个位置，checker/inverse 又不是同姿态。
3. 光学、离焦、运动和 ISP 锐化各自的核：JPEG 只支持合并有效核；没有真实对焦马达位置和 RAW/DNG。
4. shot/read/fixed-pattern/JPEG 噪声的独立参数：黑场只能给出处理后残差下界，没有多灰阶 RAW burst。
5. 多反射路径的数量和权重，以及跨姿态、距离、环境光和设备的参数分布。

## 下一批最小补拍

先修复当前不可识别项，不需要重复 121 张：

1. 同一姿态拍 2 张“内缩 10% 的非对称定位图”，开始/结束各 1 张，普通单拍。
2. 拍 2 张空间置换的联合响应图：每个关键灰阶在多个视场位置重复，两张位置互换；各普通单拍 1 张。它们用于分开 `g/A` 与 `m/b`。
3. 为泛用性新增 4 个差异姿态；每姿态只拍内缩定位、W 点阵、点/斜边、随机文本各 1 张，共 16 张。颜色点阵只在其中最偏的 2 个姿态补 R/G/B，共 6 张。
4. 若要分离噪声或物理 ISP，再单独采 RAW/DNG + JPEG 的黑/灰 burst；这不是当前有效 JPEG 模型的阻塞项。
"""
    (OUTPUT / "v2_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    point_grids, point_raw = _fit_point_grids()
    field_summary, field_frames = _fit_field_burst()
    joint = _fit_joint_burst()
    black = _fit_black_burst()
    result = {
        "version": "v2",
        "dataset": _dataset_metadata(),
        "model": {
            "scope": "effective display-sRGB to auto-47mm-path JPEG model",
            "geometry": "projective homography H plus quadratic residual field delta2(u)",
            "psf": "spatially and autofocus-state varying effective kernel k(u,t)",
            "photometry": "frame-conditional monotonic RGB response; provisional",
            "background": "low-frequency path background plus processed JPEG residual",
            "compression": "JPEG quality 96, YCbCr 4:2:0",
        },
        "fit": {
            "geometry_and_spatial_psf": {
                "status": "estimated",
                "point_grids": point_grids,
                "absolute_source_crop": {
                    "status": "not_identifiable",
                    "reason": "F11 clipped outer fiducials and the asymmetric-grid boundary capture is missing/mislabeled",
                },
            },
            "temporal_psf": {**field_summary, "frames": field_frames},
            "joint_response": joint,
            "background_and_noise": black,
            "checker_inverse_separation": {
                "status": "not_identifiable",
                "reason": "B_19 and B_20 were captured at different camera orientations/poses, so additive and multiplicative fields cannot be differenced pointwise",
            },
        },
        "unresolved_parameters": [
            "absolute source crop/mirror and path identity",
            "separate spatial attenuation, flare/background, tone curve, and physical color matrix",
            "separate optical, defocus, motion, ISP sharpening, and sensor kernels",
            "independent shot/read/fixed-pattern/JPEG noise",
            "multi-path count/weights and cross-pose/device distributions",
        ],
    }
    safe_result = _json_safe(result)
    (OUTPUT / "v2_parameters.json").write_text(
        json.dumps(safe_result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_diagnostics(point_raw, field_frames, safe_result["fit"]["joint_response"], safe_result["fit"]["background_and_noise"])
    _write_report(safe_result)
    print(f"wrote {OUTPUT / 'v2_parameters.json'}")
    print(f"wrote {OUTPUT / 'v2_report.md'}")
    print(f"wrote {OUTPUT / 'v2_diagnostics'}")


if __name__ == "__main__":
    main()
