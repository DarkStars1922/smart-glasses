from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.degradation.calibration_c import (
    fit_permuted_photometry,
    fit_point_grid_capture,
    refine_repeated_patch_grid,
)
from analysis.degradation.fitting import _read_exif, inspect_jpeg


ROOT = Path("images/real/c_follow")
PATTERN_ROOT = Path("images/origin/calibration_v2_followup")
OUTPUT = Path("analysis/results")
DIAGNOSTICS = OUTPUT / "v2_1_diagnostics"
SOURCE_SIZE = (2500, 1600)
GRID_SOURCE_CORNERS = np.asarray(
    ((430.0, 290.0), (2069.0, 290.0), (430.0, 1309.0), (2069.0, 1309.0)),
    dtype=np.float32,
)

POINT_CONFIG = {
    "G": {
        "path": ROOT / "03/IMG_20260722_140837.jpg",
        "channel_index": 1,
        "anchors": ((1477, 927), (2031, 986), (1426, 1757), (1960, 1810)),
    },
    "B": {
        "path": ROOT / "04/IMG_20260722_141807.jpg",
        "channel_index": 2,
        "anchors": ((1905, 1160), (2437, 1189), (1837, 1918), (2355, 1974)),
    },
}

# Centers are read from the five visible shapes, not from an image boundary.
LARGE_LOCATOR_SOURCE = np.asarray(
    (
        (318.6551724, 230.6724138),
        (2179.0, 230.0),
        (296.6666667, 1392.3333333),
        (2177.5276236, 1367.9455810),
    ),
    dtype=np.float32,
)
LARGE_LOCATOR_CENTER = np.asarray((1250.2240854, 812.5259146), dtype=np.float32)
LOCATOR_CONFIG = (
    {
        "path": ROOT / "00/IMG_20260722_140443.jpg",
        "observed_corners": ((1747.2, 637.5), (1548.6, 1442.8), (1233.8, 467.5), (1065.9, 1294.1)),
        "observed_center": (1377.6, 995.0),
    },
    {
        "path": ROOT / "00/IMG_20260722_141727.jpg",
        "observed_corners": ((1089.5, 503.2), (1840.1, 636.7), (1013.8, 1003.5), (1760.1, 1117.4)),
        "observed_center": (1457.8, 834.0),
    },
)

SMALL_LOCATOR_SOURCE = np.asarray(
    ((305.0, 215.0), (2194.0, 215.0), (286.6666667, 1402.3333333), (2194.0, 1384.0)),
    dtype=np.float32,
)
RESPONSE_CONFIG = (
    {
        "name": "A",
        "path": ROOT / "01/IMG_20260722_140540.jpg",
        "manifest_key": "response_A",
        "observed_markers": ((1882.0, 1562.0), (1886.0, 2269.0), (1467.0, 1377.0), (1370.0, 2185.0)),
    },
    {
        "name": "B",
        "path": ROOT / "02/IMG_20260722_140612.jpg",
        "manifest_key": "response_B",
        "observed_markers": ((1753.0, 801.0), (1638.0, 1564.0), (1253.0, 690.0), (1160.0, 1471.0)),
    },
)


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


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


def _quartiles(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "q25": np.percentile(array, 25, axis=0),
        "median": np.median(array, axis=0),
        "q75": np.percentile(array, 75, axis=0),
    }


def _fit_absolute_locators() -> dict[str, Any]:
    frame = np.asarray(
        ((0.0, 0.0), (2499.0, 0.0), (2499.0, 1599.0), (0.0, 1599.0)),
        dtype=np.float32,
    )
    rows = []
    for config in LOCATOR_CONFIG:
        observed = np.asarray(config["observed_corners"], dtype=np.float32)
        homography = cv2.getPerspectiveTransform(LARGE_LOCATOR_SOURCE, observed)
        projected_frame = cv2.perspectiveTransform(frame[None], homography)[0]
        predicted_center = cv2.perspectiveTransform(
            LARGE_LOCATOR_CENTER.reshape(1, 1, 2), homography
        )[0, 0]
        center_residual = float(
            np.linalg.norm(predicted_center - np.asarray(config["observed_center"]))
        )
        center_probe = np.asarray(
            ((1250.0, 800.0), (1251.0, 800.0), (1250.0, 801.0)),
            dtype=np.float32,
        )
        projected_probe = cv2.perspectiveTransform(center_probe[None], homography)[0]
        jacobian = np.column_stack(
            (projected_probe[1] - projected_probe[0], projected_probe[2] - projected_probe[0])
        )
        width, height = Image.open(config["path"]).size
        inside = bool(
            np.all((projected_frame[:, 0] >= 0) & (projected_frame[:, 0] < width))
            and np.all((projected_frame[:, 1] >= 0) & (projected_frame[:, 1] < height))
        )
        rows.append(
            {
                "capture": str(config["path"]),
                "image_size": (width, height),
                "homography_source_to_camera": homography,
                "projected_full_source_corners_camera_xy": projected_frame,
                "full_source_inside_camera_frame": inside,
                "orientation_jacobian_determinant": np.linalg.det(jacobian),
                "mirrored": bool(np.linalg.det(jacobian) < 0.0),
                "center_validation_residual_px": center_residual,
                "center_validation_observed_xy": config["observed_center"],
                "center_validation_predicted_xy": predicted_center,
            }
        )
    return {
        "status": "estimated",
        "reason": (
            "all four inset asymmetric fiducials are visible in each capture; the central cross is held out as a low-order distortion check"
        ),
        "frame_count": len(rows),
        "full_source_boundary_resolved": all(row["full_source_inside_camera_frame"] for row in rows),
        "mirror_resolved": True,
        "same_pose_drift_status": "not_applicable",
        "same_pose_drift_reason": "the two locator captures have different camera orientation and pose",
        "frames": rows,
    }


def _summarize_point_result(result: dict[str, Any]) -> dict[str, Any]:
    points = [point for point in result["points"] if point["inlier"]]
    fwhm = np.asarray([point["psf_fwhm_minor_major_angle"][:2] for point in points])
    contrast = np.asarray([point["contrast"] for point in points])
    scales = np.asarray([point["scale_singular_values"] for point in points])
    return {
        "inlier_count": len(points),
        "rejected_indices": np.flatnonzero(~result["warp"]["inlier_mask"]),
        "quadratic_coefficients_normalized_source_to_camera": result["warp"]["coefficients"],
        "homography_source_to_camera": result["warp"]["homography"],
        "quadratic_residual_px": {
            "median": result["warp"]["quadratic_median_residual_px"],
            "p95": result["warp"]["quadratic_p95_residual_px"],
        },
        "homography_residual_px": {
            "median": result["warp"]["homography_median_residual_px"],
            "p95": result["warp"]["homography_p95_residual_px"],
        },
        "local_scale_camera_per_source": _quartiles(scales),
        "point_contrast": _quartiles(contrast),
        "psf_fwhm_camera_px": _quartiles(fwhm),
    }


def _fit_large_color_points() -> tuple[dict[str, Any], dict[str, dict[int, dict[str, Any]]]]:
    baseline = json.loads((OUTPUT / "v2_parameters.json").read_text(encoding="utf-8"))
    summaries: dict[str, Any] = {}
    raw: dict[str, dict[int, dict[str, Any]]] = {}
    for color, config in POINT_CONFIG.items():
        image = _load_rgb(config["path"])
        radius_results: dict[int, dict[str, Any]] = {}
        radius_summaries: dict[int, dict[str, Any]] = {}
        for radius in (18, 22, 26):
            result = fit_point_grid_capture(
                image,
                source_size=SOURCE_SIZE,
                anchor_corners=np.asarray(config["anchors"]),
                channel_index=int(config["channel_index"]),
                square_side_px=31,
                search_radius=60,
                measurement_radius=radius,
            )
            radius_results[radius] = result
            radius_summaries[radius] = _summarize_point_result(result)
        primary = radius_summaries[22]
        fwhm_across = np.asarray(
            [radius_summaries[radius]["psf_fwhm_camera_px"]["median"] for radius in (18, 22, 26)]
        )
        old = baseline["fit"]["geometry_and_spatial_psf"]["point_grids"][color]
        summaries[color] = {
            "capture": str(config["path"]),
            "source_square_side_px": 31,
            "reported_focal_mm": 47,
            "geometry_status": "estimated" if primary["inlier_count"] >= 23 else "provisional",
            "psf_status": "provisional",
            "psf_reason": (
                "visibility is now sufficient, but finite-square moment deconvolution remains sensitive to the 18/22/26 px measurement window"
            ),
            "primary_measurement_radius_px": 22,
            **primary,
            "measurement_window_sensitivity": {
                "radii_px": (18, 22, 26),
                "median_psf_fwhm_camera_px_by_radius": fwhm_across,
                "minor_range": (float(fwhm_across[:, 0].min()), float(fwhm_across[:, 0].max())),
                "major_range": (float(fwhm_across[:, 1].min()), float(fwhm_across[:, 1].max())),
            },
            "improvement_over_v2_15px": {
                "old_median_contrast": old["point_contrast"]["median"],
                "new_median_contrast": primary["point_contrast"]["median"],
                "old_status": old["status"],
            },
        }
        raw[color] = radius_results
    return summaries, raw


def _fit_response(manifest: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    observed_rows = []
    source_rgb_rows = []
    source_xy_rows = []
    frame_rows = []
    refinements = []
    for frame_index, config in enumerate(RESPONSE_CONFIG):
        rows = manifest[config["manifest_key"]]
        centers = np.asarray([row["center_xy"] for row in rows], dtype=np.float32)
        marker_homography = cv2.getPerspectiveTransform(
            SMALL_LOCATOR_SOURCE,
            np.asarray(config["observed_markers"], dtype=np.float32),
        )
        initial_grid = cv2.perspectiveTransform(
            GRID_SOURCE_CORNERS[None], marker_homography
        )[0]
        refinement = refine_repeated_patch_grid(
            _load_rgb(config["path"]),
            source_centers=centers,
            patch_ids=[row["patch_id"] for row in rows],
            source_grid_corners=GRID_SOURCE_CORNERS,
            initial_camera_corners=initial_grid,
        )
        refinements.append(
            {
                "name": config["name"],
                "capture": str(config["path"]),
                **refinement,
            }
        )
        observed_rows.append(refinement["observed_rgb"])
        source_rgb_rows.extend(
            np.asarray([row["source_rgb"] for row in rows], dtype=np.float64) / 255.0
        )
        source_xy_rows.extend(centers)
        frame_rows.extend([frame_index] * len(rows))

    fitted = fit_permuted_photometry(
        np.vstack(observed_rows),
        np.asarray(source_rgb_rows),
        np.asarray(source_xy_rows),
        np.asarray(frame_rows),
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    spatial = np.asarray(fitted["spatial_multiplier_rgb"])
    frames = np.asarray(frame_rows)
    spatial_ranges = []
    for frame_index in range(len(RESPONSE_CONFIG)):
        spatial_ranges.append(
            {
                "q05": np.percentile(spatial[frames == frame_index], 5, axis=0),
                "median": np.median(spatial[frames == frame_index], axis=0),
                "q95": np.percentile(spatial[frames == frame_index], 95, axis=0),
            }
        )
    fitted.update(
        {
            "status": "provisional",
            "channel_status": {
                "R": {
                    "tone": "not_identifiable",
                    "spatial_attenuation": "not_identifiable",
                    "reason": "red-path response is close to the fitted background and the unconstrained gray response is non-monotonic",
                },
                "G": {"tone": "provisional", "spatial_attenuation": "provisional"},
                "B": {"tone": "provisional", "spatial_attenuation": "provisional"},
            },
            "effective_color_matrix_status": "not_identifiable",
            "effective_color_matrix_reason": (
                "the nonnegative row-stochastic candidate cannot overcome the failed red response and leaves a high color-patch residual"
            ),
            "spatial_multiplier_percentiles_by_frame": spatial_ranges,
        }
    )
    return fitted, refinements


def _dataset_metadata() -> dict[str, Any]:
    files = sorted(ROOT.glob("*/*.jpg"))
    rows = []
    for path in files:
        tags = _read_exif(path)
        jpeg = inspect_jpeg(path)
        rows.append(
            {
                "path": str(path),
                "image_size": Image.open(path).size,
                "focal_35mm": (
                    float(tags["FocalLengthIn35mmFilm"])
                    if tags.get("FocalLengthIn35mmFilm") is not None
                    else None
                ),
                "exposure_seconds": (
                    float(tags["ExposureTime"])
                    if tags.get("ExposureTime") is not None
                    else None
                ),
                "iso": (
                    float(tags["ISOSpeedRatings"])
                    if tags.get("ISOSpeedRatings") is not None
                    else None
                ),
                "exposure_bias": (
                    float(tags["ExposureBiasValue"])
                    if tags.get("ExposureBiasValue") is not None
                    else None
                ),
                "jpeg_quality": jpeg["equivalent_quality"],
                "jpeg_subsampling": jpeg["subsampling"],
            }
        )
    return {"root": str(ROOT), "frame_count": len(files), "frames": rows}


def _write_diagnostics(
    point_raw: dict[str, dict[int, dict[str, Any]]],
    response_refinements: list[dict[str, Any]],
    photometry: dict[str, Any],
) -> None:
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    for axis, color in zip(axes, ("G", "B")):
        result = point_raw[color][22]
        image = _load_rgb(POINT_CONFIG[color]["path"])
        points = result["observed_points"]
        left, right = int(points[:, 0].min()) - 80, int(points[:, 0].max()) + 80
        top, bottom = int(points[:, 1].min()) - 80, int(points[:, 1].max()) + 80
        axis.imshow(image[top:bottom, left:right])
        for point in result["points"]:
            x, y = np.asarray(point["observed_xy"]) - (left, top)
            axis.plot(x, y, "go" if point["inlier"] else "rx", markersize=4)
            axis.text(x + 3, y, str(point["index"]), color="yellow", fontsize=6)
        axis.set_title(f"{color}, 31 px source squares")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "large_GB_point_grids.png", dpi=160)
    plt.close(figure)

    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(12, 7))
    for axis, config, refinement in zip(axes, RESPONSE_CONFIG, response_refinements):
        image = _load_rgb(config["path"])
        points = np.asarray(refinement["observed_centers"])
        left, right = int(points[:, 0].min()) - 80, int(points[:, 0].max()) + 80
        top, bottom = int(points[:, 1].min()) - 80, int(points[:, 1].max()) + 80
        axis.imshow(image[top:bottom, left:right])
        rows = manifest[config["manifest_key"]]
        for row, (x, y) in zip(rows, points):
            color = "red" if row["patch_id"] == "K000" else "yellow"
            axis.plot(x - left, y - top, "o", markerfacecolor="none", markeredgecolor=color, markersize=4)
        axis.set_title(f"response {config['name']}: refined centers")
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "response_grid_refinement.png", dpi=160)
    plt.close(figure)

    levels = np.asarray(photometry["gray_levels"])
    tone = np.asarray(photometry["tone_values_rgb"])
    raw = np.asarray(photometry["raw_tone_values_rgb"])
    figure, axis = plt.subplots(figsize=(7, 5))
    for channel, (name, color) in enumerate(zip(("R", "G", "B"), ("red", "green", "blue"))):
        axis.plot(levels, raw[channel], "x--", color=color, alpha=0.45)
        axis.plot(levels, tone[channel], "o-", color=color, label=name)
    axis.set_xlabel("source gray")
    axis.set_ylabel("normalized effective response")
    axis.legend()
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "permuted_tone_response.png", dpi=160)
    plt.close(figure)


def _write_report(result: dict[str, Any]) -> None:
    locators = result["fit"]["absolute_geometry"]
    points = result["fit"]["large_color_point_grids"]
    response = result["fit"]["permuted_photometry"]
    equation = r"""\[
\widehat Y_t(\mathbf u)=\mathcal J_{96,4:2:0}\!\left\{\operatorname{clip}\!\left[b_t(\mathbf u)+g_t\odot m_t(\mathbf u)\odot f\!\left(A\,\mathcal D_s\left[k_{\mathbf u,t}*\mathcal W_{H_t,\delta_2}(X)\right]\right)+\varepsilon_t(\mathbf u)\right]\right\}.
\]"""
    report = f"""# c_follow v2.1 有效退化模型拟合报告

## 本轮结论

`c_follow` 的 6 张图已经进入实际拟合。内缩定位图解决了 F11 截角造成的绝对边界缺失：两张图的四个非对称标记均可见，完整 2500 x 1600 源图边界投影都位于相机画面内，方向可判定；两张图姿态和横竖方向不同，故不把它们的差值解释为同姿态漂移。中心十字作为未参与单应拟合的检查点，残差分别为 {locators['frames'][0]['center_validation_residual_px']:.1f} 和 {locators['frames'][1]['center_validation_residual_px']:.1f} px，说明绝对单应可确定，但仍需与点阵拟合的二次残差场组合使用。

31 x 31 的 G/B 方点显著提高了可见度：G、B 中位对比分别为 {points['G']['point_contrast']['median']:.3f}、{points['B']['point_contrast']['median']:.3f}。B 有 {points['B']['inlier_count']}/25 个二次场内点；G 有 {points['G']['inlier_count']}/25 个，缺失主要集中在受反射干扰的最上行。22 px 测量窗口下的有限方点反卷积 PSF 中位数为 G {points['G']['psf_fwhm_camera_px']['median'][0]:.2f} x {points['G']['psf_fwhm_camera_px']['median'][1]:.2f} px、B {points['B']['psf_fwhm_camera_px']['median'][0]:.2f} x {points['B']['psf_fwhm_camera_px']['median'][1]:.2f} px。18/22/26 px 窗口之间的 minor 轴仍明显变化，因此几何场可用，G/B PSF 数值继续标为 `provisional`。

## 光度解耦结果

置换响应图经重复 patch 类内一致性精修后，灰阶设计矩阵秩为 {response['gray_design_rank']}/{response['gray_design_columns']}，条件数 {response['gray_design_condition']:.1f}。采用的规范化是 `f_c(0)=0`、`f_c(1)=1`、每帧样本上的 `mean(log m_tc)=0`，并令候选颜色矩阵每行非负且和为 1。背景在每帧仅有 4 个黑块，故只能拟合仿射场，不能支持二次背景场。

G/B 的无约束灰阶响应保持单调，可得到暂定 tone curve 与逐帧二次 `log m`；R 的无约束响应不单调，且红通道信号接近背景，R 的 tone 和空间衰减均判为 `not_identifiable`。联合拟合灰阶 MAE 为 {response['gray_fit_mae']:.3f}，颜色块 MAE 为 {response['color_fit_mae']:.3f}；后者仍过高，所以 JSON 中保留非负颜色矩阵候选值用于诊断，但完整 3 x 3 有效颜色矩阵不启用。

## 当前整体模型

当前 47 mm 自动域的路径级模型写为

{equation}

其中绝对姿态 `H_t`、内部二次几何场 `delta_2`、W/R PSF、自动对焦时变有效核、JPEG 96/4:2:0 已有可用估计；G/B 空间 PSF、G/B tone、逐帧空间衰减和仿射背景是暂定项；R tone/R 空间衰减、完整颜色矩阵、独立物理噪声与核分解仍未可靠确定。
"""
    (OUTPUT / "v2_1_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    absolute = _fit_absolute_locators()
    points, point_raw = _fit_large_color_points()
    response, response_refinements = _fit_response(manifest)
    result = {
        "version": "v2.1",
        "dataset": _dataset_metadata(),
        "model_equation": (
            "Yhat_t(u)=J_96,420{clip[b_t(u)+g_t*m_t(u)*f(A D_s[k_(u,t)*W_(H_t,delta2)(X)])+epsilon_t(u)]}"
        ),
        "fit": {
            "absolute_geometry": absolute,
            "large_color_point_grids": points,
            "permuted_photometry": response,
            "response_grid_refinement": response_refinements,
        },
        "parameter_status": {
            "forward_operator_with_monotonic_tone_curve": "implemented",
            "absolute_source_boundary_and_orientation": "estimated",
            "interior_quadratic_geometry": "estimated_in_v2",
            "W_R_effective_psf": "estimated_in_v2",
            "G_B_effective_psf": "provisional",
            "autofocus_temporal_psf": "provisional_in_v2",
            "per_frame_affine_background": "provisional",
            "G_B_tone_and_spatial_attenuation": "provisional",
            "R_tone_and_spatial_attenuation": "not_identifiable",
            "complete_effective_color_matrix": "not_identifiable",
            "independent_physical_kernel_and_noise_terms": "not_identifiable",
        },
        "unresolved_parameters": [
            "red-channel monotonic response and spatial attenuation in the selected cyan-biased path",
            "a reliable full 3 x 3 effective color matrix",
            "quadratic or higher-order additive background beyond the fitted per-frame affine field",
            "measurement-window-stable G/B effective PSF moments",
            "separate optical, defocus, motion, ISP, sensor-noise, and JPEG components",
        ],
    }
    safe = _json_safe(result)
    (OUTPUT / "v2_1_parameters.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_diagnostics(point_raw, response_refinements, safe["fit"]["permuted_photometry"])
    _write_report(safe)
    print(f"wrote {OUTPUT / 'v2_1_parameters.json'}")
    print(f"wrote {OUTPUT / 'v2_1_report.md'}")
    print(f"wrote {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
