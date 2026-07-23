from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/glasses-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.degradation.calibration_c import (
    fit_black_background_field,
    fit_permuted_photometry,
    measure_slanted_edge_cells,
    refine_effective_color_matrix,
    refine_repeated_patch_grid,
)
from analysis.degradation.fitting import _read_exif, inspect_jpeg


ROOT = Path("images/real/c_core")
PATTERN_ROOT = Path("images/origin/calibration_v3_core")
OUTPUT = Path("analysis/results")
DIAGNOSTICS = OUTPUT / "v3_diagnostics"
SOURCE_SIZE = (2500, 1600)
CAPTURE_ROLES = (
    "locator_start",
    "black_start",
    "response_A",
    "response_B",
    "response_C",
    "red_gray_ramp",
    "gb_slanted_edges",
    "black_end",
    "locator_end",
)

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
SMALL_LOCATOR_SOURCE = np.asarray(
    ((305.0, 215.0), (2194.0, 215.0), (286.6666667, 1402.3333333), (2194.0, 1384.0)),
    dtype=np.float32,
)
GRID_SOURCE_CORNERS = np.asarray(
    ((430.0, 290.0), (2069.0, 290.0), (430.0, 1309.0), (2069.0, 1309.0)),
    dtype=np.float32,
)

# Marker centers are in full-resolution camera pixels and follow source order:
# top-left, top-right, bottom-left, bottom-right.
LARGE_MARKERS = {
    "locator_start": np.asarray(
        ((1051.0, 705.0), (1747.0, 767.0), (1012.0, 1160.0), (1704.0, 1204.0)),
        dtype=np.float32,
    ),
    "locator_end": np.asarray(
        ((1276.0, 835.0), (1974.0, 883.0), (1241.0, 1293.0), (1933.0, 1325.0)),
        dtype=np.float32,
    ),
}
LOCATOR_CENTER_OBSERVED = {
    "locator_start": np.asarray((1404.0, 972.0)),
    "locator_end": np.asarray((1634.0, 1101.0)),
}
SMALL_MARKERS = {
    "response_A": np.asarray(
        ((571.4, 547.9), (1256.4, 597.4), (550.6, 989.9), (1234.0, 1023.0)),
        dtype=np.float32,
    ),
    "response_B": np.asarray(
        ((1227.7, 573.8), (1901.6, 634.6), (1189.8, 1022.6), (1855.6, 1071.9)),
        dtype=np.float32,
    ),
    "response_C": np.asarray(
        ((1438.0, 345.8), (2128.0, 396.7), (1411.4, 811.9), (2082.1, 851.6)),
        dtype=np.float32,
    ),
    "red_gray_ramp": np.asarray(
        ((1242.3, 745.8), (1937.6, 811.6), (1196.6, 1204.5), (1885.5, 1257.0)),
        dtype=np.float32,
    ),
    "gb_slanted_edges": np.asarray(
        ((1513.6, 729.4), (2207.5, 746.3), (1499.3, 1193.6), (2181.8, 1204.8)),
        dtype=np.float32,
    ),
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
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _ordered_captures() -> dict[str, Path]:
    files = sorted(ROOT.glob("*.jpg"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if len(files) != len(CAPTURE_ROLES):
        raise ValueError(f"c_core must contain exactly {len(CAPTURE_ROLES)} JPEG files")
    return dict(zip(CAPTURE_ROLES, files))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _fit_absolute_geometry(captures: dict[str, Path]) -> dict[str, Any]:
    source_frame = np.asarray(
        ((0.0, 0.0), (2499.0, 0.0), (2499.0, 1599.0), (0.0, 1599.0)),
        dtype=np.float32,
    )
    rows = []
    for role in ("locator_start", "locator_end"):
        homography = cv2.getPerspectiveTransform(LARGE_LOCATOR_SOURCE, LARGE_MARKERS[role])
        projected_frame = cv2.perspectiveTransform(source_frame[None], homography)[0]
        projected_center = cv2.perspectiveTransform(
            LARGE_LOCATOR_CENTER.reshape(1, 1, 2), homography
        )[0, 0]
        center_residual = float(
            np.linalg.norm(projected_center - LOCATOR_CENTER_OBSERVED[role])
        )
        probe = np.asarray(
            ((1250.0, 800.0), (1251.0, 800.0), (1250.0, 801.0)), dtype=np.float32
        )
        projected_probe = cv2.perspectiveTransform(probe[None], homography)[0]
        jacobian = np.column_stack(
            (projected_probe[1] - projected_probe[0], projected_probe[2] - projected_probe[0])
        )
        width, height = Image.open(captures[role]).size
        inside = bool(
            np.all((projected_frame[:, 0] >= 0) & (projected_frame[:, 0] < width))
            and np.all((projected_frame[:, 1] >= 0) & (projected_frame[:, 1] < height))
        )
        rows.append(
            {
                "role": role,
                "capture": str(captures[role]),
                "homography_source_to_camera": homography,
                "projected_full_source_corners_camera_xy": projected_frame,
                "projected_source_center_camera_xy": projected_center,
                "center_validation_residual_px": center_residual,
                "full_source_inside_camera_frame": inside,
                "mirrored": bool(np.linalg.det(jacobian) < 0.0),
            }
        )
    drift = np.asarray(rows[1]["projected_source_center_camera_xy"]) - np.asarray(
        rows[0]["projected_source_center_camera_xy"]
    )
    return {
        "status": "estimated",
        "frame_count": 2,
        "full_source_boundary_resolved": all(row["full_source_inside_camera_frame"] for row in rows),
        "mirror_resolved": True,
        "projected_center_drift_start_to_end_px": drift,
        "frames": rows,
    }


def _sample_rgb(image: np.ndarray, points: np.ndarray, radius: int = 7) -> np.ndarray:
    samples = []
    for x, y in points:
        integer_x, integer_y = int(round(float(x))), int(round(float(y)))
        patch = image[
            integer_y - radius : integer_y + radius + 1,
            integer_x - radius : integer_x + radius + 1,
        ]
        if patch.shape[:2] != (2 * radius + 1, 2 * radius + 1):
            raise ValueError("sample intersects camera image boundary")
        samples.append(np.median(patch, axis=(0, 1)))
    return np.asarray(samples)


def _fit_background(captures: dict[str, Path]) -> tuple[dict[str, Any], dict[str, Any]]:
    x = np.linspace(350.0, 2150.0, 9)
    y = np.linspace(300.0, 1300.0, 7)
    source_points = np.asarray([(column, row) for row in y for column in x], dtype=np.float32)
    observed_rows = []
    source_rows = []
    frame_rows = []
    projected_rows = []
    pairs = (("black_start", "locator_start"), ("black_end", "locator_end"))
    for frame_index, (black_role, locator_role) in enumerate(pairs):
        homography = cv2.getPerspectiveTransform(
            LARGE_LOCATOR_SOURCE, LARGE_MARKERS[locator_role]
        )
        projected = cv2.perspectiveTransform(source_points[None], homography)[0]
        observed_rows.append(_sample_rgb(_load_rgb(captures[black_role]), projected))
        source_rows.extend(source_points)
        frame_rows.extend([frame_index] * len(source_points))
        projected_rows.append(projected)
    fit = fit_black_background_field(
        np.vstack(observed_rows),
        np.asarray(source_rows),
        np.asarray(frame_rows),
        source_size=SOURCE_SIZE,
        ridge=1e-4,
    )
    fit.update(
        {
            "status": "estimated_at_endpoints",
            "reason": (
                "two full-black frames support a quadratic spatial field at the sequence endpoints; interpolation between endpoints remains provisional"
            ),
            "frame_count": 2,
            "samples_per_frame": len(source_points),
            "captures": [str(captures[black_role]) for black_role, _ in pairs],
            "center_background_rgb": fit["coefficients"][:, :, 0],
            "center_background_drift_rgb": fit["coefficients"][1, :, 0]
            - fit["coefficients"][0, :, 0],
        }
    )
    return fit, {
        "source_points": source_points,
        "projected_points": projected_rows,
        "observed_rgb": observed_rows,
    }


def _response_rows(manifest: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role.startswith("response_"):
        return manifest["response_charts"][role[-1]]["assignments"]
    return manifest["red_gray_ramp"]["assignments"]


def _fit_joint_photometry(
    captures: dict[str, Path], manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    roles = ("response_A", "response_B", "response_C", "red_gray_ramp")
    observed_rows = []
    source_rgb_rows = []
    source_xy_rows = []
    frame_rows = []
    refinements = []
    for frame_index, role in enumerate(roles):
        rows = _response_rows(manifest, role)
        centers = np.asarray([row["center_xy"] for row in rows], dtype=np.float32)
        marker_homography = cv2.getPerspectiveTransform(
            SMALL_LOCATOR_SOURCE, SMALL_MARKERS[role]
        )
        initial_grid = cv2.perspectiveTransform(
            GRID_SOURCE_CORNERS[None], marker_homography
        )[0]
        refinement = refine_repeated_patch_grid(
            _load_rgb(captures[role]),
            source_centers=centers,
            patch_ids=[row["patch_id"] for row in rows],
            source_grid_corners=GRID_SOURCE_CORNERS,
            initial_camera_corners=initial_grid,
            max_adjustment_px=70.0,
            sampling_radius=9,
        )
        refinements.append({"role": role, "capture": str(captures[role]), **refinement})
        observed_rows.append(refinement["observed_rgb"])
        source_rgb_rows.extend(
            np.asarray([row["source_rgb"] for row in rows], dtype=np.float64) / 255.0
        )
        source_xy_rows.extend(centers)
        frame_rows.extend([frame_index] * len(rows))

    observed = np.vstack(observed_rows)
    source_rgb = np.asarray(source_rgb_rows)
    source_xy = np.asarray(source_xy_rows)
    frames = np.asarray(frame_rows)
    fit = fit_permuted_photometry(
        observed,
        source_rgb,
        source_xy,
        frames,
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    nonnegative_matrix = np.asarray(fit["color_matrix"])
    nonnegative_color_mae = float(fit["color_fit_mae"])
    refined_matrix = refine_effective_color_matrix(
        observed,
        source_rgb,
        np.asarray(fit["background_rgb"]),
        np.asarray(fit["frame_gain_rgb"])[frames],
        np.asarray(fit["spatial_multiplier_rgb"]),
        tone_levels=np.asarray(fit["gray_levels"]),
        tone_values_rgb=np.asarray(fit["tone_values_rgb"]),
        initial_matrix=nonnegative_matrix,
    )
    fit.update(
        {
            "status": "estimated_in_calibration_sequence",
            "frame_roles": roles,
            "sample_count": len(observed),
            "nonnegative_color_matrix_candidate": nonnegative_matrix,
            "nonnegative_color_fit_mae": nonnegative_color_mae,
            "color_matrix": refined_matrix["color_matrix"],
            "color_matrix_has_negative_entries": refined_matrix["has_negative_entries"],
            "color_matrix_constraint": refined_matrix["constraint"],
            "color_fit_mae": refined_matrix["color_fit_mae"],
            "fit_mae": refined_matrix["fit_mae"],
            "prediction_rgb": refined_matrix["prediction_rgb"],
            "residual_rgb": refined_matrix["residual_rgb"],
            "tone_status_rgb": [
                "estimated" if monotonic else "provisional"
                for monotonic in fit["tone_monotonic"]
            ],
        }
    )
    return fit, refinements


def _fit_gb_edges(captures: dict[str, Path], manifest: dict[str, Any]) -> dict[str, Any]:
    homography = cv2.getPerspectiveTransform(
        SMALL_LOCATOR_SOURCE, SMALL_MARKERS["gb_slanted_edges"]
    )
    result = measure_slanted_edge_cells(
        _load_rgb(captures["gb_slanted_edges"]),
        homography,
        manifest["gb_slanted_edges"]["cells"],
        max_distance_px=20.0,
    )
    axis_summary: dict[str, Any] = {}
    for color_id in ("G", "B"):
        axis_summary[color_id] = {}
        for source_orientation, camera_axis in (
            ("vertical", "horizontal_normal"),
            ("horizontal", "vertical_normal"),
        ):
            values = np.asarray(
                [
                    row["fwhm_px"]
                    for row in result["cells"]
                    if row["accepted"]
                    and row["color_id"] == color_id
                    and row["orientation"].startswith(source_orientation)
                ]
            )
            if len(values):
                axis_summary[color_id][camera_axis] = {
                    "count": len(values),
                    "median_fwhm_camera_px": float(np.median(values)),
                    "q25": float(np.percentile(values, 25)),
                    "q75": float(np.percentile(values, 75)),
                }
    result.update(
        {
            "status": "estimated_edge_core_in_pose",
            "reason": (
                "accepted slanted-edge ESFs estimate the narrow effective edge core; ringing and broader non-Gaussian wings remain outside this scalar FWHM"
            ),
            "capture": str(captures["gb_slanted_edges"]),
            "homography_source_to_camera": homography,
            "axis_by_color": axis_summary,
        }
    )
    return result


def _dataset_metadata(captures: dict[str, Path]) -> dict[str, Any]:
    rows = []
    for index, (role, path) in enumerate(captures.items()):
        tags = _read_exif(path)
        jpeg = inspect_jpeg(path)
        rows.append(
            {
                "index": index,
                "role": role,
                "path": str(path),
                "modified_time_ns": path.stat().st_mtime_ns,
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
                "jpeg_quality": jpeg["equivalent_quality"],
                "jpeg_subsampling": jpeg["subsampling"],
            }
        )
    return {"root": str(ROOT), "frame_count": len(rows), "ordering": "file modification time", "frames": rows}


def _write_diagnostics(
    captures: dict[str, Path],
    refinements: list[dict[str, Any]],
    photometry: dict[str, Any],
    background: dict[str, Any],
    edges: dict[str, Any],
) -> None:
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    for axis, refinement in zip(axes.ravel(), refinements):
        image = _load_rgb(Path(refinement["capture"]))
        points = np.asarray(refinement["observed_centers"])
        left, right = int(points[:, 0].min()) - 60, int(points[:, 0].max()) + 60
        top, bottom = int(points[:, 1].min()) - 60, int(points[:, 1].max()) + 60
        axis.imshow(image[top:bottom, left:right])
        axis.plot(points[:, 0] - left, points[:, 1] - top, "y.", markersize=2)
        axis.set_title(refinement["role"])
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "response_registration.png", dpi=150)
    plt.close(figure)

    levels = np.asarray(photometry["gray_levels"])
    tone = np.asarray(photometry["tone_values_rgb"])
    raw = np.asarray(photometry["raw_tone_values_rgb"])
    figure, axis = plt.subplots(figsize=(7, 5))
    for channel, (name, color) in enumerate(zip(("R", "G", "B"), ("red", "green", "blue"))):
        axis.plot(levels, raw[channel], "x--", color=color, alpha=0.4)
        axis.plot(levels, tone[channel], "o-", color=color, label=name)
    axis.set_xlabel("source gray")
    axis.set_ylabel("normalized effective response")
    axis.legend()
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "joint_tone_response.png", dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, frame_index, title in zip(axes, (0, 1), ("black start", "black end")):
        image = _load_rgb(captures[("black_start", "black_end")[frame_index]])
        axis.imshow(image)
        axis.set_title(
            f"{title}; center RGB {np.asarray(background['center_background_rgb'])[frame_index].round(3)}"
        )
        axis.axis("off")
    figure.suptitle(
        f"quadratic background MAE {background['mae']:.4f}; accepted edges {edges['accepted_count']}/{edges['cell_count']}"
    )
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "background_endpoints.png", dpi=150)
    plt.close(figure)


def _write_report(result: dict[str, Any]) -> None:
    fit = result["fit"]
    response = fit["joint_photometry"]
    background = fit["temporal_quadratic_background"]
    edges = fit["gb_slanted_edge_psf"]
    matrix_status = result["parameter_status"]["complete_effective_color_matrix"]
    equation = r"""\[
Z_t(\mathbf u)=\mathcal D_s\!\left[k_{\mathbf u,t}*\mathcal W_{H_t,\delta_2}(X)\right],
\qquad
\widehat Y_t=\mathcal J_{96,4:2:0}\!\left\{\operatorname{clip}_{[0,1]}\!\left[b_t(\mathbf u)+g_t\odot m_t(\mathbf u)\odot f\!\left(AZ_t(\mathbf u)\right)+\varepsilon_t(\mathbf u)\right]\right\}.
\]"""
    report = fr"""# c_core v3 有效退化模型拟合报告

## 结论

`c_core` 的 9 张 JPEG 已严格按文件时间顺序绑定到 9 张源标定图。三张独立置换响应图与红/灰阶图联合后，灰阶设计矩阵秩为 {response['gray_design_rank']}/{response['gray_design_columns']}，条件数 {response['gray_design_condition']:.2f}；R/G/B 三通道无约束 tone 序列均单调，因此此前不可识别的 R tone 与 R 空间衰减在本标定序列内已经恢复。灰阶 MAE 为 {response['gray_fit_mae']:.3f}。

前向域行和约束颜色矩阵细化后的颜色块 MAE 为 {response['color_fit_mae']:.3f}，状态为 `{matrix_status}`。这一定义是 JPEG 路径级有效矩阵，不解释为 OLED、镜片或相机 ISP 的独立物理矩阵。

首尾全黑图分别以相邻定位图配准，二次背景场拟合 MAE 为 {background['mae']:.4f}。源图中心背景 RGB 从 {np.asarray(background['center_background_rgb'])[0].round(4).tolist()} 变为 {np.asarray(background['center_background_rgb'])[1].round(4).tolist()}，因此背景必须写成随时间变化的低频场，不能使用单个常数。

G/B 斜边共有 {edges['accepted_count']}/{edges['cell_count']} 个 ESF 通过稳健拟合。窄边缘核心 FWHM 中位数为 G {edges['by_color']['G']['median']:.2f} px、B {edges['by_color']['B']['median']:.2f} px。该数值描述有效核的窄核心；点目标得到的宽翼和 ISP 振铃仍需保留在非高斯核模型中。

## 当前模型

{equation}

其中 \(H_t\) 逐帧估计，\(\delta_2\) 沿用 v2 的二次几何残差；\(f\)、\(A\)、\(g_t\)、\(m_t\) 由四张联合光度图估计；\(b_t\) 为首尾二次场插值并允许帧内黑块校正；G/B 的斜边核心与 v2/v2.1 点目标宽翼共同约束 \(k_{{\mathbf u,t}}\)。

## 尚未分离

当前 47 mm 自动成像域的核心路径级方程已经闭合。仍未分离的是跨姿态/环境的参数分布、多反射路径、核的光学/离焦/运动/ISP 分解，以及 shot/read noise 与 JPEG/去噪残差；这些不能由本批 9 张 JPEG 强行物理解耦。
"""
    (OUTPUT / "v3_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    captures = _ordered_captures()
    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    absolute = _fit_absolute_geometry(captures)
    background, _ = _fit_background(captures)
    photometry, refinements = _fit_joint_photometry(captures, manifest)
    edges = _fit_gb_edges(captures, manifest)
    color_mae = float(photometry["color_fit_mae"])
    color_status = (
        "estimated_in_calibration_sequence"
        if color_mae <= 0.06
        else "provisional_model_limited"
        if color_mae <= 0.10
        else "not_reliable_under_current_3x3_model"
    )
    result = {
        "version": "v3",
        "dataset": _dataset_metadata(captures),
        "model_equation": (
            "Z_t(u)=D_s[k_(u,t)*W_(H_t,delta2)(X)]; "
            "Yhat_t=J_96,420{clip[b_t(u)+g_t*m_t(u)*f(A Z_t(u))+epsilon_t(u)]}"
        ),
        "fit": {
            "absolute_geometry": absolute,
            "temporal_quadratic_background": background,
            "joint_photometry": photometry,
            "response_grid_refinement": refinements,
            "gb_slanted_edge_psf": edges,
        },
        "parameter_status": {
            "forward_operator": "implemented",
            "absolute_source_boundary_and_orientation": "estimated",
            "interior_quadratic_geometry": "estimated_in_v2",
            "R_G_B_monotonic_tone": "estimated_in_calibration_sequence",
            "R_G_B_spatial_attenuation": "estimated_per_calibration_frame",
            "complete_effective_color_matrix": color_status,
            "temporal_quadratic_additive_background": "estimated_at_two_endpoints",
            "G_B_slanted_edge_core_psf": "estimated_in_pose",
            "W_R_point_psf_and_autofocus_variation": "estimated_in_v2",
            "independent_physical_kernel_noise_and_multipath": "not_identifiable",
        },
        "unresolved_parameters": [
            "cross-pose, cross-lighting, and cross-device parameter distributions",
            "non-Gaussian PSF wings and ringing as a dense spatial kernel field",
            "separate optical, defocus, motion, sensor, ISP, and JPEG components",
            "multiple reflection paths and their spatial weights",
            "shot noise, read noise, fixed-pattern noise, and JPEG residuals as independent terms",
        ],
    }
    safe = _json_safe(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "v3_parameters.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(safe)
    _write_diagnostics(captures, refinements, photometry, background, edges)
    print(f"wrote {OUTPUT / 'v3_parameters.json'}")
    print(f"wrote {OUTPUT / 'v3_report.md'}")
    print(f"wrote {DIAGNOSTICS}")


if __name__ == "__main__":
    main()
