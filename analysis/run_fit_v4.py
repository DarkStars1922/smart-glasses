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
    apply_anchor_tone_curve,
    apply_trilinear_color_lut,
    fit_black_background_field,
    fit_anchor_tone_normalization,
    fit_constrained_linear_color_matrix,
    fit_joint_color_nodes,
    refine_repeated_patch_grid,
)
from analysis.degradation.fitting import _read_exif, inspect_jpeg


ROOT = Path("images/real/c_color")
PATTERN_ROOT = Path("images/origin/calibration_v4_color_operator")
OUTPUT = Path("analysis/results")
DIAGNOSTICS = OUTPUT / "v4_diagnostics"
SOURCE_SIZE = (2500, 1600)
CAPTURE_ROLES = (
    "black_start",
    "train_A",
    "train_B",
    "train_C",
    "holdout",
    "black_end",
)
SMALL_LOCATOR_SOURCE = np.asarray(
    ((305.0, 215.0), (2194.0, 215.0), (286.6666667, 1402.3333333), (2194.0, 1384.0)),
    dtype=np.float32,
)
GRID_SOURCE_CORNERS = np.asarray(
    ((430.0, 290.0), (2069.0, 290.0), (430.0, 1309.0), (2069.0, 1309.0)),
    dtype=np.float32,
)

# Full-resolution camera coordinates, in TL/TR/BL/BR source-marker order.
SMALL_MARKERS = {
    "train_A": np.asarray(
        ((1538.0, 663.0), (2229.2, 721.5), (1496.7, 1122.6), (2175.2, 1170.1)),
        dtype=np.float32,
    ),
    "train_B": np.asarray(
        ((987.7, 120.0), (1695.5, 215.5), (946.9, 595.7), (1641.2, 674.3)),
        dtype=np.float32,
    ),
    "train_C": np.asarray(
        ((1061.9, 236.7), (1668.3, 305.4), (1024.4, 641.1), (1626.6, 701.8)),
        dtype=np.float32,
    ),
    "holdout": np.asarray(
        ((1314.1, 104.5), (1983.7, 160.4), (1284.2, 559.7), (1937.7, 607.8)),
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
        raise ValueError(f"c_color must contain exactly {len(CAPTURE_ROLES)} JPEG files")
    return dict(zip(CAPTURE_ROLES, files))


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _chart_rows(manifest: dict[str, Any], role: str) -> list[dict[str, Any]]:
    if role.startswith("train_"):
        return manifest["training_charts"][role[-1]]["assignments"]
    if role == "holdout":
        return manifest["holdout_chart"]["assignments"]
    raise ValueError(f"no chart rows for {role}")


def _sample_chart(
    capture: Path,
    role: str,
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    centers = np.asarray([row["center_xy"] for row in rows], dtype=np.float32)
    marker_homography = cv2.getPerspectiveTransform(SMALL_LOCATOR_SOURCE, SMALL_MARKERS[role])
    initial_grid = cv2.perspectiveTransform(GRID_SOURCE_CORNERS[None], marker_homography)[0]
    refinement = refine_repeated_patch_grid(
        _load_rgb(capture),
        source_centers=centers,
        patch_ids=[row["patch_id"] for row in rows],
        source_grid_corners=GRID_SOURCE_CORNERS,
        initial_camera_corners=initial_grid,
        max_adjustment_px=70.0,
        sampling_radius=9,
    )
    refinement.update({"role": role, "capture": str(capture)})
    source_rgb = np.asarray([row["source_rgb"] for row in rows], dtype=np.float64) / 255.0
    return (
        refinement,
        np.asarray(refinement["observed_rgb"]),
        source_rgb,
        centers,
        [row["patch_id"] for row in rows],
    )


def _fit_training_and_holdout(
    captures: dict[str, Path], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    refinements = []
    observed_rows = []
    source_rows = []
    position_rows = []
    patch_ids: list[str] = []
    frame_rows = []
    training_roles = ("train_A", "train_B", "train_C")
    for frame_index, role in enumerate(training_roles):
        sampled = _sample_chart(captures[role], role, _chart_rows(manifest, role))
        refinement, observed, source, positions, labels = sampled
        refinements.append(refinement)
        observed_rows.append(observed)
        source_rows.append(source)
        position_rows.append(positions)
        patch_ids.extend(labels)
        frame_rows.extend([frame_index] * len(labels))
    training = fit_joint_color_nodes(
        np.vstack(observed_rows),
        np.vstack(source_rows),
        np.vstack(position_rows),
        patch_ids,
        np.asarray(frame_rows),
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    training.update(
        {
            "status": "estimated_from_three_permuted_charts",
            "frame_count": 3,
            "frame_roles": training_roles,
            "sample_count": int(sum(len(rows) for rows in observed_rows)),
        }
    )

    role = "holdout"
    refinement, observed, source, positions, labels = _sample_chart(
        captures[role], role, _chart_rows(manifest, role)
    )
    refinements.append(refinement)
    holdout = fit_joint_color_nodes(
        observed,
        source,
        positions,
        labels,
        np.zeros(len(labels), dtype=np.int64),
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    holdout.update(
        {
            "status": "independently_normalized_validation_chart",
            "frame_count": 1,
            "frame_roles": (role,),
            "sample_count": len(observed),
            "used_for_training": False,
            "validation_color_count": int(sum(node["patch_id"].startswith("V_") for node in holdout["nodes"])),
        }
    )
    return training, holdout, refinements


def _tone_normalized_samples(
    fit: dict[str, Any],
    refinements: list[dict[str, Any]],
    manifest: dict[str, Any],
    roles: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    refinement_by_role = {row["role"]: row for row in refinements}
    observed = np.vstack([refinement_by_role[role]["observed_rgb"] for role in roles])
    labels: list[str] = []
    source_rows = []
    frame_rows = []
    for frame_index, role in enumerate(roles):
        rows = _chart_rows(manifest, role)
        labels.extend(row["patch_id"] for row in rows)
        source_rows.extend(np.asarray([row["source_rgb"] for row in rows]) / 255.0)
        frame_rows.extend([frame_index] * len(rows))
    frames = np.asarray(frame_rows)
    gain = np.asarray(fit["frame_gain_rgb"])
    spatial = np.asarray(fit["spatial_multiplier_rgb"])
    background = np.asarray(fit["background_rgb"])
    normalized = np.maximum((observed - background) / (gain[frames] * spatial), 0.0)
    tone = fit_anchor_tone_normalization(normalized, labels, frames)
    latent = np.asarray(tone.pop("latent_rgb"))
    tone.pop("reconstruction_rgb")
    tone.update(
        {
            "status": "estimated_from_K128_and_K255_only",
            "frame_roles": roles,
            "color_patches_used": False,
            "frame_gain_after_white_anchor_rgb": gain
            * np.asarray(tone["frame_white_scale_rgb"]),
        }
    )
    return tone, {
        "observed_rgb": observed,
        "source_rgb": np.asarray(source_rows),
        "patch_ids": np.asarray(labels),
        "frame_indices": frames,
        "latent_rgb": latent,
        "background_rgb": background,
        "spatial_multiplier_rgb": spatial,
    }


def _aggregate_latent_nodes(samples: dict[str, Any]) -> list[dict[str, Any]]:
    labels = np.asarray(samples["patch_ids"])
    source = np.asarray(samples["source_rgb"])
    latent = np.asarray(samples["latent_rgb"])
    nodes = []
    for label in dict.fromkeys(str(value) for value in labels):
        rows = labels == label
        response = np.median(latent[rows], axis=0)
        if label == "K000":
            response = np.zeros(3)
        elif label == "K128":
            response = np.full(3, 0.5)
        elif label == "K255":
            response = np.ones(3)
        nodes.append(
            {
                "patch_id": label,
                "source_rgb": source[np.flatnonzero(rows)[0]],
                "response_rgb": response,
                "sample_count": int(np.count_nonzero(rows)),
            }
        )
    return nodes


def _fit_color_operator(
    training: dict[str, Any],
    holdout: dict[str, Any],
    manifest: dict[str, Any],
    refinements: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    training_tone, training_samples = _tone_normalized_samples(
        training, refinements, manifest, ("train_A", "train_B", "train_C")
    )
    holdout_tone, holdout_samples = _tone_normalized_samples(
        holdout, refinements, manifest, ("holdout",)
    )
    levels = np.asarray(manifest["training_levels_rgb"], dtype=np.float64) / 255.0
    training_nodes = {
        node["patch_id"]: node for node in _aggregate_latent_nodes(training_samples)
    }
    lut = np.empty((3, 3, 3, 3), dtype=np.float64)
    all_training_nodes = list(training_nodes.values())
    for node in all_training_nodes:
        indices = tuple(int(np.argmin(np.abs(levels - value))) for value in node["source_rgb"])
        if not np.allclose(levels[list(indices)], node["source_rgb"], atol=1e-8):
            raise ValueError(f"training node {node['patch_id']} is off the LUT grid")
        lut[indices] = node["response_rgb"]

    training_source = np.asarray([node["source_rgb"] for node in all_training_nodes])
    training_response = np.asarray([node["response_rgb"] for node in all_training_nodes])
    linear = fit_constrained_linear_color_matrix(training_source, training_response, ridge=1e-8)

    validation_nodes = [
        node
        for node in _aggregate_latent_nodes(holdout_samples)
        if node["patch_id"].startswith("V_")
    ]
    validation_source = np.asarray([node["source_rgb"] for node in validation_nodes])
    validation_response = np.asarray([node["response_rgb"] for node in validation_nodes])
    lut_prediction = apply_trilinear_color_lut(validation_source, levels=levels, lut_rgb=lut)
    linear_prediction = validation_source @ np.asarray(linear["color_matrix"]).T
    lut_residual = lut_prediction - validation_response
    linear_residual = linear_prediction - validation_response
    lut_response_mae = float(np.mean(np.abs(lut_residual)))
    linear_response_mae = float(np.mean(np.abs(linear_residual)))

    sample_labels = np.asarray(holdout_samples["patch_ids"])
    validation_mask = np.char.startswith(sample_labels.astype(str), "V_")
    sample_source = np.asarray(holdout_samples["source_rgb"])[validation_mask]
    sample_frames = np.asarray(holdout_samples["frame_indices"])[validation_mask]
    sample_lut_latent = apply_trilinear_color_lut(sample_source, levels=levels, lut_rgb=lut)
    sample_linear_latent = sample_source @ np.asarray(linear["color_matrix"]).T
    midpoint = np.asarray(holdout_tone["frame_midpoint_response_rgb"])
    lut_tone_response = apply_anchor_tone_curve(
        sample_lut_latent,
        midpoint_response_rgb=midpoint,
        frame_indices=sample_frames,
    )
    linear_tone_response = apply_anchor_tone_curve(
        sample_linear_latent,
        midpoint_response_rgb=midpoint,
        frame_indices=sample_frames,
    )
    adjusted_gain = np.asarray(holdout_tone["frame_gain_after_white_anchor_rgb"])
    background = np.asarray(holdout_samples["background_rgb"])[validation_mask]
    spatial = np.asarray(holdout_samples["spatial_multiplier_rgb"])[validation_mask]
    observed = np.asarray(holdout_samples["observed_rgb"])[validation_mask]
    lut_raw_prediction = background + adjusted_gain[sample_frames] * spatial * lut_tone_response
    linear_raw_prediction = (
        background + adjusted_gain[sample_frames] * spatial * linear_tone_response
    )
    lut_raw_residual = lut_raw_prediction - observed
    linear_raw_residual = linear_raw_prediction - observed
    lut_raw_mae = float(np.mean(np.abs(lut_raw_residual)))
    linear_raw_mae = float(np.mean(np.abs(linear_raw_residual)))
    selected = (
        "trilinear_3x3x3_lut"
        if lut_raw_mae < linear_raw_mae
        else "constrained_linear_matrix"
    )
    selected_mae = min(lut_raw_mae, linear_raw_mae)
    operator = {
        "status": "selected_by_independent_holdout",
        "selected_model": selected,
        "selected_holdout_jpeg_mae": selected_mae,
        "selection_rule": "lower JPEG-domain mean absolute RGB error on the 24 V_ holdout colors",
        "training_levels_rgb": levels,
        "lut_shape": list(lut.shape),
        "lut_rgb": lut,
        "lut_response_range": (float(np.min(lut)), float(np.max(lut))),
        "linear_color_matrix": linear["color_matrix"],
        "linear_row_sums": np.asarray(linear["color_matrix"]).sum(axis=1),
        "linear_has_negative_entries": bool(
            np.any(np.asarray(linear["color_matrix"]) < 0.0)
        ),
        "linear_training_node_mae": linear["mae"],
        "holdout_response_mae_lut": lut_response_mae,
        "holdout_response_mae_linear": linear_response_mae,
        "holdout_jpeg_mae_lut": lut_raw_mae,
        "holdout_jpeg_mae_linear": linear_raw_mae,
        "holdout_jpeg_mae_rgb_lut": np.mean(np.abs(lut_raw_residual), axis=0),
        "holdout_jpeg_mae_rgb_linear": np.mean(np.abs(linear_raw_residual), axis=0),
        "holdout_jpeg_improvement_fraction_linear_over_lut": (
            (lut_raw_mae - linear_raw_mae) / lut_raw_mae if lut_raw_mae > 0.0 else 0.0
        ),
        "holdout_source_rgb": validation_source,
        "holdout_measured_response_rgb": validation_response,
        "holdout_lut_prediction_rgb": lut_prediction,
        "holdout_linear_prediction_rgb": linear_prediction,
        "training_node_ids": sorted(training_nodes),
    }
    return operator, {"training": training_tone, "holdout": holdout_tone}


def _sample_rgb(image: np.ndarray, points: np.ndarray, radius: int = 7) -> np.ndarray:
    samples = []
    for x, y in points:
        ix, iy = int(round(float(x))), int(round(float(y)))
        patch = image[iy - radius : iy + radius + 1, ix - radius : ix + radius + 1]
        if patch.shape[:2] != (2 * radius + 1, 2 * radius + 1):
            raise ValueError("background sample intersects image boundary")
        samples.append(np.median(patch, axis=(0, 1)))
    return np.asarray(samples)


def _fit_background(
    captures: dict[str, Path], refinements: list[dict[str, Any]]
) -> dict[str, Any]:
    refinement_by_role = {row["role"]: row for row in refinements}
    x = np.linspace(500.0, 2000.0, 9)
    y = np.linspace(350.0, 1250.0, 7)
    source_points = np.asarray([(column, row) for row in y for column in x], dtype=np.float32)
    observed_rows = []
    positions = []
    frames = []
    pairs = (("black_start", "train_A"), ("black_end", "holdout"))
    for frame_index, (black_role, chart_role) in enumerate(pairs):
        homography = np.asarray(refinement_by_role[chart_role]["homography_source_to_camera"])
        projected = cv2.perspectiveTransform(source_points[None], homography)[0]
        observed_rows.append(_sample_rgb(_load_rgb(captures[black_role]), projected))
        positions.append(source_points)
        frames.extend([frame_index] * len(source_points))
    fit = fit_black_background_field(
        np.vstack(observed_rows),
        np.vstack(positions),
        np.asarray(frames),
        source_size=SOURCE_SIZE,
        ridge=1e-4,
    )
    fit.update(
        {
            "status": "estimated_at_sequence_endpoints",
            "frame_count": 2,
            "samples_per_frame": len(source_points),
            "captures": [str(captures[pair[0]]) for pair in pairs],
            "center_background_rgb": np.asarray(fit["coefficients"])[:, :, 0],
            "center_background_drift_rgb": np.asarray(fit["coefficients"])[1, :, 0]
            - np.asarray(fit["coefficients"])[0, :, 0],
        }
    )
    return fit


def _dataset_metadata(captures: dict[str, Path]) -> dict[str, Any]:
    rows = []
    for index, (role, path) in enumerate(captures.items()):
        tags = _read_exif(path)
        jpeg = inspect_jpeg(path)
        with Image.open(path) as image:
            image_size = image.size
        rows.append(
            {
                "index": index,
                "role": role,
                "path": str(path),
                "image_size": image_size,
                "focal_35mm": float(tags["FocalLengthIn35mmFilm"])
                if tags.get("FocalLengthIn35mmFilm") is not None
                else None,
                "exposure_seconds": float(tags["ExposureTime"])
                if tags.get("ExposureTime") is not None
                else None,
                "iso": float(tags["ISOSpeedRatings"])
                if tags.get("ISOSpeedRatings") is not None
                else None,
                "jpeg_quality": jpeg["equivalent_quality"],
                "jpeg_subsampling": jpeg["subsampling"],
            }
        )
    return {
        "root": str(ROOT),
        "frame_count": len(rows),
        "ordering": "file modification time",
        "frames": rows,
    }


def _write_diagnostics(
    refinements: list[dict[str, Any]], operator: dict[str, Any]
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
    figure.savefig(DIAGNOSTICS / "color_chart_registration.png", dpi=150)
    plt.close(figure)

    measured = np.asarray(operator["holdout_measured_response_rgb"])
    lut_prediction = np.asarray(operator["holdout_lut_prediction_rgb"])
    linear_prediction = np.asarray(operator["holdout_linear_prediction_rgb"])
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    for channel, (axis, name, color) in enumerate(
        zip(axes, ("R", "G", "B"), ("red", "green", "blue"))
    ):
        axis.scatter(measured[:, channel], linear_prediction[:, channel], marker="x", color="gray", label="3x3")
        axis.scatter(measured[:, channel], lut_prediction[:, channel], facecolors="none", edgecolors=color, label="LUT")
        axis.plot((0, 1), (0, 1), "k--", linewidth=1)
        axis.set(xlabel="measured", ylabel="predicted", title=name, xlim=(0, 1.05), ylim=(0, 1.05))
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(DIAGNOSTICS / "holdout_color_validation.png", dpi=150)
    plt.close(figure)


def _write_report(result: dict[str, Any]) -> None:
    training = result["fit"]["training_node_fit"]
    holdout = result["fit"]["holdout_validation"]
    operator = result["fit"]["color_operator"]
    background = result["fit"]["temporal_quadratic_background"]
    report = f"""# c_color v4 联合颜色算子拟合报告

## 结论

3 张置换训练色卡共同估计 27 个 RGB 网格节点；联合设计矩阵秩为 {training['design_rank']}/{training['design_columns']}，训练观测 MAE 为 {training['fit_mae']:.4f}。第 4 张色卡只用于验证，没有参与参数拟合；其 24 个内部颜色构成独立留出集。

每帧仅用黑/中灰/白锚点估计 ISP tone 状态，不使用留出颜色真值。完成该归一化后，留出集的 JPEG 域 MAE 为：3x3 线性颜色矩阵 {operator['holdout_jpeg_mae_linear']:.4f}，3x3x3 三线性 LUT {operator['holdout_jpeg_mae_lut']:.4f}。按预先固定的“留出 MAE 较低”规则，当前选择 `{operator['selected_model']}`，其留出 MAE 为 {operator['selected_holdout_jpeg_mae']:.4f}。

首尾黑场二次背景拟合 MAE 为 {background['mae']:.4f}，中心背景漂移 RGB 为 {background['center_background_drift_rgb']}。它仍属于整条 OLED-镜片-相机 ISP-JPEG 路径的有效模型，不解释为任一物理部件的独立响应。

## 当前颜色退化项

颜色项写成 `b_t(u) + g_t * m_t(u) * h_t(C(X))`。其中 `C` 由本次留出验证选出的颜色算子实现，`h_t` 是黑/中灰/白锚定的逐帧单调 tone；逐帧增益 `g_t`、逐帧二次空间场 `m_t` 和黑电平 `b_t` 在拟合时联合解耦。留出色卡只用共享锚点估计这些干扰项，因此 24 个验证颜色没有泄漏进训练。

## 边界

本批数据完成的是单设备、单拍摄设置下的有效颜色算子。跨姿态、跨环境照明和跨设备的参数分布仍需多组完整序列估计；这不影响当前序列内的模型闭合。
"""
    (OUTPUT / "v4_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    captures = _ordered_captures()
    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    training, holdout, refinements = _fit_training_and_holdout(captures, manifest)
    operator, anchor_tone = _fit_color_operator(
        training, holdout, manifest, refinements
    )
    background = _fit_background(captures, refinements)
    selected = operator["selected_model"]
    result = {
        "version": "v4",
        "dataset": _dataset_metadata(captures),
        "model_equation": (
            "Z_t(u)=D_s[k_(u,t)*W_(H_t,delta2)(X)]; "
            "Yhat_t=J_96,420{clip[b_t(u)+g_t*m_t(u)*h_t(C(Z_t(u)))+epsilon_t(u)]}"
        ),
        "fit": {
            "training_node_fit": training,
            "holdout_validation": holdout,
            "color_operator": operator,
            "anchor_tone_normalization": anchor_tone,
            "response_grid_refinement": refinements,
            "temporal_quadratic_background": background,
        },
        "parameter_status": {
            "effective_color_operator": f"{selected}_selected_by_independent_holdout",
            "R_G_B_spatial_attenuation": "estimated_per_calibration_frame",
            "frame_dependent_anchor_tone": "estimated_per_calibration_frame",
            "temporal_quadratic_additive_background": "estimated_at_two_endpoints",
            "cross_pose_environment_device_distribution": "not_identifiable_from_one_sequence",
        },
        "unresolved_parameters": [
            "cross-pose, cross-lighting, and cross-device parameter distributions",
            "separate optical, display, sensor, ISP, and JPEG color transforms",
            "frame-level temporal interpolation of the endpoint background fields",
        ],
    }
    safe = _json_safe(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "v4_parameters.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(safe)
    _write_diagnostics(refinements, operator)
    print(f"training design rank: {training['design_rank']}/{training['design_columns']}")
    print(f"holdout JPEG LUT MAE: {operator['holdout_jpeg_mae_lut']:.6f}")
    print(f"holdout JPEG linear MAE: {operator['holdout_jpeg_mae_linear']:.6f}")
    print(f"selected: {selected}")
    print(f"wrote {OUTPUT / 'v4_parameters.json'}")


if __name__ == "__main__":
    main()
