from __future__ import annotations

import json
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
    fit_black_background_field,
    fit_constrained_linear_color_matrix,
    fit_joint_color_nodes,
    refine_repeated_patch_grid,
)
from analysis.degradation.fitting import _read_exif, inspect_jpeg
from analysis.run_fit_v4 import (
    CAPTURE_ROLES,
    GRID_SOURCE_CORNERS,
    SOURCE_SIZE,
    _aggregate_latent_nodes,
    _chart_rows,
    _json_safe,
    _sample_rgb,
    _tone_normalized_samples,
)


PATTERN_ROOT = Path("images/origin/calibration_v4_color_operator")
OUTPUT = Path("analysis/results")
DIAGNOSTICS = OUTPUT / "v5_diagnostics"
SESSION_CONFIG = {
    "baseline": {
        "root": Path("images/real/c_color"),
        "quarter_turns_ccw": 0,
        "sampling_radius": 9,
        "max_adjustment_px": 10.0,
        "initial_grid_corners": None,
    },
    "repeat": {
        "root": Path("images/real/c_color_repeat"),
        "quarter_turns_ccw": 0,
        "sampling_radius": 5,
        "max_adjustment_px": 55.0,
        "initial_grid_corners": {
            "train_A": ((990, 725), (1570, 735), (980, 1120), (1560, 1120)),
            "train_B": ((1040, 540), (1510, 560), (1030, 880), (1510, 900)),
            "train_C": ((1410, 500), (2000, 510), (1400, 860), (1990, 880)),
            "holdout": ((1390, 570), (1920, 580), (1390, 880), (1920, 900)),
        },
    },
    "pose_left": {
        "root": Path("images/real/c_color_pose_left"),
        "quarter_turns_ccw": 0,
        "sampling_radius": 5,
        "max_adjustment_px": 55.0,
        "initial_grid_corners": {
            "train_A": ((1000, 350), (1570, 370), (990, 680), (1590, 660)),
            "train_B": ((1110, 230), (1580, 240), (1120, 550), (1610, 530)),
            "train_C": ((650, 250), (1200, 300), (650, 520), (1220, 550)),
            "holdout": ((850, 380), (1320, 350), (880, 670), (1360, 650)),
        },
    },
    "pose_right": {
        "root": Path("images/real/c_color_pose_right"),
        "quarter_turns_ccw": 1,
        "sampling_radius": 5,
        "max_adjustment_px": 55.0,
        "initial_grid_corners": {
            "train_A": ((1650, 2270), (2200, 2250), (1650, 2620), (2250, 2600)),
            "train_B": ((1550, 1600), (2050, 1580), (1550, 1900), (2070, 1880)),
            "train_C": ((1000, 2000), (1480, 2020), (1000, 2300), (1510, 2250)),
            "holdout": ((1100, 1750), (1650, 1700), (1100, 2020), (1660, 2050)),
        },
    },
}


def _ordered_captures(root: Path) -> dict[str, Path]:
    files = sorted(root.glob("*.jpg"), key=lambda path: (path.stat().st_mtime_ns, path.name))
    if len(files) != len(CAPTURE_ROLES):
        raise ValueError(f"{root} must contain exactly {len(CAPTURE_ROLES)} JPEG files")
    return dict(zip(CAPTURE_ROLES, files))


def _load_rgb(path: Path, quarter_turns_ccw: int) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return np.rot90(rgb, quarter_turns_ccw).copy() if quarter_turns_ccw else rgb


def _quarter_turns_for_role(config: dict[str, Any], role: str) -> int:
    role_overrides = config.get("role_quarter_turns_ccw", {})
    return int(role_overrides.get(role, config["quarter_turns_ccw"]))


def _baseline_grid_corners() -> dict[str, np.ndarray]:
    parameters = json.loads((OUTPUT / "v4_parameters.json").read_text(encoding="utf-8"))
    return {
        row["role"]: np.asarray(row["refined_camera_corners"], dtype=np.float32)
        for row in parameters["fit"]["response_grid_refinement"]
    }


def _sample_session_charts(
    captures: dict[str, Path],
    manifest: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    initial_by_role = config["initial_grid_corners"] or _baseline_grid_corners()
    refinements = []
    for role in ("train_A", "train_B", "train_C", "holdout"):
        rows = _chart_rows(manifest, role)
        turns = _quarter_turns_for_role(config, role)
        refinement = refine_repeated_patch_grid(
            _load_rgb(captures[role], turns),
            source_centers=np.asarray([row["center_xy"] for row in rows], dtype=np.float32),
            patch_ids=[row["patch_id"] for row in rows],
            source_grid_corners=GRID_SOURCE_CORNERS,
            initial_camera_corners=np.asarray(initial_by_role[role], dtype=np.float32),
            max_adjustment_px=float(config["max_adjustment_px"]),
            sampling_radius=int(config["sampling_radius"]),
        )
        refinement.update(
            {
                "role": role,
                "capture": str(captures[role]),
                "orientation_quarter_turns_ccw": turns,
            }
        )
        refinements.append(refinement)
    return refinements


def _fit_node_models(
    refinements: list[dict[str, Any]], manifest: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    by_role = {row["role"]: row for row in refinements}
    observed_rows = []
    source_rows = []
    position_rows = []
    labels: list[str] = []
    frames = []
    for frame_index, role in enumerate(("train_A", "train_B", "train_C")):
        rows = _chart_rows(manifest, role)
        observed_rows.append(np.asarray(by_role[role]["observed_rgb"]))
        source_rows.extend(np.asarray([row["source_rgb"] for row in rows]) / 255.0)
        position_rows.extend(row["center_xy"] for row in rows)
        labels.extend(row["patch_id"] for row in rows)
        frames.extend([frame_index] * len(rows))
    training = fit_joint_color_nodes(
        np.vstack(observed_rows),
        np.asarray(source_rows),
        np.asarray(position_rows),
        labels,
        np.asarray(frames),
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    training.update(
        {
            "status": "estimated_from_three_permuted_charts",
            "frame_count": 3,
            "sample_count": len(labels),
        }
    )

    rows = _chart_rows(manifest, "holdout")
    holdout = fit_joint_color_nodes(
        np.asarray(by_role["holdout"]["observed_rgb"]),
        np.asarray([row["source_rgb"] for row in rows]) / 255.0,
        np.asarray([row["center_xy"] for row in rows]),
        [row["patch_id"] for row in rows],
        np.zeros(len(rows), dtype=np.int64),
        source_size=SOURCE_SIZE,
        spatial_ridge=0.2,
    )
    holdout.update(
        {
            "status": "independently_normalized_validation_chart",
            "frame_count": 1,
            "sample_count": len(rows),
            "used_for_training": False,
            "validation_color_count": 24,
        }
    )
    return training, holdout


def _fit_background(
    captures: dict[str, Path],
    refinements: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    by_role = {row["role"]: row for row in refinements}
    x = np.linspace(500.0, 2000.0, 9)
    y = np.linspace(350.0, 1250.0, 7)
    source_points = np.asarray([(column, row) for row in y for column in x], dtype=np.float32)
    observed_rows = []
    source_rows = []
    frame_rows = []
    pairs = (("black_start", "train_A"), ("black_end", "holdout"))
    for frame_index, (black_role, chart_role) in enumerate(pairs):
        homography = np.asarray(by_role[chart_role]["homography_source_to_camera"])
        projected = cv2.perspectiveTransform(source_points[None], homography)[0]
        observed_rows.append(
            _sample_rgb(
                _load_rgb(
                    captures[black_role],
                    _quarter_turns_for_role(config, black_role),
                ),
                projected,
                radius=5,
            )
        )
        source_rows.append(source_points)
        frame_rows.extend([frame_index] * len(source_points))
    fit = fit_black_background_field(
        np.vstack(observed_rows),
        np.vstack(source_rows),
        np.asarray(frame_rows),
        source_size=SOURCE_SIZE,
        ridge=1e-4,
    )
    fit.update(
        {
            "status": "estimated_at_session_endpoints",
            "frame_count": 2,
            "samples_per_frame": len(source_points),
            "center_background_rgb": np.asarray(fit["coefficients"])[:, :, 0],
            "center_background_drift_rgb": np.asarray(fit["coefficients"])[1, :, 0]
            - np.asarray(fit["coefficients"])[0, :, 0],
        }
    )
    return fit


def _evaluate_matrix(
    matrix: np.ndarray,
    tone: dict[str, Any],
    samples: dict[str, Any],
) -> dict[str, Any]:
    labels = np.asarray(samples["patch_ids"]).astype(str)
    mask = np.char.startswith(labels, "V_")
    source = np.asarray(samples["source_rgb"])[mask]
    frames = np.asarray(samples["frame_indices"])[mask]
    latent = source @ np.asarray(matrix).T
    tone_response = apply_anchor_tone_curve(
        latent,
        midpoint_response_rgb=np.asarray(tone["frame_midpoint_response_rgb"]),
        frame_indices=frames,
    )
    prediction = (
        np.asarray(samples["background_rgb"])[mask]
        + np.asarray(tone["frame_gain_after_white_anchor_rgb"])[frames]
        * np.asarray(samples["spatial_multiplier_rgb"])[mask]
        * tone_response
    )
    observed = np.asarray(samples["observed_rgb"])[mask]
    residual = prediction - observed
    return {
        "jpeg_mae": float(np.mean(np.abs(residual))),
        "jpeg_mae_rgb": np.mean(np.abs(residual), axis=0),
        "jpeg_rmse": float(np.sqrt(np.mean(residual**2))),
        "jpeg_absolute_error_p95": float(np.percentile(np.abs(residual), 95)),
        "sample_count": int(np.count_nonzero(mask)),
    }


def _fit_session(
    session_id: str,
    config: dict[str, Any],
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    captures = _ordered_captures(config["root"])
    refinements = _sample_session_charts(captures, manifest, config)
    training, holdout = _fit_node_models(refinements, manifest)
    training_tone, training_samples = _tone_normalized_samples(
        training, refinements, manifest, ("train_A", "train_B", "train_C")
    )
    holdout_tone, holdout_samples = _tone_normalized_samples(
        holdout, refinements, manifest, ("holdout",)
    )
    training_nodes = _aggregate_latent_nodes(training_samples)
    source = np.asarray([node["source_rgb"] for node in training_nodes])
    response = np.asarray([node["response_rgb"] for node in training_nodes])
    matrix_fit = fit_constrained_linear_color_matrix(source, response, ridge=1e-8)
    evaluation = _evaluate_matrix(matrix_fit["color_matrix"], holdout_tone, holdout_samples)
    background = _fit_background(captures, refinements, config)
    grid_centers = {
        row["role"]: np.asarray(row["observed_centers"]).mean(axis=0)
        for row in refinements
    }
    result = {
        "session_id": session_id,
        "root": str(config["root"]),
        "orientation_quarter_turns_ccw": int(config["quarter_turns_ccw"]),
        "orientation_quarter_turns_ccw_by_role": {
            role: _quarter_turns_for_role(config, role) for role in CAPTURE_ROLES
        },
        "training_node_fit": training,
        "holdout_validation": holdout,
        "anchor_tone_normalization": {
            "training": training_tone,
            "holdout": holdout_tone,
        },
        "session_color_matrix": matrix_fit["color_matrix"],
        "session_color_matrix_row_sums": np.asarray(matrix_fit["color_matrix"]).sum(axis=1),
        "session_color_matrix_has_negative_entries": bool(
            np.any(np.asarray(matrix_fit["color_matrix"]) < 0.0)
        ),
        "training_node_mae": matrix_fit["mae"],
        "own_holdout_evaluation": evaluation,
        "temporal_quadratic_background": background,
        "chart_grid_centers_camera_xy": grid_centers,
        "response_grid_refinement": refinements,
    }
    internal = {
        "training_source_rgb": source,
        "training_response_rgb": response,
        "holdout_tone": holdout_tone,
        "holdout_samples": holdout_samples,
    }
    return result, internal


def _summarize_evaluations(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray([row["jpeg_mae"] for row in rows.values()])
    return {
        "by_session": rows,
        "mean_jpeg_mae": float(np.mean(values)),
        "median_jpeg_mae": float(np.median(values)),
        "max_jpeg_mae": float(np.max(values)),
    }


def _cross_pose_generalization(
    sessions: dict[str, dict[str, Any]], internals: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    session_ids = list(sessions)
    session_matrices = {
        session_id: np.asarray(sessions[session_id]["session_color_matrix"])
        for session_id in session_ids
    }
    pooled_fit = fit_constrained_linear_color_matrix(
        np.vstack([internals[key]["training_source_rgb"] for key in session_ids]),
        np.vstack([internals[key]["training_response_rgb"] for key in session_ids]),
        ridge=1e-8,
    )
    baseline_matrix = session_matrices["baseline"]
    baseline_rows = {
        key: _evaluate_matrix(
            baseline_matrix,
            internals[key]["holdout_tone"],
            internals[key]["holdout_samples"],
        )
        for key in session_ids
    }
    pooled_rows = {
        key: _evaluate_matrix(
            pooled_fit["color_matrix"],
            internals[key]["holdout_tone"],
            internals[key]["holdout_samples"],
        )
        for key in session_ids
    }
    specific_rows = {
        key: _evaluate_matrix(
            session_matrices[key],
            internals[key]["holdout_tone"],
            internals[key]["holdout_samples"],
        )
        for key in session_ids
    }
    leave_one_out = {}
    for target in session_ids:
        training_ids = [key for key in session_ids if key != target]
        fit = fit_constrained_linear_color_matrix(
            np.vstack([internals[key]["training_source_rgb"] for key in training_ids]),
            np.vstack([internals[key]["training_response_rgb"] for key in training_ids]),
            ridge=1e-8,
        )
        leave_one_out[target] = {
            "training_sessions": training_ids,
            "color_matrix": fit["color_matrix"],
            **_evaluate_matrix(
                fit["color_matrix"],
                internals[target]["holdout_tone"],
                internals[target]["holdout_samples"],
            ),
        }
    loo_values = np.asarray([row["jpeg_mae"] for row in leave_one_out.values()])
    specific_values = np.asarray([row["jpeg_mae"] for row in specific_rows.values()])
    pooled_summary = _summarize_evaluations(pooled_rows)
    status = (
        "supported_across_sampled_poses"
        if float(np.max(loo_values)) <= 0.08
        and float(np.mean(loo_values)) <= 0.06
        and pooled_summary["mean_jpeg_mae"] <= float(np.mean(specific_values)) + 0.015
        else "pose_conditioning_required"
    )
    matrix_stack = np.stack(list(session_matrices.values()))
    return {
        "status": status,
        "decision_rule": (
            "shared C is supported when leave-one-session-out max/mean JPEG MAE are <=0.08/0.06 "
            "and pooled mean MAE is within 0.015 of pose-specific mean MAE"
        ),
        "pooled_color_matrix": pooled_fit["color_matrix"],
        "pooled_matrix_row_sums": np.asarray(pooled_fit["color_matrix"]).sum(axis=1),
        "pooled_training_node_mae": pooled_fit["mae"],
        "baseline_matrix_on_all_holdouts": _summarize_evaluations(baseline_rows),
        "pooled_matrix_on_all_holdouts": pooled_summary,
        "pose_specific_matrix_on_own_holdout": _summarize_evaluations(specific_rows),
        "leave_one_session_out": leave_one_out,
        "leave_one_session_out_mean_jpeg_mae": float(np.mean(loo_values)),
        "leave_one_session_out_max_jpeg_mae": float(np.max(loo_values)),
        "session_matrix_mean": np.mean(matrix_stack, axis=0),
        "session_matrix_std": np.std(matrix_stack, axis=0),
        "session_matrix_max_frobenius_distance_from_pooled": float(
            max(np.linalg.norm(matrix - pooled_fit["color_matrix"]) for matrix in matrix_stack)
        ),
    }


def _dataset_metadata(
    session_captures: dict[str, dict[str, Path]],
    session_config: dict[str, dict[str, Any]] = SESSION_CONFIG,
) -> dict[str, Any]:
    rows = []
    for session_id, captures in session_captures.items():
        for index, (role, path) in enumerate(captures.items()):
            turns = _quarter_turns_for_role(session_config[session_id], role)
            tags = _read_exif(path)
            jpeg = inspect_jpeg(path)
            with Image.open(path) as image:
                original_size = image.size
            normalized_size = original_size[::-1] if turns % 2 else original_size
            rows.append(
                {
                    "session_id": session_id,
                    "index": index,
                    "role": role,
                    "path": str(path),
                    "original_image_size": original_size,
                    "normalized_image_size": normalized_size,
                    "orientation_quarter_turns_ccw": turns,
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
        "session_count": len(session_captures),
        "frame_count": len(rows),
        "sessions": {key: str(session_config[key]["root"]) for key in session_captures},
        "frames": rows,
    }


def _write_diagnostics(
    sessions: dict[str, dict[str, Any]],
    generalization: dict[str, Any],
    diagnostics: Path = DIAGNOSTICS,
) -> None:
    diagnostics.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(len(sessions), 4, figsize=(14, 2.75 * len(sessions)))
    axes = np.atleast_2d(axes)
    for row_index, (session_id, session) in enumerate(sessions.items()):
        for column_index, refinement in enumerate(session["response_grid_refinement"]):
            axis = axes[row_index, column_index]
            turns = int(refinement["orientation_quarter_turns_ccw"])
            image = _load_rgb(Path(refinement["capture"]), turns)
            points = np.asarray(refinement["observed_centers"])
            left, right = int(points[:, 0].min()) - 45, int(points[:, 0].max()) + 45
            top, bottom = int(points[:, 1].min()) - 45, int(points[:, 1].max()) + 45
            axis.imshow(image[top:bottom, left:right])
            axis.plot(points[:, 0] - left, points[:, 1] - top, "y.", markersize=1.5)
            axis.set_title(f"{session_id}: {refinement['role']}", fontsize=9)
            axis.axis("off")
    figure.tight_layout()
    figure.savefig(diagnostics / "cross_pose_registration.png", dpi=150)
    plt.close(figure)

    session_ids = list(sessions)
    groups = (
        ("baseline", generalization["baseline_matrix_on_all_holdouts"]["by_session"]),
        ("pooled", generalization["pooled_matrix_on_all_holdouts"]["by_session"]),
        ("pose-specific", generalization["pose_specific_matrix_on_own_holdout"]["by_session"]),
        ("leave-one-out", generalization["leave_one_session_out"]),
    )
    x = np.arange(len(session_ids))
    width = 0.19
    figure, axis = plt.subplots(figsize=(9, 4.5))
    for group_index, (label, rows) in enumerate(groups):
        axis.bar(
            x + (group_index - 1.5) * width,
            [rows[key]["jpeg_mae"] for key in session_ids],
            width,
            label=label,
        )
    axis.axhline(0.08, color="black", linestyle="--", linewidth=1, label="max criterion")
    axis.set_xticks(x, session_ids)
    axis.set_ylabel("holdout JPEG RGB MAE")
    axis.legend(ncol=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(diagnostics / "cross_pose_holdout_mae.png", dpi=150)
    plt.close(figure)


def _write_report(result: dict[str, Any]) -> None:
    generalization = result["fit"]["cross_pose_generalization"]
    pooled = generalization["pooled_matrix_on_all_holdouts"]
    specific = generalization["pose_specific_matrix_on_own_holdout"]
    report = f"""# c_color v5 跨姿态泛化报告

## 结论

四个短序列共 24 张 JPEG 均完成配准，其中右侧姿态在读取时逆时针旋转 90 度后进入统一坐标系。每个序列的三张训练色卡拟合颜色与空间干扰项，第四张色卡始终只作留出验证。

四组训练节点汇总得到的共享颜色矩阵，在四张独立留出图上的 JPEG 域 MAE 均值/最大值为 {pooled['mean_jpeg_mae']:.4f}/{pooled['max_jpeg_mae']:.4f}；逐姿态独立矩阵的均值为 {specific['mean_jpeg_mae']:.4f}。留一姿态交叉验证 MAE 均值/最大值为 {generalization['leave_one_session_out_mean_jpeg_mae']:.4f}/{generalization['leave_one_session_out_max_jpeg_mae']:.4f}。按预设规则，跨本次采样姿态的状态为 `{generalization['status']}`。

## 泛化模型

共享颜色项仍写成 `b_t(u) + g_t * m_t(u) * h_t(C(Z_t))`。矩阵 `C` 跨会话共享；逐帧锚点 tone `h_t`、增益 `g_t`、空间场 `m_t`、背景 `b_t` 和几何映射保持姿态/帧条件化。该结论只覆盖本次设备、环境和左右水平姿态，不外推到其他设备或环境光。
"""
    (OUTPUT / "v5_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    sessions = {}
    internals = {}
    captures_by_session = {}
    for session_id, config in SESSION_CONFIG.items():
        captures_by_session[session_id] = _ordered_captures(config["root"])
        sessions[session_id], internals[session_id] = _fit_session(
            session_id, config, manifest
        )
    generalization = _cross_pose_generalization(sessions, internals)
    result = {
        "version": "v5",
        "dataset": _dataset_metadata(captures_by_session),
        "model_equation": (
            "Z_t(u)=D_s[k_(u,t)*W_(H_t,delta2)(X)]; "
            "Yhat_t=J_96,420{clip[b_t(u)+g_t*m_t(u)*h_t(C(Z_t(u)))+epsilon_t(u)]}"
        ),
        "fit": {
            "sessions": sessions,
            "cross_pose_generalization": generalization,
        },
        "parameter_status": {
            "shared_effective_color_operator": generalization["status"],
            "pose_frame_conditioned_geometry_spatial_tone_background": "required",
            "sampled_pose_support": "baseline_repeat_left_right_horizontal",
            "cross_environment_device_distribution": "not_identifiable",
        },
        "unresolved_parameters": [
            "vertical and distance pose coverage",
            "cross-environment and cross-device parameter distributions",
            "separate optical, display, sensor, ISP, and JPEG transforms",
        ],
    }
    safe = _json_safe(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "v5_parameters.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(safe)
    _write_diagnostics(sessions, generalization)
    print(f"cross-pose status: {generalization['status']}")
    print(
        "pooled holdout mean/max: "
        f"{generalization['pooled_matrix_on_all_holdouts']['mean_jpeg_mae']:.6f}/"
        f"{generalization['pooled_matrix_on_all_holdouts']['max_jpeg_mae']:.6f}"
    )
    print(
        "leave-one-out mean/max: "
        f"{generalization['leave_one_session_out_mean_jpeg_mae']:.6f}/"
        f"{generalization['leave_one_session_out_max_jpeg_mae']:.6f}"
    )
    print(f"wrote {OUTPUT / 'v5_parameters.json'}")


if __name__ == "__main__":
    main()
