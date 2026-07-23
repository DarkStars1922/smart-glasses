from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.run_fit_v5 import (
    OUTPUT,
    PATTERN_ROOT,
    SESSION_CONFIG,
    _cross_pose_generalization,
    _dataset_metadata,
    _fit_session,
    _json_safe,
    _ordered_captures,
    _write_diagnostics,
)


DIAGNOSTICS = OUTPUT / "v6_diagnostics"
SESSION_CONFIG_V6 = {
    **SESSION_CONFIG,
    "pose_down": {
        "root": Path("images/real/c_color_pose_down"),
        "quarter_turns_ccw": 1,
        "role_quarter_turns_ccw": {"black_start": 0},
        "sampling_radius": 5,
        "max_adjustment_px": 70.0,
        "initial_grid_corners": {
            "train_A": ((745, 510), (1255, 545), (735, 825), (1225, 865)),
            "train_B": ((660, 560), (1155, 580), (670, 880), (1155, 880)),
            "train_C": ((790, 355), (1335, 440), (790, 735), (1325, 690)),
            "holdout": ((1030, 270), (1520, 280), (1060, 660), (1525, 640)),
        },
    },
}


def _write_report(result: dict) -> None:
    generalization = result["fit"]["cross_pose_generalization"]
    pooled = generalization["pooled_matrix_on_all_holdouts"]
    specific = generalization["pose_specific_matrix_on_own_holdout"]
    down = pooled["by_session"]["pose_down"]
    report = f"""# c_color v6 下移姿态泛化报告

## 结论

五个短序列共 30 张 JPEG 完成配准。`c_color_pose_down` 的四张色卡和定位标记均完整可见，所有色块中心均进入拟合；色卡与末张黑底逆时针旋转 90 度进入源图方向，首张已横向保存的黑底保持不旋转。每个序列的前三张色卡拟合颜色与空间干扰项，第四张色卡始终只作独立留出验证。

五组训练节点汇总得到的共享颜色矩阵，在五张独立留出图上的 JPEG 域 MAE 均值/最大值为 {pooled['mean_jpeg_mae']:.4f}/{pooled['max_jpeg_mae']:.4f}，下移姿态为 {down['jpeg_mae']:.4f}；逐姿态独立矩阵的均值为 {specific['mean_jpeg_mae']:.4f}。留一姿态交叉验证 MAE 均值/最大值为 {generalization['leave_one_session_out_mean_jpeg_mae']:.4f}/{generalization['leave_one_session_out_max_jpeg_mae']:.4f}。按预设规则，跨本次采样姿态的状态为 `{generalization['status']}`。

## 模型边界

共享颜色项仍写成 `b_t(u) + g_t * m_t(u) * h_t(C(Z_t))`。矩阵 `C` 跨已采样会话共享；几何映射、逐帧锚点 tone、增益、空间场和背景保持姿态/帧条件化。本批下移色卡没有发生硬截断，因此它验证了下移姿态下的颜色泛化，但不能识别可见性边界；可见性掩膜仍需含真实截断边界的数据。
"""
    (OUTPUT / "v6_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    manifest = json.loads((PATTERN_ROOT / "pattern_manifest.json").read_text(encoding="utf-8"))
    sessions = {}
    internals = {}
    captures_by_session = {}
    for session_id, config in SESSION_CONFIG_V6.items():
        captures_by_session[session_id] = _ordered_captures(config["root"])
        sessions[session_id], internals[session_id] = _fit_session(
            session_id, config, manifest
        )
    generalization = _cross_pose_generalization(sessions, internals)
    result = {
        "version": "v6",
        "dataset": _dataset_metadata(captures_by_session, SESSION_CONFIG_V6),
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
            "sampled_pose_support": "baseline_repeat_left_right_horizontal_and_down_vertical",
            "pose_down_chart_visibility": "complete_no_hard_clipping_observed",
            "visibility_mask": "not_identifiable_from_fully_visible_charts",
            "cross_environment_device_distribution": "not_identifiable",
        },
        "unresolved_parameters": [
            "upward vertical pose coverage and hard visibility boundaries",
            "cross-environment and cross-device parameter distributions",
            "separate optical, display, sensor, ISP, and JPEG transforms",
        ],
    }
    safe = _json_safe(result)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "v6_parameters.json").write_text(
        json.dumps(safe, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(safe)
    _write_diagnostics(sessions, generalization, DIAGNOSTICS)
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
    print(f"wrote {OUTPUT / 'v6_parameters.json'}")


if __name__ == "__main__":
    main()
