from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .model import DegradationParameters, degrade
from .schema import CalibrationManifest, CaptureGroup


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _preview(image: Image.Image, max_side: int = 1024) -> tuple[Image.Image, float]:
    scale = min(1.0, max_side / max(image.size))
    if scale == 1.0:
        return image.copy(), scale
    size = tuple(max(1, int(round(value * scale))) for value in image.size)
    return image.resize(size, Image.Resampling.LANCZOS), scale


def _write_roi_overlay(group: CaptureGroup, output: Path) -> None:
    with Image.open(group.frames[0]) as image:
        preview, scale = _preview(image.convert("RGB"))
    draw = ImageDraw.Draw(preview)
    x0, y0, x1, y1 = group.roi_xyxy
    rectangle = tuple(int(round(value * scale)) for value in (x0, y0, x1, y1))
    line_width = max(2, int(round(5 * scale)))
    draw.rectangle(rectangle, outline=(255, 48, 48), width=line_width)
    label = f"{group.id} | {group.domain} | {group.path}"
    font = ImageFont.load_default()
    text_box = draw.textbbox((rectangle[0], rectangle[1]), label, font=font)
    text_y = max(0, rectangle[1] - (text_box[3] - text_box[1]) - 6)
    text_width = text_box[2] - text_box[0] + 8
    text_height = text_box[3] - text_box[1] + 6
    draw.rectangle(
        (rectangle[0], text_y, rectangle[0] + text_width, text_y + text_height),
        fill=(16, 16, 16),
    )
    draw.text((rectangle[0] + 4, text_y + 3), label, fill=(255, 255, 255), font=font)
    preview.save(output, format="PNG", optimize=False)


def _representative_roi(group: CaptureGroup) -> np.ndarray:
    frames: list[np.ndarray] = []
    for frame in group.frames:
        with Image.open(frame) as image:
            frames.append(np.asarray(image.convert("RGB").crop(group.roi_xyxy), np.float32) / 255.0)
    return np.median(np.stack(frames), axis=0).astype(np.float32)


def _comparison_group(groups: list[CaptureGroup]) -> CaptureGroup:
    role_rank = {"psf": 0, "geometry": 1, "validation": 2, "external_validation": 3}
    return min(
        groups,
        key=lambda group: min((role_rank.get(role, 9) for role in group.roles), default=9),
    )


def _params_for_comparison(
    source: np.ndarray,
    effective: Mapping[str, Any],
    real: np.ndarray,
) -> DegradationParameters:
    scale = effective.get("scale_camera_per_source") or [None, None]
    if all(value is not None and float(value) > 0.0 for value in scale):
        output_width = max(32, int(round(source.shape[1] * float(scale[0]))))
        output_height = max(32, int(round(source.shape[0] * float(scale[1]))))
    else:
        output_width, output_height = real.shape[1], real.shape[0]
    largest = max(output_width, output_height)
    if largest > 1200:
        factor = 1200.0 / largest
        output_width = max(32, int(round(output_width * factor)))
        output_height = max(32, int(round(output_height * factor)))

    blur = effective.get("blur_fwhm_camera_px") or [0.0, 0.0]
    blur_pair = tuple(float(value or 0.0) for value in blur)
    status = effective.get("status", {})
    matrix = effective.get("color_matrix") if status.get("color") == "estimated" else None
    color_matrix = (
        tuple(tuple(float(value) for value in row) for row in matrix)
        if matrix is not None
        else ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    bias = effective.get("color_bias_rgb") if status.get("color") == "estimated" else None
    bias_rgb = tuple(float(value) for value in bias) if bias is not None else (0.0, 0.0, 0.0)
    slope = effective.get("noise_slope_rgb") if status.get("noise") == "estimated" else None
    intercept = (
        effective.get("noise_intercept_rgb") if status.get("noise") == "estimated" else None
    )
    noise_slope = tuple(float(value) for value in slope) if slope is not None else (0.0, 0.0, 0.0)
    noise_intercept = (
        tuple(float(value) for value in intercept) if intercept is not None else (0.0, 0.0, 0.0)
    )
    return DegradationParameters(
        output_size=(output_width, output_height),
        blur_fwhm_px=blur_pair,
        blur_angle_deg=float(effective.get("blur_angle_deg") or 0.0),
        color_matrix=color_matrix,
        bias_rgb=bias_rgb,
        noise_slope=noise_slope,
        noise_intercept=noise_intercept,
        jpeg_quality=int(effective["jpeg_quality"]),
        jpeg_subsampling=2,
    )


def _write_comparison(
    domain: str,
    groups: list[CaptureGroup],
    domain_result: Mapping[str, Any],
    output: Path,
    seed: int,
) -> None:
    group = _comparison_group(groups)
    with Image.open(group.source) as image:
        source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    real = _representative_roi(group)
    params = _params_for_comparison(source, domain_result["effective_parameters"], real)
    synthetic = degrade(source, params, seed=seed, encode_jpeg=True)
    real_resized = np.asarray(
        Image.fromarray(np.rint(real * 255.0).astype(np.uint8), "RGB").resize(
            (synthetic.shape[1], synthetic.shape[0]), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    difference_rgb = real_resized - synthetic
    difference = np.clip(
        np.tensordot(
            difference_rgb,
            np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32),
            axes=([-1], [0]),
        ),
        -0.5,
        0.5,
    )

    figure, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=120)
    axes[0].imshow(source)
    axes[0].set_title("Source pattern")
    axes[1].imshow(synthetic)
    axes[1].set_title("Effective synthesis")
    axes[2].imshow(real_resized)
    axes[2].set_title("Real ROI (resized)")
    axes[3].imshow(difference, vmin=-0.5, vmax=0.5, cmap="coolwarm")
    axes[3].set_title("Unregistered difference")
    for axis in axes:
        axis.set_axis_off()
    figure.suptitle(f"{domain} | group {group.id} | diagnostic only", fontsize=10)
    figure.tight_layout()
    figure.savefig(output, format="png", dpi=120, metadata={"Software": "glasses-fit-v1"})
    plt.close(figure)


def _domain_focal_label(domain: Mapping[str, Any]) -> str:
    values = [
        payload["metadata"].get("focal_35mm_median")
        for payload in domain["groups"].values()
        if payload["metadata"].get("focal_35mm_median") is not None
    ]
    return f"{int(round(float(np.median(values))))} mm" if values else "unknown focal domain"


def _format_pair(values: Any, digits: int = 3) -> str:
    if not values or any(value is None for value in values):
        return "not identifiable"
    return " x ".join(f"{float(value):.{digits}f}" for value in values)


def _markdown_report(result: Mapping[str, Any]) -> str:
    jpeg = result["dataset"]["jpeg"]
    lines = [
        "# Smart-glasses effective degradation fit v1",
        "",
        "## Scope",
        "",
        "This report fits the selected readable reflection path in JPEG space. It does not separate OLED, lens, sensor, and ISP physics.",
        "",
        "The path-level effective forward model is:",
        "",
        r"\[",
        r"\widehat Y_{d,p,j,t}=\mathcal J_{Q_d}\left\{\operatorname{clip}_{[0,1]}\left[g_{d,p,j,t}\left(A_{d,p}\mathcal D_{s_{d,p}}\left[k_{d,p,\mathbf u}*\left(m_{d,p}(\mathbf u)\odot\mathcal W_{H_{d,p,j,t}}(X_j)\right)\right]+b_{d,p,j,t}(\mathbf u)\right)+\varepsilon_{d,p,j,t}\right]\right\}.",
        r"\]",
        "",
        r"For multiple visible copies, replace the single optical path by \(Z(\mathbf u)=\sum_p\alpha_p(\mathbf u)\mathcal D_{s_p}[k_{p,\mathbf u}*(m_p\odot\mathcal W_{H_p}(X))]\). The current data supports only the selected primary path.",
        "",
        "## Dataset and encoding",
        "",
        f"- Groups: `{result['dataset']['group_count']}`; frames: `{result['dataset']['frame_count']}`.",
        f"- JPEG quality {jpeg['equivalent_quality']}; chroma subsampling `{jpeg['subsampling']}`.",
        f"- Quantization-table SHA-256: `{jpeg['qtable_sha256']}`.",
        "",
    ]
    for domain_name, domain in result["domains"].items():
        effective = domain["effective_parameters"]
        lines.extend(
            [
                f"## {_domain_focal_label(domain)} domain",
                "",
                f"- Domain ID: `{domain_name}`.",
                f"- Local scale (camera/source): `{_format_pair(effective['scale_camera_per_source'], 4)}`.",
                f"- Effective blur FWHM (camera px): `{_format_pair(effective['blur_fwhm_camera_px'], 2)}`.",
                f"- Scale status: `{effective['status']['scale']}`; blur status: `{effective['status']['blur']}`; color status: `{effective['status']['color']}`; noise status: `{effective['status']['noise']}`.",
                "",
                "### Identifiability",
                "",
            ]
        )
        for name, item in domain["identifiability"].items():
            lines.append(f"- `{name}`: `{item['status']}`. {item['reason']}")
        lines.append("")
    lines.extend(
        [
            "## Validation and diagnostic interpretation",
            "",
            "- ROI overlays verify that configuration, rather than core code, owns path coordinates.",
            "- Comparison figures show a canonical effective synthesis beside a resized real ROI.",
            "- Their unregistered difference is diagnostic only and is not reported as independent PSNR/SSIM validation.",
            "- The fixed text target is held out from PSF fitting; its current role is qualitative content-domain validation.",
            "",
            "## 不可识别的参数",
            "",
            "- Absolute crop, mirror state, full homography, reflection-path count, and path weights are unresolved because the geometry chart is periodic.",
            "- A unique response curve, color matrix, spatial attenuation, and black level are unresolved because auto exposure/white balance changed across targets.",
            "- Shot/read/fixed-pattern noise cannot be separated from ISP denoising, sharpening, and JPEG without RAW and controlled gray bursts.",
            "- Spatially varying PSF, controlled motion blur, and cross-device/pose distributions require broader calibrated captures.",
            "",
            "## 下一轮拍摄",
            "",
            "| Priority | Capture | Frames per pose | Identifies |",
            "|---|---|---:|---|",
            r"| P0 | 每个目标前后的非对称定位点阵 | 各 10 | 绝对裁剪、镜像、\(H_{p,t}\)、路径身份 |",
            r"| P0 | 单亮点扫描：白/R/G/B 亮点放在屏幕 5 x 5 位置 | 每位置每颜色 10 | \(P, H_p, \alpha_p\)、空间 PSF、色散 |",
            r"| P0 | 同一帧内黑块 + 16 级灰阶 + RGB/CMY/灰色色块和 fiducial | 10 | \(g,A,m,b\) |",
            r"| P0 | 3 x 3 位置稀疏点和水平/垂直/对角斜边 | 各 10 | \(k_{\mathbf u}\)、MTF、ringing |",
            "| P0 | 棋盘、逐像素反相棋盘、全黑图；同曝光不移动 | 各 10 | 加性 flare 与乘性衰减 |",
            r"| P1 | 每个 ISO/曝光档的黑场和多个灰阶 burst | 各 30 | \(\operatorname{Var}(\varepsilon_c\mid\mu_c)=a_c\mu_c+b_c\) |",
            "| P1 | 关键标定图的 RAW/DNG + JPEG 同步对 | 各 10 | 传感器、ISP、tone curve、JPEG 分层 |",
            "| P1 | 受控运动、不同快门时间的点阵 | 各 10 | 条件运动核 |",
            "| P1 | 至少 20 张独立随机文本图 | 每张 5--10 | OCR/内容泛化验证 |",
            "",
            "Generalization protocol: for each device, collect full calibration at nine anchor poses and asymmetric-grid plus random-text pairs at at least 30 coverage poses. Cover near/mid/far distance, yaw, pitch, and dark/office/strong-side-light conditions; use at least three devices or glasses and three independent sessions.",
            "",
        ]
    )
    return "\n".join(lines)


def write_results(
    result: Mapping[str, Any],
    manifest: CalibrationManifest,
    output: Path,
    *,
    seed: int,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = output / "v1_diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for stale in diagnostics.glob("*.png"):
        stale.unlink()

    (output / "v1_parameters.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output / "v1_report.md").write_text(_markdown_report(result), encoding="utf-8")
    for group in manifest.groups:
        _write_roi_overlay(group, diagnostics / f"roi_{_safe_name(group.id)}.png")
    for domain in manifest.domains:
        groups = [group for group in manifest.groups if group.domain == domain]
        _write_comparison(
            domain,
            groups,
            result["domains"][domain],
            diagnostics / f"comparison_{_safe_name(domain)}.png",
            seed,
        )
