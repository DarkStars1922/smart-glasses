from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from statistics import median

import numpy as np
from PIL import ExifTags, Image


_BURST_INDEX = re.compile(r"TIMEBURST(\d+)", re.IGNORECASE)
_EXIF_IFD = 0x8769

# These ROIs select the principal readable reflection in the current pilot capture.
_POINT_GRID_ROI = (2850, 1100, 3536, 2300)
_CHECKER_ROI = (1100, 100, 2500, 1150)
_CHECKER_EDGE_ROI = (1300, 220, 2300, 1000)
_PHOTOMETRY_ROI = (2240, 820, 2460, 1080)


def _burst_sort_key(path: Path) -> tuple[int, str]:
    match = _BURST_INDEX.search(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def index_dataset(root: Path) -> dict[str, list[Path]]:
    return {
        group.name: sorted(group.glob("*.jpg"), key=_burst_sort_key)
        for group in sorted(root.iterdir(), key=lambda path: path.name)
        if group.is_dir()
    }


def _read_exif(path: Path) -> dict[str, object]:
    with Image.open(path) as image:
        exif = image.getexif()
        tags = {ExifTags.TAGS.get(key, key): value for key, value in exif.items()}
        try:
            tags.update(
                {
                    ExifTags.TAGS.get(key, key): value
                    for key, value in exif.get_ifd(_EXIF_IFD).items()
                }
            )
        except (KeyError, TypeError):
            pass
    return tags


def _numeric(tags: dict[str, object], key: str) -> float | None:
    value = tags.get(key)
    if value is None:
        return None
    return float(value)


def _range(values: list[float]) -> list[float]:
    return [round(min(values), 6), round(max(values), 6)] if values else []


def _summarize_group(files: list[Path]) -> dict[str, object]:
    exif_rows = [_read_exif(path) for path in files]
    focal = [value for row in exif_rows if (value := _numeric(row, "FocalLengthIn35mmFilm")) is not None]
    shutter = [value for row in exif_rows if (value := _numeric(row, "ExposureTime")) is not None]
    iso = [value for row in exif_rows if (value := _numeric(row, "ISOSpeedRatings")) is not None]
    exposure_mode = [value for row in exif_rows if (value := _numeric(row, "ExposureMode")) is not None]
    white_balance = [value for row in exif_rows if (value := _numeric(row, "WhiteBalance")) is not None]

    with Image.open(files[0]) as image:
        size = list(image.size)

    return {
        "count": len(files),
        "image_size": size,
        "focal_35mm_median": round(float(median(focal)), 2),
        "focal_35mm_range": _range(focal),
        "shutter_seconds_range": _range(shutter),
        "iso_range": _range(iso),
        "auto_exposure": any(value == 0 for value in exposure_mode) if exposure_mode else None,
        "exposure_mode_source": "exif" if exposure_mode else "missing_in_timeburst_exif",
        "auto_white_balance": any(value == 0 for value in white_balance),
    }


def _connected_components(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> list[dict[str, object]]:
    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    components: list[dict[str, object]] = []
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        points: list[tuple[int, int]] = []
        while stack:
            current_y, current_x = stack.pop()
            points.append((current_y, current_x))
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    next_y = current_y + delta_y
                    next_x = current_x + delta_x
                    if (
                        0 <= next_y < height
                        and 0 <= next_x < width
                        and mask[next_y, next_x]
                        and not seen[next_y, next_x]
                    ):
                        seen[next_y, next_x] = True
                        stack.append((next_y, next_x))
        if not min_area <= len(points) <= max_area:
            continue
        point_y = np.fromiter((point[0] for point in points), dtype=np.float64)
        point_x = np.fromiter((point[1] for point in points), dtype=np.float64)
        components.append(
            {
                "area": len(points),
                "center": (float(point_x.mean()), float(point_y.mean())),
                "bbox": (
                    int(point_x.min()),
                    int(point_y.min()),
                    int(point_x.max()),
                    int(point_y.max()),
                ),
            }
        )
    return components


def _point_components(score: np.ndarray) -> list[dict[str, object]]:
    components = _connected_components(score > 45.0, 8, 200)
    height, width = score.shape
    return [
        component
        for component in components
        if 20 < component["center"][0] < width - 20
        and 20 < component["center"][1] < height - 20
    ]


def _dot_sigmas(score: np.ndarray, components: list[dict[str, object]]) -> list[list[float]]:
    sigmas: list[list[float]] = []
    for component in components:
        center_x, center_y = component["center"]
        x0 = int(round(center_x)) - 12
        y0 = int(round(center_y)) - 12
        patch = score[y0 : y0 + 25, x0 : x0 + 25]
        if patch.shape != (25, 25):
            continue
        border = np.concatenate(
            (patch[:3].ravel(), patch[-3:].ravel(), patch[:, :3].ravel(), patch[:, -3:].ravel())
        )
        weight = np.maximum(patch - np.median(border), 0.0)
        if weight.sum() < 30.0:
            continue
        grid_y, grid_x = np.indices(weight.shape)
        mean_x = float((weight * grid_x).sum() / weight.sum())
        mean_y = float((weight * grid_y).sum() / weight.sum())
        covariance = np.array(
            [
                [
                    (weight * (grid_x - mean_x) ** 2).sum(),
                    (weight * (grid_x - mean_x) * (grid_y - mean_y)).sum(),
                ],
                [
                    (weight * (grid_x - mean_x) * (grid_y - mean_y)).sum(),
                    (weight * (grid_y - mean_y) ** 2).sum(),
                ],
            ],
            dtype=np.float64,
        ) / weight.sum()
        eigenvalues = np.linalg.eigvalsh(covariance)
        if eigenvalues[0] > 0.0 and eigenvalues[1] < 80.0:
            sigmas.append(np.sqrt(eigenvalues).tolist())
    return sigmas


def _neighbor_spacing(centers: np.ndarray) -> tuple[float, float]:
    horizontal: list[float] = []
    vertical: list[float] = []
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            delta_x, delta_y = centers[second] - centers[first]
            distance = float(np.hypot(delta_x, delta_y))
            if not 40.0 < distance < 100.0:
                continue
            if abs(delta_y) < 0.25 * abs(delta_x):
                horizontal.append(distance)
            if abs(delta_x) < 0.25 * abs(delta_y):
                vertical.append(distance)
    if not horizontal or not vertical:
        raise RuntimeError("could not recover both point-grid axes")
    return float(np.median(horizontal)), float(np.median(vertical))


def _fit_point_grid(files: list[Path]) -> dict[str, object]:
    all_sigmas: list[list[float]] = []
    dot_counts: list[int] = []
    representative_centers: np.ndarray | None = None
    for index, path in enumerate(files):
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB").crop(_POINT_GRID_ROI), dtype=np.float32)
        score = 0.5 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] - 0.4 * rgb[:, :, 0]
        components = _point_components(score)
        dot_counts.append(len(components))
        all_sigmas.extend(_dot_sigmas(score, components))
        if index == 5:
            representative_centers = np.asarray(
                [component["center"] for component in components], dtype=np.float64
            )

    if representative_centers is None or not all_sigmas:
        raise RuntimeError("point-grid fitting produced no samples")
    spacing_x, spacing_y = _neighbor_spacing(representative_centers)
    scale = [spacing_x / 150.0, spacing_y / 150.0]
    sigma = np.median(np.asarray(all_sigmas), axis=0)
    observed_fwhm = 2.355 * sigma
    source_fwhm_range = [observed_fwhm[0] / max(scale), observed_fwhm[1] / min(scale)]
    return {
        "source_dot_size_px": 7,
        "source_grid_spacing_px": 150,
        "detected_dot_samples": len(all_sigmas),
        "dots_per_frame_range": [min(dot_counts), max(dot_counts)],
        "camera_grid_spacing_px": [round(spacing_x, 3), round(spacing_y, 3)],
        "scale_camera_per_source": [round(value, 4) for value in scale],
        "observed_fwhm_camera_px": [round(float(value), 3) for value in observed_fwhm],
        "effective_fwhm_source_px_approx": [round(float(value), 2) for value in source_fwhm_range],
    }


def _checker_centers(score: np.ndarray) -> np.ndarray:
    components = _connected_components(score > 145.0, 500, 5000)
    centers: list[tuple[float, float]] = []
    for component in components:
        left, top, right, bottom = component["bbox"]
        width = right - left + 1
        height = bottom - top + 1
        if 35 <= width <= 90 and 35 <= height <= 90:
            centers.append(component["center"])
    return np.asarray(centers, dtype=np.float64)


def _checker_scale(centers: np.ndarray) -> tuple[float, float]:
    horizontal: list[float] = []
    vertical: list[float] = []
    for first in range(len(centers)):
        for second in range(first + 1, len(centers)):
            delta_x, delta_y = centers[second] - centers[first]
            distance = float(np.hypot(delta_x, delta_y))
            if not 70.0 < distance < 180.0:
                continue
            angle = (float(np.degrees(np.arctan2(delta_y, delta_x))) + 180.0) % 180.0
            if 0.0 <= angle < 25.0:
                horizontal.append(distance)
            if 75.0 <= angle < 110.0:
                vertical.append(distance)
    if not horizontal or not vertical:
        raise RuntimeError("could not recover both checkerboard axes")
    # Equal-colored square centers are two 120 px cells apart.
    return float(np.median(horizontal)) / 240.0, float(np.median(vertical)) / 240.0


def _peak_fwhm(profile: np.ndarray) -> list[int]:
    peaks: list[int] = []
    threshold = float(np.percentile(profile, 70))
    for index in range(15, len(profile) - 15):
        if profile[index] != max(profile[index - 8 : index + 9]) or profile[index] <= threshold:
            continue
        if not peaks or index - peaks[-1] > 25:
            peaks.append(index)
        elif profile[index] > profile[peaks[-1]]:
            peaks[-1] = index

    widths: list[int] = []
    for index in peaks:
        local = profile[index - 12 : index + 13]
        baseline = float(np.percentile(local, 10))
        half_height = baseline + (float(profile[index]) - baseline) / 2.0
        left = index
        while left > index - 12 and profile[left] >= half_height:
            left -= 1
        right = index
        while right < index + 12 and profile[right] >= half_height:
            right += 1
        widths.append(right - left - 1)
    return widths


def _checker_edge_fwhm(path: Path) -> float:
    with Image.open(path) as image:
        rotated = image.convert("RGB").crop(_CHECKER_EDGE_ROI).rotate(
            -7.0, resample=Image.Resampling.BICUBIC, expand=False
        )
    rgb = np.asarray(rotated, dtype=np.float32)
    gray = 0.5 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2]
    center = gray[100:-100, 100:-100]
    vertical_profile = np.abs(np.diff(center, axis=1)).mean(axis=0)
    horizontal_profile = np.abs(np.diff(center, axis=0)).mean(axis=1)
    widths = _peak_fwhm(vertical_profile) + _peak_fwhm(horizontal_profile)
    if not widths:
        raise RuntimeError("checkerboard edge fitting produced no edge widths")
    return float(np.median(widths))


def _fit_checker(files: list[Path]) -> dict[str, object]:
    representative = files[5]
    with Image.open(representative) as image:
        rgb = np.asarray(image.convert("RGB").crop(_CHECKER_ROI), dtype=np.float32)
    score = 0.5 * rgb[:, :, 1] + 0.5 * rgb[:, :, 2] - 0.25 * rgb[:, :, 0]
    centers = _checker_centers(score)
    scale_x, scale_y = _checker_scale(centers)
    edge_fwhm = _checker_edge_fwhm(representative)
    return {
        "source_cell_size_px": 120,
        "detected_square_centers": len(centers),
        "scale_camera_per_source": [round(scale_x, 4), round(scale_y, 4)],
        "edge_gradient_fwhm_camera_px": round(edge_fwhm, 3),
        "edge_gradient_fwhm_source_px_approx": round(
            edge_fwhm / float(np.mean([scale_x, scale_y])), 2
        ),
        "absolute_screen_mapping_identifiable": False,
        "reason": "periodic checkerboard reveals only a cropped screen region",
    }


def _patch_statistics(files: list[Path]) -> dict[str, object]:
    means: list[np.ndarray] = []
    within_standard_deviation: list[np.ndarray] = []
    exposure_gain: list[float] = []
    for path in files:
        with Image.open(path) as image:
            patch = np.asarray(image.convert("RGB").crop(_PHOTOMETRY_ROI), dtype=np.float32)
        means.append(patch.mean(axis=(0, 1)))
        within_standard_deviation.append(patch.std(axis=(0, 1)))
        tags = _read_exif(path)
        shutter = _numeric(tags, "ExposureTime")
        iso = _numeric(tags, "ISOSpeedRatings")
        if shutter is not None and iso is not None:
            exposure_gain.append(shutter * iso)

    means_array = np.asarray(means)
    within_array = np.asarray(within_standard_deviation)
    return {
        "median_rgb": np.round(np.median(means_array, axis=0), 3).tolist(),
        "between_frame_rgb_std": np.round(means_array.std(axis=0), 3).tolist(),
        "median_within_patch_rgb_std": np.round(np.median(within_array, axis=0), 3).tolist(),
        "exposure_time_x_iso_range": _range(exposure_gain),
    }


def _fit_photometry(groups: dict[str, list[Path]]) -> dict[str, object]:
    black = _patch_statistics(groups["00"])
    white = _patch_statistics(groups["01"])
    gray = _patch_statistics(groups["05"])
    ratio = np.asarray(white["median_rgb"]) / np.maximum(np.asarray(gray["median_rgb"]), 1e-6)
    exposure_ranges = {
        tuple(black["exposure_time_x_iso_range"]),
        tuple(white["exposure_time_x_iso_range"]),
        tuple(gray["exposure_time_x_iso_range"]),
    }
    return {
        "auto_exposure_detected": len(exposure_ranges) > 1,
        "black": black,
        "white": white,
        "gray128": gray,
        "white_to_gray128_rgb_ratio": np.round(ratio, 3).tolist(),
        "identifiability": "relative JPEG-space statistics only; response gamma and sensor noise are confounded with auto exposure and ISP",
    }


def fit_dataset(root: Path) -> dict[str, object]:
    groups = index_dataset(root)
    group_summary = {name: _summarize_group(files) for name, files in groups.items()}
    point_grid = _fit_point_grid(groups["23"])
    checker = _fit_checker(groups["19_2"])
    photometry = _fit_photometry(groups)

    return {
        "schema_version": "v0",
        "model_scope": "effective super-macro JPEG degradation; not a separated sensor-physics model",
        "groups": group_summary,
        "domains": {
            "47mm": {
                "groups": ["00", "01", "05", "18", "19", "22", "23", "24", "28"],
                "point_grid_7px": point_grid,
            },
            "69mm": {
                "groups": ["19_2", "30", "30_2"],
                "matched_pair": ["19_2", "30_2"],
                "checkerboard": checker,
            },
        },
        "photometry": photometry,
        "current_conclusions": [
            "The 69 mm matched checker/text pair is valid for local geometry and text recoverability.",
            "The 7 px point grid is visible; the 3 px point grid is below the reliable optical/ISP cutoff.",
            "Text at 40 px and above is visibly recoverable in the 69 mm domain; 16-24 px text remains marginal.",
            "Multiple reflection paths require path-specific crops or a mixture-of-warps term.",
        ],
        "limitations": [
            "A periodic checkerboard cannot identify the absolute source crop or reflection orientation.",
            "Automatic exposure and white balance prevent a unique gamma, color matrix, or sensor-noise fit.",
            "Only one fixed text chart is available, so OCR generalization cannot yet be measured.",
            "The 47 mm and 69 mm captures must remain separate conditional domains.",
        ],
        "recommended_v2": {
            "required_patterns": [
                "21_asymmetric_grid",
                "29_spatial_color_patches",
                "32_gray_ramp_with_fiducials",
                "33_random_text_with_fiducials",
            ],
            "optional_patterns": [
                "25_resolution_lines_horizontal",
                "26_resolution_lines_vertical",
                "27_resolution_lines_diagonal",
            ],
            "capture_protocol": [
                "Use 69 mm as the primary recovery domain and 47 mm as a secondary domain.",
                "At each pose, capture the asymmetric grid immediately before every target without moving the phone or glasses.",
                "Wait 3 seconds after switching patterns, then keep the sharpest 5-8 frames from a 10-frame burst.",
                "Capture the gray ramp and RGB/CMY patches in one frame so auto exposure is shared across all levels.",
                "Capture at least 20 independently generated text charts; keep 5-10 frames per chart.",
                "Preserve full JPEGs and EXIF; never mix focal domains without a domain label.",
            ],
        },
    }


def _markdown_report(result: dict[str, object]) -> str:
    point = result["domains"]["47mm"]["point_grid_7px"]
    checker = result["domains"]["69mm"]["checkerboard"]
    photometry = result["photometry"]
    required = result["recommended_v2"]["required_patterns"]
    protocol = result["recommended_v2"]["capture_protocol"]
    limitations = result["limitations"]
    conclusions = result["current_conclusions"]

    lines = [
        "# Smart-glasses degradation fit v0",
        "",
        "## Scope",
        "",
        "This fit describes the effective super-macro JPEG pipeline. Optical blur, demosaicing, denoising, sharpening, auto exposure, color processing, and JPEG are not fully separable in the current data.",
        "",
        "## 47 mm domain",
        "",
        f"- Point-grid local scale: `{point['scale_camera_per_source'][0]:.4f} x {point['scale_camera_per_source'][1]:.4f}` camera pixels per source pixel.",
        f"- Observed 7 px dot FWHM: `{point['observed_fwhm_camera_px'][0]:.2f} x {point['observed_fwhm_camera_px'][1]:.2f}` camera pixels.",
        f"- Approximate source-coordinate effective FWHM: `{point['effective_fwhm_source_px_approx'][0]:.1f}-{point['effective_fwhm_source_px_approx'][1]:.1f}` source pixels.",
        f"- Valid dot samples: `{point['detected_dot_samples']}`.",
        "",
        "## 69 mm domain",
        "",
        f"- Checker local scale: `{checker['scale_camera_per_source'][0]:.4f} x {checker['scale_camera_per_source'][1]:.4f}` camera pixels per source pixel.",
        f"- Edge-gradient FWHM: `{checker['edge_gradient_fwhm_camera_px']:.2f}` camera pixels, approximately `{checker['edge_gradient_fwhm_source_px_approx']:.1f}` source pixels.",
        "- `19_2` and `30_2` share the same approximately 69 mm focal domain.",
        "- Absolute mapping is unresolved because the visible checkerboard is periodic and spatially cropped.",
        "",
        "## Photometry",
        "",
        f"- Black median RGB: `{photometry['black']['median_rgb']}`.",
        f"- White median RGB: `{photometry['white']['median_rgb']}`.",
        f"- Gray-128 median RGB: `{photometry['gray128']['median_rgb']}`.",
        f"- White/gray RGB ratio: `{photometry['white_to_gray128_rgb_ratio']}`.",
        "- Auto exposure and auto white balance are active, so these are JPEG-space relative measurements rather than a unique display/camera response curve.",
        "",
        "## Conclusions",
        "",
    ]
    lines.extend(f"- {item}" for item in conclusions)
    lines.extend(["", "## Limitations", ""])
    lines.extend(f"- {item}" for item in limitations)
    lines.extend(["", "## 第二版拍摄", "", "Required patterns:", ""])
    lines.extend(f"- `{item}`" for item in required)
    lines.extend(["", "Capture protocol:", ""])
    lines.extend(f"- {item}" for item in protocol)
    lines.append("")
    return "\n".join(lines)


def write_results(result: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "v0_parameters.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "v0_report.md").write_text(_markdown_report(result), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the v0 effective smart-glasses degradation model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("images/real/calibration_B"),
        help="Calibration burst directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/results"),
        help="Output directory for JSON parameters and Markdown report",
    )
    arguments = parser.parse_args()
    result = fit_dataset(arguments.dataset)
    write_results(result, arguments.output)
    print(f"wrote {arguments.output / 'v0_parameters.json'}")
    print(f"wrote {arguments.output / 'v0_report.md'}")


if __name__ == "__main__":
    main()
