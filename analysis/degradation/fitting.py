from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from io import BytesIO
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
from PIL import ExifTags, Image, JpegImagePlugin

from .schema import CalibrationManifest, CaptureGroup


_EXIF_IFD = 0x8769


@dataclass(frozen=True)
class FitResult:
    status: str
    reason: str
    sample_count: int
    value: Any
    units: str
    coordinate_system: str
    uncertainty: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.status not in {"estimated", "provisional", "not_identifiable"}:
            raise ValueError(f"invalid fit status {self.status!r}")
        if self.sample_count < 0:
            raise ValueError("sample_count must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return round(number, 8) if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _result(
    status: str,
    reason: str,
    sample_count: int,
    value: Any,
    units: str,
    coordinate_system: str,
    uncertainty: Mapping[str, Any] | None = None,
) -> FitResult:
    return FitResult(
        status=status,
        reason=reason,
        sample_count=sample_count,
        value=_json_safe(value),
        units=units,
        coordinate_system=coordinate_system,
        uncertainty=_json_safe(uncertainty or {}),
    )


def not_identifiable(reason: str) -> FitResult:
    return _result(
        "not_identifiable",
        reason,
        0,
        None,
        "not applicable",
        "not applicable",
    )


def load_rgb_rois(group: CaptureGroup) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for frame in group.frames:
        with Image.open(frame) as image:
            arrays.append(
                np.asarray(image.convert("RGB").crop(group.roi_xyxy), dtype=np.float32)
                / 255.0
            )
    return arrays


def _components(
    mask: np.ndarray,
    min_area: int,
    max_area: int,
) -> list[dict[str, Any]]:
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components: list[dict[str, Any]] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if min_area <= area <= max_area:
            components.append(
                {
                    "area": area,
                    "center": (float(centers[index, 0]), float(centers[index, 1])),
                    "bbox": (x, y, x + width - 1, y + height - 1),
                }
            )
    return components


def _point_components(score: np.ndarray) -> list[dict[str, Any]]:
    height, width = score.shape
    components = _components(score > (45.0 / 255.0), 8, 250)
    return [
        component
        for component in components
        if 14 < component["center"][0] < width - 14
        and 14 < component["center"][1] < height - 14
    ]


def _dot_shape(
    score: np.ndarray,
    component: Mapping[str, Any],
    radius: int,
) -> tuple[float, float, float] | None:
    center_x, center_y = component["center"]
    x0 = int(round(center_x)) - radius
    y0 = int(round(center_y)) - radius
    patch = score[y0 : y0 + 2 * radius + 1, x0 : x0 + 2 * radius + 1]
    expected_size = 2 * radius + 1
    if patch.shape != (expected_size, expected_size):
        return None
    border = np.concatenate(
        (patch[:3].ravel(), patch[-3:].ravel(), patch[:, :3].ravel(), patch[:, -3:].ravel())
    )
    weight = np.maximum(patch - float(np.median(border)), 0.0)
    if float(weight.sum()) < 0.1:
        return None
    grid_y, grid_x = np.indices(weight.shape, dtype=np.float64)
    total = float(weight.sum())
    mean_x = float((weight * grid_x).sum() / total)
    mean_y = float((weight * grid_y).sum() / total)
    delta_x = grid_x - mean_x
    delta_y = grid_y - mean_y
    covariance = np.asarray(
        [
            [(weight * delta_x * delta_x).sum(), (weight * delta_x * delta_y).sum()],
            [(weight * delta_x * delta_y).sum(), (weight * delta_y * delta_y).sum()],
        ],
        dtype=np.float64,
    ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[0] <= 0.0 or eigenvalues[1] >= 100.0:
        return None
    major_vector = eigenvectors[:, 1]
    angle = math.degrees(math.atan2(float(major_vector[1]), float(major_vector[0])))
    fwhm = 2.354820045 * np.sqrt(eigenvalues)
    return float(fwhm[0]), float(fwhm[1]), angle


def _axis_neighbor_spacing(centers: np.ndarray) -> tuple[float, float]:
    horizontal: list[float] = []
    vertical: list[float] = []
    for first in range(len(centers)):
        horizontal_candidates: list[float] = []
        vertical_candidates: list[float] = []
        for second in range(len(centers)):
            if first == second:
                continue
            delta_x, delta_y = centers[second] - centers[first]
            if abs(delta_x) > 3.0 and abs(delta_x) > 2.0 * abs(delta_y):
                horizontal_candidates.append(abs(float(delta_x)))
            if abs(delta_y) > 3.0 and abs(delta_y) > 2.0 * abs(delta_x):
                vertical_candidates.append(abs(float(delta_y)))
        if horizontal_candidates:
            horizontal.append(min(horizontal_candidates))
        if vertical_candidates:
            vertical.append(min(vertical_candidates))
    if len(horizontal) < 3 or len(vertical) < 3:
        raise ValueError("could not recover both grid axes")
    return float(np.median(horizontal)), float(np.median(vertical))


def _quartiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": round(float(np.median(array)), 6),
        "q25": round(float(np.percentile(array, 25)), 6),
        "q75": round(float(np.percentile(array, 75)), 6),
    }


def fit_point_grid(frames: Sequence[np.ndarray], features: Mapping[str, Any]) -> FitResult:
    spacing_source = float(features.get("point_spacing_px", 0.0))
    if spacing_source <= 0.0:
        return not_identifiable("point spacing is missing from the manifest")
    frame_scale_x: list[float] = []
    frame_scale_y: list[float] = []
    shapes: list[tuple[float, float, float]] = []
    detected_counts: list[int] = []
    for frame in frames:
        score = 0.5 * frame[:, :, 1] + 0.5 * frame[:, :, 2] - 0.4 * frame[:, :, 0]
        components = _point_components(score)
        detected_counts.append(len(components))
        centers = np.asarray([component["center"] for component in components], np.float64)
        patch_radius = 12
        if len(centers) >= 6:
            pairwise = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
            pairwise[pairwise == 0.0] = np.inf
            nearest_spacing = float(np.median(pairwise.min(axis=1)))
            patch_radius = min(12, max(4, int(nearest_spacing // 2) - 1))
            try:
                spacing_x, spacing_y = _axis_neighbor_spacing(centers)
            except ValueError:
                pass
            else:
                frame_scale_x.append(spacing_x / spacing_source)
                frame_scale_y.append(spacing_y / spacing_source)
        for component in components:
            shape = _dot_shape(score, component, patch_radius)
            if shape is not None:
                shapes.append(shape)

    if not frame_scale_x or not frame_scale_y or not shapes:
        return _result(
            "not_identifiable",
            "too few isolated points to estimate scale and PSF",
            sum(detected_counts),
            None,
            "camera pixels",
            "path ROI",
        )
    shape_array = np.asarray(shapes, dtype=np.float64)
    scale = (float(np.median(frame_scale_x)), float(np.median(frame_scale_y)))
    fwhm = tuple(float(value) for value in np.median(shape_array[:, :2], axis=0))
    return _result(
        "estimated",
        "isolated point-grid components support local scale and effective PSF",
        len(shapes),
        {
            "scale_camera_per_source": scale,
            "fwhm_camera_px": fwhm,
            "fwhm_source_px": (
                fwhm[0] / max(scale),
                fwhm[1] / min(scale),
            ),
            "major_axis_angle_deg": float(np.median(shape_array[:, 2])),
            "points_per_frame_range": (min(detected_counts), max(detected_counts)),
        },
        "pixels and degrees",
        "path ROI; FWHM sorted by minor/major covariance axis",
        {
            "scale_x": _quartiles(frame_scale_x),
            "scale_y": _quartiles(frame_scale_y),
            "fwhm_minor": _quartiles(shape_array[:, 0]),
            "fwhm_major": _quartiles(shape_array[:, 1]),
            "interval_type": "descriptive burst IQR",
        },
    )


def _checker_components(score: np.ndarray) -> list[dict[str, Any]]:
    components = _components(score > (145.0 / 255.0), 300, 12_000)
    selected: list[dict[str, Any]] = []
    for component in components:
        left, top, right, bottom = component["bbox"]
        width = right - left + 1
        height = bottom - top + 1
        ratio = width / max(height, 1)
        if 20 <= width <= 130 and 20 <= height <= 130 and 0.45 <= ratio <= 2.2:
            selected.append(component)
    return selected


def _peak_fwhm(profile: np.ndarray) -> list[int]:
    if len(profile) < 32:
        return []
    peaks: list[int] = []
    threshold = float(np.percentile(profile, 70))
    for index in range(12, len(profile) - 12):
        if profile[index] < threshold:
            continue
        local = profile[index - 6 : index + 7]
        if profile[index] != local.max():
            continue
        if not peaks or index - peaks[-1] > 20:
            peaks.append(index)
        elif profile[index] > profile[peaks[-1]]:
            peaks[-1] = index
    widths: list[int] = []
    for index in peaks:
        local = profile[index - 10 : index + 11]
        baseline = float(np.percentile(local, 10))
        half = baseline + (float(profile[index]) - baseline) / 2.0
        left = index
        while left > index - 10 and profile[left] >= half:
            left -= 1
        right = index
        while right < index + 10 and profile[right] >= half:
            right += 1
        widths.append(right - left - 1)
    return widths


def _edge_widths(frame: np.ndarray) -> list[int]:
    gray = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_RGB2GRAY)
    margin_y = min(80, gray.shape[0] // 6)
    margin_x = min(80, gray.shape[1] // 6)
    center = gray[margin_y : gray.shape[0] - margin_y, margin_x : gray.shape[1] - margin_x]
    if center.size == 0:
        return []
    vertical = np.abs(np.diff(center, axis=1)).mean(axis=0)
    horizontal = np.abs(np.diff(center, axis=0)).mean(axis=1)
    return _peak_fwhm(vertical) + _peak_fwhm(horizontal)


def fit_checker_geometry(
    frames: Sequence[np.ndarray], features: Mapping[str, Any]
) -> FitResult:
    cell_size = float(features.get("cell_size_px", 0.0))
    if cell_size <= 0.0:
        return not_identifiable("checker cell size is missing from the manifest")
    scales_x: list[float] = []
    scales_y: list[float] = []
    widths: list[int] = []
    center_counts: list[int] = []
    for frame in frames:
        score = 0.5 * frame[:, :, 1] + 0.5 * frame[:, :, 2] - 0.25 * frame[:, :, 0]
        components = _checker_components(score)
        center_counts.append(len(components))
        centers = np.asarray([component["center"] for component in components], np.float64)
        if len(centers) >= 6:
            try:
                spacing_x, spacing_y = _axis_neighbor_spacing(centers)
            except ValueError:
                pass
            else:
                scales_x.append(spacing_x / (2.0 * cell_size))
                scales_y.append(spacing_y / (2.0 * cell_size))
        widths.extend(_edge_widths(frame))
    if not scales_x or not scales_y:
        return _result(
            "not_identifiable",
            "periodic checker did not yield enough square centers",
            sum(center_counts),
            None,
            "camera pixels per source pixel",
            "path ROI",
        )
    edge_width = float(np.median(widths)) if widths else None
    value = {
        "scale_camera_per_source": (
            float(np.median(scales_x)),
            float(np.median(scales_y)),
        ),
        "edge_gradient_fwhm_camera_px": edge_width,
        "absolute_source_crop": None,
        "absolute_source_crop_status": "not_identifiable",
        "square_centers_per_frame_range": (min(center_counts), max(center_counts)),
    }
    return _result(
        "estimated",
        "checker supports local scale and edge width but not absolute periodic crop",
        sum(center_counts),
        value,
        "pixels",
        "path ROI",
        {
            "scale_x": _quartiles(scales_x),
            "scale_y": _quartiles(scales_y),
            "edge_width": _quartiles(widths) if widths else {},
            "interval_type": "descriptive burst IQR",
        },
    )


def fit_edge_width(frames: Sequence[np.ndarray], features: Mapping[str, Any]) -> FitResult:
    widths = [width for frame in frames for width in _edge_widths(frame)]
    if not widths:
        return not_identifiable("no stable edge-gradient peaks were found")
    return _result(
        "provisional",
        "ROI contains the slanted-edge chart, but exact patch correspondences are unavailable",
        len(widths),
        {"edge_gradient_fwhm_camera_px": float(np.median(widths))},
        "camera pixels",
        "path ROI",
        {"edge_width": _quartiles(widths), "interval_type": "descriptive burst IQR"},
    )


def _extract_bars(
    frame: np.ndarray,
    colors: np.ndarray,
    axis: str,
    order: str,
) -> np.ndarray:
    count = len(colors)
    observations: list[np.ndarray] = []
    if axis == "x":
        extent = frame.shape[1]
        other0, other1 = frame.shape[0] // 4, 3 * frame.shape[0] // 4
        for index in range(count):
            left = int((index + 0.25) * extent / count)
            right = int((index + 0.75) * extent / count)
            observations.append(np.median(frame[other0:other1, left:right], axis=(0, 1)))
    elif axis == "y":
        extent = frame.shape[0]
        other0, other1 = frame.shape[1] // 4, 3 * frame.shape[1] // 4
        for index in range(count):
            top = int((index + 0.25) * extent / count)
            bottom = int((index + 0.75) * extent / count)
            observations.append(np.median(frame[top:bottom, other0:other1], axis=(0, 1)))
    else:
        raise ValueError("bar_axis must be 'x' or 'y'")
    result = np.asarray(observations, np.float64)
    if order == "reverse":
        result = result[::-1]
    elif order != "forward":
        raise ValueError("bar_order must be 'forward' or 'reverse'")
    return result


def _fit_diagonal(source: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gains = np.empty(3, np.float64)
    bias = np.empty(3, np.float64)
    for channel in range(3):
        design = np.column_stack((source[:, channel], np.ones(len(source))))
        gains[channel], bias[channel] = np.linalg.lstsq(
            design, observed[:, channel], rcond=None
        )[0]
    return gains, bias


def _fit_full_color(source: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    design = np.column_stack((source, np.ones(len(source))))
    coefficients = np.linalg.lstsq(design, observed, rcond=None)[0]
    return coefficients[:3].T, coefficients[3]


def _color_predict(
    source: np.ndarray,
    model: str,
    first: np.ndarray,
    bias: np.ndarray,
) -> np.ndarray:
    if model == "diagonal":
        return source * first + bias
    return source @ first.T + bias


def _leave_one_out_color(source: np.ndarray, observed: np.ndarray, model: str) -> float:
    errors: list[float] = []
    for held in range(len(source)):
        keep = np.arange(len(source)) != held
        if model == "diagonal":
            first, bias = _fit_diagonal(source[keep], observed[keep])
        else:
            first, bias = _fit_full_color(source[keep], observed[keep])
        prediction = _color_predict(source[held : held + 1], model, first, bias)[0]
        errors.extend(np.abs(prediction - observed[held]).tolist())
    return float(np.mean(errors))


def fit_color_response(
    frames: Sequence[np.ndarray], features: Mapping[str, Any]
) -> FitResult:
    raw_colors = features.get("colors_rgb")
    if not isinstance(raw_colors, list) or len(raw_colors) < 5:
        medians = np.median(np.stack(frames), axis=(0, 1, 2))
        return _result(
            "provisional",
            "uniform target supports only a JPEG-space RGB level for this exposure",
            len(frames),
            {"median_rgb": medians},
            "normalized JPEG code value",
            "path ROI",
        )
    source = np.asarray(raw_colors, dtype=np.float64) / 255.0
    observed_per_frame = [
        _extract_bars(
            frame,
            source,
            str(features.get("bar_axis", "x")),
            str(features.get("bar_order", "forward")),
        )
        for frame in frames
    ]
    observed = np.median(np.stack(observed_per_frame), axis=0)
    diagonal_gain, diagonal_bias = _fit_diagonal(source, observed)
    full_matrix, full_bias = _fit_full_color(source, observed)
    diagonal_error = _leave_one_out_color(source, observed, "diagonal")
    full_error = _leave_one_out_color(source, observed, "full")
    full_is_physical = bool(np.all(full_matrix >= -0.05))
    selected = "full" if full_is_physical and full_error < 0.9 * diagonal_error else "diagonal"
    if selected == "full":
        matrix = full_matrix
        bias = full_bias
        selected_error = full_error
    else:
        matrix = np.diag(diagonal_gain)
        bias = diagonal_bias
        selected_error = diagonal_error
    status = "estimated" if selected_error < 0.08 else "provisional"
    return _result(
        status,
        "color model selected by leave-one-bar-out error and nonnegative response",
        len(frames) * len(source),
        {
            "selected_model": selected,
            "color_matrix": matrix,
            "bias_rgb": bias,
            "leave_one_out_mae": selected_error,
            "diagonal_leave_one_out_mae": diagonal_error,
            "full_leave_one_out_mae": full_error,
            "observed_bar_rgb": observed,
        },
        "normalized JPEG code value",
        "path ROI; one shared exposure per frame",
        {
            "observed_bar_rgb_iqr": np.subtract(
                np.percentile(observed_per_frame, 75, axis=0),
                np.percentile(observed_per_frame, 25, axis=0),
            ),
            "interval_type": "descriptive burst IQR",
        },
    )


def _binned_variance_fit(mean: np.ndarray, variance: np.ndarray) -> tuple[float, float, int]:
    edges = np.linspace(0.02, 0.98, 21)
    bin_x: list[float] = []
    bin_y: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (mean >= low) & (mean < high)
        if int(mask.sum()) >= 40:
            bin_x.append(float(np.median(mean[mask])))
            bin_y.append(float(np.median(variance[mask])))
    if len(bin_x) < 3:
        return 0.0, float(np.median(variance)), len(bin_x)
    design = np.column_stack((bin_x, np.ones(len(bin_x))))
    slope, intercept = np.linalg.lstsq(design, np.asarray(bin_y), rcond=None)[0]
    return max(float(slope), 0.0), max(float(intercept), 0.0), len(bin_x)


def fit_burst_noise(frames: Sequence[np.ndarray], features: Mapping[str, Any]) -> FitResult:
    if len(frames) < 3:
        return not_identifiable("at least three burst frames are required")
    stack = np.stack(frames).astype(np.float32)
    mean = stack.mean(axis=0)
    variance = stack.var(axis=0, ddof=1)
    gray = cv2.cvtColor(mean, cv2.COLOR_RGB2GRAY)
    gradient_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(gradient_x, gradient_y)
    low_gradient = gradient <= np.percentile(gradient, 70)
    slopes: list[float] = []
    intercepts: list[float] = []
    bins: list[int] = []
    for channel in range(3):
        valid = low_gradient & (mean[:, :, channel] > 0.05) & (mean[:, :, channel] < 0.75)
        slope, intercept, count = _binned_variance_fit(
            mean[:, :, channel][valid], variance[:, :, channel][valid]
        )
        slopes.append(slope)
        intercepts.append(intercept)
        bins.append(count)
    dynamic_range = float(np.percentile(mean, 95) - np.percentile(mean, 5))
    status = "estimated" if min(bins) >= 6 and dynamic_range >= 0.3 else "provisional"
    reason = (
        "low-gradient temporal variance supports a signal-dependent JPEG-space fit"
        if status == "estimated"
        else "burst has insufficient within-exposure intensity support for separating slope and intercept"
    )
    return _result(
        status,
        reason,
        len(frames),
        {
            "slope_rgb": slopes,
            "intercept_rgb": intercepts,
            "intensity_bin_count_rgb": bins,
            "dynamic_range": dynamic_range,
            "median_temporal_variance_rgb": np.median(variance, axis=(0, 1)),
        },
        "variance of normalized JPEG code value",
        "registered/nominally static path ROI",
        {"interval_type": "temporal burst estimate; not independent sensor noise"},
    )


def _read_exif(path: Path) -> dict[str, Any]:
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


def _numeric(tags: Mapping[str, Any], key: str) -> float | None:
    value = tags.get(key)
    return float(value) if value is not None else None


@lru_cache(maxsize=1)
def _standard_qtables() -> dict[int, Mapping[int, Sequence[int]]]:
    tables: dict[int, Mapping[int, Sequence[int]]] = {}
    image = Image.new("RGB", (16, 16))
    for quality in range(1, 101):
        buffer = BytesIO()
        image.save(buffer, "JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            tables[quality] = {
                key: tuple(values) for key, values in encoded.quantization.items()
            }
    return tables


def _equivalent_quality(quantization: Mapping[int, Sequence[int]]) -> int:
    def distance(candidate: Mapping[int, Sequence[int]]) -> int:
        if set(candidate) != set(quantization):
            return 1_000_000
        return sum(
            abs(int(first) - int(second))
            for key in quantization
            for first, second in zip(quantization[key], candidate[key])
        )

    return min(_standard_qtables(), key=lambda quality: distance(_standard_qtables()[quality]))


def inspect_jpeg(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        quantization = {
            key: tuple(values) for key, values in image.quantization.items()
        }
        sampling_code = JpegImagePlugin.get_sampling(image)
    sampling = {0: "4:4:4", 1: "4:2:2", 2: "4:2:0"}.get(
        sampling_code, f"unknown:{sampling_code}"
    )
    return {
        "quantization_signature": tuple(
            (key, tuple(values)) for key, values in sorted(quantization.items())
        ),
        "equivalent_quality": _equivalent_quality(quantization),
        "subsampling": sampling,
    }


def _fit_group(group: CaptureGroup) -> dict[str, FitResult]:
    frames = load_rgb_rois(group)
    kind = str(group.features.get("kind", ""))
    results: dict[str, FitResult] = {}
    if "geometry" in group.roles:
        if kind == "point_grid":
            results["geometry"] = fit_point_grid(frames, group.features)
        elif kind == "checkerboard":
            results["geometry"] = fit_checker_geometry(frames, group.features)
        else:
            results["geometry"] = not_identifiable(
                f"feature kind {kind!r} has no geometry estimator"
            )
    if "psf" in group.roles:
        if kind == "point_grid":
            results["psf"] = fit_point_grid(frames, group.features)
        elif kind == "checkerboard":
            checker = fit_checker_geometry(frames, group.features)
            results["psf"] = checker
        else:
            results["psf"] = not_identifiable(f"feature kind {kind!r} has no PSF estimator")
    if "mtf" in group.roles:
        results["mtf"] = fit_edge_width(frames, group.features)
    if "photometry" in group.roles:
        results["photometry"] = fit_color_response(frames, group.features)
    if "noise" in group.roles:
        results["noise"] = fit_burst_noise(frames, group.features)
    if "background" in group.roles:
        median_rgb = np.median(np.stack(frames), axis=(0, 1, 2))
        results["background"] = _result(
            "provisional",
            "black display burst includes ambient light, scattering, auto exposure, and JPEG black level",
            len(frames),
            {"median_rgb": median_rgb},
            "normalized JPEG code value",
            "path ROI",
        )
    if "validation" in group.roles:
        results["validation"] = not_identifiable(
            "validation is computed after synthesizing from fitted domain parameters"
        )
    if "external_validation" in group.roles:
        results["external_validation"] = not_identifiable(
            "neighboring pose/focal capture is reserved for external validation"
        )
    return results


def _group_metadata(group: CaptureGroup) -> dict[str, Any]:
    tags = [_read_exif(frame) for frame in group.frames]
    focal = [value for row in tags if (value := _numeric(row, "FocalLengthIn35mmFilm")) is not None]
    exposure = [value for row in tags if (value := _numeric(row, "ExposureTime")) is not None]
    iso = [value for row in tags if (value := _numeric(row, "ISOSpeedRatings")) is not None]
    white_balance = [value for row in tags if (value := _numeric(row, "WhiteBalance")) is not None]
    return {
        "frame_count": len(group.frames),
        "focal_35mm_median": float(median(focal)) if focal else None,
        "focal_35mm_range": [min(focal), max(focal)] if focal else [],
        "exposure_seconds_range": [min(exposure), max(exposure)] if exposure else [],
        "iso_range": [min(iso), max(iso)] if iso else [],
        "auto_white_balance": any(value == 0 for value in white_balance),
        "roi_xyxy": group.roi_xyxy,
        "roles": sorted(group.roles),
        "feature_kind": group.features.get("kind"),
    }


def _ranked_results(groups: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    rank = {"estimated": 2, "provisional": 1, "not_identifiable": 0}
    candidates = [
        payload[key]
        for payload in groups.values()
        if key in payload and payload[key].get("value") is not None
    ]
    return sorted(
        candidates,
        key=lambda item: (rank.get(str(item.get("status")), -1), int(item.get("sample_count", 0))),
        reverse=True,
    )


def _effective_domain_parameters(
    groups: Mapping[str, Any],
    *,
    jpeg_quality: int,
    jpeg_subsampling: str,
) -> dict[str, Any]:
    geometry = _ranked_results(groups, "geometry")
    psf = _ranked_results(groups, "psf")
    photometry = _ranked_results(groups, "photometry")
    noise = _ranked_results(groups, "noise")
    background = _ranked_results(groups, "background")

    scale: list[float | None] = [None, None]
    scale_status = "not_identifiable"
    for candidate in geometry + psf:
        candidate_scale = candidate["value"].get("scale_camera_per_source")
        if candidate_scale and len(candidate_scale) == 2:
            scale = list(candidate_scale)
            scale_status = str(candidate["status"])
            break

    blur: list[float | None] = [None, None]
    blur_angle: float | None = None
    blur_status = "not_identifiable"
    for candidate in psf + geometry:
        value = candidate["value"]
        if value.get("fwhm_camera_px"):
            blur = list(value["fwhm_camera_px"])
            blur_angle = value.get("major_axis_angle_deg")
            blur_status = str(candidate["status"])
            break
        if value.get("edge_gradient_fwhm_camera_px") is not None:
            width = float(value["edge_gradient_fwhm_camera_px"])
            blur = [width, width]
            blur_angle = 0.0
            blur_status = "provisional"
            break

    color_matrix: list[list[float]] | None = None
    color_bias: list[float] | None = None
    color_status = "not_identifiable"
    for candidate in photometry:
        value = candidate["value"]
        if value.get("color_matrix") is not None:
            color_matrix = value["color_matrix"]
            color_bias = value["bias_rgb"]
            color_status = str(candidate["status"])
            break

    noise_slope: list[float] | None = None
    noise_intercept: list[float] | None = None
    noise_status = "not_identifiable"
    if noise:
        noise_slope = noise[0]["value"].get("slope_rgb")
        noise_intercept = noise[0]["value"].get("intercept_rgb")
        noise_status = str(noise[0]["status"])

    background_rgb: list[float] | None = None
    background_status = "not_identifiable"
    if background:
        background_rgb = background[0]["value"].get("median_rgb")
        background_status = str(background[0]["status"])

    return {
        "scale_camera_per_source": scale,
        "blur_fwhm_camera_px": blur,
        "blur_angle_deg": blur_angle,
        "color_matrix": color_matrix,
        "color_bias_rgb": color_bias,
        "noise_slope_rgb": noise_slope,
        "noise_intercept_rgb": noise_intercept,
        "background_rgb": background_rgb,
        "jpeg_quality": jpeg_quality,
        "jpeg_subsampling": jpeg_subsampling,
        "status": {
            "scale": scale_status,
            "blur": blur_status,
            "color": color_status,
            "noise": noise_status,
            "background": background_status,
        },
    }


def fit_manifest(manifest: CalibrationManifest, *, seed: int) -> dict[str, Any]:
    del seed  # Reserved for deterministic bootstrap resampling and synthesis diagnostics.
    jpeg_rows = [inspect_jpeg(frame) for group in manifest.groups for frame in group.frames]
    quantization_signatures = {row["quantization_signature"] for row in jpeg_rows}
    qualities = {row["equivalent_quality"] for row in jpeg_rows}
    samplings = {row["subsampling"] for row in jpeg_rows}
    if len(quantization_signatures) != 1 or len(qualities) != 1 or len(samplings) != 1:
        raise ValueError("dataset mixes JPEG quantization or subsampling settings")

    domains: dict[str, Any] = {}
    for domain_name in manifest.domains:
        domain_groups = [group for group in manifest.groups if group.domain == domain_name]
        groups: dict[str, Any] = {}
        for group in domain_groups:
            fitted = _fit_group(group)
            groups[group.id] = {
                "metadata": _group_metadata(group),
                **{name: result.to_dict() for name, result in fitted.items()},
            }
        domains[domain_name] = {
            "paths": sorted({group.path for group in domain_groups}),
            "groups": groups,
            "effective_parameters": _effective_domain_parameters(
                groups,
                jpeg_quality=next(iter(qualities)),
                jpeg_subsampling=next(iter(samplings)),
            ),
            "identifiability": {
                "display_camera_response": not_identifiable(
                    "auto exposure and white balance vary across separately displayed targets"
                ).to_dict(),
                "independent_sensor_noise": not_identifiable(
                    "JPEG-only bursts confound sensor noise with demosaicing, denoising, sharpening, and compression"
                ).to_dict(),
                "absolute_reflection_geometry": not_identifiable(
                    "captured geometry charts are periodic and lack an asymmetric absolute orientation marker"
                ).to_dict(),
            },
        }

    return _json_safe(
        {
            "schema_version": "1.0",
            "model_scope": "path-specific effective super-macro JPEG degradation",
            "dataset": {
                "group_count": len(manifest.groups),
                "frame_count": sum(len(group.frames) for group in manifest.groups),
                "jpeg": {
                    "equivalent_quality": next(iter(qualities)),
                    "subsampling": next(iter(samplings)),
                },
            },
            "domains": domains,
        }
    )
