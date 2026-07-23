from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import ndtr


def _quadratic_basis(
    points: np.ndarray, source_size: tuple[int, int]
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    width, height = source_size
    x = 2.0 * points[:, 0] / width - 1.0
    y = 2.0 * points[:, 1] / height - 1.0
    return np.column_stack((np.ones(len(points)), x, y, x * x, x * y, y * y))


def predict_quadratic_warp(
    points: np.ndarray,
    coefficients: np.ndarray,
    *,
    source_size: tuple[int, int],
) -> np.ndarray:
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.shape != (2, 6):
        raise ValueError("coefficients must have shape (2, 6)")
    return _quadratic_basis(points, source_size) @ coefficients.T


def fit_quadratic_warp(
    source_points: np.ndarray,
    observed_points: np.ndarray,
    *,
    source_size: tuple[int, int],
    minimum_inliers: int = 12,
) -> dict[str, Any]:
    source = np.asarray(source_points, dtype=np.float64)
    observed = np.asarray(observed_points, dtype=np.float64)
    if source.shape != observed.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source_points and observed_points must both have shape (n, 2)")
    if len(source) < 6:
        raise ValueError("at least six correspondences are required")

    basis = _quadratic_basis(source, source_size)
    minimum_inliers = max(6, min(int(minimum_inliers), len(source)))
    generator = np.random.default_rng(20260721)
    best_inliers = np.ones(len(source), dtype=bool)
    best_score = (-1, -np.inf)
    for _ in range(768):
        sample = generator.choice(len(source), size=6, replace=False)
        if np.linalg.cond(basis[sample]) > 1e5:
            continue
        candidate = np.linalg.lstsq(basis[sample], observed[sample], rcond=None)[0].T
        candidate_residuals = np.linalg.norm(basis @ candidate.T - observed, axis=1)
        candidate_inliers = candidate_residuals <= 4.0
        count = int(candidate_inliers.sum())
        if count < minimum_inliers:
            continue
        score = (count, -float(np.median(candidate_residuals[candidate_inliers])))
        if score > best_score:
            best_score = score
            best_inliers = candidate_inliers

    inliers = best_inliers.copy()
    while True:
        coefficients = np.linalg.lstsq(basis[inliers], observed[inliers], rcond=None)[0].T
        residuals = np.linalg.norm(basis @ coefficients.T - observed, axis=1)
        active = residuals[inliers]
        median = float(np.median(active))
        mad = float(np.median(np.abs(active - median)))
        threshold = max(1.5, median + 4.0 * 1.4826 * mad)
        worst_index = int(np.argmax(np.where(inliers, residuals, -np.inf)))
        if residuals[worst_index] <= threshold or int(inliers.sum()) <= minimum_inliers:
            break
        inliers[worst_index] = False

    coefficients = np.linalg.lstsq(basis[inliers], observed[inliers], rcond=None)[0].T
    quadratic_residuals = np.linalg.norm(basis @ coefficients.T - observed, axis=1)
    homography, _ = cv2.findHomography(
        source[inliers].astype(np.float64),
        observed[inliers].astype(np.float64),
        method=0,
    )
    if homography is None:
        homography_residuals = np.full(len(source), np.nan, dtype=np.float64)
    else:
        projected = cv2.perspectiveTransform(
            source.reshape(1, -1, 2).astype(np.float64), homography
        ).reshape(-1, 2)
        homography_residuals = np.linalg.norm(projected - observed, axis=1)

    return {
        "coefficients": coefficients,
        "inlier_mask": inliers,
        "residuals_px": quadratic_residuals,
        "quadratic_median_residual_px": float(np.median(quadratic_residuals[inliers])),
        "quadratic_p95_residual_px": float(np.percentile(quadratic_residuals[inliers], 95)),
        "homography": homography,
        "homography_median_residual_px": float(np.nanmedian(homography_residuals[inliers])),
        "homography_p95_residual_px": float(np.nanpercentile(homography_residuals[inliers], 95)),
    }


def quadratic_warp_jacobian(
    point: np.ndarray,
    coefficients: np.ndarray,
    *,
    source_size: tuple[int, int],
) -> np.ndarray:
    x, y = np.asarray(point, dtype=np.float64)
    width, height = source_size
    normalized_x = 2.0 * x / width - 1.0
    normalized_y = 2.0 * y / height - 1.0
    coefficients = np.asarray(coefficients, dtype=np.float64)
    derivative_x = np.asarray(
        [0.0, 2.0 / width, 0.0, 4.0 * normalized_x / width, 2.0 * normalized_y / width, 0.0]
    )
    derivative_y = np.asarray(
        [0.0, 0.0, 2.0 / height, 0.0, 2.0 * normalized_x / height, 4.0 * normalized_y / height]
    )
    return np.column_stack((coefficients @ derivative_x, coefficients @ derivative_y))


def deconvolve_square_psf_covariance(
    observed_covariance: np.ndarray,
    jacobian: np.ndarray,
    *,
    square_side_px: int,
) -> np.ndarray:
    if square_side_px <= 0:
        raise ValueError("square_side_px must be positive")
    observed = np.asarray(observed_covariance, dtype=np.float64)
    jacobian = np.asarray(jacobian, dtype=np.float64)
    if observed.shape != (2, 2) or jacobian.shape != (2, 2):
        raise ValueError("covariance and jacobian must have shape (2, 2)")
    source_variance = (float(square_side_px) ** 2 - 1.0) / 12.0
    covariance = observed - source_variance * jacobian @ jacobian.T
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def _weighted_center_covariance(
    score: np.ndarray, center: np.ndarray, radius: int
) -> tuple[np.ndarray, np.ndarray, float]:
    center_x, center_y = (float(value) for value in center)
    integer_x, integer_y = int(round(center_x)), int(round(center_y))
    y0, y1 = integer_y - radius, integer_y + radius + 1
    x0, x1 = integer_x - radius, integer_x + radius + 1
    patch = score[max(y0, 0) : min(y1, score.shape[0]), max(x0, 0) : min(x1, score.shape[1])]
    if patch.shape != (2 * radius + 1, 2 * radius + 1):
        raise ValueError("point patch intersects the image boundary")
    border = np.concatenate((patch[:2].ravel(), patch[-2:].ravel(), patch[:, :2].ravel(), patch[:, -2:].ravel()))
    background = float(np.median(border))
    weights = np.maximum(patch - background, 0.0)
    total = float(weights.sum())
    if total <= 1e-6:
        raise ValueError("point patch has insufficient contrast")
    grid_y, grid_x = np.indices(patch.shape, dtype=np.float64)
    mean = np.asarray(
        [(weights * grid_x).sum() / total, (weights * grid_y).sum() / total]
    )
    centered_x = grid_x - mean[0]
    centered_y = grid_y - mean[1]
    covariance = np.asarray(
        [
            [(weights * centered_x**2).sum(), (weights * centered_x * centered_y).sum()],
            [(weights * centered_x * centered_y).sum(), (weights * centered_y**2).sum()],
        ]
    ) / total
    global_center = mean + np.asarray((x0, y0), dtype=np.float64)
    contrast = float(patch.max() - background)
    return global_center, covariance, contrast


def _covariance_shape(covariance: np.ndarray) -> tuple[float, float, float]:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    fwhm = 2.354820045 * np.sqrt(eigenvalues)
    major_vector = eigenvectors[:, 1]
    angle = np.degrees(np.arctan2(major_vector[1], major_vector[0]))
    return float(fwhm[0]), float(fwhm[1]), float(angle)


def fit_point_grid_capture(
    image_rgb: np.ndarray,
    *,
    source_size: tuple[int, int],
    anchor_corners: np.ndarray,
    channel_index: int | None,
    square_side_px: int,
    search_radius: int = 35,
    measurement_radius: int | None = None,
) -> dict[str, Any]:
    image = np.asarray(image_rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if channel_index is None:
        score = image.max(axis=2)
    elif channel_index in {0, 1, 2}:
        score = image[:, :, channel_index]
    else:
        raise ValueError("channel_index must be 0, 1, 2, or None")

    width, height = source_size
    fractions = (0.1, 0.3, 0.5, 0.7, 0.9)
    source = np.asarray(
        [
            (column * (width - 1), row * (height - 1))
            for row in fractions
            for column in fractions
        ],
        dtype=np.float64,
    )
    corner_indices = (0, 4, 20, 24)
    anchor_corners = np.asarray(anchor_corners, dtype=np.float64)
    if anchor_corners.shape != (4, 2):
        raise ValueError("anchor_corners must have shape (4, 2)")
    seed_homography = cv2.getPerspectiveTransform(
        source[list(corner_indices)].astype(np.float32), anchor_corners.astype(np.float32)
    )
    predictions = cv2.perspectiveTransform(
        source.reshape(1, -1, 2).astype(np.float32), seed_homography
    ).reshape(-1, 2)
    top_hat = score - cv2.GaussianBlur(score, (0, 0), 12.0)

    centers: list[np.ndarray] = []
    observed_covariances: list[np.ndarray] = []
    contrasts: list[float] = []
    if measurement_radius is None:
        patch_radius = min(16, max(5, search_radius - 2))
    else:
        patch_radius = int(measurement_radius)
        if patch_radius < 5 or patch_radius >= search_radius:
            raise ValueError("measurement_radius must be at least 5 and smaller than search_radius")
    for prediction in predictions:
        predicted_x, predicted_y = (int(round(value)) for value in prediction)
        x0 = max(0, predicted_x - search_radius)
        x1 = min(score.shape[1], predicted_x + search_radius + 1)
        y0 = max(0, predicted_y - search_radius)
        y1 = min(score.shape[0], predicted_y + search_radius + 1)
        search = top_hat[y0:y1, x0:x1]
        if search.size == 0:
            raise ValueError("an anchor projects outside the image")
        peak_y, peak_x = np.unravel_index(int(np.argmax(search)), search.shape)
        peak = np.asarray((x0 + peak_x, y0 + peak_y), dtype=np.float64)
        center, covariance, contrast = _weighted_center_covariance(score, peak, patch_radius)
        centers.append(center)
        observed_covariances.append(covariance)
        contrasts.append(contrast)

    observed = np.asarray(centers, dtype=np.float64)
    warp = fit_quadratic_warp(
        source,
        observed,
        source_size=source_size,
        minimum_inliers=12,
    )
    point_results: list[dict[str, Any]] = []
    for index, (source_point, observed_point, observed_covariance, contrast) in enumerate(
        zip(source, observed, observed_covariances, contrasts)
    ):
        jacobian = quadratic_warp_jacobian(
            source_point, warp["coefficients"], source_size=source_size
        )
        psf_covariance = deconvolve_square_psf_covariance(
            observed_covariance, jacobian, square_side_px=square_side_px
        )
        point_results.append(
            {
                "index": index,
                "source_xy": source_point,
                "observed_xy": observed_point,
                "inlier": bool(warp["inlier_mask"][index]),
                "contrast": contrast,
                "jacobian": jacobian,
                "scale_singular_values": np.linalg.svd(jacobian, compute_uv=False)[::-1],
                "observed_covariance": observed_covariance,
                "psf_covariance": psf_covariance,
                "psf_fwhm_minor_major_angle": _covariance_shape(psf_covariance),
            }
        )
    return {
        "detected_count": len(observed),
        "measurement_radius": patch_radius,
        "source_points": source,
        "observed_points": observed,
        "warp": warp,
        "points": point_results,
    }


def _field_basis(
    source_xy: np.ndarray, source_size: tuple[int, int]
) -> np.ndarray:
    points = np.asarray(source_xy, dtype=np.float64)
    width, height = source_size
    x = 2.0 * points[:, 0] / width - 1.0
    y = 2.0 * points[:, 1] / height - 1.0
    return np.column_stack((x, y, x * x, x * y, y * y))


def _robust_ridge_lstsq(
    design: np.ndarray,
    target: np.ndarray,
    ridge: np.ndarray,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    ridge = np.asarray(ridge, dtype=np.float64)
    coefficients = np.linalg.lstsq(
        np.vstack((design, np.diag(np.sqrt(ridge)))),
        np.concatenate((target, np.zeros(design.shape[1]))),
        rcond=None,
    )[0]
    for _ in range(12):
        residual = design @ coefficients - target
        center = float(np.median(residual))
        scale = 1.4826 * float(np.median(np.abs(residual - center)))
        if scale <= 1e-8:
            break
        limit = 1.5 * scale
        weights = np.minimum(1.0, limit / np.maximum(np.abs(residual - center), 1e-12))
        weighted_design = design * np.sqrt(weights)[:, None]
        weighted_target = target * np.sqrt(weights)
        updated = np.linalg.lstsq(
            np.vstack((weighted_design, np.diag(np.sqrt(ridge)))),
            np.concatenate((weighted_target, np.zeros(design.shape[1]))),
            rcond=None,
        )[0]
        if np.max(np.abs(updated - coefficients)) < 1e-9:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def fit_black_background_field(
    observed_rgb: np.ndarray,
    source_xy: np.ndarray,
    frame_indices: np.ndarray,
    *,
    source_size: tuple[int, int],
    ridge: float = 1e-5,
) -> dict[str, Any]:
    observed = np.asarray(observed_rgb, dtype=np.float64)
    positions = np.asarray(source_xy, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed_rgb must have shape (n, 3)")
    if positions.shape != (len(observed), 2) or frames.shape != (len(observed),):
        raise ValueError("source_xy and frame_indices must align with observed_rgb")
    unique_frames = np.unique(frames)
    if not np.array_equal(unique_frames, np.arange(len(unique_frames))):
        raise ValueError("frame_indices must be contiguous and start at zero")

    basis = _quadratic_basis(positions, source_size)
    coefficients = np.empty((len(unique_frames), 3, 6), dtype=np.float64)
    prediction = np.empty_like(observed)
    ridge_vector = np.asarray((0.0, ridge, ridge, ridge, ridge, ridge))
    for frame in unique_frames:
        rows = frames == frame
        if np.count_nonzero(rows) < 6:
            raise ValueError("each background frame requires at least six samples")
        for channel in range(3):
            coefficients[frame, channel] = _robust_ridge_lstsq(
                basis[rows], observed[rows, channel], ridge_vector
            )
        prediction[rows] = basis[rows] @ coefficients[frame].T
    residual = prediction - observed
    return {
        "basis": ("1", "x", "y", "x^2", "x*y", "y^2"),
        "coefficients": coefficients,
        "prediction_rgb": prediction,
        "residual_rgb": residual,
        "mae": float(np.mean(np.abs(residual))),
        "per_frame_mae": np.asarray(
            [np.mean(np.abs(residual[frames == frame])) for frame in unique_frames]
        ),
    }


def fit_edge_spread(distance_px: np.ndarray, intensity: np.ndarray) -> dict[str, float]:
    distance = np.asarray(distance_px, dtype=np.float64).ravel()
    values = np.asarray(intensity, dtype=np.float64).ravel()
    valid = np.isfinite(distance) & np.isfinite(values)
    distance = distance[valid]
    values = values[valid]
    if len(distance) < 80:
        raise ValueError("at least 80 finite edge samples are required")
    span = float(np.ptp(distance))
    if span <= 1.0 or float(np.ptp(values)) <= 1e-4:
        raise ValueError("edge samples have insufficient range")

    order = np.argsort(distance)
    tail_count = max(12, len(distance) // 8)
    low = float(np.median(values[order[:tail_count]]))
    high = float(np.median(values[order[-tail_count:]]))
    amplitude = high - low
    midpoint = low + 0.5 * amplitude
    center = float(distance[np.argmin(np.abs(values - midpoint))])
    initial = np.asarray((low, amplitude, center, max(0.8, span / 18.0)))

    def predict(parameters: np.ndarray) -> np.ndarray:
        baseline, scale, location, sigma = parameters
        return baseline + scale * ndtr((distance - location) / sigma)

    scale_hint = max(0.005, 0.03 * abs(amplitude))
    optimization = least_squares(
        lambda parameters: predict(parameters) - values,
        initial,
        bounds=(
            (-1.0, -2.0, float(distance.min()), 0.15),
            (2.0, 2.0, float(distance.max()), max(1.0, span)),
        ),
        loss="soft_l1",
        f_scale=scale_hint,
        max_nfev=500,
    )
    if not optimization.success:
        raise ValueError(f"edge-spread fit failed: {optimization.message}")
    baseline, amplitude, center, sigma = optimization.x
    fitted = predict(optimization.x)
    residual = fitted - values
    total = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0.0 else 0.0
    return {
        "baseline": float(baseline),
        "amplitude": float(amplitude),
        "center_px": float(center),
        "sigma_px": float(sigma),
        "fwhm_px": float(2.354820045 * sigma),
        "mae": float(np.mean(np.abs(residual))),
        "r_squared": r_squared,
        "sample_count": int(len(distance)),
    }


def measure_slanted_edge_cells(
    image_rgb: np.ndarray,
    homography_source_to_camera: np.ndarray,
    cells: list[dict[str, Any]],
    *,
    max_distance_px: float = 30.0,
) -> dict[str, Any]:
    image = np.asarray(image_rgb, dtype=np.float64)
    homography = np.asarray(homography_source_to_camera, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if homography.shape != (3, 3):
        raise ValueError("homography_source_to_camera must have shape (3, 3)")
    if max_distance_px <= 0.0:
        raise ValueError("max_distance_px must be positive")

    results: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        x0, y0, x1, y1 = (float(value) for value in cell["bbox_xyxy"])
        center_x = 0.5 * (x0 + x1)
        center_y = 0.5 * (y0 + y1)
        slope = float(cell["slope"])
        orientation = str(cell["orientation"])
        if orientation.startswith("vertical"):
            source_line = np.asarray(
                (
                    (center_x + slope * (y0 - center_y), y0),
                    (center_x + slope * (y1 - center_y), y1),
                ),
                dtype=np.float64,
            )
        elif orientation.startswith("horizontal"):
            source_line = np.asarray(
                (
                    (x0, center_y + slope * (x0 - center_x)),
                    (x1, center_y + slope * (x1 - center_x)),
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(f"unsupported edge orientation: {orientation}")

        margin_x = 0.1 * (x1 - x0)
        margin_y = 0.1 * (y1 - y0)
        source_polygon = np.asarray(
            (
                (x0 + margin_x, y0 + margin_y),
                (x1 - margin_x, y0 + margin_y),
                (x1 - margin_x, y1 - margin_y),
                (x0 + margin_x, y1 - margin_y),
            ),
            dtype=np.float64,
        )
        projected_line = cv2.perspectiveTransform(
            source_line.reshape(1, -1, 2), homography
        )[0]
        projected_polygon = cv2.perspectiveTransform(
            source_polygon.reshape(1, -1, 2), homography
        )[0]
        left = max(0, int(np.floor(projected_polygon[:, 0].min())))
        right = min(image.shape[1], int(np.ceil(projected_polygon[:, 0].max())) + 1)
        top = max(0, int(np.floor(projected_polygon[:, 1].min())))
        bottom = min(image.shape[0], int(np.ceil(projected_polygon[:, 1].max())) + 1)
        if right - left < 10 or bottom - top < 10:
            raise ValueError("projected edge cell is outside the camera image")

        polygon_mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
        local_polygon = np.rint(projected_polygon - (left, top)).astype(np.int32)
        cv2.fillConvexPoly(polygon_mask, local_polygon, 1)
        grid_y, grid_x = np.indices(polygon_mask.shape, dtype=np.float64)
        grid_x += left
        grid_y += top
        line_start, line_end = projected_line
        direction = line_end - line_start
        length = float(np.linalg.norm(direction))
        if length <= 1e-8:
            raise ValueError("projected edge line is degenerate")
        distance = (
            (grid_x - line_start[0]) * direction[1]
            - (grid_y - line_start[1]) * direction[0]
        ) / length
        sample_mask = (polygon_mask > 0) & (np.abs(distance) <= max_distance_px)
        channel = {"R": 0, "G": 1, "B": 2}[str(cell["color_id"])]
        fit = fit_edge_spread(
            distance[sample_mask], image[top:bottom, left:right, channel][sample_mask]
        )
        accepted = bool(
            fit["r_squared"] >= 0.85
            and abs(fit["amplitude"]) >= 0.03
            and fit["sigma_px"] < 0.5 * max_distance_px
        )
        results.append({"index": index, **cell, **fit, "accepted": accepted})

    accepted_rows = [row for row in results if row["accepted"]]
    summary: dict[str, Any] = {
        "cell_count": len(results),
        "accepted_count": len(accepted_rows),
        "cells": results,
        "by_color": {},
    }
    for color_id in ("G", "B"):
        values = np.asarray(
            [row["fwhm_px"] for row in accepted_rows if row["color_id"] == color_id]
        )
        if len(values):
            summary["by_color"][color_id] = {
                "count": len(values),
                "q25": float(np.percentile(values, 25)),
                "median": float(np.median(values)),
                "q75": float(np.percentile(values, 75)),
            }
    return summary


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    blocks: list[list[float]] = []
    for index, value in enumerate(np.asarray(values, dtype=np.float64)):
        weight = 1e6 if index == len(values) - 1 else 1.0
        blocks.append([float(value), weight, 1.0])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            combined_weight = left[1] + right[1]
            blocks.append(
                [
                    (left[0] * left[1] + right[0] * right[1]) / combined_weight,
                    combined_weight,
                    left[2] + right[2],
                ]
            )
    fitted = np.concatenate(
        [np.repeat(block[0], int(block[2])) for block in blocks]
    )
    return fitted - fitted[-1]


def refine_repeated_patch_grid(
    image_rgb: np.ndarray,
    *,
    source_centers: np.ndarray,
    patch_ids: list[str],
    source_grid_corners: np.ndarray,
    initial_camera_corners: np.ndarray,
    max_adjustment_px: float = 70.0,
    sampling_radius: int = 9,
) -> dict[str, Any]:
    image = np.asarray(image_rgb, dtype=np.float64)
    source_centers = np.asarray(source_centers, dtype=np.float32)
    source_corners = np.asarray(source_grid_corners, dtype=np.float32)
    initial = np.asarray(initial_camera_corners, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if source_centers.ndim != 2 or source_centers.shape[1] != 2:
        raise ValueError("source_centers must have shape (n, 2)")
    if len(patch_ids) != len(source_centers):
        raise ValueError("patch_ids must align with source_centers")
    if source_corners.shape != (4, 2) or initial.shape != (4, 2):
        raise ValueError("source and camera grid corners must have shape (4, 2)")

    smooth = cv2.medianBlur(
        np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8), 19
    ).astype(np.float64) / 255.0
    labels = np.asarray(patch_ids)
    unique_labels = np.unique(labels)

    def project(flat_corners: np.ndarray) -> np.ndarray:
        homography = cv2.getPerspectiveTransform(
            source_corners, flat_corners.reshape(4, 2).astype(np.float32)
        )
        return cv2.perspectiveTransform(source_centers[None], homography)[0]

    def smooth_samples(points: np.ndarray) -> np.ndarray:
        return cv2.remap(
            smooth,
            points[:, 0].astype(np.float32).reshape(-1, 1),
            points[:, 1].astype(np.float32).reshape(-1, 1),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        ).reshape(-1, 3)

    def objective(flat_corners: np.ndarray) -> float:
        values = smooth_samples(project(flat_corners))
        within_class = 0.0
        for label in unique_labels:
            group = values[labels == label]
            within_class += float(
                np.mean(np.abs(group - np.median(group, axis=0)))
            )
        regularization = 2e-5 * float(
            np.mean((flat_corners - initial.ravel()) ** 2)
        )
        return within_class + regularization

    initial_flat = initial.ravel()
    bounds = [
        (value - max_adjustment_px, value + max_adjustment_px)
        for value in initial_flat
    ]
    translation_values = np.arange(
        -max_adjustment_px, max_adjustment_px + 0.1, 4.0, dtype=np.float64
    )
    seed = initial.copy()
    seed_objective = objective(initial_flat)
    seed_translation = np.zeros(2, dtype=np.float64)
    for dx in translation_values:
        for dy in translation_values:
            candidate = initial + (dx, dy)
            candidate_objective = objective(candidate.ravel())
            if candidate_objective < seed_objective:
                seed = candidate
                seed_objective = candidate_objective
                seed_translation = np.asarray((dx, dy), dtype=np.float64)
    optimization = minimize(
        objective,
        seed.ravel(),
        method="Powell",
        bounds=bounds,
        options={"maxiter": 180, "xtol": 0.05, "ftol": 1e-7},
    )
    if not optimization.success:
        raise ValueError(f"response grid refinement failed: {optimization.message}")
    refined_corners = optimization.x.reshape(4, 2)
    homography = cv2.getPerspectiveTransform(
        source_corners, refined_corners.astype(np.float32)
    )
    points = cv2.perspectiveTransform(source_centers[None], homography)[0]
    values = []
    for x, y in points:
        integer_x, integer_y = int(round(x)), int(round(y))
        patch = image[
            integer_y - sampling_radius : integer_y + sampling_radius + 1,
            integer_x - sampling_radius : integer_x + sampling_radius + 1,
        ]
        if patch.shape[:2] != (2 * sampling_radius + 1, 2 * sampling_radius + 1):
            raise ValueError("a refined response sample intersects the image boundary")
        values.append(np.median(patch, axis=(0, 1)))
    return {
        "initial_objective": objective(initial_flat),
        "translation_seed_objective": seed_objective,
        "translation_seed_xy": seed_translation,
        "refined_objective": objective(optimization.x),
        "initial_camera_corners": initial,
        "refined_camera_corners": refined_corners,
        "corner_adjustment_px": refined_corners - initial,
        "homography_source_to_camera": homography,
        "observed_centers": points,
        "observed_rgb": np.asarray(values),
    }


def fit_permuted_photometry(
    observed_rgb: np.ndarray,
    source_rgb: np.ndarray,
    source_xy: np.ndarray,
    frame_indices: np.ndarray,
    *,
    source_size: tuple[int, int],
    spatial_ridge: float = 0.04,
) -> dict[str, Any]:
    """Fit a JPEG-space response under explicit scale and black-level gauges.

    The fitted model is y = b_t(u) + g_tc m_tc(u) f_c((A x)_c).
    Background is affine per frame, log-m is quadratic per frame with zero
    sample mean, f(0)=0, f(1)=1, and each row of A is nonnegative and sums to 1.
    """
    observed = np.asarray(observed_rgb, dtype=np.float64)
    source = np.asarray(source_rgb, dtype=np.float64)
    positions = np.asarray(source_xy, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if observed.shape != source.shape or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed_rgb and source_rgb must both have shape (n, 3)")
    if positions.shape != (len(observed), 2) or frames.shape != (len(observed),):
        raise ValueError("source_xy and frame_indices must align with RGB samples")
    unique_frames = np.unique(frames)
    if not np.array_equal(unique_frames, np.arange(len(unique_frames))):
        raise ValueError("frame_indices must be contiguous and start at zero")

    raw_basis = _field_basis(positions, source_size)
    basis_centers = np.vstack(
        [raw_basis[frames == frame].mean(axis=0) for frame in unique_frames]
    )
    centered_basis = raw_basis - basis_centers[frames]
    gray_mask = np.max(np.abs(source - source[:, :1]), axis=1) < 1e-10
    black_mask = gray_mask & (source[:, 0] < 1e-10)
    if any(np.count_nonzero(black_mask & (frames == frame)) < 3 for frame in unique_frames):
        raise ValueError("at least three black patches per frame are required")

    background_coefficients = np.empty((len(unique_frames), 3, 3), dtype=np.float64)
    background = np.empty_like(observed)
    for frame in unique_frames:
        frame_rows = frames == frame
        x = 2.0 * positions[:, 0] / source_size[0] - 1.0
        y = 2.0 * positions[:, 1] / source_size[1] - 1.0
        affine = np.column_stack((np.ones(len(observed)), x, y))
        fit_rows = black_mask & frame_rows
        for channel in range(3):
            coefficients = np.linalg.lstsq(
                affine[fit_rows], observed[fit_rows, channel], rcond=None
            )[0]
            background_coefficients[frame, channel] = coefficients
            background[frame_rows, channel] = affine[frame_rows] @ coefficients

    nonblack_gray = gray_mask & ~black_mask
    gray_levels = np.unique(source[gray_mask, 0])
    nonblack_levels = gray_levels[gray_levels > 0.0]
    if len(nonblack_levels) < 3 or abs(nonblack_levels[-1] - 1.0) > 1e-8:
        raise ValueError("gray patches must include multiple nonzero levels and white")
    gray_indices = np.flatnonzero(nonblack_gray)
    frame_count = len(unique_frames)
    tone_column_count = len(nonblack_levels) - 1
    column_count = frame_count + 5 * frame_count + tone_column_count
    design = np.zeros((len(gray_indices), column_count), dtype=np.float64)
    for row, sample_index in enumerate(gray_indices):
        frame = int(frames[sample_index])
        design[row, frame] = 1.0
        start = frame_count + 5 * frame
        design[row, start : start + 5] = centered_basis[sample_index]
        level = source[sample_index, 0]
        match = np.flatnonzero(np.isclose(nonblack_levels[:-1], level))
        if len(match):
            design[row, frame_count + 5 * frame_count + int(match[0])] = 1.0
    ridge = np.zeros(column_count, dtype=np.float64)
    ridge[frame_count : frame_count + 5 * frame_count] = float(spatial_ridge)

    frame_gain = np.empty((frame_count, 3), dtype=np.float64)
    spatial_coefficients = np.empty((frame_count, 3, 5), dtype=np.float64)
    spatial = np.empty_like(observed)
    tone_values = np.empty((3, len(gray_levels)), dtype=np.float64)
    raw_tone_values = np.empty_like(tone_values)
    tone_monotonic: list[bool] = []
    for channel in range(3):
        corrected = np.maximum(
            observed[gray_indices, channel] - background[gray_indices, channel],
            1e-5,
        )
        coefficients = _robust_ridge_lstsq(design, np.log(corrected), ridge)
        frame_gain[:, channel] = np.exp(coefficients[:frame_count])
        for frame in unique_frames:
            start = frame_count + 5 * frame
            spatial_coefficients[frame, channel] = coefficients[start : start + 5]
        raw_log_tone = np.concatenate(
            (
                coefficients[frame_count + 5 * frame_count :],
                np.asarray((0.0,)),
            )
        )
        monotonic = bool(np.all(np.diff(raw_log_tone) >= -1e-8))
        tone_monotonic.append(monotonic)
        raw_tone_values[channel] = np.concatenate(((0.0,), np.exp(raw_log_tone)))
        log_tone = _isotonic_non_decreasing(raw_log_tone)
        tone_values[channel] = np.concatenate(((0.0,), np.exp(log_tone)))
        for frame in unique_frames:
            frame_rows = frames == frame
            spatial[frame_rows, channel] = np.exp(
                centered_basis[frame_rows]
                @ spatial_coefficients[frame, channel]
            )

    normalized_observed = np.maximum(
        (observed - background) / (frame_gain[frames] * spatial), 0.0
    )
    latent_observed = np.empty_like(normalized_observed)
    for channel in range(3):
        interpolation_x = np.maximum.accumulate(tone_values[channel])
        interpolation_x = interpolation_x + np.arange(len(interpolation_x)) * 1e-10
        latent_observed[:, channel] = np.interp(
            normalized_observed[:, channel], interpolation_x, gray_levels
        )

    matrix = np.empty((3, 3), dtype=np.float64)
    matrix_rows = ~black_mask
    matrix_source = source[matrix_rows]
    for channel in range(3):
        target = latent_observed[matrix_rows, channel]
        objective = lambda row: float(np.mean((matrix_source @ row - target) ** 2))
        optimization = minimize(
            objective,
            np.eye(3)[channel] * 0.8 + np.ones(3) / 15.0,
            method="SLSQP",
            bounds=((0.0, 1.0),) * 3,
            constraints=({"type": "eq", "fun": lambda row: float(row.sum() - 1.0)},),
            options={"ftol": 1e-12, "maxiter": 500},
        )
        if not optimization.success:
            raise ValueError(f"color matrix fit failed: {optimization.message}")
        matrix[channel] = optimization.x

    latent_prediction = np.clip(source @ matrix.T, 0.0, 1.0)
    normalized_prediction = np.empty_like(latent_prediction)
    for channel in range(3):
        normalized_prediction[:, channel] = np.interp(
            latent_prediction[:, channel], gray_levels, tone_values[channel]
        )
    prediction = background + frame_gain[frames] * spatial * normalized_prediction
    residual = prediction - observed
    black_residual = residual[black_mask]
    return {
        "gauge_constraints": {
            "black_response": "f_c(0)=0",
            "white_response": "f_c(1)=1",
            "spatial_scale": "mean log(m_tc) over chart samples = 0",
            "color_scale": "A rows are nonnegative and sum to 1",
        },
        "background_model": "per-frame affine field in normalized source coordinates",
        "background_coefficients_intercept_x_y": background_coefficients,
        "frame_gain_rgb": frame_gain,
        "spatial_log_quadratic_basis": ("x", "y", "x^2", "x*y", "y^2"),
        "spatial_basis_centers": basis_centers,
        "spatial_log_coefficients": spatial_coefficients,
        "gray_levels": gray_levels,
        "tone_values_rgb": tone_values,
        "raw_tone_values_rgb": raw_tone_values,
        "tone_monotonic": tone_monotonic,
        "color_matrix": matrix,
        "gray_design_rank": int(np.linalg.matrix_rank(design)),
        "gray_design_columns": int(design.shape[1]),
        "gray_design_condition": float(np.linalg.cond(design)),
        "fit_mae": float(np.mean(np.abs(residual))),
        "gray_fit_mae": float(np.mean(np.abs(residual[gray_mask]))),
        "color_fit_mae": float(np.mean(np.abs(residual[~gray_mask]))),
        "black_fit_mae": float(np.mean(np.abs(black_residual))),
        "prediction_rgb": prediction,
        "residual_rgb": residual,
        "background_rgb": background,
        "spatial_multiplier_rgb": spatial,
    }


def refine_effective_color_matrix(
    observed_rgb: np.ndarray,
    source_rgb: np.ndarray,
    background_rgb: np.ndarray,
    gain_rgb: np.ndarray,
    spatial_multiplier_rgb: np.ndarray,
    *,
    tone_levels: np.ndarray,
    tone_values_rgb: np.ndarray,
    initial_matrix: np.ndarray,
) -> dict[str, Any]:
    observed = np.asarray(observed_rgb, dtype=np.float64)
    source = np.asarray(source_rgb, dtype=np.float64)
    background = np.asarray(background_rgb, dtype=np.float64)
    gain = np.asarray(gain_rgb, dtype=np.float64)
    spatial = np.asarray(spatial_multiplier_rgb, dtype=np.float64)
    levels = np.asarray(tone_levels, dtype=np.float64)
    tone = np.asarray(tone_values_rgb, dtype=np.float64)
    matrix_initial = np.asarray(initial_matrix, dtype=np.float64)
    if not (
        observed.shape
        == source.shape
        == background.shape
        == gain.shape
        == spatial.shape
    ) or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("photometric arrays must all have shape (n, 3)")
    if tone.shape != (3, len(levels)) or matrix_initial.shape != (3, 3):
        raise ValueError("tone curve or initial matrix has an invalid shape")
    color_mask = (
        np.max(np.abs(source - source[:, :1]), axis=1) > 1e-10
    ) & (np.max(source, axis=1) > 0.0)
    if np.count_nonzero(color_mask) < 6 or np.linalg.matrix_rank(source[color_mask]) < 3:
        raise ValueError("at least six full-rank non-gray samples are required")

    def predict(flat_matrix: np.ndarray) -> np.ndarray:
        matrix = flat_matrix.reshape(3, 3)
        latent = np.clip(source @ matrix.T, 0.0, 1.0)
        normalized = np.empty_like(latent)
        for channel in range(3):
            normalized[:, channel] = np.interp(
                latent[:, channel], levels, tone[channel]
            )
        return background + gain * spatial * normalized

    delta = 0.03

    def objective(flat_matrix: np.ndarray) -> float:
        residual = predict(flat_matrix)[color_mask] - observed[color_mask]
        return float(
            np.mean(delta * delta * (np.sqrt(1.0 + (residual / delta) ** 2) - 1.0))
        )

    initial_flat = matrix_initial.ravel()
    initial_prediction = predict(initial_flat)
    optimization = minimize(
        objective,
        initial_flat,
        method="SLSQP",
        bounds=((-1.5, 2.5),) * 9,
        constraints=(
            {
                "type": "eq",
                "fun": lambda flat_matrix: flat_matrix.reshape(3, 3).sum(axis=1)
                - 1.0,
            },
        ),
        options={"ftol": 1e-13, "maxiter": 1000},
    )
    if not optimization.success:
        raise ValueError(f"color matrix refinement failed: {optimization.message}")
    matrix = optimization.x.reshape(3, 3)
    prediction = predict(optimization.x)
    residual = prediction - observed
    return {
        "constraint": "each matrix row sums to one; signed entries are allowed",
        "color_matrix": matrix,
        "has_negative_entries": bool(np.any(matrix < 0.0)),
        "initial_color_fit_mae": float(
            np.mean(np.abs(initial_prediction[color_mask] - observed[color_mask]))
        ),
        "color_fit_mae": float(np.mean(np.abs(residual[color_mask]))),
        "fit_mae": float(np.mean(np.abs(residual))),
        "prediction_rgb": prediction,
        "residual_rgb": residual,
        "color_sample_count": int(np.count_nonzero(color_mask)),
    }


def fit_joint_color_nodes(
    observed_rgb: np.ndarray,
    source_rgb: np.ndarray,
    source_xy: np.ndarray,
    patch_ids: list[str],
    frame_indices: np.ndarray,
    *,
    source_size: tuple[int, int],
    spatial_ridge: float = 0.2,
) -> dict[str, Any]:
    observed = np.asarray(observed_rgb, dtype=np.float64)
    source = np.asarray(source_rgb, dtype=np.float64)
    positions = np.asarray(source_xy, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    labels = np.asarray(patch_ids)
    if observed.shape != source.shape or observed.ndim != 2 or observed.shape[1] != 3:
        raise ValueError("observed_rgb and source_rgb must both have shape (n, 3)")
    if positions.shape != (len(observed), 2) or frames.shape != (len(observed),):
        raise ValueError("source_xy and frame_indices must align with RGB samples")
    if labels.shape != (len(observed),):
        raise ValueError("patch_ids must align with RGB samples")
    unique_frames = np.unique(frames)
    if not np.array_equal(unique_frames, np.arange(len(unique_frames))):
        raise ValueError("frame_indices must be contiguous and start at zero")

    black_mask = np.max(np.abs(source), axis=1) < 1e-10
    white_mask = np.max(np.abs(source - 1.0), axis=1) < 1e-10
    if any(np.count_nonzero(black_mask & (frames == frame)) < 3 for frame in unique_frames):
        raise ValueError("at least three black patches per frame are required")
    white_labels = np.unique(labels[white_mask])
    if len(white_labels) != 1:
        raise ValueError("exactly one white patch ID is required for the response gauge")
    white_label = str(white_labels[0])

    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    source_by_label: dict[str, np.ndarray] = {}
    for label in unique_labels:
        values = source[labels == label]
        if not np.allclose(values, values[0], atol=1e-12):
            raise ValueError(f"patch ID {label!r} maps to multiple source colors")
        source_by_label[label] = values[0]
    nonblack_labels = [
        label for label in unique_labels if np.max(np.abs(source_by_label[label])) >= 1e-10
    ]
    fitted_labels = [label for label in nonblack_labels if label != white_label]

    width, height = source_size
    normalized_x = 2.0 * positions[:, 0] / width - 1.0
    normalized_y = 2.0 * positions[:, 1] / height - 1.0
    affine = np.column_stack((np.ones(len(observed)), normalized_x, normalized_y))
    background_coefficients = np.empty((len(unique_frames), 3, 3), dtype=np.float64)
    background = np.empty_like(observed)
    for frame in unique_frames:
        frame_rows = frames == frame
        fit_rows = black_mask & frame_rows
        for channel in range(3):
            coefficients = np.linalg.lstsq(
                affine[fit_rows], observed[fit_rows, channel], rcond=None
            )[0]
            background_coefficients[frame, channel] = coefficients
            background[frame_rows, channel] = affine[frame_rows] @ coefficients

    raw_basis = _field_basis(positions, source_size)
    basis_centers = np.vstack(
        [raw_basis[frames == frame].mean(axis=0) for frame in unique_frames]
    )
    centered_basis = raw_basis - basis_centers[frames]
    nonblack_indices = np.flatnonzero(~black_mask)
    frame_count = len(unique_frames)
    column_count = frame_count + 5 * frame_count + len(fitted_labels)
    design = np.zeros((len(nonblack_indices), column_count), dtype=np.float64)
    label_columns = {label: index for index, label in enumerate(fitted_labels)}
    for row, sample_index in enumerate(nonblack_indices):
        frame = int(frames[sample_index])
        design[row, frame] = 1.0
        spatial_start = frame_count + 5 * frame
        design[row, spatial_start : spatial_start + 5] = centered_basis[sample_index]
        label = str(labels[sample_index])
        if label != white_label:
            design[row, frame_count + 5 * frame_count + label_columns[label]] = 1.0
    ridge = np.zeros(column_count, dtype=np.float64)
    ridge[frame_count : frame_count + 5 * frame_count] = float(spatial_ridge)

    frame_gain = np.empty((frame_count, 3), dtype=np.float64)
    spatial_coefficients = np.empty((frame_count, 3, 5), dtype=np.float64)
    node_response = {label: np.zeros(3, dtype=np.float64) for label in unique_labels}
    node_response[white_label] = np.ones(3, dtype=np.float64)
    spatial = np.empty_like(observed)
    for channel in range(3):
        corrected = np.maximum(
            observed[nonblack_indices, channel] - background[nonblack_indices, channel],
            1e-6,
        )
        coefficients = _robust_ridge_lstsq(design, np.log(corrected), ridge)
        frame_gain[:, channel] = np.exp(coefficients[:frame_count])
        for frame in unique_frames:
            start = frame_count + 5 * frame
            spatial_coefficients[frame, channel] = coefficients[start : start + 5]
            frame_rows = frames == frame
            spatial[frame_rows, channel] = np.exp(
                centered_basis[frame_rows] @ spatial_coefficients[frame, channel]
            )
        node_start = frame_count + 5 * frame_count
        for label, column in label_columns.items():
            node_response[label][channel] = np.exp(coefficients[node_start + column])

    response_samples = np.asarray([node_response[str(label)] for label in labels])
    prediction = background + frame_gain[frames] * spatial * response_samples
    residual = prediction - observed
    nodes = [
        {
            "patch_id": label,
            "source_rgb": source_by_label[label],
            "response_rgb": node_response[label],
            "sample_count": int(np.count_nonzero(labels == label)),
        }
        for label in unique_labels
    ]
    return {
        "gauge_constraints": {
            "black_response": "C(0,0,0)=0",
            "white_response": "C(1,1,1)=(1,1,1)",
            "spatial_scale": "mean log(m_tc) over frame samples = 0",
        },
        "background_model": "per-frame affine field in normalized source coordinates",
        "background_coefficients_intercept_x_y": background_coefficients,
        "frame_gain_rgb": frame_gain,
        "spatial_log_quadratic_basis": ("x", "y", "x^2", "x*y", "y^2"),
        "spatial_basis_centers": basis_centers,
        "spatial_log_coefficients": spatial_coefficients,
        "nodes": nodes,
        "design_rank": int(np.linalg.matrix_rank(design)),
        "design_columns": int(design.shape[1]),
        "design_condition": float(np.linalg.cond(design)),
        "fit_mae": float(np.mean(np.abs(residual))),
        "nonblack_fit_mae": float(np.mean(np.abs(residual[~black_mask]))),
        "black_fit_mae": float(np.mean(np.abs(residual[black_mask]))),
        "prediction_rgb": prediction,
        "residual_rgb": residual,
        "background_rgb": background,
        "spatial_multiplier_rgb": spatial,
    }


def fit_constrained_linear_color_matrix(
    source_rgb: np.ndarray,
    response_rgb: np.ndarray,
    *,
    ridge: float = 0.0,
) -> dict[str, Any]:
    source = np.asarray(source_rgb, dtype=np.float64)
    response = np.asarray(response_rgb, dtype=np.float64)
    if source.shape != response.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_rgb and response_rgb must both have shape (n, 3)")
    if len(source) < 4 or np.linalg.matrix_rank(source) < 3:
        raise ValueError("at least four full-rank colors are required")
    constraint = np.ones(3, dtype=np.float64)
    system = np.block(
        [
            [source.T @ source + float(ridge) * np.eye(3), constraint[:, None]],
            [constraint[None, :], np.zeros((1, 1))],
        ]
    )
    matrix = np.empty((3, 3), dtype=np.float64)
    for channel in range(3):
        target = np.concatenate((source.T @ response[:, channel], np.asarray((1.0,))))
        matrix[channel] = np.linalg.solve(system, target)[:3]
    prediction = source @ matrix.T
    residual = prediction - response
    return {
        "constraint": "each matrix row sums to one",
        "color_matrix": matrix,
        "prediction_rgb": prediction,
        "residual_rgb": residual,
        "mae": float(np.mean(np.abs(residual))),
    }


def apply_anchor_tone_curve(
    latent_rgb: np.ndarray,
    *,
    midpoint_response_rgb: np.ndarray,
    frame_indices: np.ndarray,
) -> np.ndarray:
    latent = np.asarray(latent_rgb, dtype=np.float64)
    midpoint = np.asarray(midpoint_response_rgb, dtype=np.float64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if latent.ndim != 2 or latent.shape[1] != 3:
        raise ValueError("latent_rgb must have shape (n, 3)")
    if frames.shape != (len(latent),):
        raise ValueError("frame_indices must align with latent_rgb")
    if midpoint.ndim != 2 or midpoint.shape[1] != 3:
        raise ValueError("midpoint_response_rgb must have shape (frame_count, 3)")
    if np.any(frames < 0) or np.any(frames >= len(midpoint)):
        raise ValueError("frame_indices fall outside midpoint_response_rgb")
    if np.any((midpoint <= 0.0) | (midpoint >= 1.0)):
        raise ValueError("midpoint responses must lie strictly between zero and one")

    response = np.empty_like(latent)
    for frame in range(len(midpoint)):
        rows = frames == frame
        for channel in range(3):
            response[rows, channel] = np.interp(
                latent[rows, channel],
                (0.0, 0.5, 1.0),
                (0.0, midpoint[frame, channel], 1.0),
            )
    return response


def fit_anchor_tone_normalization(
    normalized_rgb: np.ndarray,
    patch_ids: list[str],
    frame_indices: np.ndarray,
) -> dict[str, Any]:
    normalized = np.asarray(normalized_rgb, dtype=np.float64)
    labels = np.asarray(patch_ids)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if normalized.ndim != 2 or normalized.shape[1] != 3:
        raise ValueError("normalized_rgb must have shape (n, 3)")
    if labels.shape != (len(normalized),) or frames.shape != (len(normalized),):
        raise ValueError("patch_ids and frame_indices must align with normalized_rgb")
    unique_frames = np.unique(frames)
    if not np.array_equal(unique_frames, np.arange(len(unique_frames))):
        raise ValueError("frame_indices must be contiguous and start at zero")

    white_scale = np.empty((len(unique_frames), 3), dtype=np.float64)
    midpoint = np.empty_like(white_scale)
    latent = np.empty_like(normalized)
    for frame in unique_frames:
        rows = frames == frame
        middle_rows = rows & (labels == "K128")
        white_rows = rows & (labels == "K255")
        if np.count_nonzero(middle_rows) < 2 or np.count_nonzero(white_rows) < 2:
            raise ValueError("each frame requires at least two K128 and two K255 anchors")
        white_scale[frame] = np.median(normalized[white_rows], axis=0)
        if np.any(white_scale[frame] <= 0.0):
            raise ValueError("white anchor response must be positive")
        scaled = normalized[rows] / white_scale[frame]
        midpoint[frame] = np.median(normalized[middle_rows], axis=0) / white_scale[frame]
        if np.any((midpoint[frame] <= 0.0) | (midpoint[frame] >= 1.0)):
            raise ValueError("K128 response must lie strictly between black and white")
        for channel in range(3):
            latent[rows, channel] = np.interp(
                scaled[:, channel],
                (0.0, midpoint[frame, channel], 1.0),
                (0.0, 0.5, 1.0),
            )

    reconstructed_scaled = apply_anchor_tone_curve(
        latent,
        midpoint_response_rgb=midpoint,
        frame_indices=frames,
    )
    return {
        "gauge_constraints": {
            "black": "h_tc(0)=0",
            "middle_gray": "h_tc(0.5)=measured K128/K255 ratio",
            "white": "h_tc(1)=1",
        },
        "frame_white_scale_rgb": white_scale,
        "frame_midpoint_response_rgb": midpoint,
        "latent_rgb": latent,
        "reconstruction_rgb": reconstructed_scaled * white_scale[frames],
    }


def apply_trilinear_color_lut(
    source_rgb: np.ndarray,
    *,
    levels: np.ndarray,
    lut_rgb: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_rgb, dtype=np.float64)
    grid_levels = np.asarray(levels, dtype=np.float64)
    lut = np.asarray(lut_rgb, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source_rgb must have shape (n, 3)")
    if grid_levels.ndim != 1 or len(grid_levels) < 2 or np.any(np.diff(grid_levels) <= 0.0):
        raise ValueError("levels must be a strictly increasing one-dimensional array")
    expected_shape = (len(grid_levels), len(grid_levels), len(grid_levels), 3)
    if lut.shape != expected_shape:
        raise ValueError(f"lut_rgb must have shape {expected_shape}")
    if np.any(source < grid_levels[0] - 1e-12) or np.any(source > grid_levels[-1] + 1e-12):
        raise ValueError("source colors fall outside the LUT level range")

    upper = np.searchsorted(grid_levels, source, side="right")
    upper = np.clip(upper, 1, len(grid_levels) - 1)
    lower = upper - 1
    lower_values = grid_levels[lower]
    upper_values = grid_levels[upper]
    weights = (source - lower_values) / (upper_values - lower_values)
    output = np.zeros((len(source), 3), dtype=np.float64)
    for red_side in (0, 1):
        for green_side in (0, 1):
            for blue_side in (0, 1):
                sides = np.asarray((red_side, green_side, blue_side))
                indices = np.where(sides, upper, lower)
                corner_weight = np.prod(np.where(sides, weights, 1.0 - weights), axis=1)
                output += corner_weight[:, None] * lut[
                    indices[:, 0], indices[:, 1], indices[:, 2]
                ]
    return output


def _field_chart_source_geometry(
    source_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = source_size
    patch_width, patch_height = 235, 255
    component_centers: list[tuple[float, float]] = []
    point_centers: list[tuple[float, float]] = []
    for row in range(3):
        for column in range(3):
            center_x = (2 * column + 1) * width // 6
            center_y = (2 * row + 1) * height // 6
            point_centers.append((center_x - 152.0, float(center_y)))
            x0 = center_x - 30
            x1 = x0 + patch_width
            y0 = center_y - patch_height // 2
            y1 = y0 + patch_height
            orientation = (row + column) % 3
            if orientation == 0:
                polygon = np.asarray(
                    [(x0 + 112, y0), (x1, y0), (x1, y1), (x0 + 124, y1)],
                    dtype=np.float32,
                )
            elif orientation == 1:
                polygon = np.asarray(
                    [(x0, y0 + 136), (x1, y0 + 119), (x1, y1), (x0, y1)],
                    dtype=np.float32,
                )
            else:
                polygon = np.asarray([(x0, y1), (x1, y0), (x1, y1)], dtype=np.float32)
            moments = cv2.moments(polygon)
            component_centers.append(
                (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            )
    return np.asarray(component_centers), np.asarray(point_centers)


def fit_field_point_edge_capture(
    image_rgb: np.ndarray,
    *,
    source_size: tuple[int, int],
    roi_xyxy: tuple[int, int, int, int],
    square_side_px: int,
) -> dict[str, Any]:
    image = np.asarray(image_rgb, dtype=np.float32)
    x0, y0, x1, y1 = roi_xyxy
    roi = image[y0:y1, x0:x1]
    score = 0.5 * roi[:, :, 1] + 0.5 * roi[:, :, 2] - 0.6 * roi[:, :, 0]
    count, _, stats, centers = cv2.connectedComponentsWithStats(
        (score > 0.18).astype(np.uint8), connectivity=8
    )
    candidates: list[tuple[int, np.ndarray]] = []
    for index in range(1, count):
        left, top, width, height, area = (int(value) for value in stats[index])
        if 300 <= area <= 30_000 and width >= 20 and height >= 20:
            center = centers[index] + np.asarray((x0, y0), dtype=np.float64)
            candidates.append((area, center))
    if len(candidates) < 9:
        raise ValueError(f"field chart yielded only {len(candidates)} large patches")
    selected = np.asarray(
        [center for _, center in sorted(candidates, key=lambda item: item[0], reverse=True)[:9]]
    )
    row_order = np.argsort(selected[:, 1])
    observed_components = np.vstack(
        [selected[row][np.argsort(selected[row, 0])] for row in np.array_split(row_order, 3)]
    )
    source_components, source_points = _field_chart_source_geometry(source_size)
    warp = fit_quadratic_warp(
        source_components,
        observed_components,
        source_size=source_size,
        minimum_inliers=7,
    )
    predicted_points = predict_quadratic_warp(
        source_points, warp["coefficients"], source_size=source_size
    )

    point_score = image.max(axis=2)
    top_hat = point_score - cv2.GaussianBlur(point_score, (0, 0), 12.0)
    point_results: list[dict[str, Any]] = []
    for index, (source_point, prediction) in enumerate(zip(source_points, predicted_points)):
        predicted_x, predicted_y = (int(round(value)) for value in prediction)
        radius = 45
        search = top_hat[
            predicted_y - radius : predicted_y + radius + 1,
            predicted_x - radius : predicted_x + radius + 1,
        ]
        if search.shape != (2 * radius + 1, 2 * radius + 1):
            continue
        peak_y, peak_x = np.unravel_index(int(np.argmax(search)), search.shape)
        peak = np.asarray(
            (predicted_x - radius + peak_x, predicted_y - radius + peak_y),
            dtype=np.float64,
        )
        try:
            center, observed_covariance, contrast = _weighted_center_covariance(
                point_score, peak, 14
            )
        except ValueError:
            continue
        jacobian = quadratic_warp_jacobian(
            source_point, warp["coefficients"], source_size=source_size
        )
        psf_covariance = deconvolve_square_psf_covariance(
            observed_covariance, jacobian, square_side_px=square_side_px
        )
        point_results.append(
            {
                "index": index,
                "source_xy": source_point,
                "observed_xy": center,
                "prediction_residual_px": float(np.linalg.norm(center - prediction)),
                "contrast": contrast,
                "jacobian": jacobian,
                "scale_singular_values": np.linalg.svd(jacobian, compute_uv=False)[::-1],
                "psf_covariance": psf_covariance,
                "psf_fwhm_minor_major_angle": _covariance_shape(psf_covariance),
            }
        )
    return {
        "large_patch_count": len(observed_components),
        "point_count": len(point_results),
        "source_components": source_components,
        "observed_components": observed_components,
        "warp": warp,
        "points": point_results,
    }
