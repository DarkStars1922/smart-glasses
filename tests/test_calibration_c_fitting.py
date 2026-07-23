from __future__ import annotations

import numpy as np

from analysis.degradation.calibration_c import (
    apply_anchor_tone_curve,
    apply_trilinear_color_lut,
    deconvolve_square_psf_covariance,
    fit_black_background_field,
    fit_constrained_linear_color_matrix,
    fit_edge_spread,
    fit_field_point_edge_capture,
    fit_joint_color_nodes,
    fit_anchor_tone_normalization,
    fit_point_grid_capture,
    fit_quadratic_warp,
    measure_slanted_edge_cells,
    predict_quadratic_warp,
    refine_effective_color_matrix,
)


def test_quadratic_warp_recovers_curved_mapping_with_one_outlier() -> None:
    x, y = np.meshgrid(
        np.linspace(250.0, 2250.0, 5),
        np.linspace(160.0, 1440.0, 5),
    )
    source = np.column_stack((x.ravel(), y.ravel()))
    normalized_x = source[:, 0] / 1250.0 - 1.0
    normalized_y = source[:, 1] / 800.0 - 1.0
    basis = np.column_stack(
        (
            np.ones(len(source)),
            normalized_x,
            normalized_y,
            normalized_x**2,
            normalized_x * normalized_y,
            normalized_y**2,
        )
    )
    coefficients = np.asarray(
        [
            [1800.0, 310.0, -18.0, 12.0, 8.0, -7.0],
            [2050.0, 25.0, 455.0, -9.0, 11.0, 16.0],
        ]
    )
    observed = basis @ coefficients.T
    observed[7] += (85.0, -70.0)

    result = fit_quadratic_warp(source, observed, source_size=(2500, 1600))
    prediction = predict_quadratic_warp(
        source, result["coefficients"], source_size=(2500, 1600)
    )

    assert result["inlier_mask"].sum() == 24
    assert np.max(np.linalg.norm(prediction[result["inlier_mask"]] - observed[result["inlier_mask"]], axis=1)) < 1e-6
    assert result["quadratic_median_residual_px"] < 1e-6
    assert result["homography_median_residual_px"] > 1.0


def test_deconvolution_removes_known_finite_square_support() -> None:
    jacobian = np.asarray([[0.42, 0.08], [-0.05, 0.51]])
    true_psf_covariance = np.asarray([[13.0, 2.5], [2.5, 21.0]])
    square_variance = (15.0**2 - 1.0) / 12.0
    observed_covariance = (
        true_psf_covariance
        + square_variance * jacobian @ jacobian.T
    )

    recovered = deconvolve_square_psf_covariance(
        observed_covariance, jacobian, square_side_px=15
    )

    np.testing.assert_allclose(recovered, true_psf_covariance, atol=1e-10)


def test_point_grid_capture_recovers_all_local_centers() -> None:
    source_size = (250, 160)
    fractions = np.asarray((0.1, 0.3, 0.5, 0.7, 0.9))
    source = np.asarray(
        [
            (x * (source_size[0] - 1), y * (source_size[1] - 1))
            for y in fractions
            for x in fractions
        ]
    )
    observed = np.column_stack(
        (
            85.0 + 1.25 * source[:, 0] + 0.0008 * (source[:, 1] - 80.0) ** 2,
            55.0 + 0.08 * source[:, 0] + 1.45 * source[:, 1],
        )
    )
    image = np.zeros((380, 480, 3), dtype=np.float32)
    yy, xx = np.indices(image.shape[:2])
    for center_x, center_y in observed:
        image[:, :, 0] += np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 2.0**2)
        )
    image = np.clip(image, 0.0, 1.0)
    anchor_indices = (0, 4, 20, 24)

    result = fit_point_grid_capture(
        image,
        source_size=source_size,
        anchor_corners=observed[list(anchor_indices)],
        channel_index=0,
        square_side_px=1,
        search_radius=15,
    )

    assert result["detected_count"] == 25
    assert result["warp"]["inlier_mask"].sum() == 25
    assert result["warp"]["quadratic_p95_residual_px"] < 0.15


def test_point_grid_capture_accepts_larger_measurement_patch() -> None:
    source_size = (250, 160)
    fractions = np.asarray((0.1, 0.3, 0.5, 0.7, 0.9))
    source = np.asarray(
        [
            (x * (source_size[0] - 1), y * (source_size[1] - 1))
            for y in fractions
            for x in fractions
        ]
    )
    observed = source * (1.7, 1.7) + (70.0, 60.0)
    image = np.zeros((400, 600, 3), dtype=np.float32)
    yy, xx = np.indices(image.shape[:2])
    for center_x, center_y in observed:
        image[:, :, 1] += np.exp(
            -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * 7.0**2)
        )
    image = np.clip(image, 0.0, 1.0)

    result = fit_point_grid_capture(
        image,
        source_size=source_size,
        anchor_corners=observed[[0, 4, 20, 24]],
        channel_index=1,
        square_side_px=1,
        search_radius=25,
        measurement_radius=22,
    )

    assert result["measurement_radius"] == 22
    assert result["warp"]["inlier_mask"].sum() == 25


def test_black_background_field_recovers_per_frame_quadratics() -> None:
    x, y = np.meshgrid(np.linspace(150.0, 2350.0, 7), np.linspace(120.0, 1480.0, 6))
    positions = np.column_stack((x.ravel(), y.ravel()))
    frame_indices = np.repeat(np.arange(2), len(positions))
    positions = np.tile(positions, (2, 1))
    normalized_x = 2.0 * positions[:, 0] / 2500.0 - 1.0
    normalized_y = 2.0 * positions[:, 1] / 1600.0 - 1.0
    basis = np.column_stack(
        (
            np.ones(len(positions)),
            normalized_x,
            normalized_y,
            normalized_x**2,
            normalized_x * normalized_y,
            normalized_y**2,
        )
    )
    coefficients = np.asarray(
        (
            ((0.08, 0.01, -0.02, 0.004, 0.002, 0.006),) * 3,
            ((0.11, -0.015, 0.01, 0.007, -0.003, 0.005),) * 3,
        )
    )
    observed = np.empty((len(positions), 3))
    for frame in range(2):
        rows = frame_indices == frame
        observed[rows] = basis[rows] @ coefficients[frame].T

    result = fit_black_background_field(
        observed,
        positions,
        frame_indices,
        source_size=(2500, 1600),
        ridge=0.0,
    )

    np.testing.assert_allclose(result["coefficients"], coefficients, atol=1e-10)
    assert result["mae"] < 1e-12


def test_edge_spread_recovers_gaussian_fwhm_with_outliers() -> None:
    from scipy.special import ndtr

    distance = np.linspace(-18.0, 18.0, 1200)
    sigma = 2.6
    intensity = 0.08 + 0.62 * ndtr((distance - 0.35) / sigma)
    intensity[::97] += 0.15

    result = fit_edge_spread(distance, intensity)

    assert abs(result["sigma_px"] - sigma) < 0.08
    assert abs(result["fwhm_px"] - 2.354820045 * sigma) < 0.2
    assert result["r_squared"] > 0.98


def test_slanted_edge_cells_measure_camera_space_blur() -> None:
    from scipy.special import ndtr

    height, width = 240, 320
    yy, xx = np.indices((height, width), dtype=np.float64)
    sigma = 3.2
    edge_x = 155.0 + 0.12 * (yy - 120.0)
    image = np.zeros((height, width, 3), dtype=np.float32)
    image[:, :, 1] = 0.04 + 0.78 * ndtr((edge_x - xx) / sigma)
    cells = [
        {
            "color_id": "G",
            "orientation": "vertical_pos",
            "slope": 0.12,
            "bbox_xyxy": (60, 35, 250, 205),
        }
    ]

    result = measure_slanted_edge_cells(
        image,
        np.eye(3),
        cells,
        max_distance_px=20.0,
    )

    assert result["accepted_count"] == 1
    assert abs(result["cells"][0]["sigma_px"] - sigma) < 0.15
    assert result["cells"][0]["r_squared"] > 0.99


def test_color_matrix_refinement_recovers_forward_mixing() -> None:
    levels = np.asarray((0.0, 0.25, 0.5, 1.0))
    tone = np.asarray(
        (
            (0.0, 0.10, 0.42, 1.0),
            (0.0, 0.14, 0.51, 1.0),
            (0.0, 0.08, 0.38, 1.0),
        )
    )
    base_colors = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
        )
    )
    source = np.vstack([base_colors * scale for scale in (0.3, 0.6, 1.0)])
    matrix = np.asarray(
        ((0.72, 0.18, 0.10), (0.12, 0.78, 0.10), (0.20, 0.16, 0.64))
    )
    latent = np.clip(source @ matrix.T, 0.0, 1.0)
    normalized = np.column_stack(
        [np.interp(latent[:, channel], levels, tone[channel]) for channel in range(3)]
    )
    background = np.full_like(normalized, 0.04)
    gain = np.tile((0.75, 0.82, 0.70), (len(source), 1))
    spatial = np.ones_like(source)
    observed = background + gain * normalized

    result = refine_effective_color_matrix(
        observed,
        source,
        background,
        gain,
        spatial,
        tone_levels=levels,
        tone_values_rgb=tone,
        initial_matrix=np.eye(3),
    )

    np.testing.assert_allclose(result["color_matrix"], matrix, atol=0.03)
    assert result["color_fit_mae"] < 0.003


def test_joint_color_nodes_separate_frame_spatial_fields() -> None:
    source_size = (100, 80)
    node_colors = np.asarray(
        (
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 0.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
            (1.0, 1.0, 1.0),
        )
    )
    matrix = np.asarray(
        ((0.65, 0.25, 0.10), (0.15, 0.70, 0.15), (0.20, 0.15, 0.65))
    )
    node_response = node_colors @ matrix.T
    generator = np.random.default_rng(20260722)
    observed_rows = []
    source_rows = []
    position_rows = []
    patch_ids = []
    frame_rows = []
    frame_gain = np.asarray(((0.72, 0.81, 0.68), (0.83, 0.74, 0.77)))
    spatial_coefficients = np.asarray(
        (
            ((0.06, -0.03, 0.02, 0.01, -0.02),) * 3,
            ((-0.04, 0.05, -0.01, 0.02, 0.01),) * 3,
        )
    )
    background_coefficients = np.asarray(
        (
            ((0.04, 0.006, -0.004),) * 3,
            ((0.07, -0.003, 0.005),) * 3,
        )
    )
    for frame in range(2):
        colors = np.vstack((np.zeros((4, 3)), np.repeat(node_colors, 2, axis=0)))
        ids = ["K000"] * 4 + [f"N{index}" for index in range(7) for _ in range(2)]
        positions = generator.uniform((8.0, 8.0), (92.0, 72.0), size=(len(colors), 2))
        x = 2.0 * positions[:, 0] / source_size[0] - 1.0
        y = 2.0 * positions[:, 1] / source_size[1] - 1.0
        affine = np.column_stack((np.ones(len(colors)), x, y))
        field = np.column_stack((x, y, x * x, x * y, y * y))
        field -= field.mean(axis=0)
        response_lookup = {f"N{index}": node_response[index] for index in range(7)}
        response = np.asarray(
            [np.zeros(3) if patch_id == "K000" else response_lookup[patch_id] for patch_id in ids]
        )
        background = affine @ background_coefficients[frame].T
        spatial = np.exp(field @ spatial_coefficients[frame].T)
        observed = background + frame_gain[frame] * spatial * response
        observed_rows.extend(observed)
        source_rows.extend(colors)
        position_rows.extend(positions)
        patch_ids.extend(ids)
        frame_rows.extend([frame] * len(colors))

    result = fit_joint_color_nodes(
        np.asarray(observed_rows),
        np.asarray(source_rows),
        np.asarray(position_rows),
        patch_ids,
        np.asarray(frame_rows),
        source_size=source_size,
        spatial_ridge=0.0,
    )

    fitted = {row["patch_id"]: row["response_rgb"] for row in result["nodes"]}
    for index in range(7):
        np.testing.assert_allclose(fitted[f"N{index}"], node_response[index], atol=1e-7)
    assert result["design_rank"] == result["design_columns"]
    assert result["fit_mae"] < 1e-10


def test_constrained_linear_color_matrix_recovers_row_sum_mapping() -> None:
    source = np.asarray(list(np.ndindex(2, 2, 2)), dtype=np.float64)
    matrix = np.asarray(
        ((0.65, 0.25, 0.10), (0.15, 0.70, 0.15), (0.20, 0.15, 0.65))
    )
    observed = source @ matrix.T

    result = fit_constrained_linear_color_matrix(source, observed)

    np.testing.assert_allclose(result["color_matrix"], matrix, atol=1e-10)
    assert result["mae"] < 1e-12


def test_trilinear_color_lut_interpolates_multilinear_response() -> None:
    levels = np.asarray((0.0, 0.5, 1.0))
    grid = np.asarray(list(np.ndindex(3, 3, 3)))
    source_nodes = levels[grid]
    lut = np.empty((3, 3, 3, 3), dtype=np.float64)
    for indices, (red, green, blue) in zip(grid, source_nodes):
        lut[tuple(indices)] = (red * green, green * blue, 0.2 * red + 0.8 * blue)
    source = np.asarray(((0.25, 0.75, 0.60), (0.90, 0.10, 0.35)))

    prediction = apply_trilinear_color_lut(source, levels=levels, lut_rgb=lut)

    expected = np.column_stack(
        (source[:, 0] * source[:, 1], source[:, 1] * source[:, 2], 0.2 * source[:, 0] + 0.8 * source[:, 2])
    )
    np.testing.assert_allclose(prediction, expected, atol=1e-12)


def test_anchor_tone_normalization_removes_frame_dependent_midtone() -> None:
    latent = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5),
            (1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.25, 0.70, 0.40),
            (0.75, 0.20, 0.85),
        )
    )
    labels = ["K000", "K000", "K128", "K128", "K255", "K255", "C1", "C2"]
    frames = np.repeat(np.arange(2), len(latent))
    latent = np.tile(latent, (2, 1))
    midpoint = np.asarray(((0.30, 0.42, 0.55), (0.62, 0.35, 0.48)))
    white_scale = np.asarray(((0.8, 0.9, 1.1), (1.2, 0.7, 0.95)))
    observed = apply_anchor_tone_curve(
        latent,
        midpoint_response_rgb=midpoint,
        frame_indices=frames,
    ) * white_scale[frames]

    result = fit_anchor_tone_normalization(observed, labels * 2, frames)

    np.testing.assert_allclose(result["frame_white_scale_rgb"], white_scale, atol=1e-12)
    np.testing.assert_allclose(result["frame_midpoint_response_rgb"], midpoint, atol=1e-12)
    np.testing.assert_allclose(result["latent_rgb"], latent, atol=1e-12)


def test_field_chart_uses_large_patches_to_recover_nine_small_points() -> None:
    source_size = (2500, 1600)
    source_centers = np.asarray(
        [
            ((2 * column + 1) * source_size[0] / 6, (2 * row + 1) * source_size[1] / 6)
            for row in range(3)
            for column in range(3)
        ]
    )
    image = np.zeros((520, 820, 3), dtype=np.float32)
    scale = np.asarray((0.24, 0.22))
    offset = np.asarray((80.0, 70.0))
    observed_centers = source_centers * scale + offset
    for source_center, observed_center in zip(source_centers, observed_centers):
        center_x, center_y = np.rint(observed_center).astype(int)
        image[center_y - 20 : center_y + 21, center_x - 24 : center_x + 25, 1:] = 0.9
        point = (source_center - (152.0, 0.0)) * scale + offset
        point_x, point_y = point
        yy, xx = np.indices(image.shape[:2])
        blob = np.exp(-((xx - point_x) ** 2 + (yy - point_y) ** 2) / 8.0)
        image[:, :, 1] = np.maximum(image[:, :, 1], blob)
        image[:, :, 2] = np.maximum(image[:, :, 2], blob)

    result = fit_field_point_edge_capture(
        image,
        source_size=source_size,
        roi_xyxy=(0, 0, image.shape[1], image.shape[0]),
        square_side_px=1,
    )

    assert result["large_patch_count"] == 9
    assert result["point_count"] == 9
    assert result["warp"]["quadratic_p95_residual_px"] < 1.0
