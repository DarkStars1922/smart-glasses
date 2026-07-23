from __future__ import annotations

import numpy as np

from analysis.degradation.calibration_c import fit_permuted_photometry


def test_permuted_photometry_separates_background_tone_spatial_gain_and_color() -> None:
    levels = np.asarray((0, 36, 73, 109, 146, 182, 219, 255), dtype=np.float64) / 255.0
    colors = np.asarray(
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 0.0),
        )
    )
    patch_types = np.vstack((np.repeat(levels[:, None], 3, axis=1), colors))
    repeated = np.repeat(patch_types, 4, axis=0)
    generator = np.random.default_rng(20260722)
    source_rgb = np.vstack(
        (repeated[generator.permutation(56)], repeated[generator.permutation(56)])
    )
    frame_indices = np.repeat((0, 1), 56)
    grid_xy = np.asarray(
        [(530.0 + 205.0 * column, 360.0 + 145.0 * row) for row in range(7) for column in range(8)]
    )
    source_xy = np.vstack((grid_xy, grid_xy))

    matrix = np.asarray(
        ((0.82, 0.13, 0.05), (0.08, 0.84, 0.08), (0.04, 0.16, 0.80))
    )
    gamma = np.asarray((0.85, 1.05, 1.20))
    gains = np.asarray(((0.72, 0.78, 0.68), (0.88, 0.83, 0.76)))
    x = 2.0 * source_xy[:, 0] / 2500.0 - 1.0
    y = 2.0 * source_xy[:, 1] / 1600.0 - 1.0
    spatial = np.empty((len(source_rgb), 3))
    background = np.empty_like(spatial)
    for index, frame in enumerate(frame_indices):
        spatial[index] = np.exp(
            np.asarray((0.08, -0.05, 0.04)) * x[index]
            + np.asarray((-0.04, 0.06, -0.03)) * y[index]
            + (0.03 if frame else -0.02) * (x[index] * y[index])
        )
        background[index] = (
            np.asarray((0.035, 0.040, 0.030))
            + np.asarray((0.008, -0.004, 0.006)) * x[index]
            + np.asarray((-0.005, 0.007, 0.003)) * y[index]
            + frame * np.asarray((0.006, 0.004, 0.005))
        )
    latent = source_rgb @ matrix.T
    observed = background + gains[frame_indices] * spatial * latent**gamma

    result = fit_permuted_photometry(
        observed,
        source_rgb,
        source_xy,
        frame_indices,
        source_size=(2500, 1600),
    )

    assert result["gray_design_rank"] == result["gray_design_columns"]
    assert result["tone_monotonic"] == [True, True, True]
    np.testing.assert_allclose(result["color_matrix"], matrix, atol=0.06)
    assert result["fit_mae"] < 0.015
