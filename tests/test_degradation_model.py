from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from analysis.degradation.model import DegradationParameters, degrade


def test_identity_forward_model_preserves_rgb() -> None:
    image = np.zeros((16, 20, 3), dtype=np.float32)
    image[4:12, 6:14] = (0.2, 0.6, 0.9)
    params = DegradationParameters.identity(output_size=(20, 16))

    actual = degrade(image, params, seed=7, encode_jpeg=False)

    np.testing.assert_allclose(actual, image, atol=1e-6)


def test_normalized_homography_supports_anisotropic_output_sampling() -> None:
    image = np.zeros((12, 18, 3), dtype=np.float32)
    image[3:9, 4:14] = 1.0
    params = DegradationParameters.identity(output_size=(9, 8))

    actual = degrade(image, params, seed=11, encode_jpeg=False)

    assert actual.shape == (8, 9, 3)
    assert 0.1 < float(actual.mean()) < 0.9


def test_quadratic_geometric_residual_moves_warped_content() -> None:
    image = np.zeros((11, 11, 3), dtype=np.float32)
    image[5, 4] = 1.0
    params = replace(
        DegradationParameters.identity(output_size=(11, 11)),
        geometric_residual_coefficients=(
            (0.2, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )

    actual = degrade(image, params, seed=12, encode_jpeg=False)

    peak_y, peak_x = np.unravel_index(np.argmax(actual[:, :, 0]), actual.shape[:2])
    assert (peak_x, peak_y) == (6, 5)


def test_color_matrix_maps_input_channels_to_output_channels() -> None:
    image = np.zeros((8, 8, 3), dtype=np.float32)
    image[:, :, 0] = 1.0
    params = replace(
        DegradationParameters.identity(output_size=(8, 8)),
        color_matrix=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    )

    actual = degrade(image, params, seed=13, encode_jpeg=False)

    np.testing.assert_allclose(actual[:, :, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(actual[:, :, 1], 1.0, atol=1e-6)
    np.testing.assert_allclose(actual[:, :, 2], 0.0, atol=1e-6)


def test_tone_curve_precedes_spatial_gain_and_additive_background() -> None:
    image = np.full((6, 6, 3), 0.5, dtype=np.float32)
    params = replace(
        DegradationParameters.identity(output_size=(6, 6)),
        tone_curve_levels=(0.0, 0.5, 1.0),
        tone_curve_rgb=(
            (0.0, 0.25, 1.0),
            (0.0, 0.25, 1.0),
            (0.0, 0.25, 1.0),
        ),
        gain_rgb=(2.0, 2.0, 2.0),
        attenuation_coefficients=(
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.5, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
        background_coefficients=(
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.1, 0.0, 0.0, 0.0, 0.0, 0.0),
        ),
    )

    actual = degrade(image, params, seed=14, encode_jpeg=False)

    np.testing.assert_allclose(actual, 0.35, atol=1e-6)


def test_rejects_non_monotonic_tone_curve() -> None:
    with pytest.raises(ValueError, match="tone curve"):
        replace(
            DegradationParameters.identity(output_size=(8, 8)),
            tone_curve_levels=(0.0, 0.5, 1.0),
            tone_curve_rgb=(
                (0.0, 0.6, 0.5),
                (0.0, 0.5, 1.0),
                (0.0, 0.5, 1.0),
            ),
        )


def test_seeded_degradation_is_reproducible() -> None:
    image = np.full((32, 32, 3), 0.5, dtype=np.float32)
    params = replace(
        DegradationParameters.identity(output_size=(32, 32)),
        noise_slope=(0.01, 0.01, 0.01),
        jpeg_quality=96,
        jpeg_subsampling=2,
    )

    first = degrade(image, params, seed=31, encode_jpeg=True)
    second = degrade(image, params, seed=31, encode_jpeg=True)
    different = degrade(image, params, seed=32, encode_jpeg=True)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, different)


def test_rejects_negative_noise_variance() -> None:
    with pytest.raises(ValueError, match="noise variance"):
        replace(
            DegradationParameters.identity(output_size=(8, 8)),
            noise_intercept=(-0.1, 0.0, 0.0),
        )
