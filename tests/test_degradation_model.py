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
