from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import math
from typing import Iterable

import cv2
import numpy as np
from PIL import Image


_IDENTITY_MATRIX = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)
_ZERO_QUADRATIC = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ONE_QUADRATIC = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_ZERO_GEOMETRIC_RESIDUAL = (_ZERO_QUADRATIC, _ZERO_QUADRATIC)


def _flatten(values: Iterable[object]) -> Iterable[float]:
    for value in values:
        if isinstance(value, (tuple, list)):
            yield from _flatten(value)
        else:
            yield float(value)


@dataclass(frozen=True)
class DegradationParameters:
    output_size: tuple[int, int]
    homography: tuple[tuple[float, float, float], ...] = _IDENTITY_MATRIX
    geometric_residual_coefficients: tuple[
        tuple[float, float, float, float, float, float], ...
    ] = _ZERO_GEOMETRIC_RESIDUAL
    attenuation_coefficients: tuple[tuple[float, float, float, float, float, float], ...] = (
        _ONE_QUADRATIC,
        _ONE_QUADRATIC,
        _ONE_QUADRATIC,
    )
    background_coefficients: tuple[tuple[float, float, float, float, float, float], ...] = (
        _ZERO_QUADRATIC,
        _ZERO_QUADRATIC,
        _ZERO_QUADRATIC,
    )
    blur_fwhm_px: tuple[float, float] = (0.0, 0.0)
    blur_angle_deg: float = 0.0
    color_matrix: tuple[tuple[float, float, float], ...] = _IDENTITY_MATRIX
    gain_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    bias_rgb: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gamma_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
    tone_curve_levels: tuple[float, ...] = (0.0, 1.0)
    tone_curve_rgb: tuple[tuple[float, ...], ...] = (
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
    )
    noise_slope: tuple[float, float, float] = (0.0, 0.0, 0.0)
    noise_intercept: tuple[float, float, float] = (0.0, 0.0, 0.0)
    jpeg_quality: int = 96
    jpeg_subsampling: int = 2

    def __post_init__(self) -> None:
        width, height = self.output_size
        if width <= 0 or height <= 0:
            raise ValueError("output_size values must be positive")
        numeric_fields = (
            self.homography,
            self.geometric_residual_coefficients,
            self.attenuation_coefficients,
            self.background_coefficients,
            self.blur_fwhm_px,
            (self.blur_angle_deg,),
            self.color_matrix,
            self.gain_rgb,
            self.bias_rgb,
            self.gamma_rgb,
            self.tone_curve_levels,
            self.tone_curve_rgb,
            self.noise_slope,
            self.noise_intercept,
        )
        if not all(math.isfinite(value) for field in numeric_fields for value in _flatten(field)):
            raise ValueError("all degradation parameters must be finite")
        if np.asarray(self.homography).shape != (3, 3):
            raise ValueError("homography must be 3 x 3")
        if abs(float(np.linalg.det(np.asarray(self.homography, dtype=np.float64)))) < 1e-12:
            raise ValueError("homography must be invertible")
        if np.asarray(self.color_matrix).shape != (3, 3):
            raise ValueError("color_matrix must be 3 x 3")
        if np.asarray(self.geometric_residual_coefficients).shape != (2, 6):
            raise ValueError("geometric_residual_coefficients must be 2 x 6")
        if np.asarray(self.attenuation_coefficients).shape != (3, 6):
            raise ValueError("attenuation_coefficients must be 3 x 6")
        if np.asarray(self.background_coefficients).shape != (3, 6):
            raise ValueError("background_coefficients must be 3 x 6")
        if any(value < 0.0 for value in self.blur_fwhm_px):
            raise ValueError("blur FWHM must be nonnegative")
        if any(value <= 0.0 for value in self.gamma_rgb):
            raise ValueError("gamma values must be positive")
        levels = np.asarray(self.tone_curve_levels, dtype=np.float64)
        tone = np.asarray(self.tone_curve_rgb, dtype=np.float64)
        if len(levels) < 2 or tone.shape != (3, len(levels)):
            raise ValueError("tone curve must have shared levels and three channel rows")
        if (
            abs(float(levels[0])) > 1e-12
            or abs(float(levels[-1]) - 1.0) > 1e-12
            or np.any(np.diff(levels) <= 0.0)
            or np.any((tone < 0.0) | (tone > 1.0))
            or np.any(np.diff(tone, axis=1) < 0.0)
        ):
            raise ValueError("tone curve levels and channel values must be monotonic in [0, 1]")
        if any(value < 0.0 for value in self.noise_slope + self.noise_intercept):
            raise ValueError("noise variance coefficients must be nonnegative")
        if not 0 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 0 and 100")
        if self.jpeg_subsampling not in {0, 1, 2}:
            raise ValueError("jpeg_subsampling must be 0, 1, or 2")

    @classmethod
    def identity(cls, output_size: tuple[int, int]) -> DegradationParameters:
        return cls(output_size=output_size)


def _pixel_homography(
    normalized: tuple[tuple[float, float, float], ...],
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    source_width, source_height = source_size
    output_width, output_height = output_size
    source_to_normalized = np.diag(
        [1.0 / max(source_width - 1, 1), 1.0 / max(source_height - 1, 1), 1.0]
    )
    normalized_to_output = np.diag(
        [max(output_width - 1, 1), max(output_height - 1, 1), 1.0]
    )
    return (
        normalized_to_output
        @ np.asarray(normalized, dtype=np.float64)
        @ source_to_normalized
    )


def _quadratic_field(
    coefficients: tuple[tuple[float, float, float, float, float, float], ...],
    width: int,
    height: int,
) -> np.ndarray:
    x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)
    basis = np.stack(
        (
            np.ones_like(grid_x),
            grid_x,
            grid_y,
            grid_x * grid_x,
            grid_x * grid_y,
            grid_y * grid_y,
        ),
        axis=-1,
    )
    return np.einsum("hwk,ck->hwc", basis, np.asarray(coefficients, np.float32))


def _apply_geometric_residual(
    image: np.ndarray,
    coefficients: tuple[tuple[float, float, float, float, float, float], ...],
) -> np.ndarray:
    if not any(abs(value) > 1e-12 for row in coefficients for value in row):
        return image
    height, width = image.shape[:2]
    displacement = _quadratic_field(coefficients, width, height)
    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    map_x = grid_x - displacement[:, :, 0] * max(width - 1, 1)
    map_y = grid_y - displacement[:, :, 1] * max(height - 1, 1)
    return cv2.remap(
        image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )


def _anisotropic_kernel(fwhm_px: tuple[float, float], angle_deg: float) -> np.ndarray | None:
    sigma = np.asarray(fwhm_px, dtype=np.float64) / 2.354820045
    if float(sigma.max()) < 1e-6:
        return None
    sigma = np.maximum(sigma, 1e-3)
    angle = math.radians(angle_deg)
    rotation = np.asarray(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    covariance = rotation @ np.diag(sigma * sigma) @ rotation.T
    inverse = np.linalg.inv(covariance)
    radius = max(1, int(math.ceil(4.0 * float(sigma.max()))))
    grid_y, grid_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    points = np.stack((grid_x, grid_y), axis=-1).astype(np.float64)
    exponent = np.einsum("...i,ij,...j->...", points, inverse, points)
    kernel = np.exp(-0.5 * exponent)
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def _jpeg_roundtrip(image: np.ndarray, quality: int, subsampling: int) -> np.ndarray:
    encoded = BytesIO()
    Image.fromarray(np.rint(image * 255.0).astype(np.uint8), "RGB").save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
        optimize=False,
        progressive=False,
    )
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        return np.asarray(decoded.convert("RGB"), dtype=np.float32) / 255.0


def _apply_tone_curve(image: np.ndarray, params: DegradationParameters) -> np.ndarray:
    levels = np.asarray(params.tone_curve_levels, dtype=np.float32)
    tone = np.asarray(params.tone_curve_rgb, dtype=np.float32)
    output = np.empty_like(image)
    for channel in range(3):
        output[:, :, channel] = np.interp(
            image[:, :, channel], levels, tone[channel]
        )
    return output


def degrade(
    image: np.ndarray,
    params: DegradationParameters,
    *,
    seed: int,
    encode_jpeg: bool = True,
) -> np.ndarray:
    """Apply the effective path-level degradation model to float RGB in [0, 1]."""
    source = np.asarray(image, dtype=np.float32)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("image must have shape (height, width, 3)")
    if not np.isfinite(source).all() or float(source.min()) < 0.0 or float(source.max()) > 1.0:
        raise ValueError("image values must be finite and within [0, 1]")

    output_width, output_height = params.output_size
    transform = _pixel_homography(
        params.homography,
        (source.shape[1], source.shape[0]),
        params.output_size,
    )
    degraded = cv2.warpPerspective(
        source,
        transform,
        params.output_size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    degraded = _apply_geometric_residual(
        degraded, params.geometric_residual_coefficients
    )
    kernel = _anisotropic_kernel(params.blur_fwhm_px, params.blur_angle_deg)
    if kernel is not None:
        degraded = cv2.filter2D(degraded, -1, kernel, borderType=cv2.BORDER_REFLECT_101)

    degraded = np.einsum(
        "hwc,oc->hwo", degraded, np.asarray(params.color_matrix, dtype=np.float32)
    )
    degraded = np.power(
        np.clip(degraded, 0.0, 1.0), np.asarray(params.gamma_rgb, np.float32)
    )
    degraded = _apply_tone_curve(degraded, params)
    attenuation = _quadratic_field(
        params.attenuation_coefficients, output_width, output_height
    )
    degraded *= np.maximum(attenuation, 0.0)
    degraded *= np.asarray(params.gain_rgb, np.float32)
    degraded += _quadratic_field(
        params.background_coefficients, output_width, output_height
    )
    degraded += np.asarray(params.bias_rgb, np.float32)

    slope = np.asarray(params.noise_slope, dtype=np.float32)
    intercept = np.asarray(params.noise_intercept, dtype=np.float32)
    variance = np.maximum(degraded * slope + intercept, 0.0)
    if float(variance.max()) > 0.0:
        generator = np.random.default_rng(seed)
        degraded += generator.normal(0.0, np.sqrt(variance), degraded.shape).astype(
            np.float32
        )
    degraded = np.clip(degraded, 0.0, 1.0).astype(np.float32)
    if encode_jpeg:
        degraded = _jpeg_roundtrip(
            degraded, params.jpeg_quality, params.jpeg_subsampling
        )
    return degraded.astype(np.float32, copy=False)
