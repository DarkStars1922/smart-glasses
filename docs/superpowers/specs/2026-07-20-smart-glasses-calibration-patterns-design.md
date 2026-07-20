# Smart-glasses calibration pattern design

## Goal

Generate a deterministic calibration suite that separates the major terms in the
smart-glasses leakage degradation model: additive background, spatial attenuation,
camera response, color cross-talk, projective geometry, optical blur, sampling, and
text-domain validation.

## Output contract

- Output directory: `images/origin/calibration_B/`
- Canvas: 2515 x 1491 pixels, matching the current source screenshot
- Format: lossless 8-bit RGB PNG in sRGB value space
- Image count: exactly 32
- Supporting files: `manifest.csv` and `README.md`
- Generator: `images/generate_calibration_b.py`
- Existing files under `images/origin/` remain unchanged

Uniform fields and hard geometric patterns use exact integer RGB values. Text is
rasterized at native canvas resolution, so antialiasing is allowed only within the
two text-validation images. JPEG compression and embedded resizing are forbidden.

## Image manifest

### Photometry and color: 19 images

| Index | Pattern |
|---|---|
| 00 | black `(0,0,0)` |
| 01 | white `(255,255,255)` |
| 02-08 | neutral gray at 32, 64, 96, 128, 160, 192, and 224 |
| 09-14 | red, green, and blue at channel levels 128 and 255 |
| 15-17 | cyan, magenta, and yellow at level 255 |
| 18 | full-level RGB/CMY/white/black color bars |

These images estimate additive background, read noise, spatial transmission,
response nonlinearity, clipping, channel gain, and a 3 x 3 color mixing matrix.

### Geometry: 3 images

| Index | Pattern |
|---|---|
| 19 | black/white checkerboard with hard edges |
| 20 | exact inverse checkerboard |
| 21 | asymmetric point grid with orientation markers |

These images estimate screen bounds, homography, optical mirroring, and lens
distortion. The asymmetric markers make all rotations and reflections identifiable.

### Blur and resolution: 7 images

| Index | Pattern |
|---|---|
| 22 | sparse 3 px white point grid on black |
| 23 | sparse 7 px white point grid on black |
| 24 | 3 x 3 spatial slanted-edge patches with alternating orientation |
| 25 | horizontal resolution bands at several spatial frequencies |
| 26 | vertical resolution bands at several spatial frequencies |
| 27 | diagonal resolution bands at several spatial frequencies |
| 28 | centered Siemens star |

These images estimate spatially varying PSF, directional blur, MTF, and the
effective resolution limit after the optical path and camera sampling.

### Spatial color and validation: 3 images

| Index | Pattern |
|---|---|
| 29 | spatial grid of black, gray, white, RGB, and CMY patches |
| 30 | white multilingual text and digits on a dark background |
| 31 | dark multilingual text and digits on a white background |

The spatial chart checks whether one global color matrix is sufficient. The two
text charts validate the calibrated degradation model; they are not used to fit
the blur kernel.

## Generator behavior

The generator creates the suite in a temporary directory, validates all images,
then replaces only the generated files in `calibration_B`. It is deterministic and
safe to rerun. `manifest.csv` records filename, category, exact parameters, and the
model terms each image estimates. `README.md` records the display and capture steps.

## Capture protocol

1. Display every image full-screen at 100% scale without interpolation.
2. Keep camera and glasses fixed for one complete 32-image run.
3. Lock exposure, ISO, focus, white balance, and focal length. Disable HDR, beauty
   processing, scene optimization, and automatic night mode.
4. Use the white image to set exposure so no RGB channel clips; do not change that
   exposure during the run.
5. Capture at least 10 frames per pattern. Prefer RAW/DNG plus JPEG when available.
6. Preserve pattern filenames in the captured-image directory or record the exact
   shooting order from `manifest.csv`.
7. Repeat the complete suite for each camera angle or distance being modeled.

## Verification

Automated tests must verify:

- exactly 32 PNG files are generated;
- every PNG is 2515 x 1491 and RGB;
- all uniform fields contain one exact expected RGB value;
- checkerboard normal and inverse images are pixelwise complements;
- structured patterns contain both foreground and background and are nonempty;
- filenames and `manifest.csv` indices are unique and contiguous;
- a second generation produces byte-identical PNG files.
