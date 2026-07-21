# Effective Smart-Glasses Degradation Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit a reproducible, path-specific effective degradation model to the current smart-glasses JPEG bursts, expose a deterministic synthesis API, update the research document, and specify the next captures needed for broader generalization.

**Architecture:** A versioned JSON manifest contains all dataset-specific paths, domain labels, roles, and ROIs. A small `analysis/degradation` package implements manifest validation, the generic forward operator, modular parameter fitting, and report/diagnostic generation; the core package contains no calibration-group or focal-length literals. The current 47 mm and 69 mm results are generated artifacts from one CLI run and explicitly mark unsupported parameters as provisional or not identifiable.

**Tech Stack:** Python 3.12, NumPy, Pillow, OpenCV, SciPy, scikit-image, Matplotlib, pytest

---

## File Map

- `analysis/degradation/schema.py`: typed manifest and parameter-result contracts.
- `analysis/degradation/model.py`: generic deterministic forward degradation operator.
- `analysis/degradation/fitting.py`: burst indexing, JPEG inspection, geometry/PSF/photometry/noise fits, and bootstrap summaries.
- `analysis/degradation/reporting.py`: JSON, Markdown, overlays, and comparison figures derived from fit results.
- `analysis/degradation/__init__.py`: stable public imports only.
- `analysis/calibration_b_v1.json`: current dataset labels and path-specific coordinates.
- `analysis/run_fit_v1.py`: command-line orchestration.
- `tests/test_degradation_schema.py`: manifest validation tests.
- `tests/test_degradation_model.py`: forward-operator tests.
- `tests/test_degradation_fitting.py`: synthetic estimator tests and current-dataset integration checks.
- `analysis/results/v1_parameters.json`: generated fitted parameter/result schema.
- `analysis/results/v1_report.md`: generated scientific report and capture recommendations.
- `analysis/results/v1_diagnostics/`: generated ROI overlays and source/synthetic/real comparisons.
- `docs/Shouldersurfing Smart Glass .md`: replace the preliminary first degradation equation with the validated model and measured values.

### Task 1: Add the generic dataset contract

**Files:**
- Create: `analysis/degradation/__init__.py`
- Create: `analysis/degradation/schema.py`
- Create: `analysis/calibration_b_v1.json`
- Create: `tests/test_degradation_schema.py`

- [ ] **Step 1: Write failing schema tests**

Test a minimal valid temporary manifest, reject an out-of-bounds ROI, reject unknown roles, and load the real manifest without relying on group names in Python code:

```python
from pathlib import Path
import json

import pytest

from analysis.degradation.schema import ManifestError, load_manifest


def test_real_manifest_is_domain_and_path_driven() -> None:
    manifest = load_manifest(Path("analysis/calibration_b_v1.json"))
    assert set(manifest.domains) == {"supermacro_47_primary", "supermacro_69_primary"}
    assert {group.path for group in manifest.groups} == {"primary_readable"}
    assert all(group.frames for group in manifest.groups)
    assert all(group.source.is_file() for group in manifest.groups)


def test_rejects_roi_outside_capture(tmp_path: Path) -> None:
    payload = {
        "schema_version": "1.0",
        "groups": [{
            "id": "sample", "domain": "d", "path": "p", "roles": ["psf"],
            "source": "source.png", "captures": ["capture.jpg"],
            "roi_xyxy": [0, 0, 20, 20]
        }]
    }
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "source.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "capture.jpg")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="outside capture bounds"):
        load_manifest(path)
```

- [ ] **Step 2: Run the schema tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_degradation_schema.py -q`

Expected: collection fails because `analysis.degradation.schema` does not exist.

- [ ] **Step 3: Implement immutable schema types and strict loading**

Define `CaptureGroup` and `CalibrationManifest` as frozen dataclasses. Resolve every source and capture path relative to the manifest's `workspace_root`, expand capture globs in natural burst order, validate RGB image dimensions, require `x0 < x1`, `y0 < y1`, bounds containment, nonempty frames, one domain/path string, and roles from:

```python
ALLOWED_ROLES = frozenset({
    "background", "photometry", "geometry", "psf", "mtf",
    "noise", "validation", "external_validation",
})
```

Expose the exact public signature `load_manifest(path: Path) -> CalibrationManifest` and export it from
`analysis.degradation`. `CalibrationManifest.domains` is a sorted tuple derived from group labels;
it is never duplicated in JSON.

The loader must not contain any focal length, group ID, ROI, or filename literal.

- [ ] **Step 4: Add the current manifest**

Use `workspace_root: ".."`, source root `images/origin/calibration_B`, and capture root `images/real/calibration_B`. Add these current path annotations:

```json
{
  "00": [2240, 820, 2460, 1080],
  "01": [2240, 820, 2460, 1080],
  "05": [2240, 820, 2460, 1080],
  "18": [2123, 349, 2537, 950],
  "19": [2300, 500, 2750, 1250],
  "22": [2850, 1100, 3536, 2300],
  "23": [2850, 1100, 3536, 2300],
  "24": [2900, 1000, 3350, 1950],
  "28": [2750, 550, 3250, 1150],
  "19_2": [1100, 100, 2500, 1250],
  "30": [850, 750, 1650, 1550],
  "30_2": [850, 750, 1650, 1550]
}
```

Record source feature metadata in the manifest: point spacing 150 source pixels and point sizes 3/7; checker cell size 120; color-bar axis `y` with eight ordered RGB values; slanted-edge layout 3 x 3. Assign `30_2` to validation and `30` to external validation.

- [ ] **Step 5: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_degradation_schema.py -q`

Expected: all schema tests pass.

Commit only the four Task 1 files with message `feat: add degradation dataset manifest`.

### Task 2: Implement the generic forward operator

**Files:**
- Create: `analysis/degradation/model.py`
- Create: `tests/test_degradation_model.py`
- Modify: `analysis/degradation/__init__.py`

- [ ] **Step 1: Write failing deterministic and behavior tests**

Define a typed parameter payload and test identity behavior without JPEG/noise, deterministic seeded noise, anisotropic output scale, non-identity color mixing, and actual JPEG metadata:

```python
from dataclasses import replace

import numpy as np

from analysis.degradation.model import DegradationParameters, degrade


def test_identity_forward_model_preserves_rgb() -> None:
    image = np.zeros((16, 20, 3), np.float32)
    image[4:12, 6:14] = (0.2, 0.6, 0.9)
    params = DegradationParameters.identity(output_size=(20, 16))
    actual = degrade(image, params, seed=7, encode_jpeg=False)
    np.testing.assert_allclose(actual, image, atol=1e-6)


def test_seeded_degradation_is_reproducible() -> None:
    image = np.full((32, 32, 3), 0.5, np.float32)
    params = replace(
        DegradationParameters.identity(output_size=(32, 32)),
        noise_slope=(0.01, 0.01, 0.01), jpeg_quality=96, jpeg_subsampling=2
    )
    first = degrade(image, params, seed=31, encode_jpeg=True)
    second = degrade(image, params, seed=31, encode_jpeg=True)
    assert np.array_equal(first, second)
```

- [ ] **Step 2: Run the model tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_degradation_model.py -q`

Expected: collection fails because `analysis.degradation.model` does not exist.

- [ ] **Step 3: Implement the operator in physical/effective order**

Implement `DegradationParameters` with normalized source-to-output homography, output size, anisotropic Gaussian FWHM/angle, 3 x 3 color matrix, RGB gain/bias/gamma, quadratic background coefficients, signal-dependent noise slope/intercept, JPEG quality and subsampling. Validate finite values, positive dimensions, positive gamma, nonnegative noise variance, normalized nonnegative blur kernels, and legal JPEG settings.

Implement:

```python
def degrade(
    image: np.ndarray,
    params: DegradationParameters,
    *,
    seed: int,
    encode_jpeg: bool = True,
) -> np.ndarray:
    """Return float32 RGB in [0, 1] after warp, attenuation/background,
    blur/sampling, color/tone response, heteroscedastic noise, and JPEG."""
```

Use `cv2.warpPerspective`, a covariance-derived anisotropic kernel, `cv2.filter2D`, NumPy color operations, `numpy.random.Generator`, and in-memory Pillow JPEG encoding. Preserve identity exactly when all optional effects are disabled.

- [ ] **Step 4: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_degradation_model.py -q`

Expected: all model tests pass.

Commit Task 2 files with message `feat: add effective degradation operator`.

### Task 3: Fit JPEG, geometry, blur, photometry, and noise generically

**Files:**
- Create: `analysis/degradation/fitting.py`
- Create: `tests/test_degradation_fitting.py`
- Modify: `analysis/degradation/__init__.py`

- [ ] **Step 1: Write synthetic estimator tests**

Generate in-memory point grids, affine checkerboards, eight color bars, and flat burst patches with known parameters. Assert:

```python
def test_point_grid_recovers_scale_and_fwhm() -> None:
    result = fit_point_grid(synthetic_point_burst(), {"point_spacing_px": 40})
    assert result.status == "estimated"
    assert result.value["scale_camera_per_source"] == pytest.approx((0.5, 0.4), abs=0.03)
    assert result.value["fwhm_camera_px"] == pytest.approx((3.0, 5.0), abs=0.8)


def test_noise_fit_ignores_high_gradient_pixels() -> None:
    result = fit_burst_noise(synthetic_heteroscedastic_burst(), {})
    assert result.status == "estimated"
    assert result.value["slope_rgb"] == pytest.approx((0.02, 0.01, 0.03), abs=0.006)


def test_color_fit_prefers_diagonal_when_full_matrix_does_not_generalize() -> None:
    result = fit_color_response(
        diagonal_observations(),
        {"colors_rgb": SOURCE_BARS, "bar_axis": "x", "bar_order": "forward"},
    )
    assert result.value["selected_model"] == "diagonal"
    assert result.value["leave_one_out_mae"] < 0.02
```

Define the synthetic helpers in the same test file with fixed seeds: a 6 x 5 point grid at 40 px
source spacing warped to scales 0.5/0.4 then blurred to 3/5 px FWHM; 24 flat 64 x 64 RGB
frames with variance slopes 0.02/0.01/0.03 plus intercept 0.0004; and eight standard RGB/CMY/
white/black bars transformed by diagonal gains `(0.5, 0.8, 1.1)` and offset `0.03`.

Also add a current-data test asserting all 120 frames are indexed, all quantization tables match, quality is 96, subsampling is 4:2:0, both domains are returned, and unsupported gamma/independent sensor noise fields are `not_identifiable`.

- [ ] **Step 2: Run estimator tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_degradation_fitting.py -q`

Expected: collection fails because fitting functions do not exist.

- [ ] **Step 3: Implement shared result contracts**

Every estimator returns a JSON-serializable dataclass containing `status`, `reason`, `sample_count`, `value`, `units`, `coordinate_system`, and `uncertainty`. Provide `estimated`, `provisional`, and `not_identifiable` constructors. Never emit NaN/Infinity; unsupported terms carry `value: null`.

- [ ] **Step 4: Implement JPEG and EXIF inspection**

Hash quantization tables, identify standard Pillow-equivalent quality by exhaustive 0--100 table comparison, read sampling factors, focal length, exposure time, ISO, and white-balance mode. Aggregate by domain but retain per-frame metadata. Reject mixed JPEG tables within a group.

- [ ] **Step 5: Implement modular robust fits**

Implement these array-level estimators and manifest orchestration without group literals:

```python
ROLE_FITTERS = {
    "geometry": fit_checker_geometry,
    "psf": fit_point_grid,
    "mtf": fit_edge_width,
    "photometry": fit_color_response,
    "noise": fit_burst_noise,
}


def fit_group(group: CaptureGroup) -> dict[str, FitResult]:
    frames = load_rgb_rois(group)
    return {
        role: ROLE_FITTERS[role](frames, group.features)
        for role in sorted(group.roles & ROLE_FITTERS.keys())
    }


def fit_manifest(manifest: CalibrationManifest) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "domains": {
            domain: {
                group.id: fit_group(group)
                for group in manifest.groups
                if group.domain == domain
            }
            for domain in manifest.domains
        },
    }
```

Use connected components and robust medians for grids; local edge-gradient profiles for checker/MTF; per-bar central medians and robust least squares for color; low-gradient, exposure-normalized temporal residuals for noise. Compare diagonal and full color models using leave-one-bar-out MAE and select the full matrix only when it improves error by at least 10% without negative channel response. Bootstrap frames with a fixed seed and store median/IQR; label intervals descriptive when fewer than 20 independent captures exist.

- [ ] **Step 6: Run focused and full tests, then commit**

Run:

```bash
.venv/bin/python -m pytest tests/test_degradation_fitting.py -q
.venv/bin/python -m pytest tests/test_degradation_schema.py tests/test_degradation_model.py tests/test_degradation_fitting.py -q
```

Expected: all tests pass.

Commit Task 3 files with message `feat: fit effective degradation parameters`.

### Task 4: Generate results, diagnostics, and scientific report

**Files:**
- Create: `analysis/degradation/reporting.py`
- Create: `analysis/run_fit_v1.py`
- Modify: `tests/test_degradation_fitting.py`
- Create: `analysis/results/v1_parameters.json`
- Create: `analysis/results/v1_report.md`
- Create: `analysis/results/v1_diagnostics/*.png`

- [ ] **Step 1: Write failing output-contract tests**

Use a temporary output directory and assert that one run produces valid JSON, Markdown, an ROI overlay per group, and at least one source/synthetic/real/residual comparison per domain. Assert rerunning with the same seed gives identical JSON and diagnostic PNG bytes. Assert the report contains `Identifiability`, `47 mm`, `69 mm`, `JPEG quality 96`, and `下一轮拍摄`.

- [ ] **Step 2: Run output tests and verify RED**

Run: `.venv/bin/python -m pytest tests/test_degradation_fitting.py -q`

Expected: fails because `write_results` and the CLI do not exist.

- [ ] **Step 3: Implement deterministic reporting**

Write JSON with sorted keys and no platform paths outside the workspace. Generate overlays showing the configured ROI and labels. Generate comparison figures at fixed pixel dimensions and DPI. The report must separate measurements, provisional approximations, not-identifiable terms, validation metrics, and limitations; do not turn the fit status into narrative certainty.

- [ ] **Step 4: Implement the CLI and generate current results**

Expose:

```bash
.venv/bin/python analysis/run_fit_v1.py \
  --manifest analysis/calibration_b_v1.json \
  --output analysis/results \
  --seed 20260721
```

Expected: writes `v1_parameters.json`, `v1_report.md`, and `v1_diagnostics/`, then prints domain names and key fitted metrics.

- [ ] **Step 5: Inspect diagnostics and fix estimator defects via tests**

Open every generated overlay and comparison. ROI labels must be inside the 4096 x 3072 frame; main paths must be visibly covered; comparison panels must not be blank; residuals must use a fixed signed color scale. Add a failing regression test before correcting any bad ROI or estimator behavior.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_degradation_schema.py tests/test_degradation_model.py tests/test_degradation_fitting.py -q`

Expected: all tests pass and a second CLI run leaves generated output byte-identical.

Commit reporting, CLI, test changes, manifest, and generated v1 results with message `analysis: fit smart-glasses degradation v1`.

### Task 5: Update the research document and next-capture matrix

**Files:**
- Modify: `docs/Shouldersurfing Smart Glass .md`
- Modify: `analysis/results/v1_report.md`

- [ ] **Step 1: Replace the preliminary degradation subsection**

Replace only the first `退化建模` subsection and preserve the remaining project proposal. Add:

1. scope: path-specific JPEG-space effective model;
2. complete forward equation and symbol table;
3. parameter hierarchy and per-frame nuisance variables;
4. actual 47/69 mm fitted values with status and uncertainty copied from generated JSON;
5. fitting objective and validation metrics;
6. explicit identifiability limitations;
7. citations to Real-ESRGAN, BSRGAN degradation modeling, Unprocessing Images, and ghosting-cue reflection modeling.

Do not copy provisional values into prose without a `暂定` marker. Do not retain the old statement that a single `k`, downsampling, noise, and additive color shift form the complete model.

- [ ] **Step 2: Add an actionable generalization capture matrix**

Specify the following next round for each combination of device, focal length, distance, yaw, pitch, and major ambient-light condition:

| Capture | Frames | Purpose |
|---|---:|---|
| asymmetric grid immediately before every target | 10 | absolute crop, mirror state, homography, path identity |
| black + 16-level gray ramp in one locked-exposure frame | 10 | response curve, black level, heteroscedastic noise |
| spatial RGB/CMY/gray patches in one frame | 10 | color matrix and spatial attenuation |
| sparse 3/7 px points at 3 x 3 field positions | 10 each | spatially varying PSF |
| horizontal/vertical/diagonal slanted edges | 10 each | anisotropic MTF and sharpening |
| checkerboard + exact inverse without moving camera | 10 each | flare/background separation |
| all-black burst at each ISO/exposure setting | 30 | read/fixed-pattern noise |
| repeated gray levels at each exposure | 30 | shot-noise slope |
| 20 independently generated text charts | 5--10 each | content-generalization validation |
| RAW/DNG + JPEG pairs | 10 per pattern | separate sensor noise from ISP/JPEG |

Require locked focus, exposure, ISO, white balance, focal length, disabled HDR/beauty/night processing, preserved EXIF, a pose ID, path ID, distance, yaw, pitch, illumination reading, and no movement between calibration/target pairs. Recommend at least three devices and three independently repeated sessions for device/session generalization.

- [ ] **Step 3: Verify document consistency**

Run a script/test that parses `v1_parameters.json` and asserts every numeric fitted value quoted in the document exists verbatim in JSON. Run `rg` to ensure the obsolete `I_blur` equation and unsupported fixed HSV/noise ranges are removed from the first subsection but later proposal content is preserved.

- [ ] **Step 4: Commit documentation**

Commit only the target document and final report update with message `docs: document fitted degradation model`.

### Task 6: Final reproducibility verification

**Files:**
- Verify all files from Tasks 1--5.

- [ ] **Step 1: Run the full relevant test suite**

Run:

```bash
.venv/bin/python -m pytest tests/test_degradation_schema.py tests/test_degradation_model.py tests/test_degradation_fitting.py -q
.venv/bin/python -m unittest tests.test_fit_v0 -v
```

Expected: all new tests and all three legacy v0 tests pass.

- [ ] **Step 2: Regenerate v1 in a clean temporary output and compare**

Run the v1 CLI to `/tmp/glasses-v1-repro`, then compare its JSON, Markdown, and diagnostic hashes with `analysis/results`. Expected: exact match apart from an explicitly excluded generation path; no timestamp fields are allowed.

- [ ] **Step 3: Inspect repository scope**

Run `git status --short` and verify real captures, cropped images, the original screenshot, and unrelated files were not staged or modified. Run `git diff --check` on all changed tracked files.

- [ ] **Step 4: Summarize scientific outcome**

Report which terms are estimated, provisional, and not identifiable; key parameter ranges; validation error; diagnostic locations; and the prioritized next-capture protocol. Do not claim physical parameter separation or cross-device generalization until the capture matrix is collected.
