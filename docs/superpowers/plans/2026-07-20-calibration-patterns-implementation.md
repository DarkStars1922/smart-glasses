# Calibration Pattern Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate and verify the approved 32-image, 2515 x 1491 smart-glasses calibration suite under `images/origin/calibration_B/`.

**Architecture:** A deterministic Pillow generator exposes a `build_suite()` function that returns typed pattern records and a `generate()` function that writes validated PNGs plus metadata. Standard-library `unittest` tests the suite contract before any files are accepted; generated images are then visually inspected as contact sheets.

**Tech Stack:** Python 3.11, Pillow, NumPy, standard-library `unittest`/`csv`/`tempfile`, Windows Git Bash

---

### Task 1: Specify the generator contract with failing tests

**Files:**
- Create: `tests/test_generate_calibration_b.py`
- Create: `images/generate_calibration_b.py`

- [ ] **Step 1: Write a failing manifest-contract test**

Create a test that imports `CANVAS_SIZE`, `EXPECTED_FILENAMES`, `build_suite`, and
`generate`, then asserts the approved filename order:

```python
EXPECTED = [
    "00_black.png", "01_white.png",
    "02_gray_032.png", "03_gray_064.png", "04_gray_096.png",
    "05_gray_128.png", "06_gray_160.png", "07_gray_192.png",
    "08_gray_224.png", "09_red_128.png", "10_red_255.png",
    "11_green_128.png", "12_green_255.png", "13_blue_128.png",
    "14_blue_255.png", "15_cyan_255.png", "16_magenta_255.png",
    "17_yellow_255.png", "18_color_bars_full.png",
    "19_checkerboard.png", "20_checkerboard_inverse.png",
    "21_asymmetric_grid.png", "22_point_grid_3px.png",
    "23_point_grid_7px.png", "24_slanted_edges_3x3.png",
    "25_resolution_lines_horizontal.png",
    "26_resolution_lines_vertical.png",
    "27_resolution_lines_diagonal.png", "28_siemens_star.png",
    "29_spatial_color_patches.png", "30_text_dark.png",
    "31_text_light.png",
]

def test_suite_has_approved_order_and_canvas(self):
    suite = build_suite()
    self.assertEqual(CANVAS_SIZE, (2515, 1491))
    self.assertEqual(EXPECTED_FILENAMES, EXPECTED)
    self.assertEqual([item.filename for item in suite], EXPECTED)
    self.assertTrue(all(item.image.size == CANVAS_SIZE for item in suite))
    self.assertTrue(all(item.image.mode == "RGB" for item in suite))
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
'/mnt/c/Users/Darkstars/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest discover -s tests -p 'test_generate_calibration_b.py' -v
```

Expected: `ModuleNotFoundError` because `images.generate_calibration_b` does not exist.

- [ ] **Step 3: Add the typed generator skeleton**

Create `Pattern` and the public API, initially constructing exact solid fields and
minimal structured images only far enough for the manifest test:

```python
@dataclass(frozen=True)
class Pattern:
    filename: str
    category: str
    parameters: str
    purpose: str
    image: Image.Image

CANVAS_SIZE = (2515, 1491)

def build_suite() -> list[Pattern]:
    patterns = build_photometric_patterns()
    patterns.extend(build_geometry_patterns())
    patterns.extend(build_resolution_patterns())
    patterns.extend(build_validation_patterns())
    if [p.filename for p in patterns] != EXPECTED_FILENAMES:
        raise RuntimeError("generated suite does not match approved manifest")
    return patterns
```

- [ ] **Step 4: Run the manifest test and verify GREEN**

Run the command from Step 2. Expected: one passing test.

### Task 2: Verify exact photometric and geometric behavior

**Files:**
- Modify: `tests/test_generate_calibration_b.py`
- Modify: `images/generate_calibration_b.py`

- [ ] **Step 1: Write failing pixel-accuracy tests**

Add tests that assert all 18 uniform fields contain exactly one expected RGB triplet,
the normal and inverse checkerboards sum to 255 in every channel, and each structured
pattern contains at least two distinct pixel values:

```python
def test_uniform_fields_are_exact(self):
    for filename, expected_rgb in EXPECTED_SOLIDS.items():
        image = next(p.image for p in build_suite() if p.filename == filename)
        self.assertEqual(image.getextrema(), tuple((v, v) for v in expected_rgb))

def test_checkerboards_are_exact_complements(self):
    by_name = {p.filename: p.image for p in build_suite()}
    normal = np.asarray(by_name["19_checkerboard.png"], dtype=np.uint16)
    inverse = np.asarray(by_name["20_checkerboard_inverse.png"], dtype=np.uint16)
    self.assertTrue(np.all(normal + inverse == 255))
```

- [ ] **Step 2: Run tests and verify the minimal structured images fail**

Expected: failures identify the minimal checkerboard and nonuniform-pattern
requirements, not import or setup errors.

- [ ] **Step 3: Implement exact pattern builders**

Implement integer-coordinate Pillow builders for color bars, checkerboards,
orientation-marked asymmetric grid, 3 px and 7 px point grids, 3 x 3 slanted-edge
patches, multi-frequency horizontal/vertical/diagonal lines, and a 144-sector
Siemens star. All calibration edges are hard; only validation text may be antialiased.

- [ ] **Step 4: Run all tests and verify GREEN**

Expected: the manifest, exact solids, checkerboard complement, and nonempty structure
tests all pass.

### Task 3: Add spatial color, text validation, and deterministic output

**Files:**
- Modify: `tests/test_generate_calibration_b.py`
- Modify: `images/generate_calibration_b.py`

- [ ] **Step 1: Write failing output-contract tests**

Use a temporary directory to verify exactly 32 PNG files, `manifest.csv`, and
`README.md`; verify contiguous manifest indices and byte-identical PNGs across two
independent generations:

```python
def test_generate_writes_deterministic_complete_output(self):
    with TemporaryDirectory() as first, TemporaryDirectory() as second:
        generate(Path(first))
        generate(Path(second))
        first_pngs = sorted(Path(first).glob("*.png"))
        second_pngs = sorted(Path(second).glob("*.png"))
        self.assertEqual([p.name for p in first_pngs], EXPECTED)
        self.assertEqual(
            [p.read_bytes() for p in first_pngs],
            [p.read_bytes() for p in second_pngs],
        )
        self.assertTrue((Path(first) / "manifest.csv").is_file())
        self.assertTrue((Path(first) / "README.md").is_file())
```

- [ ] **Step 2: Run the test and verify RED**

Expected: missing output metadata or incomplete `generate()` behavior.

- [ ] **Step 3: Implement the final three images and writer**

Generate a separated spatial color-patch grid and two multilingual text charts using
Microsoft YaHei when available, with deterministic font fallback. Write all files to
a sibling temporary directory, validate the suite, replace only the requested output
directory, and write UTF-8 `manifest.csv` and capture-oriented `README.md`.

- [ ] **Step 4: Run the full test suite and verify GREEN**

Run:

```bash
'/mnt/c/Users/Darkstars/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' -m unittest discover -s tests -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 5: Commit generator and tests with Windows Git Bash**

```bash
'/mnt/d/Code_Tools/Git/bin/bash.exe' -lc \
  'cd /d/Worker/work/glasses && git add images/generate_calibration_b.py tests/test_generate_calibration_b.py && git commit -m "feat: add calibration pattern generator"'
```

### Task 4: Generate, visually inspect, and commit the calibration suite

**Files:**
- Create: `images/origin/calibration_B/*.png`
- Create: `images/origin/calibration_B/manifest.csv`
- Create: `images/origin/calibration_B/README.md`

- [ ] **Step 1: Generate the approved suite**

```bash
'/mnt/c/Users/Darkstars/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' images/generate_calibration_b.py
```

Expected: reports 32 PNG images written to `images/origin/calibration_B`.

- [ ] **Step 2: Run fresh output verification**

Run the full unittest command and a separate script that opens every generated PNG,
checks mode and dimensions, verifies manifest row count, and reports total bytes.

- [ ] **Step 3: Build and inspect category contact sheets**

Create temporary contact sheets outside `images/origin/calibration_B`, inspect them
with the image viewer, and confirm hard edges, orientation markers, frequency bands,
star geometry, color-patch separation, and text containment. Fix any visual defect via
a failing regression test before changing the generator.

- [ ] **Step 4: Commit generated assets with Windows Git Bash**

```bash
'/mnt/d/Code_Tools/Git/bin/bash.exe' -lc \
  'cd /d/Worker/work/glasses && git add images/origin/calibration_B && git commit -m "assets: add smart-glasses calibration suite"'
```

- [ ] **Step 5: Confirm unrelated files remain untracked**

Run `git status --short` and verify the pre-existing source screenshot, real photos,
crop outputs, and shoulder-surfing document were not accidentally committed.
