# v3 Core Calibration Image Set Design

## Objective

Generate one self-contained, ordered set of nine 2500 x 1600 RGB PNG files for finishing the unresolved parameters of the current-pose 47 mm effective degradation model. The set targets only red-channel photometry, the effective color matrix, spatial additive background, and stable G/B PSF measurement. Cross-pose genericity is outside this set.

## Output

Output directory: `images/origin/calibration_v3_core/`

| Order | Filename | Content | Parameter target |
|---:|---|---|---|
| 00 | `00_locator_start.png` | Inset asymmetric locator | Initial absolute geometry |
| 01 | `01_full_black_start.png` | Uniform RGB zero image | Initial additive background field |
| 02 | `02_permuted_response_A.png` | Existing 14-class response multiset, permutation A | Tone, attenuation, color mixing |
| 03 | `03_permuted_response_B.png` | Existing 14-class response multiset, permutation B | Tone, attenuation, color mixing |
| 04 | `04_permuted_response_C.png` | Same multiset, independent permutation C | Reduce response-position coupling |
| 05 | `05_red_gray_ramp.png` | Spatially shuffled red and neutral-gray ramps | Red response and red attenuation |
| 06 | `06_GB_slanted_edges.png` | Sixteen alternating G/B slanted-edge cells | G/B edge-spread PSF without square-support deconvolution |
| 07 | `07_full_black_end.png` | Uniform RGB zero image | Background repeat and drift check |
| 08 | `08_locator_end.png` | Inset asymmetric locator | Final geometry and sequence drift check |

`00` and `08` are pixel-identical by design. `01` and `07` are also pixel-identical; they are separate filenames because both exposures are required in the sequence.

## Pattern Constraints

- Every PNG is exactly 2500 x 1600, RGB, and lossless.
- Response A/B/C use the same patch geometry and the same multiset: eight neutral levels plus R/G/B/C/M/Y, each repeated four times. Only spatial assignment changes.
- Response C is independently shuffled and is not a relabeling of A or B.
- The red-gray ramp fills all 56 response cells: every red level is repeated four times and every neutral-gray level three times for the eight levels 0, 36, 73, 109, 146, 182, 219, and 255.
- The G/B chart contains eight green and eight blue cells distributed over a 4 x 4 field grid. Each color has both near-vertical and near-horizontal edges with positive and negative slants.
- Response, ramp, and edge charts retain inset asymmetric locators. Full-black frames contain no locator light.
- A JSON manifest records filenames, patch assignments, source coordinates, colors, and edge orientation.

## Capture Contract

- Capture once per file, strictly from `00` through `08`; do not use burst mode.
- Keep phone, glasses, and display fixed. Only change the fullscreen PNG.
- Use the existing 47 mm automatic path: EV -2/3, S/ISO/WB/F automatic, universal tracking and telephoto adaptation enabled, HDR/AI scene optimization/night mode/beauty/filter disabled.
- Wait 2-3 seconds after each image change before taking the photograph.
- A batch is valid only if start/end locator registration shows no material pose change. The fitting stage will quantify this before combining response frames.

## Verification

- Automated tests check the exact nine filenames, dimensions, color mode, duplicate locator/black pairs, response multisets, independent C permutation, ramp counts, G/B cell counts, and manifest consistency.
- The generator deletes only obsolete PNG files inside its own output directory.
- Generated files are visually inspected as a contact sheet before handoff.
