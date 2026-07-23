# Smart-glasses degradation fit v0

## Scope

This fit describes the effective super-macro JPEG pipeline. Optical blur, demosaicing, denoising, sharpening, auto exposure, color processing, and JPEG are not fully separable in the current data.

## 47 mm domain

- Point-grid local scale: `0.4945 x 0.4210` camera pixels per source pixel.
- Observed 7 px dot FWHM: `7.83 x 9.65` camera pixels.
- Approximate source-coordinate effective FWHM: `15.8-22.9` source pixels.
- Valid dot samples: `558`.

## 69 mm domain

- Checker local scale: `0.4801 x 0.5087` camera pixels per source pixel.
- Edge-gradient FWHM: `6.00` camera pixels, approximately `12.1` source pixels.
- `19_2` and `30_2` share the same approximately 69 mm focal domain.
- Absolute mapping is unresolved because the visible checkerboard is periodic and spatially cropped.

## Photometry

- Black median RGB: `[19.159000396728516, 19.222000122070312, 16.841999053955078]`.
- White median RGB: `[58.21200180053711, 128.7830047607422, 169.67300415039062]`.
- Gray-128 median RGB: `[30.968000411987305, 69.62899780273438, 88.43000030517578]`.
- White/gray RGB ratio: `[1.88, 1.85, 1.919]`.
- Auto exposure and auto white balance are active, so these are JPEG-space relative measurements rather than a unique display/camera response curve.

## Conclusions

- The 69 mm matched checker/text pair is valid for local geometry and text recoverability.
- The 7 px point grid is visible; the 3 px point grid is below the reliable optical/ISP cutoff.
- Text at 40 px and above is visibly recoverable in the 69 mm domain; 16-24 px text remains marginal.
- Multiple reflection paths require path-specific crops or a mixture-of-warps term.

## Limitations

- A periodic checkerboard cannot identify the absolute source crop or reflection orientation.
- Automatic exposure and white balance prevent a unique gamma, color matrix, or sensor-noise fit.
- Only one fixed text chart is available, so OCR generalization cannot yet be measured.
- The 47 mm and 69 mm captures must remain separate conditional domains.

## 第二版拍摄

Required patterns:

- `21_asymmetric_grid`
- `29_spatial_color_patches`
- `32_gray_ramp_with_fiducials`
- `33_random_text_with_fiducials`

Capture protocol:

- Use 69 mm as the primary recovery domain and 47 mm as a secondary domain.
- At each pose, capture the asymmetric grid immediately before every target without moving the phone or glasses.
- Wait 3 seconds after switching patterns, then keep the sharpest 5-8 frames from a 10-frame burst.
- Capture the gray ramp and RGB/CMY patches in one frame so auto exposure is shared across all levels.
- Capture at least 20 independently generated text charts; keep 5-10 frames per chart.
- Preserve full JPEGs and EXIF; never mix focal domains without a domain label.
