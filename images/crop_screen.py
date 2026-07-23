from pathlib import Path

from PIL import Image, ImageEnhance


SOURCE_DIR = Path(__file__).parent / "real"
OUTPUT_DIR = Path(__file__).parent / "cropped"

# Shared acquisition setup: all three burst images place the leaked display here.
ROI = (2050, 1680, 3536, 3072)

# Coordinates inside ROI, ordered for PIL's QUAD mapping as
# output NW, SW, SE, NE. The visible display is rotated roughly 56 degrees.
DISPLAY_QUAD = (519, 273, 191, 481, 482, 944, 807, 742)
RECTIFIED_SIZE = (1000, 600)


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for source in sorted(SOURCE_DIR.glob("*.jpg")):
        with Image.open(source) as image:
            crop = image.crop(ROI)
            crop.save(OUTPUT_DIR / f"{source.stem}_screen_roi.png")

            # Brightened only for locating the display boundary; never use it for
            # photometric parameter estimation.
            preview = ImageEnhance.Brightness(crop).enhance(2.2)
            preview.save(OUTPUT_DIR / f"{source.stem}_screen_roi_preview.png")

            rectified = crop.transform(
                RECTIFIED_SIZE,
                Image.Transform.QUAD,
                DISPLAY_QUAD,
                resample=Image.Resampling.BICUBIC,
            )
            # The observed optical path vertically mirrors the source display:
            # text size decreases from top to bottom while digit order is kept.
            rectified = rectified.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            rectified.save(OUTPUT_DIR / f"{source.stem}_screen_rectified.png")
            rectified_preview = ImageEnhance.Brightness(rectified).enhance(2.2)
            rectified_preview.save(
                OUTPUT_DIR / f"{source.stem}_screen_rectified_preview.png"
            )


if __name__ == "__main__":
    main()
