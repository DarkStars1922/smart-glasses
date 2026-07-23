from __future__ import annotations

import json
from pathlib import Path
import random

from PIL import Image, ImageDraw


WIDTH = 2500
HEIGHT = 1600
OUTPUT_DIR = Path(__file__).parent / "origin" / "calibration_v2_followup"
PATCH_COLORS = {
    "K000": (0, 0, 0),
    "K036": (36, 36, 36),
    "K073": (73, 73, 73),
    "K109": (109, 109, 109),
    "K146": (146, 146, 146),
    "K182": (182, 182, 182),
    "K219": (219, 219, 219),
    "K255": (255, 255, 255),
    "R": (255, 0, 0),
    "G": (0, 255, 0),
    "B": (0, 0, 255),
    "C": (0, 255, 255),
    "M": (255, 0, 255),
    "Y": (255, 255, 0),
}
PATCH_IDS = tuple(PATCH_COLORS)
GRID_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
LARGE_POINT_DIAMETER = 31


def inset_asymmetric_locator() -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    color = (255, 255, 255)
    left = round(0.1 * WIDTH)
    top = round(0.1 * HEIGHT)
    right = round(0.9 * WIDTH) - 1
    bottom = round(0.9 * HEIGHT) - 1
    size = 140
    stroke = 18

    draw.rectangle((left, top, left + size, top + size), outline=color, width=stroke)
    draw.rectangle(
        (left + 50, top + 35, left + 76, top + 112), fill=color
    )
    draw.ellipse((right - size, top, right, top + size), outline=color, width=stroke)
    draw.ellipse((right - 88, top + 52, right - 52, top + 88), fill=color)
    draw.polygon(
        ((left, bottom), (left + size, bottom), (left, bottom - size)), fill=color
    )
    draw.rectangle(
        (right - size, bottom - size, right, bottom), fill=color
    )
    draw.rectangle(
        (right - size + 28, bottom - size + 28, right - 20, bottom - 20),
        fill=(0, 0, 0),
    )
    draw.rectangle(
        (right - 64, bottom - 58, right - 36, bottom - 30), fill=color
    )
    draw.line((WIDTH // 2 - 90, HEIGHT // 2, WIDTH // 2 + 90, HEIGHT // 2), fill=color, width=14)
    draw.line((WIDTH // 2, HEIGHT // 2 - 55, WIDTH // 2, HEIGHT // 2 + 105), fill=color, width=14)
    return image


def _draw_small_locator(draw: ImageDraw.ImageDraw) -> None:
    color = (210, 210, 210)
    left = round(0.1 * WIDTH)
    top = round(0.1 * HEIGHT)
    right = round(0.9 * WIDTH) - 1
    bottom = round(0.9 * HEIGHT) - 1
    size = 110
    stroke = 12
    draw.rectangle((left, top, left + size, top + size), outline=color, width=stroke)
    draw.ellipse((right - size, top, right, top + size), outline=color, width=stroke)
    draw.polygon(((left, bottom), (left + size, bottom), (left, bottom - size)), fill=color)
    draw.rectangle((right - size, bottom - size, right, bottom), outline=color, width=stroke)


def _response_assignments() -> tuple[list[str], list[str]]:
    first = list(PATCH_IDS) * 4
    random.Random(20260721).shuffle(first)
    shift = len(PATCH_IDS) // 2
    second = [PATCH_IDS[(PATCH_IDS.index(patch_id) + shift) % len(PATCH_IDS)] for patch_id in first]
    return first, second


def response_chart(assignments: list[str]) -> tuple[Image.Image, list[dict[str, object]]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), (12, 12, 12))
    draw = ImageDraw.Draw(image)
    _draw_small_locator(draw)
    rows, columns = 7, 8
    grid_left, grid_top = 430, 290
    grid_width, grid_height = 1640, 1020
    gap = 8
    patch_width = (grid_width - (columns - 1) * gap) // columns
    patch_height = (grid_height - (rows - 1) * gap) // rows
    manifest: list[dict[str, object]] = []
    for index, patch_id in enumerate(assignments):
        row, column = divmod(index, columns)
        x0 = grid_left + column * (patch_width + gap)
        y0 = grid_top + row * (patch_height + gap)
        x1 = x0 + patch_width - 1
        y1 = y0 + patch_height - 1
        color = PATCH_COLORS[patch_id]
        draw.rectangle((x0, y0, x1, y1), fill=color)
        manifest.append(
            {
                "index": index,
                "row": row,
                "column": column,
                "patch_id": patch_id,
                "source_rgb": color,
                "bbox_xyxy": (x0, y0, x1, y1),
                "center_xy": ((x0 + x1) / 2, (y0 + y1) / 2),
            }
        )
    return image, manifest


def large_point_grid(color: tuple[int, int, int]) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    radius = LARGE_POINT_DIAMETER // 2
    for row_fraction in GRID_FRACTIONS:
        for column_fraction in GRID_FRACTIONS:
            center_x = round(column_fraction * (WIDTH - 1))
            center_y = round(row_fraction * (HEIGHT - 1))
            draw.rectangle(
                (
                    center_x - radius,
                    center_y - radius,
                    center_x + radius,
                    center_y + radius,
                ),
                fill=color,
            )
    return image


def build_patterns() -> tuple[list[tuple[str, Image.Image]], dict[str, object]]:
    assignments_a, assignments_b = _response_assignments()
    response_a, manifest_a = response_chart(assignments_a)
    response_b, manifest_b = response_chart(assignments_b)
    patterns = [
        ("00_inset_asymmetric_locator.png", inset_asymmetric_locator()),
        ("01_permuted_response_A.png", response_a),
        ("02_permuted_response_B.png", response_b),
        ("03_pointgrid_G_5x5_31px.png", large_point_grid((0, 255, 0))),
        ("04_pointgrid_B_5x5_31px.png", large_point_grid((0, 0, 255))),
    ]
    manifest = {
        "source_size": (WIDTH, HEIGHT),
        "patch_colors": PATCH_COLORS,
        "response_A": manifest_a,
        "response_B": manifest_b,
        "large_point_diameter_px": LARGE_POINT_DIAMETER,
        "large_point_grid_fractions": GRID_FRACTIONS,
        "large_point_colors": {"G": (0, 255, 0), "B": (0, 0, 255)},
    }
    return patterns, manifest


def main() -> None:
    patterns, manifest = build_patterns()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in OUTPUT_DIR.glob("*.png"):
        old_image.unlink()
    for filename, image in patterns:
        image.save(OUTPUT_DIR / filename, format="PNG", compress_level=9)
    (OUTPUT_DIR / "pattern_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated {len(patterns)} PNG files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
