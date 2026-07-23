from __future__ import annotations

import json
from pathlib import Path
import random

from PIL import Image, ImageDraw

try:
    from images.generate_calibration_v2_followup import (
        HEIGHT,
        PATCH_COLORS,
        PATCH_IDS,
        WIDTH,
        _draw_small_locator,
        _response_assignments,
        inset_asymmetric_locator,
        response_chart,
    )
except ModuleNotFoundError:
    from generate_calibration_v2_followup import (
        HEIGHT,
        PATCH_COLORS,
        PATCH_IDS,
        WIDTH,
        _draw_small_locator,
        _response_assignments,
        inset_asymmetric_locator,
        response_chart,
    )


OUTPUT_DIR = Path(__file__).parent / "origin" / "calibration_v3_core"
LEVELS = (0, 36, 73, 109, 146, 182, 219, 255)
EDGE_ORIENTATIONS = (
    "vertical_pos",
    "vertical_neg",
    "horizontal_pos",
    "horizontal_neg",
)


def _third_response_assignments() -> list[str]:
    assignments = list(PATCH_IDS) * 4
    random.Random(20260722).shuffle(assignments)
    return assignments


def _patch_chart(
    assignments: list[str], colors: dict[str, tuple[int, int, int]]
) -> tuple[Image.Image, list[dict[str, object]]]:
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
        color = colors[patch_id]
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


def red_gray_ramp() -> tuple[Image.Image, list[dict[str, object]]]:
    colors: dict[str, tuple[int, int, int]] = {}
    assignments: list[str] = []
    for level in LEVELS:
        colors[f"R{level:03d}"] = (level, 0, 0)
        colors[f"K{level:03d}"] = (level, level, level)
        assignments.extend([f"R{level:03d}"] * 4)
        assignments.extend([f"K{level:03d}"] * 3)
    random.Random(20260723).shuffle(assignments)
    return _patch_chart(assignments, colors)


def _draw_slanted_half_plane(
    draw: ImageDraw.ImageDraw,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    orientation: str,
) -> float:
    x0, y0, x1, y1 = bbox
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    slope = 0.12 if orientation.endswith("pos") else -0.12

    if orientation.startswith("vertical"):
        edge_top = center_x + slope * (y0 - center_y)
        edge_bottom = center_x + slope * (y1 - center_y)
        polygon = ((x0, y0), (round(edge_top), y0), (round(edge_bottom), y1), (x0, y1))
    else:
        edge_left = center_y + slope * (x0 - center_x)
        edge_right = center_y + slope * (x1 - center_x)
        polygon = ((x0, y0), (x1, y0), (x1, round(edge_right)), (x0, round(edge_left)))
    draw.polygon(polygon, fill=color)
    return slope


def gb_slanted_edges() -> tuple[Image.Image, list[dict[str, object]]]:
    image = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(image)
    _draw_small_locator(draw)
    rows, columns = 4, 4
    grid_left, grid_top = 430, 290
    grid_width, grid_height = 1640, 1020
    gap = 40
    cell_width = (grid_width - (columns - 1) * gap) // columns
    cell_height = (grid_height - (rows - 1) * gap) // rows
    cells: list[dict[str, object]] = []

    for index in range(rows * columns):
        row, column = divmod(index, columns)
        x0 = grid_left + column * (cell_width + gap)
        y0 = grid_top + row * (cell_height + gap)
        x1 = x0 + cell_width - 1
        y1 = y0 + cell_height - 1
        color_id = "G" if index % 2 == 0 else "B"
        color = PATCH_COLORS[color_id]
        orientation = EDGE_ORIENTATIONS[(index // 2) % len(EDGE_ORIENTATIONS)]
        slope = _draw_slanted_half_plane(draw, (x0, y0, x1, y1), color, orientation)
        cells.append(
            {
                "index": index,
                "row": row,
                "column": column,
                "color_id": color_id,
                "source_rgb": color,
                "orientation": orientation,
                "slope": slope,
                "bbox_xyxy": (x0, y0, x1, y1),
                "center_xy": ((x0 + x1) / 2, (y0 + y1) / 2),
            }
        )

    return image, cells


def build_patterns() -> tuple[dict[str, Image.Image], dict[str, object]]:
    assignments_a, assignments_b = _response_assignments()
    assignments_c = _third_response_assignments()
    response_a, manifest_a = response_chart(assignments_a)
    response_b, manifest_b = response_chart(assignments_b)
    response_c, manifest_c = response_chart(assignments_c)
    ramp, ramp_manifest = red_gray_ramp()
    edges, edge_manifest = gb_slanted_edges()
    locator = inset_asymmetric_locator()
    black = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))

    patterns = {
        "00_locator_start.png": locator.copy(),
        "01_full_black_start.png": black.copy(),
        "02_permuted_response_A.png": response_a,
        "03_permuted_response_B.png": response_b,
        "04_permuted_response_C.png": response_c,
        "05_red_gray_ramp.png": ramp,
        "06_GB_slanted_edges.png": edges,
        "07_full_black_end.png": black.copy(),
        "08_locator_end.png": locator.copy(),
    }
    manifest = {
        "source_size": (WIDTH, HEIGHT),
        "sequence": list(patterns),
        "patch_colors": PATCH_COLORS,
        "response_charts": {
            "A": {"assignments": manifest_a},
            "B": {"assignments": manifest_b},
            "C": {"assignments": manifest_c},
        },
        "red_gray_ramp": {"levels": LEVELS, "assignments": ramp_manifest},
        "gb_slanted_edges": {"cells": edge_manifest},
    }
    return patterns, manifest


def main() -> None:
    patterns, manifest = build_patterns()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in OUTPUT_DIR.glob("*.png"):
        old_image.unlink()
    for filename, image in patterns.items():
        image.save(OUTPUT_DIR / filename, format="PNG", compress_level=9)
    (OUTPUT_DIR / "pattern_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(f"generated {len(patterns)} PNG files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
