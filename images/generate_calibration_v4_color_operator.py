from __future__ import annotations

from itertools import product
import json
from pathlib import Path
import random

from PIL import Image

try:
    from images.generate_calibration_v3_core import HEIGHT, WIDTH, _patch_chart
except ModuleNotFoundError:
    from generate_calibration_v3_core import HEIGHT, WIDTH, _patch_chart


OUTPUT_DIR = Path(__file__).parent / "origin" / "calibration_v4_color_operator"
ANCHOR_COLORS = {
    "K000": (0, 0, 0),
    "K128": (128, 128, 128),
    "K255": (255, 255, 255),
}


def _non_gray_color_nodes(
    levels: tuple[int, int, int], prefix: str
) -> dict[str, tuple[int, int, int]]:
    return {
        f"{prefix}_{red:03d}_{green:03d}_{blue:03d}": (red, green, blue)
        for red, green, blue in product(levels, repeat=3)
        if not (red == green == blue)
    }


TRAINING_COLORS = _non_gray_color_nodes((0, 128, 255), "T")
HOLDOUT_COLORS = _non_gray_color_nodes((64, 128, 192), "V")
TRAINING_COLOR_IDS = tuple(TRAINING_COLORS)
HOLDOUT_COLOR_IDS = tuple(HOLDOUT_COLORS)


def _chart_assignments(color_ids: tuple[str, ...], seed: int) -> list[str]:
    assignments = list(color_ids) * 2
    assignments.extend(("K000",) * 4)
    assignments.extend(("K128",) * 2)
    assignments.extend(("K255",) * 2)
    random.Random(seed).shuffle(assignments)
    return assignments


def build_patterns() -> tuple[dict[str, Image.Image], dict[str, object]]:
    training_palette = {**ANCHOR_COLORS, **TRAINING_COLORS}
    holdout_palette = {**ANCHOR_COLORS, **HOLDOUT_COLORS}
    training_charts: dict[str, dict[str, object]] = {}
    chart_images: dict[str, Image.Image] = {}
    for name, seed in zip(("A", "B", "C"), (20260731, 20260801, 20261007)):
        assignments = _chart_assignments(TRAINING_COLOR_IDS, seed)
        image, rows = _patch_chart(assignments, training_palette)
        chart_images[name] = image
        training_charts[name] = {"seed": seed, "assignments": rows}

    holdout_assignments = _chart_assignments(HOLDOUT_COLOR_IDS, 20260803)
    holdout_image, holdout_rows = _patch_chart(
        holdout_assignments, holdout_palette
    )
    black = Image.new("RGB", (WIDTH, HEIGHT), (0, 0, 0))
    patterns = {
        "00_full_black_start.png": black.copy(),
        "01_joint_color_train_A.png": chart_images["A"],
        "02_joint_color_train_B.png": chart_images["B"],
        "03_joint_color_train_C.png": chart_images["C"],
        "04_joint_color_holdout.png": holdout_image,
        "05_full_black_end.png": black.copy(),
    }
    manifest = {
        "source_size": (WIDTH, HEIGHT),
        "sequence": list(patterns),
        "training_levels_rgb": (0, 128, 255),
        "holdout_levels_rgb": (64, 128, 192),
        "anchor_colors": ANCHOR_COLORS,
        "training_colors": TRAINING_COLORS,
        "holdout_colors": HOLDOUT_COLORS,
        "training_charts": training_charts,
        "holdout_chart": {
            "seed": 20260803,
            "assignments": holdout_rows,
            "role": "validation_only_do_not_fit",
        },
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
