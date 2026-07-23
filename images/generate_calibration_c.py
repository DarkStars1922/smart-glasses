from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw

try:
    from images.generate_calibration_b import HEIGHT, WIDTH, find_font, solid
except ModuleNotFoundError:
    from generate_calibration_b import HEIGHT, WIDTH, find_font, solid


OUTPUT_DIR = Path(__file__).parent / "origin" / "calibration_C"
GRID_FRACTIONS = (0.1, 0.3, 0.5, 0.7, 0.9)
POINT_DIAMETER = 15
POINT_COLORS = {
    "W": (255, 255, 255),
    "R": (255, 0, 0),
    "G": (0, 255, 0),
    "B": (0, 0, 255),
}


def add_fiducials(image: Image.Image, level: int = 112) -> None:
    draw = ImageDraw.Draw(image)
    color = (level, level, level)
    margin = 28
    size = 76
    width = 10

    draw.rectangle(
        (margin, margin, margin + size, margin + size), outline=color, width=width
    )
    draw.ellipse(
        (WIDTH - margin - size, margin, WIDTH - margin, margin + size),
        outline=color,
        width=width,
    )
    draw.polygon(
        [
            (margin, HEIGHT - margin),
            (margin + size, HEIGHT - margin),
            (margin, HEIGHT - margin - size),
        ],
        fill=color,
    )
    draw.rectangle(
        (
            WIDTH - margin - size,
            HEIGHT - margin - size,
            WIDTH - margin,
            HEIGHT - margin,
        ),
        fill=color,
    )
    inset = 18
    draw.rectangle(
        (
            WIDTH - margin - size + inset,
            HEIGHT - margin - size + inset,
            WIDTH - margin - inset,
            HEIGHT - margin - inset,
        ),
        fill=(0, 0, 0),
    )


def joint_response_chart() -> Image.Image:
    image = solid((12, 12, 12))
    draw = ImageDraw.Draw(image)
    add_fiducials(image)

    gray_values = [round(index * 255 / 15) for index in range(16)]
    columns = 8
    margin_x = 145
    gap = 12
    patch_width = (WIDTH - 2 * margin_x - (columns - 1) * gap) // columns
    patch_height = 245
    start_y = 125
    for index, value in enumerate(gray_values):
        row, column = divmod(index, columns)
        x0 = margin_x + column * (patch_width + gap)
        y0 = start_y + row * (patch_height + gap)
        draw.rectangle(
            (x0, y0, x0 + patch_width - 1, y0 + patch_height - 1),
            fill=(value, value, value),
        )

    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (64, 64, 64),
        (128, 128, 128),
        (192, 192, 192),
        (255, 255, 255),
    ]
    color_gap = 14
    color_width = (WIDTH - 2 * margin_x - (len(colors) - 1) * color_gap) // len(
        colors
    )
    color_y0 = 900
    color_y1 = 1435
    for index, color in enumerate(colors):
        x0 = margin_x + index * (color_width + color_gap)
        draw.rectangle((x0, color_y0, x0 + color_width - 1, color_y1), fill=color)
    return image


def combined_point_grid(color: tuple[int, int, int]) -> Image.Image:
    image = solid((0, 0, 0))
    add_fiducials(image, level=80)
    draw = ImageDraw.Draw(image)
    radius = POINT_DIAMETER // 2
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


def field_point_edge_chart() -> Image.Image:
    image = solid((0, 0, 0))
    add_fiducials(image)
    draw = ImageDraw.Draw(image)
    patch_width = 235
    patch_height = 255

    for row in range(3):
        for column in range(3):
            center_x = (2 * column + 1) * WIDTH // 6
            center_y = (2 * row + 1) * HEIGHT // 6
            point_radius = POINT_DIAMETER // 2
            point_x = center_x - 152
            draw.rectangle(
                (
                    point_x - point_radius,
                    center_y - point_radius,
                    point_x + point_radius,
                    center_y + point_radius,
                ),
                fill=(255, 255, 255),
            )

            x0 = center_x - 30
            x1 = x0 + patch_width
            y0 = center_y - patch_height // 2
            y1 = y0 + patch_height
            orientation = (row + column) % 3
            if orientation == 0:
                draw.polygon(
                    [(x0 + 112, y0), (x1, y0), (x1, y1), (x0 + 124, y1)],
                    fill=(255, 255, 255),
                )
            elif orientation == 1:
                draw.polygon(
                    [(x0, y0 + 136), (x1, y0 + 119), (x1, y1), (x0, y1)],
                    fill=(255, 255, 255),
                )
            else:
                draw.polygon(
                    [(x0, y1), (x1, y0), (x1, y1)], fill=(255, 255, 255)
                )
    return image


def random_text_chart(index: int) -> Image.Image:
    rng = random.Random(20260721 + index)
    dark = index % 2 == 0
    background = (18, 18, 18) if dark else (242, 242, 242)
    foreground = (245, 245, 245) if dark else (8, 8, 8)
    image = solid(background)
    add_fiducials(image, level=104 if dark else 152)
    draw = ImageDraw.Draw(image)
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ"
    lowercase = "abcdefghijkmnopqrstuvwxyz"
    top = 115
    rows = 11
    row_height = (HEIGHT - 2 * top) // rows

    for row in range(rows):
        font_size = rng.choice((18, 22, 26, 30, 36, 42, 50, 60, 72))
        font = find_font(font_size)
        digits = "".join(rng.choice("0123456789") for _ in range(12))
        upper = "".join(rng.choice(alphabet) for _ in range(10))
        lower = "".join(rng.choice(lowercase) for _ in range(10))
        text = f"C{index:02d}-{row:02d}  {digits}  {upper}  {lower}"
        box = draw.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = max(125, (WIDTH - text_width) // 2)
        y = top + row * row_height + (row_height - text_height) // 2
        draw.text((x, y), text, fill=foreground, font=font)
    return image


def build_patterns() -> list[tuple[str, Image.Image]]:
    patterns = [("00_joint_response.png", joint_response_chart())]
    for sequence, (color_name, color) in enumerate(POINT_COLORS.items(), start=1):
        patterns.append(
            (
                f"{sequence:02d}_pointgrid_{color_name}_5x5_15px.png",
                combined_point_grid(color),
            )
        )

    patterns.append(("05_field_point_edge.png", field_point_edge_chart()))
    for index in range(20):
        polarity = "dark" if index % 2 == 0 else "light"
        patterns.append(
            (f"{6 + index:02d}_text_{index:02d}_{polarity}.png", random_text_chart(index))
        )
    return patterns


def main() -> None:
    patterns = build_patterns()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for old_image in OUTPUT_DIR.glob("*.png"):
        old_image.unlink()
    for filename, image in patterns:
        image.save(OUTPUT_DIR / filename, format="PNG", compress_level=9)
    print(f"generated {len(patterns)} PNG files in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
