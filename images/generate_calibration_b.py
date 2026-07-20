from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


WIDTH = 2515
HEIGHT = 1491
SIZE = (WIDTH, HEIGHT)
OUTPUT_DIR = Path(__file__).parent / "origin" / "calibration_B"


def solid(rgb: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", SIZE, rgb)


def color_bars() -> Image.Image:
    image = solid((0, 0, 0))
    draw = ImageDraw.Draw(image)
    colors = [
        (255, 255, 255),
        (255, 255, 0),
        (0, 255, 255),
        (0, 255, 0),
        (255, 0, 255),
        (255, 0, 0),
        (0, 0, 255),
        (0, 0, 0),
    ]
    for index, color in enumerate(colors):
        draw.rectangle(
            (index * WIDTH // 8, 0, (index + 1) * WIDTH // 8 - 1, HEIGHT - 1),
            fill=color,
        )
    return image


def checkerboard() -> Image.Image:
    image = solid((0, 0, 0))
    draw = ImageDraw.Draw(image)
    cell = 120
    for y in range(0, HEIGHT, cell):
        for x in range(0, WIDTH, cell):
            if (x // cell + y // cell) % 2 == 0:
                draw.rectangle(
                    (x, y, min(x + cell - 1, WIDTH - 1), min(y + cell - 1, HEIGHT - 1)),
                    fill=(255, 255, 255),
                )
    return image


def asymmetric_grid() -> Image.Image:
    image = solid((0, 0, 0))
    draw = ImageDraw.Draw(image)
    spacing = 150
    radius = 9
    for y in range(145, HEIGHT - 120, spacing):
        for x in range(145, WIDTH - 120, spacing):
            draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(255, 255, 255),
            )

    draw.rectangle((35, 35, 115, 115), outline=(255, 255, 255), width=12)
    draw.ellipse(
        (WIDTH - 125, 35, WIDTH - 35, 125),
        outline=(255, 255, 255),
        width=12,
    )
    draw.polygon(
        [(45, HEIGHT - 45), (125, HEIGHT - 45), (45, HEIGHT - 125)],
        fill=(255, 255, 255),
    )
    draw.rectangle(
        (WIDTH - 125, HEIGHT - 125, WIDTH - 35, HEIGHT - 35),
        fill=(255, 255, 255),
    )
    draw.rectangle(
        (WIDTH - 105, HEIGHT - 105, WIDTH - 55, HEIGHT - 55),
        fill=(0, 0, 0),
    )
    return image


def point_grid(diameter: int) -> Image.Image:
    image = solid((0, 0, 0))
    draw = ImageDraw.Draw(image)
    half = diameter // 2
    for y in range(100, HEIGHT - 80, 150):
        for x in range(100, WIDTH - 80, 150):
            draw.rectangle(
                (x - half, y - half, x - half + diameter - 1, y - half + diameter - 1),
                fill=(255, 255, 255),
            )
    return image


def slanted_edges() -> Image.Image:
    image = solid((0, 0, 0))
    draw = ImageDraw.Draw(image)
    patch_width = 620
    patch_height = 350
    for row in range(3):
        for column in range(3):
            center_x = (2 * column + 1) * WIDTH // 6
            center_y = (2 * row + 1) * HEIGHT // 6
            x0 = center_x - patch_width // 2
            x1 = center_x + patch_width // 2
            y0 = center_y - patch_height // 2
            y1 = center_y + patch_height // 2
            if (row + column) % 2 == 0:
                draw.polygon(
                    [(center_x - 16, y0), (x1, y0), (x1, y1), (center_x + 16, y1)],
                    fill=(255, 255, 255),
                )
            else:
                draw.polygon(
                    [(x0, center_y + 26), (x1, center_y - 26), (x1, y1), (x0, y1)],
                    fill=(255, 255, 255),
                )
    return image


def resolution_charts() -> tuple[Image.Image, Image.Image, Image.Image]:
    widths = [1, 2, 4, 8, 16, 32]
    y_index, x_index = np.indices((HEIGHT, WIDTH), dtype=np.int32)
    horizontal = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    vertical = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    diagonal = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

    for index, line_width in enumerate(widths):
        x0 = index * WIDTH // len(widths)
        x1 = (index + 1) * WIDTH // len(widths)
        horizontal[:, x0:x1] = (
            (y_index[:, x0:x1] // line_width) % 2 * 255
        ).astype(np.uint8)
        diagonal[:, x0:x1] = (
            ((x_index[:, x0:x1] + y_index[:, x0:x1]) // line_width) % 2 * 255
        ).astype(np.uint8)

        y0 = index * HEIGHT // len(widths)
        y1 = (index + 1) * HEIGHT // len(widths)
        vertical[y0:y1, :] = (
            (x_index[y0:y1, :] // line_width) % 2 * 255
        ).astype(np.uint8)

    return tuple(
        Image.fromarray(array, mode="L").convert("RGB")
        for array in (horizontal, vertical, diagonal)
    )


def siemens_star() -> Image.Image:
    image = solid((128, 128, 128))
    draw = ImageDraw.Draw(image)
    center_x = WIDTH // 2
    center_y = HEIGHT // 2
    radius = min(WIDTH, HEIGHT) // 2 - 70
    sectors = 144
    for index in range(sectors):
        angle0 = 2 * math.pi * index / sectors
        angle1 = 2 * math.pi * (index + 1) / sectors
        color = (255, 255, 255) if index % 2 == 0 else (0, 0, 0)
        point0 = (
            center_x + int(radius * math.cos(angle0)),
            center_y + int(radius * math.sin(angle0)),
        )
        point1 = (
            center_x + int(radius * math.cos(angle1)),
            center_y + int(radius * math.sin(angle1)),
        )
        draw.polygon([(center_x, center_y), point0, point1], fill=color)
    draw.ellipse(
        (center_x - 10, center_y - 10, center_x + 10, center_y + 10),
        fill=(128, 128, 128),
    )
    return image


def spatial_color_patches() -> Image.Image:
    image = solid((16, 16, 16))
    draw = ImageDraw.Draw(image)
    colors = [
        (0, 0, 0),
        (64, 64, 64),
        (128, 128, 128),
        (255, 255, 255),
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),
    ]
    gap = 34
    for row in range(4):
        for column in range(4):
            x0 = column * WIDTH // 4 + gap
            x1 = (column + 1) * WIDTH // 4 - gap - 1
            y0 = row * HEIGHT // 4 + gap
            y1 = (row + 1) * HEIGHT // 4 - gap - 1
            draw.rectangle((x0, y0, x1, y1), fill=colors[row * 4 + column])
    return image


def find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("/mnt/c/Windows/Fonts/msyh.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def text_chart(
    background: tuple[int, int, int], foreground: tuple[int, int, int]
) -> Image.Image:
    image = solid(background)
    draw = ImageDraw.Draw(image)
    rows = [
        (16, "0123456789  ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        (20, "0123456789  abcdefghijklmnopqrstuvwxyz"),
        (24, "ID: 310101199001011234  Account: 6222 0000 1234 5678"),
        (32, "0 1 2 3 4 5 6 7 8 9"),
        (40, "The quick brown fox jumps over 0123456789"),
        (48, "中文字符样例  一二三四  五六七八  九十"),
        (64, "Aa Bb 0123 测试"),
        (80, "0123456789"),
    ]
    top = 95
    step = (HEIGHT - 2 * top) // len(rows)
    for index, (size, text) in enumerate(rows):
        font = find_font(size)
        box = draw.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        x = max(30, (WIDTH - text_width) // 2)
        y = top + index * step + (step - text_height) // 2
        draw.text((x, y), text, font=font, fill=foreground)
    return image


def build_patterns() -> list[tuple[str, Image.Image]]:
    patterns: list[tuple[str, Image.Image]] = []
    solid_specs = [
        ("00_black.png", (0, 0, 0)),
        ("01_white.png", (255, 255, 255)),
        ("02_gray_032.png", (32, 32, 32)),
        ("03_gray_064.png", (64, 64, 64)),
        ("04_gray_096.png", (96, 96, 96)),
        ("05_gray_128.png", (128, 128, 128)),
        ("06_gray_160.png", (160, 160, 160)),
        ("07_gray_192.png", (192, 192, 192)),
        ("08_gray_224.png", (224, 224, 224)),
        ("09_red_128.png", (128, 0, 0)),
        ("10_red_255.png", (255, 0, 0)),
        ("11_green_128.png", (0, 128, 0)),
        ("12_green_255.png", (0, 255, 0)),
        ("13_blue_128.png", (0, 0, 128)),
        ("14_blue_255.png", (0, 0, 255)),
        ("15_cyan_255.png", (0, 255, 255)),
        ("16_magenta_255.png", (255, 0, 255)),
        ("17_yellow_255.png", (255, 255, 0)),
    ]
    patterns.extend((name, solid(rgb)) for name, rgb in solid_specs)

    normal_checkerboard = checkerboard()
    horizontal, vertical, diagonal = resolution_charts()
    patterns.extend(
        [
            ("18_color_bars_full.png", color_bars()),
            ("19_checkerboard.png", normal_checkerboard),
            ("20_checkerboard_inverse.png", ImageChops.invert(normal_checkerboard)),
            ("21_asymmetric_grid.png", asymmetric_grid()),
            ("22_point_grid_3px.png", point_grid(3)),
            ("23_point_grid_7px.png", point_grid(7)),
            ("24_slanted_edges_3x3.png", slanted_edges()),
            ("25_resolution_lines_horizontal.png", horizontal),
            ("26_resolution_lines_vertical.png", vertical),
            ("27_resolution_lines_diagonal.png", diagonal),
            ("28_siemens_star.png", siemens_star()),
            ("29_spatial_color_patches.png", spatial_color_patches()),
            ("30_text_dark.png", text_chart((24, 24, 24), (255, 255, 255))),
            ("31_text_light.png", text_chart((245, 245, 245), (8, 8, 8))),
        ]
    )
    if len(patterns) != 32 or len({name for name, _ in patterns}) != 32:
        raise RuntimeError(f"expected 32 unique images, got {len(patterns)}")
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
