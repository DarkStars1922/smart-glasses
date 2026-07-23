from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "images" / "generate_calibration_v2_followup.py"


def load_generator():
    assert SCRIPT.exists(), "v2 follow-up generator is missing"
    spec = importlib.util.spec_from_file_location("generate_calibration_v2_followup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_followup_contains_locator_and_two_permuted_response_charts() -> None:
    module = load_generator()
    patterns, manifest = module.build_patterns()

    assert [name for name, _ in patterns] == [
        "00_inset_asymmetric_locator.png",
        "01_permuted_response_A.png",
        "02_permuted_response_B.png",
        "03_pointgrid_G_5x5_31px.png",
        "04_pointgrid_B_5x5_31px.png",
    ]
    assert all(image.size == (2500, 1600) for _, image in patterns)
    assert len(manifest["response_A"]) == 56
    assert len(manifest["response_B"]) == 56


def test_large_color_points_have_known_31_pixel_support() -> None:
    module = load_generator()
    patterns, manifest = module.build_patterns()
    images = dict(patterns)

    assert manifest["large_point_diameter_px"] == 31
    for name, channel in (
        ("03_pointgrid_G_5x5_31px.png", 1),
        ("04_pointgrid_B_5x5_31px.png", 2),
    ):
        image = images[name]
        center_x = round(module.GRID_FRACTIONS[2] * (module.WIDTH - 1))
        center_y = round(module.GRID_FRACTIONS[2] * (module.HEIGHT - 1))
        crop = image.crop((center_x - 15, center_y - 15, center_x + 16, center_y + 16))
        pixels = crop.load()
        assert pixels[15, 15][channel] == 255
        assert sum(pixel[channel] == 255 for pixel in crop.getdata()) == 31 * 31


def test_response_levels_repeat_and_swap_at_every_position() -> None:
    module = load_generator()
    _, manifest = module.build_patterns()
    first = manifest["response_A"]
    second = manifest["response_B"]

    first_ids = [patch["patch_id"] for patch in first]
    second_ids = [patch["patch_id"] for patch in second]
    assert set(first_ids) == set(module.PATCH_IDS)
    assert all(first_ids.count(patch_id) == 4 for patch_id in module.PATCH_IDS)
    assert all(second_ids.count(patch_id) == 4 for patch_id in module.PATCH_IDS)
    assert all(a != b for a, b in zip(first_ids, second_ids))


def test_locator_marks_are_inside_the_safe_ten_percent_boundary() -> None:
    module = load_generator()
    patterns, _ = module.build_patterns()
    locator = dict(patterns)["00_inset_asymmetric_locator.png"]
    safe_box = (
        round(0.1 * module.WIDTH),
        round(0.1 * module.HEIGHT),
        round(0.9 * module.WIDTH),
        round(0.9 * module.HEIGHT),
    )

    assert locator.crop(safe_box).getbbox() is not None
    assert locator.crop((0, 0, safe_box[0], module.HEIGHT)).getbbox() is None
    assert locator.crop((safe_box[2], 0, module.WIDTH, module.HEIGHT)).getbbox() is None


def test_readme_lists_the_28_single_frame_followup_batch() -> None:
    readme = (
        ROOT / "images" / "origin" / "calibration_v2_followup" / "README.md"
    ).read_text(encoding="utf-8")

    assert "共 28 张普通单拍" in readme
    assert "EV=-0.7" in readme
    assert "S、ISO、WB、F：自动" in readme
    assert "长焦自适应：开启" in readme
    assert "01_pointgrid_W_5x5_15px.png" in readme
