import importlib.util
from collections import Counter
from pathlib import Path

from PIL import ImageChops


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "images" / "generate_calibration_v3_core.py"


def load_generator():
    spec = importlib.util.spec_from_file_location("generate_calibration_v3_core", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builds_exact_nine_capture_patterns():
    generator = load_generator()
    patterns, _ = generator.build_patterns()

    assert list(patterns) == [
        "00_locator_start.png",
        "01_full_black_start.png",
        "02_permuted_response_A.png",
        "03_permuted_response_B.png",
        "04_permuted_response_C.png",
        "05_red_gray_ramp.png",
        "06_GB_slanted_edges.png",
        "07_full_black_end.png",
        "08_locator_end.png",
    ]
    assert all(image.size == (2500, 1600) for image in patterns.values())
    assert all(image.mode == "RGB" for image in patterns.values())


def test_start_and_end_reference_frames_are_identical():
    generator = load_generator()
    patterns, _ = generator.build_patterns()

    assert ImageChops.difference(
        patterns["00_locator_start.png"], patterns["08_locator_end.png"]
    ).getbbox() is None
    assert ImageChops.difference(
        patterns["01_full_black_start.png"], patterns["07_full_black_end.png"]
    ).getbbox() is None
    assert patterns["01_full_black_start.png"].getextrema() == ((0, 0), (0, 0), (0, 0))


def test_response_charts_have_equal_patch_counts_and_distinct_orders():
    generator = load_generator()
    _, manifest = generator.build_patterns()
    charts = manifest["response_charts"]

    orders = []
    for name in ("A", "B", "C"):
        assignments = charts[name]["assignments"]
        assert Counter(item["patch_id"] for item in assignments) == {
            patch_id: 4 for patch_id in generator.PATCH_IDS
        }
        orders.append([item["patch_id"] for item in assignments])

    assert orders[0] != orders[1]
    assert orders[0] != orders[2]
    assert orders[1] != orders[2]


def test_red_gray_ramp_repeats_each_level_as_designed():
    generator = load_generator()
    _, manifest = generator.build_patterns()
    assignments = manifest["red_gray_ramp"]["assignments"]
    counts = Counter(item["patch_id"] for item in assignments)

    for level in generator.LEVELS:
        assert counts[f"R{level:03d}"] == 4
        assert counts[f"K{level:03d}"] == 3
    assert len(assignments) == 56


def test_gb_chart_balances_colors_and_edge_orientations():
    generator = load_generator()
    patterns, manifest = generator.build_patterns()
    cells = manifest["gb_slanted_edges"]["cells"]

    assert Counter(cell["color_id"] for cell in cells) == {"G": 8, "B": 8}
    for color_id in ("G", "B"):
        assert Counter(
            cell["orientation"] for cell in cells if cell["color_id"] == color_id
        ) == {orientation: 2 for orientation in generator.EDGE_ORIENTATIONS}

    colors = patterns["06_GB_slanted_edges.png"].getcolors(maxcolors=100)
    assert colors is not None
    assert (0, 255, 0) in {color for _, color in colors}
    assert (0, 0, 255) in {color for _, color in colors}
