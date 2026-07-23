import importlib.util
from collections import Counter
from pathlib import Path

from PIL import ImageChops


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "images" / "generate_calibration_v4_color_operator.py"


def load_generator():
    spec = importlib.util.spec_from_file_location(
        "generate_calibration_v4_color_operator", SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builds_exact_six_color_operator_patterns():
    generator = load_generator()
    patterns, _ = generator.build_patterns()

    assert list(patterns) == [
        "00_full_black_start.png",
        "01_joint_color_train_A.png",
        "02_joint_color_train_B.png",
        "03_joint_color_train_C.png",
        "04_joint_color_holdout.png",
        "05_full_black_end.png",
    ]
    assert all(image.size == (2500, 1600) for image in patterns.values())
    assert all(image.mode == "RGB" for image in patterns.values())


def test_black_endpoint_patterns_are_identical_and_pure_black():
    generator = load_generator()
    patterns, _ = generator.build_patterns()
    start = patterns["00_full_black_start.png"]
    end = patterns["05_full_black_end.png"]

    assert ImageChops.difference(start, end).getbbox() is None
    assert start.getextrema() == ((0, 0), (0, 0), (0, 0))


def test_training_charts_repeat_each_color_and_have_distinct_orders():
    generator = load_generator()
    _, manifest = generator.build_patterns()
    orders = []

    for name in ("A", "B", "C"):
        assignments = manifest["training_charts"][name]["assignments"]
        counts = Counter(row["patch_id"] for row in assignments)
        assert counts["K000"] == 4
        assert counts["K128"] == 2
        assert counts["K255"] == 2
        assert all(counts[patch_id] == 2 for patch_id in generator.TRAINING_COLOR_IDS)
        assert len(assignments) == 56
        orders.append([row["patch_id"] for row in assignments])

    assert len({tuple(order) for order in orders}) == 3


def test_holdout_colors_are_disjoint_from_training_colors():
    generator = load_generator()
    _, manifest = generator.build_patterns()
    assignments = manifest["holdout_chart"]["assignments"]
    counts = Counter(row["patch_id"] for row in assignments)

    assert set(generator.TRAINING_COLORS.values()).isdisjoint(
        set(generator.HOLDOUT_COLORS.values())
    )
    assert counts["K000"] == 4
    assert counts["K128"] == 2
    assert counts["K255"] == 2
    assert all(counts[patch_id] == 2 for patch_id in generator.HOLDOUT_COLOR_IDS)
    assert len(assignments) == 56


def test_training_colors_cover_multiple_field_positions():
    generator = load_generator()
    _, manifest = generator.build_patterns()

    for patch_id in generator.TRAINING_COLOR_IDS:
        rows = []
        columns = []
        for name in ("A", "B", "C"):
            for assignment in manifest["training_charts"][name]["assignments"]:
                if assignment["patch_id"] == patch_id:
                    rows.append(assignment["row"])
                    columns.append(assignment["column"])
        assert max(rows) - min(rows) >= 3
        assert max(columns) - min(columns) >= 4
