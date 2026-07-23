from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "images" / "generate_calibration_c.py"


def load_generator():
    assert SCRIPT.exists(), "calibration C generator is missing"
    import importlib.util

    spec = importlib.util.spec_from_file_location("generate_calibration_c", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_calibration_c_inventory_matches_model_requirements():
    module = load_generator()
    patterns = module.build_patterns()
    names = [name for name, _ in patterns]

    assert len(names) == 26
    assert len(set(names)) == 26
    assert sum("pointgrid_" in name for name in names) == 4
    assert sum("text_" in name for name in names) == 20
    assert "00_joint_response.png" in names
    assert "01_pointgrid_W_5x5_15px.png" in names
    assert "05_field_point_edge.png" in names


def test_calibration_c_images_match_display_format():
    module = load_generator()

    for _, image in module.build_patterns():
        assert image.size == (2500, 1600)
        assert image.mode == "RGB"


def test_point_target_is_visible_and_has_known_support():
    module = load_generator()
    image = module.combined_point_grid((255, 255, 255))
    center_x = round(module.GRID_FRACTIONS[2] * (module.WIDTH - 1))
    center_y = round(module.GRID_FRACTIONS[2] * (module.HEIGHT - 1))
    half = module.POINT_DIAMETER // 2

    assert module.POINT_DIAMETER == 15
    assert image.crop(
        (center_x - half, center_y - half, center_x + half + 1, center_y + half + 1)
    ).getbbox() == (0, 0, 15, 15)


def test_capture_notes_separate_auto_47mm_and_manual_23mm_domains():
    readme = (ROOT / "images" / "origin" / "calibration_C" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "47 mm 自动域" in readme
    assert "23 mm 手动对照域" in readme
    assert "长焦自适应：开启" in readme
    assert "关闭万物追焦、长焦自适应" not in readme


def test_capture_notes_use_a_small_adaptive_first_batch():
    readme = (ROOT / "images" / "origin" / "calibration_C" / "README.md").read_text(
        encoding="utf-8"
    )

    assert "核心批次：43 张实拍" in readme
    assert "只使用三次 10 张定时连拍" in readme
    assert "核心批次：51 张实拍" not in readme
    assert "核心批次：42 张实拍" not in readme
    assert "核心批次：66 张实拍" not in readme
    assert "全部单点扫描" not in readme
