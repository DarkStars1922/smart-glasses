from __future__ import annotations

import json
from pathlib import Path


DOCUMENT = Path("docs/Shouldersurfing Smart Glass .md")
PARAMETERS = Path("analysis/results/v1_parameters.json")


def test_document_uses_generated_core_fit_values() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))
    domain_47 = parameters["domains"]["supermacro_47_primary"]["effective_parameters"]
    domain_69 = parameters["domains"]["supermacro_69_primary"]["effective_parameters"]

    expected = [
        f"{domain_47['scale_camera_per_source'][0]:.4f}",
        f"{domain_47['scale_camera_per_source'][1]:.4f}",
        f"{domain_47['blur_fwhm_camera_px'][0]:.2f}",
        f"{domain_47['blur_fwhm_camera_px'][1]:.2f}",
        f"{domain_69['scale_camera_per_source'][0]:.4f}",
        f"{domain_69['scale_camera_per_source'][1]:.4f}",
        f"{domain_69['blur_fwhm_camera_px'][0]:.2f}",
    ]
    assert all(value in document for value in expected)


def test_document_contains_complete_model_and_identifiability_sections() -> None:
    document = DOCUMENT.read_text(encoding="utf-8")

    assert "I_{blur}" not in document
    assert "ψ\\_blur" not in document
    assert "JPEG 压缩 quality ∈" not in document
    assert "\\mathcal W" in document
    assert "\\mathcal D" in document
    assert "\\mathcal J" in document
    assert "当前可识别参数" in document
    assert "仍不可识别的参数" in document
    assert "下一轮必拍数据" in document
    assert "单亮点扫描" in document
    assert "RAW/DNG + JPEG" in document
    assert "特征级多帧融合识别模型" in document
