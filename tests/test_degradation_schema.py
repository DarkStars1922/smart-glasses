from __future__ import annotations

import json
from pathlib import Path

from PIL import Image
import pytest

from analysis.degradation.schema import ManifestError, load_manifest


def _write_manifest(tmp_path: Path, **group_overrides: object) -> Path:
    Image.new("RGB", (8, 8), (255, 255, 255)).save(tmp_path / "source.png")
    Image.new("RGB", (8, 8), (32, 64, 96)).save(tmp_path / "capture.jpg")
    group: dict[str, object] = {
        "id": "sample",
        "domain": "test_domain",
        "path": "primary",
        "roles": ["psf"],
        "source": "source.png",
        "captures": ["capture.jpg"],
        "roi_xyxy": [0, 0, 8, 8],
        "features": {"point_spacing_px": 4},
    }
    group.update(group_overrides)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "workspace_root": ".",
                "groups": [group],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_real_manifest_is_domain_and_path_driven() -> None:
    manifest = load_manifest(Path("analysis/calibration_b_v1.json"))

    assert set(manifest.domains) == {
        "supermacro_47_primary",
        "supermacro_69_primary",
    }
    assert {group.path for group in manifest.groups} == {"primary_readable"}
    assert sum(len(group.frames) for group in manifest.groups) == 120
    assert all(group.source.is_file() for group in manifest.groups)


def test_loads_minimal_valid_manifest(tmp_path: Path) -> None:
    manifest = load_manifest(_write_manifest(tmp_path))

    assert manifest.schema_version == "1.0"
    assert manifest.domains == ("test_domain",)
    assert manifest.groups[0].roi_xyxy == (0, 0, 8, 8)
    assert manifest.groups[0].features["point_spacing_px"] == 4


def test_rejects_roi_outside_capture(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, roi_xyxy=[0, 0, 20, 20])

    with pytest.raises(ManifestError, match="outside capture bounds"):
        load_manifest(path)


def test_rejects_unknown_role(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path, roles=["invented_role"])

    with pytest.raises(ManifestError, match="unknown roles"):
        load_manifest(path)
