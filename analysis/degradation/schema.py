from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from PIL import Image


ALLOWED_ROLES = frozenset(
    {
        "background",
        "photometry",
        "geometry",
        "psf",
        "mtf",
        "noise",
        "validation",
        "external_validation",
    }
)
_BURST_INDEX = re.compile(r"TIMEBURST(\d+)", re.IGNORECASE)


class ManifestError(ValueError):
    """Raised when a degradation calibration manifest is invalid."""


@dataclass(frozen=True)
class CaptureGroup:
    id: str
    domain: str
    path: str
    roles: frozenset[str]
    source: Path
    frames: tuple[Path, ...]
    roi_xyxy: tuple[int, int, int, int]
    features: Mapping[str, Any]


@dataclass(frozen=True)
class CalibrationManifest:
    schema_version: str
    workspace_root: Path
    groups: tuple[CaptureGroup, ...]
    domains: tuple[str, ...]


def _natural_frame_key(path: Path) -> tuple[int, str]:
    match = _BURST_INDEX.search(path.stem)
    return (int(match.group(1)) if match else 0, path.name)


def _require_string(payload: Mapping[str, Any], key: str, group_id: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"group {group_id!r} requires nonempty {key!r}")
    return value


def _resolve_frames(root: Path, patterns: object, group_id: str) -> tuple[Path, ...]:
    if not isinstance(patterns, list) or not patterns or not all(
        isinstance(item, str) and item for item in patterns
    ):
        raise ManifestError(f"group {group_id!r} requires a nonempty captures list")
    frames: list[Path] = []
    for pattern in patterns:
        if any(character in pattern for character in "*?["):
            frames.extend(root.glob(pattern))
        else:
            frames.append(root / pattern)
    frames = sorted({frame.resolve() for frame in frames}, key=_natural_frame_key)
    if not frames:
        raise ManifestError(f"group {group_id!r} captures matched no files")
    missing = [frame for frame in frames if not frame.is_file()]
    if missing:
        raise ManifestError(f"group {group_id!r} capture is missing: {missing[0]}")
    return tuple(frames)


def _parse_roi(payload: object, group_id: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(payload, list)
        or len(payload) != 4
        or not all(isinstance(value, int) for value in payload)
    ):
        raise ManifestError(f"group {group_id!r} roi_xyxy must contain four integers")
    x0, y0, x1, y1 = payload
    if x0 < 0 or y0 < 0 or x0 >= x1 or y0 >= y1:
        raise ManifestError(f"group {group_id!r} has invalid roi_xyxy")
    return x0, y0, x1, y1


def _validate_images(
    source: Path,
    frames: tuple[Path, ...],
    roi: tuple[int, int, int, int],
    group_id: str,
) -> None:
    if not source.is_file():
        raise ManifestError(f"group {group_id!r} source is missing: {source}")
    with Image.open(source) as image:
        if image.mode != "RGB":
            raise ManifestError(f"group {group_id!r} source must be RGB")

    expected_size: tuple[int, int] | None = None
    for frame in frames:
        with Image.open(frame) as image:
            if image.mode != "RGB":
                raise ManifestError(f"group {group_id!r} capture must be RGB: {frame}")
            if expected_size is None:
                expected_size = image.size
            elif image.size != expected_size:
                raise ManifestError(f"group {group_id!r} has mixed capture dimensions")
    assert expected_size is not None
    if roi[2] > expected_size[0] or roi[3] > expected_size[1]:
        raise ManifestError(f"group {group_id!r} roi is outside capture bounds")


def load_manifest(path: Path) -> CalibrationManifest:
    manifest_path = path.resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ManifestError(f"cannot read manifest {manifest_path}: {error}") from error
    if not isinstance(payload, dict):
        raise ManifestError("manifest root must be an object")

    schema_version = payload.get("schema_version")
    if schema_version != "1.0":
        raise ManifestError("manifest schema_version must be '1.0'")
    root_value = payload.get("workspace_root", ".")
    if not isinstance(root_value, str) or not root_value:
        raise ManifestError("workspace_root must be a nonempty string")
    workspace_root = (manifest_path.parent / root_value).resolve()

    raw_groups = payload.get("groups")
    if not isinstance(raw_groups, list) or not raw_groups:
        raise ManifestError("manifest groups must be a nonempty list")
    groups: list[CaptureGroup] = []
    seen_ids: set[str] = set()
    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ManifestError("every group must be an object")
        group_id = _require_string(raw_group, "id", "<unknown>")
        if group_id in seen_ids:
            raise ManifestError(f"duplicate group id {group_id!r}")
        seen_ids.add(group_id)
        domain = _require_string(raw_group, "domain", group_id)
        path_name = _require_string(raw_group, "path", group_id)

        raw_roles = raw_group.get("roles")
        if not isinstance(raw_roles, list) or not raw_roles or not all(
            isinstance(role, str) for role in raw_roles
        ):
            raise ManifestError(f"group {group_id!r} requires a nonempty roles list")
        roles = frozenset(raw_roles)
        unknown_roles = roles - ALLOWED_ROLES
        if unknown_roles:
            raise ManifestError(
                f"group {group_id!r} has unknown roles: {sorted(unknown_roles)}"
            )

        source_value = _require_string(raw_group, "source", group_id)
        source = (workspace_root / source_value).resolve()
        frames = _resolve_frames(workspace_root, raw_group.get("captures"), group_id)
        roi = _parse_roi(raw_group.get("roi_xyxy"), group_id)
        features = raw_group.get("features", {})
        if not isinstance(features, dict):
            raise ManifestError(f"group {group_id!r} features must be an object")
        _validate_images(source, frames, roi, group_id)
        groups.append(
            CaptureGroup(
                id=group_id,
                domain=domain,
                path=path_name,
                roles=roles,
                source=source,
                frames=frames,
                roi_xyxy=roi,
                features=dict(features),
            )
        )

    return CalibrationManifest(
        schema_version=schema_version,
        workspace_root=workspace_root,
        groups=tuple(groups),
        domains=tuple(sorted({group.domain for group in groups})),
    )
