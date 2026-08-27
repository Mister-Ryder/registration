from __future__ import annotations

import json
from pathlib import Path

from registration_v5 import __version__
from registration_v5.routing.capture import CaptureRouterConfig


def test_capture_radius_is_exactly_24_native_voxels() -> None:
    config = CaptureRouterConfig()
    assert config.residual_grid_spacing_native_voxels == 6
    assert config.residual_displacement_half_width_cells == 4
    assert config.residual_capture_radius_native_voxels == 24


def test_manifest_matches_package_version_and_result() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "VERSION_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == __version__
    assert manifest["imports_from_registration_v4_experiment_directories"] is False
    assert manifest["canonical_v2_public8"]["cases"] == 8
    assert manifest["canonical_v2_public8"]["mean_dice"] == 0.7859497458400165
