"""Thin adapter around the official ConvexAdam Python implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

from ..contract import PairTask
from ..flow import xyz_last_to_dzyx
from ..io import load_pair_on_fixed_grid
from .base import AdapterResult


class ConvexAdamAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        repo = Path(str(config["repo"])).resolve()
        source = repo / "src"
        if not source.is_dir():
            raise FileNotFoundError(f"ConvexAdam source directory missing: {source}")
        sys.path.insert(0, str(source))
        try:
            from convexAdam.convex_adam_MIND import convex_adam_pt
        finally:
            sys.path.pop(0)
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        kwargs = {
            "mind_r": int(config.get("mind_r", 1)),
            "mind_d": int(config.get("mind_d", 2)),
            "lambda_weight": float(config.get("lambda_weight", 1.25)),
            "grid_sp": int(config.get("grid_sp", 6)),
            "disp_hw": int(config.get("disp_hw", 4)),
            "selected_niter": int(config.get("selected_niter", 80)),
            "grid_sp_adam": int(config.get("grid_sp_adam", 2)),
            "ic": bool(config.get("inverse_consistency", True)),
        }
        native = convex_adam_pt(fixed.data_xyz, moving.data_xyz, **kwargs)
        return AdapterResult(
            xyz_last_to_dzyx(native),
            {"implementation": "official_convexAdam.convex_adam_pt", "parameters": kwargs},
        )

