"""Official Locor CLI bridge with canonical displacement conversion."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from ..contract import PairTask
from ..flow import load_native_xyz_last_displacement
from ..io import load_pair_on_fixed_grid, save_nifti
from .base import AdapterResult


class LocorAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        repo = Path(str(config["repo"])).resolve()
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        fixed_path = work_dir / "fixed.nii.gz"
        moving_path = work_dir / "moving_on_fixed.nii.gz"
        native_flow = work_dir / "locor_reference_displacement.nii.gz"
        save_nifti(fixed.data_xyz, fixed.affine, fixed_path)
        save_nifti(moving.data_xyz, fixed.affine, moving_path)
        command = [
            str(config.get("python", sys.executable)), "-m", "locor", str(fixed_path), str(moving_path),
            "--displacement-field-reference", str(native_flow),
            "--device", str(config.get("device", "cuda:0")),
            "--dtype", str(config.get("dtype", "float32")),
        ]
        if config.get("config"):
            command.extend(["--config", str(Path(str(config["config"])).resolve())])
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo / "src") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, check=True, cwd=str(repo), env=env)
        flow = load_native_xyz_last_displacement(native_flow, fixed.affine)
        return AdapterResult(flow, {"implementation": "official locor CLI", "command": command})

