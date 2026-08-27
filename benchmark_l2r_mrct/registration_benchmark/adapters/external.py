"""Strict subprocess bridge for released methods with repository-specific training CLIs.

This is intentionally not a permissive "guess the flow" wrapper. A method config
must declare both its exact command and native output convention. The bridge
rejects unknown formats and never exposes label paths to the subprocess.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from ..contract import PairTask
from ..flow import load_flow, load_native_xyz_last_displacement, validate_flow, xyz_last_to_dzyx
from ..io import load_pair_on_fixed_grid, save_nifti
from .base import AdapterResult


class ExternalCommandAdapter:
    def __init__(self, method_name: str = "external") -> None:
        self.method_name = method_name

    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        fixed_path = work_dir / "fixed.nii.gz"
        moving_path = work_dir / "moving_on_fixed.nii.gz"
        output_name = Path(str(config.get("native_output_name", "native_flow.npz")))
        if output_name.is_absolute() or len(output_name.parts) != 1 or output_name.name in {".", ".."}:
            raise ValueError(f"Unsafe native_output_name: {output_name}")
        output_path = work_dir / output_name
        save_nifti(fixed.data_xyz, fixed.affine, fixed_path)
        save_nifti(moving.data_xyz, fixed.affine, moving_path)
        context = {
            "python": str(config.get("python", sys.executable)),
            "repo": str(Path(str(config.get("repo", "."))).resolve()),
            "fixed": str(fixed_path),
            "moving": str(moving_path),
            "output": str(output_path),
            "checkpoint": str(Path(str(config.get("checkpoint", ""))).resolve()) if config.get("checkpoint") else "",
            "pair_id": task.pair_id,
            "fixed_phase": task.fixed.phase,
            "moving_phase": task.moving.phase,
        }
        if config.get("checkpoint") and not Path(context["checkpoint"]).is_file():
            raise FileNotFoundError(
                f"{self.method_name}: checkpoint missing: {context['checkpoint']}. "
                "Set the method-specific REGBENCH_*_CHECKPOINT environment variable."
            )
        template = config.get("infer_command")
        if not isinstance(template, list) or not template:
            raise ValueError(f"{self.method_name}: infer_command must be a non-empty YAML list.")
        command = [str(token).format_map(context) for token in template]
        env = os.environ.copy()
        benchmark_root = Path(__file__).resolve().parents[2]
        env["PYTHONPATH"] = str(benchmark_root) + os.pathsep + env.get("PYTHONPATH", "")
        for key, value in dict(config.get("environment", {})).items():
            env[str(key)] = str(value).format_map(context)
        subprocess.run(command, check=True, cwd=context["repo"], env=env)
        if not output_path.is_file():
            raise FileNotFoundError(f"{self.method_name} did not create {output_path}.")
        native_format = str(config.get("native_flow_format", "")).lower()
        if native_format == "canonical_npz":
            flow, _, _ = load_flow(output_path)
        elif native_format == "xyz_last_voxel_nifti":
            flow = load_native_xyz_last_displacement(output_path, fixed.affine)
        elif native_format == "xyz_last_voxel_npy":
            flow = xyz_last_to_dzyx(np.load(output_path, allow_pickle=False))
        elif native_format == "dzyx_voxel_npy":
            flow = validate_flow(np.load(output_path, allow_pickle=False), fixed.shape_xyz)
        else:
            raise ValueError(f"{self.method_name}: unsupported native_flow_format {native_format!r}.")
        return AdapterResult(
            flow,
            {"implementation": f"external official-code bridge: {self.method_name}", "command": command, "native_flow_format": native_format},
        )
