"""Bridge to a versioned native PRA-CM package without changing flow semantics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contract import PairTask
from .base import AdapterResult


class OursAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        project_root = Path(str(config["project_root"])).resolve()
        package = str(config.get("package", "registration_v3_pracm"))
        native = work_dir / str(config.get("native_output_name", f"{package}.flow.npz"))
        qa = work_dir / f"{package}.qa.json"
        command = [
            str(config.get("python", sys.executable)), "-m", f"{package}.scripts.infer",
            "--config", str(Path(str(config["model_config"])).resolve()),
            "--checkpoint", str(Path(str(config["checkpoint"])).resolve()),
            "--fixed", str(task.fixed.path), "--moving", str(task.moving.path),
            "--fixed-phase", task.fixed.phase, "--moving-phase", task.moving.phase,
            "--output-flow", str(native), "--output-qa", str(qa),
            "--device", str(config.get("device", "cuda")),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "code") + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, check=True, cwd=str(project_root), env=env)
        with np.load(native, allow_pickle=False) as archive:
            flow = np.asarray(archive["flow_dzyx_voxels"], dtype=np.float32)
            upstream_version = str(archive["version"].item())
        return AdapterResult(
            flow,
            {
                "implementation": f"local {package} native 3-D inference",
                "upstream_version": upstream_version,
                "command": command,
            },
        )
