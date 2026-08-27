from pathlib import Path
from typing import Any, Mapping

from ..contract import PairTask
from ..flow import zero_flow
from ..io import load_scalar
from .base import AdapterResult


class IdentityAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        fixed = load_scalar(task.fixed.path)
        return AdapterResult(zero_flow(fixed.shape_xyz), {"implementation": "internal_exact_identity"})

