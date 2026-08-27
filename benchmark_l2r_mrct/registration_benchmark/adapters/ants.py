"""ANTs SyN+MI adapter matching the Mok paper's no-affine command."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contract import PairTask
from ..flow import xyz_last_to_dzyx
from ..io import load_pair_on_fixed_grid
from .base import AdapterResult


def _geometry(affine: np.ndarray):
    matrix = affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    direction = matrix / spacing[None, :]
    return tuple(spacing.tolist()), tuple(affine[:3, 3].tolist()), direction


class ANTsAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        try:
            import ants
        except ImportError as exc:
            raise RuntimeError("ANTs adapter requires the pinned ANTsPy environment.") from exc
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        spacing, origin, direction = _geometry(fixed.affine)
        fixed_ants = ants.from_numpy(fixed.data_xyz, spacing=spacing, origin=origin, direction=direction)
        moving_ants = ants.from_numpy(moving.data_xyz, spacing=spacing, origin=origin, direction=direction)
        reg_iterations = tuple(int(v) for v in config.get("iterations", [200, 100, 50]))
        result = ants.registration(
            fixed=fixed_ants,
            moving=moving_ants,
            type_of_transform=str(config.get("type_of_transform", "SyNOnly")),
            syn_metric=str(config.get("metric", "mattes")),
            syn_sampling=int(config.get("metric_bins", 32)),
            grad_step=float(config.get("grad_step", 0.25)),
            flow_sigma=float(config.get("flow_sigma", 9.0)),
            total_sigma=float(config.get("total_sigma", 0.2)),
            reg_iterations=reg_iterations,
            initial_transform=str(config.get("initial_transform", "Identity")),
            verbose=bool(config.get("verbose", False)),
        )
        warp_paths = [Path(value) for value in result["fwdtransforms"] if str(value).lower().endswith((".nii", ".nii.gz"))]
        if len(warp_paths) != 1:
            raise RuntimeError(f"Expected one SyN displacement image, got {result['fwdtransforms']}")
        physical_xyz_last = np.asarray(ants.image_read(str(warp_paths[0])).numpy(), dtype=np.float32)
        if physical_xyz_last.shape != (*fixed.shape_xyz, 3):
            raise ValueError(f"Unexpected ANTs warp shape: {physical_xyz_last.shape}")
        linear = np.asarray(fixed.affine[:3, :3], dtype=np.float64)
        native = np.einsum("ij,...j->...i", np.linalg.inv(linear), physical_xyz_last).astype(np.float32)
        return AdapterResult(
            xyz_last_to_dzyx(native),
            {
                "implementation": "ANTsPy ants.registration/apply_transforms",
                "type_of_transform": str(config.get("type_of_transform", "SyNOnly")),
                "metric": str(config.get("metric", "mattes")),
                "affine_iterations": 0,
                "initial_transform": str(config.get("initial_transform", "Identity")),
                "native_warp": "ANTs physical displacement converted through inverse fixed affine",
                "iterations": list(reg_iterations),
            },
        )
