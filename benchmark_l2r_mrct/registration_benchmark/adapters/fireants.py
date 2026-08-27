"""FireANTs SyN adapter using its physical-coordinate-aware API."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ..contract import PairTask
from ..io import load_pair_on_fixed_grid, save_nifti
from .base import AdapterResult


class FireANTsAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        repo = Path(str(config["repo"])).resolve()
        sys.path.insert(0, str(repo))
        try:
            from fireants.io.image import BatchedImages, Image
            from fireants.registration.syn import SyNRegistration
        finally:
            sys.path.pop(0)
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        fixed_path = work_dir / "fixed.nii.gz"
        moving_path = work_dir / "moving_on_fixed.nii.gz"
        save_nifti(fixed.data_xyz, fixed.affine, fixed_path)
        save_nifti(moving.data_xyz, fixed.affine, moving_path)
        device = torch.device(str(config.get("device", "cuda:0" if torch.cuda.is_available() else "cpu")))
        dtype = getattr(torch, str(config.get("dtype", "float32")))
        fixed_batch = BatchedImages([Image.load_file(str(fixed_path), device=device, dtype=dtype)])
        moving_batch = BatchedImages([Image.load_file(str(moving_path), device=device, dtype=dtype)])
        parameters = {
            "scales": [float(v) for v in config.get("scales", [4, 2, 1])],
            "iterations": [int(v) for v in config.get("iterations", [200, 100, 50])],
            "loss_type": str(config.get("loss_type", "mi")),
            "deformation_type": str(config.get("deformation_type", "compositive")),
            "optimizer": str(config.get("optimizer", "Adam")),
            "optimizer_lr": float(config.get("optimizer_lr", 0.1)),
            "smooth_warp_sigma": float(config.get("smooth_warp_sigma", 0.5)),
            "smooth_grad_sigma": float(config.get("smooth_grad_sigma", 1.0)),
            "progress_bar": bool(config.get("progress_bar", False)),
        }
        registration = SyNRegistration(fixed_images=fixed_batch, moving_images=moving_batch, **parameters)
        registration.optimize()
        normalized = registration.get_warped_coordinates(fixed_batch, moving_batch).detach().cpu().numpy()[0]
        d, h, w = normalized.shape[:3]
        zz, yy, xx = np.meshgrid(
            np.linspace(-1, 1, d, dtype=np.float32),
            np.linspace(-1, 1, h, dtype=np.float32),
            np.linspace(-1, 1, w, dtype=np.float32),
            indexing="ij",
        )
        flow = np.stack([
            (normalized[..., 2] - zz) * max(d - 1, 1) / 2.0,
            (normalized[..., 1] - yy) * max(h - 1, 1) / 2.0,
            (normalized[..., 0] - xx) * max(w - 1, 1) / 2.0,
        ], axis=0).astype(np.float32)
        return AdapterResult(flow, {"implementation": "official FireANTs SyNRegistration", "parameters": parameters})

