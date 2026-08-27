"""Label-free bridge to the official DINO-Reg pair inference class."""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..contract import PairTask
from ..flow import xyz_last_to_dzyx
from ..io import load_pair_on_fixed_grid
from ..io import robust_unit_interval
from ..provenance import sha256_file
from .base import AdapterResult


class DINORegAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        repo = Path(str(config["repo"])).resolve()
        if not (repo / "inference_l2rmrct.py").is_file():
            raise FileNotFoundError(f"DINO-Reg repository missing: {repo}")
        previous_cwd = Path.cwd()
        previous_xformers = os.environ.get("XFORMERS_DISABLED")
        if bool(config.get("disable_xformers", True)):
            os.environ["XFORMERS_DISABLED"] = "1"
        sys.path.insert(0, str(repo))
        try:
            os.chdir(repo)
            import inference_l2rmrct as upstream
            upstream.save_feature = False
            upstream.output_dir_0 = str(work_dir)
            upstream.eigenvalue_array = []
            upstream.configs = {
                "smooth_weight": float(config.get("smooth_weight", 2)),
                "lr": float(config.get("lr", 3)),
                "num_iter": int(config.get("num_iter", 1000)),
                "fm_downsample": int(config.get("fm_downsample", 1)),
                "feature_size": tuple(int(v) for v in config.get("feature_size", [112, 96])),
                "useSavedPCA": False,
                "DINOReg_useMask": bool(config.get("use_mask", True)),
                "window": True,
                "convex": False,
                "ztrans": False,
                "iter_smooth_num": int(config.get("iter_smooth_num", 5)),
                "iter_smooth_kernel": int(config.get("iter_smooth_kernel", 7)),
                "final_upsample": int(config.get("final_upsample", 1)),
                "mask": "slice fill stack",
            }
            fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
            model = upstream.dinoReg(
                device_id=int(config.get("device_id", 0)),
                lr=upstream.configs["lr"],
                smooth_weight=upstream.configs["smooth_weight"],
                num_iter=upstream.configs["num_iter"],
                feat_size=upstream.configs["feature_size"],
            )
            preprocessing = "official_l2r_mrct"
            if task.task_domain == "multiphase_ct":
                preprocessing = "symmetric_ct_robust_normalization"
                def symmetric_ct_preprocess(instance, moving_array, fixed_array):
                    # DINO-Reg's released L2R path hard-codes MR normalization for
                    # fixed and CT windowing for moving.  That is invalid for a
                    # CT-to-CT phase task, so only this modality preprocessing is
                    # made symmetric; the released DINO/PCA/ConvexAdam path stays intact.
                    fixed_std = fixed_array.std(axis=(0, 1))
                    moving_std = moving_array.std(axis=(0, 1))
                    keep = np.flatnonzero((fixed_std > 1e-6) | (moving_std > 1e-6))
                    if keep.size == 0:
                        keep = np.arange(fixed_array.shape[2])
                    fixed_value = robust_unit_interval(fixed_array[:, :, keep])
                    moving_value = robust_unit_interval(moving_array[:, :, keep])
                    fixed_mask = (fixed_value > 0.01).astype(np.float32)
                    moving_mask = (moving_value > 0.01).astype(np.float32)
                    if fixed_mask.sum() < 64:
                        fixed_mask[...] = 1.0
                    if moving_mask.sum() < 64:
                        moving_mask[...] = 1.0
                    return moving_value, fixed_value, keep, fixed_value.shape, fixed_mask, moving_mask
                model.case_preprocess = types.MethodType(symmetric_ct_preprocess, model)
            native = model.case_inference(
                moving.data_xyz,
                fixed.data_xyz,
                fixed.shape_xyz,
                fixed.affine,
                case_id=task.pair_id,
                disp_init=None,
                grid_sp_adam=upstream.configs["fm_downsample"],
                DINOReg_useMask=upstream.configs["DINOReg_useMask"],
            )
        finally:
            os.chdir(previous_cwd)
            sys.path.pop(0)
            if previous_xformers is None:
                os.environ.pop("XFORMERS_DISABLED", None)
            else:
                os.environ["XFORMERS_DISABLED"] = previous_xformers
        weight_path = repo / "models/dinov2/dinov2_vitl14_reg4_pretrain.pth"
        return AdapterResult(
            xyz_last_to_dzyx(native),
            {
                "implementation": "official DINO-Reg dinoReg.case_inference",
                "labels_loaded": False,
                "preprocessing": preprocessing,
                "foundation_weight": {
                    "path": str(weight_path),
                    "sha256": sha256_file(weight_path) if weight_path.is_file() else None,
                },
                "parameters": dict(upstream.configs),
            },
        )
