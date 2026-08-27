"""Run the pinned official DINO-Reg MR<-CT anchor and save canonical V5 flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from ..flow import save_flow, xyz_last_to_dzyx
from ..io import load_pair_on_fixed_grid


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_dino_anchor(
    *,
    fixed: Path,
    moving: Path,
    pair_id: str,
    repo: Path,
    output_flow: Path,
    work_dir: Path,
    device_id: int = 0,
) -> dict:
    """Execute the exact B04 anchor without labels or case-specific tuning."""
    repo = repo.expanduser().resolve()
    fixed = fixed.expanduser().resolve()
    moving = moving.expanduser().resolve()
    output_flow = output_flow.expanduser().resolve()
    work_dir = work_dir.expanduser().resolve()
    if not (repo / "inference_l2rmrct.py").is_file():
        raise FileNotFoundError(f"Official DINO-Reg repository missing: {repo}")
    if output_flow.exists():
        raise FileExistsError(output_flow)
    fixed_volume, moving_volume = load_pair_on_fixed_grid(fixed, moving)
    previous_cwd = Path.cwd()
    previous_xformers = os.environ.get("XFORMERS_DISABLED")
    os.environ["XFORMERS_DISABLED"] = "1"
    sys.path.insert(0, str(repo))
    work_dir.mkdir(parents=True, exist_ok=True)
    parameters = {
        "smooth_weight": 2.0,
        "lr": 3.0,
        "num_iter": 1000,
        "fm_downsample": 1,
        "feature_size": (112, 96),
        "useSavedPCA": False,
        "DINOReg_useMask": True,
        "window": True,
        "convex": False,
        "ztrans": False,
        "iter_smooth_num": 5,
        "iter_smooth_kernel": 7,
        "final_upsample": 1,
        "mask": "slice fill stack",
    }
    try:
        os.chdir(repo)
        import inference_l2rmrct as upstream

        upstream.save_feature = False
        upstream.output_dir_0 = str(work_dir)
        upstream.eigenvalue_array = []
        upstream.configs = dict(parameters)
        model = upstream.dinoReg(
            device_id=int(device_id),
            lr=parameters["lr"],
            smooth_weight=parameters["smooth_weight"],
            num_iter=parameters["num_iter"],
            feat_size=parameters["feature_size"],
        )
        native = model.case_inference(
            moving_volume.data_xyz,
            fixed_volume.data_xyz,
            fixed_volume.shape_xyz,
            fixed_volume.affine,
            case_id=pair_id,
            disp_init=None,
            grid_sp_adam=parameters["fm_downsample"],
            DINOReg_useMask=parameters["DINOReg_useMask"],
        )
    finally:
        os.chdir(previous_cwd)
        sys.path.pop(0)
        if previous_xformers is None:
            os.environ.pop("XFORMERS_DISABLED", None)
        else:
            os.environ["XFORMERS_DISABLED"] = previous_xformers
    weight = repo / "models/dinov2/dinov2_vitl14_reg4_pretrain.pth"
    metadata = {
        "schema": "registration_v5_dino_anchor_qa_v1",
        "pair_id": pair_id,
        "labels_used": False,
        "implementation": "official DINO-Reg dinoReg.case_inference",
        "preprocessing": "official_l2r_mrct",
        "parameters": parameters,
        "foundation_weight": {
            "path": str(weight),
            "sha256": _sha256(weight) if weight.is_file() else None,
        },
    }
    save_flow(
        output_flow,
        xyz_last_to_dzyx(np.asarray(native)),
        fixed_volume.affine,
        metadata,
    )
    qa_path = output_flow.with_suffix(".qa.json")
    qa_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"flow": str(output_flow), "qa": str(qa_path), **metadata}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed", type=Path, required=True, help="Fixed MR NIfTI")
    parser.add_argument("--moving", type=Path, required=True, help="Moving CT NIfTI")
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-flow", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device-id", type=int, default=0)
    args = parser.parse_args(argv)
    result = run_dino_anchor(**vars(args))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
