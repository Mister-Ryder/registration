"""Full-resolution DSIR plus descriptor ConvexAdam pair inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from registration_benchmark.dns.faithful_v2 import (
    canonical_mrct_modality,
    normalize_mrct_intensity,
)
from registration_benchmark.io import load_pair_on_fixed_grid

from .. import __version__
from ..convex_solver import descriptor_convex_adam
from ..data import raw_foreground_xyz
from ..model import V4FinalModel
from ..protocol import load_protocol
from ..state import canonical_hash, load_checkpoint, source_hash


def _tensor_xyz(data: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(data, dtype=np.float32).transpose(2, 1, 0).copy())[
        None, None
    ].to(device)


def _descriptor_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[name]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed-phase", required=True)
    parser.add_argument("--moving-phase", required=True)
    parser.add_argument("--output-flow", required=True)
    parser.add_argument("--output-qa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    args = parser.parse_args(argv)

    protocol = load_protocol(args.config)
    payload = load_checkpoint(args.checkpoint, map_location="cpu")
    if payload["protocol_sha256"] != canonical_hash(protocol.to_dict()) and not args.allow_config_mismatch:
        raise ValueError("Inference protocol differs from the checkpoint.")
    observed_source = source_hash()
    if payload["source_sha256"] != observed_source and not args.allow_source_mismatch:
        raise ValueError("Inference source differs from the checkpoint.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable.")
    model = V4FinalModel(protocol.descriptor).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)

    fixed_volume, moving_volume = load_pair_on_fixed_grid(
        Path(args.fixed).resolve(), Path(args.moving).resolve()
    )
    if fixed_volume.data_xyz.shape != moving_volume.data_xyz.shape:
        raise RuntimeError("Moving-to-fixed preprocessing did not produce one common pair grid.")
    fixed_modality = canonical_mrct_modality(args.fixed_phase)
    moving_modality = canonical_mrct_modality(args.moving_phase)
    fixed_mask_xyz = raw_foreground_xyz(
        fixed_volume.data_xyz,
        fixed_modality,
        ct_min_hu=protocol.inference.ct_foreground_min_hu,
    )
    moving_mask_xyz = raw_foreground_xyz(
        moving_volume.data_xyz,
        moving_modality,
        ct_min_hu=protocol.inference.ct_foreground_min_hu,
    )
    fixed_normalized = normalize_mrct_intensity(fixed_volume.data_xyz, fixed_modality)
    moving_normalized = normalize_mrct_intensity(moving_volume.data_xyz, moving_modality)
    fixed_normalized[fixed_mask_xyz == 0] = 0.0
    moving_normalized[moving_mask_xyz == 0] = 0.0
    fixed = _tensor_xyz(fixed_normalized, device)
    moving = _tensor_xyz(moving_normalized, device)
    fixed_mask = _tensor_xyz(fixed_mask_xyz.astype(np.float32), device) > 0.5
    moving_mask = _tensor_xyz(moving_mask_xyz.astype(np.float32), device) > 0.5

    dtype = _descriptor_dtype(protocol.inference.descriptor_precision)
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda",
            dtype=dtype or torch.float16,
            enabled=device.type == "cuda" and dtype is not None,
        ):
            fixed_descriptor = model(fixed)
            moving_descriptor = model(moving)
        fixed_descriptor = F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6)
        moving_descriptor = F.normalize(moving_descriptor.float(), dim=1, eps=1e-6)
        common_mask = fixed_mask & moving_mask
        identity_distance = 1.0 - (fixed_descriptor * moving_descriptor).sum(
            dim=1, keepdim=True
        )
        mean_identity_distance = (
            float(identity_distance[common_mask].mean().cpu()) if common_mask.any() else None
        )
    result = descriptor_convex_adam(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=protocol.solver,
    )
    output_flow = Path(args.output_flow).expanduser().resolve()
    if output_flow.suffix.lower() != ".npz":
        raise ValueError("--output-flow must end in .npz.")
    output_flow.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_flow,
        flow_dzyx_voxels=result.flow_dzyx_voxels[0].cpu().numpy().astype(np.float32),
        fixed_grid_affine=fixed_volume.affine,
        component_order=np.asarray("dz,dy,dx"),
        units=np.asarray("fixed_grid_voxels"),
        mapping=np.asarray("fixed_output_grid_to_moving_input_sampling"),
        fixed_phase=np.asarray(args.fixed_phase),
        moving_phase=np.asarray(args.moving_phase),
        version=np.asarray(__version__),
        solver=np.asarray("descriptor_convexadam"),
        checkpoint_source_sha256=np.asarray(payload["source_sha256"]),
    )
    qa = {
        "schema": "pracm_v4_final_inference_qa_v1",
        "version": __version__,
        "fixed": str(Path(args.fixed).resolve()),
        "moving": str(Path(args.moving).resolve()),
        "fixed_phase": args.fixed_phase,
        "moving_phase": args.moving_phase,
        "flow_convention": "fixed grid to moving sampling location, dzyx native voxels",
        "descriptor": "faithful_v2 full-resolution DSIR, 24 channels",
        "solver": "descriptor ConvexAdam",
        "fixed_foreground_fraction": float(fixed_mask.float().mean().cpu()),
        "moving_foreground_fraction": float(moving_mask.float().mean().cpu()),
        "identity_descriptor_cosine_distance": mean_identity_distance,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_logical_step": int(payload["logical_step"]),
        "checkpoint_optimizer_step": int(payload["optimizer_step"]),
        "checkpoint_source_sha256": payload["source_sha256"],
        "observed_source_sha256": observed_source,
        "source_mismatch_allowed": bool(args.allow_source_mismatch),
        "config_mismatch_allowed": bool(args.allow_config_mismatch),
        "solver_diagnostics": result.diagnostics,
    }
    qa_path = (
        Path(args.output_qa).expanduser().resolve()
        if args.output_qa
        else output_flow.with_suffix(".qa.json")
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(
        json.dumps(qa, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"flow": str(output_flow), "qa": str(qa_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

