"""Pairwise 3-D NIfTI inference for a frozen PRA-CM checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .. import __version__
from ..config import load_config
from ..data.nifti import (
    l2r_foreground_mask,
    load_pair_on_fixed_grid,
    normalize_intensity,
    tensor_to_xyz,
    validate_plc_uint8_nifti,
    voxel_spacing_dzyx,
)
from ..inference import register_volume
from ..ops.spatial import jacobian_determinant, warp
from ..training.checkpoint import config_sha256, load_checkpoint, source_tree_sha256
from ..training.module_v4 import PRACMTrainingModule


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixed", required=True)
    parser.add_argument("--moving", required=True)
    parser.add_argument("--fixed-phase", required=True)
    parser.add_argument("--moving-phase", required=True)
    parser.add_argument("--output-flow", required=True)
    parser.add_argument("--output-warped")
    parser.add_argument("--output-qa")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-source-mismatch", action="store_true")
    parser.add_argument("--allow-config-mismatch", action="store_true")
    parser.add_argument(
        "--l2r-domain-mode",
        choices=("raw-foreground", "rectangular"),
        default="raw-foreground",
        help="Use raw modality foreground (core fix) or the legacy rectangular NIfTI domain.",
    )
    parser.add_argument(
        "--response-gate-override",
        choices=("neutral",),
        help="Inference-only diagnostic that makes the candidate response gate exactly neutral.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    if checkpoint["config_sha256"] != config_sha256(config.to_dict()) and not args.allow_config_mismatch:
        raise ValueError("Inference config differs from the checkpoint; use only an explicit audited override.")
    observed_source = source_tree_sha256()
    if checkpoint["source_sha256"] != observed_source and not args.allow_source_mismatch:
        raise ValueError("Inference source differs from the checkpoint.")
    module = PRACMTrainingModule(config.model, config.losses, config.augmentation, config.training)
    module.load_state_dict(checkpoint["module_state"], strict=True)
    if args.response_gate_override:
        module.model.response_gate.mode = args.response_gate_override
    device = torch.device(args.device)
    module.model.to(device).eval()
    if config.data.intensity_mode == "plc_uint8":
        validate_plc_uint8_nifti(args.fixed)
        validate_plc_uint8_nifti(args.moving)
    pair = load_pair_on_fixed_grid(args.fixed, args.moving)
    if config.data.dataset == "l2r_mrct" and args.l2r_domain_mode == "raw-foreground":
        fixed_domain = l2r_foreground_mask(
            pair.fixed.tensor,
            args.fixed_phase,
            domain=pair.fixed.domain,
        )
        moving_domain = l2r_foreground_mask(
            pair.moving.tensor,
            args.moving_phase,
            domain=pair.moving.domain,
        )
    else:
        fixed_domain = pair.fixed.domain
        moving_domain = pair.moving.domain
    fixed = normalize_intensity(
        pair.fixed.tensor,
        config.data.intensity_mode,
        hu_window=config.data.hu_window,
        domain=fixed_domain,
    ).to(device)
    moving = normalize_intensity(
        pair.moving.tensor,
        config.data.intensity_mode,
        hu_window=config.data.hu_window,
        domain=moving_domain,
    ).to(device)
    result = register_volume(
        module.model,
        fixed,
        moving,
        fixed_phase=args.fixed_phase,
        moving_phase=args.moving_phase,
        config=config.inference,
        fixed_domain=fixed_domain.to(device),
        moving_domain=moving_domain.to(device),
        spacing_dzyx=voxel_spacing_dzyx(pair.fixed.affine).to(device),
    )
    output_flow = Path(args.output_flow).expanduser().resolve()
    if output_flow.suffix.lower() != ".npz":
        raise ValueError("--output-flow must use the .npz suffix.")
    output_flow.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_flow,
        flow_dzyx_voxels=result.flow_dzyx[0].cpu().numpy().astype(np.float32),
        variance_dzyx_voxels2=result.variance_dzyx[0].cpu().numpy().astype(np.float32),
        entropy=result.entropy[0, 0].cpu().numpy().astype(np.float32),
        maximum_probability=result.maximum_probability[0, 0].cpu().numpy().astype(np.float32),
        response_gate=result.expected_gate[0, 0].cpu().numpy().astype(np.float32),
        candidate_coverage=result.candidate_coverage[0, 0].cpu().numpy().astype(np.float32),
        support_radius_mm=result.support_radius_mm[0, 0].cpu().numpy().astype(np.float32),
        posterior_solver_correction=result.posterior_solver_correction[0, 0].cpu().numpy().astype(np.float32),
        endpoint_valid=result.endpoint_valid[0, 0].cpu().numpy().astype(np.uint8),
        fixed_grid_affine=pair.fixed.affine,
        component_order=np.asarray("dz,dy,dx"),
        spatial_axis_order=np.asarray("zyx"),
        units=np.asarray("fixed_common_grid_voxels"),
        mapping=np.asarray("fixed_grid_to_resampled_moving_sampling_location"),
        fixed_phase=np.asarray(args.fixed_phase),
        moving_phase=np.asarray(args.moving_phase),
        version=np.asarray(__version__),
        checkpoint_source_sha256=np.asarray(checkpoint["source_sha256"]),
    )
    if args.output_warped:
        warped_original = warp(
            pair.moving.tensor.to(device), result.flow_dzyx
        ).masked_fill(~result.endpoint_valid, 0)
        header = pair.fixed.header.copy()
        header.set_data_dtype(np.float32)
        warped_path = Path(args.output_warped).expanduser().resolve()
        warped_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(tensor_to_xyz(warped_original), pair.fixed.affine, header=header),
            str(warped_path),
        )
    determinant = jacobian_determinant(result.flow_dzyx)
    evidence = result.endpoint_valid
    selected = determinant[evidence]
    flow_magnitude_voxels = result.flow_dzyx.square().sum(dim=1, keepdim=True).sqrt()
    spacing = voxel_spacing_dzyx(pair.fixed.affine).to(device).view(1, 3, 1, 1, 1)
    flow_magnitude_mm = (result.flow_dzyx * spacing).square().sum(dim=1, keepdim=True).sqrt()
    selected_flow_voxels = flow_magnitude_voxels[evidence]
    selected_flow_mm = flow_magnitude_mm[evidence]
    qa = {
        "schema": "pracm_3d_inference_qa_v1",
        "version": __version__,
        "fixed": str(Path(args.fixed).expanduser().resolve()),
        "moving": str(Path(args.moving).expanduser().resolve()),
        "fixed_phase": args.fixed_phase,
        "moving_phase": args.moving_phase,
        "flow_convention": "fixed->moving sampling displacement, dzyx, common-grid voxels",
        "mean_entropy": float(result.entropy[evidence].mean().cpu()) if evidence.any() else None,
        "mean_maximum_probability": float(result.maximum_probability[evidence].mean().cpu()) if evidence.any() else None,
        "mean_candidate_coverage": float(result.candidate_coverage[evidence].mean().cpu()) if evidence.any() else None,
        "mean_response_gate": float(result.expected_gate[evidence].mean().cpu()) if evidence.any() else None,
        "mean_displacement_voxels": float(selected_flow_voxels.mean().cpu()) if selected_flow_voxels.numel() else None,
        "p95_displacement_voxels": float(torch.quantile(selected_flow_voxels.float(), 0.95).cpu()) if selected_flow_voxels.numel() else None,
        "mean_displacement_mm": float(selected_flow_mm.mean().cpu()) if selected_flow_mm.numel() else None,
        "p95_displacement_mm": float(torch.quantile(selected_flow_mm.float(), 0.95).cpu()) if selected_flow_mm.numel() else None,
        "fixed_foreground_fraction": float(fixed_domain.float().mean()),
        "moving_foreground_fraction": float(moving_domain.float().mean()),
        "l2r_domain_mode": args.l2r_domain_mode,
        "response_gate_override": args.response_gate_override,
        "evidence_fraction": float(evidence.float().mean().cpu()),
        "nonpositive_jacobian_fraction": float((selected <= 0).float().mean().cpu()) if selected.numel() else None,
        "minimum_jacobian": float(selected.min().cpu()) if selected.numel() else None,
        "source_mismatch_allowed": bool(args.allow_source_mismatch),
        "config_mismatch_allowed": bool(args.allow_config_mismatch),
    }
    qa_path = Path(args.output_qa).expanduser().resolve() if args.output_qa else output_flow.with_suffix(".qa.json")
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(json.dumps(qa, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"flow": str(output_flow), "qa": str(qa_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
