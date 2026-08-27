"""Run frozen 24-channel DSIR with the audited V5 dense-corefix solver."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .. import __version__
from ..data import raw_foreground_xyz
from ..dns.faithful_v2 import canonical_mrct_modality, normalize_mrct_intensity
from ..flow import save_flow
from ..io import load_pair_on_fixed_grid
from ..network import V4FinalRegistrationModel
from ..protocol import load_protocol
from ..solver.release import descriptor_convex_adam_corefix
from ..state import canonical_hash, load_checkpoint, source_hash


FROZEN_V4_CORE_BEST_SHA256 = (
    "6ba1c54ab260f4fb830b019caeaaf8414c1b45aae435adb4ff9eb68592d5bb70"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_xyz(data: np.ndarray, device: torch.device) -> torch.Tensor:
    value = np.asarray(data, dtype=np.float32).transpose(2, 1, 0).copy()
    return torch.from_numpy(value)[None, None].to(device)


def _descriptor_dtype(name: str):
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[name]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--moving", type=Path, required=True)
    parser.add_argument("--fixed-phase", default="mr")
    parser.add_argument("--moving-phase", default="ct")
    parser.add_argument("--output-flow", type=Path, required=True)
    parser.add_argument("--output-qa", type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    checkpoint = args.checkpoint.expanduser().resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != FROZEN_V4_CORE_BEST_SHA256:
        raise ValueError(
            "V5 formal inference accepts only the frozen V4-Core best checkpoint; "
            f"observed SHA-256 {checkpoint_sha}."
        )
    protocol = load_protocol(args.config)
    payload = load_checkpoint(checkpoint, map_location="cpu")
    if payload["protocol_sha256"] != canonical_hash(protocol.to_dict()):
        raise ValueError("Protocol differs from the frozen descriptor checkpoint.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but CUDA is unavailable.")
    model = V4FinalRegistrationModel(protocol.descriptor).to(device).eval()
    model.load_state_dict(payload["model"], strict=True)

    fixed_volume, moving_volume = load_pair_on_fixed_grid(
        args.fixed.expanduser().resolve(), args.moving.expanduser().resolve()
    )
    fixed_modality = canonical_mrct_modality(args.fixed_phase)
    moving_modality = canonical_mrct_modality(args.moving_phase)
    if (fixed_modality, moving_modality) != ("mr", "ct"):
        raise ValueError("The frozen V5 result contract is MR fixed and CT moving.")
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
    fixed_value = normalize_mrct_intensity(fixed_volume.data_xyz, fixed_modality)
    moving_value = normalize_mrct_intensity(moving_volume.data_xyz, moving_modality)
    fixed_value[fixed_mask_xyz == 0] = 0.0
    moving_value[moving_mask_xyz == 0] = 0.0
    fixed = _tensor_xyz(fixed_value, device)
    moving = _tensor_xyz(moving_value, device)
    fixed_mask = _tensor_xyz(fixed_mask_xyz.astype(np.float32), device) > 0.5
    moving_mask = _tensor_xyz(moving_mask_xyz.astype(np.float32), device) > 0.5

    dtype = _descriptor_dtype(protocol.inference.descriptor_precision)
    with torch.no_grad(), torch.autocast(
        device_type="cuda",
        dtype=dtype or torch.float16,
        enabled=device.type == "cuda" and dtype is not None,
    ):
        fixed_descriptor = model(fixed)
        moving_descriptor = model(moving)
    fixed_descriptor = F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6)
    moving_descriptor = F.normalize(moving_descriptor.float(), dim=1, eps=1e-6)
    result = descriptor_convex_adam_corefix(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=protocol.solver,
    )
    output_flow = args.output_flow.expanduser().resolve()
    if output_flow.exists():
        raise FileExistsError(output_flow)
    metadata = {
        "schema": "registration_v5_dense_corefix_qa_v1",
        "version": __version__,
        "labels_used": False,
        "flow_convention": "fixed grid to moving sampling location, dzyx native voxels",
        "descriptor": "frozen V4-Core faithful full-resolution 24-channel DSIR",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_source_sha256": payload["source_sha256"],
        "observed_v5_source_sha256": source_hash(),
        "legacy_source_mismatch_reason": "audited code consolidation; weights frozen by file SHA-256",
        "solver_diagnostics": result.diagnostics,
    }
    save_flow(
        output_flow,
        result.flow_dzyx_voxels[0].detach().cpu().numpy().astype(np.float32),
        fixed_volume.affine,
        metadata,
    )
    qa_path = (
        args.output_qa.expanduser().resolve()
        if args.output_qa
        else output_flow.with_suffix(".qa.json")
    )
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"flow": str(output_flow), "qa": str(qa_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
