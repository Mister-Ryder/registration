"""Three-case diagnostic: frozen legacy B05 descriptor plus ConvexAdam."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from registration_benchmark.contract import load_registration_manifest
from registration_benchmark.dns.model import MASRNet
from registration_benchmark.flow import save_flow
from registration_benchmark.io import load_pair_on_fixed_grid, robust_unit_interval

from ..convex_solver import descriptor_convex_adam
from ..data import raw_foreground_xyz
from ..protocol import load_protocol


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_xyz(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value, dtype=np.float32).transpose(2, 1, 0).copy())[
        None, None
    ].to(device)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pair-ids", nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    protocol = load_protocol(args.config)
    payload = _load_torch(args.checkpoint.resolve())
    if payload.get("schema") != "mok_masr_checkpoint_v1":
        raise ValueError("Probe requires the frozen legacy B05 MASR checkpoint.")
    architecture = dict(payload.get("architecture", {}))
    model = MASRNet(
        channels=architecture.get("channels", [8, 16, 32, 64]),
        feature_channels=int(architecture.get("feature_channels", 4)),
        descriptor_channels=int(architecture.get("descriptor_channels", 24)),
        dns_dilation=int(architecture.get("dns_dilation", 2)),
    )
    model.load_state_dict(payload["model"], strict=True)
    device = torch.device(args.device)
    model.to(device).eval()
    selected_ids = set(args.pair_ids)
    tasks = [
        task
        for task in load_registration_manifest(args.manifest.resolve(), require_files=True)
        if not selected_ids or task.pair_id in selected_ids
    ][: int(args.limit)]
    if not tasks:
        raise KeyError("No requested public pair was found.")
    summaries = []
    for task in tasks:
        fixed_volume, moving_volume = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        fixed_raw_mask = raw_foreground_xyz(
            fixed_volume.data_xyz,
            task.fixed.modality,
            ct_min_hu=protocol.inference.ct_foreground_min_hu,
        )
        moving_raw_mask = raw_foreground_xyz(
            moving_volume.data_xyz,
            task.moving.modality,
            ct_min_hu=protocol.inference.ct_foreground_min_hu,
        )
        # Preserve exactly the legacy descriptor's training-time robust scaling;
        # this probe changes only the solver and foreground evidence handling.
        fixed = robust_unit_interval(fixed_volume.data_xyz)
        moving = robust_unit_interval(moving_volume.data_xyz)
        fixed[fixed_raw_mask == 0] = 0
        moving[moving_raw_mask == 0] = 0
        fixed_tensor = _tensor_xyz(fixed, device)
        moving_tensor = _tensor_xyz(moving, device)
        fixed_mask = _tensor_xyz(fixed_raw_mask.astype(np.float32), device) > 0.5
        moving_mask = _tensor_xyz(moving_raw_mask.astype(np.float32), device) > 0.5
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            fixed_descriptor = model(fixed_tensor)
            moving_descriptor = model(moving_tensor)
        fixed_descriptor = F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6)
        moving_descriptor = F.normalize(moving_descriptor.float(), dim=1, eps=1e-6)
        result = descriptor_convex_adam(
            fixed_descriptor,
            moving_descriptor,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=protocol.solver,
        )
        pair_dir = (
            args.output_dir.resolve()
            / "B05_legacy_descriptor_convex"
            / task.task_domain
            / task.pair_id
        )
        pair_dir.mkdir(parents=True, exist_ok=True)
        save_flow(
            pair_dir / "flow.npz",
            result.flow_dzyx_voxels[0].cpu().numpy().astype(np.float32),
            fixed_volume.affine,
            {
                "method_id": "B05_legacy_descriptor_convex",
                "pair_id": task.pair_id,
                "task_domain": task.task_domain,
                "task_group": task.task_group,
                "fixed_phase": task.fixed.phase,
                "moving_phase": task.moving.phase,
            },
        )
        diagnostic = {
            "schema": "legacy_b05_descriptor_convex_probe_v1",
            "pair_id": task.pair_id,
            "checkpoint": str(args.checkpoint.resolve()),
            "descriptor": "legacy protocol300 B05 MASRNet",
            "changed_component": "solver: legacy dense IO -> mask-aware descriptor ConvexAdam",
            "solver": result.diagnostics,
        }
        (pair_dir / "probe_qa.json").write_text(
            json.dumps(diagnostic, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (pair_dir / "status.json").write_text(
            json.dumps(
                {
                    "schema": "registration_pair_status_v1",
                    "method_id": "B05_legacy_descriptor_convex",
                    "pair_id": task.pair_id,
                    "task_domain": task.task_domain,
                    "task_group": task.task_group,
                    "success": True,
                    "flow": "flow.npz",
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        summaries.append({"pair_id": task.pair_id, **result.diagnostics})
        del fixed_descriptor, moving_descriptor, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    summary_path = args.output_dir.resolve() / "B05_legacy_descriptor_convex" / "probe_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pairs": len(tasks), "summary": str(summary_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

