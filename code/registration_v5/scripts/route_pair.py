"""Select one frozen flow using the predeclared capture-range rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from ..routing.capture import decide_from_corefix_qa


SCHEMA = "fixed_to_moving_sampling_displacement_dzyx_fixed_voxel_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_flow(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "flow",
            "fixed_affine",
            "schema",
            "component_order",
            "units",
            "mapping",
        }
        if not required.issubset(archive.files):
            raise ValueError(f"Incomplete canonical flow: {path}")
        if str(archive["schema"].item()) != SCHEMA:
            raise ValueError(f"Wrong flow schema: {path}")
        flow = np.asarray(archive["flow"], dtype=np.float32)
        metadata = {
            key: str(archive[key].item())
            for key in ("component_order", "units", "mapping")
        }
    if flow.ndim != 4 or flow.shape[0] != 3 or not np.isfinite(flow).all():
        raise ValueError(f"Invalid canonical displacement: {path}")
    if metadata != {
        "component_order": "dz,dy,dx",
        "units": "fixed_grid_voxels",
        "mapping": "fixed_output_grid_to_moving_input_sampling",
    }:
        raise ValueError(f"Flow convention mismatch: {path}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--dino-flow", type=Path, required=True)
    parser.add_argument("--corefix-flow", type=Path, required=True)
    parser.add_argument("--corefix-qa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method-id", required=True)
    args = parser.parse_args(argv)

    dino = args.dino_flow.expanduser().resolve()
    corefix = args.corefix_flow.expanduser().resolve()
    corefix_qa = args.corefix_qa.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (dino, corefix, corefix_qa):
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_flow(dino)
    _validate_flow(corefix)
    decision = decide_from_corefix_qa(
        json.loads(corefix_qa.read_text(encoding="utf-8"))
    )
    selected = corefix if decision.use_dense_corefix else dino
    selected_name = "dense_corefix" if decision.use_dense_corefix else "frozen_B04_DINO_Reg"

    output.mkdir(parents=True)
    flow_path = output / "flow.npz"
    # The selected candidate is copied byte-for-byte.  No re-serialization is
    # allowed because B04 is the immutable default anchor.
    shutil.copy2(selected, flow_path)
    selected_hash = _sha256(selected)
    materialized_hash = _sha256(flow_path)
    if materialized_hash != selected_hash:
        raise RuntimeError("Selected flow was not materialized byte-identically.")

    qa = {
        "schema": "v4_solver_capture_router_dino_anchor_public8_pair_qa_v1",
        "pair_id": args.pair_id,
        "method_id": args.method_id,
        "labels_used": False,
        "default_anchor": "frozen_B04_DINO_Reg",
        "decision": {
            "selected": selected_name,
            "coarse_body_p95_displacement_native_voxels": (
                decision.coarse_body_p95_displacement_native_voxels
            ),
            "residual_capture_radius_native_voxels": (
                decision.residual_capture_radius_native_voxels
            ),
            "strictly_exceeds_residual_capture": (
                decision.strictly_exceeds_residual_capture
            ),
            "dense_coarse_topology_safe": decision.dense_coarse_topology_safe,
            "dense_coarse_forward_backward_safe": (
                decision.dense_coarse_forward_backward_safe
            ),
        },
        "decision_evidence": decision.evidence,
        "candidate_hashes": {
            "frozen_B04_DINO_Reg_flow": _sha256(dino),
            "dense_corefix_flow": _sha256(corefix),
            "dense_corefix_qa": _sha256(corefix_qa),
        },
        "selected_flow": str(selected),
        "selected_flow_sha256": selected_hash,
        "materialized_flow_sha256": materialized_hash,
        "materialized_byte_identically": True,
    }
    (output / "registration_v5.qa.json").write_text(
        json.dumps(qa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "status.json").write_text(
        json.dumps(
            {
                "schema": "registration_pair_status_v1",
                "method_id": args.method_id,
                "pair_id": args.pair_id,
                "task_domain": "l2r_mrct",
                "task_group": "mr_target",
                "success": True,
                "runtime_seconds": 0.0,
                "flow": "flow.npz",
                "warped_moving": None,
                "native_output_deleted": False,
                "preprocessed_inputs_deleted": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(qa, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
