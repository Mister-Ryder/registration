"""Route between frozen Core and dense-corefix using solver capture range only."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class CaptureRouterConfig:
    residual_grid_spacing_native_voxels: int = 6
    residual_displacement_half_width_cells: int = 4
    maximum_fold_fraction: float = 1.0e-4

    @property
    def residual_capture_radius_native_voxels(self) -> int:
        return int(self.residual_grid_spacing_native_voxels) * int(
            self.residual_displacement_half_width_cells
        )

    def __post_init__(self) -> None:
        if self.residual_grid_spacing_native_voxels < 1:
            raise ValueError("Residual spacing must be positive.")
        if self.residual_displacement_half_width_cells < 1:
            raise ValueError("Residual half width must be positive.")
        if not 0 <= self.maximum_fold_fraction < 1:
            raise ValueError("Fold threshold must lie in [0,1).")


@dataclass(frozen=True)
class CaptureDecision:
    selected: str
    use_dense_corefix: bool
    coarse_body_p95_displacement_native_voxels: float
    residual_capture_radius_native_voxels: int
    strictly_exceeds_residual_capture: bool
    dense_coarse_topology_safe: bool
    dense_coarse_forward_backward_safe: bool
    evidence: Dict[str, Any]


def _finite_mapping(value: Mapping[str, Any]) -> bool:
    numeric = [item for item in value.values() if isinstance(item, (int, float))]
    return bool(numeric) and all(math.isfinite(float(item)) for item in numeric)


def decide_from_corefix_qa(
    qa: Mapping[str, Any], config: CaptureRouterConfig | None = None
) -> CaptureDecision:
    selected = config or CaptureRouterConfig()
    diagnostics = qa.get("solver_diagnostics", {})
    if diagnostics.get("solver_implementation_id") != (
        "v4_dense_coarse_compose_identity_corefix_v1"
    ):
        raise ValueError("QA is not from the frozen dense-corefix solver.")
    parameters = diagnostics.get("parameters", {})
    if (
        int(parameters.get("residual_grid_spacing", -1))
        != selected.residual_grid_spacing_native_voxels
        or int(parameters.get("residual_displacement_half_width", -1))
        != selected.residual_displacement_half_width_cells
    ):
        raise ValueError("QA residual capture configuration differs from the router.")
    coarse_stage = diagnostics.get("coarse_stage", {})
    if (
        int(coarse_stage.get("grid_spacing_native_voxels", -1)) != 12
        or int(coarse_stage.get("displacement_half_width_cells", -1)) != 4
        or int(coarse_stage.get("search_radius_native_voxels", -1)) != 48
    ):
        raise ValueError("QA does not contain the frozen spacing12 dense capture.")
    coarse_selection = diagnostics.get("coarse_selection", {})
    dense = coarse_selection.get("candidate_qa", {}).get("dense_coarse_discrete", {})
    body_p95 = float(dense.get("p95_displacement_native_voxels", float("nan")))
    if not math.isfinite(body_p95) or body_p95 < 0:
        raise ValueError("Dense coarse body p95 displacement is invalid.")
    topology_safe = bool(dense.get("topology_safe")) and (
        float(dense.get("fold_fraction", float("inf")))
        <= selected.maximum_fold_fraction
    ) and float(dense.get("minimum_jacobian", float("-inf"))) > 0.0
    chain = diagnostics.get("algorithm_chain", ())
    inverse_iterations = int(parameters.get("inverse_consistency_iterations", 0))
    forward = coarse_stage.get("forward", {})
    reverse = coarse_stage.get("reverse", {})
    fb_safe = (
        "forward_backward_inverse_consistency" in chain
        and inverse_iterations > 0
        and _finite_mapping(forward)
        and _finite_mapping(reverse)
    )
    radius = selected.residual_capture_radius_native_voxels
    exceeds = body_p95 > float(radius)
    use_dense = bool(exceeds and topology_safe and fb_safe)
    return CaptureDecision(
        selected="dense_corefix" if use_dense else "frozen_core",
        use_dense_corefix=use_dense,
        coarse_body_p95_displacement_native_voxels=body_p95,
        residual_capture_radius_native_voxels=radius,
        strictly_exceeds_residual_capture=exceeds,
        dense_coarse_topology_safe=topology_safe,
        dense_coarse_forward_backward_safe=fb_safe,
        evidence={
            "schema": "v4_solver_capture_router_evidence_v1",
            "labels_used": False,
            "case_score_used": False,
            "formula": "dense_corefix iff coarse_body_p95 > spacing6*half_width4 and topology_safe and fb_safe",
            "parameters": asdict(selected),
            "dense_coarse_candidate_qa": dict(dense),
            "coarse_forward_cost_qa": dict(forward),
            "coarse_reverse_cost_qa": dict(reverse),
            "inverse_consistency_iterations": inverse_iterations,
        },
    )


__all__ = ["CaptureDecision", "CaptureRouterConfig", "decide_from_corefix_qa"]

