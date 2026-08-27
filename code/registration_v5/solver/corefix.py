"""Dense coarse-to-fine correspondence correction for frozen V4 descriptors.

This package changes only the explicit correspondence/solve stage.  The input
contract remains one pair of L2-normalized dense descriptors in
``[1,C,D,H,W]`` layout, accompanied by raw fixed/moving foreground masks.  All
flows are fixed-grid to moving-sampling displacements in native ``dzyx``
voxels.

The correction has three deliberately narrow parts:

* the spacing-12 level retains a *dense* coupled-convex field instead of
  collapsing the cost volume to one constant translation;
* discrete mask evidence uses the same fixed-denominator overlap/unmatched
  objective as Adam refinement and final QA;
* identity/the previous accepted field is an explicit candidate at every
  stage, evaluated with the same full-resolution objective and topology gate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, Iterable, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..inference import dns_convex_solver as _base


@dataclass(frozen=True)
class SolverCorefixConfig:
    coarse_grid_spacing: int = 12
    coarse_displacement_half_width: int = 4
    residual_grid_spacing: int = 6
    residual_displacement_half_width: int = 4
    adam_grid_spacing: int = 2
    diffusion_weight: float = 1.25
    adam_iterations: int = 80
    adam_learning_rate: float = 1.0
    inverse_consistency_iterations: int = 15
    invalid_similarity_cost: float = 8.0
    correspondence_temperature: float = 0.25
    jacobian_weight: float = 50.0
    jacobian_margin: float = 0.05
    maximum_fold_fraction: float = 1.0e-4
    minimum_relative_objective_improvement: float = 1.0e-4
    minimum_absolute_objective_improvement: float = 1.0e-6
    maximum_valid_fraction_drop: float = 1.0e-2
    jacobian_backtrack_bisection_steps: int = 12

    def __post_init__(self) -> None:
        positive_ints = (
            self.coarse_grid_spacing,
            self.coarse_displacement_half_width,
            self.residual_grid_spacing,
            self.residual_displacement_half_width,
            self.adam_grid_spacing,
            self.adam_iterations,
            self.jacobian_backtrack_bisection_steps,
        )
        if any(int(value) < 1 for value in positive_ints):
            raise ValueError("Solver spacings, widths, iterations and bisection must be positive.")
        if self.coarse_grid_spacing <= self.residual_grid_spacing:
            raise ValueError("The dense coarse spacing must exceed the residual spacing.")
        if self.diffusion_weight < 0 or self.adam_learning_rate <= 0:
            raise ValueError("Invalid Adam/diffusion configuration.")
        if self.inverse_consistency_iterations < 0:
            raise ValueError("inverse_consistency_iterations cannot be negative.")
        if self.invalid_similarity_cost <= 4.0:
            raise ValueError(
                "invalid_similarity_cost must exceed the maximum unit-descriptor SSD (4)."
            )
        if self.correspondence_temperature <= 0:
            raise ValueError("correspondence_temperature must be positive.")
        if self.jacobian_weight < 0 or not 0 <= self.jacobian_margin < 1:
            raise ValueError("Invalid Jacobian regularization configuration.")
        if not 0 <= self.maximum_fold_fraction < 1:
            raise ValueError("maximum_fold_fraction must lie in [0,1).")
        if self.minimum_relative_objective_improvement < 0:
            raise ValueError("minimum_relative_objective_improvement cannot be negative.")
        if self.minimum_absolute_objective_improvement < 0:
            raise ValueError("minimum_absolute_objective_improvement cannot be negative.")
        if not 0 <= self.maximum_valid_fraction_drop < 1:
            raise ValueError("maximum_valid_fraction_drop must lie in [0,1).")

    @classmethod
    def from_protocol(cls, protocol: Any) -> "SolverCorefixConfig":
        """Map a frozen V4 ``ConvexAdamProtocol`` without changing descriptors."""

        residual_spacing = int(protocol.grid_spacing)
        return cls(
            coarse_grid_spacing=int(
                getattr(protocol, "global_grid_spacing", 2 * residual_spacing)
            ),
            coarse_displacement_half_width=int(
                getattr(
                    protocol,
                    "global_displacement_half_width",
                    protocol.displacement_half_width,
                )
            ),
            residual_grid_spacing=residual_spacing,
            residual_displacement_half_width=int(protocol.displacement_half_width),
            adam_grid_spacing=int(protocol.adam_grid_spacing),
            diffusion_weight=float(protocol.diffusion_weight),
            adam_iterations=max(int(protocol.adam_iterations), 1),
            adam_learning_rate=float(protocol.adam_learning_rate),
            inverse_consistency_iterations=(
                int(protocol.inverse_consistency_iterations)
                if bool(protocol.inverse_consistency)
                else 0
            ),
            invalid_similarity_cost=float(protocol.invalid_candidate_penalty),
        )


@dataclass(frozen=True)
class CorefixResult:
    flow_native_dzyx_voxels: torch.Tensor
    dense_coarse_flow_native_dzyx_voxels: torch.Tensor
    discrete_composed_flow_native_dzyx_voxels: torch.Tensor
    diagnostics: Dict[str, Any]
    mapping: str = "fixed grid to moving sampling location"


@dataclass(frozen=True)
class _DiscreteStage:
    flow_native_dzyx_voxels: torch.Tensor
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class _Candidate:
    name: str
    flow: torch.Tensor


def fixed_denominator_candidate_cost(
    descriptor_ssd: torch.Tensor,
    fixed_occupancy: torch.Tensor,
    moving_occupancy: torch.Tensor,
    invalid_similarity_cost: float,
) -> torch.Tensor:
    """Pointwise numerator shared by discrete search, Adam, and final QA.

    Extra moving support is not invalid.  Only fixed evidence without sampled
    moving support receives the invalid penalty.
    """

    evidence = fixed_occupancy * moving_occupancy
    unmatched = (fixed_occupancy - evidence).clamp_min(0.0)
    return descriptor_ssd * evidence + float(invalid_similarity_cost) * unmatched


def _double_box(value: torch.Tensor) -> torch.Tensor:
    value = F.avg_pool3d(value, 3, stride=1, padding=1)
    return F.avg_pool3d(value, 3, stride=1, padding=1)


def _masked_fixed_denominator_correlate(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_occupancy: torch.Tensor,
    moving_occupancy: torch.Tensor,
    *,
    displacement_half_width: int,
    grid_spacing: int,
    native_shape: Tuple[int, int, int],
    invalid_similarity_cost: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official candidate layout with a fixed-denominator masked data term."""

    depth, height, width = (int(value) for value in native_shape)
    low_shape = (
        depth // int(grid_spacing),
        height // int(grid_spacing),
        width // int(grid_spacing),
    )
    if tuple(fixed.shape[-3:]) != low_shape or tuple(moving.shape[-3:]) != low_shape:
        raise ValueError("Descriptors do not match the requested pooled search grid.")
    radius = int(displacement_half_width)
    size = 2 * radius + 1
    channels = int(fixed.shape[1])

    with torch.no_grad():
        moving_unfold = F.unfold(
            F.pad(moving, (radius, radius, radius, radius, radius, radius)).squeeze(0),
            size,
        ).view(channels, -1, size**2, low_shape[1], low_shape[2])
        moving_occupancy_unfold = F.unfold(
            F.pad(
                moving_occupancy,
                (radius, radius, radius, radius, radius, radius),
            ).squeeze(0),
            size,
        ).view(1, -1, size**2, low_shape[1], low_shape[2])

        fixed_layout = fixed.permute(1, 2, 0, 3, 4)
        fixed_occupancy_layout = fixed_occupancy.permute(1, 2, 0, 3, 4).float()
        fixed_denominator = _double_box(fixed_occupancy.float())
        has_fixed_evidence = fixed_denominator > 1.0e-6
        cost = fixed.new_empty((size**3, *low_shape), dtype=torch.float32)

        for depth_offset in range(size):
            moving_candidates = moving_unfold[
                :, depth_offset : depth_offset + low_shape[0]
            ]
            candidate_occupancy = moving_occupancy_unfold[
                :, depth_offset : depth_offset + low_shape[0]
            ].float()
            descriptor_ssd = (
                fixed_layout.float() - moving_candidates.float()
            ).square().sum(dim=0, keepdim=True)
            numerator = fixed_denominator_candidate_cost(
                descriptor_ssd,
                fixed_occupancy_layout,
                candidate_occupancy,
                invalid_similarity_cost,
            )
            local_numerator = _double_box(numerator.transpose(2, 1))
            local_cost = local_numerator / fixed_denominator.clamp_min(1.0e-6)
            local_cost = torch.where(
                has_fixed_evidence,
                local_cost,
                torch.zeros_like(local_cost),
            )
            cost[depth_offset::size] = local_cost.squeeze(0)

        # Preserve the official ConvexAdam candidate ordering.  The 2-D
        # ``unfold`` trick above temporarily stores the z/y axes transposed;
        # the displacement mesh is dzyx only after this exact reshape.
        cost = (
            cost.view(size, size, size, *low_shape)
            .transpose(1, 0)
            .reshape(size**3, *low_shape)
        )

        mesh = _base._candidate_mesh(
            radius, device=fixed.device, dtype=torch.float32
        )[:, :, 0]
        identity_index = int(torch.argmin(mesh.square().sum(dim=0)).item())
        argmin = torch.argmin(cost, dim=0)
        argmin = torch.where(
            has_fixed_evidence[0, 0],
            argmin,
            argmin.new_full((), identity_index),
        )
    return cost, argmin


def _weighted_cost_diagnostics(
    cost: torch.Tensor,
    occupancy: torch.Tensor,
    *,
    temperature: float,
) -> Dict[str, float]:
    probability = torch.softmax(-cost.float() / float(temperature), dim=0)
    normalized_entropy = -(
        probability * probability.clamp_min(1.0e-8).log()
    ).sum(dim=0) / math.log(float(cost.shape[0]))
    max_probability = probability.max(dim=0).values
    mesh = _base._candidate_mesh(
        (round(cost.shape[0] ** (1.0 / 3.0)) - 1) // 2,
        device=cost.device,
        dtype=torch.float32,
    )[:, :, 0]
    identity_index = int(torch.argmin(mesh.square().sum(dim=0)).item())
    best = cost.min(dim=0).values.float()
    identity = cost[identity_index].float()
    weight = occupancy[0, 0].float()
    denominator = weight.sum().clamp_min(1.0)
    return {
        "normalized_entropy": float(
            (normalized_entropy * weight).sum().div(denominator).cpu()
        ),
        "mean_max_probability": float(
            (max_probability * weight).sum().div(denominator).cpu()
        ),
        "mean_identity_minus_best_cost": float(
            ((identity - best) * weight).sum().div(denominator).cpu()
        ),
    }


def _coupled_convex_boundary_preserving(
    cost: torch.Tensor,
    argmin: torch.Tensor,
    displacement_mesh: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    grid_spacing: int,
) -> torch.Tensor:
    """Official coupled-convex schedule without artificial zero-pad shrinkage."""

    low_shape = tuple(int(value) // int(grid_spacing) for value in native_shape)

    def smooth(field: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            field,
            3,
            stride=1,
            padding=1,
            count_include_pad=False,
        )

    field = smooth(
        displacement_mesh.view(3, -1)[:, argmin.reshape(-1)].reshape(
            1, 3, *low_shape
        )
    )
    coefficients = cost.new_tensor((0.003, 0.01, 0.03, 0.1, 0.3, 1.0))
    for coefficient in coefficients:
        selected = torch.zeros_like(argmin)
        with torch.no_grad():
            for depth_index in range(low_shape[0]):
                coupled = cost[:, depth_index] + coefficient * (
                    displacement_mesh
                    - field[:, :, depth_index].reshape(3, 1, -1)
                ).square().sum(dim=0).view(-1, low_shape[1], low_shape[2])
                selected[depth_index] = torch.argmin(coupled, dim=0)
        field = smooth(
            displacement_mesh.view(3, -1)[:, selected.reshape(-1)].reshape(
                1, 3, *low_shape
            )
        )
    return field


def _check_inputs(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: SolverCorefixConfig,
) -> Tuple[int, int, int]:
    if fixed.ndim != 5 or fixed.shape != moving.shape or fixed.shape[0] != 1:
        raise ValueError("Descriptors must be equal [1,C,D,H,W] tensors.")
    expected_mask = (1, 1, *fixed.shape[-3:])
    if tuple(fixed_mask.shape) != expected_mask or tuple(moving_mask.shape) != expected_mask:
        raise ValueError("Masks must be [1,1,D,H,W] on the descriptor grid.")
    if fixed.device != moving.device:
        raise ValueError("Descriptors must share a device.")
    if not torch.isfinite(fixed).all() or not torch.isfinite(moving).all():
        raise ValueError("Descriptors contain non-finite values.")
    shape = tuple(int(value) for value in fixed.shape[-3:])
    for spacing in (config.coarse_grid_spacing, config.residual_grid_spacing):
        if any(value // int(spacing) < 3 for value in shape):
            raise ValueError(
                f"Every spacing-{spacing} search dimension must contain at least three cells."
            )
    if any(value // config.adam_grid_spacing < 3 for value in shape):
        raise ValueError("Every Adam dimension must contain at least three cells.")
    if int(fixed_mask.sum()) < 8 or int(moving_mask.sum()) < 8:
        raise ValueError("Foreground masks have insufficient support.")
    return shape


def _dense_discrete_stage(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    grid_spacing: int,
    displacement_half_width: int,
    config: SolverCorefixConfig,
) -> _DiscreteStage:
    fixed, fixed_occupancy = _base._masked_pool_descriptor(
        fixed_descriptor, fixed_mask, int(grid_spacing)
    )
    moving, moving_occupancy = _base._masked_pool_descriptor(
        moving_descriptor, moving_mask, int(grid_spacing)
    )
    search_dtype = torch.float16 if fixed_descriptor.is_cuda else torch.float32
    fixed = fixed.to(search_dtype)
    moving = moving.to(search_dtype)
    fixed_occupancy = fixed_occupancy.to(search_dtype)
    moving_occupancy = moving_occupancy.to(search_dtype)

    correlate, _, inverse_consistency = _base._load_official_primitives()
    del correlate  # Candidate layout is retained; only its masked cost is corrected.
    forward_cost, forward_argmin = _masked_fixed_denominator_correlate(
        fixed,
        moving,
        fixed_occupancy,
        moving_occupancy,
        displacement_half_width=int(displacement_half_width),
        grid_spacing=int(grid_spacing),
        native_shape=native_shape,
        invalid_similarity_cost=config.invalid_similarity_cost,
    )
    reverse_cost, reverse_argmin = _masked_fixed_denominator_correlate(
        moving,
        fixed,
        moving_occupancy,
        fixed_occupancy,
        displacement_half_width=int(displacement_half_width),
        grid_spacing=int(grid_spacing),
        native_shape=native_shape,
        invalid_similarity_cost=config.invalid_similarity_cost,
    )
    mesh = _base._candidate_mesh(
        int(displacement_half_width), device=fixed.device, dtype=torch.float32
    ).to(forward_cost.dtype)
    forward = _coupled_convex_boundary_preserving(
        forward_cost,
        forward_argmin,
        mesh,
        grid_spacing=int(grid_spacing),
        native_shape=native_shape,
    )
    reverse = _coupled_convex_boundary_preserving(
        reverse_cost,
        reverse_argmin,
        mesh,
        grid_spacing=int(grid_spacing),
        native_shape=native_shape,
    )
    flow_native = _base._inverse_consistent_native_flow(
        forward,
        reverse,
        native_shape=native_shape,
        grid_spacing=int(grid_spacing),
        iterations=int(config.inverse_consistency_iterations),
        inverse_consistency=inverse_consistency,
    )
    diagnostics: Dict[str, Any] = {
        "grid_spacing_native_voxels": int(grid_spacing),
        "displacement_half_width_cells": int(displacement_half_width),
        "search_radius_native_voxels": int(grid_spacing)
        * int(displacement_half_width),
        "field_shape_dzyx": tuple(int(value) for value in forward.shape[-3:]),
        "cost_definition": (
            "(descriptor_ssd * fixed_occ * moving_occ + invalid * "
            "fixed_occ * (1-moving_occ)) / local_fixed_occ"
        ),
        "forward": _weighted_cost_diagnostics(
            forward_cost,
            fixed_occupancy,
            temperature=config.correspondence_temperature,
        ),
        "reverse": _weighted_cost_diagnostics(
            reverse_cost,
            moving_occupancy,
            temperature=config.correspondence_temperature,
        ),
    }
    return _DiscreteStage(flow_native.float().detach(), diagnostics)


def _sampling_grid(flow_native_dzyx: torch.Tensor) -> torch.Tensor:
    return _base._grid_from_refinement_flow(flow_native_dzyx)


def _prewarp_moving(
    moving_descriptor: torch.Tensor,
    moving_mask: torch.Tensor,
    base_flow_native_dzyx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    grid = _sampling_grid(base_flow_native_dzyx)
    descriptor = F.grid_sample(
        moving_descriptor.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    mask = F.grid_sample(
        moving_mask.float(),
        grid,
        mode="nearest",
        padding_mode="zeros",
        align_corners=False,
    ) > 0.5
    descriptor = F.normalize(descriptor, dim=1, eps=1.0e-6)
    descriptor = descriptor * mask.to(descriptor.dtype)
    return descriptor, mask


def compose_base_after_residual_native(
    base_flow_native_dzyx: torch.Tensor,
    residual_flow_native_dzyx: torch.Tensor,
) -> torch.Tensor:
    """Apply residual first, then sample the fixed-to-moving base map."""

    if base_flow_native_dzyx.shape != residual_flow_native_dzyx.shape:
        raise ValueError("Base and residual flows must have identical native-grid shapes.")
    sampled_base = F.grid_sample(
        base_flow_native_dzyx.float(),
        _sampling_grid(residual_flow_native_dzyx.float()),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return residual_flow_native_dzyx.float() + sampled_base


def _qa_dict(values: Sequence[float]) -> Dict[str, float]:
    keys = (
        "objective",
        "valid_fraction_of_fixed",
        "fold_fraction",
        "minimum_jacobian",
        "mean_displacement_native_voxels",
        "p95_displacement_native_voxels",
    )
    return {key: float(value) for key, value in zip(keys, values)}


def _candidate_qa(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    flow: torch.Tensor,
    invalid_similarity_cost: float,
) -> tuple[float, float, float, float, float, float]:
    return _base._full_resolution_qa(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        flow,
        float(invalid_similarity_cost),
    )


def _largest_safe_backtrack(
    previous: torch.Tensor,
    proposal: torch.Tensor,
    *,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: SolverCorefixConfig,
) -> tuple[torch.Tensor, float, tuple[float, float, float, float, float, float], int]:
    low = 0.0
    high = 1.0
    safe_flow = previous
    safe_qa = _candidate_qa(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        previous,
        config.invalid_similarity_cost,
    )
    evaluations = 0
    delta = proposal - previous
    for _ in range(int(config.jacobian_backtrack_bisection_steps)):
        alpha = 0.5 * (low + high)
        candidate = previous + alpha * delta
        qa = _candidate_qa(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            candidate,
            config.invalid_similarity_cost,
        )
        evaluations += 1
        if qa[2] <= config.maximum_fold_fraction:
            low = alpha
            safe_flow = candidate
            safe_qa = qa
        else:
            high = alpha
    return safe_flow, float(low), safe_qa, evaluations


def _select_against_previous(
    previous: _Candidate,
    proposals: Iterable[_Candidate],
    *,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: SolverCorefixConfig,
) -> tuple[_Candidate, Dict[str, Any]]:
    previous_qa = _candidate_qa(
        fixed,
        moving,
        fixed_mask,
        moving_mask,
        previous.flow,
        config.invalid_similarity_cost,
    )
    if previous_qa[2] > config.maximum_fold_fraction:
        raise RuntimeError("The previous/identity candidate is not topology-safe.")
    required_improvement = max(
        float(config.minimum_absolute_objective_improvement),
        abs(float(previous_qa[0]))
        * float(config.minimum_relative_objective_improvement),
    )
    records: Dict[str, Any] = {
        previous.name: {
            **_qa_dict(previous_qa),
            "accepted": True,
            "role": "previous_identity_preservation_candidate",
            "backtrack_alpha": 0.0,
        }
    }
    best = previous
    best_qa = previous_qa
    best_objective_limit = float(previous_qa[0]) - required_improvement

    for proposal in proposals:
        qa = _candidate_qa(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            proposal.flow,
            config.invalid_similarity_cost,
        )
        candidate = proposal
        alpha = 1.0
        backtrack_evaluations = 0
        if qa[2] > config.maximum_fold_fraction:
            flow, alpha, qa, backtrack_evaluations = _largest_safe_backtrack(
                previous.flow,
                proposal.flow,
                fixed=fixed,
                moving=moving,
                fixed_mask=fixed_mask,
                moving_mask=moving_mask,
                config=config,
            )
            candidate = _Candidate(f"{proposal.name}_jacbacktracked", flow)
        safe = qa[2] <= config.maximum_fold_fraction
        valid = qa[1] >= (
            previous_qa[1] - float(config.maximum_valid_fraction_drop)
        )
        improves_previous = qa[0] <= best_objective_limit
        records[proposal.name] = {
            **_qa_dict(qa),
            "topology_safe": bool(safe),
            "valid_fraction_acceptable": bool(valid),
            "improves_previous_by_required_margin": bool(improves_previous),
            "backtrack_alpha": float(alpha),
            "backtrack_evaluations": int(backtrack_evaluations),
        }
        if safe and valid and improves_previous and qa[0] < best_qa[0]:
            best = candidate
            best_qa = qa

    for value in records.values():
        value["accepted"] = False
    selected_record = records.get(best.name)
    if selected_record is None and best.name.endswith("_jacbacktracked"):
        selected_record = records[best.name.removesuffix("_jacbacktracked")]
    assert selected_record is not None
    selected_record["accepted"] = True
    return best, {
        "previous_candidate": previous.name,
        "selected_candidate": best.name,
        "required_objective_improvement": float(required_improvement),
        "candidate_qa": records,
        "selected_qa": _qa_dict(best_qa),
    }


def _residual_refinement_config(config: SolverCorefixConfig) -> _base.DNSConvexConfig:
    return _base.DNSConvexConfig(
        grid_spacing=int(config.residual_grid_spacing),
        displacement_half_width=int(config.residual_displacement_half_width),
        adam_grid_spacing=int(config.adam_grid_spacing),
        lambda_weight=float(config.diffusion_weight),
        adam_iterations=int(config.adam_iterations),
        adam_learning_rate=float(config.adam_learning_rate),
        inverse_consistency_iterations=int(config.inverse_consistency_iterations),
        mask_cost_weight=float(config.invalid_similarity_cost),
        invalid_similarity_cost=float(config.invalid_similarity_cost),
        correspondence_temperature=float(config.correspondence_temperature),
        jacobian_weight=float(config.jacobian_weight),
        jacobian_margin=float(config.jacobian_margin),
        maximum_fold_fraction=float(config.maximum_fold_fraction),
        fail_on_excess_folding=False,
    )


def solve_corefix(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: SolverCorefixConfig | None = None,
) -> CorefixResult:
    """Run dense spacing-12 -> composed spacing-6 residual registration."""

    started = time.perf_counter()
    selected = config or SolverCorefixConfig()
    native_shape = _check_inputs(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask,
        moving_mask,
        selected,
    )
    fixed_mask = fixed_mask.to(device=fixed_descriptor.device).bool()
    moving_mask = moving_mask.to(device=fixed_descriptor.device).bool()
    fixed = F.normalize(fixed_descriptor.detach().float(), dim=1, eps=1.0e-6)
    moving = F.normalize(moving_descriptor.detach().float(), dim=1, eps=1.0e-6)
    zero = fixed.new_zeros((1, 3, *native_shape))

    with torch.no_grad():
        coarse_stage = _dense_discrete_stage(
            fixed,
            moving,
            fixed_mask,
            moving_mask,
            native_shape=native_shape,
            grid_spacing=selected.coarse_grid_spacing,
            displacement_half_width=selected.coarse_displacement_half_width,
            config=selected,
        )
    accepted_coarse, coarse_selection = _select_against_previous(
        _Candidate("identity", zero),
        (_Candidate("dense_coarse_discrete", coarse_stage.flow_native_dzyx_voxels),),
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=selected,
    )
    coarse_flow = accepted_coarse.flow.detach()
    shifted_moving, shifted_moving_mask = _prewarp_moving(
        moving, moving_mask, coarse_flow
    )

    with torch.no_grad():
        residual_stage = _dense_discrete_stage(
            fixed,
            shifted_moving,
            fixed_mask,
            shifted_moving_mask,
            native_shape=native_shape,
            grid_spacing=selected.residual_grid_spacing,
            displacement_half_width=selected.residual_displacement_half_width,
            config=selected,
        )
    residual_discrete = residual_stage.flow_native_dzyx_voxels
    refinement_config = _residual_refinement_config(selected)
    residual_refined, _, refinement_regularization, refinement_jacobian = (
        _base._refine_with_adam(
            fixed,
            shifted_moving,
            fixed_mask,
            shifted_moving_mask,
            residual_discrete,
            native_shape=native_shape,
            config=refinement_config,
        )
    )
    composed_discrete = compose_base_after_residual_native(
        coarse_flow, residual_discrete
    )
    composed_refined = compose_base_after_residual_native(
        coarse_flow, residual_refined
    )
    final, residual_selection = _select_against_previous(
        _Candidate("previous_dense_coarse", coarse_flow),
        (
            _Candidate("composed_residual_discrete", composed_discrete),
            _Candidate("composed_residual_refined", composed_refined),
        ),
        fixed=fixed,
        moving=moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=selected,
    )

    diagnostics: Dict[str, Any] = {
        "schema": "v4_solver_corefix_dense_coarse_compose_identity_v1",
        "flow_convention": (
            "fixed grid to moving sampling location, dzyx native voxels"
        ),
        "algorithm_chain": [
            "fixed_denominator_masked_spacing12_dense_coupled_convex",
            "forward_backward_inverse_consistency",
            "identity_vs_dense_coarse_same_objective_selection",
            "prewarp_moving_by_accepted_dense_coarse",
            "fixed_denominator_masked_spacing6_residual_coupled_convex",
            "adam_residual_refinement",
            "pullback_composition_residual_plus_warped_coarse",
            "previous_vs_discrete_vs_refined_same_objective_selection",
            "jacobian_safe_backtracking_when_needed",
        ],
        "parameters": asdict(selected),
        "coarse_stage": coarse_stage.diagnostics,
        "coarse_selection": coarse_selection,
        "residual_stage": residual_stage.diagnostics,
        "residual_selection": residual_selection,
        "refinement_regularization": float(refinement_regularization),
        "refinement_jacobian_penalty": float(refinement_jacobian),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return CorefixResult(
        flow_native_dzyx_voxels=final.flow.detach(),
        dense_coarse_flow_native_dzyx_voxels=coarse_flow.detach(),
        discrete_composed_flow_native_dzyx_voxels=composed_discrete.detach(),
        diagnostics=diagnostics,
    )


__all__ = [
    "CorefixResult",
    "SolverCorefixConfig",
    "compose_base_after_residual_native",
    "fixed_denominator_candidate_cost",
    "solve_corefix",
]
