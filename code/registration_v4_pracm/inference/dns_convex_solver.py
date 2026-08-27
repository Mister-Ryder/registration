"""Descriptor-agnostic explicit correspondence solver for PRA-CM V4.

The discrete stage intentionally reuses the vendored ConvexAdam correlation,
coupled-convex and inverse-consistency primitives.  Inputs and outputs follow
the rest of PRA-CM: tensors are [B,C,D,H,W], and flow is fixed-grid to moving
sampling displacement in native dzyx voxels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import time
from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from .instance_optimization import (
    _jacobian_determinant_from_normalized,
    jacobian_determinant_native_voxel,
    native_voxel_to_normalized_dzyx,
)


@dataclass(frozen=True)
class DNSConvexConfig:
    grid_spacing: int = 6
    displacement_half_width: int = 4
    adam_grid_spacing: int = 2
    lambda_weight: float = 1.25
    adam_iterations: int = 80
    adam_learning_rate: float = 1.0
    inverse_consistency_iterations: int = 15
    mask_cost_weight: float = 2.0
    invalid_similarity_cost: float = 2.0
    correspondence_temperature: float = 0.25
    jacobian_weight: float = 50.0
    jacobian_margin: float = 0.05
    maximum_fold_fraction: float = 1.0e-4
    fail_on_excess_folding: bool = True

    def __post_init__(self) -> None:
        if self.grid_spacing < 1 or self.displacement_half_width < 1:
            raise ValueError("grid_spacing and displacement_half_width must be positive.")
        if self.adam_grid_spacing < 1 or self.adam_iterations < 1:
            raise ValueError("Adam spacing and iteration count must be positive.")
        if self.lambda_weight < 0 or self.adam_learning_rate <= 0:
            raise ValueError("Invalid Adam refinement parameters.")
        if self.inverse_consistency_iterations < 0:
            raise ValueError("inverse_consistency_iterations cannot be negative.")
        if self.mask_cost_weight < 0 or self.invalid_similarity_cost <= 0:
            raise ValueError("Mask costs must be non-negative and invalid cost positive.")
        if self.correspondence_temperature <= 0:
            raise ValueError("correspondence_temperature must be positive.")
        if self.jacobian_weight < 0 or not 0 <= self.jacobian_margin < 1:
            raise ValueError("Invalid Jacobian regularization parameters.")
        if not 0 <= self.maximum_fold_fraction < 1:
            raise ValueError("maximum_fold_fraction must lie in [0,1).")


@dataclass(frozen=True)
class DNSConvexDiagnostics:
    native_shape_dzyx: Tuple[int, int, int]
    coarse_shape_dzyx: Tuple[int, int, int]
    refinement_shape_dzyx: Tuple[int, int, int]
    search_radius_native_voxels: int
    forward_normalized_entropy: float
    forward_mean_max_probability: float
    reverse_normalized_entropy: float
    reverse_mean_max_probability: float
    discrete_similarity: float
    refined_similarity: float
    refinement_regularization: float
    refinement_jacobian_penalty: float
    valid_fraction_of_fixed: float
    fold_fraction: float
    minimum_jacobian: float
    mean_displacement_native_voxels: float
    p95_displacement_native_voxels: float
    refinement_accepted: bool
    elapsed_seconds: float


@dataclass(frozen=True)
class DNSConvexResult:
    flow_native_dzyx_voxels: torch.Tensor
    discrete_flow_native_dzyx_voxels: torch.Tensor
    diagnostics: DNSConvexDiagnostics
    mapping: str = "fixed grid to moving sampling location"


def _load_official_primitives() -> tuple[Callable, Callable, Callable]:
    try:
        from convexAdam.convex_adam_utils import (  # type: ignore
            correlate,
            coupled_convex,
            inverse_consistency,
        )
    except ImportError:
        project_root = Path(__file__).resolve().parents[3]
        candidates = (
            project_root / "benchmark_l2r_mrct" / "third_party" / "convexadam" / "src",
            project_root / "third_party" / "convexadam" / "src",
        )
        vendor = next((candidate for candidate in candidates if candidate.is_dir()), None)
        if vendor is None:
            raise ImportError(
                "Vendored ConvexAdam source is missing; checked: "
                + ", ".join(str(candidate) for candidate in candidates)
            )
        vendor_text = str(vendor)
        if vendor_text not in sys.path:
            sys.path.insert(0, vendor_text)
        from convexAdam.convex_adam_utils import (  # type: ignore
            correlate,
            coupled_convex,
            inverse_consistency,
        )
    return correlate, coupled_convex, inverse_consistency


def _check_inputs(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: DNSConvexConfig,
) -> Tuple[int, int, int]:
    if fixed.ndim != 5 or fixed.shape != moving.shape or fixed.shape[0] != 1:
        raise ValueError("Descriptors must be equal [1,C,D,H,W] tensors.")
    expected_mask = (1, 1, *fixed.shape[-3:])
    if tuple(fixed_mask.shape) != expected_mask or tuple(moving_mask.shape) != expected_mask:
        raise ValueError("Masks must be [1,1,D,H,W] on the descriptor grid.")
    if fixed.device != moving.device:
        raise ValueError("Fixed and moving descriptors must share a device.")
    if not torch.isfinite(fixed).all() or not torch.isfinite(moving).all():
        raise ValueError("Descriptors contain non-finite values.")
    shape = tuple(int(value) for value in fixed.shape[-3:])
    if any(value // config.grid_spacing < 3 for value in shape):
        raise ValueError("Every coarse search dimension must contain at least three cells.")
    if any(value // config.adam_grid_spacing < 3 for value in shape):
        raise ValueError("Every Adam refinement dimension must contain at least three cells.")
    if int(fixed_mask.sum()) < 8 or int(moving_mask.sum()) < 8:
        raise ValueError("Foreground masks have insufficient support.")
    return shape


def _masked_pool_descriptor(
    descriptor: torch.Tensor,
    mask: torch.Tensor,
    spacing: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    occupancy = F.avg_pool3d(mask.float(), spacing, stride=spacing)
    pooled = F.avg_pool3d(descriptor * mask.float(), spacing, stride=spacing)
    pooled = pooled / occupancy.clamp_min(1e-6)
    pooled = F.normalize(pooled, dim=1, eps=1e-6)
    pooled = pooled * (occupancy > 1e-6).to(pooled.dtype)
    return pooled, occupancy


def _candidate_mesh(
    half_width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    width = 2 * half_width + 1
    theta = half_width * torch.eye(3, 4, device=device, dtype=dtype)[None]
    # ConvexAdam's cost-volume reshape makes these returned channels dzyx even
    # though affine_grid itself emits xyz. Keep the official construction.
    return F.affine_grid(
        theta,
        (1, 1, width, width, width),
        align_corners=True,
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)


def _cost_volume_qa(
    cost: torch.Tensor,
    fixed_occupancy: torch.Tensor,
    temperature: float,
) -> tuple[float, float]:
    probability = torch.softmax(-cost.float() / float(temperature), dim=0)
    normalized_entropy = -(
        probability * probability.clamp_min(1e-8).log()
    ).sum(dim=0) / math.log(float(cost.shape[0]))
    max_probability = probability.max(dim=0).values
    weight = fixed_occupancy[0, 0].float()
    denominator = weight.sum().clamp_min(1.0)
    return (
        float((normalized_entropy * weight).sum().div(denominator).cpu()),
        float((max_probability * weight).sum().div(denominator).cpu()),
    )


def _coarse_displacement(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_occupancy: torch.Tensor,
    moving_occupancy: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    config: DNSConvexConfig,
    correlate: Callable,
    coupled_convex: Callable,
) -> tuple[torch.Tensor, float, float]:
    spacing = config.grid_spacing
    channel_count = int(fixed.shape[1])
    cost, _ = correlate(
        fixed,
        moving,
        config.displacement_half_width,
        spacing,
        native_shape,
        channel_count,
    )
    if config.mask_cost_weight > 0:
        mask_cost, _ = correlate(
            fixed_occupancy,
            moving_occupancy,
            config.displacement_half_width,
            spacing,
            native_shape,
            1,
        )
        cost = cost + float(config.mask_cost_weight) * mask_cost
    entropy, max_probability = _cost_volume_qa(
        cost, fixed_occupancy, config.correspondence_temperature
    )
    argmin = torch.argmin(cost, dim=0)
    mesh = _candidate_mesh(
        config.displacement_half_width,
        device=fixed.device,
        dtype=fixed.dtype,
    )
    displacement = coupled_convex(cost, argmin, mesh, spacing, native_shape)
    return displacement, entropy, max_probability


def _inverse_consistent_native_flow(
    forward_coarse_dzyx: torch.Tensor,
    reverse_coarse_dzyx: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    grid_spacing: int,
    iterations: int,
    inverse_consistency: Callable,
) -> torch.Tensor:
    coarse_shape = tuple(int(value) for value in forward_coarse_dzyx.shape[-3:])
    factors_dzyx = forward_coarse_dzyx.new_tensor(
        tuple(value / 2.0 for value in coarse_shape)
    ).view(1, 3, 1, 1, 1)
    forward_normalized_xyz = (forward_coarse_dzyx / factors_dzyx).flip(1)
    reverse_normalized_xyz = (reverse_coarse_dzyx / factors_dzyx).flip(1)
    if iterations > 0:
        forward_normalized_xyz, _ = inverse_consistency(
            forward_normalized_xyz,
            reverse_normalized_xyz,
            iter=int(iterations),
        )
    forward_coarse_dzyx = forward_normalized_xyz.flip(1) * factors_dzyx
    flow_native = F.interpolate(
        forward_coarse_dzyx * float(grid_spacing),
        size=native_shape,
        mode="trilinear",
        align_corners=False,
    )
    return flow_native.float()


def _grid_from_refinement_flow(flow_refine_dzyx_voxels: torch.Tensor) -> torch.Tensor:
    shape = tuple(int(value) for value in flow_refine_dzyx_voxels.shape[-3:])
    identity = F.affine_grid(
        torch.eye(
            3,
            4,
            device=flow_refine_dzyx_voxels.device,
            dtype=flow_refine_dzyx_voxels.dtype,
        )[None],
        (flow_refine_dzyx_voxels.shape[0], 1, *shape),
        align_corners=False,
    )
    factor_dzyx = flow_refine_dzyx_voxels.new_tensor(
        tuple(value / 2.0 for value in shape)
    ).view(1, 3, 1, 1, 1)
    normalized_dzyx = flow_refine_dzyx_voxels / factor_dzyx
    return identity + torch.stack(
        (normalized_dzyx[:, 2], normalized_dzyx[:, 1], normalized_dzyx[:, 0]),
        dim=-1,
    )


def _triple_box_smooth(value: torch.Tensor) -> torch.Tensor:
    for _ in range(3):
        value = F.avg_pool3d(value, 3, stride=1, padding=1)
    return value


def _diffusion_regularization(flow_refine_dzyx: torch.Tensor) -> torch.Tensor:
    terms = []
    for dimension in (2, 3, 4):
        current = flow_refine_dzyx.narrow(
            dimension, 1, flow_refine_dzyx.shape[dimension] - 1
        )
        previous = flow_refine_dzyx.narrow(
            dimension, 0, flow_refine_dzyx.shape[dimension] - 1
        )
        terms.append((current - previous).square().mean())
    return sum(terms)


def _refinement_objective(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_occupancy: torch.Tensor,
    moving_occupancy: torch.Tensor,
    flow_refine_dzyx: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    config: DNSConvexConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    grid = _grid_from_refinement_flow(flow_refine_dzyx)
    warped = F.grid_sample(
        moving.float(), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    warped = F.normalize(warped, dim=1, eps=1e-6)
    moving_evidence = F.grid_sample(
        moving_occupancy.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).clamp(0.0, 1.0)
    evidence = fixed_occupancy * moving_evidence
    fixed_count = fixed_occupancy.sum().clamp_min(1.0)
    descriptor_ssd = (fixed.float() - warped).square().sum(dim=1, keepdim=True)
    unmatched = (fixed_occupancy - evidence).clamp_min(0.0)
    similarity = (
        descriptor_ssd * evidence
        + float(config.invalid_similarity_cost) * unmatched
    ).sum() / fixed_count
    regularization = _diffusion_regularization(flow_refine_dzyx)

    refine_shape = tuple(int(value) for value in flow_refine_dzyx.shape[-3:])
    ratios = flow_refine_dzyx.new_tensor(
        tuple(native / refine for native, refine in zip(native_shape, refine_shape))
    ).view(1, 3, 1, 1, 1)
    flow_native_at_refine = flow_refine_dzyx * ratios
    normalized_native = native_voxel_to_normalized_dzyx(
        flow_native_at_refine, native_shape, align_corners=False
    )
    determinant = _jacobian_determinant_from_normalized(
        normalized_native, native_shape, align_corners=False
    )
    jacobian_penalty = F.relu(
        float(config.jacobian_margin) - determinant
    ).square().mean()
    return similarity, regularization, jacobian_penalty, evidence


def _refine_with_adam(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    initial_flow_native_dzyx: torch.Tensor,
    *,
    native_shape: Tuple[int, int, int],
    config: DNSConvexConfig,
) -> tuple[torch.Tensor, float, float, float]:
    spacing = config.adam_grid_spacing
    fixed, fixed_occupancy = _masked_pool_descriptor(
        fixed_descriptor, fixed_mask, spacing
    )
    moving, moving_occupancy = _masked_pool_descriptor(
        moving_descriptor, moving_mask, spacing
    )
    refine_shape = tuple(int(value) for value in fixed.shape[-3:])
    ratios = initial_flow_native_dzyx.new_tensor(
        tuple(native / refine for native, refine in zip(native_shape, refine_shape))
    ).view(1, 3, 1, 1, 1)
    initial_refine = F.interpolate(
        initial_flow_native_dzyx,
        size=refine_shape,
        mode="trilinear",
        align_corners=False,
    ) / ratios
    parameter = torch.nn.Parameter(initial_refine.detach().float())
    optimizer = torch.optim.Adam([parameter], lr=float(config.adam_learning_rate))

    for _ in range(config.adam_iterations):
        smoothed = _triple_box_smooth(parameter)
        similarity, regularization, jacobian_penalty, _ = _refinement_objective(
            fixed,
            moving,
            fixed_occupancy,
            moving_occupancy,
            smoothed,
            native_shape=native_shape,
            config=config,
        )
        total = (
            similarity
            + float(config.lambda_weight) * regularization
            + float(config.jacobian_weight) * jacobian_penalty
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Non-finite DNS ConvexAdam refinement loss.")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

    with torch.no_grad():
        smoothed = _triple_box_smooth(parameter)
        similarity, regularization, jacobian_penalty, _ = _refinement_objective(
            fixed,
            moving,
            fixed_occupancy,
            moving_occupancy,
            smoothed,
            native_shape=native_shape,
            config=config,
        )
        flow_native_at_refine = smoothed * ratios
        flow_native = F.interpolate(
            flow_native_at_refine,
            size=native_shape,
            mode="trilinear",
            align_corners=False,
        )
    return (
        flow_native.detach(),
        float(similarity.cpu()),
        float(regularization.cpu()),
        float(jacobian_penalty.cpu()),
    )


def _full_resolution_qa(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    flow_native_dzyx: torch.Tensor,
    invalid_cost: float,
) -> tuple[float, float, float, float, float, float]:
    native_shape = tuple(int(value) for value in flow_native_dzyx.shape[-3:])
    grid = _grid_from_refinement_flow(flow_native_dzyx)
    moving_warped = F.grid_sample(
        moving_descriptor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    moving_warped = F.normalize(moving_warped, dim=1, eps=1e-6)
    moving_evidence = F.grid_sample(
        moving_mask.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).clamp(0.0, 1.0)
    fixed_weight = fixed_mask.float()
    evidence = fixed_weight * moving_evidence
    count = fixed_weight.sum().clamp_min(1.0)
    ssd = (fixed_descriptor - moving_warped).square().sum(dim=1, keepdim=True)
    similarity = (
        ssd * evidence + float(invalid_cost) * (fixed_weight - evidence).clamp_min(0)
    ).sum() / count
    determinant = jacobian_determinant_native_voxel(flow_native_dzyx)
    magnitude = flow_native_dzyx.square().sum(dim=1, keepdim=True).sqrt()[fixed_mask]
    return (
        float(similarity.cpu()),
        float((evidence.sum() / count).cpu()),
        float((determinant <= 0).float().mean().cpu()),
        float(determinant.min().cpu()),
        float(magnitude.mean().cpu()),
        float(torch.quantile(magnitude.float(), 0.95).cpu()),
    )


def solve_dns_convex(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: DNSConvexConfig | None = None,
) -> DNSConvexResult:
    """Run DSIR -> discrete correlation -> convex/IC -> Adam refinement."""

    started = time.perf_counter()
    selected = config or DNSConvexConfig()
    native_shape = _check_inputs(
        fixed_descriptor, moving_descriptor, fixed_mask, moving_mask, selected
    )
    correlate, coupled_convex, inverse_consistency = _load_official_primitives()
    dtype = torch.float16 if fixed_descriptor.is_cuda else torch.float32
    fixed_mask = fixed_mask.to(device=fixed_descriptor.device).bool()
    moving_mask = moving_mask.to(device=fixed_descriptor.device).bool()
    fixed_full = F.normalize(fixed_descriptor.detach().float(), dim=1, eps=1e-6)
    moving_full = F.normalize(moving_descriptor.detach().float(), dim=1, eps=1e-6)

    with torch.no_grad():
        fixed_coarse, fixed_occupancy = _masked_pool_descriptor(
            fixed_full, fixed_mask, selected.grid_spacing
        )
        moving_coarse, moving_occupancy = _masked_pool_descriptor(
            moving_full, moving_mask, selected.grid_spacing
        )
        fixed_coarse = fixed_coarse.to(dtype)
        moving_coarse = moving_coarse.to(dtype)
        fixed_occupancy = fixed_occupancy.to(dtype)
        moving_occupancy = moving_occupancy.to(dtype)
        forward, forward_entropy, forward_maxp = _coarse_displacement(
            fixed_coarse,
            moving_coarse,
            fixed_occupancy,
            moving_occupancy,
            native_shape=native_shape,
            config=selected,
            correlate=correlate,
            coupled_convex=coupled_convex,
        )
        reverse, reverse_entropy, reverse_maxp = _coarse_displacement(
            moving_coarse,
            fixed_coarse,
            moving_occupancy,
            fixed_occupancy,
            native_shape=native_shape,
            config=selected,
            correlate=correlate,
            coupled_convex=coupled_convex,
        )
        discrete = _inverse_consistent_native_flow(
            forward,
            reverse,
            native_shape=native_shape,
            grid_spacing=selected.grid_spacing,
            iterations=selected.inverse_consistency_iterations,
            inverse_consistency=inverse_consistency,
        )

    discrete_qa = _full_resolution_qa(
        fixed_full,
        moving_full,
        fixed_mask,
        moving_mask,
        discrete,
        selected.invalid_similarity_cost,
    )
    refined, _, refinement_reg, refinement_jac = _refine_with_adam(
        fixed_full,
        moving_full,
        fixed_mask,
        moving_mask,
        discrete,
        native_shape=native_shape,
        config=selected,
    )
    refined_qa = _full_resolution_qa(
        fixed_full,
        moving_full,
        fixed_mask,
        moving_mask,
        refined,
        selected.invalid_similarity_cost,
    )

    refined_safe = refined_qa[2] <= selected.maximum_fold_fraction
    discrete_safe = discrete_qa[2] <= selected.maximum_fold_fraction
    refinement_improves = refined_qa[0] <= discrete_qa[0]
    refinement_accepted = refined_safe and (refinement_improves or not discrete_safe)
    if refinement_accepted:
        final = refined
        final_qa = refined_qa
    elif discrete_safe:
        final = discrete
        final_qa = discrete_qa
    elif selected.fail_on_excess_folding:
        raise RuntimeError(
            "Both discrete and refined DNS-Convex flows exceed the fold-fraction gate: "
            f"discrete={discrete_qa[2]:.6f}, refined={refined_qa[2]:.6f}."
        )
    else:
        final = refined if refined_qa[2] <= discrete_qa[2] else discrete
        final_qa = refined_qa if final is refined else discrete_qa

    diagnostics = DNSConvexDiagnostics(
        native_shape_dzyx=native_shape,
        coarse_shape_dzyx=tuple(int(value) for value in forward.shape[-3:]),
        refinement_shape_dzyx=tuple(
            int(value // selected.adam_grid_spacing) for value in native_shape
        ),
        search_radius_native_voxels=(
            selected.grid_spacing * selected.displacement_half_width
        ),
        forward_normalized_entropy=forward_entropy,
        forward_mean_max_probability=forward_maxp,
        reverse_normalized_entropy=reverse_entropy,
        reverse_mean_max_probability=reverse_maxp,
        discrete_similarity=discrete_qa[0],
        refined_similarity=refined_qa[0],
        refinement_regularization=refinement_reg,
        refinement_jacobian_penalty=refinement_jac,
        valid_fraction_of_fixed=final_qa[1],
        fold_fraction=final_qa[2],
        minimum_jacobian=final_qa[3],
        mean_displacement_native_voxels=final_qa[4],
        p95_displacement_native_voxels=final_qa[5],
        refinement_accepted=refinement_accepted,
        elapsed_seconds=float(time.perf_counter() - started),
    )
    return DNSConvexResult(
        flow_native_dzyx_voxels=final.detach(),
        discrete_flow_native_dzyx_voxels=discrete.detach(),
        diagnostics=diagnostics,
    )
