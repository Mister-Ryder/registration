"""Standalone masked multi-level descriptor instance optimization.

This module is intentionally not imported by PRA-CM's legacy A/B/Full path.
It optimizes a per-case fixed-grid to moving-grid displacement in normalized
dzyx coordinates and can consume any dense full-resolution descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class InstanceOptimizationConfig:
    """Frozen optimization schedule.

    smoothness_weights multiply squared derivatives of native-voxel
    displacement with respect to native-voxel coordinates. The derivatives
    and the Jacobian determinant are dimensionless and level-independent even
    though the optimized parameter remains normalized-grid dzyx.
    """

    scales: Tuple[float, ...] = (0.25, 0.5, 1.0)
    iterations: Tuple[int, ...] = (100, 80, 50)
    learning_rates: Tuple[float, ...] = (0.08, 0.04, 0.02)
    smoothness_weights: Tuple[float, ...] = (0.02, 0.01, 0.005)
    jacobian_weight: float = 0.25
    jacobian_margin: float = 0.10
    invalid_similarity_cost: float = 1.25
    parameterization: str = "svf"
    scaling_squaring_steps: int = 6
    align_corners: bool = True
    minimum_valid_voxels: int = 8

    def __post_init__(self) -> None:
        count = len(self.scales)
        if count < 2:
            raise ValueError("Instance optimization requires at least two pyramid levels.")
        if not (
            len(self.iterations)
            == len(self.learning_rates)
            == len(self.smoothness_weights)
            == count
        ):
            raise ValueError("Every pyramid level needs iterations, learning rate and smoothness.")
        if any(not 0 < float(scale) <= 1 for scale in self.scales):
            raise ValueError("Pyramid scales must lie in (0,1].")
        if tuple(sorted(float(value) for value in self.scales)) != tuple(
            float(value) for value in self.scales
        ):
            raise ValueError("Pyramid scales must be ordered coarse to fine.")
        if abs(float(self.scales[-1]) - 1.0) > 1e-8:
            raise ValueError("The final pyramid level must be full descriptor resolution.")
        if any(int(value) < 1 for value in self.iterations):
            raise ValueError("Every pyramid level needs at least one iteration.")
        if any(float(value) <= 0 for value in self.learning_rates):
            raise ValueError("Learning rates must be positive.")
        if any(float(value) < 0 for value in self.smoothness_weights):
            raise ValueError("Smoothness weights cannot be negative.")
        if float(self.jacobian_weight) < 0:
            raise ValueError("jacobian_weight cannot be negative.")
        if not 0 <= float(self.jacobian_margin) < 1:
            raise ValueError("jacobian_margin must lie in [0,1).")
        if float(self.invalid_similarity_cost) <= 0:
            raise ValueError("invalid_similarity_cost must be positive.")
        if self.parameterization not in {"svf", "dense"}:
            raise ValueError("parameterization must be svf or dense.")
        if int(self.scaling_squaring_steps) < 0:
            raise ValueError("scaling_squaring_steps cannot be negative.")
        if int(self.minimum_valid_voxels) < 2:
            raise ValueError("minimum_valid_voxels must be at least two.")


@dataclass(frozen=True)
class LevelDiagnostics:
    level_index: int
    scale: float
    shape_dzyx: Tuple[int, int, int]
    iterations: int
    learning_rate_normalized: float
    smoothness_weight_dimensionless: float
    initial_similarity_1_minus_cosine: float
    final_similarity_1_minus_cosine: float
    final_regularization_dimensionless: float
    final_jacobian_penalty: float
    final_total_loss: float
    valid_fraction_of_fixed: float
    fold_fraction: float
    minimum_jacobian: float
    mean_displacement_native_voxels: float
    p95_displacement_native_voxels: float


@dataclass(frozen=True)
class InstanceOptimizationResult:
    flow_native_dzyx_voxels: torch.Tensor
    flow_normalized_dzyx: torch.Tensor
    diagnostics: Tuple[LevelDiagnostics, ...]
    align_corners: bool
    mapping: str = "fixed grid to moving sampling location"


def _check_flow(flow: torch.Tensor) -> None:
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError(f"Flow must be [B,3,D,H,W], got {tuple(flow.shape)}.")


def _normalization_factors(
    shape_dzyx: Sequence[int],
    *,
    align_corners: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    shape = tuple(int(value) for value in shape_dzyx)
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError("A positive 3-D shape is required.")
    if align_corners:
        factors = tuple(max(value - 1, 1) / 2.0 for value in shape)
    else:
        factors = tuple(value / 2.0 for value in shape)
    return torch.tensor(factors, device=device, dtype=dtype).view(1, 3, 1, 1, 1)


def native_voxel_to_normalized_dzyx(
    flow_native_dzyx: torch.Tensor,
    native_shape_dzyx: Optional[Sequence[int]] = None,
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """Convert native dzyx voxel displacement to grid_sample normalized dzyx."""

    _check_flow(flow_native_dzyx)
    shape = tuple(
        int(value)
        for value in (
            native_shape_dzyx
            if native_shape_dzyx is not None
            else flow_native_dzyx.shape[-3:]
        )
    )
    factors = _normalization_factors(
        shape,
        align_corners=align_corners,
        device=flow_native_dzyx.device,
        dtype=flow_native_dzyx.dtype,
    )
    return flow_native_dzyx / factors


def normalized_to_native_voxel_dzyx(
    flow_normalized_dzyx: torch.Tensor,
    native_shape_dzyx: Sequence[int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    """Convert grid_sample normalized dzyx displacement to native voxels."""

    _check_flow(flow_normalized_dzyx)
    factors = _normalization_factors(
        native_shape_dzyx,
        align_corners=align_corners,
        device=flow_normalized_dzyx.device,
        dtype=flow_normalized_dzyx.dtype,
    )
    return flow_normalized_dzyx * factors


def _base_grid(
    batch: int,
    shape_dzyx: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
    align_corners: bool,
) -> torch.Tensor:
    depth, height, width = (int(value) for value in shape_dzyx)
    if align_corners:
        axes = (
            torch.linspace(-1, 1, depth, device=device, dtype=dtype),
            torch.linspace(-1, 1, height, device=device, dtype=dtype),
            torch.linspace(-1, 1, width, device=device, dtype=dtype),
        )
    else:
        axes = (
            (torch.arange(depth, device=device, dtype=dtype) + 0.5) * (2.0 / depth) - 1.0,
            (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / height) - 1.0,
            (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / width) - 1.0,
        )
    z, y, x = torch.meshgrid(*axes, indexing="ij")
    return torch.stack((x, y, z), dim=-1)[None].expand(batch, -1, -1, -1, -1)


def _sampling_grid(
    flow_normalized_dzyx: torch.Tensor, *, align_corners: bool
) -> torch.Tensor:
    _check_flow(flow_normalized_dzyx)
    base = _base_grid(
        flow_normalized_dzyx.shape[0],
        flow_normalized_dzyx.shape[-3:],
        device=flow_normalized_dzyx.device,
        dtype=flow_normalized_dzyx.dtype,
        align_corners=align_corners,
    )
    delta_xyz = torch.stack(
        (
            flow_normalized_dzyx[:, 2],
            flow_normalized_dzyx[:, 1],
            flow_normalized_dzyx[:, 0],
        ),
        dim=-1,
    )
    return base + delta_xyz


def _warp_vector_field(
    field_dzyx: torch.Tensor,
    sampling_flow_normalized_dzyx: torch.Tensor,
    *,
    align_corners: bool,
) -> torch.Tensor:
    return F.grid_sample(
        field_dzyx,
        _sampling_grid(
            sampling_flow_normalized_dzyx, align_corners=align_corners
        ),
        mode="bilinear",
        padding_mode="border",
        align_corners=align_corners,
    )


def _exponentiate_stationary_velocity(
    velocity_normalized_dzyx: torch.Tensor,
    *,
    steps: int,
    align_corners: bool,
) -> torch.Tensor:
    """Scaling-and-squaring exp(v), retaining normalized dzyx units."""

    displacement = velocity_normalized_dzyx / float(2 ** int(steps))
    for _ in range(int(steps)):
        displacement = displacement + _warp_vector_field(
            displacement,
            displacement,
            align_corners=align_corners,
        )
    return displacement


def _compose_base_and_residual(
    base_flow_normalized_dzyx: torch.Tensor,
    residual_flow_normalized_dzyx: torch.Tensor,
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Apply residual first, then an optional fixed-to-moving base map."""

    return residual_flow_normalized_dzyx + _warp_vector_field(
        base_flow_normalized_dzyx,
        residual_flow_normalized_dzyx,
        align_corners=align_corners,
    )


def _parameter_to_displacement(
    parameter_normalized_dzyx: torch.Tensor,
    *,
    parameterization: str,
    scaling_squaring_steps: int,
    align_corners: bool,
) -> torch.Tensor:
    if parameterization == "dense":
        return parameter_normalized_dzyx
    return _exponentiate_stationary_velocity(
        parameter_normalized_dzyx,
        steps=scaling_squaring_steps,
        align_corners=align_corners,
    )


def _boundary_valid(grid_xyz: torch.Tensor, *, align_corners: bool) -> torch.Tensor:
    depth, height, width = grid_xyz.shape[1:4]
    if align_corners:
        limits_xyz = grid_xyz.new_tensor((1.0, 1.0, 1.0))
    else:
        limits_xyz = grid_xyz.new_tensor(
            (
                1.0 - 1.0 / width,
                1.0 - 1.0 / height,
                1.0 - 1.0 / depth,
            )
        )
    return (grid_xyz.abs() <= limits_xyz).all(dim=-1, keepdim=False).unsqueeze(1)


def _resize_mask(mask: torch.Tensor, size: Sequence[int]) -> torch.Tensor:
    return F.interpolate(mask.float(), size=tuple(size), mode="nearest") > 0.5


def _native_coordinate_steps(
    native_shape_dzyx: Sequence[int],
    level_shape_dzyx: Sequence[int],
    *,
    align_corners: bool,
) -> Tuple[float, float, float]:
    native = tuple(int(value) for value in native_shape_dzyx)
    level = tuple(int(value) for value in level_shape_dzyx)
    if align_corners:
        return tuple(
            max(native_size - 1, 1) / max(level_size - 1, 1)
            for native_size, level_size in zip(native, level)
        )
    return tuple(
        native_size / level_size
        for native_size, level_size in zip(native, level)
    )


def _forward_difference(
    value: torch.Tensor, spatial_dimension: int, coordinate_step: float
) -> torch.Tensor:
    current = value.narrow(
        spatial_dimension, 1, value.shape[spatial_dimension] - 1
    )
    previous = value.narrow(
        spatial_dimension, 0, value.shape[spatial_dimension] - 1
    )
    difference = (current - previous) / float(coordinate_step)
    tail = difference.select(
        spatial_dimension, difference.shape[spatial_dimension] - 1
    ).unsqueeze(spatial_dimension)
    return torch.cat((difference, tail), dim=spatial_dimension)


def _native_flow_derivatives(
    flow_normalized_dzyx: torch.Tensor,
    native_shape_dzyx: Sequence[int],
    *,
    align_corners: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flow_native = normalized_to_native_voxel_dzyx(
        flow_normalized_dzyx,
        native_shape_dzyx,
        align_corners=align_corners,
    )
    steps = _native_coordinate_steps(
        native_shape_dzyx,
        flow_normalized_dzyx.shape[-3:],
        align_corners=align_corners,
    )
    return tuple(
        _forward_difference(flow_native, dimension, step)
        for dimension, step in zip((2, 3, 4), steps)
    )


def _dimensionless_smoothness(
    flow_normalized_dzyx: torch.Tensor,
    native_shape_dzyx: Sequence[int],
    *,
    align_corners: bool,
) -> torch.Tensor:
    """Mean squared du_native_voxels / dx_native_voxels over the full field."""

    derivatives = _native_flow_derivatives(
        flow_normalized_dzyx,
        native_shape_dzyx,
        align_corners=align_corners,
    )
    return torch.stack(
        [value.square().mean() for value in derivatives]
    ).mean()


def _jacobian_determinant_from_normalized(
    flow_normalized_dzyx: torch.Tensor,
    native_shape_dzyx: Sequence[int],
    *,
    align_corners: bool,
) -> torch.Tensor:
    """det(I + du_native_voxels / dx_native_voxels) on the level grid."""

    dz, dy, dx = _native_flow_derivatives(
        flow_normalized_dzyx,
        native_shape_dzyx,
        align_corners=align_corners,
    )
    j00, j01, j02 = 1 + dz[:, 0], dy[:, 0], dx[:, 0]
    j10, j11, j12 = dz[:, 1], 1 + dy[:, 1], dx[:, 1]
    j20, j21, j22 = dz[:, 2], dy[:, 2], 1 + dx[:, 2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    ).unsqueeze(1)


def jacobian_determinant_native_voxel(
    flow_native_dzyx_voxels: torch.Tensor,
) -> torch.Tensor:
    """Jacobian determinant for a native-grid dzyx voxel displacement."""

    _check_flow(flow_native_dzyx_voxels)
    normalized = native_voxel_to_normalized_dzyx(
        flow_native_dzyx_voxels,
        flow_native_dzyx_voxels.shape[-3:],
        align_corners=True,
    )
    return _jacobian_determinant_from_normalized(
        normalized,
        flow_native_dzyx_voxels.shape[-3:],
        align_corners=True,
    )


def _objective(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    flow_normalized_dzyx: torch.Tensor,
    *,
    native_shape_dzyx: Sequence[int],
    align_corners: bool,
    jacobian_margin: float,
    invalid_similarity_cost: float,
    minimum_valid_voxels: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    grid = _sampling_grid(
        flow_normalized_dzyx, align_corners=align_corners
    )
    warped_moving = F.grid_sample(
        moving_descriptor,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    )
    warped_moving = F.normalize(warped_moving, dim=1, eps=1e-6)
    warped_moving_evidence = F.grid_sample(
        moving_mask.float(),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=align_corners,
    ).clamp_(0.0, 1.0)
    fixed_evidence = fixed_mask.float()
    fixed_count = fixed_evidence.sum()
    if int(fixed_count.item()) < int(minimum_valid_voxels):
        raise RuntimeError("The fixed mask has insufficient support at this level.")
    if int(moving_mask.sum().item()) < int(minimum_valid_voxels):
        raise RuntimeError("The moving mask has insufficient support at this level.")

    # Use a fixed denominator and charge lost overlap explicitly. This prevents
    # optimization from improving its score merely by moving hard voxels beyond
    # the moving foreground or image boundary. Bilinear mask sampling makes the
    # coverage term differentiable at both boundaries.
    evidence = fixed_evidence * warped_moving_evidence
    cosine_distance = 1.0 - (
        fixed_descriptor * warped_moving
    ).sum(dim=1, keepdim=True)
    unmatched = (fixed_evidence - evidence).clamp_min(0.0)
    similarity = (
        cosine_distance * evidence
        + float(invalid_similarity_cost) * unmatched
    ).sum() / fixed_count.clamp_min(1.0)
    valid = (
        fixed_mask
        & (warped_moving_evidence > 0.5)
        & _boundary_valid(grid, align_corners=align_corners)
    )
    regularization = _dimensionless_smoothness(
        flow_normalized_dzyx,
        native_shape_dzyx,
        align_corners=align_corners,
    )
    determinant = _jacobian_determinant_from_normalized(
        flow_normalized_dzyx,
        native_shape_dzyx,
        align_corners=align_corners,
    )
    jacobian_penalty = F.relu(
        float(jacobian_margin) - determinant
    ).square().mean()
    return (
        similarity,
        regularization,
        jacobian_penalty,
        valid,
        determinant,
        evidence,
    )

def _level_shape(native_shape: Sequence[int], scale: float) -> Tuple[int, int, int]:
    return tuple(
        min(int(size), max(2, int(round(int(size) * float(scale)))))
        for size in native_shape
    )


def _prepare_mask(
    mask: Optional[torch.Tensor],
    descriptor: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            (descriptor.shape[0], 1, *descriptor.shape[-3:]),
            device=descriptor.device,
            dtype=torch.bool,
        )
    if mask.shape != (descriptor.shape[0], 1, *descriptor.shape[-3:]):
        raise ValueError(f"{name} must be [B,1,D,H,W] on the descriptor grid.")
    return mask.detach().to(device=descriptor.device).bool().clone()


def optimize_masked_descriptor_flow(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: Optional[torch.Tensor] = None,
    moving_mask: Optional[torch.Tensor] = None,
    initial_flow: Optional[torch.Tensor] = None,
    initial_flow_units: str = "native_voxels",
    config: Optional[InstanceOptimizationConfig] = None,
) -> InstanceOptimizationResult:
    """Optimize a dense fixed-to-moving sampling displacement.

    The optimized parameter is normalized dzyx. By default it is a stationary
    velocity exponentiated with scaling-and-squaring; ``parameterization=dense``
    retains a direct-displacement ablation. ``initial_flow`` is optional and may
    use native dzyx voxels or normalized dzyx. A learned residual is composed
    before this fixed base map. Pyramid interpolation never rescales normalized
    displacement or velocity values.
    """

    selected_config = config or InstanceOptimizationConfig()
    if (
        fixed_descriptor.ndim != 5
        or moving_descriptor.shape != fixed_descriptor.shape
        or fixed_descriptor.shape[1] < 1
    ):
        raise ValueError("Descriptors must be equal [B,C,D,H,W] tensors.")
    if not torch.isfinite(fixed_descriptor).all() or not torch.isfinite(
        moving_descriptor
    ).all():
        raise ValueError("Descriptors contain non-finite values.")
    native_shape = tuple(int(value) for value in fixed_descriptor.shape[-3:])
    if any(value < 2 for value in native_shape):
        raise ValueError("Every descriptor dimension must contain at least two voxels.")
    if initial_flow_units not in {"native_voxels", "normalized"}:
        raise ValueError("initial_flow_units must be native_voxels or normalized.")

    with torch.inference_mode(False):
        fixed_full = fixed_descriptor.detach().float().clone()
        moving_full = moving_descriptor.detach().float().clone()
        fixed_support = _prepare_mask(fixed_mask, fixed_full, name="fixed_mask")
        moving_support = _prepare_mask(moving_mask, moving_full, name="moving_mask")

        base_flow_full = None
        if initial_flow is not None:
            _check_flow(initial_flow)
            if (
                initial_flow.shape[0] != fixed_full.shape[0]
                or tuple(initial_flow.shape[-3:]) != native_shape
            ):
                raise ValueError("initial_flow must be on the native descriptor grid.")
            base_flow_full = initial_flow.detach().to(
                device=fixed_full.device, dtype=torch.float32
            ).clone()
            if not torch.isfinite(base_flow_full).all():
                raise ValueError("initial_flow contains non-finite values.")
            if initial_flow_units == "native_voxels":
                base_flow_full = native_voxel_to_normalized_dzyx(
                    base_flow_full,
                    native_shape,
                    align_corners=selected_config.align_corners,
                )

        parameter_state = None
        normalized_flow = None
        diagnostics = []
        for level_index, (
            scale,
            iteration_count,
            learning_rate,
            smoothness_weight,
        ) in enumerate(
            zip(
                selected_config.scales,
                selected_config.iterations,
                selected_config.learning_rates,
                selected_config.smoothness_weights,
            )
        ):
            shape = _level_shape(native_shape, float(scale))
            fixed_level = F.normalize(
                F.interpolate(
                    fixed_full,
                    size=shape,
                    mode="trilinear",
                    align_corners=selected_config.align_corners,
                ),
                dim=1,
                eps=1e-6,
            )
            moving_level = F.normalize(
                F.interpolate(
                    moving_full,
                    size=shape,
                    mode="trilinear",
                    align_corners=selected_config.align_corners,
                ),
                dim=1,
                eps=1e-6,
            )
            fixed_mask_level = _resize_mask(fixed_support, shape)
            moving_mask_level = _resize_mask(moving_support, shape)
            base_flow_level = (
                None
                if base_flow_full is None
                else F.interpolate(
                    base_flow_full,
                    size=shape,
                    mode="trilinear",
                    align_corners=selected_config.align_corners,
                )
            )
            if parameter_state is None:
                level_initial = fixed_level.new_zeros(
                    (fixed_level.shape[0], 3, *shape)
                )
            else:
                level_initial = F.interpolate(
                    parameter_state,
                    size=shape,
                    mode="trilinear",
                    align_corners=selected_config.align_corners,
                )
            flow_parameter = torch.nn.Parameter(level_initial)
            optimizer = torch.optim.Adam(
                [flow_parameter], lr=float(learning_rate)
            )

            def effective_flow() -> torch.Tensor:
                residual = _parameter_to_displacement(
                    flow_parameter,
                    parameterization=selected_config.parameterization,
                    scaling_squaring_steps=selected_config.scaling_squaring_steps,
                    align_corners=selected_config.align_corners,
                )
                if base_flow_level is None:
                    return residual
                return _compose_base_and_residual(
                    base_flow_level,
                    residual,
                    align_corners=selected_config.align_corners,
                )

            with torch.no_grad():
                initial_similarity, _, _, _, _, _ = _objective(
                    fixed_level,
                    moving_level,
                    fixed_mask_level,
                    moving_mask_level,
                    effective_flow(),
                    native_shape_dzyx=native_shape,
                    align_corners=selected_config.align_corners,
                    jacobian_margin=selected_config.jacobian_margin,
                    invalid_similarity_cost=selected_config.invalid_similarity_cost,
                    minimum_valid_voxels=selected_config.minimum_valid_voxels,
                )
            for _ in range(int(iteration_count)):
                current_flow = effective_flow()
                similarity, regularization, jacobian_penalty, _, _, _ = _objective(
                    fixed_level,
                    moving_level,
                    fixed_mask_level,
                    moving_mask_level,
                    current_flow,
                    native_shape_dzyx=native_shape,
                    align_corners=selected_config.align_corners,
                    jacobian_margin=selected_config.jacobian_margin,
                    invalid_similarity_cost=selected_config.invalid_similarity_cost,
                    minimum_valid_voxels=selected_config.minimum_valid_voxels,
                )
                total = (
                    similarity
                    + float(smoothness_weight) * regularization
                    + float(selected_config.jacobian_weight) * jacobian_penalty
                )
                if not torch.isfinite(total):
                    raise FloatingPointError("Non-finite instance-optimization loss.")
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()

            with torch.no_grad():
                final_effective_flow = effective_flow()
                (
                    final_similarity,
                    final_regularization,
                    final_jacobian_penalty,
                    final_valid,
                    final_determinant,
                    final_evidence,
                ) = _objective(
                    fixed_level,
                    moving_level,
                    fixed_mask_level,
                    moving_mask_level,
                    final_effective_flow,
                    native_shape_dzyx=native_shape,
                    align_corners=selected_config.align_corners,
                    jacobian_margin=selected_config.jacobian_margin,
                    invalid_similarity_cost=selected_config.invalid_similarity_cost,
                    minimum_valid_voxels=selected_config.minimum_valid_voxels,
                )
                native_units_at_level = normalized_to_native_voxel_dzyx(
                    final_effective_flow,
                    native_shape,
                    align_corners=selected_config.align_corners,
                )
                magnitude = native_units_at_level.square().sum(
                    dim=1, keepdim=True
                ).sqrt()
                # QA is measured over the fixed foreground, not only surviving
                # overlap; otherwise out-of-FOV displacement would be hidden.
                selected_magnitude = magnitude[fixed_mask_level].float()
                fixed_count = fixed_mask_level.sum().clamp_min(1)
                diagnostics.append(
                    LevelDiagnostics(
                        level_index=level_index,
                        scale=float(scale),
                        shape_dzyx=shape,
                        iterations=int(iteration_count),
                        learning_rate_normalized=float(learning_rate),
                        smoothness_weight_dimensionless=float(smoothness_weight),
                        initial_similarity_1_minus_cosine=float(
                            initial_similarity.cpu()
                        ),
                        final_similarity_1_minus_cosine=float(
                            final_similarity.cpu()
                        ),
                        final_regularization_dimensionless=float(
                            final_regularization.cpu()
                        ),
                        final_jacobian_penalty=float(
                            final_jacobian_penalty.cpu()
                        ),
                        final_total_loss=float(
                            (
                                final_similarity
                                + float(smoothness_weight) * final_regularization
                                + float(selected_config.jacobian_weight)
                                * final_jacobian_penalty
                            ).cpu()
                        ),
                        valid_fraction_of_fixed=float(
                            (final_evidence.sum() / fixed_count).cpu()
                        ),
                        fold_fraction=float(
                            (final_determinant <= 0).float().mean().cpu()
                        ),
                        minimum_jacobian=float(final_determinant.min().cpu()),
                        mean_displacement_native_voxels=float(
                            selected_magnitude.mean().cpu()
                        ),
                        p95_displacement_native_voxels=float(
                            torch.quantile(selected_magnitude, 0.95).cpu()
                        ),
                    )
                )
            parameter_state = flow_parameter.detach()
            normalized_flow = final_effective_flow.detach()

        assert normalized_flow is not None
        if tuple(normalized_flow.shape[-3:]) != native_shape:
            raise AssertionError("Final pyramid level did not preserve native shape.")
        native_flow = normalized_to_native_voxel_dzyx(
            normalized_flow,
            native_shape,
            align_corners=selected_config.align_corners,
        )
        return InstanceOptimizationResult(
            flow_native_dzyx_voxels=native_flow.detach(),
            flow_normalized_dzyx=normalized_flow.detach(),
            diagnostics=tuple(diagnostics),
            align_corners=selected_config.align_corners,
        )
