"""Mask-aware descriptor ConvexAdam adapted from the official B02 core.

The algorithmic chain is deliberately unchanged: discrete SSD correlation,
coupled-convex regularisation, forward/backward inverse consistency, then Adam
instance refinement with diffusion regularisation.  The only semantic change
is that callers supply arbitrary dense descriptors instead of MIND-SSC.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .protocol import ConvexAdamProtocol


@dataclass(frozen=True)
class DescriptorConvexAdamResult:
    flow_dzyx_voxels: torch.Tensor
    diagnostics: Dict[str, object]


def _check_inputs(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
) -> None:
    if fixed.ndim != 5 or moving.shape != fixed.shape or fixed.shape[0] != 1:
        raise ValueError("Descriptors must be equal [1,C,D,H,W] tensors.")
    expected_mask = (1, 1, *fixed.shape[-3:])
    if fixed_mask.shape != expected_mask or moving_mask.shape != expected_mask:
        raise ValueError("Raw masks must be [1,1,D,H,W] on the descriptor grid.")
    if not torch.isfinite(fixed).all() or not torch.isfinite(moving).all():
        raise ValueError("Descriptors contain non-finite values.")


def _masked_correlate(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    fixed_valid: torch.Tensor,
    moving_valid: torch.Tensor,
    *,
    displacement_half_width: int,
    grid_spacing: int,
    native_shape: Tuple[int, int, int],
    invalid_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Official 3-D unfold correlation with invalid candidates excluded."""

    depth, height, width = native_shape
    radius = int(displacement_half_width)
    size = 2 * radius + 1
    channels = fixed.shape[1]
    with torch.no_grad():
        moving_unfold = F.unfold(
            F.pad(moving, (radius, radius, radius, radius, radius, radius)).squeeze(0),
            size,
        )
        moving_unfold = moving_unfold.view(
            channels, -1, size**2, height // grid_spacing, width // grid_spacing
        )
        valid_unfold = F.unfold(
            F.pad(moving_valid.float(), (radius, radius, radius, radius, radius, radius)).squeeze(0),
            size,
        ).view(1, -1, size**2, height // grid_spacing, width // grid_spacing)

    ssd = fixed.new_zeros(
        (size**3, depth // grid_spacing, height // grid_spacing, width // grid_spacing)
    )
    fixed_layout = fixed.permute(1, 2, 0, 3, 4)
    fixed_valid_layout = fixed_valid.permute(1, 2, 0, 3, 4)
    with torch.no_grad():
        for offset in range(size):
            difference = fixed_layout - moving_unfold[:, offset : offset + depth // grid_spacing]
            cost = difference.square().sum(0, keepdim=True)
            candidate_valid = (
                fixed_valid_layout
                & (valid_unfold[:, offset : offset + depth // grid_spacing] > 0.5)
            )
            cost = torch.where(
                candidate_valid,
                cost,
                cost.new_full((), float(invalid_penalty) * max(channels, 1)),
            )
            # Outside fixed anatomy, let the coupled/diffusion terms extend the
            # nearby field instead of treating background equality as evidence.
            cost = torch.where(fixed_valid_layout, cost, torch.zeros_like(cost))
            smoothed = F.avg_pool3d(
                F.avg_pool3d(cost.transpose(2, 1), 3, stride=1, padding=1),
                3,
                stride=1,
                padding=1,
            ).squeeze(1)
            ssd[offset::size] = smoothed
        ssd = (
            ssd.view(
                size,
                size,
                size,
                depth // grid_spacing,
                height // grid_spacing,
                width // grid_spacing,
            )
            .transpose(1, 0)
            .reshape(size**3, depth // grid_spacing, height // grid_spacing, width // grid_spacing)
        )
        argmin = torch.argmin(ssd, dim=0)
    return ssd, argmin


def _coupled_convex(
    ssd: torch.Tensor,
    argmin: torch.Tensor,
    displacement_mesh: torch.Tensor,
    *,
    grid_spacing: int,
    native_shape: Tuple[int, int, int],
) -> torch.Tensor:
    depth, height, width = native_shape
    low_shape = (
        depth // grid_spacing,
        height // grid_spacing,
        width // grid_spacing,
    )
    field = F.avg_pool3d(
        displacement_mesh.view(3, -1)[:, argmin.reshape(-1)].reshape(1, 3, *low_shape),
        3,
        padding=1,
        stride=1,
    )
    coefficients = ssd.new_tensor((0.003, 0.01, 0.03, 0.1, 0.3, 1.0))
    for coefficient in coefficients:
        coupled_argmin = torch.zeros_like(argmin)
        with torch.no_grad():
            for depth_index in range(low_shape[0]):
                coupled = ssd[:, depth_index] + coefficient * (
                    displacement_mesh - field[:, :, depth_index].reshape(3, 1, -1)
                ).square().sum(0).view(-1, low_shape[1], low_shape[2])
                coupled_argmin[depth_index] = torch.argmin(coupled, dim=0)
        field = F.avg_pool3d(
            displacement_mesh.view(3, -1)[:, coupled_argmin.reshape(-1)].reshape(
                1, 3, *low_shape
            ),
            3,
            padding=1,
            stride=1,
        )
    return field


def _inverse_consistency(
    forward: torch.Tensor,
    backward: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _, depth, height, width = forward.shape
    with torch.no_grad():
        consistent_forward = forward.clone()
        consistent_backward = backward.clone()
        identity = F.affine_grid(
            torch.eye(3, 4, device=forward.device, dtype=forward.dtype)[None],
            (batch, 1, depth, height, width),
            align_corners=False,
        ).permute(0, 4, 1, 2, 3)
        for _ in range(int(iterations)):
            current_forward = consistent_forward.clone()
            current_backward = consistent_backward.clone()
            consistent_forward = 0.5 * (
                current_forward
                - F.grid_sample(
                    current_backward,
                    (identity + current_forward).permute(0, 2, 3, 4, 1),
                    align_corners=False,
                )
            )
            consistent_backward = 0.5 * (
                current_backward
                - F.grid_sample(
                    current_forward,
                    (identity + current_backward).permute(0, 2, 3, 4, 1),
                    align_corners=False,
                )
            )
    return consistent_forward, consistent_backward


def _jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
    dz, dy, dx = torch.gradient(flow.float(), dim=(2, 3, 4))
    j00, j01, j02 = 1 + dz[:, 0], dy[:, 0], dx[:, 0]
    j10, j11, j12 = dz[:, 1], 1 + dy[:, 1], dx[:, 1]
    j20, j21, j22 = dz[:, 2], dy[:, 2], 1 + dx[:, 2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    )


def descriptor_convex_adam(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: ConvexAdamProtocol,
) -> DescriptorConvexAdamResult:
    """Register fixed-grid descriptors to moving sampling coordinates."""

    _check_inputs(fixed_descriptor, moving_descriptor, fixed_mask, moving_mask)
    fixed = F.normalize(fixed_descriptor.detach(), dim=1, eps=1e-6)
    moving = F.normalize(moving_descriptor.detach(), dim=1, eps=1e-6)
    fixed = fixed * fixed_mask.to(fixed.dtype)
    moving = moving * moving_mask.to(moving.dtype)
    depth, height, width = (int(value) for value in fixed.shape[-3:])
    grid_spacing = int(config.grid_spacing)
    if min(depth, height, width) < 2 * grid_spacing:
        raise ValueError("Descriptor grid is too small for the selected ConvexAdam spacing.")
    search_dtype = torch.float16 if fixed.is_cuda else torch.float32
    fixed = fixed.to(search_dtype)
    moving = moving.to(search_dtype)
    with torch.no_grad():
        fixed_coarse = F.avg_pool3d(fixed, grid_spacing, stride=grid_spacing)
        moving_coarse = F.avg_pool3d(moving, grid_spacing, stride=grid_spacing)
        fixed_valid = F.avg_pool3d(
            fixed_mask.float(), grid_spacing, stride=grid_spacing
        ) >= float(config.mask_pool_threshold)
        moving_valid = F.avg_pool3d(
            moving_mask.float(), grid_spacing, stride=grid_spacing
        ) >= float(config.mask_pool_threshold)
    native_shape = (depth, height, width)
    ssd, argmin = _masked_correlate(
        fixed_coarse,
        moving_coarse,
        fixed_valid,
        moving_valid,
        displacement_half_width=config.displacement_half_width,
        grid_spacing=grid_spacing,
        native_shape=native_shape,
        invalid_penalty=config.invalid_candidate_penalty,
    )
    radius = int(config.displacement_half_width)
    displacement_mesh = F.affine_grid(
        radius * torch.eye(3, 4, device=fixed.device, dtype=search_dtype)[None],
        (1, 1, 2 * radius + 1, 2 * radius + 1, 2 * radius + 1),
        align_corners=True,
    ).permute(0, 4, 1, 2, 3).reshape(3, -1, 1)
    forward_coarse = _coupled_convex(
        ssd,
        argmin,
        displacement_mesh,
        grid_spacing=grid_spacing,
        native_shape=native_shape,
    )
    reverse_ssd = None
    if config.inverse_consistency:
        reverse_ssd, reverse_argmin = _masked_correlate(
            moving_coarse,
            fixed_coarse,
            moving_valid,
            fixed_valid,
            displacement_half_width=config.displacement_half_width,
            grid_spacing=grid_spacing,
            native_shape=native_shape,
            invalid_penalty=config.invalid_candidate_penalty,
        )
        backward_coarse = _coupled_convex(
            reverse_ssd,
            reverse_argmin,
            displacement_mesh,
            grid_spacing=grid_spacing,
            native_shape=native_shape,
        )
        normalization = fixed.new_tensor(
            (
                max(depth // grid_spacing - 1, 1),
                max(height // grid_spacing - 1, 1),
                max(width // grid_spacing - 1, 1),
            )
        ).view(1, 3, 1, 1, 1) / 2
        consistent, _ = _inverse_consistency(
            (forward_coarse / normalization).flip(1),
            (backward_coarse / normalization).flip(1),
            iterations=config.inverse_consistency_iterations,
        )
        full_flow = F.interpolate(
            consistent.flip(1) * normalization * grid_spacing,
            size=native_shape,
            mode="trilinear",
            align_corners=False,
        )
    else:
        full_flow = F.interpolate(
            forward_coarse * grid_spacing,
            size=native_shape,
            mode="trilinear",
            align_corners=False,
        )

    initial_flow = full_flow.float().detach()
    final_data_loss = float("nan")
    final_regularization = float("nan")
    if config.adam_iterations > 0 and config.diffusion_weight > 0:
        spacing = int(config.adam_grid_spacing)
        with torch.no_grad():
            patch_fixed = F.avg_pool3d(fixed.float(), spacing, stride=spacing)
            patch_moving = F.avg_pool3d(moving.float(), spacing, stride=spacing)
            patch_fixed_mask = F.avg_pool3d(
                fixed_mask.float(), spacing, stride=spacing
            ) >= float(config.mask_pool_threshold)
            patch_moving_mask = F.avg_pool3d(
                moving_mask.float(), spacing, stride=spacing
            ) >= float(config.mask_pool_threshold)
        low_shape = (depth // spacing, height // spacing, width // spacing)
        low_flow = F.interpolate(
            initial_flow, size=low_shape, mode="trilinear", align_corners=False
        )
        net = nn.Sequential(nn.Conv3d(3, 1, low_shape, bias=False)).to(fixed.device)
        net[0].weight.data.copy_(low_flow.cpu().data / spacing)
        net.to(fixed.device)
        optimizer = torch.optim.Adam(net.parameters(), lr=float(config.adam_learning_rate))
        identity = F.affine_grid(
            torch.eye(3, 4, device=fixed.device)[None],
            (1, 1, *low_shape),
            align_corners=False,
        )
        scale = fixed.new_tensor(
            tuple(max(value - 1, 1) / 2 for value in low_shape), dtype=torch.float32
        )[None]
        for _ in range(int(config.adam_iterations)):
            optimizer.zero_grad(set_to_none=True)
            displacement = F.avg_pool3d(
                F.avg_pool3d(
                    F.avg_pool3d(net[0].weight, 3, stride=1, padding=1),
                    3,
                    stride=1,
                    padding=1,
                ),
                3,
                stride=1,
                padding=1,
            ).permute(0, 2, 3, 4, 1)
            regularization = float(config.diffusion_weight) * (
                (displacement[0, :, 1:] - displacement[0, :, :-1]).square().mean()
                + (displacement[0, 1:] - displacement[0, :-1]).square().mean()
                + (displacement[0, :, :, 1:] - displacement[0, :, :, :-1]).square().mean()
            )
            sample_grid = identity + (displacement.reshape(-1, 3) / scale).flip(1).view(
                1, *low_shape, 3
            )
            sampled_moving = F.grid_sample(
                patch_moving,
                sample_grid,
                align_corners=False,
                mode="bilinear",
                padding_mode="zeros",
            )
            sampled_mask = F.grid_sample(
                patch_moving_mask.float(),
                sample_grid,
                align_corners=False,
                mode="nearest",
                padding_mode="zeros",
            ) > 0.5
            valid = patch_fixed_mask & sampled_mask
            if int(valid.sum().item()) < 8:
                raise RuntimeError("ConvexAdam refinement has insufficient masked overlap.")
            cost = (sampled_moving - patch_fixed).square().mean(dim=1, keepdim=True) * 12.0
            data_loss = cost[valid].mean()
            total = data_loss + regularization
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite descriptor ConvexAdam refinement loss.")
            total.backward()
            optimizer.step()
            final_data_loss = float(data_loss.detach().cpu())
            final_regularization = float(regularization.detach().cpu())
        fitted = displacement.detach().permute(0, 4, 1, 2, 3)
        full_flow = F.interpolate(
            fitted * spacing, size=native_shape, mode="trilinear", align_corners=False
        )

    if config.selected_smoothing > 0:
        kernel = int(config.selected_smoothing)
        if kernel % 2 == 0:
            kernel += 1
        padding = kernel // 2
        for _ in range(3):
            full_flow = F.avg_pool3d(
                full_flow, kernel, stride=1, padding=padding
            )
    flow = full_flow.float().detach()
    determinant = _jacobian_determinant(flow)
    magnitude = flow.square().sum(dim=1, keepdim=True).sqrt()
    selected = magnitude[fixed_mask]
    diagnostics: Dict[str, object] = {
        "schema": "descriptor_convexadam_diagnostics_v1",
        "flow_convention": "fixed grid to moving sampling location, dzyx native voxels",
        "algorithm_chain": [
            "discrete_ssd_correlation",
            "coupled_convex",
            "inverse_consistency" if config.inverse_consistency else "no_inverse_consistency",
            "adam_diffusion_refinement",
        ],
        "parameters": asdict(config),
        "fixed_foreground_fraction": float(fixed_mask.float().mean().cpu()),
        "moving_foreground_fraction": float(moving_mask.float().mean().cpu()),
        "coarse_cost_mean": float(ssd.float().mean().cpu()),
        "reverse_coarse_cost_mean": (
            float(reverse_ssd.float().mean().cpu()) if reverse_ssd is not None else None
        ),
        "adam_final_data_loss": final_data_loss,
        "adam_final_regularization": final_regularization,
        "fold_fraction": float((determinant <= 0).float().mean().cpu()),
        "minimum_jacobian": float(determinant.min().cpu()),
        "mean_displacement_voxels": float(selected.mean().cpu()),
        "p95_displacement_voxels": float(torch.quantile(selected.float(), 0.95).cpu()),
    }
    return DescriptorConvexAdamResult(flow, diagnostics)


__all__ = ["DescriptorConvexAdamResult", "descriptor_convex_adam"]

