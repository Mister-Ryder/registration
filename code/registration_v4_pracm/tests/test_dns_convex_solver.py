"""Regression tests for the descriptor-agnostic DNS ConvexAdam solver."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from registration_v4_pracm.inference.dns_convex_solver import (
    DNSConvexConfig,
    solve_dns_convex,
)


def _coordinate_descriptor(shape: tuple[int, int, int], translation_x: float):
    depth, height, width = shape
    z = torch.arange(depth, dtype=torch.float32)
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    wz = 2.0 * torch.pi / 73.0
    wy = 2.0 * torch.pi / 97.0
    wx1 = 2.0 * torch.pi / 137.0
    wx2 = 2.0 * torch.pi / 53.0
    wx3 = 2.0 * torch.pi / 29.0
    wx4 = 2.0 * torch.pi / 19.0

    def build(x_coordinate: torch.Tensor) -> torch.Tensor:
        channels = (
            torch.sin(wz * z)[:, None, None].expand(depth, height, width),
            torch.cos(wz * z)[:, None, None].expand(depth, height, width),
            torch.sin(wy * y)[None, :, None].expand(depth, height, width),
            torch.cos(wy * y)[None, :, None].expand(depth, height, width),
            torch.sin(wx1 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.cos(wx1 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.sin(wx2 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.cos(wx2 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.sin(wx3 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.cos(wx3 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.sin(wx4 * x_coordinate)[None, None, :].expand(depth, height, width),
            torch.cos(wx4 * x_coordinate)[None, None, :].expand(depth, height, width),
        )
        return F.normalize(torch.stack(channels, dim=0)[None], dim=1)

    fixed = build(x)
    moving = build(x - float(translation_x))
    fixed_mask = torch.zeros((1, 1, *shape), dtype=torch.bool)
    moving_mask = torch.zeros_like(fixed_mask)
    shift = int(round(translation_x))
    fixed_mask[..., 3:-3, 3:-3, 3 : width - shift - 3] = True
    moving_mask[..., 3:-3, 3:-3, 3 + shift : width - 3] = True
    return fixed, moving, fixed_mask, moving_mask


def test_dns_convex_recovers_twenty_voxel_translation_with_low_folding():
    translation = 20.0
    fixed, moving, fixed_mask, moving_mask = _coordinate_descriptor(
        (30, 36, 72), translation
    )
    result = solve_dns_convex(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=DNSConvexConfig(),
    )
    estimated_x = result.flow_native_dzyx_voxels[:, 2:3][fixed_mask]
    assert abs(float(estimated_x.mean()) - translation) < 2.5
    assert float((estimated_x - translation).abs().mean()) < 2.5
    assert result.diagnostics.search_radius_native_voxels == 24
    assert result.diagnostics.p95_displacement_native_voxels > 15.0
    assert result.diagnostics.fold_fraction <= 1.0e-4
    assert result.diagnostics.minimum_jacobian > 0.0
    assert result.diagnostics.valid_fraction_of_fixed > 0.90
    assert result.diagnostics.refined_similarity < 0.5
    assert result.diagnostics.refinement_accepted
    return result
