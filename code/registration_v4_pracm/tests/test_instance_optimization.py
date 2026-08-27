"""Regression tests for the independent normalized-grid instance optimizer."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from registration_v4_pracm.inference.instance_optimization import (
    InstanceOptimizationConfig,
    native_voxel_to_normalized_dzyx,
    normalized_to_native_voxel_dzyx,
    optimize_masked_descriptor_flow,
)


def test_normalized_native_roundtrip_is_exact_for_both_grid_conventions():
    shape = (13, 17, 65)
    generator = torch.Generator().manual_seed(20260826)
    native = torch.randn((1, 3, *shape), generator=generator)
    for align_corners in (True, False):
        normalized = native_voxel_to_normalized_dzyx(
            native, shape, align_corners=align_corners
        )
        restored = normalized_to_native_voxel_dzyx(
            normalized, shape, align_corners=align_corners
        )
        assert torch.allclose(restored, native, atol=1e-6, rtol=1e-6)


def _translation_case(device: torch.device, *, translation_x: float = 20.0):
    depth, height, width = 12, 16, 65
    x = torch.arange(width, device=device, dtype=torch.float32)
    angular_frequency = 2.0 * torch.pi / 128.0
    fixed_line = torch.stack(
        (
            torch.sin(angular_frequency * x),
            torch.cos(angular_frequency * x),
        ),
        dim=0,
    )
    moving_line = torch.stack(
        (
            torch.sin(angular_frequency * (x - translation_x)),
            torch.cos(angular_frequency * (x - translation_x)),
        ),
        dim=0,
    )
    fixed = fixed_line[None, :, None, None].expand(
        1, 2, depth, height, width
    ).contiguous()
    moving = moving_line[None, :, None, None].expand_as(fixed).contiguous()
    fixed = F.normalize(fixed, dim=1)
    moving = F.normalize(moving, dim=1)

    fixed_mask = torch.zeros(
        (1, 1, depth, height, width), device=device, dtype=torch.bool
    )
    moving_mask = torch.zeros_like(fixed_mask)
    fixed_mask[..., 2:-2, 2:-2, 4 : width - int(translation_x) - 4] = True
    moving_mask[..., 1:-1, 1:-1, 4:-4] = True

    config = InstanceOptimizationConfig(
        scales=(0.25, 0.5, 1.0),
        iterations=(60, 40, 25),
        learning_rates=(0.08, 0.04, 0.02),
        smoothness_weights=(0.02, 0.01, 0.005),
        align_corners=True,
        minimum_valid_voxels=8,
    )
    result = optimize_masked_descriptor_flow(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=config,
    )
    estimated = result.flow_native_dzyx_voxels[:, 2:3][fixed_mask]
    mean_x = float(estimated.mean().cpu())
    p95_magnitude = result.diagnostics[-1].p95_displacement_native_voxels
    assert abs(mean_x - translation_x) < 1.5
    assert p95_magnitude > 15.0
    assert result.diagnostics[-1].mean_displacement_native_voxels > 15.0
    assert result.diagnostics[-1].valid_fraction_of_fixed > 0.95
    assert result.diagnostics[-1].fold_fraction == 0.0
    assert result.diagnostics[-1].minimum_jacobian > 0.0
    assert (
        result.diagnostics[-1].final_similarity_1_minus_cosine
        < result.diagnostics[0].initial_similarity_1_minus_cosine
    )
    return result


def test_cpu_recovers_twenty_voxel_translation_without_five_voxel_cap():
    _translation_case(torch.device("cpu"))


def test_cuda_recovers_twenty_voxel_translation_when_available():
    if not torch.cuda.is_available():
        return
    _translation_case(torch.device("cuda"))


def test_synthetic_nonrigid_target_improves_without_folding():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    depth, height, width = 10, 12, 49
    x = torch.arange(width, device=device, dtype=torch.float32)
    deformation_amplitude = 5.0
    deformation_frequency = 2.0 * torch.pi / float(width - 1)
    target_x = deformation_amplitude * torch.sin(deformation_frequency * x)

    # g(x)=x+u(x) is diffeomorphic because min g'(x)>0.34. Build the
    # moving descriptor analytically from g^{-1}, so moving(g(x))=fixed(x).
    inverse = x.clone()
    for _ in range(16):
        residual = (
            inverse
            + deformation_amplitude
            * torch.sin(deformation_frequency * inverse)
            - x
        )
        derivative = (
            1.0
            + deformation_amplitude
            * deformation_frequency
            * torch.cos(deformation_frequency * inverse)
        )
        inverse = inverse - residual / derivative

    descriptor_frequency = 2.0 * torch.pi / 96.0
    fixed_line = torch.stack(
        (
            torch.sin(descriptor_frequency * x),
            torch.cos(descriptor_frequency * x),
        ),
        dim=0,
    )
    moving_line = torch.stack(
        (
            torch.sin(descriptor_frequency * inverse),
            torch.cos(descriptor_frequency * inverse),
        ),
        dim=0,
    )
    fixed = fixed_line[None, :, None, None].expand(
        1, 2, depth, height, width
    ).contiguous()
    moving = moving_line[None, :, None, None].expand_as(fixed).contiguous()
    fixed = F.normalize(fixed, dim=1)
    moving = F.normalize(moving, dim=1)
    mask = torch.zeros(
        (1, 1, depth, height, width), device=device, dtype=torch.bool
    )
    mask[..., 2:-2, 2:-2, 4:-4] = True

    result = optimize_masked_descriptor_flow(
        fixed,
        moving,
        fixed_mask=mask,
        moving_mask=mask,
        config=InstanceOptimizationConfig(
            scales=(0.25, 0.5, 1.0),
            iterations=(70, 45, 30),
            learning_rates=(0.07, 0.035, 0.015),
            smoothness_weights=(0.03, 0.02, 0.01),
            jacobian_weight=0.20,
            jacobian_margin=0.10,
            align_corners=True,
        ),
    )
    target = target_x.view(1, 1, 1, 1, width).expand_as(mask)
    estimated = result.flow_native_dzyx_voxels[:, 2:3]
    identity_error = target[mask].abs().mean()
    final_error = (estimated[mask] - target[mask]).abs().mean()
    determinant = result.diagnostics[-1]
    assert float(final_error.cpu()) < 0.6 * float(identity_error.cpu())
    assert determinant.fold_fraction <= 1.0e-4
    assert determinant.minimum_jacobian > 0.0
    assert (
        determinant.final_similarity_1_minus_cosine
        < result.diagnostics[0].initial_similarity_1_minus_cosine
    )
    return result
