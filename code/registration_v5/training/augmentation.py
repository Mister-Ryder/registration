"""Geometry/appearance-decoupled synthetic correspondence generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..augmentation_config import AugmentationConfig
from ..ops.spatial import compose_flows, warp


@dataclass
class SyntheticPair3D:
    fixed: torch.Tensor
    moving: torch.Tensor
    ground_truth_flow: torch.Tensor
    fixed_domain: torch.Tensor
    moving_domain: torch.Tensor


@dataclass
class AppearanceViews3D:
    first: torch.Tensor
    second: torch.Tensor
    domain: torch.Tensor


def _smooth(value: torch.Tensor, passes: int = 3) -> torch.Tensor:
    for _ in range(passes):
        value = F.avg_pool3d(
            F.pad(value, (1, 1, 1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )
    return value


def random_diffeomorphic_flow(
    reference: torch.Tensor,
    config: AugmentationConfig,
) -> torch.Tensor:
    """Integrate a random smooth stationary velocity by scaling and squaring."""

    if reference.ndim != 5:
        raise ValueError("reference must be [B,C,D,H,W].")
    batch = reference.shape[0]
    control = torch.randn(
        (batch, 3, *config.velocity_control_shape),
        device=reference.device,
        dtype=reference.dtype,
    )
    velocity = F.interpolate(
        control,
        size=reference.shape[-3:],
        mode="trilinear",
        align_corners=True,
    )
    velocity = _smooth(velocity)
    translation = torch.empty(
        (batch, 3, 1, 1, 1), device=reference.device, dtype=reference.dtype
    ).uniform_(-0.25, 0.25) * config.maximum_velocity_voxels
    velocity = velocity + translation
    norm = velocity.square().sum(dim=1, keepdim=True).sqrt().flatten(2).amax(dim=2)
    desired = torch.empty_like(norm).uniform_(0.35, 1.0) * config.maximum_velocity_voxels
    velocity = velocity * (desired / norm.clamp_min(1e-6)).view(batch, 1, 1, 1, 1)
    flow = velocity / float(2**config.integration_steps)
    for _ in range(config.integration_steps):
        flow = compose_flows(flow, flow, align_corners=True)
    return flow.detach()


def nonlinear_intensity_transform(
    image: torch.Tensor,
    config: AugmentationConfig,
) -> torch.Tensor:
    """Geometry-preserving monotonic remapping plus low-frequency gain field."""

    batch = image.shape[0]
    gamma = torch.empty((batch, 1, 1, 1, 1), device=image.device, dtype=image.dtype).uniform_(
        *config.gamma_range
    )
    gain = torch.empty_like(gamma).uniform_(*config.gain_range)
    normalized = image.clamp(0, 1)
    curve = torch.empty_like(gamma).uniform_(
        -config.piecewise_strength, config.piecewise_strength
    )
    # A smooth monotonic bend whose derivative stays positive for the frozen
    # strength range.  It supplies a different response law for every view.
    normalized = normalized + curve * normalized * (1 - normalized) * (2 * normalized - 1)
    transformed = gain * normalized.clamp_min(1e-6).pow(gamma)
    field = torch.randn(
        (batch, 1, 3, 4, 4), device=image.device, dtype=image.dtype
    )
    field = F.interpolate(field, size=image.shape[-3:], mode="trilinear", align_corners=True)
    field = torch.tanh(_smooth(field, passes=2)) * config.bias_field_strength
    transformed = transformed * (1 + field)
    if config.noise_std > 0:
        transformed = transformed + torch.randn_like(transformed) * config.noise_std
    invert = torch.rand(
        (batch, 1, 1, 1, 1), device=image.device
    ) < config.inversion_probability
    transformed = torch.where(invert, 1 - transformed, transformed)
    return transformed.clamp(0, 1)


def make_appearance_views(
    image: torch.Tensor,
    config: AugmentationConfig,
    domain: Optional[torch.Tensor] = None,
) -> AppearanceViews3D:
    """Two independently rendered views with exactly zero geometric displacement."""

    support = torch.ones_like(image, dtype=torch.bool) if domain is None else domain.bool()
    if support.shape != image.shape:
        raise ValueError("Appearance-view domain must match the image shape.")
    first = nonlinear_intensity_transform(image, config).masked_fill(~support, 0)
    second = nonlinear_intensity_transform(image, config).masked_fill(~support, 0)
    return AppearanceViews3D(first, second, support)

def make_synthetic_pair(
    image: torch.Tensor,
    config: AugmentationConfig,
    domain: Optional[torch.Tensor] = None,
) -> SyntheticPair3D:
    """Independent appearance views around an exact fixed->moving deformation."""

    moving_domain = (
        torch.ones_like(image, dtype=torch.bool) if domain is None else domain.bool()
    )
    if moving_domain.shape != image.shape:
        raise ValueError("Synthetic source domain must have the same shape as image.")
    flow = random_diffeomorphic_flow(image, config)
    fixed, valid = warp(
        image,
        flow,
        padding_mode="zeros",
        align_corners=True,
        return_valid=True,
    )
    sampled_domain = warp(
        moving_domain.float(),
        flow,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    ) > 0.5
    valid &= sampled_domain
    moving = nonlinear_intensity_transform(image, config)
    fixed = nonlinear_intensity_transform(fixed, config)
    fixed = fixed.masked_fill(~valid, 0)
    return SyntheticPair3D(
        fixed=fixed,
        moving=moving.masked_fill(~moving_domain, 0),
        ground_truth_flow=flow,
        fixed_domain=valid,
        moving_domain=moving_domain,
    )
