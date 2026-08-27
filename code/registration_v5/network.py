"""Authoritative V4-final network and faithful first-stage objective."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dns.faithful_v2 import (
    FullResolutionDSIRExtractor,
    MASRNetFaithfulV2,
    stochastic_nonlinear_transform_faithful_v2,
)
from .augmentation_config import AugmentationConfig
from .ops.spatial import warp
from .training.augmentation import random_diffeomorphic_flow

from .convex_solver_v2 import descriptor_convex_adam_v2
from .losses import anatomy_correspondence_loss
from .protocol import ConvexAdamProtocol, DescriptorProtocol, RepresentationTrainingProtocol


@dataclass(frozen=True)
class RepresentationResult:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    task: str


@dataclass(frozen=True)
class FinalRegistrationResult:
    flow_dzyx_voxels: torch.Tensor
    diagnostics: Dict[str, object]


class V4FinalRegistrationModel(nn.Module):
    """Full-resolution DSIR; the only public register path is ConvexAdam."""

    def __init__(self, descriptor: Optional[DescriptorProtocol] = None) -> None:
        super().__init__()
        selected = descriptor or DescriptorProtocol()
        self.extractor = FullResolutionDSIRExtractor(
            MASRNetFaithfulV2(
                descriptor_channels=selected.descriptor_channels,
                dns_dilation=selected.dns_dilation_voxels,
            ),
            normalize_output=True,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.extractor(image)

    def register(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        solver: ConvexAdamProtocol,
        descriptor_autocast_dtype: Optional[torch.dtype] = None,
    ) -> FinalRegistrationResult:
        enabled = fixed.is_cuda and descriptor_autocast_dtype is not None
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=descriptor_autocast_dtype or torch.float16,
            enabled=enabled,
        ):
            fixed_descriptor = self(fixed)
            moving_descriptor = self(moving)
        result = descriptor_convex_adam_v2(
            F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6),
            F.normalize(moving_descriptor.float(), dim=1, eps=1e-6),
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=solver,
        )
        return FinalRegistrationResult(result.flow_dzyx_voxels, result.diagnostics)


def _geometry_pair(
    image: torch.Tensor,
    mask: torch.Tensor,
    training: RepresentationTrainingProtocol,
    generator: Optional[torch.Generator],
):
    augmentation = AugmentationConfig(
        maximum_velocity_voxels=training.maximum_velocity_voxels,
        velocity_control_shape=training.velocity_control_shape,
        integration_steps=training.integration_steps,
    )
    flow = random_diffeomorphic_flow(image, augmentation)
    fixed_base, valid = warp(
        image, flow, padding_mode="zeros", align_corners=True, return_valid=True
    )
    valid &= warp(
        mask.float(), flow, mode="nearest", padding_mode="zeros", align_corners=True
    ) > 0.5
    fixed = stochastic_nonlinear_transform_faithful_v2(
        fixed_base, generator=generator
    ).masked_fill(~valid, 0)
    moving = stochastic_nonlinear_transform_faithful_v2(
        image, generator=generator
    ).masked_fill(~mask, 0)
    return fixed, moving, valid, flow


def faithful_representation_objective(
    model,
    image: torch.Tensor,
    mask: torch.Tensor,
    training: RepresentationTrainingProtocol,
    *,
    task: str,
    generator: Optional[torch.Generator] = None,
) -> RepresentationResult:
    """Paper-faithful original-vs-one-aug view; geometry is optional ablation."""

    if task == "appearance":
        first = image.masked_fill(~mask, 0)
        second = stochastic_nonlinear_transform_faithful_v2(
            image, generator=generator
        ).masked_fill(~mask, 0)
        support = mask
        first_descriptor = model(first)
        second_descriptor = model(second)
    elif task == "geometry":
        fixed, moving, support, flow = _geometry_pair(image, mask, training, generator)
        first_descriptor = model(fixed)
        second_descriptor = F.normalize(
            warp(model(moving), flow, padding_mode="zeros", align_corners=True),
            dim=1,
            eps=1e-6,
        )
    else:
        raise ValueError("Representation task must be appearance or geometry.")
    diagnostic = anatomy_correspondence_loss(
        first_descriptor,
        second_descriptor,
        support,
        samples=training.samples_per_view,
        temperature=training.contrastive_temperature,
        chunk_size=training.contrastive_chunk_size,
        variance_floor=training.variance_floor,
        generator=generator,
    )
    total = diagnostic.loss + training.variance_weight * diagnostic.variance_penalty
    return RepresentationResult(
        loss=total,
        task=task,
        metrics={
            "total": total,
            "contrastive": diagnostic.loss,
            "variance": diagnostic.variance_penalty,
            "positive_cosine": diagnostic.positive_cosine.detach(),
            "top1": diagnostic.top1_accuracy.detach(),
            "sampled_locations": total.new_tensor(float(diagnostic.sampled_locations)),
        },
    )


__all__ = [
    "FinalRegistrationResult",
    "RepresentationResult",
    "V4FinalRegistrationModel",
    "faithful_representation_objective",
]
