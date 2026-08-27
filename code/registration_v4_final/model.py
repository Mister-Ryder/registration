"""One V4-final method: faithful full-resolution DSIR and robust real-pair IO."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from registration_benchmark.dns.faithful_v2 import (
    FullResolutionDSIRExtractor,
    MASRNetFaithfulV2,
    stochastic_nonlinear_transform_faithful_v2,
)
from registration_v4_pracm.config import AugmentationConfig
from registration_v4_pracm.inference.instance_optimization import (
    InstanceOptimizationConfig,
    InstanceOptimizationResult,
    optimize_masked_descriptor_flow,
)
from registration_v4_pracm.ops.spatial import warp
from registration_v4_pracm.training.augmentation import random_diffeomorphic_flow

from .config import DescriptorConfig, SolverConfig, TrainingConfig
from .losses import anatomy_correspondence_loss


@dataclass(frozen=True)
class RepresentationResult:
    loss: torch.Tensor
    metrics: Dict[str, torch.Tensor]
    task: str


@dataclass(frozen=True)
class RegistrationResult:
    flow_dzyx_voxels: torch.Tensor
    solver: InstanceOptimizationResult
    identity_cosine_distance: float
    fixed_descriptor: torch.Tensor
    moving_descriptor: torch.Tensor


class V4FinalModel(nn.Module):
    """Trainable DSIR extractor with an inference-only explicit solver."""

    def __init__(self, descriptor_config: Optional[DescriptorConfig] = None) -> None:
        super().__init__()
        selected = descriptor_config or DescriptorConfig()
        network = MASRNetFaithfulV2(
            descriptor_channels=selected.descriptor_channels,
            dns_dilation=selected.dns_dilation_voxels,
        )
        self.extractor = FullResolutionDSIRExtractor(network, normalize_output=True)
        self.descriptor_config = selected

    def forward(self, normalized_image: torch.Tensor) -> torch.Tensor:
        return self.extractor(normalized_image)

    @staticmethod
    def _masked_identity_distance(
        fixed: torch.Tensor,
        moving: torch.Tensor,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
    ) -> float:
        valid = fixed_mask.bool() & moving_mask.bool()
        if not valid.any():
            return float("nan")
        distance = 1.0 - (fixed.float() * moving.float()).sum(dim=1, keepdim=True)
        return float(distance[valid].mean().cpu())

    def register(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        solver_config: SolverConfig,
        descriptor_autocast_dtype: Optional[torch.dtype] = None,
        initial_flow: Optional[torch.Tensor] = None,
    ) -> RegistrationResult:
        if fixed.shape != moving.shape or fixed_mask.shape != fixed.shape or moving_mask.shape != moving.shape:
            raise ValueError("Images and raw masks must share [B,1,D,H,W].")
        device_type = fixed.device.type
        use_autocast = descriptor_autocast_dtype is not None and device_type == "cuda"
        with torch.no_grad():
            with torch.autocast(
                device_type="cuda",
                dtype=descriptor_autocast_dtype or torch.float16,
                enabled=use_autocast,
            ):
                fixed_descriptor = self(fixed)
                moving_descriptor = self(moving)
            # Each pyramid level is re-normalized inside the solver.  Keeping
            # the full-resolution contract explicit here prevents accidental
            # re-use of the old /4 structural head.
            fixed_descriptor = F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6)
            moving_descriptor = F.normalize(moving_descriptor.float(), dim=1, eps=1e-6)
        identity_distance = self._masked_identity_distance(
            fixed_descriptor, moving_descriptor, fixed_mask, moving_mask
        )
        solver = optimize_masked_descriptor_flow(
            fixed_descriptor,
            moving_descriptor,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            initial_flow=initial_flow,
            initial_flow_units="native_voxels",
            config=InstanceOptimizationConfig(
                scales=solver_config.scales,
                iterations=solver_config.iterations,
                learning_rates=solver_config.learning_rates,
                smoothness_weights=solver_config.smoothness_weights,
                jacobian_weight=solver_config.jacobian_weight,
                jacobian_margin=solver_config.jacobian_margin,
                align_corners=solver_config.align_corners,
                minimum_valid_voxels=solver_config.minimum_valid_voxels,
            ),
        )
        return RegistrationResult(
            flow_dzyx_voxels=solver.flow_native_dzyx_voxels,
            solver=solver,
            identity_cosine_distance=identity_distance,
            fixed_descriptor=fixed_descriptor,
            moving_descriptor=moving_descriptor,
        )


def _appearance_views(
    image: torch.Tensor,
    mask: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    first = stochastic_nonlinear_transform_faithful_v2(image, generator=generator)
    second = stochastic_nonlinear_transform_faithful_v2(image, generator=generator)
    return first.masked_fill(~mask, 0), second.masked_fill(~mask, 0), mask


def _geometry_views(
    image: torch.Tensor,
    mask: torch.Tensor,
    training: TrainingConfig,
    *,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    augmentation = AugmentationConfig(
        maximum_velocity_voxels=training.maximum_velocity_voxels,
        velocity_control_shape=training.velocity_control_shape,
        integration_steps=training.integration_steps,
    )
    ground_truth_flow = random_diffeomorphic_flow(image, augmentation)
    fixed_base, boundary = warp(
        image,
        ground_truth_flow,
        padding_mode="zeros",
        align_corners=True,
        return_valid=True,
    )
    fixed_mask = boundary & (
        warp(
            mask.float(),
            ground_truth_flow,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        )
        > 0.5
    )
    fixed = stochastic_nonlinear_transform_faithful_v2(fixed_base, generator=generator)
    moving = stochastic_nonlinear_transform_faithful_v2(image, generator=generator)
    return (
        fixed.masked_fill(~fixed_mask, 0),
        moving.masked_fill(~mask, 0),
        fixed_mask,
        ground_truth_flow,
    )


def representation_objective(
    model: V4FinalModel,
    image: torch.Tensor,
    mask: torch.Tensor,
    training: TrainingConfig,
    *,
    task: str,
    generator: Optional[torch.Generator] = None,
) -> RepresentationResult:
    """Train intensity invariance and known-flow equivariance as separate tasks."""

    if task == "appearance":
        first, second, support = _appearance_views(image, mask, generator=generator)
        first_descriptor = model(first)
        second_descriptor = model(second)
    elif task == "geometry":
        fixed, moving, support, ground_truth_flow = _geometry_views(
            image, mask, training, generator=generator
        )
        first_descriptor = model(fixed)
        moving_descriptor = model(moving)
        second_descriptor = warp(
            moving_descriptor,
            ground_truth_flow,
            padding_mode="zeros",
            align_corners=True,
        )
        second_descriptor = F.normalize(second_descriptor, dim=1, eps=1e-6)
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
    "RegistrationResult",
    "RepresentationResult",
    "V4FinalModel",
    "representation_objective",
]

