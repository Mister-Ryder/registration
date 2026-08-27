"""Phase-Response-Aware Correspondence Modeling in native 3-D."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from ..ops.spatial import resize_flow, sampling_valid, warp
from ..phase import phase_indices
from .correlation import CorrelationDistribution3D, DynamicCorrelation3D
from .encoder import EncodedPyramid, SharedStructuralResponseEncoder
from .response import DirectedResponseCompatibility
from .update import CorrespondenceUpdate3D


@dataclass
class IterationOutput3D:
    level: int
    iteration: int
    base_flow: torch.Tensor
    distribution: CorrelationDistribution3D
    flow_increment: torch.Tensor
    flow_after: torch.Tensor
    update_gain: torch.Tensor
    update_evidence: torch.Tensor


@dataclass
class LevelOutput3D:
    level: int
    flow: torch.Tensor
    variance: torch.Tensor
    entropy: torch.Tensor
    maximum_probability: torch.Tensor
    expected_gate: torch.Tensor
    candidate_coverage: torch.Tensor
    iterations: Tuple[IterationOutput3D, ...]


@dataclass
class RegistrationOutput3D:
    flow_dzyx: torch.Tensor
    variance_dzyx: torch.Tensor
    entropy: torch.Tensor
    maximum_probability: torch.Tensor
    expected_gate: torch.Tensor
    candidate_coverage: torch.Tensor
    endpoint_valid: torch.Tensor
    levels: Tuple[LevelOutput3D, ...]
    fixed_encoded: EncodedPyramid
    moving_encoded: EncodedPyramid

    @property
    def flow(self) -> torch.Tensor:
        return self.flow_dzyx

    @property
    def variance(self) -> torch.Tensor:
        return self.variance_dzyx

    @property
    def confidence(self) -> torch.Tensor:
        return (
            (1 - self.entropy)
            * self.maximum_probability
            * self.candidate_coverage
            * self.endpoint_valid.to(self.entropy.dtype)
        )


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float):
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class PRACM3D(nn.Module):
    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.encoder = SharedStructuralResponseEncoder(
            self.config.encoder_channels,
            self.config.descriptor_channels,
            self.config.response_channels,
            self.config.dns_dilations,
        )
        self.response_gate = DirectedResponseCompatibility(
            self.config.response_channels,
            self.config.phase_embedding_channels,
            self.config.response_gate_floor,
            len(self.config.acquisition_identities),
            self.config.response_gate_mode,
        )
        self.updater = CorrespondenceUpdate3D(
            self.config.descriptor_channels,
            self.config.response_channels,
            self.config.update_hidden_channels,
            uncertainty_floor=self.config.uncertainty_update_floor,
            maximum_bias=self.config.maximum_learned_bias,
        )
        finest = self.config.correlation_levels[-1]
        self.response_reconstruction = nn.Sequential(
            nn.Conv3d(self.config.response_channels, self.config.response_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(self.config.response_channels, 1, 1),
        )
        self.response_phase_classifier = nn.Linear(
            self.config.response_channels, len(self.config.acquisition_identities)
        )
        self.structural_phase_classifier = nn.Linear(
            self.config.descriptor_channels, len(self.config.acquisition_identities)
        )
        self.finest_auxiliary_level = finest

    def encode(self, image: torch.Tensor) -> EncodedPyramid:
        return self.encoder(image)

    @staticmethod
    def _mask_at(mask: Optional[torch.Tensor], reference: torch.Tensor) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.ndim != 5 or mask.shape[1] != 1:
            raise ValueError("Domain masks must be [B,1,D,H,W].")
        return F.interpolate(mask.float(), size=reference.shape[-3:], mode="nearest") > 0.5

    @staticmethod
    def _resize_variance(variance: torch.Tensor, size) -> torch.Tensor:
        old = variance.shape[-3:]
        new = tuple(int(value) for value in size)
        if old == new:
            return variance
        value = F.interpolate(variance, size=new, mode="trilinear", align_corners=True)
        scale = [
            (new[i] - 1) / (old[i] - 1) if old[i] > 1 and new[i] > 1 else 0.0
            for i in range(3)
        ]
        return value * value.new_tensor(scale).square().view(1, 3, 1, 1, 1)

    def auxiliary_predictions(
        self,
        encoded: EncodedPyramid,
        output_size,
        *,
        domain: Optional[torch.Tensor] = None,
        adversarial_scale: float = 1.0,
    ):
        response = encoded.response[self.finest_auxiliary_level]
        structural = encoded.structural[self.finest_auxiliary_level]
        reconstruction = F.interpolate(
            self.response_reconstruction(response),
            size=tuple(output_size),
            mode="trilinear",
            align_corners=self.config.align_corners,
        )
        if domain is None:
            response_summary = response.mean(dim=(-3, -2, -1))
            structural_summary = structural.mean(dim=(-3, -2, -1))
        else:
            mask = self._mask_at(domain, response).to(response.dtype)
            denominator = mask.sum(dim=(-3, -2, -1)).clamp_min(1)
            response_summary = (response * mask).sum(dim=(-3, -2, -1)) / denominator
            structural_summary = (structural * mask).sum(dim=(-3, -2, -1)) / denominator
        response_logits = self.response_phase_classifier(response_summary)
        reversed_structural = GradientReversal.apply(
            structural_summary, adversarial_scale
        )
        structural_logits = self.structural_phase_classifier(reversed_structural)
        return reconstruction, response_logits, structural_logits

    def forward(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_phase="p",
        moving_phase="c2",
        fixed_domain: Optional[torch.Tensor] = None,
        moving_domain: Optional[torch.Tensor] = None,
        retain_distributions: Optional[bool] = None,
    ) -> RegistrationOutput3D:
        if fixed.shape != moving.shape or fixed.ndim != 5 or fixed.shape[1] != 1:
            raise ValueError("fixed and moving must be equal [B,1,D,H,W] tensors.")
        if retain_distributions is None:
            retain_distributions = self.training
        batch = fixed.shape[0]
        fixed_phase_ids = phase_indices(
            fixed_phase,
            batch=batch,
            device=fixed.device,
            identities=self.config.acquisition_identities,
        )
        moving_phase_ids = phase_indices(
            moving_phase,
            batch=batch,
            device=fixed.device,
            identities=self.config.acquisition_identities,
        )

        joint = self.encoder(torch.cat((fixed, moving), dim=0))
        fixed_encoded = EncodedPyramid(
            tuple(value[:batch] for value in joint.backbone),
            tuple(value[:batch] for value in joint.structural),
            tuple(value[:batch] for value in joint.response),
        )
        moving_encoded = EncodedPyramid(
            tuple(value[batch:] for value in joint.backbone),
            tuple(value[batch:] for value in joint.structural),
            tuple(value[batch:] for value in joint.response),
        )

        flow = None
        variance = None
        level_outputs = []
        for level, radius, iteration_count in zip(
            self.config.correlation_levels,
            self.config.search_radii,
            self.config.recurrent_iterations,
        ):
            fixed_s = fixed_encoded.structural[level]
            moving_s = moving_encoded.structural[level]
            fixed_r = fixed_encoded.response[level]
            moving_r = moving_encoded.response[level]
            if flow is None:
                flow = fixed_s.new_zeros((batch, 3, *fixed_s.shape[-3:]))
                variance = torch.zeros_like(flow)
            else:
                variance = self._resize_variance(variance, fixed_s.shape[-3:])
                flow = resize_flow(
                    flow, fixed_s.shape[-3:], align_corners=self.config.align_corners
                )
            fixed_mask = self._mask_at(fixed_domain, fixed_s)
            moving_mask = self._mask_at(moving_domain, moving_s)
            correlation = DynamicCorrelation3D(
                radius,
                self.response_gate,
                temperature=self.config.correlation_temperature,
                chunk_size=self.config.candidate_chunk_size,
                align_corners=self.config.align_corners,
            )
            hidden = None
            iterations = []
            final_distribution = None
            for iteration in range(iteration_count):
                base_flow = flow
                distribution = correlation(
                    fixed_s,
                    moving_s,
                    fixed_r,
                    moving_r,
                    base_flow,
                    fixed_phase_ids,
                    moving_phase_ids,
                    fixed_domain=fixed_mask,
                    moving_domain=moving_mask,
                    retain_volume=bool(retain_distributions),
                )
                warped_moving_s = warp(
                    moving_s,
                    base_flow,
                    align_corners=self.config.align_corners,
                )
                update = self.updater(
                    fixed_s,
                    warped_moving_s,
                    fixed_r,
                    base_flow,
                    distribution.mean_residual,
                    distribution.variance_residual,
                    distribution.entropy,
                    distribution.maximum_probability,
                    distribution.expected_gate,
                    distribution.candidate_coverage,
                    hidden,
                )
                hidden = update.hidden
                flow = base_flow + update.flow_increment
                variance = variance + update.transformed_variance
                iterations.append(
                    IterationOutput3D(
                        level,
                        iteration,
                        base_flow,
                        distribution,
                        update.flow_increment,
                        flow,
                        update.gain,
                        update.evidence,
                    )
                )
                final_distribution = distribution
            assert final_distribution is not None and variance is not None
            level_outputs.append(
                LevelOutput3D(
                    level,
                    flow,
                    variance,
                    final_distribution.entropy,
                    final_distribution.maximum_probability,
                    final_distribution.expected_gate,
                    final_distribution.candidate_coverage,
                    tuple(iterations),
                )
            )

        assert flow is not None and variance is not None
        full_size = fixed.shape[-3:]
        flow_full = resize_flow(flow, full_size, align_corners=self.config.align_corners)
        variance_full = self._resize_variance(variance, full_size)
        final = level_outputs[-1]
        scalar_kwargs = dict(size=full_size, mode="trilinear", align_corners=self.config.align_corners)
        entropy = F.interpolate(final.entropy, **scalar_kwargs).clamp(0, 1)
        maximum = F.interpolate(final.maximum_probability, **scalar_kwargs).clamp(0, 1)
        gate = F.interpolate(final.expected_gate, **scalar_kwargs).clamp(0, 1)
        coverage = F.interpolate(final.candidate_coverage, **scalar_kwargs).clamp(0, 1)
        endpoint = sampling_valid(flow_full)
        if fixed_domain is not None:
            endpoint &= fixed_domain.bool()
        if moving_domain is not None:
            sampled_moving_domain = warp(
                moving_domain.float(),
                flow_full,
                mode="nearest",
                padding_mode="zeros",
                align_corners=self.config.align_corners,
            ) > 0.5
            endpoint &= sampled_moving_domain
        return RegistrationOutput3D(
            flow_full,
            variance_full,
            entropy,
            maximum,
            gate,
            coverage,
            endpoint,
            tuple(level_outputs),
            fixed_encoded,
            moving_encoded,
        )
