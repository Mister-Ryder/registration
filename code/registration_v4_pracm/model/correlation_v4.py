"""Physical, uncertainty-adaptive displacement posterior for PRA-CM V4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from ..ops.spatial import candidate_grid, candidate_offsets, sample_candidate_grid
from .response import DirectedResponseCompatibility


@dataclass
class CorrelationDistributionV4:
    probabilities: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    offsets_dzyx: torch.Tensor
    mean_residual: torch.Tensor
    mode_residual: torch.Tensor
    variance_residual: torch.Tensor
    entropy: torch.Tensor
    maximum_probability: torch.Tensor
    expected_gate: torch.Tensor
    candidate_coverage: torch.Tensor
    support_radius_mm: torch.Tensor
    topk_probabilities: torch.Tensor
    topk_residuals: torch.Tensor
    valid: Optional[torch.Tensor]


class AdaptiveCorrelation3D:
    """Explicit posterior with entropy-adaptive support in physical coordinates."""

    def __init__(
        self,
        radius: int,
        gate: DirectedResponseCompatibility,
        *,
        temperature: float,
        chunk_size: int,
        align_corners: bool,
        adaptive_support: bool,
        minimum_support_mm: float,
        maximum_support_mm: float,
        mode_centered: bool,
        mode_radius: float,
        topk: int,
    ) -> None:
        self.radius = int(radius)
        self.gate = gate
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)
        self.align_corners = bool(align_corners)
        self.adaptive_support = bool(adaptive_support)
        self.minimum_support_mm = float(minimum_support_mm)
        self.maximum_support_mm = float(maximum_support_mm)
        self.mode_centered = bool(mode_centered)
        self.mode_radius = float(mode_radius)
        self.topk = int(topk)

    def __call__(
        self,
        fixed_structural: torch.Tensor,
        moving_structural: torch.Tensor,
        fixed_response: torch.Tensor,
        moving_response: torch.Tensor,
        base_flow: torch.Tensor,
        fixed_phase: torch.Tensor,
        moving_phase: torch.Tensor,
        *,
        feature_spacing_dzyx: torch.Tensor,
        previous_entropy: Optional[torch.Tensor] = None,
        fixed_domain: Optional[torch.Tensor] = None,
        moving_domain: Optional[torch.Tensor] = None,
        retain_volume: bool = True,
    ) -> CorrelationDistributionV4:
        if fixed_structural.shape != moving_structural.shape:
            raise ValueError("Fixed and moving structural descriptors must share a grid.")
        if fixed_response.shape != moving_response.shape:
            raise ValueError("Fixed and moving response features must share a grid.")
        if fixed_structural.shape[0] != base_flow.shape[0] or fixed_structural.shape[-3:] != base_flow.shape[-3:]:
            raise ValueError("Descriptor and base-flow grids differ.")
        batch = base_flow.shape[0]
        spacing = torch.as_tensor(
            feature_spacing_dzyx, device=base_flow.device, dtype=base_flow.dtype
        )
        if spacing.ndim == 1:
            spacing = spacing[None].expand(batch, -1)
        if spacing.shape != (batch, 3) or (spacing <= 0).any():
            raise ValueError("feature_spacing_dzyx must be positive [B,3].")
        if fixed_domain is None:
            fixed_domain = torch.ones_like(base_flow[:, :1], dtype=torch.bool)
        if moving_domain is None:
            moving_domain = torch.ones_like(base_flow[:, :1], dtype=torch.bool)

        if self.adaptive_support:
            if previous_entropy is None:
                uncertainty = torch.ones_like(base_flow[:, :1])
            else:
                uncertainty = F.interpolate(
                    previous_entropy.float(),
                    size=base_flow.shape[-3:],
                    mode="trilinear",
                    align_corners=self.align_corners,
                ).to(base_flow.dtype).clamp(0, 1)
            support_radius = self.minimum_support_mm + uncertainty * (
                self.maximum_support_mm - self.minimum_support_mm
            )
        else:
            support_radius = base_flow.new_full(
                (batch, 1, *base_flow.shape[-3:]), self.maximum_support_mm
            )

        offsets = candidate_offsets(
            self.radius, device=base_flow.device, dtype=base_flow.dtype
        )
        physical_distance = (
            offsets[None] * spacing[:, None]
        ).square().sum(dim=2).sqrt()
        moving_joint = torch.cat((moving_structural, moving_response), dim=1)
        structural_channels = moving_structural.shape[1]
        logits_parts = []
        gate_parts = []
        valid_parts = []
        for start in range(0, offsets.shape[0], self.chunk_size):
            current = offsets[start : start + self.chunk_size]
            grid, endpoint_valid = candidate_grid(
                base_flow, current, align_corners=self.align_corners
            )
            sampled = sample_candidate_grid(
                moving_joint, grid, align_corners=self.align_corners
            )
            sampled_structural = sampled[:, :, :structural_channels]
            sampled_response = sampled[:, :, structural_channels:]
            sampled_domain = sample_candidate_grid(
                moving_domain.to(base_flow.dtype),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=self.align_corners,
            )[:, :, 0] > 0.5
            structural = (
                fixed_structural[:, None]
                * F.normalize(sampled_structural, dim=2, eps=1e-6)
            ).sum(dim=2)
            if torch.is_grad_enabled() and self.gate.mode == "calibrated":
                # Recompute each candidate chunk's pointwise gate activations
                # during backward. The forward function is unchanged, while the
                # full-resolution V4-Full graph fits a 32-GiB device.
                evidence = checkpoint(
                    lambda fixed_value, sampled_value: self.gate(
                        fixed_value,
                        sampled_value,
                        fixed_phase,
                        moving_phase,
                    ),
                    fixed_response,
                    sampled_response,
                    use_reentrant=False,
                )
            else:
                evidence = self.gate(
                    fixed_response, sampled_response, fixed_phase, moving_phase
                )
            if self.adaptive_support:
                physical_valid = (
                    physical_distance[:, start : start + current.shape[0], None, None, None]
                    <= support_radius
                )
            else:
                # V4-A retains the frozen dense voxel cube; physical posterior
                # support is introduced only by the V4-B mechanism.
                physical_valid = torch.ones_like(endpoint_valid)
            valid = (
                endpoint_valid
                & sampled_domain
                & fixed_domain[:, None, 0]
                & physical_valid
            )
            logits_parts.append(
                structural / self.temperature + evidence.clamp_min(1e-6).log()
            )
            gate_parts.append(evidence)
            valid_parts.append(valid)

        logits = torch.cat(logits_parts, dim=1)
        gates = torch.cat(gate_parts, dim=1)
        valid = torch.cat(valid_parts, dim=1)
        support_count = valid.sum(dim=1, keepdim=True)
        unsupported = support_count == 0
        centre = offsets.abs().sum(dim=1).argmin()
        safe_valid = valid.clone()
        safe_valid[:, centre : centre + 1] |= unsupported
        masked_logits = logits.masked_fill(~safe_valid, torch.finfo(logits.dtype).min)
        probabilities = torch.softmax(masked_logits.float(), dim=1)

        offset_map = offsets.float().view(1, -1, 3, 1, 1, 1)
        mean = (probabilities[:, :, None] * offset_map).sum(dim=1)
        if self.mode_centered:
            mode_index = probabilities.argmax(dim=1)
            mode_offset = offsets.float()[mode_index].permute(0, 4, 1, 2, 3)
            local = (
                offset_map - mode_offset[:, None]
            ).square().sum(dim=2).sqrt() <= self.mode_radius
            local_probability = probabilities * local
            local_probability = local_probability / local_probability.sum(
                dim=1, keepdim=True
            ).clamp_min(1e-8)
            mode_mean = (local_probability[:, :, None] * offset_map).sum(dim=1)
        else:
            mode_mean = mean
        centred = offset_map - mode_mean[:, None]
        variance = (probabilities[:, :, None] * centred.square()).sum(dim=1)
        raw_entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        entropy = torch.where(
            support_count > 1,
            raw_entropy
            / support_count.clamp_min(2).to(raw_entropy.dtype).log(),
            torch.ones_like(raw_entropy),
        ).clamp(0, 1)
        maximum = probabilities.max(dim=1, keepdim=True).values
        expected_gate = (probabilities * gates.float()).sum(dim=1, keepdim=True)
        coverage = support_count.to(logits.dtype) / float(offsets.shape[0])

        topk_count = min(self.topk, probabilities.shape[1])
        topk_probability, topk_index = probabilities.topk(topk_count, dim=1)
        topk_residual = offsets.float()[topk_index].permute(0, 1, 5, 2, 3, 4)

        mean = mean.to(logits.dtype).masked_fill(unsupported, 0)
        mode_mean = mode_mean.to(logits.dtype).masked_fill(unsupported, 0)
        variance = variance.to(logits.dtype).masked_fill(unsupported, 0)
        entropy = entropy.to(logits.dtype)
        maximum = maximum.to(logits.dtype).masked_fill(unsupported, 0)
        expected_gate = expected_gate.to(logits.dtype).masked_fill(unsupported, 0)
        return CorrelationDistributionV4(
            probabilities=probabilities if retain_volume else None,
            logits=masked_logits.float() if retain_volume else None,
            offsets_dzyx=offsets,
            mean_residual=mean,
            mode_residual=mode_mean,
            variance_residual=variance,
            entropy=entropy,
            maximum_probability=maximum,
            expected_gate=expected_gate,
            candidate_coverage=coverage,
            support_radius_mm=support_radius,
            topk_probabilities=topk_probability,
            topk_residuals=topk_residual,
            valid=valid if retain_volume else None,
        )

