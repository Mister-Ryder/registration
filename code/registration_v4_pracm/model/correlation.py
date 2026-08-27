"""Memory-bounded explicit 3-D displacement-space correlation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from ..ops.spatial import candidate_grid, candidate_offsets, sample_candidate_grid
from .response import DirectedResponseCompatibility


@dataclass
class CorrelationDistribution3D:
    probabilities: Optional[torch.Tensor]
    logits: Optional[torch.Tensor]
    offsets_dzyx: torch.Tensor
    mean_residual: torch.Tensor
    variance_residual: torch.Tensor
    entropy: torch.Tensor
    maximum_probability: torch.Tensor
    expected_gate: torch.Tensor
    candidate_coverage: torch.Tensor
    valid: Optional[torch.Tensor]


class DynamicCorrelation3D:
    def __init__(
        self,
        radius: int,
        gate: DirectedResponseCompatibility,
        *,
        temperature: float,
        chunk_size: int,
        align_corners: bool,
    ) -> None:
        self.radius = int(radius)
        self.gate = gate
        self.temperature = float(temperature)
        self.chunk_size = int(chunk_size)
        self.align_corners = bool(align_corners)

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
        fixed_domain: Optional[torch.Tensor] = None,
        moving_domain: Optional[torch.Tensor] = None,
        retain_volume: bool = True,
    ) -> CorrelationDistribution3D:
        if fixed_structural.shape != moving_structural.shape:
            raise ValueError("Fixed and moving structural descriptors must share a grid.")
        if fixed_response.shape != moving_response.shape:
            raise ValueError("Fixed and moving response features must share a grid.")
        if fixed_structural.shape[0] != base_flow.shape[0] or fixed_structural.shape[-3:] != base_flow.shape[-3:]:
            raise ValueError("Descriptor and base-flow grids differ.")
        if fixed_domain is None:
            fixed_domain = torch.ones_like(base_flow[:, :1], dtype=torch.bool)
        if moving_domain is None:
            moving_domain = torch.ones_like(base_flow[:, :1], dtype=torch.bool)

        offsets = candidate_offsets(
            self.radius, device=base_flow.device, dtype=base_flow.dtype
        )
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
            gate = self.gate(
                fixed_response, sampled_response, fixed_phase, moving_phase
            )
            valid = endpoint_valid & sampled_domain & fixed_domain[:, None, 0]
            logits_parts.append(structural / self.temperature + gate.clamp_min(1e-6).log())
            gate_parts.append(gate)
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
        # Keep the retained distribution in float32.  With 125-729 candidates,
        # valid low probabilities routinely underflow in fp16 and would turn
        # exact-correspondence NLL into Inf even though the softmax itself is
        # well defined.
        probabilities_fp32 = torch.softmax(masked_logits.float(), dim=1)

        offset_map = offsets.view(1, -1, 3, 1, 1, 1)
        offset_map_fp32 = offset_map.float()
        mean_fp32 = (probabilities_fp32[:, :, None] * offset_map_fp32).sum(dim=1)
        centred = offset_map_fp32 - mean_fp32[:, None]
        variance_fp32 = (
            probabilities_fp32[:, :, None] * centred.square()
        ).sum(dim=1)
        raw_entropy_fp32 = -(
            probabilities_fp32 * probabilities_fp32.clamp_min(1e-8).log()
        ).sum(
            dim=1, keepdim=True
        )
        entropy_fp32 = torch.where(
            support_count > 1,
            raw_entropy_fp32
            / support_count.clamp_min(2).to(raw_entropy_fp32.dtype).log(),
            torch.ones_like(raw_entropy_fp32),
        ).clamp(0, 1)
        coverage = support_count.to(logits.dtype) / float(offsets.shape[0])
        maximum_fp32 = probabilities_fp32.max(dim=1, keepdim=True).values
        expected_gate_fp32 = (
            probabilities_fp32 * gates.float()
        ).sum(dim=1, keepdim=True)
        mean = mean_fp32.to(logits.dtype)
        variance = variance_fp32.to(logits.dtype)
        entropy = entropy_fp32.to(logits.dtype)
        maximum = maximum_fp32.to(logits.dtype)
        expected_gate = expected_gate_fp32.to(logits.dtype)
        mean = mean.masked_fill(unsupported, 0)
        variance = variance.masked_fill(unsupported, 0)
        maximum = maximum.masked_fill(unsupported, 0)
        expected_gate = expected_gate.masked_fill(unsupported, 0)

        return CorrelationDistribution3D(
            probabilities=probabilities_fp32 if retain_volume else None,
            logits=masked_logits.float() if retain_volume else None,
            offsets_dzyx=offsets,
            mean_residual=mean,
            variance_residual=variance,
            entropy=entropy,
            maximum_probability=maximum,
            expected_gate=expected_gate,
            candidate_coverage=coverage,
            valid=valid if retain_volume else None,
        )
