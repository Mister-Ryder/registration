"""Correspondence-constrained recurrent update (convolutional, no Transformer)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 6, 4, 3, 2):
        if channels % value == 0:
            return value
    return 1


class ConvGRU3D(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv3d(input_channels + hidden_channels, 2 * hidden_channels, 3, padding=1)
        self.candidate = nn.Conv3d(input_channels + hidden_channels, hidden_channels, 3, padding=1)

    def forward(self, value: torch.Tensor, hidden: Optional[torch.Tensor]) -> torch.Tensor:
        if hidden is None:
            hidden = value.new_zeros((value.shape[0], self.hidden_channels, *value.shape[-3:]))
        reset, update = torch.sigmoid(self.gates(torch.cat((value, hidden), dim=1))).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat((value, reset * hidden), dim=1)))
        return (1 - update) * hidden + update * candidate


@dataclass
class RecurrentUpdate:
    flow_increment: torch.Tensor
    transformed_variance: torch.Tensor
    hidden: torch.Tensor
    gain: torch.Tensor
    bias: torch.Tensor
    evidence: torch.Tensor


class CorrespondenceUpdate3D(nn.Module):
    """Regularize a correlation mean without replacing it by free flow regression."""

    def __init__(
        self,
        descriptor_channels: int,
        response_channels: int,
        hidden_channels: int,
        *,
        uncertainty_floor: float,
        maximum_bias: float,
    ) -> None:
        super().__init__()
        raw_channels = 3 * descriptor_channels + response_channels + 13
        self.context = nn.Sequential(
            nn.Conv3d(raw_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_groups(hidden_channels), hidden_channels),
            nn.SiLU(inplace=True),
        )
        self.gru = ConvGRU3D(hidden_channels, hidden_channels)
        self.head = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden_channels, 6, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.uncertainty_floor = float(uncertainty_floor)
        self.maximum_bias = float(maximum_bias)

    def forward(
        self,
        fixed_structural: torch.Tensor,
        warped_moving_structural: torch.Tensor,
        fixed_response: torch.Tensor,
        current_flow: torch.Tensor,
        mean_residual: torch.Tensor,
        variance_residual: torch.Tensor,
        entropy: torch.Tensor,
        maximum_probability: torch.Tensor,
        expected_gate: torch.Tensor,
        candidate_coverage: torch.Tensor,
        hidden: Optional[torch.Tensor],
    ) -> RecurrentUpdate:
        raw = torch.cat(
            (
                fixed_structural,
                warped_moving_structural,
                fixed_structural - warped_moving_structural,
                fixed_response,
                current_flow,
                mean_residual,
                # sqrt has an infinite derivative at exactly zero. Candidate
                # variance is legitimately zero when only one hypothesis is
                # valid, so floor it before sqrt to keep AMP gradients finite.
                variance_residual.clamp_min(1e-6).sqrt().clamp_max(10),
                entropy,
                maximum_probability,
                expected_gate,
                candidate_coverage,
            ),
            dim=1,
        )
        hidden = self.gru(self.context(raw), hidden)
        gain_raw, bias_raw = self.head(hidden).chunk(2, dim=1)
        gain = 0.5 + torch.sigmoid(gain_raw)
        bias = self.maximum_bias * torch.tanh(bias_raw)
        evidence = (
            self.uncertainty_floor + (1 - self.uncertainty_floor) * (1 - entropy)
        )
        # Pruning implausible hypotheses changes support size, not evidence.
        evidence = evidence.masked_fill(candidate_coverage <= 0, 0)
        increment = evidence * (gain * mean_residual + bias)
        transformed_variance = variance_residual * (evidence * gain).square()
        return RecurrentUpdate(increment, transformed_variance, hidden, gain, bias, evidence)
