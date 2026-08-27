"""Candidate-specific, directed phase-response compatibility."""

from __future__ import annotations

import torch
import torch.nn as nn

class DirectedResponseCompatibility(nn.Module):
    """Score response compatibility without forcing cross-phase equality.

    A different learned context is used for every fixed/moving phase direction.
    The gate has a non-zero floor, so response evidence can re-rank structurally
    plausible candidates but can never erase the structural correspondence path.
    """

    def __init__(
        self,
        response_channels: int,
        embedding_channels: int,
        gate_floor: float,
        acquisition_count: int,
        mode: str = "learned",
        logit_bound: float = 0.50,
    ) -> None:
        super().__init__()
        self.fixed_embedding = nn.Embedding(acquisition_count, embedding_channels)
        self.moving_embedding = nn.Embedding(acquisition_count, embedding_channels)
        context_channels = 3 * embedding_channels
        input_channels = 4 * response_channels + context_channels
        hidden = max(24, 2 * response_channels)
        self.network = nn.Sequential(
            nn.Conv3d(input_channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, 1, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)
        self.gate_floor = float(gate_floor)
        self.logit_bound = float(logit_bound)
        if mode not in {"learned", "neutral", "calibrated"}:
            raise ValueError("Response compatibility mode must be learned, neutral, or calibrated.")
        self.mode = mode

    def forward(
        self,
        fixed: torch.Tensor,
        moving_candidates: torch.Tensor,
        fixed_phase: torch.Tensor,
        moving_phase: torch.Tensor,
    ) -> torch.Tensor:
        if fixed.ndim != 5 or moving_candidates.ndim != 6:
            raise ValueError("Response gate expects fixed [B,C,D,H,W], candidates [B,K,C,D,H,W].")
        batch, count, channels, depth, height, width = moving_candidates.shape
        if fixed.shape != (batch, channels, depth, height, width):
            raise ValueError("Fixed/candidate response shapes are incompatible.")
        if self.mode == "neutral":
            # Heterogeneous unpaired modalities provide no identifiable target
            # for candidate-specific appearance compatibility.  Returning one
            # leaves the structural correlation logits unchanged.
            return fixed.new_ones((batch, count, depth, height, width))
        fixed_expanded = fixed[:, None].expand(-1, count, -1, -1, -1, -1)
        difference = moving_candidates - fixed_expanded
        fixed_context = self.fixed_embedding(fixed_phase)
        moving_context = self.moving_embedding(moving_phase)
        context = torch.cat(
            (fixed_context, moving_context, moving_context - fixed_context), dim=1
        )
        context = context[:, None, :, None, None, None].expand(
            -1, count, -1, depth, height, width
        )
        value = torch.cat(
            (
                fixed_expanded,
                moving_candidates,
                difference,
                difference.abs(),
                context,
            ),
            dim=2,
        ).reshape(batch * count, -1, depth, height, width)
        raw = self.network(value).reshape(
            batch, count, depth, height, width
        )
        if self.mode == "calibrated":
            # Exactly neutral at initialization; exact candidate NLL then learns
            # only a bounded acquisition-conditioned likelihood correction.
            return torch.exp(self.logit_bound * torch.tanh(raw))
        gate = torch.sigmoid(raw)
        return self.gate_floor + (1.0 - self.gate_floor) * gate
