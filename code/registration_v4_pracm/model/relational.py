"""Explicit multi-radius neighbour-to-neighbour structural representation for PRA-CM V4."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    for value in (8, 6, 4, 3, 2):
        if channels % value == 0:
            return value
    return 1


_AXIAL_DIRECTIONS: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)

# Twelve neighbour-to-neighbour relations.  Every pair lies on two distinct
# anatomical axes; the centre voxel is deliberately absent from the relation.
_ORTHOGONAL_PAIRS: Tuple[Tuple[int, int], ...] = (
    (0, 2), (0, 3), (1, 2), (1, 3),
    (0, 4), (0, 5), (1, 4), (1, 5),
    (2, 4), (2, 5), (3, 4), (3, 5),
)


def _shift_no_wrap(value: torch.Tensor, offset: Tuple[int, int, int]) -> torch.Tensor:
    dz, dy, dx = offset
    shifted = torch.roll(value, shifts=(-dz, -dy, -dx), dims=(-3, -2, -1))
    valid = torch.ones((1, 1, *value.shape[-3:]), device=value.device, dtype=value.dtype)
    if dz > 0:
        valid[..., -dz:, :, :] = 0
    elif dz < 0:
        valid[..., : -dz, :, :] = 0
    if dy > 0:
        valid[..., -dy:, :] = 0
    elif dy < 0:
        valid[..., : -dy, :] = 0
    if dx > 0:
        valid[..., -dx:] = 0
    elif dx < 0:
        valid[..., : -dx] = 0
    return shifted * valid


class RelationalSelfSimilarity3D(nn.Module):
    """Encode second-order local geometry instead of first-order centre similarity.

    For each dilation, six axial neighbours are sampled and twelve orthogonal
    neighbour pairs are compared by cosine similarity.  The resulting relation
    bank is locally standardized before a learned projection.  Consequently,
    a descriptor cannot solve the task by retaining only the centre intensity.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilations: Sequence[int],
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        relation_channels = len(_ORTHOGONAL_PAIRS) * len(self.dilations)
        hidden = max(output_channels, relation_channels)
        self.feature_preconditioner = nn.Conv3d(
            input_channels, input_channels, 1, bias=False
        )
        self.project = nn.Sequential(
            nn.Conv3d(relation_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, output_channels, 3, padding=1, bias=False),
        )

    @property
    def relation_channels(self) -> int:
        return len(_ORTHOGONAL_PAIRS) * len(self.dilations)

    def raw_relations(self, feature: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(self.feature_preconditioner(feature), dim=1, eps=1e-6)
        relations = []
        for dilation in self.dilations:
            neighbours = [
                _shift_no_wrap(
                    normalized,
                    tuple(dilation * component for component in direction),
                )
                for direction in _AXIAL_DIRECTIONS
            ]
            for first, second in _ORTHOGONAL_PAIRS:
                relations.append(
                    (neighbours[first] * neighbours[second]).sum(dim=1, keepdim=True)
                )
        relation_bank = torch.cat(relations, dim=1)
        mean = relation_bank.mean(dim=1, keepdim=True)
        scale = relation_bank.var(dim=1, keepdim=True, unbiased=False).add(1e-4).sqrt()
        return (relation_bank - mean) / scale

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.project(self.raw_relations(feature)), dim=1, eps=1e-6)

