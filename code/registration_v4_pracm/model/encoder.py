"""Shared 3-D encoder and learned neighbourhood self-similarity descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .relational import RelationalSelfSimilarity3D


def _groups(channels: int) -> int:
    for value in (8, 6, 4, 3, 2):
        if channels % value == 0:
            return value
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(
            input_channels, output_channels, 3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.conv2 = nn.Conv3d(output_channels, output_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(output_channels), output_channels)
        self.skip = (
            nn.Identity()
            if stride == 1 and input_channels == output_channels
            else nn.Conv3d(input_channels, output_channels, 1, stride=stride, bias=False)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.skip(value)
        value = F.silu(self.norm1(self.conv1(value)), inplace=True)
        value = self.norm2(self.conv2(value))
        return F.silu(value + residual, inplace=True)


def _shift_no_wrap(value: torch.Tensor, offset: Tuple[int, int, int]) -> torch.Tensor:
    dz, dy, dx = offset
    shifted = torch.roll(value, shifts=(-dz, -dy, -dx), dims=(-3, -2, -1))
    valid = torch.ones(
        (1, 1, *value.shape[-3:]), device=value.device, dtype=value.dtype
    )
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


_DIRECTIONS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (-1, -1, 0),
    (1, 0, 1),
    (1, 0, -1),
    (-1, 0, 1),
    (-1, 0, -1),
    (0, 1, 1),
    (0, 1, -1),
    (0, -1, 1),
    (0, -1, -1),
)


class DeepSelfSimilarity3D(nn.Module):
    """Learn a descriptor from 3-D neighbour-to-neighbour relations.

    The descriptor does not directly expose absolute intensity.  Its input is a
    bank of cosine self-similarities at multiple dilations, which is then shaped
    by the exact-correspondence training losses.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        dilations: Sequence[int],
    ) -> None:
        super().__init__()
        relation_channels = len(_DIRECTIONS) * len(tuple(dilations))
        self.dilations = tuple(int(value) for value in dilations)
        hidden = max(output_channels, relation_channels)
        self.project = nn.Sequential(
            nn.Conv3d(relation_channels, hidden, 1, bias=False),
            nn.GroupNorm(_groups(hidden), hidden),
            nn.SiLU(inplace=True),
            nn.Conv3d(hidden, output_channels, 3, padding=1, bias=False),
        )
        self.feature_preconditioner = nn.Conv3d(
            input_channels, input_channels, 1, bias=False
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        normalized = F.normalize(self.feature_preconditioner(feature), dim=1, eps=1e-6)
        relations = []
        for dilation in self.dilations:
            for direction in _DIRECTIONS:
                offset = tuple(dilation * value for value in direction)
                neighbour = _shift_no_wrap(normalized, offset)
                relations.append((normalized * neighbour).sum(dim=1, keepdim=True))
        descriptor = self.project(torch.cat(relations, dim=1))
        return F.normalize(descriptor, dim=1, eps=1e-6)


@dataclass
class EncodedPyramid:
    backbone: Tuple[torch.Tensor, ...]
    structural: Tuple[torch.Tensor, ...]
    response: Tuple[torch.Tensor, ...]


class SharedStructuralResponseEncoder(nn.Module):
    """Siamese encoder with distinct structural and response heads."""

    def __init__(
        self,
        channels: Sequence[int],
        descriptor_channels: int,
        response_channels: int,
        dns_dilations: Sequence[int],
    ) -> None:
        super().__init__()
        stages: List[nn.Module] = []
        previous = 1
        for output in channels:
            stages.append(
                nn.Sequential(
                    ResidualBlock3D(previous, output, stride=2),
                    ResidualBlock3D(output, output),
                )
            )
            previous = output
        self.stages = nn.ModuleList(stages)
        self.structural_heads = nn.ModuleList(
            RelationalSelfSimilarity3D(value, descriptor_channels, dns_dilations)
            for value in channels
        )
        self.response_heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv3d(value, response_channels, 3, padding=1, bias=False),
                nn.GroupNorm(_groups(response_channels), response_channels),
                nn.SiLU(inplace=True),
                nn.Conv3d(response_channels, response_channels, 1),
            )
            for value in channels
        )

    def forward(self, image: torch.Tensor) -> EncodedPyramid:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError("Encoder input must be [B,1,D,H,W].")
        backbone = []
        structural = []
        response = []
        value = image
        for stage, structural_head, response_head in zip(
            self.stages, self.structural_heads, self.response_heads
        ):
            value = stage(value)
            backbone.append(value)
            structural.append(structural_head(value))
            response.append(response_head(value))
        return EncodedPyramid(tuple(backbone), tuple(structural), tuple(response))

