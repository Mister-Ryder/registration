"""MASR-Net and duo-layout Deep Neighbourhood Self-similarity.

Paper-specified: four-level 3-D encoder-decoder, BlurPool, trilinear
upsampling, direct and dilated 6-neighbour layouts, 24 relation channels,
and a two-layer 3x3x3 squeezing head. The paper does not state the dilated
layout radius; this reproduction freezes it to two voxels and records that
assumption in its config/checkpoint.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BlurPool3d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        kernel_1d = torch.tensor([1.0, 2.0, 1.0])
        kernel = kernel_1d[:, None, None] * kernel_1d[None, :, None] * kernel_1d[None, None, :]
        kernel = kernel / kernel.sum()
        self.register_buffer("kernel", kernel[None, None].repeat(channels, 1, 1, 1, 1))
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv3d(F.pad(x, (1, 1, 1, 1, 1, 1), mode="replicate"), self.kernel, stride=2, groups=self.channels)


class FeatureUNet(nn.Module):
    def __init__(self, channels: Sequence[int] = (8, 16, 32, 64), feature_channels: int = 4) -> None:
        super().__init__()
        c0, c1, c2, c3 = [int(v) for v in channels]
        self.e0 = ConvBlock(1, c0)
        self.d0 = BlurPool3d(c0)
        self.e1 = ConvBlock(c0, c1)
        self.d1 = BlurPool3d(c1)
        self.e2 = ConvBlock(c1, c2)
        self.d2 = BlurPool3d(c2)
        self.e3 = ConvBlock(c2, c3)
        self.u2 = ConvBlock(c3 + c2, c2)
        self.u1 = ConvBlock(c2 + c1, c1)
        self.u0 = ConvBlock(c1 + c0, c0)
        self.out = nn.Conv3d(c0, feature_channels, 3, padding=1)

    @staticmethod
    def _up(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=reference.shape[-3:], mode="trilinear", align_corners=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e0 = self.e0(x)
        e1 = self.e1(self.d0(e0))
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        u2 = self.u2(torch.cat([self._up(e3, e2), e2], dim=1))
        u1 = self.u1(torch.cat([self._up(u2, e1), e1], dim=1))
        u0 = self.u0(torch.cat([self._up(u1, e0), e0], dim=1))
        return self.out(u0)


def _six_offsets(radius: int) -> List[Tuple[int, int, int]]:
    return [(-radius, 0, 0), (radius, 0, 0), (0, -radius, 0), (0, radius, 0), (0, 0, -radius), (0, 0, radius)]


def _twelve_pairs(radius: int) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    offsets = _six_offsets(radius)
    pairs = []
    for left, right in combinations(offsets, 2):
        distance2 = sum((a - b) ** 2 for a, b in zip(left, right))
        if distance2 == 2 * radius * radius:
            pairs.append((left, right))
    if len(pairs) != 12:
        raise AssertionError(f"Expected 12 sqrt(2)*r pairs, got {len(pairs)}")
    return pairs


def _shift_replicate(x: torch.Tensor, offset: Tuple[int, int, int], radius: int) -> torch.Tensor:
    dz, dy, dx = offset
    padded = F.pad(x, (radius, radius, radius, radius, radius, radius), mode="replicate")
    d, h, w = x.shape[-3:]
    return padded[..., radius + dz:radius + dz + d, radius + dy:radius + dy + h, radius + dx:radius + dx + w]


class DeepNeighbourhoodSelfSimilarity(nn.Module):
    def __init__(self, feature_channels: int = 4, dilation: int = 2, descriptor_channels: int = 24) -> None:
        super().__init__()
        self.feature_channels = int(feature_channels)
        self.dilation = int(dilation)
        self.relation_projection = nn.Parameter(torch.empty(24, feature_channels))
        self.relation_bias = nn.Parameter(torch.zeros(24))
        nn.init.xavier_uniform_(self.relation_projection)
        self.head = nn.Sequential(
            nn.Conv3d(24, descriptor_channels, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(descriptor_channels, descriptor_channels, 3, padding=1),
        )

    def _layout(self, features: torch.Tensor, radius: int) -> torch.Tensor:
        distances = []
        for left, right in _twelve_pairs(radius):
            a = _shift_replicate(features, left, radius)
            b = _shift_replicate(features, right, radius)
            distances.append((a - b).square())
        distance = torch.stack(distances, dim=2)  # B,C,12,D,H,W
        sigma2 = distance.mean(dim=2, keepdim=True).clamp_min(1e-6)
        return torch.exp(-distance / sigma2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        direct = self._layout(features, 1)
        dilated = self._layout(features, self.dilation)
        relations = torch.cat([direct, dilated], dim=2)
        compact = torch.einsum("bcrdhw,rc->brdhw", relations, self.relation_projection)
        compact = compact + self.relation_bias[None, :, None, None, None]
        descriptor = self.head(compact)
        return F.normalize(descriptor, p=2, dim=1, eps=1e-6)


class MASRNet(nn.Module):
    def __init__(
        self,
        channels: Sequence[int] = (8, 16, 32, 64),
        feature_channels: int = 4,
        descriptor_channels: int = 24,
        dns_dilation: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = FeatureUNet(channels, feature_channels)
        self.dns = DeepNeighbourhoodSelfSimilarity(feature_channels, dns_dilation, descriptor_channels)
        self.architecture = {
            "channels": list(channels),
            "feature_channels": feature_channels,
            "descriptor_channels": descriptor_channels,
            "dns_dilation": dns_dilation,
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"MASRNet expects [B,1,D,H,W], got {tuple(image.shape)}")
        return self.dns(self.encoder(image))

