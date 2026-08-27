"""Full-volume or overlap-tiled pairwise inference with evidence blending."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Optional, Sequence, Tuple

import torch

from ..config import InferenceConfig
from ..model.pracm_v4 import PRACM3D
from ..ops.spatial import sampling_valid, warp


@dataclass
class InferenceOutput3D:
    flow_dzyx: torch.Tensor
    variance_dzyx: torch.Tensor
    entropy: torch.Tensor
    maximum_probability: torch.Tensor
    expected_gate: torch.Tensor
    candidate_coverage: torch.Tensor
    support_radius_mm: torch.Tensor
    posterior_solver_correction: torch.Tensor
    endpoint_valid: torch.Tensor
    warped_moving: torch.Tensor


def _starts(length: int, tile: int, overlap: int):
    if length <= tile:
        return (0,)
    stride = tile - overlap
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return tuple(values)


def _window(size: Sequence[int], *, device, dtype) -> torch.Tensor:
    axes = []
    for length in size:
        if length <= 2:
            axes.append(torch.ones(length, device=device, dtype=dtype))
        else:
            axes.append(torch.hann_window(length, periodic=False, device=device, dtype=dtype).clamp_min(0.05))
    return (axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :])[None, None]


def _pad_to_tile(value: torch.Tensor, tile: Sequence[int], mode: str):
    spatial = value.shape[-3:]
    after = [max(0, tile[i] - spatial[i]) for i in range(3)]
    if not any(after):
        return value, spatial
    padding = (0, after[2], 0, after[1], 0, after[0])
    return torch.nn.functional.pad(value, padding, mode=mode), spatial


@torch.no_grad()
def register_volume(
    model: PRACM3D,
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    fixed_phase,
    moving_phase,
    config: Optional[InferenceConfig] = None,
    fixed_domain: Optional[torch.Tensor] = None,
    moving_domain: Optional[torch.Tensor] = None,
    spacing_dzyx: Optional[torch.Tensor] = None,
) -> InferenceOutput3D:
    if fixed.shape != moving.shape or fixed.shape[0] != 1:
        raise ValueError("Volume inference requires equal [1,1,D,H,W] tensors.")
    config = config or InferenceConfig()
    fixed_domain = torch.ones_like(fixed, dtype=torch.bool) if fixed_domain is None else fixed_domain.bool()
    moving_domain = torch.ones_like(moving, dtype=torch.bool) if moving_domain is None else moving_domain.bool()
    original_shape = fixed.shape[-3:]
    fixed, _ = _pad_to_tile(fixed, config.tile_size, config.padding_mode)
    moving, _ = _pad_to_tile(moving, config.tile_size, config.padding_mode)
    fixed_domain, _ = _pad_to_tile(fixed_domain.float(), config.tile_size, "constant")
    moving_domain, _ = _pad_to_tile(moving_domain.float(), config.tile_size, "constant")
    fixed_domain = fixed_domain > 0.5
    moving_domain = moving_domain > 0.5
    spatial = fixed.shape[-3:]
    accumulators = [fixed.new_zeros((1, channels, *spatial)) for channels in (3, 3, 1, 1, 1, 1, 1, 1)]
    weights = fixed.new_zeros((1, 1, *spatial))
    starts = [_starts(spatial[i], config.tile_size[i], config.tile_overlap[i]) for i in range(3)]
    window = _window(config.tile_size, device=fixed.device, dtype=fixed.dtype)
    was_training = model.training
    model.eval()
    try:
        for z, y, x in product(*starts):
            region = (
                slice(z, z + config.tile_size[0]),
                slice(y, y + config.tile_size[1]),
                slice(x, x + config.tile_size[2]),
            )
            output = model(
                fixed[(..., *region)],
                moving[(..., *region)],
                fixed_phase=fixed_phase,
                moving_phase=moving_phase,
                fixed_domain=fixed_domain[(..., *region)],
                moving_domain=moving_domain[(..., *region)],
                spacing_dzyx=spacing_dzyx,
                retain_distributions=False,
            )
            values = (
                output.flow,
                output.variance,
                output.entropy,
                output.maximum_probability,
                output.expected_gate,
                output.candidate_coverage,
                output.support_radius_mm,
                output.posterior_solver_correction,
            )
            for accumulator, value in zip(accumulators, values):
                accumulator[(..., *region)] += value * window
            weights[(..., *region)] += window
    finally:
        model.train(was_training)
    values = [value / weights.clamp_min(1e-6) for value in accumulators]
    crop = tuple(slice(0, size) for size in original_shape)
    flow, variance, entropy, maximum, gate, coverage, support, solver_correction = [value[(..., *crop)] for value in values]
    fixed = fixed[(..., *crop)]
    moving = moving[(..., *crop)]
    fixed_domain = fixed_domain[(..., *crop)]
    moving_domain = moving_domain[(..., *crop)]
    endpoint = sampling_valid(flow) & fixed_domain
    endpoint &= warp(
        moving_domain.float(), flow, mode="nearest", padding_mode="zeros"
    ) > 0.5
    warped = warp(moving, flow).masked_fill(~endpoint, 0)
    return InferenceOutput3D(
        flow,
        variance,
        entropy.clamp(0, 1),
        maximum.clamp(0, 1),
        gate.clamp(0, 1),
        coverage.clamp(0, 1),
        support.clamp_min(0),
        solver_correction.clamp_min(0),
        endpoint,
        warped,
    )

