"""Stochastic monotonic nonlinear intensity transform from Mok et al."""

from __future__ import annotations

import math
from typing import Optional

import torch


def monotonic_bezier_lut(n_control_points: int, samples: int, *, device: torch.device, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if n_control_points < 2:
        raise ValueError("n_control_points must include at least two endpoints.")
    degree = n_control_points - 1
    if n_control_points == 2:
        control_y = torch.tensor([0.0, 1.0], device=device)
    else:
        interior = torch.sort(torch.rand(n_control_points - 2, device=device, generator=generator))[0]
        control_y = torch.cat([torch.zeros(1, device=device), interior, torch.ones(1, device=device)])
    t = torch.linspace(0.0, 1.0, samples, device=device)
    curve = torch.zeros_like(t)
    for index in range(n_control_points):
        basis = math.comb(degree, index) * (1 - t).pow(degree - index) * t.pow(index)
        curve = curve + control_y[index] * basis
    return curve.clamp(0.0, 1.0)


def stochastic_nonlinear_transform(
    image: torch.Tensor,
    *,
    n_control_points: int = 3,
    inversion_threshold: float = 0.5,
    lut_samples: int = 1024,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply a monotone Bezier mapping and randomly invert its input."""
    source = image.clamp(0.0, 1.0)
    probability = torch.rand((), device=image.device, generator=generator)
    if float(probability) <= inversion_threshold:
        source = 1.0 - source
    lut = monotonic_bezier_lut(n_control_points, lut_samples, device=image.device, generator=generator)
    position = source * (lut_samples - 1)
    lower = position.floor().long().clamp(0, lut_samples - 1)
    upper = (lower + 1).clamp(0, lut_samples - 1)
    fraction = position - lower.float()
    return lut[lower] * (1.0 - fraction) + lut[upper] * fraction

