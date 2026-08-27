"""The exact augmentation contract required by the frozen DSIR training code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class AugmentationConfig:
    maximum_velocity_voxels: float = 8.0
    velocity_control_shape: Tuple[int, int, int] = (5, 7, 7)
    integration_steps: int = 5
    gamma_range: Tuple[float, float] = (0.55, 1.80)
    gain_range: Tuple[float, float] = (0.75, 1.25)
    bias_field_strength: float = 0.15
    noise_std: float = 0.015
    inversion_probability: float = 0.25
    piecewise_strength: float = 0.35

    def __post_init__(self) -> None:
        if self.maximum_velocity_voxels <= 0 or self.integration_steps < 0:
            raise ValueError("Invalid synthetic deformation settings.")
        if any(value < 2 for value in self.velocity_control_shape):
            raise ValueError("velocity_control_shape must contain sizes >=2.")
        if not 0 < self.gamma_range[0] <= self.gamma_range[1]:
            raise ValueError("Invalid gamma_range.")
        if not 0 < self.gain_range[0] <= self.gain_range[1]:
            raise ValueError("Invalid gain_range.")
        if self.bias_field_strength < 0 or self.noise_std < 0:
            raise ValueError("Appearance perturbation strengths cannot be negative.")
        if not 0 <= self.inversion_probability <= 1:
            raise ValueError("inversion_probability must lie in [0,1].")
        if not 0 <= self.piecewise_strength <= 1:
            raise ValueError("piecewise_strength must lie in [0,1].")


__all__ = ["AugmentationConfig"]
