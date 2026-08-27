"""Frozen protocol objects for the V4-final ConvexAdam main path.

This module supersedes the early integration scaffold in ``config.py``.  It is
kept separate so the final path cannot silently fall back to dense normalized
IO, which is retained only as a diagnostic control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar, Union

import yaml


@dataclass(frozen=True)
class DescriptorProtocol:
    feature_channels: int = 4
    descriptor_channels: int = 24
    dns_dilation_voxels: int = 2

    def __post_init__(self) -> None:
        if (self.feature_channels, self.descriptor_channels) != (4, 24):
            raise ValueError("V4-final freezes the faithful Figure-9 4ch -> DSIR 24ch design.")
        if self.dns_dilation_voxels < 2:
            raise ValueError("The second DNS layout must be genuinely dilated.")


@dataclass(frozen=True)
class ConvexAdamProtocol:
    backend: str = "descriptor_convexadam"
    grid_spacing: int = 6
    displacement_half_width: int = 4
    diffusion_weight: float = 1.25
    adam_iterations: int = 80
    adam_grid_spacing: int = 2
    adam_learning_rate: float = 1.0
    inverse_consistency: bool = True
    inverse_consistency_iterations: int = 15
    selected_smoothing: int = 0
    mask_pool_threshold: float = 0.25
    invalid_candidate_penalty: float = 8.0

    def __post_init__(self) -> None:
        if self.backend != "descriptor_convexadam":
            raise ValueError("The final main protocol is descriptor_convexadam.")
        if self.grid_spacing < 1 or self.displacement_half_width < 1:
            raise ValueError("ConvexAdam search parameters must be positive.")
        if self.diffusion_weight < 0 or self.adam_iterations < 0:
            raise ValueError("ConvexAdam refinement parameters are invalid.")
        if self.adam_grid_spacing < 1 or self.adam_learning_rate <= 0:
            raise ValueError("ConvexAdam Adam-grid parameters are invalid.")
        if self.inverse_consistency_iterations < 1 or self.selected_smoothing < 0:
            raise ValueError("ConvexAdam consistency/smoothing parameters are invalid.")
        if not 0 < self.mask_pool_threshold <= 1 or self.invalid_candidate_penalty <= 0:
            raise ValueError("Mask-aware cost parameters are invalid.")


@dataclass(frozen=True)
class RepresentationTrainingProtocol:
    protocol_id: str = "v4_final_protocol300"
    epochs: int = 300
    seed: int = 20260826
    crop_shape_dzyx: Tuple[int, int, int] = (96, 96, 96)
    minimum_foreground_fraction: float = 0.10
    samples_per_view: int = 8196
    contrastive_chunk_size: int = 512
    contrastive_temperature: float = 0.07
    variance_floor: float = 0.08
    variance_weight: float = 0.10
    geometry_probability: float = 0.50
    maximum_velocity_voxels: float = 8.0
    velocity_control_shape: Tuple[int, int, int] = (5, 7, 7)
    integration_steps: int = 5
    learning_rate: float = 1.0e-4
    warmup_epochs: int = 10
    minimum_learning_rate_factor: float = 0.02
    gradient_clip_norm: float = 2.0
    precision: str = "bf16"
    validation_volumes: int = 5
    iteration_log_every_optimizer_steps: int = 5
    checkpoint_every_epochs: int = 100
    minimum_free_disk_gib: float = 8.0

    def __post_init__(self) -> None:
        if not self.protocol_id.strip() or self.epochs < 1:
            raise ValueError("Training protocol and epochs must be valid.")
        if len(self.crop_shape_dzyx) != 3 or any(value < 32 for value in self.crop_shape_dzyx):
            raise ValueError("crop_shape_dzyx must contain three values >=32.")
        if not 0 < self.minimum_foreground_fraction <= 1:
            raise ValueError("minimum_foreground_fraction must lie in (0,1].")
        if self.samples_per_view < 32 or self.contrastive_chunk_size < 1:
            raise ValueError("Contrastive sample/chunk settings are invalid.")
        if self.contrastive_temperature <= 0 or self.variance_floor <= 0:
            raise ValueError("Contrastive temperature/variance floor must be positive.")
        if self.variance_weight < 0 or not 0 <= self.geometry_probability <= 1:
            raise ValueError("Representation task weights are invalid.")
        if self.maximum_velocity_voxels <= 0 or self.integration_steps < 0:
            raise ValueError("Synthetic deformation settings are invalid.")
        if len(self.velocity_control_shape) != 3 or any(value < 2 for value in self.velocity_control_shape):
            raise ValueError("velocity_control_shape must contain three values >=2.")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Optimizer settings must be positive.")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must lie in [0,epochs).")
        if not 0 < self.minimum_learning_rate_factor <= 1:
            raise ValueError("minimum_learning_rate_factor must lie in (0,1].")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16 or bf16.")
        if self.validation_volumes < 1 or self.iteration_log_every_optimizer_steps < 1:
            raise ValueError("Validation/log settings must be positive.")
        if self.checkpoint_every_epochs < 1 or self.minimum_free_disk_gib < 0:
            raise ValueError("Checkpoint settings are invalid.")


@dataclass(frozen=True)
class InferenceProtocol:
    descriptor_precision: str = "fp16"
    ct_foreground_min_hu: float = -500.0
    require_common_pair_grid: bool = True

    def __post_init__(self) -> None:
        if self.descriptor_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("descriptor_precision must be fp32, fp16 or bf16.")


@dataclass(frozen=True)
class V4FinalProtocol:
    descriptor: DescriptorProtocol = DescriptorProtocol()
    solver: ConvexAdamProtocol = ConvexAdamProtocol()
    training: RepresentationTrainingProtocol = RepresentationTrainingProtocol()
    inference: InferenceProtocol = InferenceProtocol()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _strict_section(cls: Type[T], raw: Mapping[str, Any], section: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {section} keys: {sorted(unknown)}")
    defaults = cls()
    converted: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(getattr(defaults, key), tuple):
            value = tuple(value)
        converted[key] = value
    return cls(**converted)


def load_protocol(path: Union[str, Path]) -> V4FinalProtocol:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("V4-final configuration root must be a mapping.")
    allowed = {"descriptor", "solver", "training", "inference"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return V4FinalProtocol(
        descriptor=_strict_section(DescriptorProtocol, payload.get("descriptor", {}), "descriptor"),
        solver=_strict_section(ConvexAdamProtocol, payload.get("solver", {}), "solver"),
        training=_strict_section(
            RepresentationTrainingProtocol, payload.get("training", {}), "training"
        ),
        inference=_strict_section(InferenceProtocol, payload.get("inference", {}), "inference"),
    )


__all__ = [
    "ConvexAdamProtocol",
    "DescriptorProtocol",
    "InferenceProtocol",
    "RepresentationTrainingProtocol",
    "V4FinalProtocol",
    "load_protocol",
]

