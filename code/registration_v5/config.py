"""Strict configuration for the isolated V4-final implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar, Union

import yaml


@dataclass(frozen=True)
class DescriptorConfig:
    feature_channels: int = 4
    descriptor_channels: int = 24
    dns_dilation_voxels: int = 2

    def __post_init__(self) -> None:
        # These values are part of the faithful-v2 architecture contract, not
        # free capacity controls for the present experiment.
        if self.feature_channels != 4:
            raise ValueError("The faithful Figure-9 extractor has four output channels.")
        if self.descriptor_channels != 24:
            raise ValueError("The faithful duo-layout DSIR has 24 output channels.")
        if self.dns_dilation_voxels < 2:
            raise ValueError("dns_dilation_voxels must separate the dilated layout.")


@dataclass(frozen=True)
class SolverConfig:
    scales: Tuple[float, ...] = (0.25, 0.5, 1.0)
    iterations: Tuple[int, ...] = (100, 80, 50)
    learning_rates: Tuple[float, ...] = (0.08, 0.04, 0.02)
    smoothness_weights: Tuple[float, ...] = (0.02, 0.01, 0.005)
    jacobian_weight: float = 0.10
    jacobian_margin: float = 0.05
    minimum_valid_voxels: int = 8
    align_corners: bool = True

    def __post_init__(self) -> None:
        count = len(self.scales)
        if count < 2 or not (
            len(self.iterations)
            == len(self.learning_rates)
            == len(self.smoothness_weights)
            == count
        ):
            raise ValueError("Every solver level needs a scale, iterations, LR and smoothness.")
        if tuple(sorted(self.scales)) != self.scales or self.scales[-1] != 1.0:
            raise ValueError("Solver scales must be coarse-to-fine and finish at 1.0.")
        if any(value <= 0 for value in self.scales + self.learning_rates):
            raise ValueError("Solver scales and learning rates must be positive.")
        if any(value < 1 for value in self.iterations):
            raise ValueError("Every solver level needs at least one iteration.")
        if any(value < 0 for value in self.smoothness_weights):
            raise ValueError("Smoothness weights cannot be negative.")
        if self.jacobian_weight < 0 or not 0 <= self.jacobian_margin < 1:
            raise ValueError("Invalid Jacobian barrier settings.")
        if self.minimum_valid_voxels < 2:
            raise ValueError("minimum_valid_voxels must be at least two.")


@dataclass(frozen=True)
class TrainingConfig:
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
            raise ValueError("Training protocol and epoch budget must be non-empty.")
        if len(self.crop_shape_dzyx) != 3 or any(v < 32 for v in self.crop_shape_dzyx):
            raise ValueError("crop_shape_dzyx must contain three values >=32.")
        if not 0 < self.minimum_foreground_fraction <= 1:
            raise ValueError("minimum_foreground_fraction must lie in (0,1].")
        if self.samples_per_view < 32 or self.contrastive_chunk_size < 1:
            raise ValueError("Contrastive sample/chunk settings are invalid.")
        if self.contrastive_temperature <= 0 or self.variance_floor <= 0:
            raise ValueError("Contrastive temperature and variance floor must be positive.")
        if self.variance_weight < 0 or not 0 <= self.geometry_probability <= 1:
            raise ValueError("Invalid representation loss weights.")
        if self.maximum_velocity_voxels <= 0 or self.integration_steps < 0:
            raise ValueError("Invalid synthetic deformation settings.")
        if len(self.velocity_control_shape) != 3 or any(v < 2 for v in self.velocity_control_shape):
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
            raise ValueError("Validation/logging settings must be positive.")
        if self.checkpoint_every_epochs < 1 or self.minimum_free_disk_gib < 0:
            raise ValueError("Checkpoint settings are invalid.")


@dataclass(frozen=True)
class InferenceConfig:
    descriptor_precision: str = "fp16"
    ct_foreground_min_hu: float = -500.0
    require_common_pair_grid: bool = True

    def __post_init__(self) -> None:
        if self.descriptor_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("descriptor_precision must be fp32, fp16 or bf16.")


@dataclass(frozen=True)
class ExperimentConfig:
    descriptor: DescriptorConfig = DescriptorConfig()
    solver: SolverConfig = SolverConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _section(cls: Type[T], raw: Mapping[str, Any], name: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(unknown)}")
    defaults = cls()
    payload: Dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(getattr(defaults, key), tuple):
            value = tuple(value)
        payload[key] = value
    return cls(**payload)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    allowed = {"descriptor", "solver", "training", "inference"}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return ExperimentConfig(
        descriptor=_section(DescriptorConfig, raw.get("descriptor", {}), "descriptor"),
        solver=_section(SolverConfig, raw.get("solver", {}), "solver"),
        training=_section(TrainingConfig, raw.get("training", {}), "training"),
        inference=_section(InferenceConfig, raw.get("inference", {}), "inference"),
    )

