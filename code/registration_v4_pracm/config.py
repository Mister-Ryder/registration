"""Typed, fail-closed configuration for PRA-CM V4 and its frozen ablations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple, Type, TypeVar, Union

import yaml


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "v4_full"
    acquisition_identities: Tuple[str, ...] = ("p", "c1", "c2", "c3")
    encoder_channels: Tuple[int, ...] = (24, 48, 72, 96)
    descriptor_channels: int = 32
    response_channels: int = 16
    correlation_levels: Tuple[int, ...] = (3, 2, 1)
    search_radii: Tuple[int, ...] = (4, 3, 2)
    recurrent_iterations: Tuple[int, ...] = (3, 2, 2)
    dns_dilations: Tuple[int, ...] = (1, 2)
    appearance_temperature: float = 0.07
    appearance_samples: int = 768
    appearance_variance_floor: float = 0.08
    maximum_search_support_mm: Tuple[float, ...] = (96.0, 40.0, 14.0)
    minimum_search_support_mm: Tuple[float, ...] = (48.0, 16.0, 6.0)
    posterior_mode_radius: float = 1.75
    posterior_topk: int = 7
    solver_iterations_train: int = 4
    solver_iterations_inference: int = 10
    solver_data_sigma: float = 1.25
    solver_spatial_weight: float = 0.35
    acquisition_logit_bound: float = 0.50
    update_hidden_channels: int = 48
    phase_embedding_channels: int = 8
    correlation_temperature: float = 0.12
    candidate_chunk_size: int = 8
    response_gate_mode: str = "learned"
    response_gate_floor: float = 0.20
    uncertainty_update_floor: float = 0.15
    maximum_learned_bias: float = 0.25
    align_corners: bool = True

    def __post_init__(self) -> None:
        if self.variant not in {"v4a", "v4b", "v4_full"}:
            raise ValueError("model.variant must be v4a, v4b, or v4_full.")
        allowed_identities = {"p", "c1", "c2", "c3", "mr", "ct"}
        if (
            not self.acquisition_identities
            or len(set(self.acquisition_identities)) != len(self.acquisition_identities)
            or any(value not in allowed_identities for value in self.acquisition_identities)
        ):
            raise ValueError("acquisition_identities must be unique canonical P/C1/C2/C3/MR/CT identities.")
        n = len(self.encoder_channels)
        if n < 3 or any(value <= 0 for value in self.encoder_channels):
            raise ValueError("encoder_channels must contain at least three positive values.")
        if not (
            len(self.correlation_levels)
            == len(self.search_radii)
            == len(self.recurrent_iterations)
            == len(self.maximum_search_support_mm)
            == len(self.minimum_search_support_mm)
        ):
            raise ValueError("Correlation levels, radii, iterations and physical supports must align.")
        if tuple(sorted(self.correlation_levels, reverse=True)) != self.correlation_levels:
            raise ValueError("correlation_levels must be ordered coarse to fine.")
        if any(level < 0 or level >= n for level in self.correlation_levels):
            raise ValueError("A correlation level is outside encoder_channels.")
        if any(value < 0 for value in self.search_radii):
            raise ValueError("search_radii must be nonnegative.")
        if any(value < 1 for value in self.recurrent_iterations):
            raise ValueError("Every recurrent level needs at least one iteration.")
        if not self.dns_dilations or any(value < 1 for value in self.dns_dilations):
            raise ValueError("dns_dilations must be positive.")
        if self.descriptor_channels < 4 or self.response_channels < 2:
            raise ValueError("Descriptor/response heads are too narrow.")
        if self.correlation_temperature <= 0 or self.candidate_chunk_size < 1:
            raise ValueError("Invalid correlation temperature or chunk size.")
        if self.response_gate_mode not in {"learned", "neutral", "calibrated"}:
            raise ValueError("response_gate_mode must be learned, neutral, or calibrated.")
        if not 0.0 <= self.response_gate_floor <= 1.0:
            raise ValueError("response_gate_floor must lie in [0,1].")
        if not 0.0 <= self.uncertainty_update_floor <= 1.0:
            raise ValueError("uncertainty_update_floor must lie in [0,1].")
        if self.appearance_temperature <= 0 or self.appearance_samples < 32:
            raise ValueError("Appearance contrastive temperature/samples are invalid.")
        if self.appearance_variance_floor <= 0:
            raise ValueError("appearance_variance_floor must be positive.")
        if any(
            low <= 0 or high < low
            for low, high in zip(
                self.minimum_search_support_mm, self.maximum_search_support_mm
            )
        ):
            raise ValueError("Every physical search-support interval must be positive and ordered.")
        if self.posterior_mode_radius <= 0 or self.posterior_topk < 2:
            raise ValueError("Posterior mode radius/top-k are invalid.")
        if self.solver_iterations_train < 1 or self.solver_iterations_inference < 1:
            raise ValueError("Posterior solver needs at least one iteration.")
        if self.solver_data_sigma <= 0 or self.solver_spatial_weight < 0:
            raise ValueError("Posterior solver scale/weight are invalid.")
        if self.acquisition_logit_bound < 0:
            raise ValueError("acquisition_logit_bound cannot be negative.")


@dataclass(frozen=True)
class LossConfig:
    structural: float = 1.0
    photometric: float = 0.10
    smooth_first: float = 0.15
    smooth_second: float = 0.25
    appearance_invariance: float = 1.0
    appearance_variance: float = 0.10
    jacobian: float = 0.05
    synthetic_flow: float = 0.50
    synthetic_candidate: float = 1.0
    synthetic_contrastive: float = 0.25
    uncertainty_calibration: float = 0.10
    posterior_solver_consistency: float = 0.10
    response_reconstruction: float = 0.10
    response_phase: float = 0.05
    structural_phase_adversarial: float = 0.02
    structural_response_orthogonality: float = 0.02
    response_gate_support: float = 0.02
    relational_moments: float = 0.25
    relational_entropy: float = 0.05
    hard_negative_margin: float = 0.20
    jacobian_margin: float = 0.05
    minimum_gate_support: float = 0.35

    def __post_init__(self) -> None:
        for field in fields(self):
            value = float(getattr(self, field.name))
            if value < 0:
                raise ValueError(f"Loss setting {field.name} cannot be negative.")


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
        if not 0 <= self.inversion_probability <= 1:
            raise ValueError("inversion_probability must lie in [0,1].")
        if not 0 <= self.piecewise_strength <= 1:
            raise ValueError("piecewise_strength must lie in [0,1].")


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "plc_r"
    manifest: str = ""
    phase_inventory: str = ""
    data_root: str = ""
    scalar_cache: str = ""

    intensity_mode: str = "plc_uint8"
    hu_window: Tuple[float, float] = (-200.0, 400.0)
    patch_size: Tuple[int, int, int] = (96, 160, 160)
    samples_per_epoch: int = 512
    triplet_probability: float = 0.50
    minimum_content_fraction: float = 0.01
    foreground_threshold: float = 1.0e-6
    cache_patients: int = 2

    def __post_init__(self) -> None:
        if self.dataset not in {"plc_r", "l2r_mrct"}:
            raise ValueError("data.dataset must be plc_r or l2r_mrct.")
        if self.intensity_mode not in {"plc_uint8", "hu_window", "zscore"}:
            raise ValueError("intensity_mode must be plc_uint8, hu_window, or zscore.")
        if len(self.patch_size) != 3 or any(value < 16 for value in self.patch_size):
            raise ValueError("patch_size must contain three values >=16.")
        if self.samples_per_epoch < 1 or not 0 <= self.triplet_probability <= 1:
            raise ValueError("Invalid sampling settings.")
        if not 0 < self.minimum_content_fraction <= 1:
            raise ValueError("minimum_content_fraction must lie in (0,1].")
        if self.foreground_threshold < 0 or self.cache_patients < 1:
            raise ValueError("foreground_threshold/cache_patients are invalid.")
        if self.dataset == "l2r_mrct" and self.triplet_probability != 0:
            raise ValueError("Two-domain L2R MR-CT training requires triplet_probability=0.")


@dataclass(frozen=True)
class TrainingConfig:
    protocol_id: str = "native"
    epochs: int = 100
    batch_size: int = 1
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 2.0
    precision: str = "fp16"
    seed: int = 20260823
    workers: int = 2
    validation_cases: int = 16
    synthetic_every_steps: int = 1
    log_every_steps: int = 10
    warmup_epochs: int = 5
    minimum_epochs: int = 30
    early_stopping_patience: int = 15
    early_stopping_enabled: bool = True
    minimum_learning_rate_factor: float = 0.05
    validation_label_selection: bool = True
    checkpoint_every_epochs: int = 50
    minimum_free_disk_gib: float = 8.0
    representation_stage_epochs: int = 40
    candidate_ramp_epochs: int = 60
    deformation_ramp_epochs: int = 100

    def __post_init__(self) -> None:
        if not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty.")
        if self.epochs < 1 or self.batch_size != 1:
            raise ValueError("3-D variable pair/triplet training currently requires batch_size=1.")
        if self.learning_rate <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("Invalid optimizer settings.")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16.")
        if self.synthetic_every_steps < 1:
            raise ValueError("synthetic_every_steps must be >=1.")
        if self.workers < 0 or self.validation_cases < 1 or self.log_every_steps < 1:
            raise ValueError("workers/validation_cases/log_every_steps are invalid.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError("warmup_epochs must lie in [0, epochs).")
        if not 1 <= self.minimum_epochs <= self.epochs:
            raise ValueError("minimum_epochs must lie in [1, epochs].")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive.")
        if not 0 < self.minimum_learning_rate_factor <= 1:
            raise ValueError("minimum_learning_rate_factor must lie in (0,1].")
        if self.checkpoint_every_epochs < 1:
            raise ValueError("checkpoint_every_epochs must be positive.")
        if self.minimum_free_disk_gib < 0:
            raise ValueError("minimum_free_disk_gib cannot be negative.")
        if not 0 <= self.representation_stage_epochs < self.epochs:
            raise ValueError("representation_stage_epochs must lie in [0, epochs).")
        if self.candidate_ramp_epochs < 1 or self.deformation_ramp_epochs < 1:
            raise ValueError("Curriculum ramp lengths must be positive.")


@dataclass(frozen=True)
class InferenceConfig:
    tile_size: Tuple[int, int, int] = (96, 192, 192)
    tile_overlap: Tuple[int, int, int] = (48, 96, 96)
    padding_mode: str = "replicate"

    def __post_init__(self) -> None:
        if len(self.tile_size) != 3 or len(self.tile_overlap) != 3:
            raise ValueError("Inference tile settings must be 3-D.")
        if any(size <= overlap for size, overlap in zip(self.tile_size, self.tile_overlap)):
            raise ValueError("Each tile size must exceed its overlap.")
        if self.padding_mode not in {"replicate", "constant"}:
            raise ValueError("Unsupported inference padding mode.")


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig = ModelConfig()
    losses: LossConfig = LossConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
    data: DataConfig = DataConfig()
    training: TrainingConfig = TrainingConfig()
    inference: InferenceConfig = InferenceConfig()

    def __post_init__(self) -> None:
        if self.data.dataset == "plc_r":
            if self.model.acquisition_identities != ("p", "c1", "c2", "c3"):
                raise ValueError(
                    "PLC-R requires the frozen P/C1/C2/C3 acquisition vocabulary."
                )
        else:
            if self.model.acquisition_identities != ("mr", "ct"):
                raise ValueError("L2R MR-CT requires acquisition_identities=[mr, ct].")
            expected_gate = "calibrated" if self.model.variant == "v4_full" else "neutral"
            if self.model.response_gate_mode != expected_gate:
                raise ValueError(
                    f"{self.model.variant} on L2R requires response_gate_mode={expected_gate}."
                )
            if self.training.validation_label_selection:
                raise ValueError("L2R public labels are evaluation-only.")
            if self.losses.relational_moments or self.losses.relational_entropy:
                raise ValueError("Two-domain unpaired L2R training has no triplet relation loss.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _section(cls: Type[T], raw: Mapping[str, Any], name: str) -> T:
    allowed = {field.name for field in fields(cls)}
    unknown = set(raw).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} configuration keys: {sorted(unknown)}")
    payload: Dict[str, Any] = {}
    defaults = cls()
    for key, value in raw.items():
        if isinstance(getattr(defaults, key), tuple):
            value = tuple(value)
        payload[key] = value
    return cls(**payload)


def load_config(path: Union[str, Path]) -> ExperimentConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("Configuration root must be a mapping.")
    allowed = {"model", "losses", "augmentation", "data", "training", "inference"}
    unknown = set(payload).difference(allowed)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    return ExperimentConfig(
        model=_section(ModelConfig, payload.get("model", {}), "model"),
        losses=_section(LossConfig, payload.get("losses", {}), "losses"),
        augmentation=_section(
            AugmentationConfig, payload.get("augmentation", {}), "augmentation"
        ),
        data=_section(DataConfig, payload.get("data", {}), "data"),
        training=_section(TrainingConfig, payload.get("training", {}), "training"),
        inference=_section(InferenceConfig, payload.get("inference", {}), "inference"),
    )
