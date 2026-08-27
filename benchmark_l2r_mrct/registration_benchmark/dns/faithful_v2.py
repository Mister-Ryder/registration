"""Versioned, paper-aligned MASR/DNS components for future B05 and PRA-CM runs.

The original :class:`registration_benchmark.dns.model.MASRNet` is deliberately
left untouched because the protocol-300 checkpoint was trained with that state
dict.  This module provides a new ``faithful_v2`` implementation for retraining
and for reuse as a differentiable, full-resolution DSIR extractor.

The CVPR 2024 paper explicitly discloses the Figure 9 channel/resolution
topology, cubic Bezier augmentation with four two-dimensional control points,
random intensity inversion, the L2R CT window, and the common 2 mm grid.  It
does not disclose every numerical convention needed by executable code.  Such
choices are collected in :data:`FAITHFUL_V2_REPRODUCTION_ASSUMPTIONS` rather
than being presented as paper facts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to
from torch import nn

from .model import BlurPool3d, DeepNeighbourhoodSelfSimilarity


FAITHFUL_V2_VERSION = "mok_masr_dns_faithful_v2"

FAITHFUL_V2_PAPER_EXPLICIT: Mapping[str, Any] = {
    "feature_extractor_channel_sequence": [1, 8, 8, 16, 16, 32, 32, 64, 64, 32, 16, 8, 4],
    "feature_extractor_downsampling": "BlurPool",
    "feature_extractor_upsampling": "trilinear",
    "bezier_degree": 3,
    "bezier_control_points": 4,
    "random_intensity_inversion": True,
    "intensity_inversion_threshold": 0.5,
    "l2r_ct_window_hu": [-200.0, 1024.0],
    "l2r_common_shape_xyz": [192, 160, 192],
    "l2r_common_spacing_mm_xyz": [2.0, 2.0, 2.0],
}

FAITHFUL_V2_REPRODUCTION_ASSUMPTIONS: Mapping[str, Any] = {
    "bezier_monotonicity": (
        "sample four 2-D points uniformly, sort both coordinates, then range-normalize "
        "the endpoint coordinates to [0,1] before evaluating the cubic curve"
    ),
    "mr_intensity_normalization": (
        "map non-zero finite foreground between the 0.5th and 99.5th percentiles to [0,1]"
    ),
    "geometry_standardization": (
        "preserve source orientation and physical center while resampling to the disclosed grid"
    ),
    "blurpool_filter": "separable [1,2,1]^3 kernel with replicate padding",
    "trilinear_align_corners": True,
    "skip_fusion": "channel concatenation followed by the Figure 9 decoder convolution",
    "dns_dilation_voxels": 2,
}


@dataclass(frozen=True)
class MRCTPreprocessConfig:
    """Executable preprocessing contract for the L2R abdomen MR-CT setting."""

    target_shape_xyz: Tuple[int, int, int] = (192, 160, 192)
    target_spacing_xyz: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    ct_window_hu: Tuple[float, float] = (-200.0, 1024.0)
    mr_percentiles: Tuple[float, float] = (0.5, 99.5)
    geometry_atol: float = 1e-4

    def __post_init__(self) -> None:
        if len(self.target_shape_xyz) != 3 or any(int(v) < 2 for v in self.target_shape_xyz):
            raise ValueError("target_shape_xyz must contain three dimensions >= 2.")
        if len(self.target_spacing_xyz) != 3 or any(float(v) <= 0 for v in self.target_spacing_xyz):
            raise ValueError("target_spacing_xyz must contain three positive values.")
        if not self.ct_window_hu[0] < self.ct_window_hu[1]:
            raise ValueError("ct_window_hu must be strictly increasing.")
        if not 0 <= self.mr_percentiles[0] < self.mr_percentiles[1] <= 100:
            raise ValueError("mr_percentiles must be increasing values in [0,100].")


@dataclass(frozen=True)
class StandardizedMRCTVolume:
    """Normalized volume and the physical grid on which it is represented."""

    data_xyz: np.ndarray
    affine_xyz: np.ndarray
    modality: str
    geometry_resampled: bool
    original_shape_xyz: Tuple[int, int, int]


@dataclass(frozen=True)
class DSIRExtraction:
    """Full-resolution DSIR plus the image geometry needed by downstream code."""

    descriptor_bcdhw: torch.Tensor
    affine_xyz: Optional[np.ndarray]
    shape_xyz: Tuple[int, int, int]
    modality: str
    geometry_resampled: bool
    implementation_version: str = FAITHFUL_V2_VERSION


def faithful_v2_provenance() -> Dict[str, Any]:
    """Return a serializable record separating paper facts from assumptions."""

    return {
        "implementation_version": FAITHFUL_V2_VERSION,
        "paper_explicit": dict(FAITHFUL_V2_PAPER_EXPLICIT),
        "reproduction_assumptions": dict(FAITHFUL_V2_REPRODUCTION_ASSUMPTIONS),
    }


def canonical_mrct_modality(modality: str) -> str:
    """Map MR/CT and multiphase-CT labels onto the two preprocessing branches."""

    token = "".join(character for character in str(modality).strip().lower() if character.isalnum())
    if token in {
        "ct", "computedtomography", "pre", "precontrast", "unenhanced",
        "arterial", "venous", "portalvenous", "delayed", "delay",
    }:
        return "ct"
    if token in {"mr", "mri", "magneticresonance", "t1", "t1w", "t2", "t2w"}:
        return "mr"
    raise ValueError(f"Unsupported MR-CT modality: {modality!r}")


def normalize_mrct_intensity(
    data_xyz: np.ndarray,
    modality: str,
    config: MRCTPreprocessConfig = MRCTPreprocessConfig(),
) -> np.ndarray:
    """Apply the versioned modality-aware intensity preprocessing.

    CT uses the paper-disclosed HU window.  The paper does not specify MR
    normalization; the foreground-only robust percentile mapping is therefore
    an explicit reproduction assumption.
    """

    values = np.asarray(data_xyz, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Expected a finite three-dimensional image.")
    canonical = canonical_mrct_modality(modality)
    if canonical == "ct":
        low, high = (float(v) for v in config.ct_window_hu)
        return np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)

    foreground = np.abs(values) > np.finfo(np.float32).eps
    samples = values[foreground]
    if samples.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    low, high = np.percentile(samples, config.mr_percentiles)
    if high <= low + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    normalized = np.clip((values - low) / (high - low), 0.0, 1.0).astype(np.float32)
    normalized[~foreground] = 0.0
    return normalized


def _voxel_sizes_xyz(affine_xyz: np.ndarray) -> np.ndarray:
    linear = np.asarray(affine_xyz, dtype=np.float64)[:3, :3]
    sizes = np.linalg.norm(linear, axis=0)
    if not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("Affine has invalid voxel sizes.")
    return sizes


def is_standard_mrct_geometry(
    shape_xyz: Sequence[int],
    affine_xyz: np.ndarray,
    config: MRCTPreprocessConfig = MRCTPreprocessConfig(),
) -> bool:
    """Return whether a volume already has the disclosed L2R shape/spacing."""

    return (
        tuple(int(v) for v in shape_xyz) == tuple(config.target_shape_xyz)
        and np.allclose(
            _voxel_sizes_xyz(affine_xyz),
            np.asarray(config.target_spacing_xyz, dtype=np.float64),
            atol=float(config.geometry_atol),
            rtol=0.0,
        )
    )


def _centered_target_affine(
    source_shape_xyz: Sequence[int],
    source_affine_xyz: np.ndarray,
    config: MRCTPreprocessConfig,
) -> np.ndarray:
    """Construct the assumed common grid while preserving orientation/center."""

    source_affine = np.asarray(source_affine_xyz, dtype=np.float64)
    direction = source_affine[:3, :3] / _voxel_sizes_xyz(source_affine)[None, :]
    target_linear = direction * np.asarray(config.target_spacing_xyz, dtype=np.float64)[None, :]
    source_center = nib.affines.apply_affine(
        source_affine, (np.asarray(source_shape_xyz, dtype=np.float64) - 1.0) / 2.0
    )
    target_center_index = (np.asarray(config.target_shape_xyz, dtype=np.float64) - 1.0) / 2.0
    target_affine = np.eye(4, dtype=np.float64)
    target_affine[:3, :3] = target_linear
    target_affine[:3, 3] = source_center - target_linear @ target_center_index
    return target_affine


def standardize_mrct_volume(
    data_xyz: np.ndarray,
    affine_xyz: np.ndarray,
    modality: str,
    config: MRCTPreprocessConfig = MRCTPreprocessConfig(),
) -> StandardizedMRCTVolume:
    """Resample to the disclosed L2R grid, then normalize by modality.

    A volume already on a 192x160x192, 2 mm grid is not interpolated again.
    For unpaired training scans, the undisclosed placement convention preserves
    source orientation and physical center.
    """

    values = np.asarray(data_xyz, dtype=np.float32)
    affine = np.asarray(affine_xyz, dtype=np.float64)
    if values.ndim != 3 or affine.shape != (4, 4):
        raise ValueError("Expected a 3-D image and a 4x4 affine.")
    if not np.isfinite(values).all() or not np.isfinite(affine).all():
        raise ValueError("Image and affine must be finite.")
    original_shape = tuple(int(v) for v in values.shape)
    canonical = canonical_mrct_modality(modality)
    resampled = not is_standard_mrct_geometry(values.shape, affine, config)
    if resampled:
        target_affine = _centered_target_affine(values.shape, affine, config)
        fill_value = float(config.ct_window_hu[0]) if canonical == "ct" else 0.0
        image = nib.Nifti1Image(values, affine)
        image = resample_from_to(
            image,
            (tuple(config.target_shape_xyz), target_affine),
            order=1,
            mode="constant",
            cval=fill_value,
        )
        values = np.asarray(image.dataobj, dtype=np.float32)
        affine = np.asarray(image.affine, dtype=np.float64)
    normalized = normalize_mrct_intensity(values, canonical, config)
    return StandardizedMRCTVolume(
        data_xyz=normalized,
        affine_xyz=affine,
        modality=canonical,
        geometry_resampled=resampled,
        original_shape_xyz=original_shape,
    )


def sample_monotonic_cubic_control_points(
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample four 2-D points and canonicalize them to a monotonic cubic curve."""

    raw = torch.rand((4, 2), device=device, dtype=dtype, generator=generator)
    x = torch.sort(raw[:, 0]).values
    y = torch.sort(raw[:, 1]).values
    eps = torch.finfo(dtype).eps
    x = (x - x[0]) / (x[-1] - x[0]).clamp_min(eps)
    y = (y - y[0]) / (y[-1] - y[0]).clamp_min(eps)
    points = torch.stack([x, y], dim=1)
    points[0] = 0.0
    points[-1] = 1.0
    return points


def cubic_bezier_lut_faithful_v2(
    samples: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    control_points: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate a monotonic cubic Bezier intensity mapping on a uniform LUT."""

    if int(samples) < 2:
        raise ValueError("samples must be >= 2.")
    points = (
        sample_monotonic_cubic_control_points(device=device, dtype=dtype, generator=generator)
        if control_points is None
        else control_points.to(device=device, dtype=dtype)
    )
    if points.shape != (4, 2):
        raise ValueError("A cubic Bezier curve requires exactly four 2-D control points.")
    if torch.any(points[1:, 0] < points[:-1, 0]) or torch.any(points[1:, 1] < points[:-1, 1]):
        raise ValueError("Control points must be monotonic in both coordinates.")

    curve_samples = max(int(samples) * 4, 256)
    t = torch.linspace(0.0, 1.0, curve_samples, device=device, dtype=dtype)
    one_minus_t = 1.0 - t
    basis = torch.stack(
        [
            one_minus_t.pow(3),
            3.0 * one_minus_t.pow(2) * t,
            3.0 * one_minus_t * t.pow(2),
            t.pow(3),
        ],
        dim=1,
    )
    curve = basis @ points
    query = torch.linspace(0.0, 1.0, int(samples), device=device, dtype=dtype)
    upper = torch.searchsorted(curve[:, 0].contiguous(), query, right=False).clamp(1, curve_samples - 1)
    lower = upper - 1
    x0, x1 = curve[lower, 0], curve[upper, 0]
    y0, y1 = curve[lower, 1], curve[upper, 1]
    fraction = (query - x0) / (x1 - x0).clamp_min(torch.finfo(dtype).eps)
    lut = y0 + fraction * (y1 - y0)
    lut = torch.cummax(lut.clamp(0.0, 1.0), dim=0).values
    lut[0] = 0.0
    lut[-1] = 1.0
    return lut


def stochastic_nonlinear_transform_faithful_v2(
    image: torch.Tensor,
    *,
    inversion_threshold: float = 0.5,
    lut_samples: int = 1024,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Apply the paper-aligned cubic mapping and stochastic intensity inversion."""

    if not 0.0 <= float(inversion_threshold) <= 1.0:
        raise ValueError("inversion_threshold must lie in [0,1].")
    source = image.clamp(0.0, 1.0)
    probability = torch.rand((), device=image.device, generator=generator)
    if float(probability) <= float(inversion_threshold):
        source = 1.0 - source
    lut = cubic_bezier_lut_faithful_v2(
        int(lut_samples), device=image.device, dtype=image.dtype, generator=generator
    )
    position = source * (int(lut_samples) - 1)
    lower = position.floor().long().clamp(0, int(lut_samples) - 1)
    upper = (lower + 1).clamp(0, int(lut_samples) - 1)
    fraction = position - lower.to(dtype=position.dtype)
    return lut[lower] * (1.0 - fraction) + lut[upper] * fraction


class Figure9FeatureExtractor(nn.Module):
    """Feature extractor with the exact channel/resolution topology in Figure 9.

    Each encoder/decoder bar that changes channels is one 3x3x3 convolution.
    Equal-channel lower-resolution bars are produced by BlurPool.  The three
    Figure 9 arrows are implemented as concatenative skip connections; that
    fusion rule is an explicit reproduction assumption because the figure does
    not label the operation.
    """

    channel_resolution_sequence = (
        (1, 1.0),
        (8, 1.0),
        (8, 0.5),
        (16, 0.5),
        (16, 0.25),
        (32, 0.25),
        (32, 0.125),
        (64, 0.125),
        (64, 0.125),
        (32, 0.25),
        (16, 0.5),
        (8, 1.0),
        (4, 1.0),
    )

    def __init__(
        self,
        feature_channels: int = 4,
        negative_slope: float = 0.2,
        align_corners: bool = True,
    ) -> None:
        super().__init__()
        if int(feature_channels) != 4:
            raise ValueError("Figure 9 explicitly fixes the output feature channels to four.")
        self.negative_slope = float(negative_slope)
        self.align_corners = bool(align_corners)
        self.conv_full_8 = nn.Conv3d(1, 8, 3, padding=1)
        self.down_full_to_half = BlurPool3d(8)
        self.conv_half_16 = nn.Conv3d(8, 16, 3, padding=1)
        self.down_half_to_quarter = BlurPool3d(16)
        self.conv_quarter_32 = nn.Conv3d(16, 32, 3, padding=1)
        self.down_quarter_to_eighth = BlurPool3d(32)
        self.conv_eighth_64 = nn.Conv3d(32, 64, 3, padding=1)
        self.conv_eighth_64_refine = nn.Conv3d(64, 64, 3, padding=1)
        self.conv_decode_quarter_32 = nn.Conv3d(64 + 32, 32, 3, padding=1)
        self.conv_decode_half_16 = nn.Conv3d(32 + 16, 16, 3, padding=1)
        self.conv_decode_full_8 = nn.Conv3d(16 + 8, 8, 3, padding=1)
        self.conv_output_4 = nn.Conv3d(8, 4, 3, padding=1)

    def _activation(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(tensor, negative_slope=self.negative_slope, inplace=False)

    def _up(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            tensor,
            size=reference.shape[-3:],
            mode="trilinear",
            align_corners=self.align_corners,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"Figure9FeatureExtractor expects [B,1,D,H,W], got {tuple(image.shape)}")
        full_8 = self._activation(self.conv_full_8(image))
        half_8 = self.down_full_to_half(full_8)
        half_16 = self._activation(self.conv_half_16(half_8))
        quarter_16 = self.down_half_to_quarter(half_16)
        quarter_32 = self._activation(self.conv_quarter_32(quarter_16))
        eighth_32 = self.down_quarter_to_eighth(quarter_32)
        eighth_64 = self._activation(self.conv_eighth_64(eighth_32))
        eighth_64 = self._activation(self.conv_eighth_64_refine(eighth_64))
        decode_quarter_32 = self._activation(
            self.conv_decode_quarter_32(torch.cat([self._up(eighth_64, quarter_32), quarter_32], dim=1))
        )
        decode_half_16 = self._activation(
            self.conv_decode_half_16(torch.cat([self._up(decode_quarter_32, half_16), half_16], dim=1))
        )
        decode_full_8 = self._activation(
            self.conv_decode_full_8(torch.cat([self._up(decode_half_16, full_8), full_8], dim=1))
        )
        return self.conv_output_4(decode_full_8)


class MASRNetFaithfulV2(nn.Module):
    """Versioned MASR-Net using the Figure 9 feature extractor."""

    def __init__(
        self,
        descriptor_channels: int = 24,
        dns_dilation: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = Figure9FeatureExtractor(feature_channels=4)
        self.dns = DeepNeighbourhoodSelfSimilarity(
            feature_channels=4,
            dilation=int(dns_dilation),
            descriptor_channels=int(descriptor_channels),
        )
        self.architecture = {
            "implementation_version": FAITHFUL_V2_VERSION,
            "feature_extractor": "figure9_exact_topology",
            "feature_channels": 4,
            "descriptor_channels": int(descriptor_channels),
            "dns_dilation": int(dns_dilation),
        }

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 5 or image.shape[1] != 1:
            raise ValueError(f"MASRNetFaithfulV2 expects [B,1,D,H,W], got {tuple(image.shape)}")
        return self.dns(self.encoder(image))


class FullResolutionDSIRExtractor(nn.Module):
    """Reusable differentiable DSIR interface for B05 and future PRA-CM code.

    ``forward`` accepts already normalized ``[B,1,D,H,W]`` tensors and keeps
    gradients, so V4 can reuse it as a normal module.  ``extract_numpy`` and
    ``extract_nifti`` provide inference conveniences with the versioned
    modality-aware preprocessing.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        preprocess_config: MRCTPreprocessConfig = MRCTPreprocessConfig(),
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else MASRNetFaithfulV2()
        self.preprocess_config = preprocess_config
        self.normalize_output = bool(normalize_output)
        self.implementation_version = FAITHFUL_V2_VERSION

    def forward(self, normalized_image: torch.Tensor) -> torch.Tensor:
        descriptor = self.model(normalized_image)
        if descriptor.shape[0] != normalized_image.shape[0] or descriptor.shape[-3:] != normalized_image.shape[-3:]:
            raise RuntimeError(
                "Full-resolution DSIR contract violated: "
                f"input={tuple(normalized_image.shape)}, output={tuple(descriptor.shape)}"
            )
        if self.normalize_output:
            descriptor = F.normalize(descriptor, p=2, dim=1, eps=1e-6)
        return descriptor

    def _device(self) -> torch.device:
        parameter = next(self.parameters(), None)
        return parameter.device if parameter is not None else torch.device("cpu")

    def preprocess_numpy(self, data_xyz: np.ndarray, modality: str) -> torch.Tensor:
        normalized = normalize_mrct_intensity(data_xyz, modality, self.preprocess_config)
        tensor = torch.from_numpy(normalized.transpose(2, 1, 0).copy())[None, None]
        return tensor.to(self._device())

    @torch.no_grad()
    def extract_numpy(self, data_xyz: np.ndarray, modality: str) -> DSIRExtraction:
        canonical = canonical_mrct_modality(modality)
        image = self.preprocess_numpy(data_xyz, canonical)
        descriptor = self.forward(image)
        return DSIRExtraction(
            descriptor_bcdhw=descriptor,
            affine_xyz=None,
            shape_xyz=tuple(int(v) for v in data_xyz.shape),
            modality=canonical,
            geometry_resampled=False,
        )

    @torch.no_grad()
    def extract_nifti(
        self,
        image: nib.spatialimages.SpatialImage,
        modality: str,
        *,
        standardize_geometry: bool = True,
    ) -> DSIRExtraction:
        data = np.asarray(image.dataobj, dtype=np.float32)
        affine = np.asarray(image.affine, dtype=np.float64)
        if standardize_geometry:
            volume = standardize_mrct_volume(data, affine, modality, self.preprocess_config)
        else:
            canonical = canonical_mrct_modality(modality)
            volume = StandardizedMRCTVolume(
                data_xyz=normalize_mrct_intensity(data, canonical, self.preprocess_config),
                affine_xyz=affine,
                modality=canonical,
                geometry_resampled=False,
                original_shape_xyz=tuple(int(v) for v in data.shape),
            )
        tensor = torch.from_numpy(volume.data_xyz.transpose(2, 1, 0).copy())[None, None].to(self._device())
        descriptor = self.forward(tensor)
        return DSIRExtraction(
            descriptor_bcdhw=descriptor,
            affine_xyz=volume.affine_xyz.copy(),
            shape_xyz=tuple(int(v) for v in volume.data_xyz.shape),
            modality=volume.modality,
            geometry_resampled=volume.geometry_resampled,
        )


__all__ = [
    "DSIRExtraction",
    "FAITHFUL_V2_PAPER_EXPLICIT",
    "FAITHFUL_V2_REPRODUCTION_ASSUMPTIONS",
    "FAITHFUL_V2_VERSION",
    "Figure9FeatureExtractor",
    "FullResolutionDSIRExtractor",
    "MASRNetFaithfulV2",
    "MRCTPreprocessConfig",
    "StandardizedMRCTVolume",
    "canonical_mrct_modality",
    "cubic_bezier_lut_faithful_v2",
    "faithful_v2_provenance",
    "is_standard_mrct_geometry",
    "normalize_mrct_intensity",
    "sample_monotonic_cubic_control_points",
    "standardize_mrct_volume",
    "stochastic_nonlinear_transform_faithful_v2",
]
