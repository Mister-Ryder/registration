"""Physical-coordinate NIfTI loading for native 3-D PRA-CM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
from nibabel.processing import resample_from_to


@dataclass
class NiftiVolume:
    tensor: torch.Tensor
    domain: torch.Tensor
    affine: np.ndarray
    header: nib.nifti1.Nifti1Header
    source_path: str


@dataclass
class NiftiPair:
    fixed: NiftiVolume
    moving: NiftiVolume


def validate_plc_uint8_nifti(path: Union[str, Path]) -> None:
    """Reject already-normalized or scaled PLC files before float conversion."""

    image = nib.load(str(Path(path).expanduser().resolve()), mmap=True)
    if len(image.shape) != 3:
        raise ValueError("PLC input must be a verified scalar 3-D NIfTI.")
    if np.dtype(image.get_data_dtype()) != np.dtype(np.uint8):
        raise ValueError(
            f"PLC observed-scale input must have uint8 storage, got {image.get_data_dtype()}. "
            "A float [0,1] file would otherwise be divided by 255 twice."
        )
    slope, intercept = image.header.get_slope_inter()
    effective_slope = 1.0 if slope is None else float(slope)
    effective_intercept = 0.0 if intercept is None else float(intercept)
    if not np.isclose(effective_slope, 1.0) or not np.isclose(effective_intercept, 0.0):
        raise ValueError(
            "PLC observed-scale NIfTI must not carry a non-identity slope/intercept."
        )


def _to_tensor(array_xyz: np.ndarray) -> torch.Tensor:
    array = np.asarray(array_xyz)
    if array.ndim != 3:
        raise ValueError(f"Only scalar 3-D NIfTI is accepted, got {array.shape}.")
    return torch.from_numpy(np.transpose(array, (2, 1, 0)).copy()).float().unsqueeze(0).unsqueeze(0)


def _load_scalar(path: Union[str, Path]) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path), mmap=True)
    if len(image.shape) != 3:
        raise ValueError(
            f"{path} is not scalar 3-D. Replicated-component PLC inputs must use the verified scalar cache."
        )
    array = np.asarray(image.get_fdata(dtype=np.float32))
    if not np.isfinite(array).all():
        raise ValueError(f"Non-finite intensities in {path}.")
    affine = np.asarray(image.affine, dtype=np.float64)
    if affine.shape != (4, 4) or abs(np.linalg.det(affine[:3, :3])) < 1e-10:
        raise ValueError(f"Invalid NIfTI affine in {path}.")
    return image, array


def voxel_spacing_dzyx(affine: np.ndarray) -> torch.Tensor:
    """Return positive voxel spacing in tensor dzyx order from a NIfTI affine."""

    matrix = np.asarray(affine, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError("NIfTI affine must be 4x4.")
    spacing_xyz = np.linalg.norm(matrix[:3, :3], axis=0)
    if not np.isfinite(spacing_xyz).all() or (spacing_xyz <= 0).any():
        raise ValueError("NIfTI affine contains invalid voxel spacing.")
    return torch.from_numpy(spacing_xyz[::-1].copy()).float()


def load_volume(path: Union[str, Path]) -> NiftiVolume:
    resolved = Path(path).expanduser().resolve()
    image, array = _load_scalar(resolved)
    tensor = _to_tensor(array)
    return NiftiVolume(
        tensor,
        torch.ones_like(tensor, dtype=torch.bool),
        np.asarray(image.affine, dtype=np.float64).copy(),
        image.header.copy(),
        str(resolved),
    )


def resample_volume(
    source: NiftiVolume,
    reference: NiftiVolume,
    *,
    interpolation: str = "linear",
) -> NiftiVolume:
    if interpolation not in {"linear", "nearest"}:
        raise ValueError("interpolation must be 'linear' or 'nearest'.")
    target_shape_xyz = tuple(reversed(reference.tensor.shape[-3:]))
    if source.tensor.shape == reference.tensor.shape and np.array_equal(source.affine, reference.affine):
        return NiftiVolume(
            source.tensor.clone(),
            source.domain.clone(),
            reference.affine.copy(),
            reference.header.copy(),
            source.source_path,
        )
    source_xyz = np.transpose(source.tensor[0, 0].numpy(), (2, 1, 0))
    source_image = nib.Nifti1Image(source_xyz, source.affine, header=source.header.copy())
    sampled = resample_from_to(
        source_image,
        (target_shape_xyz, reference.affine),
        order=1 if interpolation == "linear" else 0,
        mode="constant",
        cval=0.0,
    )
    domain_xyz = np.transpose(source.domain[0, 0].numpy().astype(np.uint8), (2, 1, 0))
    sampled_domain = resample_from_to(
        nib.Nifti1Image(domain_xyz, source.affine),
        (target_shape_xyz, reference.affine),
        order=0,
        mode="constant",
        cval=0.0,
    )
    return NiftiVolume(
        _to_tensor(np.asarray(sampled.get_fdata(dtype=np.float32))),
        _to_tensor(np.asarray(sampled_domain.dataobj) > 0).bool(),
        reference.affine.copy(),
        reference.header.copy(),
        source.source_path,
    )


def load_pair_on_fixed_grid(
    fixed_path: Union[str, Path], moving_path: Union[str, Path]
) -> NiftiPair:
    fixed = load_volume(fixed_path)
    moving = resample_volume(load_volume(moving_path), fixed)
    return NiftiPair(fixed, moving)


def l2r_foreground_mask(
    tensor: torch.Tensor,
    modality: str,
    *,
    domain: Optional[torch.Tensor] = None,
    ct_body_threshold_hu: float = -500.0,
    background_epsilon: float = 1.0e-6,
) -> torch.Tensor:
    """Build an L2R MR/CT content mask from raw intensities.

    The public auxiliary volumes contain two distinct kinds of non-content:
    exact-zero resampling/crop padding in both modalities, and approximately
    -1000 HU air around the CT body. Neither can be identified after z-score
    normalization because normalization maps them to ordinary positive values.
    This function must therefore be called before normalize_intensity.

    The mask is label-free. It is deliberately conservative: exact-zero MR
    background is excluded, while CT support must be nonzero and lie above a
    broad body/air threshold. domain still limits the physical FOV after a
    volume has been resampled to another grid.
    """

    value = tensor.float()
    valid = torch.ones_like(value, dtype=torch.bool) if domain is None else domain.bool()
    if valid.shape != value.shape:
        raise ValueError("L2R raw-intensity domain must match the image shape.")
    identity = str(modality).strip().lower()
    nonpadding = value.abs() > float(background_epsilon)
    if identity == "mr":
        foreground = valid & nonpadding
    elif identity == "ct":
        foreground = valid & nonpadding & (value > float(ct_body_threshold_hu))
    else:
        raise ValueError("L2R foreground masking accepts only 'mr' or 'ct'.")
    if foreground.sum().item() < 2:
        raise ValueError(f"{identity.upper()} volume has insufficient raw foreground support.")
    return foreground


def normalize_intensity(
    tensor: torch.Tensor,
    mode: str,
    *,
    hu_window: Tuple[float, float] = (-200.0, 400.0),
    domain: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    value = tensor.float()
    if mode == "plc_uint8":
        if value.min() < 0 or value.max() > 255:
            raise ValueError("plc_uint8 input lies outside the frozen observed [0,255] scale.")
        result = value / 255.0
    elif mode == "hu_window":
        low, high = map(float, hu_window)
        if not low < high:
            raise ValueError("HU window must be increasing.")
        result = (value.clamp(low, high) - low) / (high - low)
    elif mode == "zscore":
        mask = torch.ones_like(value, dtype=torch.bool) if domain is None else domain.bool()
        selected = value[mask]
        if selected.numel() < 2:
            raise ValueError("zscore input has insufficient valid support.")
        normalized = (value - selected.mean()) / selected.std().clamp_min(1e-6)
        result = ((normalized.clamp(-3, 3) + 3) / 6).clamp(0, 1)
    else:
        raise ValueError(f"Unknown intensity mode {mode!r}.")
    return result if domain is None else result.masked_fill(~domain.bool(), 0)


def tensor_to_xyz(tensor: torch.Tensor) -> np.ndarray:
    value = tensor.detach().cpu()
    if value.ndim == 5:
        if value.shape[:2] != (1, 1):
            raise ValueError("Scalar export requires [1,1,D,H,W].")
        value = value[0, 0]
    if value.ndim != 3:
        raise ValueError("Scalar export requires a 3-D tensor.")
    return np.transpose(value.numpy(), (2, 1, 0)).copy()
