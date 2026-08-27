"""NIfTI loading and deterministic moving-to-fixed resampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


@dataclass(frozen=True)
class Volume:
    data_xyz: np.ndarray
    affine: np.ndarray
    path: Path

    @property
    def shape_xyz(self) -> Tuple[int, int, int]:
        return tuple(int(v) for v in self.data_xyz.shape)  # type: ignore[return-value]


def load_scalar(path: Path, dtype=np.float32) -> Volume:
    image = nib.load(str(path), mmap=True)
    data = np.asanyarray(image.dataobj)
    while data.ndim > 3 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected scalar 3-D NIfTI, got {data.shape} for {path}.")
    data = np.asarray(data, dtype=dtype)
    if not np.isfinite(data).all():
        raise ValueError(f"Non-finite values in {path}.")
    return Volume(data, np.asarray(image.affine, dtype=np.float64), path.resolve())


def load_pair_on_fixed_grid(fixed_path: Path, moving_path: Path, order: int = 1) -> Tuple[Volume, Volume]:
    fixed_image = nib.load(str(fixed_path), mmap=True)
    moving_image = nib.load(str(moving_path), mmap=True)
    fixed = load_scalar(fixed_path)
    if moving_image.shape[:3] == fixed_image.shape[:3] and np.allclose(moving_image.affine, fixed_image.affine, atol=1e-5):
        moving = load_scalar(moving_path)
    else:
        resampled = resample_from_to(moving_image, (fixed_image.shape[:3], fixed_image.affine), order=order)
        data = np.asarray(resampled.dataobj, dtype=np.float32)
        if data.ndim != 3 or not np.isfinite(data).all():
            raise ValueError(f"Invalid resampled moving image: {moving_path}")
        moving = Volume(data, fixed.affine.copy(), moving_path.resolve())
    return fixed, moving


def robust_unit_interval(data: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    values = np.asarray(data, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Cannot normalize an empty/non-finite image.")
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo + 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def save_nifti(data_xyz: np.ndarray, affine: np.ndarray, path: Path, dtype=np.float32) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data_xyz, dtype=dtype), affine), str(path))

