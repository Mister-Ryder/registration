"""Label-free image inventory, modality preprocessing and raw foreground masks."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nibabel.processing import resample_from_to

from registration_benchmark.contract import load_registration_manifest
from registration_benchmark.dns.faithful_v2 import (
    MRCTPreprocessConfig,
    canonical_mrct_modality,
    standardize_mrct_volume,
)


@dataclass(frozen=True, order=True)
class Anchor:
    path: Path
    modality: str


@dataclass(frozen=True)
class AnchorInventory:
    train: Tuple[Anchor, ...]
    validation: Tuple[Anchor, ...]
    manifest_sha256: str


@dataclass(frozen=True)
class PreparedVolume:
    image_dzyx: torch.Tensor
    foreground_dzyx: torch.Tensor
    affine_xyz: np.ndarray
    modality: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_anchor_inventory(manifest: Path) -> AnchorInventory:
    """Deduplicate standalone train/validation images without using labels."""

    resolved = manifest.expanduser().resolve()
    tasks = load_registration_manifest(resolved, require_files=True)
    by_split = {"train": {}, "validation": {}}
    for task in tasks:
        if task.split not in by_split:
            continue
        for image in (task.fixed, task.moving):
            path = image.path.resolve()
            modality = canonical_mrct_modality(image.modality)
            previous = by_split[task.split].get(path)
            if previous is not None and previous != modality:
                raise ValueError(f"One image has conflicting modalities: {path}")
            by_split[task.split][path] = modality
    train = tuple(sorted(Anchor(path, modality) for path, modality in by_split["train"].items()))
    validation = tuple(
        sorted(Anchor(path, modality) for path, modality in by_split["validation"].items())
    )
    if not train or not validation:
        raise ValueError("V4-final needs non-empty, label-free train and validation anchors.")
    overlap = {anchor.path for anchor in train}.intersection(anchor.path for anchor in validation)
    if overlap:
        raise ValueError(f"Train/validation image leakage: {sorted(overlap)[:3]}")
    return AnchorInventory(train, validation, sha256_file(resolved))


def raw_foreground_xyz(
    data_xyz: np.ndarray,
    modality: str,
    *,
    ct_min_hu: float = -500.0,
) -> np.ndarray:
    """Compute the core-fix mask before any z-score/window normalization."""

    values = np.asarray(data_xyz)
    if values.ndim != 3 or not np.isfinite(values).all():
        raise ValueError("Foreground masking expects one finite 3-D raw volume.")
    canonical = canonical_mrct_modality(modality)
    nonzero = np.abs(values) > np.finfo(np.float32).eps
    if canonical == "ct":
        return (nonzero & (values > float(ct_min_hu))).astype(np.uint8)
    return nonzero.astype(np.uint8)


@lru_cache(maxsize=32)
def _prepare_cached(
    path_text: str,
    modality: str,
    ct_min_hu: float,
) -> PreparedVolume:
    path = Path(path_text)
    image = nib.load(str(path), mmap=True)
    raw = np.asarray(image.dataobj, dtype=np.float32)
    while raw.ndim > 3 and raw.shape[-1] == 1:
        raw = raw[..., 0]
    if raw.ndim != 3 or not np.isfinite(raw).all():
        raise ValueError(f"Expected one finite scalar NIfTI: {path}")
    preprocess = MRCTPreprocessConfig()
    standardized = standardize_mrct_volume(raw, image.affine, modality, preprocess)
    foreground = raw_foreground_xyz(raw, modality, ct_min_hu=ct_min_hu)
    if standardized.geometry_resampled:
        mask_image = nib.Nifti1Image(foreground.astype(np.float32), image.affine)
        mask_image = resample_from_to(
            mask_image,
            (standardized.data_xyz.shape, standardized.affine_xyz),
            order=0,
            mode="constant",
            cval=0.0,
        )
        foreground = np.asarray(mask_image.dataobj) > 0.5
    else:
        foreground = foreground > 0
    normalized = np.asarray(standardized.data_xyz, dtype=np.float32)
    normalized[~foreground] = 0.0
    image_dzyx = torch.from_numpy(normalized.transpose(2, 1, 0).copy())[None]
    mask_dzyx = torch.from_numpy(foreground.transpose(2, 1, 0).copy())[None].bool()
    if int(mask_dzyx.sum()) < 2:
        raise ValueError(f"Raw foreground mask is empty: {path}")
    return PreparedVolume(
        image_dzyx=image_dzyx,
        foreground_dzyx=mask_dzyx,
        affine_xyz=np.asarray(standardized.affine_xyz, dtype=np.float64),
        modality=canonical_mrct_modality(modality),
    )


def prepare_anchor(anchor: Anchor, *, ct_min_hu: float = -500.0) -> PreparedVolume:
    return _prepare_cached(str(anchor.path.resolve()), anchor.modality, float(ct_min_hu))


def _pad_to_shape(value: torch.Tensor, shape: Sequence[int], fill: float) -> torch.Tensor:
    pads = []
    for current, wanted in zip(reversed(value.shape[-3:]), reversed(shape)):
        total = max(int(wanted) - int(current), 0)
        pads.extend((total // 2, total - total // 2))
    return F.pad(value, tuple(pads), mode="constant", value=float(fill)) if any(pads) else value


def deterministic_foreground_crop(
    volume: PreparedVolume,
    shape_dzyx: Sequence[int],
    *,
    seed: int,
    minimum_foreground_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Crop a reproducible full-resolution patch with non-trivial anatomy."""

    target = tuple(int(value) for value in shape_dzyx)
    image = _pad_to_shape(volume.image_dzyx, target, 0.0)
    mask = _pad_to_shape(volume.foreground_dzyx.float(), target, 0.0).bool()
    rng = random.Random(int(seed))
    best = None
    best_fraction = -1.0
    for _ in range(32):
        starts = tuple(
            rng.randint(0, int(current) - wanted) if int(current) > wanted else 0
            for current, wanted in zip(image.shape[-3:], target)
        )
        slices = tuple(slice(start, start + wanted) for start, wanted in zip(starts, target))
        candidate_mask = mask[(...,) + slices]
        fraction = float(candidate_mask.float().mean())
        if fraction > best_fraction:
            best = (slices, candidate_mask)
            best_fraction = fraction
        if fraction >= float(minimum_foreground_fraction):
            break
    assert best is not None
    slices, selected_mask = best
    if best_fraction < float(minimum_foreground_fraction):
        raise RuntimeError(
            f"Could not sample foreground crop: {best_fraction:.4f} < "
            f"{minimum_foreground_fraction:.4f}"
        )
    selected_image = image[(...,) + slices].masked_fill(~selected_mask, 0.0)
    return selected_image.contiguous(), selected_mask.contiguous()


def distributed_epoch_anchors(
    anchors: Sequence[Anchor],
    *,
    epoch: int,
    seed: int,
    rank: int,
    world_size: int,
) -> tuple[Tuple[Anchor, ...], int]:
    """Shuffle once globally, pad deterministically, then shard by rank."""

    if not anchors or world_size < 1 or not 0 <= rank < world_size:
        raise ValueError("Invalid distributed anchor request.")
    order = list(anchors)
    random.Random(int(seed) + int(epoch) * 1_000_003).shuffle(order)
    repeat_count = (-len(order)) % int(world_size)
    for offset in range(repeat_count):
        order.append(order[(epoch + offset) % len(order)])
    return tuple(order[rank::world_size]), repeat_count


__all__ = [
    "Anchor",
    "AnchorInventory",
    "PreparedVolume",
    "deterministic_foreground_crop",
    "distributed_epoch_anchors",
    "load_anchor_inventory",
    "prepare_anchor",
    "raw_foreground_xyz",
    "sha256_file",
]

