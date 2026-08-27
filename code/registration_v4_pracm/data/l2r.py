"""Label-free Learn2Reg MR/CT domain sampling for PRA-CM.

The auxiliary L2R images are explicitly *unpaired*. They are therefore used
as two independent image domains. No arbitrary MR/CT cross-product is ever
treated as an anatomical registration target: exact correspondence supervision
is generated later, independently within each native-grid volume.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import DataConfig
from .nifti import (
    l2r_foreground_mask,
    load_volume,
    normalize_intensity,
    voxel_spacing_dzyx,
)


class L2RMRCTDataset(Dataset):
    """Sample independent native-grid MR and CT patches without labels."""

    _MANIFEST_SCHEMA = "registration_pair_manifest_v1"

    def __init__(
        self,
        config: DataConfig,
        *,
        split: str,
        seed: int,
        samples: Optional[int] = None,
    ) -> None:
        if config.dataset != "l2r_mrct":
            raise ValueError("L2RMRCTDataset requires data.dataset=l2r_mrct.")
        self.config = config
        self.split = split.lower()
        self.seed = int(seed)
        self.epoch = 0
        self.manifest_path = Path(config.manifest).expanduser().resolve()
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema") != self._MANIFEST_SCHEMA:
            raise ValueError(
                f"PRA-CM L2R training requires {self._MANIFEST_SCHEMA}, "
                f"got {payload.get('schema')!r}."
            )

        pools: dict[str, dict[str, Path]] = {"mr": {}, "ct": {}}
        observed_splits: dict[tuple[str, str], set[str]] = {}
        for raw in payload.get("pairs", []):
            raw_split = str(raw.get("split", "")).strip().lower()
            fixed, moving = raw.get("fixed", {}), raw.get("moving", {})
            identities = (
                str(fixed.get("modality", "")).lower(),
                str(moving.get("modality", "")).lower(),
            )
            if identities != ("mr", "ct"):
                raise ValueError("Every L2R manifest row must describe MR <- CT domains.")
            for modality, image in zip(identities, (fixed, moving)):
                path = self._resolve(image["path"])
                key = path.as_posix()
                observed_splits.setdefault((modality, key), set()).add(raw_split)
                if raw_split == self.split:
                    pools[modality][key] = path

        leakage = [
            (modality, path, sorted(splits))
            for (modality, path), splits in observed_splits.items()
            if len(splits) > 1
        ]
        if leakage:
            raise ValueError(
                "L2R auxiliary images leak across train/validation splits: "
                f"{leakage[:3]}."
            )
        self.pools = {
            modality: tuple(path for _, path in sorted(values.items()))
            for modality, values in pools.items()
        }
        if not self.pools["mr"] or not self.pools["ct"]:
            raise ValueError(
                f"No independent MR/CT {self.split} domain pools in {self.manifest_path}."
            )
        self.samples = int(samples if samples is not None else config.samples_per_epoch)
        if self.samples < 1:
            raise ValueError("L2R dataset samples must be positive.")
        self._cache: OrderedDict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = OrderedDict()

    def _resolve(self, value: str) -> Path:
        path = Path(str(value))
        path = path if path.is_absolute() else self.manifest_path.parent / path
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def source_identity(self, *, include_observed_inputs: bool = False) -> Mapping[str, object]:
        return {
            "protocol": "L2R-MRCT-unpaired-native-synthetic-correspondence-v3-raw-foreground",
            "manifest": str(self.manifest_path),
            "manifest_sha256": hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
            "dataset_split": self.split,
            "mr_volume_count": len(self.pools["mr"]),
            "ct_volume_count": len(self.pools["ct"]),
            "sample_count": self.samples,
            "cross_modal_pair_supervision": False,
            "native_grid_sampling": True,
            "foreground_from_raw_intensity": True,
            "ct_body_threshold_hu": -500.0,
            "exact_zero_padding_excluded": True,
            "segmentation_labels_loaded": False,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _crop_pad(value: torch.Tensor, start: Sequence[int], size: Sequence[int]) -> torch.Tensor:
        slices, pads = [], []
        for begin, length, available in zip(start, size, value.shape[-3:]):
            end = begin + length
            slices.append(slice(max(begin, 0), min(end, available)))
            pads.append((max(0, -begin), max(0, end - available)))
        result = value[(..., *slices)]
        return F.pad(
            result,
            (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]),
        )

    def _load(
        self, path: Path, modality: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = f"{modality}:{path.as_posix()}"
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        volume = load_volume(path)
        foreground = l2r_foreground_mask(
            volume.tensor,
            modality,
            domain=volume.domain,
        )
        image = normalize_intensity(
            volume.tensor,
            self.config.intensity_mode,
            hu_window=self.config.hu_window,
            domain=foreground,
        )[0]
        value = (image, foreground[0], voxel_spacing_dzyx(volume.affine))
        self._cache[key] = value
        while len(self._cache) > self.config.cache_patients:
            self._cache.popitem(last=False)
        return value

    def _patch_one(
        self, loaded: tuple[torch.Tensor, torch.Tensor, torch.Tensor], rng: random.Random
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, domain, spacing = loaded
        content = domain & (image.abs() > self.config.foreground_threshold)
        coordinates = content[0].nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            coordinates = domain[0].nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("L2R auxiliary volume has no valid physical support.")
        best = 0.0
        for _ in range(min(16, coordinates.shape[0])):
            centre = coordinates[rng.randrange(coordinates.shape[0])]
            start = [
                int(centre[i]) - self.config.patch_size[i] // 2 for i in range(3)
            ]
            image_patch = self._crop_pad(image, start, self.config.patch_size)
            support = self._crop_pad(domain.float(), start, self.config.patch_size) > 0.5
            content_fraction = float(
                (support & (image_patch.abs() > self.config.foreground_threshold))
                .float()
                .mean()
            )
            best = max(best, content_fraction)
            if content_fraction >= self.config.minimum_content_fraction:
                return image_patch.masked_fill(~support, 0), support, spacing
        raise RuntimeError(
            "No valid native-grid L2R patch after bounded retries; "
            f"best content fraction={best:.4f}."
        )

    def __getitem__(self, index: int):
        index = int(index)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        mr_path = self.pools["mr"][(index + self.epoch * 104729) % len(self.pools["mr"])]
        ct_path = self.pools["ct"][(index * 65537 + self.epoch * 130363) % len(self.pools["ct"])]
        mr, mr_domain, mr_spacing = self._patch_one(self._load(mr_path, "mr"), rng)
        ct, ct_domain, ct_spacing = self._patch_one(self._load(ct_path, "ct"), rng)
        return {
            "kind": "unpaired_domains",
            "patient_id": f"unpaired__{mr_path.stem}__{ct_path.stem}",
            "sample_id": index,
            "augmentation_seed": self.seed + index,
            "mr": mr,
            "ct": ct,
            "mr_domain": mr_domain,
            "ct_domain": ct_domain,
            "mr_spacing_dzyx": mr_spacing,
            "ct_spacing_dzyx": ct_spacing,
        }
