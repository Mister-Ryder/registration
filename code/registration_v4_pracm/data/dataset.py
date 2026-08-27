"""Four-phase PLC-R volume sampling with a frozen cohort and optional validation labels."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ..config import DataConfig
from .nifti import (
    load_volume,
    normalize_intensity,
    resample_volume,
    validate_plc_uint8_nifti,
    voxel_spacing_dzyx,
)


PHASE_SOURCES = (("P", "p"), ("C1", "c1"), ("C2", "c2"), ("C3", "c3"))
MAIN_DIRECTIONS = (("p", "c1"), ("p", "c2"), ("p", "c3"))
RELATIONAL_TRIANGLES = (("p", "c1", "c2"), ("p", "c2", "c3"))
EXPECTED_SPLIT_COUNTS = {"train": 172, "validation": 29, "test": 49}


class PLCRVolumeDataset(Dataset):
    """Patient-balanced P<-Ci pairs and P/Ci/Cj relational triangles.

    CT intensities never use segmentation labels.  When ``load_labels`` is true,
    liver masks are returned solely for validation-time checkpoint selection.
    """

    def __init__(
        self,
        config: DataConfig,
        *,
        split: str,
        samples: Optional[int] = None,
        seed: int = 20260823,
        load_labels: bool = False,
        pair_only: bool = False,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("Training code exposes only frozen train/validation splits.")
        self.config = config
        self.split = split
        self.seed = int(seed)
        self.epoch = 0
        self.load_labels = bool(load_labels)
        self.pair_only = bool(pair_only)
        self.manifest_path = Path(config.manifest).expanduser().resolve()
        if not config.phase_inventory:
            raise ValueError("data.phase_inventory is required for the independent P phase.")
        self.inventory_path = Path(config.phase_inventory).expanduser().resolve()
        self.data_root = Path(config.data_root).expanduser().resolve()
        self.scalar_cache = (
            None if not config.scalar_cache else Path(config.scalar_cache).expanduser().resolve()
        )
        payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if payload.get("protocol_id") != "PLC-R" or payload.get("status") != "frozen":
            raise ValueError("Training accepts only a frozen PLC-R manifest.")
        embedded_hash = str(payload.get("manifest_content_sha256", ""))
        unhashed = dict(payload)
        unhashed.pop("manifest_content_sha256", None)
        observed_hash = hashlib.sha256(
            json.dumps(
                unhashed, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if not embedded_hash or observed_hash != embedded_hash:
            raise ValueError("PLC-R manifest embedded content hash is invalid.")
        self.manifest_content_sha256 = embedded_hash
        self.manifest_file_sha256 = self._sha256(self.manifest_path)
        self.manifest_sources = payload.get("sources", {})
        self.inventory_sha256 = self._sha256(self.inventory_path)
        expected_inventory_hash = str(
            self.manifest_sources.get("inventory", {}).get("sha256", "")
        )
        if not expected_inventory_hash or self.inventory_sha256 != expected_inventory_hash:
            raise ValueError("The four-phase inventory does not match the frozen manifest source.")

        patients = payload.get("patients")
        if not isinstance(patients, list):
            raise ValueError("PLC-R manifest must contain a patients list.")
        eligible = [patient for patient in patients if patient.get("eligible") is True]
        split_counts = Counter(str(patient.get("split")) for patient in eligible)
        if len(eligible) != sum(EXPECTED_SPLIT_COUNTS.values()) or dict(split_counts) != EXPECTED_SPLIT_COUNTS:
            raise ValueError(
                f"Frozen cohort must be exactly 250 patients with splits {EXPECTED_SPLIT_COUNTS}; "
                f"observed {dict(split_counts)}."
            )
        self._inventory = self._load_inventory(eligible)
        self.patients = tuple(
            patient
            for patient in eligible
            if str(patient.get("split")) == split
            and patient.get("complete_triplet") is True
        )
        if len(self.patients) != EXPECTED_SPLIT_COUNTS[split]:
            raise ValueError(
                f"Expected {EXPECTED_SPLIT_COUNTS[split]} eligible {split} patients, "
                f"got {len(self.patients)}."
            )
        self.samples = int(samples or config.samples_per_epoch)
        self._cache: OrderedDict[
            str, Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
        ] = OrderedDict()
        self._verified_paths: set[Path] = set()
        self._observed_input_hashes: Dict[str, str] = {}

    def _load_inventory(self, eligible) -> Dict[str, Dict[str, Mapping[str, str]]]:
        selected_ids = {str(patient["patient_id"]) for patient in eligible}
        result: Dict[str, Dict[str, Mapping[str, str]]] = {
            patient_id: {} for patient_id in selected_ids
        }
        with self.inventory_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                patient_id = str(row.get("patient_id", "")).strip()
                source_phase = str(row.get("phase", "")).strip().upper()
                if patient_id not in selected_ids or source_phase not in {p[0] for p in PHASE_SOURCES}:
                    continue
                if source_phase in result[patient_id]:
                    raise ValueError(f"Duplicate inventory row for {patient_id}/{source_phase}.")
                if str(row.get("error", "")).strip():
                    raise ValueError(f"Inventory reports an error for {patient_id}/{source_phase}.")
                if not str(row.get("ct_path", "")).strip():
                    raise ValueError(f"Inventory lacks CT path for {patient_id}/{source_phase}.")
                if self.load_labels and not str(row.get("liver_path", "")).strip():
                    raise ValueError(f"Inventory lacks liver mask for {patient_id}/{source_phase}.")
                if self.load_labels and str(row.get("ct_liver_geometry_match", "")).lower() != "true":
                    raise ValueError(
                        f"Inventory does not verify CT/liver geometry for {patient_id}/{source_phase}."
                    )
                result[patient_id][source_phase] = dict(row)
        expected_phases = {phase for phase, _ in PHASE_SOURCES}
        incomplete = {
            patient_id: sorted(expected_phases.difference(rows))
            for patient_id, rows in result.items()
            if set(rows) != expected_phases
        }
        if incomplete:
            preview = dict(list(sorted(incomplete.items()))[:5])
            raise ValueError(f"Frozen cohort is not complete for P/C1/C2/C3: {preview}.")

        for patient in eligible:
            patient_id = str(patient["patient_id"])
            manifest_phases = patient.get("phases")
            if not isinstance(manifest_phases, Mapping):
                raise ValueError(f"Patient {patient_id} has no manifest phase mapping.")
            for source_phase in ("C1", "C2", "C3"):
                manifest_entry = manifest_phases.get(source_phase)
                inventory_entry = result[patient_id][source_phase]
                if not isinstance(manifest_entry, Mapping):
                    raise ValueError(f"Manifest lacks {patient_id}/{source_phase}.")
                if str(manifest_entry.get("source_path", "")).replace("\\", "/") != str(
                    inventory_entry.get("ct_path", "")
                ).replace("\\", "/"):
                    raise ValueError(f"Manifest/inventory CT path mismatch for {patient_id}/{source_phase}.")
        return result

    def source_identity(self, *, include_observed_inputs: bool = False) -> Mapping[str, object]:
        identity: Dict[str, object] = {
            "protocol": "PLC-R-four-phase-star-v1",
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_content_sha256": self.manifest_content_sha256,
            "phase_inventory_sha256": self.inventory_sha256,
            "cohort_size": 250,
            "split_counts": dict(EXPECTED_SPLIT_COUNTS),
            "phases": ["P", "C1", "C2", "C3"],
            "main_directions_fixed_moving": [list(item) for item in MAIN_DIRECTIONS],
            "relational_triangles": [list(item) for item in RELATIONAL_TRIANGLES],
            "dataset_split": self.split,
            "dataset_patients": len(self.patients),
            "labels_loaded": self.load_labels,
            "pair_only": self.pair_only,
            "label_centered_patches": self.load_labels and self.pair_only,
        }
        if include_observed_inputs:
            identity["observed_input_sha256"] = dict(sorted(self._observed_input_hashes.items()))
        return identity

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _safe_relative_path(relative: str) -> Path:
        normalized = relative.replace("\\", "/").strip()
        candidate = Path(normalized)
        if not normalized or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError(f"Unsafe PLC-R relative path {relative!r}.")
        return candidate

    def _source_entry(
        self, patient: Mapping[str, object], source_phase: str
    ) -> Mapping[str, object]:
        patient_id = str(patient["patient_id"])
        inventory = self._inventory[patient_id][source_phase]
        if source_phase == "P":
            return {
                "source_path": inventory["ct_path"],
                "scalar_layout": inventory.get("ct_scalar_layout_verified")
                or inventory.get("ct_scalar_layout")
                or "native_3d",
            }
        manifest_entry = patient["phases"][source_phase]
        return manifest_entry

    def _source_path(
        self,
        entry: Mapping[str, object],
        *,
        patient_id: str,
        source_phase: str,
    ) -> Path:
        relative = self._safe_relative_path(str(entry.get("source_path", "")))
        layout = str(entry.get("scalar_layout", "native_3d"))
        root = self.data_root if layout == "native_3d" else self.scalar_cache
        if root is None:
            raise ValueError(
                "Replicated-component PLC input requires data.scalar_cache built by registration_v3."
            )
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Resolved PLC-R path escapes its configured root.") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        if path not in self._verified_paths:
            expected_source = str(entry.get("source_sha256", ""))
            observed_source = self._sha256(path)
            self._observed_input_hashes[f"{patient_id}/{source_phase}"] = observed_source
            if layout == "native_3d":
                if expected_source and observed_source != expected_source:
                    raise ValueError(
                        f"Frozen source checksum changed for {patient_id}/{source_phase}."
                    )
            else:
                provenance_path = path.with_suffix(path.suffix + ".provenance.json")
                if not provenance_path.is_file():
                    raise FileNotFoundError(
                        f"Scalar cache lacks provenance for {patient_id}/{source_phase}: {provenance_path}"
                    )
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                sources = self.manifest_sources
                expected = {
                    "patient_id": patient_id,
                    "source_phase": source_phase,
                    "source_sha256": expected_source,
                    "source_relative_path": relative.as_posix(),
                    "inventory_sha256": str(sources.get("inventory", {}).get("sha256", "")),
                    "split_sha256": str(sources.get("split", {}).get("embedded_sha256", "")),
                    "exclusions_sha256": str(
                        sources.get("quality_exclusions", {}).get("embedded_sha256", "")
                    ),
                }
                for name, value in expected.items():
                    if value and str(provenance.get(name, "")) != value:
                        raise ValueError(
                            f"Scalar-cache provenance mismatch for {patient_id}/{source_phase}: {name}."
                        )
                if observed_source != str(provenance.get("output_sha256", "")):
                    raise ValueError(
                        f"Scalar-cache checksum mismatch for {patient_id}/{source_phase}."
                    )
            self._verified_paths.add(path)
        if self.config.intensity_mode == "plc_uint8":
            validate_plc_uint8_nifti(path)
        return path

    def _mask_path(self, patient_id: str, source_phase: str) -> Path:
        row = self._inventory[patient_id][source_phase]
        relative = self._safe_relative_path(str(row.get("liver_path", "")))
        if str(row.get("liver_scalar_layout_verified", "native_3d")) != "native_3d":
            raise ValueError(f"Liver mask must be scalar 3-D for {patient_id}/{source_phase}.")
        path = (self.data_root / relative).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError("Resolved liver-mask path escapes data.data_root.") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @staticmethod
    def _validate_inventory_geometry(volume, row, patient_id: str, source_phase: str) -> None:
        expected_shape = tuple(int(row[f"ct_size_{axis}"]) for axis in ("x", "y", "z"))
        observed_shape = tuple(reversed(volume.tensor.shape[-3:]))
        if observed_shape != expected_shape:
            raise ValueError(
                f"Inventory shape changed for {patient_id}/{source_phase}: "
                f"expected {expected_shape}, observed {observed_shape}."
            )
        affine_text = str(row.get("ct_affine_json", "")).strip()
        if affine_text:
            expected_affine = np.asarray(json.loads(affine_text), dtype=np.float64)
            if expected_affine.shape != (4, 4) or not np.allclose(
                volume.affine, expected_affine, rtol=0.0, atol=1.0e-6
            ):
                raise ValueError(f"Inventory affine changed for {patient_id}/{source_phase}.")

    def _load_patient(self, patient: Mapping[str, object]):
        patient_id = str(patient["patient_id"])
        if patient_id in self._cache:
            value = self._cache.pop(patient_id)
            self._cache[patient_id] = value
            return value
        reference_entry = self._source_entry(patient, "P")
        reference = load_volume(
            self._source_path(reference_entry, patient_id=patient_id, source_phase="P")
        )
        self._validate_inventory_geometry(
            reference, self._inventory[patient_id]["P"], patient_id, "P"
        )
        spacing = voxel_spacing_dzyx(reference.affine)
        loaded: Dict[str, Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]] = {}
        for source_phase, phase in PHASE_SOURCES:
            entry = self._source_entry(patient, source_phase)
            if source_phase == "P":
                volume = reference
            else:
                native_volume = load_volume(
                    self._source_path(entry, patient_id=patient_id, source_phase=source_phase)
                )
                self._validate_inventory_geometry(
                    native_volume,
                    self._inventory[patient_id][source_phase],
                    patient_id,
                    source_phase,
                )
                volume = resample_volume(native_volume, reference)
            tensor = normalize_intensity(
                volume.tensor,
                self.config.intensity_mode,
                hu_window=self.config.hu_window,
                domain=volume.domain,
            )[0]
            label = None
            if self.load_labels:
                mask = load_volume(self._mask_path(patient_id, source_phase))
                if source_phase != "P":
                    mask = resample_volume(mask, reference, interpolation="nearest")
                label = (mask.tensor[0] > 0.5) & volume.domain[0]
            loaded[phase] = (tensor, volume.domain[0], label, spacing)
        self._cache[patient_id] = loaded
        while len(self._cache) > max(1, self.config.cache_patients):
            self._cache.popitem(last=False)
        return loaded

    @staticmethod
    def _crop_pad(value: torch.Tensor, start: Sequence[int], size: Sequence[int]) -> torch.Tensor:
        spatial = value.shape[-3:]
        slices = []
        pads = []
        for begin, length, available in zip(start, size, spatial):
            end = begin + length
            source_begin = max(begin, 0)
            source_end = min(end, available)
            slices.append(slice(source_begin, source_end))
            pads.append((max(0, -begin), max(0, end - available)))
        result = value[(..., *slices)]
        return F.pad(
            result,
            (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]),
        )

    def _patch(self, volumes, rng: random.Random, required_phases: Sequence[str]):
        intersection = torch.stack([volumes[p][1] for p in required_phases]).all(dim=0)
        content = torch.stack([volumes[p][0].abs() for p in required_phases]).amax(dim=0)
        eligible = intersection & (content > self.config.foreground_threshold)
        coordinates = eligible[0].nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            coordinates = intersection[0].nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("Required patient phases have no common physical support.")
        preferred_centre = None
        if self.load_labels and self.pair_only:
            fixed_label = volumes[required_phases[0]][2]
            label_coordinates = (intersection & fixed_label)[0].nonzero(as_tuple=False)
            if label_coordinates.numel() == 0:
                raise ValueError("Validation patch has no fixed-label support in the common domain.")
            preferred_centre = label_coordinates.float().mean(dim=0).round().long()
        best_fraction = 0.0
        attempts = min(16, coordinates.shape[0])
        for attempt in range(attempts):
            centre = (
                preferred_centre
                if attempt == 0 and preferred_centre is not None
                else coordinates[rng.randrange(coordinates.shape[0])]
            )
            start = [int(centre[i]) - self.config.patch_size[i] // 2 for i in range(3)]
            required_domains = [
                self._crop_pad(volumes[p][1].float(), start, self.config.patch_size) > 0.5
                for p in required_phases
            ]
            common_fraction = float(
                torch.stack(required_domains).all(dim=0).float().mean()
            )
            best_fraction = max(best_fraction, common_fraction)
            if common_fraction < self.config.minimum_content_fraction:
                continue
            result = {}
            for phase, (image, domain, label, spacing) in volumes.items():
                result[phase] = (
                    self._crop_pad(image, start, self.config.patch_size),
                    self._crop_pad(domain.float(), start, self.config.patch_size) > 0.5,
                    None
                    if label is None
                    else self._crop_pad(label.float(), start, self.config.patch_size) > 0.5,
                    spacing,
                )
            return result
        raise RuntimeError(
            f"No valid patch after bounded retries; best common-domain fraction "
            f"{best_fraction:.4f} is below the configured minimum."
        )

    def __getitem__(self, index: int):
        index = int(index)
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        patient = self.patients[(index + self.epoch * 104729) % len(self.patients)]
        schedule_index = index + self.epoch * self.samples
        is_triplet = False if self.pair_only else rng.random() < self.config.triplet_probability
        if is_triplet:
            phases = RELATIONAL_TRIANGLES[schedule_index % len(RELATIONAL_TRIANGLES)]
            required_phases = phases
        else:
            fixed_phase, moving_phase = MAIN_DIRECTIONS[schedule_index % len(MAIN_DIRECTIONS)]
            required_phases = (fixed_phase, moving_phase)
        patch = self._patch(self._load_patient(patient), rng, required_phases)
        patient_id = str(patient["patient_id"])
        if is_triplet:
            result = {
                "kind": "triplet",
                "patient_id": patient_id,
                "first": patch[phases[0]][0],
                "second": patch[phases[1]][0],
                "third": patch[phases[2]][0],
                "phases": phases,
                "spacing_dzyx": patch[phases[0]][3],
                "domains": tuple(patch[phase][1] for phase in phases),
            }
            if self.load_labels:
                result["labels"] = tuple(patch[phase][2] for phase in phases)
            return result
        result = {
            "kind": "pair",
            "patient_id": patient_id,
            "fixed": patch[fixed_phase][0],
            "moving": patch[moving_phase][0],
            "fixed_phase": fixed_phase,
            "moving_phase": moving_phase,
            "fixed_domain": patch[fixed_phase][1],
            "spacing_dzyx": patch[fixed_phase][3],
            "moving_domain": patch[moving_phase][1],
        }
        if self.load_labels:
            result["fixed_label"] = patch[fixed_phase][2]
            result["moving_label"] = patch[moving_phase][2]
        return result


def single_case_collate(items):
    if len(items) != 1:
        raise ValueError("PRA-CM's variable pair/triplet collate requires batch_size=1.")
    sample = items[0]
    result = dict(sample)
    if sample["kind"] == "pair":
        for key in (
            "fixed",
            "moving",
            "fixed_domain",
            "spacing_dzyx",
            "moving_domain",
            "fixed_label",
            "moving_label",
        ):
            if key in sample:
                result[key] = sample[key].unsqueeze(0)
    elif sample["kind"] == "triplet":
        for key in ("first", "second", "third"):
            result[key] = sample[key].unsqueeze(0)
        result["spacing_dzyx"] = sample["spacing_dzyx"].unsqueeze(0)
        result["domains"] = tuple(value.unsqueeze(0) for value in sample["domains"])
        if "labels" in sample:
            result["labels"] = tuple(value.unsqueeze(0) for value in sample["labels"])
    elif sample["kind"] == "unpaired_domains":
        for key in ("mr", "ct", "mr_domain", "ct_domain", "mr_spacing_dzyx", "ct_spacing_dzyx"):
            result[key] = sample[key].unsqueeze(0)
    else:
        raise ValueError(f"Unknown PRA-CM sample kind {sample['kind']!r}.")
    return result
