"""Strict, label-isolated task manifests used by every method adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REGISTRATION_SCHEMA = "registration_pair_manifest_v1"
EVALUATION_SCHEMA = "registration_evaluation_manifest_v1"
FLOW_SCHEMA = "fixed_to_moving_sampling_displacement_dzyx_fixed_voxel_v1"
MULTIPHASE_GROUPS: Mapping[str, Tuple[Tuple[str, str], ...]] = {
    "precontrast_target": (("p", "v"), ("p", "a")),
    "arterial_target": (("a", "v"), ("a", "p")),
    "venous_target": (("v", "a"), ("v", "p")),
}
MULTIPHASE_TRAINING_DIRECTIONS: Tuple[Tuple[str, str], ...] = tuple(
    (fixed, moving) for fixed in ("p", "a", "v", "d") for moving in ("p", "a", "v", "d") if fixed != moving
)
PLCR_SPLIT_COUNTS: Mapping[str, int] = {"train": 172, "validation": 29, "test": 49}
PLCR_PHASES: Tuple[str, ...] = ("p", "a", "v", "d")


def _norm_phase(value: str) -> str:
    aliases = {
        "p": "p", "plain": "p", "pre": "p", "precontrast": "p", "pre-contrast": "p", "c0": "p",
        "a": "a", "arterial": "a", "c1": "a",
        "v": "v", "venous": "v", "portal": "v", "portal_venous": "v", "c2": "v",
        "d": "d", "delayed": "d", "c3": "d",
        "mr": "mr", "ct": "ct",
    }
    key = str(value).strip().lower().replace(" ", "_")
    if key not in aliases:
        raise ValueError(f"Unknown phase/modality label: {value!r}")
    return aliases[key]


def _norm_split(value: str) -> str:
    aliases = {
        "train": "train", "training": "train",
        "val": "validation", "valid": "validation", "validation": "validation",
        "test": "test", "testing": "test",
    }
    key = str(value).strip().lower()
    if key not in aliases:
        raise ValueError(f"Unknown dataset split: {value!r}")
    return aliases[key]


def canonical_json_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    if not token:
        token = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
    return token[:160]


@dataclass(frozen=True)
class ImageRecord:
    path: Path
    phase: str
    modality: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base: Path) -> "ImageRecord":
        path = Path(str(raw["path"]))
        if not path.is_absolute():
            path = (base / path).resolve()
        return cls(
            path=path,
            phase=_norm_phase(str(raw.get("phase", raw.get("modality", "")))),
            modality=_norm_phase(str(raw.get("modality", raw.get("phase", "")))),
        )

    def as_dict(self, relative_to: Optional[Path] = None) -> Dict[str, str]:
        path = self.path
        if relative_to is not None:
            try:
                path = path.relative_to(relative_to)
            except ValueError:
                pass
        return {"path": path.as_posix(), "phase": self.phase, "modality": self.modality}


@dataclass(frozen=True)
class PairTask:
    pair_id: str
    subject_id: str
    split: str
    task_domain: str
    task_group: str
    fixed: ImageRecord
    moving: ImageRecord

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any], base: Path) -> "PairTask":
        pair_id = str(raw["pair_id"]).strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", pair_id) or pair_id in {".", ".."}:
            raise ValueError(f"Unsafe pair_id: {pair_id!r}")
        fixed = ImageRecord.from_dict(raw["fixed"], base)
        moving = ImageRecord.from_dict(raw["moving"], base)
        domain = str(raw["task_domain"]).strip().lower()
        group = str(raw["task_group"]).strip().lower()
        if domain == "multiphase_ct":
            allowed = set(MULTIPHASE_TRAINING_DIRECTIONS if group == "training_all_phases" else MULTIPHASE_GROUPS.get(group, ()))
            if (fixed.phase, moving.phase) not in allowed:
                raise ValueError(
                    f"{pair_id}: direction {fixed.phase}<-{moving.phase} is not allowed for {group}."
                )
            if fixed.modality != "ct" or moving.modality != "ct":
                raise ValueError(f"{pair_id}: multiphase task must be CT-to-CT.")
        elif domain == "l2r_mrct":
            if (fixed.modality, moving.modality) != ("mr", "ct"):
                raise ValueError(f"{pair_id}: L2R protocol is fixed MR <- moving CT.")
        else:
            raise ValueError(f"{pair_id}: unsupported task_domain {domain!r}.")
        return cls(
            pair_id=pair_id,
            subject_id=str(raw["subject_id"]),
            split=_norm_split(str(raw["split"])),
            task_domain=domain,
            task_group=group,
            fixed=fixed,
            moving=moving,
        )

    def as_dict(self, relative_to: Optional[Path] = None) -> Dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "subject_id": self.subject_id,
            "split": self.split,
            "task_domain": self.task_domain,
            "task_group": self.task_group,
            "fixed": self.fixed.as_dict(relative_to),
            "moving": self.moving.as_dict(relative_to),
        }


def load_registration_manifest(path: Path, require_files: bool = True) -> List[PairTask]:
    path = path.resolve()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("schema") != REGISTRATION_SCHEMA:
        raise ValueError(f"Expected {REGISTRATION_SCHEMA}, got {raw.get('schema')!r}.")
    embedded = raw.get("content_sha256")
    if embedded:
        unhashed = dict(raw)
        unhashed.pop("content_sha256", None)
        observed = canonical_json_hash(unhashed)
        if observed != embedded:
            raise ValueError(f"Registration manifest hash mismatch: {observed} != {embedded}.")
    tasks = [PairTask.from_dict(item, path.parent) for item in raw.get("pairs", [])]
    if not tasks:
        raise ValueError("Registration manifest contains no pairs.")
    ids = [task.pair_id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Registration manifest contains duplicate pair_id values.")
    if require_files:
        missing = [str(image.path) for task in tasks for image in (task.fixed, task.moving) if not image.path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing registration images ({len(missing)}): {missing[:3]}")
    return tasks


def write_registration_manifest(path: Path, pairs: Sequence[PairTask], dataset_id: str) -> None:
    path = path.resolve()
    if not pairs:
        raise ValueError("Cannot write an empty registration manifest.")
    ids = [pair.pair_id for pair in pairs]
    if len(ids) != len(set(ids)):
        raise ValueError("Cannot write duplicate pair_id values; check subject identifier normalization.")
    payload: Dict[str, Any] = {
        "schema": REGISTRATION_SCHEMA,
        "dataset_id": dataset_id,
        "label_paths_present": False,
        "pairs": [pair.as_dict(path.parent) for pair in pairs],
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_multiphase_manifests(
    inventory_csv: Path,
    registration_output: Path,
    evaluation_output: Path,
    *,
    dataset_id: str,
    splits: Sequence[str] = ("train", "validation", "test"),
    task_groups: Sequence[str] = ("precontrast_target",),
    training_output: Optional[Path] = None,
    evaluation_splits: Sequence[str] = ("test",),
) -> Tuple[int, int]:
    """Build image-only and label-only manifests from a four-phase inventory.

    Required columns: subject_id, split, phase, image_path.  label_path is optional.
    By default the publication-facing task is ``Pre-contrast <- Arterial &
    Venous``. Passing all keys of ``MULTIPHASE_GROUPS`` reproduces the three
    target-phase reporting layout used by Mok et al. Internally, each target
    task contains two pairwise calls; they are pooled by the aggregator rather
    than reported as six separate experiments. D/C3 remains available to
    separate training manifests but is deliberately excluded here.
    """
    inventory_csv = inventory_csv.resolve()
    rows: Dict[Tuple[str, str], Dict[str, str]] = {}
    allowed_splits = {_norm_split(value) for value in splits}
    normalized_evaluation_splits = {_norm_split(value) for value in evaluation_splits}
    with inventory_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject_id", "split", "phase", "image_path"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Inventory missing columns: {sorted(missing)}")
        for raw in reader:
            normalized_split = _norm_split(raw["split"])
            if normalized_split not in allowed_splits:
                continue
            phase = _norm_phase(raw["phase"])
            key = (raw["subject_id"], phase)
            if key in rows:
                raise ValueError(f"Duplicate subject/phase: {key}")
            rows[key] = {key: str(value or "").strip() for key, value in raw.items()}
            rows[key]["split"] = normalized_split

    unknown_groups = set(task_groups).difference(MULTIPHASE_GROUPS)
    if unknown_groups:
        raise ValueError(f"Unknown multiphase task groups: {sorted(unknown_groups)}")
    subjects = sorted({subject for subject, _ in rows})
    for subject in subjects:
        subject_splits = {_norm_split(row["split"]) for (row_subject, _), row in rows.items() if row_subject == subject}
        if len(subject_splits) != 1:
            raise ValueError(f"Subject {subject!r} has phase rows assigned to different splits: {sorted(subject_splits)}")
    tasks: List[PairTask] = []
    evaluation: List[Dict[str, Any]] = []
    for subject in subjects:
        if not all((subject, phase) in rows for phase in ("p", "a", "v")):
            continue
        subject_split = _norm_split(rows[(subject, "p")]["split"])
        if subject_split not in normalized_evaluation_splits:
            continue
        for group in task_groups:
            directions = MULTIPHASE_GROUPS[group]
            for fixed_phase, moving_phase in directions:
                fixed_row, moving_row = rows[(subject, fixed_phase)], rows[(subject, moving_phase)]
                fixed_path = Path(fixed_row["image_path"])
                moving_path = Path(moving_row["image_path"])
                if not fixed_path.is_absolute():
                    fixed_path = (inventory_csv.parent / fixed_path).resolve()
                if not moving_path.is_absolute():
                    moving_path = (inventory_csv.parent / moving_path).resolve()
                pair_id = f"{_safe_token(subject)}__{fixed_phase}_from_{moving_phase}"
                tasks.append(PairTask(
                    pair_id=pair_id,
                    subject_id=subject,
                    split=subject_split,
                    task_domain="multiphase_ct",
                    task_group=group,
                    fixed=ImageRecord(fixed_path, fixed_phase, "ct"),
                    moving=ImageRecord(moving_path, moving_phase, "ct"),
                ))
                fixed_label, moving_label = fixed_row.get("label_path", ""), moving_row.get("label_path", "")
                if fixed_label and moving_label:
                    def resolve_label(value: str) -> str:
                        label = Path(value)
                        if not label.is_absolute():
                            label = (inventory_csv.parent / label).resolve()
                        return label.as_posix()
                    evaluation.append({
                        "pair_id": pair_id,
                        "fixed_label": resolve_label(fixed_label),
                        "moving_label": resolve_label(moving_label),
                    })
    if not tasks:
        raise ValueError("No complete P/A/V subjects were found.")
    write_registration_manifest(registration_output, tasks, dataset_id)
    if training_output is not None:
        training_tasks: List[PairTask] = []
        for subject in subjects:
            if not all((subject, phase) in rows for phase in ("p", "a", "v", "d")):
                continue
            split = _norm_split(rows[(subject, "p")]["split"])
            if split not in {"train", "validation"}:
                continue
            for fixed_phase, moving_phase in MULTIPHASE_TRAINING_DIRECTIONS:
                fixed_row, moving_row = rows[(subject, fixed_phase)], rows[(subject, moving_phase)]
                fixed_path, moving_path = Path(fixed_row["image_path"]), Path(moving_row["image_path"])
                if not fixed_path.is_absolute(): fixed_path = (inventory_csv.parent / fixed_path).resolve()
                if not moving_path.is_absolute(): moving_path = (inventory_csv.parent / moving_path).resolve()
                training_tasks.append(PairTask(
                    pair_id=f"{_safe_token(subject)}__train_{fixed_phase}_from_{moving_phase}", subject_id=subject,
                    split=split, task_domain="multiphase_ct",
                    task_group="training_all_phases", fixed=ImageRecord(fixed_path, fixed_phase, "ct"),
                    moving=ImageRecord(moving_path, moving_phase, "ct"),
                ))
        if not training_tasks:
            raise ValueError("No complete four-phase train/validation subjects were found for training_output.")
        write_registration_manifest(training_output, training_tasks, dataset_id + "__all_four_phase_training")
    eval_payload = {
        "schema": EVALUATION_SCHEMA,
        "dataset_id": dataset_id,
        "registration_manifest_sha256": canonical_json_hash(json.loads(registration_output.read_text(encoding="utf-8"))),
        "labels_are_evaluation_only": True,
        "pairs": evaluation,
    }
    eval_payload["content_sha256"] = canonical_json_hash(eval_payload)
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(json.dumps(eval_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(tasks), len(evaluation)


def audit_plcr_inventory(
    inventory_csv: Path,
    *,
    expected_split_counts: Mapping[str, int] = PLCR_SPLIT_COUNTS,
) -> Dict[str, Any]:
    """Validate only the frozen PLC-R subject assignment, not dataset integrity.

    PLC-R is not repartitioned here. The authoritative local split is checked
    for 172/29/49 subjects and subject disjointness. Image/label existence,
    geometry, readability, and phase completeness are intentionally out of
    scope because dataset integrity is managed separately by the data owner.
    """
    inventory_csv = inventory_csv.resolve()
    normalized_expected = {_norm_split(key): int(value) for key, value in expected_split_counts.items()}
    if set(normalized_expected) != {"train", "validation", "test"}:
        raise ValueError("PLC-R expected_split_counts must define train, validation, and test.")
    subject_splits: Dict[str, str] = {}
    with inventory_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"subject_id", "split"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"PLC-R inventory missing columns: {sorted(missing)}")
        for row_index, raw in enumerate(reader, start=2):
            subject = str(raw.get("subject_id", "")).strip()
            if not subject:
                raise ValueError(f"PLC-R inventory row {row_index} has an empty subject_id.")
            split = _norm_split(str(raw.get("split", "")))
            previous_split = subject_splits.setdefault(subject, split)
            if previous_split != split:
                raise ValueError(
                    f"PLC-R subject {subject!r} leaks across splits: {previous_split!r} and {split!r}."
                )

    observed_counts = {split: 0 for split in normalized_expected}
    for subject in subject_splits:
        observed_counts[subject_splits[subject]] += 1
    if observed_counts != normalized_expected:
        raise ValueError(
            f"PLC-R split mismatch: observed {observed_counts}, expected {normalized_expected}. "
            "Use the frozen local assignment; do not generate a new random split."
        )
    return {
        "schema": "plcr_split_audit_v1",
        "dataset_id": "PLC-R-250",
        "subject_count": len(subject_splits),
        "split_subject_counts": observed_counts,
        "subject_disjoint": True,
        "dataset_integrity_checked": False,
        "inventory_sha256": hashlib.sha256(inventory_csv.read_bytes()).hexdigest(),
    }


def build_plcr_manifests(
    inventory_csv: Path,
    registration_output: Path,
    evaluation_output: Path,
    training_output: Path,
    *,
    dataset_id: str = "PLC-R-250",
    task_groups: Sequence[str] = ("precontrast_target",),
) -> Dict[str, Any]:
    """Audit the frozen 172/29/49 PLC-R split and build its formal manifests."""
    audit = audit_plcr_inventory(inventory_csv)
    pair_count, label_count = build_multiphase_manifests(
        inventory_csv,
        registration_output,
        evaluation_output,
        dataset_id=dataset_id,
        splits=("train", "validation", "test"),
        task_groups=task_groups,
        training_output=training_output,
        evaluation_splits=("test",),
    )
    training_payload = json.loads(training_output.resolve().read_text(encoding="utf-8"))
    return {
        **audit,
        "registration_pair_count": pair_count,
        "evaluation_pair_count": label_count,
        "training_pair_count": len(training_payload.get("pairs", [])),
        "task_groups": list(task_groups),
        "main_table_result_policy": "reproduced_local_only",
    }


def build_l2r_public_manifests(
    inventory: Path, registration_output: Path, evaluation_output: Path, *,
    dataset_id: str = "L2R-MRCT-P8", dataset_root: Optional[Path] = None,
) -> Tuple[int, int]:
    """Convert the frozen public-8 inventory while physically isolating labels."""
    inventory = inventory.resolve()
    if inventory.suffix.lower() == ".json":
        rows = json.loads(inventory.read_text(encoding="utf-8"))
    else:
        with inventory.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
    tasks: List[PairTask] = []; evaluation: List[Dict[str, Any]] = []
    for index, row in enumerate(rows):
        def resolve(key: str) -> Path:
            value = Path(str(row[key]))
            if value.is_absolute():
                return value.resolve()
            candidates = []
            if dataset_root is not None:
                candidates.append(dataset_root.resolve() / value)
            candidates.extend([inventory.parent / value, inventory.parent.parent / value])
            for candidate in candidates:
                if candidate.is_file():
                    return candidate.resolve()
            return candidates[0].resolve()
        pair_id = _safe_token(str(row.get("pair_id") or f"l2r_mrct_{index:04d}"))
        tasks.append(PairTask(
            pair_id=pair_id, subject_id=str(row.get("official_subject_id") or row.get("subject_id") or pair_id),
            split="test", task_domain="l2r_mrct", task_group="mr_target",
            fixed=ImageRecord(resolve("fixed_image"), "mr", "mr"),
            moving=ImageRecord(resolve("moving_image"), "ct", "ct"),
        ))
        evaluation.append({
            "pair_id": pair_id, "fixed_label": resolve("fixed_label").as_posix(),
            "moving_label": resolve("moving_label").as_posix(),
            "label_values": [1, 2, 3, 4],
            "label_names": {"1": "liver", "2": "spleen", "3": "right_kidney", "4": "left_kidney"},
        })
    if len(tasks) != 8:
        raise ValueError(f"L2R-MRCT-P8 requires exactly 8 public pairs, got {len(tasks)}.")
    write_registration_manifest(registration_output, tasks, dataset_id)
    payload: Dict[str, Any] = {
        "schema": EVALUATION_SCHEMA, "dataset_id": dataset_id, "labels_are_evaluation_only": True,
        "pairs": evaluation,
    }
    payload["content_sha256"] = canonical_json_hash(payload)
    evaluation_output.parent.mkdir(parents=True, exist_ok=True)
    evaluation_output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(tasks), len(evaluation)


def build_l2r_unpaired_training_manifest(
    inventory_csv: Path, output: Path, *, dataset_id: str = "L2R-MRCT-unpaired90",
    dataset_root: Optional[Path] = None, validation_count: int = 5, seed: int = 2024,
) -> Tuple[int, int]:
    """Create deterministic cross-product MR<-CT training pairs from unpaired images."""
    inventory_csv = inventory_csv.resolve()
    with inventory_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_modality: Dict[str, List[Tuple[str, Path]]] = {"mr": [], "ct": []}
    for index, row in enumerate(rows):
        modality = _norm_phase(str(row.get("modality", "")))
        if modality not in by_modality:
            continue
        raw_value = str(row.get("image_path") or row.get("image") or "").strip()
        if not raw_value:
            raise ValueError("Unpaired inventory requires image or image_path.")
        raw_path = Path(raw_value)
        if raw_path.is_absolute():
            path = raw_path.resolve()
        else:
            candidates = []
            if dataset_root is not None:
                candidates.append(dataset_root.resolve() / raw_path)
            candidates.extend([inventory_csv.parent / raw_path, inventory_csv.parent.parent / raw_path])
            path = next((candidate.resolve() for candidate in candidates if candidate.is_file()), candidates[0].resolve())
        image_id = str(row.get("auxiliary_id") or f"{modality}_{index:04d}")
        by_modality[modality].append((image_id, path))
    if not by_modality["mr"] or not by_modality["ct"]:
        raise ValueError("Unpaired training inventory must contain both MR and CT images.")
    if validation_count < 2 or validation_count >= len(by_modality["mr"]) + len(by_modality["ct"]):
        raise ValueError("validation_count must leave training data and include both modalities.")
    rng = random.Random(seed)
    for values in by_modality.values():
        rng.shuffle(values)
    mr_val_count = max(1, round(validation_count * len(by_modality["mr"]) / sum(map(len, by_modality.values()))))
    mr_val_count = min(mr_val_count, len(by_modality["mr"]) - 1)
    ct_val_count = validation_count - mr_val_count
    ct_val_count = min(max(1, ct_val_count), len(by_modality["ct"]) - 1)
    if mr_val_count + ct_val_count != validation_count:
        raise ValueError("validation_count cannot be stratified while retaining both training modalities.")
    split_values = {
        "train": (by_modality["mr"][mr_val_count:], by_modality["ct"][ct_val_count:]),
        "validation": (by_modality["mr"][:mr_val_count], by_modality["ct"][:ct_val_count]),
    }
    tasks: List[PairTask] = []
    for split, (mr_values, ct_values) in split_values.items():
        for mr_id, mr_path in mr_values:
            for ct_id, ct_path in ct_values:
                safe_mr = _safe_token(mr_id)
                safe_ct = _safe_token(ct_id)
                tasks.append(PairTask(
                    pair_id=f"unpaired_{split}__{safe_mr}_from_{safe_ct}",
                    subject_id=f"unpaired__{safe_mr}__{safe_ct}", split=split,
                    task_domain="l2r_mrct", task_group="unpaired_training",
                    fixed=ImageRecord(mr_path, "mr", "mr"), moving=ImageRecord(ct_path, "ct", "ct"),
                ))
    write_registration_manifest(output, tasks, dataset_id)
    return len(split_values["train"][0]) * len(split_values["train"][1]), len(split_values["validation"][0]) * len(split_values["validation"][1])


__all__ = [
    "EVALUATION_SCHEMA", "FLOW_SCHEMA", "ImageRecord", "MULTIPHASE_GROUPS",
    "MULTIPHASE_TRAINING_DIRECTIONS", "PLCR_PHASES", "PLCR_SPLIT_COUNTS", "PairTask", "REGISTRATION_SCHEMA",
    "audit_plcr_inventory", "build_l2r_public_manifests", "build_l2r_unpaired_training_manifest",
    "build_multiphase_manifests", "build_plcr_manifests",
    "canonical_json_hash", "load_registration_manifest", "write_registration_manifest",
]
