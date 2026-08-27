"""Independent label evaluator; deliberately not imported by the inference runner."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy.ndimage import binary_erosion, distance_transform_edt

from .contract import EVALUATION_SCHEMA, PairTask, canonical_json_hash, load_registration_manifest
from .flow import load_flow, validate_flow, warp_resampled_xyz


def _load_evaluation_manifest(path: Path) -> Dict[str, Mapping[str, Any]]:
    raw = json.loads(path.resolve().read_text(encoding="utf-8"))
    if raw.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"Expected {EVALUATION_SCHEMA}, got {raw.get('schema')!r}.")
    if raw.get("labels_are_evaluation_only") is not True:
        raise ValueError("Evaluation manifest must explicitly isolate labels from registration.")
    embedded = raw.get("content_sha256")
    if embedded:
        unhashed = dict(raw); unhashed.pop("content_sha256", None)
        if canonical_json_hash(unhashed) != embedded:
            raise ValueError("Evaluation manifest content hash mismatch.")
    pairs = {str(item["pair_id"]): item for item in raw.get("pairs", [])}
    if not pairs:
        raise ValueError("Evaluation manifest contains no labeled pairs.")
    if len(pairs) != len(raw.get("pairs", [])):
        raise ValueError("Duplicate pair_id in evaluation manifest.")
    return pairs


def _resolve_label(path_value: str, manifest_path: Path) -> Path:
    path = Path(path_value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def _label_on_fixed_grid(path: Path, fixed_shape: Sequence[int], fixed_affine: np.ndarray) -> np.ndarray:
    image = nib.load(str(path), mmap=True)
    if tuple(image.shape[:3]) != tuple(fixed_shape) or not np.allclose(image.affine, fixed_affine, atol=1e-5):
        image = resample_from_to(image, (tuple(fixed_shape), fixed_affine), order=0)
    label = np.asarray(image.dataobj)
    while label.ndim > 3 and label.shape[-1] == 1:
        label = label[..., 0]
    if label.ndim != 3 or not np.isfinite(label).all():
        raise ValueError(f"Invalid label map: {path}")
    return np.rint(label).astype(np.int32)


def _surface(mask: np.ndarray) -> np.ndarray:
    return mask & ~binary_erosion(mask, structure=np.ones((3, 3, 3), dtype=bool), border_value=0)


def overlap_and_surface_metrics(
    fixed_mask_xyz: np.ndarray,
    warped_mask_xyz: np.ndarray,
    spacing_xyz: Sequence[float],
) -> Dict[str, float]:
    fixed = np.asarray(fixed_mask_xyz, dtype=bool)
    moving = np.asarray(warped_mask_xyz, dtype=bool)
    if not fixed.any() and not moving.any():
        return {"dice": 1.0, "hd95_mm": 0.0, "assd_mm": 0.0}
    if not fixed.any() or not moving.any():
        return {"dice": 0.0, "hd95_mm": float("nan"), "assd_mm": float("nan")}
    dice = 2.0 * np.count_nonzero(fixed & moving) / (np.count_nonzero(fixed) + np.count_nonzero(moving))
    fixed_zyx, moving_zyx = fixed.transpose(2, 1, 0), moving.transpose(2, 1, 0)
    fixed_surface, moving_surface = _surface(fixed_zyx), _surface(moving_zyx)
    sampling = tuple(float(v) for v in reversed(spacing_xyz))
    distance_to_fixed = distance_transform_edt(~fixed_surface, sampling=sampling)
    distance_to_moving = distance_transform_edt(~moving_surface, sampling=sampling)
    moving_to_fixed = distance_to_fixed[moving_surface]
    fixed_to_moving = distance_to_moving[fixed_surface]
    return {
        "dice": float(dice),
        "hd95_mm": float(max(np.percentile(moving_to_fixed, 95), np.percentile(fixed_to_moving, 95))),
        "assd_mm": float(0.5 * (np.mean(moving_to_fixed) + np.mean(fixed_to_moving))),
    }


def deformation_metrics(flow_dzyx: np.ndarray, spacing_xyz: Sequence[float]) -> Dict[str, float]:
    flow = validate_flow(flow_dzyx)
    gradients = [[np.gradient(flow[i], axis=j) for j in range(3)] for i in range(3)]
    jacobian = np.empty((*flow.shape[1:], 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            jacobian[..., i, j] = gradients[i][j] + (1.0 if i == j else 0.0)
    determinant = np.linalg.det(jacobian)
    positive = determinant > 0
    sd_log_jacobian = float(np.std(np.log(determinant[positive]))) if positive.any() else float("nan")
    spacing_zyx = np.asarray(tuple(reversed(spacing_xyz)), dtype=np.float32)[:, None, None, None]
    displacement_mm = np.sqrt(np.sum((flow * spacing_zyx) ** 2, axis=0))
    d, h, w = flow.shape[1:]
    zz, yy, xx = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing="ij")
    valid = (
        (zz + flow[0] >= 0) & (zz + flow[0] <= d - 1)
        & (yy + flow[1] >= 0) & (yy + flow[1] <= h - 1)
        & (xx + flow[2] >= 0) & (xx + flow[2] <= w - 1)
    )
    return {
        "fold_fraction": float(np.mean(determinant <= 0)),
        "sd_log_jacobian": sd_log_jacobian,
        "mean_displacement_mm": float(np.mean(displacement_mm)),
        "p95_displacement_mm": float(np.percentile(displacement_mm, 95)),
        "valid_sampling_fraction": float(np.mean(valid)),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def evaluate_outputs(
    registration_manifest: Path,
    evaluation_manifest: Path,
    results_root: Path,
    method_ids: Sequence[str],
    output_dir: Path,
) -> Dict[str, Any]:
    tasks = {task.pair_id: task for task in load_registration_manifest(registration_manifest)}
    labels = _load_evaluation_manifest(evaluation_manifest)
    unknown = set(labels).difference(tasks)
    if unknown:
        raise ValueError(f"Evaluation pairs absent from registration manifest: {sorted(unknown)[:3]}")
    pair_rows: List[Dict[str, Any]] = []
    organ_rows: List[Dict[str, Any]] = []
    for method_id in method_ids:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", method_id):
            raise ValueError(f"Unsafe method_id: {method_id!r}")
        for pair_id, item in labels.items():
            task = tasks[pair_id]
            pair_dir = results_root.resolve() / method_id / task.task_domain / pair_id
            status_path, flow_path = pair_dir / "status.json", pair_dir / "flow.npz"
            if not status_path.is_file() or not flow_path.is_file():
                raise FileNotFoundError(f"Missing successful output for {method_id}/{pair_id}")
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("success") is not True:
                raise RuntimeError(f"Failed registration output: {method_id}/{pair_id}")
            flow, fixed_affine, _ = load_flow(flow_path)
            fixed_image = nib.load(str(task.fixed.path), mmap=True)
            fixed_shape = tuple(int(v) for v in fixed_image.shape[:3])
            if not np.allclose(fixed_affine, fixed_image.affine, atol=1e-5):
                raise ValueError(f"Fixed affine mismatch in {flow_path}")
            fixed_label = _label_on_fixed_grid(
                _resolve_label(str(item["fixed_label"]), evaluation_manifest.resolve()), fixed_shape, fixed_affine
            )
            moving_label = _label_on_fixed_grid(
                _resolve_label(str(item["moving_label"]), evaluation_manifest.resolve()), fixed_shape, fixed_affine
            )
            warped_label = np.rint(warp_resampled_xyz(moving_label, flow, order=0, cval=0)).astype(np.int32)
            spacing = nib.affines.voxel_sizes(fixed_affine)[:3]
            requested_values = item.get("label_values")
            values = [int(v) for v in requested_values] if requested_values else sorted(
                (set(np.unique(fixed_label)) | set(np.unique(moving_label))) - {0}
            )
            names = {int(key): str(value) for key, value in dict(item.get("label_names", {})).items()}
            if not values:
                raise ValueError(f"No foreground labels for {pair_id}")
            pair_organ_metrics = []
            for value in values:
                fixed_mask = fixed_label == value
                moving_mask = moving_label == value
                fixed_present = bool(fixed_mask.any())
                moving_present = bool(moving_mask.any())
                label_available = fixed_present and moving_present
                metrics = (
                    overlap_and_surface_metrics(fixed_mask, warped_label == value, spacing)
                    if label_available
                    else {"dice": float("nan"), "hd95_mm": float("nan"), "assd_mm": float("nan")}
                )
                row = {
                    "method_id": method_id, "pair_id": pair_id, "subject_id": task.subject_id,
                    "task_domain": task.task_domain, "task_group": task.task_group,
                    "fixed_phase": task.fixed.phase, "moving_phase": task.moving.phase,
                    "label_value": value, "label_name": names.get(value, str(value)),
                    "label_available": label_available,
                    "fixed_label_present": fixed_present,
                    "moving_label_present": moving_present,
                    "availability_reason": "" if label_available else "missing_in_fixed_or_moving_label",
                    **metrics,
                }
                organ_rows.append(row); pair_organ_metrics.append(metrics)
            evaluable_organ_count = sum(np.isfinite(float(m["dice"])) for m in pair_organ_metrics)
            if evaluable_organ_count == 0:
                raise ValueError(f"No jointly available foreground labels for {pair_id}")
            topology = deformation_metrics(flow, spacing)
            pair_rows.append({
                "method_id": method_id, "pair_id": pair_id, "subject_id": task.subject_id,
                "task_domain": task.task_domain, "task_group": task.task_group,
                "fixed_phase": task.fixed.phase, "moving_phase": task.moving.phase,
                "mean_dice": float(np.nanmean([m["dice"] for m in pair_organ_metrics])),
                "mean_hd95_mm": float(np.nanmean([m["hd95_mm"] for m in pair_organ_metrics])),
                "mean_assd_mm": float(np.nanmean([m["assd_mm"] for m in pair_organ_metrics])),
                "n_requested_organs": len(pair_organ_metrics),
                "n_evaluable_organs": evaluable_organ_count,
                "runtime_seconds": float(status.get("runtime_seconds", float("nan"))), **topology,
            })
    _write_csv(output_dir / "pair_metrics.csv", pair_rows)
    _write_csv(output_dir / "organ_metrics.csv", organ_rows)
    summary = aggregate_pair_metrics(pair_rows, organ_rows=organ_rows)
    _write_csv(output_dir / "summary.csv", summary)
    metadata = {
        "schema": "registration_evaluation_result_v2", "method_ids": list(method_ids),
        "evaluated_pairs": len(pair_rows), "labels_loaded_by_runner": False,
        "label_availability_policy": "exclude_if_missing_in_fixed_or_moving_label",
        "headline_overlap_aggregation": "equal_weight_label_macro_after_excluding_unavailable",
        "pair_metrics": "pair_metrics.csv", "organ_metrics": "organ_metrics.csv", "summary": "summary.csv",
    }
    (output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def _label_is_available(item: Mapping[str, Any]) -> bool:
    value = item.get("label_available", True)
    return value if isinstance(value, bool) else str(value).strip().lower() == "true"


def aggregate_pair_metrics(
    rows: Sequence[Mapping[str, Any]],
    organ_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Equal-weight each composite task; never report its two directions as headline tasks."""
    numeric = [
        "mean_dice", "mean_hd95_mm", "mean_assd_mm", "fold_fraction", "sd_log_jacobian",
        "mean_displacement_mm", "p95_displacement_mm", "valid_sampling_fraction", "runtime_seconds",
    ]
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["method_id"]), str(row["task_domain"]), str(row["task_group"]))
        grouped.setdefault(key, []).append(row)
    grouped_organs: Dict[Tuple[str, str, str], List[Mapping[str, Any]]] = {}
    for row in organ_rows or ():
        key = (str(row["method_id"]), str(row["task_domain"]), str(row["task_group"]))
        grouped_organs.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    group_summaries: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for (method, domain, group), items in sorted(grouped.items()):
        summary: Dict[str, Any] = {
            "method_id": method, "task_domain": domain, "task_group": group,
            "reporting_unit": "composite_target_task", "n_pair_calls": len(items),
            "n_subjects": len({str(item["subject_id"]) for item in items}),
        }
        for metric in numeric:
            summary[metric] = float(np.nanmean([float(item[metric]) for item in items]))
        organ_items = grouped_organs.get((method, domain, group), [])
        if organ_items:
            for summary_metric, organ_metric in (
                ("mean_dice", "dice"),
                ("mean_hd95_mm", "hd95_mm"),
                ("mean_assd_mm", "assd_mm"),
            ):
                label_means = []
                label_values = sorted({int(item["label_value"]) for item in organ_items})
                for label_value in label_values:
                    values = [
                        float(item[organ_metric]) for item in organ_items
                        if int(item["label_value"]) == label_value
                        and _label_is_available(item)
                        and np.isfinite(float(item[organ_metric]))
                    ]
                    if values:
                        label_means.append(float(np.mean(values)))
                summary[summary_metric] = float(np.mean(label_means)) if label_means else float("nan")
            summary["n_requested_organ_cases"] = len(organ_items)
            summary["n_evaluable_organ_cases"] = sum(
                _label_is_available(item) and np.isfinite(float(item["dice"]))
                for item in organ_items
            )
            summary["n_unavailable_organ_cases"] = (
                summary["n_requested_organ_cases"] - summary["n_evaluable_organ_cases"]
            )
            summary["n_evaluable_labels"] = len({
                int(item["label_value"]) for item in organ_items
                if _label_is_available(item) and np.isfinite(float(item["dice"]))
            })
        output.append(summary); group_summaries.setdefault((method, domain), []).append(summary)
    for (method, domain), items in sorted(group_summaries.items()):
        if len(items) < 2:
            continue
        average: Dict[str, Any] = {
            "method_id": method, "task_domain": domain, "task_group": "average_across_selected_tasks",
            "reporting_unit": "equal_weight_task_average", "n_pair_calls": sum(int(v["n_pair_calls"]) for v in items),
            "n_subjects": max(int(v["n_subjects"]) for v in items),
        }
        for metric in numeric:
            average[metric] = float(np.nanmean([float(item[metric]) for item in items]))
        output.append(average)
    return output


__all__ = ["aggregate_pair_metrics", "deformation_metrics", "evaluate_outputs", "overlap_and_surface_metrics"]
