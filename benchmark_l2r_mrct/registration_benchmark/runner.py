"""One leakage-safe pair runner shared by every task domain and method."""

from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

from .adapters import create_adapter
from .contract import PairTask, load_registration_manifest
from .flow import load_flow, save_flow, validate_flow, warp_resampled_xyz
from .io import load_pair_on_fixed_grid, save_nifti
from .provenance import environment_identity, git_identity, sha256_file


def load_method_config(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    expanded = os.path.expandvars(path.read_text(encoding="utf-8"))
    if "${REGBENCH_" in expanded or "$REGBENCH_" in expanded:
        raise EnvironmentError(f"Unresolved REGBENCH environment variable in {path}.")
    raw = yaml.safe_load(expanded)
    if not isinstance(raw, dict) or not raw.get("method_id") or not raw.get("adapter"):
        raise ValueError("Method config requires method_id and adapter.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", str(raw["method_id"])):
        raise ValueError(f"Unsafe method_id: {raw['method_id']!r}")
    for key in ("repo", "checkpoint", "project_root", "model_config", "config"):
        if raw.get(key):
            value = Path(str(raw[key]))
            raw[key] = str(value if value.is_absolute() else (path.parent / value).resolve())
    raw["_config_path"] = str(path)
    return raw


def run_pair(task: PairTask, method_config_path: Path, output_root: Path) -> Dict[str, Any]:
    config_path = method_config_path.resolve()
    config = load_method_config(config_path)
    method_id = str(config["method_id"])
    pair_dir = output_root.resolve() / method_id / task.task_domain / task.pair_id
    if pair_dir.exists() and any(pair_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing pair output: {pair_dir}")
    pair_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status: Dict[str, Any] = {
        "schema": "registration_pair_status_v1",
        "method_id": method_id,
        "pair_id": task.pair_id,
        "task_domain": task.task_domain,
        "task_group": task.task_group,
        "success": False,
    }
    try:
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        adapter = create_adapter(str(config["adapter"]))
        result = adapter.register(task, config, pair_dir)
        flow = validate_flow(result.flow_dzyx, fixed.shape_xyz)
        save_warped_output = bool(config.get("save_warped", True))
        save_flow(
            pair_dir / "flow.npz",
            flow,
            fixed.affine,
            {
                "method_id": method_id,
                "pair_id": task.pair_id,
                "task_domain": task.task_domain,
                "task_group": task.task_group,
                "fixed_phase": task.fixed.phase,
                "moving_phase": task.moving.phase,
            },
        )
        native_output_deleted = False
        if bool(config.get("delete_native_output", False)):
            native_name = Path(str(config.get("native_output_name", "")))
            if native_name.is_absolute() or len(native_name.parts) != 1 or native_name.name in {"", ".", "..", "flow.npz"}:
                raise ValueError(f"Unsafe native output cleanup target: {native_name}")
            native_path = pair_dir / native_name
            if native_path.is_file():
                native_path.unlink()
                native_output_deleted = True
        preprocessed_inputs_deleted = False
        if bool(config.get("delete_preprocessed_inputs", False)):
            for intermediate_name in ("fixed.nii.gz", "moving_on_fixed.nii.gz"):
                intermediate = pair_dir / intermediate_name
                if intermediate.is_file():
                    intermediate.unlink()
                    preprocessed_inputs_deleted = True
        if save_warped_output:
            warped = warp_resampled_xyz(moving.data_xyz, flow, order=1)
            save_nifti(warped, fixed.affine, pair_dir / "warped_moving.nii.gz")
        provenance = {
            "schema": "registration_pair_provenance_v1",
            "method_config": str(config_path),
            "method_config_sha256": sha256_file(config_path),
            "fixed_image": str(task.fixed.path),
            "fixed_image_sha256": sha256_file(task.fixed.path),
            "moving_image": str(task.moving.path),
            "moving_image_sha256": sha256_file(task.moving.path),
            "environment": environment_identity(),
            "adapter_diagnostics": result.diagnostics,
        }
        if config.get("repo"):
            provenance["third_party_repo"] = git_identity(Path(str(config["repo"])))
        for key in ("checkpoint", "model_config"):
            if config.get(key):
                artifact = Path(str(config[key])).resolve()
                if not artifact.is_file():
                    raise FileNotFoundError(f"Configured {key} is missing: {artifact}")
                provenance[key] = {"path": str(artifact), "sha256": sha256_file(artifact), "bytes": artifact.stat().st_size}
        (pair_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        status.update({
            "success": True,
            "runtime_seconds": time.perf_counter() - started,
            "flow": "flow.npz",
            "warped_moving": "warped_moving.nii.gz" if save_warped_output else None,
            "native_output_deleted": native_output_deleted,
            "preprocessed_inputs_deleted": preprocessed_inputs_deleted,
        })
    except Exception as exc:
        status.update({
            "runtime_seconds": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        (pair_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise
    (pair_dir / "status.json").write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return status


def run_manifest(
    manifest_path: Path,
    method_config_path: Path,
    output_root: Path,
    *,
    pair_id: Optional[str] = None,
    splits: Optional[set[str]] = None,
    task_groups: Optional[set[str]] = None,
    resume: bool = False,
    continue_on_error: bool = False,
) -> Dict[str, Any]:
    tasks = load_registration_manifest(manifest_path, require_files=True)
    selected = [
        task for task in tasks
        if (pair_id is None or task.pair_id == pair_id)
        and (not splits or task.split in splits)
        and (not task_groups or task.task_group in task_groups)
    ]
    if not selected:
        raise KeyError("No pair matched pair-id/split/task-group filters.")
    completed, skipped, failures = [], [], []
    resolved_method_config = load_method_config(method_config_path)
    method_id = str(resolved_method_config["method_id"])
    for task in selected:
        pair_dir = output_root.resolve() / method_id / task.task_domain / task.pair_id
        if resume and (pair_dir / "status.json").is_file() and (pair_dir / "flow.npz").is_file():
            status = json.loads((pair_dir / "status.json").read_text(encoding="utf-8"))
            if status.get("success") is True:
                _, _, flow_metadata = load_flow(pair_dir / "flow.npz")
                provenance_path = pair_dir / "provenance.json"
                if not provenance_path.is_file():
                    raise RuntimeError(f"Cannot resume unverifiable output without provenance: {pair_dir}")
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
                expected = {
                    "method_id": method_id, "pair_id": task.pair_id,
                    "config": sha256_file(method_config_path.resolve()),
                    "fixed": sha256_file(task.fixed.path), "moving": sha256_file(task.moving.path),
                }
                observed = {
                    "method_id": flow_metadata.get("method_id"), "pair_id": flow_metadata.get("pair_id"),
                    "config": provenance.get("method_config_sha256"),
                    "fixed": provenance.get("fixed_image_sha256"), "moving": provenance.get("moving_image_sha256"),
                }
                if resolved_method_config.get("checkpoint"):
                    checkpoint = Path(str(resolved_method_config["checkpoint"])).resolve()
                    expected["checkpoint"] = sha256_file(checkpoint)
                    observed["checkpoint"] = dict(provenance.get("checkpoint", {})).get("sha256")
                if resolved_method_config.get("model_config"):
                    model_config = Path(str(resolved_method_config["model_config"])).resolve()
                    expected["model_config"] = sha256_file(model_config)
                    observed["model_config"] = dict(provenance.get("model_config", {})).get("sha256")
                if resolved_method_config.get("repo"):
                    expected["repo_commit"] = git_identity(Path(str(resolved_method_config["repo"]))).get("commit")
                    observed["repo_commit"] = dict(provenance.get("third_party_repo", {})).get("commit")
                if observed != expected:
                    raise RuntimeError(f"Resume provenance mismatch for {pair_dir}: {observed} != {expected}")
                skipped.append(task.pair_id)
                continue
        try:
            completed.append(run_pair(task, method_config_path, output_root))
        except Exception as exc:
            failures.append({"pair_id": task.pair_id, "error_type": type(exc).__name__, "error": str(exc)})
            if not continue_on_error:
                raise
    return {
        "method_id": method_id, "selected": len(selected), "completed": len(completed),
        "skipped_successful": len(skipped), "failed": len(failures),
        "skipped_pair_ids": skipped, "failures": failures, "statuses": completed,
    }
