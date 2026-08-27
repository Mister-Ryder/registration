"""Machine-readable safeguards for the dual-dataset experiment policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml


POLICY_SCHEMA = "dual_dataset_experiment_policy_v1"
DATASET_ALIASES = {
    "plcr": "plcr", "plc-r": "plcr", "plc_r": "plcr",
    "l2r": "l2r_mrct", "l2r_mrct": "l2r_mrct", "l2r-mrct": "l2r_mrct",
}
RESULT_SOURCES = {"reproduced_local", "reported_literature"}


def default_policy_path() -> Path:
    return Path(__file__).resolve().parent.parent / "registries" / "experiment_policy_v1.yaml"


def normalize_dataset(value: str) -> str:
    key = str(value).strip().lower()
    if key not in DATASET_ALIASES:
        raise ValueError(f"Unknown benchmark dataset {value!r}; expected PLC-R or L2R-MRCT.")
    return DATASET_ALIASES[key]


def load_experiment_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    policy_path = (path or default_policy_path()).resolve()
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"Expected policy schema {POLICY_SCHEMA!r} in {policy_path}.")
    datasets = raw.get("datasets")
    methods = raw.get("methods")
    if not isinstance(datasets, dict) or set(datasets) != {"plcr", "l2r_mrct"}:
        raise ValueError("Experiment policy must define exactly plcr and l2r_mrct datasets.")
    if not isinstance(methods, dict) or not methods:
        raise ValueError("Experiment policy has no method entries.")
    for method_id, method in methods.items():
        if not isinstance(method, dict):
            raise ValueError(f"Invalid method policy for {method_id}.")
        for dataset in datasets:
            entry = method.get(dataset)
            if not isinstance(entry, dict):
                raise ValueError(f"Method {method_id} has no {dataset} policy.")
            if dataset == "plcr" and entry.get("reported_reference_allowed") is not False:
                raise ValueError(f"PLC-R literature results are prohibited, including {method_id}.")
            if entry.get("reported_reference_allowed") and entry.get("reported_fallback_grade") not in {"A", "B"}:
                raise ValueError(f"Method {method_id} needs an A/B fallback grade before literature use is allowed.")
            if entry.get("test_inference") not in {"required", "required_if_reproduced"}:
                raise ValueError(f"Method {method_id} has an invalid {dataset} inference policy.")
    return raw


def validate_result_source(
    dataset: str,
    method_id: str,
    result_source: str,
    *,
    policy: Optional[Mapping[str, Any]] = None,
    protocol_verified: bool = False,
    citation: Optional[str] = None,
) -> None:
    dataset_key = normalize_dataset(dataset)
    if result_source not in RESULT_SOURCES:
        raise ValueError(f"Unknown result source {result_source!r}.")
    active = dict(policy or load_experiment_policy())
    method = dict(active.get("methods", {})).get(method_id)
    if not isinstance(method, dict):
        raise KeyError(f"Unknown method in experiment policy: {method_id}")
    method_dataset = dict(method[dataset_key])
    if result_source == "reported_literature" and not method_dataset.get("reported_reference_allowed", False):
        raise ValueError(
            f"{dataset_key}/{method_id} cannot use a literature value. PLC-R is reproduced-only; "
            "L2R references require explicit method-level eligibility."
        )
    if result_source == "reported_literature":
        if method_dataset.get("protocol_verification_required", True) and not protocol_verified:
            raise ValueError(f"{dataset_key}/{method_id} literature value has not passed protocol verification.")
        if method_dataset.get("citation_required", True) and not str(citation or "").strip():
            raise ValueError(f"{dataset_key}/{method_id} literature value requires a citation.")


def build_execution_plan(dataset: str, *, policy_path: Optional[Path] = None) -> Dict[str, Any]:
    dataset_key = normalize_dataset(dataset)
    policy = load_experiment_policy(policy_path)
    rows: List[Dict[str, Any]] = []
    for method_id, method in policy["methods"].items():
        dataset_policy = dict(method[dataset_key])
        rows.append({
            "method_id": method_id,
            "training_class": method["training_class"],
            "target_training": dataset_policy["target_training"],
            "validation_calibration": dataset_policy["validation_calibration"],
            "test_inference": dataset_policy["test_inference"],
            "reported_reference_allowed": dataset_policy["reported_reference_allowed"],
            "reported_role": dataset_policy.get("reported_role"),
            "reported_fallback_grade": dataset_policy.get("reported_fallback_grade"),
            "protocol_verification_required": dataset_policy.get("protocol_verification_required", False),
            "citation_required": dataset_policy.get("citation_required", False),
        })
    return {
        "schema": "registration_execution_plan_v1",
        "dataset": dataset_key,
        "dataset_policy": policy["datasets"][dataset_key],
        "methods": rows,
    }


def write_execution_plan(output: Path, dataset: str, *, policy_path: Optional[Path] = None) -> Dict[str, Any]:
    plan = build_execution_plan(dataset, policy_path=policy_path)
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return plan


__all__ = [
    "POLICY_SCHEMA", "build_execution_plan", "default_policy_path", "load_experiment_policy",
    "normalize_dataset", "validate_result_source", "write_execution_plan",
]
