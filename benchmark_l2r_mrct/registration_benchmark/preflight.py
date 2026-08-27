"""Fail-fast deployment checks without running registration experiments."""

from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping

from .adapters import create_adapter
from .runner import load_method_config


IMPORTS = {
    "ants_syn_mi": ["ants", "nibabel", "numpy"],
    "convexadam_mind": ["torch", "SimpleITK", "nibabel"],
    "fireants": ["torch", "SimpleITK", "nibabel"],
    "dino_reg": ["torch", "torchvision", "skimage", "sklearn"],
    "mok_dns_io": ["torch", "nibabel"],
    "synmse": ["torch", "nibabel"],
    "locor": ["torch", "nibabel"],
    "dgmir_u": ["torch", "nibabel"],
    "m2m_reg": ["torch", "icon_registration", "itk"],
    "transmorph_mind": ["torch", "ml_collections"],
    "corrmlp_mind": ["torch", "einops"],
    "ours": ["torch", "nibabel"],
    "identity": ["numpy"],
}


def check_method(config_path: Path) -> Dict[str, Any]:
    config = load_method_config(config_path)
    adapter = str(config["adapter"]).lower().replace("-", "_")
    create_adapter(adapter)
    checks = []
    for module in IMPORTS.get(adapter, []):
        ok = importlib.util.find_spec(module) is not None
        checks.append({"kind": "python_import", "target": module, "ok": ok})
    repo_value = config.get("repo")
    if repo_value:
        repo = Path(str(repo_value))
        ok = repo.is_dir()
        item: Dict[str, Any] = {"kind": "upstream_repo", "target": str(repo), "ok": ok}
        if ok and (repo / ".git").exists():
            proc = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
            item["commit"] = proc.stdout.strip() if proc.returncode == 0 else None
        checks.append(item)
    checkpoint_value = config.get("checkpoint")
    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value))
        checks.append({"kind": "checkpoint", "target": str(checkpoint), "ok": checkpoint.is_file()})
    model_config = config.get("model_config")
    if model_config:
        checks.append({"kind": "model_config", "target": str(model_config), "ok": Path(str(model_config)).is_file()})
    failed = [item for item in checks if not item["ok"]]
    return {
        "schema": "registration_method_preflight_v1", "method_id": config["method_id"],
        "adapter": adapter, "ready": not failed, "checks": checks, "failed_checks": failed,
    }


__all__ = ["check_method"]
