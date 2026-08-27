"""Complete, dependency-aware V4-final checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch

from .dns import faithful_v2
from .dns import model as dns_model
from .inference import instance_optimization
from .ops import spatial
from .training import augmentation

from . import __version__


CHECKPOINT_SCHEMA = "pracm_v4_final_checkpoint_v1"


def config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _python_files() -> Sequence[Path]:
    root = Path(__file__).resolve().parent
    own = [path for path in root.rglob("*.py") if "__pycache__" not in path.parts]
    dependencies = [
        Path(faithful_v2.__file__).resolve(),
        Path(dns_model.__file__).resolve(),
        Path(instance_optimization.__file__).resolve(),
        Path(spatial.__file__).resolve(),
        Path(augmentation.__file__).resolve(),
    ]
    return tuple(sorted(set(own + dependencies), key=lambda path: str(path)))


def source_tree_sha256() -> str:
    digest = hashlib.sha256()
    for path in _python_files():
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def rng_state() -> Mapping[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state["cuda"].cpu())


def checkpoint_payload(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    config: Mapping[str, Any],
    epoch: int,
    logical_step: int,
    optimizer_step: int,
    best_validation: float,
    manifest_path: Path,
    manifest_sha256: str,
    tensorboard_dir: Path,
    world_size: int,
    rng_by_rank: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": __version__,
        "implementation": "full_resolution_faithful_v2_dsir_plus_masked_explicit_io",
        "source_sha256": source_tree_sha256(),
        "config": dict(config),
        "config_sha256": config_sha256(config),
        "epoch": int(epoch),
        "global_step": int(logical_step),
        "logical_step": int(logical_step),
        "optimizer_step": int(optimizer_step),
        "best_validation": float(best_validation),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": str(manifest_sha256),
        "tensorboard_dir": str(tensorboard_dir.resolve()),
        "world_size": int(world_size),
        "model": model.state_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "rng_by_rank": tuple(rng_by_rank),
        "architecture": {
            "implementation_version": "mok_masr_dns_faithful_v2",
            "feature_extractor": "figure9_full_resolution",
            "descriptor": "duo_layout_dns_feature_squeezing_24ch",
            "registration": "masked_multilevel_explicit_real_pair_optimization",
        },
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
        torch.save(payload, temporary)
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def load_checkpoint(path: Union[str, Path], *, map_location="cpu") -> Mapping[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        payload = torch.load(resolved, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(resolved, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Not a V4-final checkpoint; legacy A/B/Full checkpoints are incompatible.")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "logical_step",
        "optimizer_step",
        "rng_by_rank",
        "config_sha256",
        "source_sha256",
        "manifest_sha256",
        "world_size",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Incomplete V4-final checkpoint: {sorted(missing)}")
    return payload


__all__ = [
    "CHECKPOINT_SCHEMA",
    "atomic_torch_save",
    "checkpoint_payload",
    "config_sha256",
    "load_checkpoint",
    "restore_rng_state",
    "rng_state",
    "source_tree_sha256",
]
