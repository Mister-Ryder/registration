"""Reproducible checkpoint state for the ConvexAdam-main V4-final protocol."""

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

from registration_benchmark.dns import faithful_v2
from registration_benchmark.dns import model as dns_model
from registration_v4_pracm.ops import spatial
from registration_v4_pracm.training import augmentation

from . import __version__


SCHEMA = "pracm_v4_final_descriptor_checkpoint_v1"


def canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def source_hash() -> str:
    package = Path(__file__).resolve().parent
    files = [path for path in package.rglob("*.py") if "__pycache__" not in path.parts]
    files.extend(
        (
            Path(faithful_v2.__file__).resolve(),
            Path(dns_model.__file__).resolve(),
            Path(spatial.__file__).resolve(),
            Path(augmentation.__file__).resolve(),
        )
    )
    digest = hashlib.sha256()
    for path in sorted(set(files), key=str):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def capture_rng() -> Mapping[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(value: Mapping[str, Any]) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.set_rng_state(value["torch"].cpu())
    if value.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(value["cuda"].cpu())


def make_checkpoint(
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    protocol: Mapping[str, Any],
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
    model_state = model.state_dict()
    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()
    scaler_state = None if scaler is None else scaler.state_dict()
    return {
        "schema": SCHEMA,
        "version": __version__,
        "source_sha256": source_hash(),
        "protocol": dict(protocol),
        "protocol_sha256": canonical_hash(protocol),
        "epoch": int(epoch),
        "global_step": int(logical_step),
        "logical_step": int(logical_step),
        "optimizer_step": int(optimizer_step),
        "best_validation": float(best_validation),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest_sha256,
        "tensorboard_dir": str(tensorboard_dir.resolve()),
        "world_size": int(world_size),
        "model": model_state,
        "model_state_dict": model_state,
        "optimizer": optimizer_state,
        "optimizer_state_dict": optimizer_state,
        "scheduler": scheduler_state,
        "scheduler_state_dict": scheduler_state,
        "scaler": scaler_state,
        "scaler_state_dict": scaler_state,
        "rng_by_rank": tuple(rng_by_rank),
        "architecture": {
            "implementation_version": "mok_masr_dns_faithful_v2",
            "full_resolution_dsir": True,
            "descriptor_channels": 24,
            "main_solver": "descriptor_convexadam",
            "legacy_v4_checkpoint_compatible": False,
        },
    }


def atomic_save(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
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
    try:
        payload = torch.load(Path(path).expanduser().resolve(), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path).expanduser().resolve(), map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema") != SCHEMA:
        raise ValueError("Checkpoint is not V4-final; old V4 A/B/Full weights are forbidden.")
    required = {
        "model",
        "optimizer",
        "scheduler",
        "rng_by_rank",
        "epoch",
        "logical_step",
        "optimizer_step",
        "protocol_sha256",
        "source_sha256",
        "manifest_sha256",
        "world_size",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Incomplete V4-final checkpoint: {sorted(missing)}")
    return payload


__all__ = [
    "SCHEMA",
    "atomic_save",
    "canonical_hash",
    "capture_rng",
    "load_checkpoint",
    "make_checkpoint",
    "restore_rng",
    "source_hash",
]

