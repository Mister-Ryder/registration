"""Versioned checkpoints with configuration and source identity."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
import torch

from .. import __version__


CHECKPOINT_SCHEMA = "pracm_v4_3d_checkpoint_v1"


def source_tree_sha256() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def config_sha256(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def rng_state() -> Mapping[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def checkpoint_payload(
    *,
    module,
    optimizer,
    scheduler,
    scaler,
    config: Mapping[str, Any],
    epoch: int,
    global_step: int,
    best_validation: float,
    data_identity: Mapping[str, Any],
    early_stopping_counter: int = 0,
) -> Mapping[str, Any]:
    return {
        "schema": CHECKPOINT_SCHEMA,
        "version": __version__,
        "source_sha256": source_tree_sha256(),
        "config": dict(config),
        "config_sha256": config_sha256(config),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "logical_step": int(global_step),
        "optimizer_step": int(global_step),
        "best_validation": float(best_validation),
        "data_identity": dict(data_identity),
        "early_stopping_counter": int(early_stopping_counter),
        "module_state": module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": None if scaler is None else scaler.state_dict(),
        "rng_state": rng_state(),
    }


def atomic_torch_save(payload: Mapping[str, Any], path: Union[str, Path]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
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
    except TypeError:  # PyTorch versions before the weights_only argument.
        payload = torch.load(resolved, map_location=map_location)
    if not isinstance(payload, Mapping) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("Not a PRA-CM 3-D checkpoint.")
    required = {
        "module_state",
        "config",
        "source_sha256",
        "data_identity",
        "epoch",
        "global_step",
        "logical_step",
        "optimizer_step",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Checkpoint is incomplete: {sorted(missing)}")
    return payload
