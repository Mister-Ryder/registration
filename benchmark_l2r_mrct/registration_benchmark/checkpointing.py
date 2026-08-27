"""Crash-safe checkpoint persistence shared by formal benchmark trainers."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


MIN_FREE_GIB_ENV = "REGBENCH_MIN_FREE_GIB"


def ensure_checkpoint_headroom(path: Path) -> None:
    """Refuse a checkpoint write before the benchmark volume becomes unsafe.

    The server protocol freezes an 8 GiB emergency reserve.  The environment
    variable exists only to make the threshold explicit in provenance and to
    permit stricter deployments; values below 8 GiB are rejected.
    """

    requested = float(os.environ.get(MIN_FREE_GIB_ENV, "8"))
    if requested < 8.0:
        raise ValueError(f"{MIN_FREE_GIB_ENV} cannot be lower than 8 GiB.")
    probe = path.resolve().parent
    probe.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(probe).free
    required_bytes = int(requested * (1024 ** 3))
    if free_bytes < required_bytes:
        raise OSError(
            f"Checkpoint write stopped: {free_bytes / (1024 ** 3):.2f} GiB "
            f"free at {probe}, below the {requested:.2f} GiB reserve."
        )


def atomic_torch_save(payload: Mapping[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_checkpoint_headroom(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
        torch.save(payload, temporary)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_training_checkpoints(
    payload: Mapping[str, Any],
    output_dir: Path,
    *,
    epoch: int,
    improved: bool,
    checkpoint_every: int,
) -> None:
    """Persist resumable last/best plus bounded periodic recovery points."""

    atomic_torch_save(payload, output_dir / "last.pt")
    if improved:
        atomic_torch_save(payload, output_dir / "best.pt")
    if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
        atomic_torch_save(
            payload,
            output_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
        )
