"""Small, dependency-light TensorBoard logging helpers for benchmark trainers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional


def create_summary_writer(output_dir: Path, *, log_dir: Optional[Path] = None):
    """Create a scalar writer in the requested benchmark log directory.

    ``log_dir`` is optional for backward compatibility with existing methods.
    Formal protocol_300 runs pass the shared top-level TensorBoard directory
    explicitly so events never consume the benchmark result tree.
    """

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError as error:  # pragma: no cover - exercised in server environments
        raise RuntimeError(
            "TensorBoard is required for formal benchmark training; install the "
            "'tensorboard' package in this method environment."
        ) from error
    log_dir = log_dir.resolve() if log_dir is not None else output_dir.resolve() / "tf-logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir), max_queue=1000, flush_secs=120)


def _flatten(prefix: str, value: Any):
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            yield from _flatten(name, child)
    elif isinstance(value, bool):
        yield prefix, int(value)
    elif isinstance(value, (int, float)):
        yield prefix, value


def log_record(writer, record: Mapping[str, Any], step: int) -> None:
    """Log every numeric metric while excluding bookkeeping-only epoch fields."""

    for name, value in _flatten("", record):
        if not name or name in {"epoch", "global_step", "steps"}:
            continue
        writer.add_scalar(name, value, global_step=step)
    writer.flush()


def log_iteration(writer, record: Mapping[str, Any], optimizer_step: int) -> None:
    """Write scalar-only, per-optimizer-iteration telemetry.

    Iteration events are intentionally not flushed one by one: SummaryWriter's
    event queue preserves every scalar while avoiding a disk synchronization in
    the hot training loop. The epoch logger flushes the queue at each boundary.
    No image, histogram, graph, or tensor payload is accepted.
    """

    for name, value in _flatten("", record):
        if not name:
            continue
        writer.add_scalar(
            f"iteration/{name}", value, global_step=int(optimizer_step)
        )
