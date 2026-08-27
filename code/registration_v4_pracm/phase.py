"""Canonical phase identities used by response-aware modules."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple, Union

import torch


PHASES: Tuple[str, ...] = ("p", "c1", "c2", "c3")
ACQUISITION_IDENTITIES: Tuple[str, ...] = (*PHASES, "mr", "ct")
_ALIASES = {
    "p": "p",
    "plain": "p",
    "pre": "p",
    "precontrast": "p",
    "c1": "c1",
    "a": "c1",
    "arterial": "c1",
    "c2": "c2",
    "v": "c2",
    "venous": "c2",
    "portal": "c2",
    "portalvenous": "c2",
    "c3": "c3",
    "d": "c3",
    "delayed": "c3",
    "mr": "mr",
    "mri": "mr",
    "ct": "ct",
}


def canonical_phase(value: object) -> str:
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    try:
        return _ALIASES[key]
    except KeyError as error:
        raise ValueError(
            f"Unsupported acquisition identity {value!r}; expected P/C1/C2/C3/MR/CT."
        ) from error


def phase_index(value: object, identities: Sequence[str] = PHASES) -> int:
    canonical = canonical_phase(value)
    try:
        return tuple(identities).index(canonical)
    except ValueError as error:
        raise ValueError(
            f"Acquisition {canonical!r} is outside this model's configured identities {tuple(identities)}."
        ) from error


def phase_indices(
    value: Union[object, Sequence[object]],
    *,
    batch: int,
    device: torch.device,
    identities: Sequence[str] = PHASES,
) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        result = value.to(device=device, dtype=torch.long).flatten()
    elif isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        result = torch.full(
            (batch,), phase_index(value, identities), device=device, dtype=torch.long
        )
    else:
        result = torch.as_tensor(
            [phase_index(item, identities) for item in value], device=device, dtype=torch.long
        )
    if result.numel() == 1 and batch > 1:
        result = result.expand(batch)
    if result.shape != (batch,):
        raise ValueError(f"Phase batch must contain {batch} entries, got {result.numel()}.")
    if ((result < 0) | (result >= len(tuple(identities)))).any():
        raise ValueError("Phase indices are outside the configured vocabulary.")
    return result
