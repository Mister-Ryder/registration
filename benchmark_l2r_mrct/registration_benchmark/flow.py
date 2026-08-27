"""Canonical fixed-grid sampling displacement and conversion utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

from .contract import FLOW_SCHEMA


def validate_flow(flow_dzyx: np.ndarray, shape_xyz: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    flow = np.asarray(flow_dzyx, dtype=np.float32)
    if flow.ndim != 4 or flow.shape[0] != 3:
        raise ValueError(f"Flow must be [3,D,H,W], got {flow.shape}.")
    if not np.isfinite(flow).all():
        raise ValueError("Flow contains non-finite values.")
    if shape_xyz is not None and tuple(reversed(flow.shape[1:])) != tuple(shape_xyz):
        raise ValueError(f"Flow grid {flow.shape[1:]} does not match fixed XYZ {shape_xyz}.")
    return flow


def zero_flow(shape_xyz: Tuple[int, int, int]) -> np.ndarray:
    x, y, z = shape_xyz
    return np.zeros((3, z, y, x), dtype=np.float32)


def xyz_last_to_dzyx(flow_xyz_last: np.ndarray) -> np.ndarray:
    """Convert [X,Y,Z,(dx,dy,dz)] to [3,D,H,W] (dz,dy,dx)."""
    value = np.asarray(flow_xyz_last, dtype=np.float32)
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"Expected [X,Y,Z,3], got {value.shape}.")
    return np.stack(
        [value[..., 2].transpose(2, 1, 0), value[..., 1].transpose(2, 1, 0), value[..., 0].transpose(2, 1, 0)],
        axis=0,
    ).astype(np.float32)


def dzyx_to_xyz_last(flow_dzyx: np.ndarray) -> np.ndarray:
    flow = validate_flow(flow_dzyx)
    return np.stack(
        [flow[2].transpose(2, 1, 0), flow[1].transpose(2, 1, 0), flow[0].transpose(2, 1, 0)],
        axis=-1,
    ).astype(np.float32)


def save_flow(path: Path, flow_dzyx: np.ndarray, fixed_affine: np.ndarray, metadata: Mapping[str, Any]) -> None:
    flow = validate_flow(flow_dzyx)
    payload: Dict[str, Any] = {
        "flow": flow,
        "schema": np.asarray(FLOW_SCHEMA),
        "component_order": np.asarray("dz,dy,dx"),
        "units": np.asarray("fixed_grid_voxels"),
        "mapping": np.asarray("fixed_output_grid_to_moving_input_sampling"),
        "fixed_affine": np.asarray(fixed_affine, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def load_flow(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        schema = str(archive["schema"].item())
        if schema != FLOW_SCHEMA:
            raise ValueError(f"Unexpected flow schema {schema!r}.")
        flow = validate_flow(archive["flow"])
        affine = np.asarray(archive["fixed_affine"], dtype=np.float64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    return flow, affine, metadata


def warp_resampled_xyz(moving_xyz: np.ndarray, flow_dzyx: np.ndarray, order: int = 1, cval: float = 0.0) -> np.ndarray:
    flow = validate_flow(flow_dzyx, tuple(int(v) for v in moving_xyz.shape))
    z, y, x = np.meshgrid(
        np.arange(flow.shape[1], dtype=np.float32),
        np.arange(flow.shape[2], dtype=np.float32),
        np.arange(flow.shape[3], dtype=np.float32),
        indexing="ij",
    )
    data_zyx = np.asarray(moving_xyz).transpose(2, 1, 0)
    warped_zyx = map_coordinates(
        data_zyx,
        [z + flow[0], y + flow[1], x + flow[2]],
        order=order,
        mode="constant",
        cval=float(cval),
        prefilter=order > 1,
    )
    return warped_zyx.transpose(2, 1, 0)


def load_native_xyz_last_displacement(path: Path, fixed_affine: np.ndarray) -> np.ndarray:
    image = nib.load(str(path))
    value = np.asarray(image.dataobj, dtype=np.float32)
    if value.ndim == 5 and value.shape[-2] == 1:
        value = value[..., 0, :]
    if value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(f"Expected vector NIfTI [X,Y,Z,3], got {value.shape}.")
    if not np.allclose(image.affine, fixed_affine, atol=1e-4):
        raise ValueError("Native displacement affine does not match the fixed grid.")
    return xyz_last_to_dzyx(value)


__all__ = [
    "dzyx_to_xyz_last", "load_flow", "load_native_xyz_last_displacement", "save_flow",
    "validate_flow", "warp_resampled_xyz", "xyz_last_to_dzyx", "zero_flow",
]
