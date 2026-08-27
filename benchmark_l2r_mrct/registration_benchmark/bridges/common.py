from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple
import random

import numpy as np
import torch
import torch.nn.functional as F

from ..io import load_pair_on_fixed_grid, load_scalar, robust_unit_interval


def load_image_tensor(path: Path, device: torch.device, shape_zyx: Sequence[int] = ()) -> torch.Tensor:
    """Load one independent domain image without inventing a paired subject."""

    volume = load_scalar(path)
    value = robust_unit_interval(volume.data_xyz).transpose(2, 1, 0).copy()
    tensor = torch.from_numpy(value)[None, None].to(device)
    if shape_zyx:
        tensor = F.interpolate(
            tensor, size=tuple(int(v) for v in shape_zyx),
            mode="trilinear", align_corners=True,
        )
    return tensor


def load_pair_tensors(fixed: Path, moving: Path, device: torch.device, shape_zyx: Sequence[int] = ()):
    fixed_volume, moving_volume = load_pair_on_fixed_grid(fixed, moving)
    def convert(data: np.ndarray) -> torch.Tensor:
        value = robust_unit_interval(data).transpose(2, 1, 0).copy()
        tensor = torch.from_numpy(value)[None, None].to(device)
        if shape_zyx:
            tensor = F.interpolate(tensor, size=tuple(int(v) for v in shape_zyx), mode="trilinear", align_corners=True)
        return tensor
    return fixed_volume, moving_volume, convert(fixed_volume.data_xyz), convert(moving_volume.data_xyz)


def resize_flow_to_native(flow_dzyx: torch.Tensor, native_shape_zyx: Sequence[int]) -> np.ndarray:
    if flow_dzyx.ndim == 4:
        flow_dzyx = flow_dzyx[None]
    if flow_dzyx.ndim != 5 or flow_dzyx.shape[1] != 3:
        raise ValueError(f"Expected [B,3,D,H,W] flow, got {tuple(flow_dzyx.shape)}")
    source_shape = flow_dzyx.shape[-3:]
    target_shape = tuple(int(v) for v in native_shape_zyx)
    result = F.interpolate(flow_dzyx, size=target_shape, mode="trilinear", align_corners=True)
    for component, (target, source) in enumerate(zip(target_shape, source_shape)):
        result[:, component] *= max(target - 1, 1) / max(source - 1, 1)
    return result[0].detach().cpu().numpy().astype(np.float32)


def checkpoint_state(path: Path, device: torch.device) -> Mapping[str, torch.Tensor]:
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict):
        for key in ("model_state_dict", "model", "state_dict", "network"):
            if key in raw and isinstance(raw[key], dict):
                return raw[key]
    if not isinstance(raw, dict):
        raise ValueError(f"Unsupported checkpoint object in {path}")
    return raw


def warp(moving: torch.Tensor, flow_dzyx: torch.Tensor) -> torch.Tensor:
    b, _, d, h, w = flow_dzyx.shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, d, device=flow_dzyx.device, dtype=flow_dzyx.dtype),
        torch.linspace(-1, 1, h, device=flow_dzyx.device, dtype=flow_dzyx.dtype),
        torch.linspace(-1, 1, w, device=flow_dzyx.device, dtype=flow_dzyx.dtype),
        indexing="ij",
    )
    base = torch.stack([x, y, z], dim=-1)[None].expand(b, -1, -1, -1, -1)
    delta = torch.stack([
        flow_dzyx[:, 2] * 2 / max(w - 1, 1),
        flow_dzyx[:, 1] * 2 / max(h - 1, 1),
        flow_dzyx[:, 0] * 2 / max(d - 1, 1),
    ], dim=-1)
    return F.grid_sample(moving, base + delta, mode="bilinear", padding_mode="border", align_corners=True)


def gradient_loss(flow: torch.Tensor) -> torch.Tensor:
    return (
        (flow[:, :, 1:] - flow[:, :, :-1]).square().mean()
        + (flow[:, :, :, 1:] - flow[:, :, :, :-1]).square().mean()
        + (flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1]).square().mean()
    ) / 3.0


def capture_rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    if not state:
        return
    # ``torch.load(..., map_location=device)`` also moves serialized RNG byte
    # tensors to CUDA.  PyTorch's RNG restoration APIs require CPU byte
    # tensors, so normalize them explicitly for reliable resume on GPU.
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])
