"""3-D sampling operations with one fixed-grid -> moving-grid convention.

All flows are ``[B,3,D,H,W]`` in ``(dz,dy,dx)`` voxel units.  Therefore
``warp(moving, flow)[z,y,x] = moving[z+dz,y+dy,x+dx]``.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F


def _check_flow(flow: torch.Tensor) -> None:
    if flow.ndim != 5 or flow.shape[1] != 3:
        raise ValueError(f"flow must be [B,3,D,H,W], got {tuple(flow.shape)}")


def base_grid(flow: torch.Tensor, *, align_corners: bool = True) -> torch.Tensor:
    _check_flow(flow)
    batch, _, depth, height, width = flow.shape
    if align_corners:
        zs = torch.linspace(-1, 1, depth, device=flow.device, dtype=flow.dtype)
        ys = torch.linspace(-1, 1, height, device=flow.device, dtype=flow.dtype)
        xs = torch.linspace(-1, 1, width, device=flow.device, dtype=flow.dtype)
    else:
        zs = (torch.arange(depth, device=flow.device, dtype=flow.dtype) + 0.5) * (2 / depth) - 1
        ys = (torch.arange(height, device=flow.device, dtype=flow.dtype) + 0.5) * (2 / height) - 1
        xs = (torch.arange(width, device=flow.device, dtype=flow.dtype) + 0.5) * (2 / width) - 1
    grid_z, grid_y, grid_x = torch.meshgrid(zs, ys, xs, indexing="ij")
    return torch.stack((grid_x, grid_y, grid_z), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1, -1)


def flow_to_normalized(flow: torch.Tensor, *, align_corners: bool = True) -> torch.Tensor:
    _check_flow(flow)
    depth, height, width = flow.shape[-3:]
    if align_corners:
        sz, sy, sx = 2 / max(depth - 1, 1), 2 / max(height - 1, 1), 2 / max(width - 1, 1)
    else:
        sz, sy, sx = 2 / depth, 2 / height, 2 / width
    return torch.stack((flow[:, 2] * sx, flow[:, 1] * sy, flow[:, 0] * sz), dim=-1)


def sampling_valid(flow: torch.Tensor) -> torch.Tensor:
    _check_flow(flow)
    depth, height, width = flow.shape[-3:]
    zs = torch.arange(depth, device=flow.device, dtype=flow.dtype)
    ys = torch.arange(height, device=flow.device, dtype=flow.dtype)
    xs = torch.arange(width, device=flow.device, dtype=flow.dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")
    return (
        (z[None] + flow[:, 0] >= 0)
        & (z[None] + flow[:, 0] <= depth - 1)
        & (y[None] + flow[:, 1] >= 0)
        & (y[None] + flow[:, 1] <= height - 1)
        & (x[None] + flow[:, 2] >= 0)
        & (x[None] + flow[:, 2] <= width - 1)
    ).unsqueeze(1)


def warp(
    source: torch.Tensor,
    flow: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
    return_valid: bool = False,
):
    _check_flow(flow)
    if source.ndim != 5 or source.shape[0] != flow.shape[0] or source.shape[-3:] != flow.shape[-3:]:
        raise ValueError("source and flow must share [B,D,H,W].")
    grid = base_grid(flow, align_corners=align_corners) + flow_to_normalized(
        flow, align_corners=align_corners
    )
    result = F.grid_sample(
        source,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return (result, sampling_valid(flow)) if return_valid else result


def resize_flow(
    flow: torch.Tensor,
    size: Sequence[int],
    *,
    align_corners: bool = True,
) -> torch.Tensor:
    _check_flow(flow)
    old = flow.shape[-3:]
    new = tuple(int(value) for value in size)
    if old == new:
        return flow
    resized = F.interpolate(flow, size=new, mode="trilinear", align_corners=align_corners)
    if align_corners:
        scale = [
            (new[i] - 1) / (old[i] - 1) if old[i] > 1 and new[i] > 1 else 0.0
            for i in range(3)
        ]
    else:
        scale = [new[i] / old[i] for i in range(3)]
    return resized * resized.new_tensor(scale).view(1, 3, 1, 1, 1)


def compose_flows(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    align_corners: bool = True,
    return_valid: bool = False,
):
    """Compose A->B ``first`` and B->C ``second`` into A->C."""

    if first.shape != second.shape:
        raise ValueError("Flow composition requires equal grids.")
    sampled, valid = warp(
        second,
        first,
        padding_mode="zeros",
        align_corners=align_corners,
        return_valid=True,
    )
    result = first + sampled
    return (result, valid) if return_valid else result


def _forward_difference(value: torch.Tensor, dim: int) -> torch.Tensor:
    head = value.narrow(dim, 1, value.shape[dim] - 1) - value.narrow(dim, 0, value.shape[dim] - 1)
    tail = head.select(dim, head.shape[dim] - 1).unsqueeze(dim)
    return torch.cat((head, tail), dim=dim)


def jacobian_determinant(flow: torch.Tensor) -> torch.Tensor:
    """Jacobian determinant of ``identity + flow`` in tensor voxel coordinates."""

    _check_flow(flow)
    dz = _forward_difference(flow, 2)
    dy = _forward_difference(flow, 3)
    dx = _forward_difference(flow, 4)
    j00, j01, j02 = 1 + dz[:, 0], dy[:, 0], dx[:, 0]
    j10, j11, j12 = dz[:, 1], 1 + dy[:, 1], dx[:, 1]
    j20, j21, j22 = dz[:, 2], dy[:, 2], 1 + dx[:, 2]
    return (
        j00 * (j11 * j22 - j12 * j21)
        - j01 * (j10 * j22 - j12 * j20)
        + j02 * (j10 * j21 - j11 * j20)
    ).unsqueeze(1)


def candidate_offsets(radius: int, *, device, dtype) -> torch.Tensor:
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    dz, dy, dx = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.stack((dz, dy, dx), dim=-1).reshape(-1, 3)


def candidate_grid(
    base_flow: torch.Tensor,
    offsets: torch.Tensor,
    *,
    align_corners: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    _check_flow(base_flow)
    if offsets.ndim != 2 or offsets.shape[1] != 3:
        raise ValueError("offsets must be [K,3] in dzyx order.")
    batch, _, depth, height, width = base_flow.shape
    count = offsets.shape[0]
    flows = base_flow[:, None] + offsets.view(1, count, 3, 1, 1, 1)
    flat = flows.reshape(batch * count, 3, depth, height, width)
    grid = base_grid(flat, align_corners=align_corners) + flow_to_normalized(
        flat, align_corners=align_corners
    )
    valid = sampling_valid(flat).reshape(batch, count, depth, height, width)
    return grid.reshape(batch, count, depth, height, width, 3), valid


def sample_candidate_grid(
    source: torch.Tensor,
    grid: torch.Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample ``source`` at K grids without copying the source K times."""

    if source.ndim != 5 or grid.ndim != 6 or grid.shape[-1] != 3:
        raise ValueError("Expected source [B,C,D,H,W], grid [B,K,D,H,W,3].")
    batch, channels, depth, height, width = source.shape
    if grid.shape[0] != batch or grid.shape[2:5] != (depth, height, width):
        raise ValueError("Candidate grid and source shapes differ.")
    count = grid.shape[1]
    packed = grid.reshape(batch, count * depth, height, width, 3).to(source.dtype)
    sampled = F.grid_sample(
        source,
        packed,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.reshape(batch, channels, count, depth, height, width).permute(
        0, 2, 1, 3, 4, 5
    )

