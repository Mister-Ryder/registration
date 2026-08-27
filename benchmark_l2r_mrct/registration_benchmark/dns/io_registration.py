"""Three-level instance optimization driven by pretrained DNS descriptors."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from ..adapters.base import AdapterResult
from ..contract import PairTask
from ..io import load_pair_on_fixed_grid, robust_unit_interval
from .model import MASRNet


def _grid(flow: torch.Tensor) -> torch.Tensor:
    """Sampling grid for a displacement expressed in normalized dzyx units."""

    b, _, d, h, w = flow.shape
    zz, yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, d, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1, 1, h, device=flow.device, dtype=flow.dtype),
        torch.linspace(-1, 1, w, device=flow.device, dtype=flow.dtype),
        indexing="ij",
    )
    base = torch.stack([xx, yy, zz], dim=-1)[None].expand(b, -1, -1, -1, -1)
    delta = torch.stack([
        flow[:, 2],
        flow[:, 1],
        flow[:, 0],
    ], dim=-1)
    return base + delta


def _normalized_displacement_to_voxels(
    flow: torch.Tensor, shape: Optional[Sequence[int]] = None
) -> torch.Tensor:
    """Convert normalized dzyx displacement to align_corners voxel units."""

    d, h, w = tuple(int(value) for value in (shape or flow.shape[-3:]))
    factors = flow.new_tensor([(d - 1) / 2, (h - 1) / 2, (w - 1) / 2])
    return flow * factors.view(1, 3, 1, 1, 1)


def _smoothness(flow: torch.Tensor) -> torch.Tensor:
    terms = []
    for axis in (-3, -2, -1):
        left = [slice(None)] * flow.ndim
        right = [slice(None)] * flow.ndim
        left[axis] = slice(1, None)
        right[axis] = slice(None, -1)
        terms.append((flow[tuple(left)] - flow[tuple(right)]).square().mean())
    return torch.stack(terms).mean()


def _gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    radius = max(int(round(3 * sigma)), 1)
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).square())
    return kernel / kernel.sum()


def _gaussian_smooth(features: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return features
    kernel = _gaussian_kernel1d(sigma, features.device, features.dtype)
    channels = features.shape[1]
    result = features
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = kernel.numel()
        weight = kernel.reshape(1, 1, *shape).repeat(channels, 1, 1, 1, 1)
        padding = [0, 0, 0]
        padding[axis] = kernel.numel() // 2
        result = F.conv3d(result, weight, padding=tuple(padding), groups=channels)
    return F.normalize(result, dim=1, eps=1e-6)


def optimize_dns_flow(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    scales: Sequence[float] = (0.5, 0.75, 1.0),
    learning_rates: Sequence[float] = (1e-2, 5e-3, 3e-3),
    iterations: Sequence[int] = (100, 80, 50),
    smoothness_weights: Sequence[float] = (0.6, 0.5, 0.4),
) -> torch.Tensor:
    if not (len(scales) == len(learning_rates) == len(iterations) == len(smoothness_weights) == 3):
        raise ValueError("Mok IO reproduction requires exactly three pyramid levels.")
    # IO optimizes only the per-case displacement. Explicit detachment prevents
    # accidental backpropagation into MASR and makes this invariant independent
    # of the caller's surrounding grad context.
    fixed_descriptor = fixed_descriptor.detach()
    moving_descriptor = moving_descriptor.detach()
    native_shape = fixed_descriptor.shape[-3:]
    flow = None
    for scale, lr, n_iter, regularization in zip(scales, learning_rates, iterations, smoothness_weights):
        shape = tuple(max(2, int(round(size * float(scale)))) for size in native_shape)
        fixed = F.normalize(
            F.interpolate(fixed_descriptor, size=shape, mode="trilinear", align_corners=True),
            dim=1,
            eps=1e-6,
        )
        moving = F.normalize(
            F.interpolate(moving_descriptor, size=shape, mode="trilinear", align_corners=True),
            dim=1,
            eps=1e-6,
        )
        if flow is None:
            flow = torch.zeros(
                (fixed.shape[0], 3, *shape), device=fixed.device, dtype=fixed.dtype
            )
        else:
            flow = F.interpolate(
                flow.detach(), size=shape, mode="trilinear", align_corners=True
            )
        flow.requires_grad_(True)
        optimizer = torch.optim.Adam([flow], lr=float(lr))
        for _ in range(int(n_iter)):
            warped = F.grid_sample(moving, _grid(flow), mode="bilinear", padding_mode="border", align_corners=True)
            similarity = 1.0 - (fixed * F.normalize(warped, dim=1, eps=1e-6)).sum(dim=1).mean()
            loss = similarity + float(regularization) * _smoothness(flow)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        flow = flow.detach()
    if tuple(flow.shape[-3:]) != tuple(native_shape):
        flow = F.interpolate(flow, size=native_shape, mode="trilinear", align_corners=True)
    return _normalized_displacement_to_voxels(flow, native_shape)


class DNSIOAdapter:
    def register(self, task: PairTask, config: Mapping[str, Any], work_dir: Path) -> AdapterResult:
        device = torch.device(str(config.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        checkpoint_path = Path(str(config["checkpoint"])).resolve()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        architecture = checkpoint.get("architecture", {})
        model = MASRNet(**architecture).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        fixed, moving = load_pair_on_fixed_grid(task.fixed.path, task.moving.path)
        def tensor(data: np.ndarray) -> torch.Tensor:
            normalized = robust_unit_interval(data).transpose(2, 1, 0)
            return torch.from_numpy(normalized.copy())[None, None].to(device)
        with torch.no_grad():
            fixed_descriptor = model(tensor(fixed.data_xyz))
            moving_descriptor = model(tensor(moving.data_xyz))
            sigma = float(config.get("descriptor_gaussian_sigma", 1.0))
            fixed_descriptor = _gaussian_smooth(fixed_descriptor, sigma)
            moving_descriptor = _gaussian_smooth(moving_descriptor, sigma)
        flow = optimize_dns_flow(
            fixed_descriptor,
            moving_descriptor,
            scales=config.get("scales", [0.5, 0.75, 1.0]),
            learning_rates=config.get("learning_rates", [1e-2, 5e-3, 3e-3]),
            iterations=config.get("iterations", [100, 80, 50]),
            smoothness_weights=config.get("smoothness_weights", [0.6, 0.5, 0.4]),
        )
        return AdapterResult(
            flow[0].cpu().numpy().astype(np.float32),
            {
                "implementation": "paper reproduction MASR-Net DNS + three-level IO",
                "checkpoint": str(checkpoint_path),
                "paper_disclosed_io": {"learning_rates": [1e-2, 5e-3, 3e-3], "iterations": [100, 80, 50], "smoothness_weights": [0.6, 0.5, 0.4]},
                "reproduction_assumptions": {
                    "pyramid_scales": list(config.get("scales", [0.5, 0.75, 1.0])),
                    "dns_dilation_voxels": architecture.get("dns_dilation", 2),
                    "io_displacement_parameterization": "normalized_grid",
                    "pyramid_descriptors_renormalized": True,
                },
            },
        )
