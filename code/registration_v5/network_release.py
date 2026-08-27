"""Public V4-final model whose register method uses the audited solver."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .convex_solver_release import descriptor_convex_adam_release
from .network import FinalRegistrationResult, V4FinalRegistrationModel
from .protocol import ConvexAdamProtocol


class V4FinalRegistrationModelRelease(V4FinalRegistrationModel):
    def register(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_mask: torch.Tensor,
        moving_mask: torch.Tensor,
        solver: ConvexAdamProtocol,
        descriptor_autocast_dtype: Optional[torch.dtype] = None,
    ) -> FinalRegistrationResult:
        enabled = fixed.is_cuda and descriptor_autocast_dtype is not None
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=descriptor_autocast_dtype or torch.float16,
            enabled=enabled,
        ):
            fixed_descriptor = self(fixed)
            moving_descriptor = self(moving)
        result = descriptor_convex_adam_release(
            F.normalize(fixed_descriptor.float(), dim=1, eps=1e-6),
            F.normalize(moving_descriptor.float(), dim=1, eps=1e-6),
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            config=solver,
        )
        return FinalRegistrationResult(result.flow_dzyx_voxels, result.diagnostics)


__all__ = ["V4FinalRegistrationModelRelease"]
