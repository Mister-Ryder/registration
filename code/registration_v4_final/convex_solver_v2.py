"""Final 24-channel DSIR ConvexAdam contract.

For L2-normalized 24-channel DSIR, ``0.5 * sum_c (f-m)^2`` is exactly
``12 * mean_c (f-m)^2``.  The official ConvexAdam adaptation uses the latter
form internally; this wrapper freezes C=24 and records the former scientific
definition so its regularization scale cannot drift with channel count.
"""

from __future__ import annotations

import torch

from .convex_solver import DescriptorConvexAdamResult, descriptor_convex_adam
from .protocol import ConvexAdamProtocol


def descriptor_convex_adam_v2(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: ConvexAdamProtocol,
) -> DescriptorConvexAdamResult:
    if fixed_descriptor.shape[1] != 24 or moving_descriptor.shape[1] != 24:
        raise ValueError("Final ConvexAdam requires the frozen 24-channel DSIR.")
    result = descriptor_convex_adam(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=config,
    )
    diagnostics = dict(result.diagnostics)
    diagnostics["adam_descriptor_cost"] = "0.5 * sum_c squared = 1-cosine for L2 DSIR"
    diagnostics["channel_contract"] = 24
    return DescriptorConvexAdamResult(result.flow_dzyx_voxels, diagnostics)


__all__ = ["descriptor_convex_adam_v2"]

