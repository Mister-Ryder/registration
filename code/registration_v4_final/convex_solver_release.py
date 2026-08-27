"""Compatibility bridge to the independently audited DNS-Convex solver.

The public V4-final runner keeps its frozen protocol/checkpoint schema while
delegating the actual explicit correspondence solve to the shared audited
implementation in ``registration_v4_pracm``.
"""

from __future__ import annotations

from dataclasses import asdict

import torch

from registration_v4_pracm.inference.dns_convex_solver import (
    DNSConvexConfig,
    solve_dns_convex,
)

from .convex_solver import DescriptorConvexAdamResult
from .protocol import ConvexAdamProtocol


def descriptor_convex_adam_release(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: ConvexAdamProtocol,
) -> DescriptorConvexAdamResult:
    if fixed_descriptor.shape[1] != 24 or moving_descriptor.shape[1] != 24:
        raise ValueError("Final V4 requires the frozen 24-channel DSIR.")
    audited = DNSConvexConfig(
        grid_spacing=config.grid_spacing,
        displacement_half_width=config.displacement_half_width,
        adam_grid_spacing=config.adam_grid_spacing,
        lambda_weight=config.diffusion_weight,
        adam_iterations=config.adam_iterations,
        adam_learning_rate=config.adam_learning_rate,
        inverse_consistency_iterations=(
            config.inverse_consistency_iterations if config.inverse_consistency else 0
        ),
        jacobian_weight=50.0,
        jacobian_margin=0.05,
        maximum_fold_fraction=1.0e-4,
        fail_on_excess_folding=True,
    )
    result = solve_dns_convex(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=audited,
    )
    diagnostics = {
        "schema": "v4_final_audited_dns_convex_diagnostics_v1",
        "flow_convention": "fixed grid to moving sampling location, dzyx native voxels",
        "descriptor_cost": "sum_c squared = 2*(1-cosine) for L2 DSIR",
        "channel_contract": 24,
        "solver": "registration_v4_pracm.inference.dns_convex_solver.solve_dns_convex",
        "parameters": asdict(audited),
        **asdict(result.diagnostics),
    }
    return DescriptorConvexAdamResult(result.flow_native_dzyx_voxels, diagnostics)


__all__ = ["descriptor_convex_adam_release"]
