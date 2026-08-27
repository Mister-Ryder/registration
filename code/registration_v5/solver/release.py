"""Frozen V4 descriptor/checkpoint adapter for the isolated solver corefix."""

from __future__ import annotations

from dataclasses import asdict

import torch

from ..convex_solver import DescriptorConvexAdamResult
from ..protocol import ConvexAdamProtocol

from .corefix import SolverCorefixConfig, solve_corefix


SOLVER_IMPLEMENTATION_ID = "v4_dense_coarse_compose_identity_corefix_v1"


def descriptor_convex_adam_corefix(
    fixed_descriptor: torch.Tensor,
    moving_descriptor: torch.Tensor,
    *,
    fixed_mask: torch.Tensor,
    moving_mask: torch.Tensor,
    config: ConvexAdamProtocol,
) -> DescriptorConvexAdamResult:
    if fixed_descriptor.ndim != 5 or moving_descriptor.shape != fixed_descriptor.shape:
        raise ValueError("Descriptors must be equal [1,24,D,H,W] tensors.")
    if fixed_descriptor.shape[0] != 1 or fixed_descriptor.shape[1] != 24:
        raise ValueError("The frozen V4-Core adapter requires one 24-channel pair.")
    selected = SolverCorefixConfig.from_protocol(config)
    result = solve_corefix(
        fixed_descriptor,
        moving_descriptor,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=selected,
    )
    diagnostics = {
        "schema": "v4_solver_corefix_release_diagnostics_v1",
        "solver_implementation_id": SOLVER_IMPLEMENTATION_ID,
        "descriptor_checkpoint_contract": "frozen_v4_core_24_channel_dsir_unchanged",
        "flow_convention": "fixed grid to moving sampling location, dzyx native voxels",
        "descriptor_cost": "sum_c squared = 2*(1-cosine) for L2 DSIR",
        "solver_only_frozen_v4_checkpoint_reuse": True,
        "corefix_parameters": asdict(selected),
        **result.diagnostics,
    }
    return DescriptorConvexAdamResult(
        flow_dzyx_voxels=result.flow_native_dzyx_voxels.detach(),
        diagnostics=diagnostics,
    )


__all__ = ["SOLVER_IMPLEMENTATION_ID", "descriptor_convex_adam_corefix"]
