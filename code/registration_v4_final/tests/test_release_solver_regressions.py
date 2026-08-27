"""Regression gates for the audited solver used by the formal V4 release."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from registration_v4_pracm.inference.dns_convex_solver import (
    DNSConvexConfig,
    _refinement_objective,
)
from registration_v4_pracm.tests.test_dns_convex_solver import _coordinate_descriptor

from registration_v4_final.convex_solver_release import descriptor_convex_adam_release
from registration_v4_final.protocol import ConvexAdamProtocol


def _to_24_channels(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(torch.cat((value, value), dim=1), dim=1)


def test_release_recovers_non_cubic_translation_on_all_dzyx_axes():
    translation = 20.0
    fixed, moving, fixed_mask, moving_mask = _coordinate_descriptor(
        (30, 36, 72), translation
    )
    # Original shift is x.  Spatial permutations place the identical physical
    # construction on y and z while retaining a deliberately non-cubic grid.
    cases = (
        (fixed, moving, fixed_mask, moving_mask, 2),
        (
            fixed.permute(0, 1, 2, 4, 3),
            moving.permute(0, 1, 2, 4, 3),
            fixed_mask.permute(0, 1, 2, 4, 3),
            moving_mask.permute(0, 1, 2, 4, 3),
            1,
        ),
        (
            fixed.permute(0, 1, 4, 3, 2),
            moving.permute(0, 1, 4, 3, 2),
            fixed_mask.permute(0, 1, 4, 3, 2),
            moving_mask.permute(0, 1, 4, 3, 2),
            0,
        ),
    )
    for fixed_case, moving_case, fixed_support, moving_support, component in cases:
        result = descriptor_convex_adam_release(
            _to_24_channels(fixed_case.contiguous()),
            _to_24_channels(moving_case.contiguous()),
            fixed_mask=fixed_support.contiguous(),
            moving_mask=moving_support.contiguous(),
            config=ConvexAdamProtocol(),
        )
        estimated = result.flow_dzyx_voxels[:, component : component + 1][fixed_support]
        assert abs(float(estimated.mean()) - translation) < 2.5
        assert float((estimated - translation).abs().mean()) < 2.5
        assert float(result.diagnostics["fold_fraction"]) <= 1.0e-4
        assert float(result.diagnostics["minimum_jacobian"]) > 0.0
        assert float(result.diagnostics["valid_fraction_of_fixed"]) > 0.90


def test_refinement_cannot_reduce_cost_by_escaping_the_moving_mask():
    shape = (12, 14, 16)
    generator = torch.Generator().manual_seed(20260826)
    fixed = F.normalize(torch.randn(1, 24, *shape, generator=generator), dim=1)
    moving = fixed.clone()
    occupancy = torch.ones(1, 1, *shape)
    identity = torch.zeros(1, 3, *shape)
    escaped = identity.clone()
    escaped[:, 2] = 4.0 * shape[2]
    config = DNSConvexConfig()
    identity_similarity, _, _, identity_evidence = _refinement_objective(
        fixed,
        moving,
        occupancy,
        occupancy,
        identity,
        native_shape=shape,
        config=config,
    )
    escaped_similarity, _, _, escaped_evidence = _refinement_objective(
        fixed,
        moving,
        occupancy,
        occupancy,
        escaped,
        native_shape=shape,
        config=config,
    )
    assert float(identity_evidence.mean()) > 0.99
    assert float(escaped_evidence.mean()) < 0.01
    assert float(identity_similarity) < 1.0e-5
    assert float(escaped_similarity) > 1.9


if __name__ == "__main__":
    test_release_recovers_non_cubic_translation_on_all_dzyx_axes()
    test_refinement_cannot_reduce_cost_by_escaping_the_moving_mask()
    print("audited release solver regressions passed")
