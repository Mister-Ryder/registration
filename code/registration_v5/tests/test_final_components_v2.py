"""Scientific gates for the authoritative faithful-DSIR/ConvexAdam path."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from registration_v5.convex_solver_v2 import descriptor_convex_adam_v2
from registration_v5.protocol import ConvexAdamProtocol, load_protocol


def test_faithful_protocol_disables_nonpaper_auxiliaries_and_uses_crop160():
    config = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "configs"
        / "v4_final_l2r_mrct_protocol300_faithful_crop160.yaml"
    )
    protocol = load_protocol(config)
    assert protocol.training.crop_shape_dzyx == (160, 160, 160)
    assert protocol.training.geometry_probability == 0.0
    assert protocol.training.variance_weight == 0.0
    assert protocol.training.epochs == 300


def test_descriptor_convexadam_captures_twelve_voxel_translation():
    """A capture-range gate, not an identity-only smoke test.

    Moving is fixed shifted +12 along native x.  The returned fixed-grid to
    moving-sampling flow must therefore be positive in dzyx component 2.
    """

    size = 48
    shift = 12
    generator = torch.Generator().manual_seed(20260826)
    fixed = F.normalize(
        torch.randn(1, 24, size, size, size, generator=generator), dim=1
    )
    moving = torch.zeros_like(fixed)
    moving[..., shift:] = fixed[..., :-shift]
    fixed_mask = torch.zeros(1, 1, size, size, size, dtype=torch.bool)
    moving_mask = torch.zeros_like(fixed_mask)
    fixed_mask[..., 4:-4, 4:-4, 4 : size - shift - 4] = True
    moving_mask[..., 4:-4, 4:-4, shift + 4 : -4] = True
    result = descriptor_convex_adam_v2(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        config=ConvexAdamProtocol(
            grid_spacing=4,
            displacement_half_width=4,
            adam_iterations=0,
            adam_grid_spacing=2,
            inverse_consistency=True,
            inverse_consistency_iterations=5,
            selected_smoothing=0,
        ),
    )
    support = fixed_mask[:, 0]
    flow = result.flow_dzyx_voxels
    median_dz = float(flow[:, 0][support].median())
    median_dy = float(flow[:, 1][support].median())
    median_dx = float(flow[:, 2][support].median())
    assert abs(median_dz) < 2.0
    assert abs(median_dy) < 2.0
    assert abs(median_dx - shift) < 3.0
    assert float(result.diagnostics["fold_fraction"]) < 0.01
    assert result.diagnostics["adam_descriptor_cost"].startswith("0.5 * sum_c")


if __name__ == "__main__":
    test_faithful_protocol_disables_nonpaper_auxiliaries_and_uses_crop160()
    test_descriptor_convexadam_captures_twelve_voxel_translation()
    print("V4-final v2 component gates passed")
