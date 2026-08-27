"""Small component gates; these do not train or touch public labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from registration_benchmark.dns.faithful_v2 import FullResolutionDSIRExtractor
from registration_v4_final.convex_solver import descriptor_convex_adam
from registration_v4_final.data import raw_foreground_xyz
from registration_v4_final.protocol import ConvexAdamProtocol, V4FinalProtocol


def test_faithful_descriptor_is_full_resolution_and_normalized():
    model = FullResolutionDSIRExtractor().eval()
    image = torch.rand(1, 1, 16, 20, 24)
    with torch.no_grad():
        descriptor = model(image)
    assert descriptor.shape == (1, 24, 16, 20, 24)
    assert torch.allclose(
        descriptor.square().sum(dim=1).sqrt(),
        torch.ones_like(descriptor[:, 0]),
        atol=2e-4,
        rtol=2e-4,
    )


def test_raw_foreground_is_computed_before_ct_windowing():
    values = torch.tensor([-1024.0, -400.0, 0.0, 50.0]).numpy().reshape(2, 2, 1)
    mask = raw_foreground_xyz(values, "ct", ct_min_hu=-500.0)
    assert mask.reshape(-1).tolist() == [0, 1, 0, 1]


def test_protocol_freezes_descriptor_convexadam_as_main_solver():
    protocol = V4FinalProtocol()
    assert protocol.solver.backend == "descriptor_convexadam"
    assert protocol.descriptor.descriptor_channels == 24


def test_descriptor_convexadam_identity_has_no_fold_on_cpu():
    generator = torch.Generator().manual_seed(20260826)
    descriptor = F.normalize(
        torch.randn(1, 4, 16, 16, 16, generator=generator), dim=1
    )
    mask = torch.zeros(1, 1, 16, 16, 16, dtype=torch.bool)
    mask[..., 2:-2, 2:-2, 2:-2] = True
    result = descriptor_convex_adam(
        descriptor,
        descriptor.clone(),
        fixed_mask=mask,
        moving_mask=mask,
        config=ConvexAdamProtocol(
            grid_spacing=4,
            displacement_half_width=1,
            adam_iterations=2,
            adam_grid_spacing=2,
            inverse_consistency_iterations=2,
        ),
    )
    assert result.flow_dzyx_voxels.shape == (1, 3, 16, 16, 16)
    assert float(result.flow_dzyx_voxels.abs().mean()) < 0.5
    assert float(result.diagnostics["fold_fraction"]) == 0.0

