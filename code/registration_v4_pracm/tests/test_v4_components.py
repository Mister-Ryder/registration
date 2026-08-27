"""Focused V4 component checks; no dataset experiment is performed."""

from dataclasses import replace
from pathlib import Path

import torch

from registration_v4_pracm.config import load_config
from registration_v4_pracm.data.l2r import L2RMRCTDataset
from registration_v4_pracm.data.nifti import l2r_foreground_mask, normalize_intensity
from registration_v4_pracm.losses.objective import _candidate_corner_loss
from registration_v4_pracm.model.pracm_v4 import PRACM3D
from registration_v4_pracm.model.update import CorrespondenceUpdate3D
from registration_v4_pracm.training.module_v4 import PRACMTrainingModule


def test_candidate_corner_loss_maps_zero_residual_to_center_candidate():
    radius = 2
    axis = torch.arange(-radius, radius + 1)
    offsets = torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    probabilities = torch.full((1, offsets.shape[0], 1, 1, 1), 1e-8)
    center = ((offsets == 0).all(dim=1)).nonzero(as_tuple=False).item()
    probabilities[:, center] = 1.0
    probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
    loss = _candidate_corner_loss(
        probabilities,
        offsets,
        torch.zeros(1, 3, 1, 1, 1),
        torch.ones(1, 1, 1, 1, 1, dtype=torch.bool),
    )
    assert loss.item() < 1e-5


def test_supported_candidate_pruning_does_not_shrink_update_evidence():
    module = CorrespondenceUpdate3D(
        descriptor_channels=2,
        response_channels=1,
        hidden_channels=4,
        uncertainty_floor=0.15,
        maximum_bias=0.5,
    ).eval()
    shape = (1, 1, 2, 2, 2)
    fixed = torch.zeros(1, 2, 2, 2, 2)
    warped = torch.zeros_like(fixed)
    response = torch.zeros(shape)
    current = torch.zeros(1, 3, 2, 2, 2)
    residual = torch.ones_like(current)
    variance = torch.ones_like(current)
    entropy = torch.full(shape, 0.5)
    maximum = torch.full(shape, 0.5)
    gate = torch.ones(shape)

    def run(coverage):
        return module(
            fixed,
            warped,
            response,
            current,
            residual,
            variance,
            entropy,
            maximum,
            gate,
            torch.full(shape, coverage),
            None,
        )

    sparse = run(0.05)
    dense = run(1.0)
    assert torch.allclose(sparse.flow_increment, dense.flow_increment)
    assert torch.allclose(sparse.evidence, dense.evidence)


def test_l2r_foreground_is_raw_label_free_and_padding_remains_zero():
    mr = torch.zeros(1, 1, 8, 8, 8)
    mr[..., 2:6, 2:6, 2:6] = torch.linspace(0.1, 1.0, 64).reshape(1, 1, 4, 4, 4)
    ct = torch.zeros_like(mr)
    ct[..., 1:7, 1:7, 1:7] = -1000.0
    ct[..., 2:6, 2:6, 2:6] = 45.0
    valid = torch.ones_like(mr, dtype=torch.bool)

    mr_support = l2r_foreground_mask(mr, "mr", domain=valid)
    ct_support = l2r_foreground_mask(ct, "ct", domain=valid)
    assert 0 < mr_support.float().mean() < 1
    assert 0 < ct_support.float().mean() < 1
    assert not mr_support[..., 0, 0, 0].item()
    assert not ct_support[..., 0, 0, 0].item()
    assert not ct_support[..., 1, 1, 1].item()

    mr_normalized = normalize_intensity(mr, "zscore", domain=mr_support)
    ct_normalized = normalize_intensity(ct, "zscore", domain=ct_support)
    assert torch.count_nonzero(mr_normalized[~mr_support]) == 0
    assert torch.count_nonzero(ct_normalized[~ct_support]) == 0

    cropped_support = L2RMRCTDataset._crop_pad(
        mr_support[0].float(), (-2, -1, -3), (8, 8, 8)
    ) > 0.5
    cropped_image = L2RMRCTDataset._crop_pad(
        mr_normalized[0], (-2, -1, -3), (8, 8, 8)
    ).masked_fill(~cropped_support, 0)
    assert torch.count_nonzero(cropped_support[..., :2, :, :]) == 0
    assert torch.count_nonzero(cropped_image[~cropped_support]) == 0


def _small_model(config, variant):
    return replace(
        config.model,
        variant=variant,
        response_gate_mode="calibrated" if variant == "v4_full" else "neutral",
        encoder_channels=(8, 12, 16, 20),
        descriptor_channels=8,
        response_channels=4,
        search_radii=(2, 1, 1),
        recurrent_iterations=(1, 1, 1),
        appearance_samples=64,
        update_hidden_channels=12,
        phase_embedding_channels=4,
        candidate_chunk_size=4,
        posterior_topk=3,
        solver_iterations_train=2,
        solver_iterations_inference=2,
    )


def test_v4_variant_mechanisms_and_gradients():
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs/pracm_v4_full_l2r_mrct_protocol300.yaml")
    image = torch.rand(1, 1, 32, 32, 32)
    domain = torch.ones_like(image, dtype=torch.bool)
    spacing = torch.tensor([[2.0, 2.0, 2.0]])

    for variant in ("v4a", "v4b", "v4_full"):
        model = PRACM3D(_small_model(config, variant)).eval()
        with torch.no_grad():
            output = model(
                image,
                image,
                fixed_phase="mr",
                moving_phase="ct",
                fixed_domain=domain,
                moving_domain=domain,
                spacing_dzyx=spacing,
                retain_distributions=False,
            )
        assert output.flow.shape == (1, 3, 32, 32, 32)
        assert torch.isfinite(output.flow).all()
        if variant == "v4a":
            assert torch.count_nonzero(output.posterior_solver_correction) == 0
        if variant == "v4_full":
            assert output.levels[-1].iterations[-1].distribution.topk_residuals.shape[1] == 3

    training = replace(
        config.training,
        representation_stage_epochs=1,
        candidate_ramp_epochs=1,
        deformation_ramp_epochs=1,
    )
    module = PRACMTrainingModule(
        _small_model(config, "v4_full"),
        config.losses,
        config.augmentation,
        training,
    )
    module.set_training_state(epoch=5, global_step=1)
    result = module(
        {
            "kind": "unpaired_domains",
            "mr": image,
            "ct": image.flip(-1),
            "mr_domain": domain,
            "ct_domain": domain,
            "mr_spacing_dzyx": spacing,
            "ct_spacing_dzyx": spacing,
        }
    )
    assert "diagnostic_candidate_top5" in result.losses
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )

