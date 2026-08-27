from __future__ import annotations

import random
import json

import nibabel as nib
import numpy as np
import torch

from registration_v3_pracm.config import AugmentationConfig, DataConfig, LossConfig, ModelConfig
from registration_v3_pracm.data import (
    L2RMRCTDataset,
    MAIN_DIRECTIONS,
    RELATIONAL_TRIANGLES,
    PLCRVolumeDataset,
)
from registration_v3_pracm.model import PRACM3D
from registration_v3_pracm.ops.spatial import compose_flows, warp
from registration_v3_pracm.phase import PHASES, canonical_phase, phase_index
from registration_v3_pracm.training.augmentation import make_synthetic_pair
from registration_v3_pracm.training.module import PRACMTrainingModule


def tiny_model_config() -> ModelConfig:
    return ModelConfig(
        encoder_channels=(8, 12, 16),
        descriptor_channels=8,
        response_channels=4,
        correlation_levels=(2, 1),
        search_radii=(1, 1),
        recurrent_iterations=(1, 1),
        dns_dilations=(1,),
        update_hidden_channels=12,
        phase_embedding_channels=4,
        candidate_chunk_size=9,
    )


def test_four_phase_protocol_keeps_plain_independent():
    assert PHASES == ("p", "c1", "c2", "c3")
    assert canonical_phase("P") == "p"
    assert canonical_phase("plain") == "p"
    assert phase_index("P") != phase_index("C1")
    assert MAIN_DIRECTIONS == (("p", "c1"), ("p", "c2"), ("p", "c3"))
    assert RELATIONAL_TRIANGLES == (("p", "c1", "c2"), ("p", "c2", "c3"))


def test_l2r_acquisition_vocabulary_does_not_change_plc_checkpoint_shapes():
    plc = PRACM3D(tiny_model_config())
    assert plc.response_phase_classifier.out_features == 4
    l2r_config = tiny_model_config()
    l2r_config = ModelConfig(
        **{
            **l2r_config.__dict__,
            "acquisition_identities": ("mr", "ct"),
        }
    )
    l2r = PRACM3D(l2r_config).eval()
    assert l2r.response_phase_classifier.out_features == 2
    image = torch.rand(1, 1, 24, 24, 24)
    with torch.no_grad():
        output = l2r(image, image, fixed_phase="mr", moving_phase="ct")
    assert output.flow.shape == (1, 3, 24, 24, 24)


def test_l2r_neutral_gate_cannot_randomly_rerank_candidates():
    base = tiny_model_config()
    config = ModelConfig(
        **{
            **base.__dict__,
            "acquisition_identities": ("mr", "ct"),
            "response_gate_mode": "neutral",
        }
    )
    model = PRACM3D(config)
    fixed = torch.randn(1, config.response_channels, 3, 4, 5)
    candidates = torch.randn(1, 7, config.response_channels, 3, 4, 5)
    gate = model.response_gate(
        fixed,
        candidates,
        torch.tensor([0]),
        torch.tensor([1]),
    )
    assert torch.equal(gate, torch.ones_like(gate))


def test_l2r_unpaired_loader_uses_independent_native_domain_patches(tmp_path):
    paths = {}
    specifications = {
        "mr_train": ((19, 21, 23), np.diag([1.1, 1.2, 1.3, 1.0])),
        "ct_train": ((27, 25, 17), np.diag([2.0, 1.5, 0.9, 1.0])),
        "mr_val": ((18, 20, 22), np.eye(4)),
        "ct_val": ((24, 26, 28), np.diag([0.8, 0.9, 1.0, 1.0])),
    }
    for name, (shape, affine) in specifications.items():
        path = tmp_path / f"{name}.nii.gz"
        data = np.linspace(0, 1, num=int(np.prod(shape)), dtype=np.float32).reshape(shape)
        nib.save(nib.Nifti1Image(data, affine), str(path))
        paths[name] = path
    pairs = []
    for split in ("train", "validation"):
        pairs.append(
            {
                "pair_id": f"unpaired_{split}",
                "subject_id": f"unpaired_{split}",
                "split": split,
                "task_domain": "l2r_mrct",
                "task_group": "unpaired_training",
                "fixed": {
                    "path": str(paths[f"mr_{'train' if split == 'train' else 'val'}"]),
                    "phase": "mr",
                    "modality": "mr",
                },
                "moving": {
                    "path": str(paths[f"ct_{'train' if split == 'train' else 'val'}"]),
                    "phase": "ct",
                    "modality": "ct",
                },
            }
        )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "registration_pair_manifest_v1", "pairs": pairs}),
        encoding="utf-8",
    )
    dataset = L2RMRCTDataset(
        DataConfig(
            dataset="l2r_mrct",
            manifest=str(manifest),
            intensity_mode="zscore",
            patch_size=(16, 16, 16),
            samples_per_epoch=1,
            triplet_probability=0.0,
            minimum_content_fraction=0.01,
        ),
        split="train",
        seed=3,
    )
    sample = dataset[0]
    assert sample["kind"] == "unpaired_domains"
    assert sample["mr"].shape == sample["ct"].shape == (1, 16, 16, 16)
    assert sample["mr_domain"].shape == sample["ct_domain"].shape
    identity = dataset.source_identity()
    assert identity["cross_modal_pair_supervision"] is False
    assert identity["native_grid_sampling"] is True


def test_validation_patch_is_fixed_label_centered():
    dataset = PLCRVolumeDataset.__new__(PLCRVolumeDataset)
    dataset.config = DataConfig(
        patch_size=(16, 16, 16),
        minimum_content_fraction=0.01,
    )
    dataset.load_labels = True
    dataset.pair_only = True
    image = torch.ones(1, 32, 32, 32)
    domain = torch.ones_like(image, dtype=torch.bool)
    label = torch.zeros_like(domain)
    label[:, 20:24, 18:22, 16:20] = True
    volumes = {
        phase: (image.clone(), domain.clone(), label.clone())
        for phase in ("p", "c1", "c2", "c3")
    }
    patch = dataset._patch(volumes, random.Random(1), ("p", "c2"))
    assert patch["p"][0].shape == (1, 16, 16, 16)
    assert patch["p"][2].any()


def test_translation_composition_uses_sampling_convention():
    first = torch.zeros(1, 3, 8, 8, 8)
    second = torch.zeros_like(first)
    first[:, 2] = 1.0
    second[:, 1] = 2.0
    composed = compose_flows(first, second)
    assert torch.allclose(composed[:, 2, :, :, :-1], torch.ones_like(composed[:, 2, :, :, :-1]))
    assert torch.allclose(composed[:, 1, :, :, :-1], 2 * torch.ones_like(composed[:, 1, :, :, :-1]))


def test_synthetic_geometry_has_exact_known_correspondence():
    image = torch.rand(1, 1, 20, 20, 20)
    config = AugmentationConfig(
        maximum_velocity_voxels=2.0,
        velocity_control_shape=(3, 3, 3),
        integration_steps=3,
        gamma_range=(1.0, 1.0),
        gain_range=(1.0, 1.0),
        bias_field_strength=0.0,
        noise_std=0.0,
    )
    pair = make_synthetic_pair(image, config)
    expected = warp(image, pair.ground_truth_flow, padding_mode="zeros")
    assert torch.allclose(pair.fixed[pair.fixed_domain], expected[pair.fixed_domain], atol=1e-5)


def test_native_3d_model_returns_distribution_moments():
    torch.manual_seed(3)
    model = PRACM3D(tiny_model_config()).eval()
    image = torch.rand(1, 1, 24, 24, 24)
    with torch.no_grad():
        output = model(image, image, fixed_phase="p", moving_phase="c2", retain_distributions=True)
    assert output.flow.shape == (1, 3, 24, 24, 24)
    assert output.variance.shape == output.flow.shape
    assert output.entropy.shape == (1, 1, 24, 24, 24)
    assert torch.isfinite(output.flow).all()
    assert torch.isfinite(output.variance).all()
    assert output.levels[-1].iterations[-1].distribution.probabilities is not None
    _, response_logits, structural_logits = model.auxiliary_predictions(
        output.fixed_encoded, image.shape[-3:]
    )
    assert response_logits.shape[-1] == 4
    assert structural_logits.shape[-1] == 4


def test_pair_training_graph_backpropagates_through_synthetic_correspondence():
    torch.manual_seed(7)
    augmentation = AugmentationConfig(
        maximum_velocity_voxels=2.0,
        velocity_control_shape=(3, 3, 3),
        integration_steps=2,
        gamma_range=(0.8, 1.2),
        gain_range=(0.9, 1.1),
        bias_field_strength=0.02,
        noise_std=0.0,
    )
    module = PRACMTrainingModule(tiny_model_config(), LossConfig(), augmentation)
    fixed = torch.rand(1, 1, 24, 24, 24)
    moving = torch.rand_like(fixed)
    result = module.forward_pair(
        fixed,
        moving,
        fixed_phase="p",
        moving_phase="c2",
        include_synthetic=True,
    )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_unpaired_mrct_training_uses_two_exact_within_domain_correspondences():
    torch.manual_seed(13)
    base = tiny_model_config()
    config = ModelConfig(
        **{
            **base.__dict__,
            "acquisition_identities": ("mr", "ct"),
            "response_gate_mode": "neutral",
        }
    )
    module = PRACMTrainingModule(
        config,
        LossConfig(response_gate_support=0.0, relational_moments=0.0, relational_entropy=0.0),
        AugmentationConfig(
            maximum_velocity_voxels=2.0,
            velocity_control_shape=(3, 3, 3),
            integration_steps=2,
            noise_std=0.0,
        ),
    )
    result = module(
        {
            "kind": "unpaired_domains",
            "mr": torch.rand(1, 1, 24, 24, 24),
            "ct": torch.rand(1, 1, 24, 24, 24),
            "mr_domain": torch.ones(1, 1, 24, 24, 24, dtype=torch.bool),
            "ct_domain": torch.ones(1, 1, 24, 24, 24, dtype=torch.bool),
        }
    )
    assert len(result.registrations) == 2
    assert "synthetic_candidate" in result.losses
    assert "relational_moments" not in result.losses
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_cuda_fp16_pair_training_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(5)
    module = PRACMTrainingModule(
        tiny_model_config(),
        LossConfig(),
        AugmentationConfig(
            maximum_velocity_voxels=2.0,
            velocity_control_shape=(3, 3, 3),
            integration_steps=2,
        ),
    ).cuda()
    fixed = torch.rand(1, 1, 24, 24, 24, device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = module.forward_pair(
            fixed,
            torch.rand_like(fixed),
            fixed_phase="p",
            moving_phase="c2",
            include_synthetic=True,
        )
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_cuda_fp16_triplet_relation_when_available():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(9)
    module = PRACMTrainingModule(tiny_model_config()).cuda()
    images = [torch.rand(1, 1, 24, 24, 24, device="cuda") for _ in range(3)]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        result = module.forward_triplet(*images, include_synthetic=False)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert torch.isfinite(
        torch.stack(
            [
                value
                for name, value in result.losses.items()
                if name in {"relational_moments", "relational_entropy"}
            ]
        )
    ).all()


def test_cpu_relational_triangle_runs_with_independent_plain_phase():
    torch.manual_seed(11)
    module = PRACMTrainingModule(tiny_model_config())
    images = [torch.rand(1, 1, 24, 24, 24) for _ in range(3)]
    result = module.forward_triplet(
        *images,
        phases=("p", "c1", "c2"),
        include_synthetic=False,
    )
    assert torch.isfinite(result.loss)
    assert len(result.registrations) == 3
