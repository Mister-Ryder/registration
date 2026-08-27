"""Staged V4 training graph: appearance invariance and geometric equivariance are separate tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..config import AugmentationConfig, LossConfig, ModelConfig, TrainingConfig
from ..losses.appearance import appearance_invariance_losses
from ..losses.objective import (
    PRACMObjective,
    decoupling_losses,
    relational_distribution_loss,
    synthetic_correspondence_losses,
)
from ..model.encoder import EncodedPyramid
from ..model.pracm_v4 import PRACM3D, RegistrationOutput3D
from ..phase import phase_indices
from .augmentation import make_appearance_views, make_synthetic_pair


@dataclass
class TrainingResult:
    loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    registrations: Sequence[RegistrationOutput3D]


def _merge_means(items: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    if not items:
        return {}
    names = set().union(*(item.keys() for item in items))
    return {
        name: torch.stack([item[name] for item in items if name in item]).mean()
        for name in names
    }


def _split_encoded(encoded: EncodedPyramid, batch: int) -> tuple[EncodedPyramid, EncodedPyramid]:
    return (
        EncodedPyramid(
            tuple(value[:batch] for value in encoded.backbone),
            tuple(value[:batch] for value in encoded.structural),
            tuple(value[:batch] for value in encoded.response),
        ),
        EncodedPyramid(
            tuple(value[batch:] for value in encoded.backbone),
            tuple(value[batch:] for value in encoded.structural),
            tuple(value[batch:] for value in encoded.response),
        ),
    )


class PRACMTrainingModule(nn.Module):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        loss_config: Optional[LossConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        super().__init__()
        self.model = PRACM3D(model_config)
        self.loss_config = loss_config or LossConfig()
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.training_config = training_config or TrainingConfig()
        self.objective = PRACMObjective(self.loss_config, self.model)
        self.current_epoch = 0
        self.current_step = 0
        self.validation_mode = False

    def set_training_state(self, epoch: int, global_step: int) -> None:
        self.current_epoch = int(epoch)
        self.current_step = int(global_step)

    def set_validation_mode(self, enabled: bool) -> None:
        self.validation_mode = bool(enabled)

    def _task_mode(self) -> str:
        if self.validation_mode:
            return "both"
        representation_end = self.training_config.representation_stage_epochs
        if self.current_epoch < representation_end:
            return "appearance"
        ramp_end = representation_end + self.training_config.candidate_ramp_epochs
        if self.current_epoch < ramp_end:
            return "appearance" if self.current_step % 2 == 0 else "geometry"
        return "appearance" if self.current_step % 4 == 0 else "geometry"

    def _curriculum_factors(self) -> Dict[str, float]:
        representation_end = self.training_config.representation_stage_epochs
        candidate_progress = (
            self.current_epoch - representation_end + 1
        ) / float(self.training_config.candidate_ramp_epochs)
        candidate = min(1.0, max(0.0, candidate_progress))
        deformation_start = representation_end + self.training_config.candidate_ramp_epochs // 2
        deformation_progress = (
            self.current_epoch - deformation_start + 1
        ) / float(self.training_config.deformation_ramp_epochs)
        deformation = min(1.0, max(0.0, deformation_progress))
        if self.validation_mode:
            candidate = deformation = 1.0
        return {
            "representation": 1.0,
            "candidate": candidate,
            "deformation": deformation,
        }

    @staticmethod
    def _loss_group(name: str) -> str:
        if name in {"appearance_invariance", "appearance_variance"}:
            return "representation"
        if name in {
            "synthetic_candidate",
            "synthetic_contrastive",
            "uncertainty_calibration",
        }:
            return "candidate"
        return "deformation"

    def _totals(self, losses: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        reference = next(iter(losses.values()))
        total = reference * 0
        selection_total = reference * 0
        factors = self._curriculum_factors()
        fields = self.loss_config.__dataclass_fields__
        for name, value in losses.items():
            if name not in fields:
                continue
            weighted = float(getattr(self.loss_config, name)) * value
            selection_total = selection_total + weighted
            total = total + factors[self._loss_group(name)] * weighted
        return total, selection_total

    def _appearance(
        self,
        image: torch.Tensor,
        domain: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        views = make_appearance_views(image, self.augmentation_config, domain)
        joint = self.model.encode(torch.cat((views.first, views.second), dim=0))
        first, second = _split_encoded(joint, image.shape[0])
        return appearance_invariance_losses(
            first,
            second,
            domain=views.domain,
            levels=self.model.config.correlation_levels,
            temperature=self.model.config.appearance_temperature,
            samples=self.model.config.appearance_samples,
            variance_floor=self.model.config.appearance_variance_floor,
        )

    def _decoupling(
        self,
        image: torch.Tensor,
        encoded: EncodedPyramid,
        phase,
        domain: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ids = phase_indices(
            phase,
            batch=image.shape[0],
            device=image.device,
            identities=self.model.config.acquisition_identities,
        )
        return decoupling_losses(self.model, image, encoded, ids, domain)

    def _geometry(
        self,
        image: torch.Tensor,
        identity,
        domain: Optional[torch.Tensor],
        spacing_dzyx: Optional[torch.Tensor],
    ) -> tuple[Dict[str, torch.Tensor], RegistrationOutput3D]:
        synthetic = make_synthetic_pair(image, self.augmentation_config, domain=domain)
        output = self.model(
            synthetic.fixed,
            synthetic.moving,
            fixed_phase=identity,
            moving_phase=identity,
            fixed_domain=synthetic.fixed_domain,
            moving_domain=synthetic.moving_domain,
            spacing_dzyx=spacing_dzyx,
            retain_distributions=True,
        )
        losses = self.objective.pair_registration(
            output, synthetic.fixed, synthetic.moving
        )
        losses.update(
            synthetic_correspondence_losses(
                output,
                synthetic.ground_truth_flow,
                synthetic.fixed_domain,
                hard_negative_margin=self.loss_config.hard_negative_margin,
            )
        )
        if any(
            getattr(self.loss_config, name) > 0
            for name in (
                "response_reconstruction",
                "response_phase",
                "structural_phase_adversarial",
                "structural_response_orthogonality",
            )
        ):
            losses.update(
                _merge_means(
                    (
                        self._decoupling(
                            synthetic.fixed,
                            output.fixed_encoded,
                            identity,
                            synthetic.fixed_domain,
                        ),
                        self._decoupling(
                            synthetic.moving,
                            output.moving_encoded,
                            identity,
                            synthetic.moving_domain,
                        ),
                    )
                )
            )
        return losses, output

    def forward_unpaired_domains(
        self,
        mr: torch.Tensor,
        ct: torch.Tensor,
        *,
        mr_domain: Optional[torch.Tensor] = None,
        ct_domain: Optional[torch.Tensor] = None,
        mr_spacing_dzyx: Optional[torch.Tensor] = None,
        ct_spacing_dzyx: Optional[torch.Tensor] = None,
    ) -> TrainingResult:
        """Never assign a false MR/CT flow target; both tasks use exact within-image anatomy."""

        mode = self._task_mode()
        losses = []
        registrations = []
        domains = (
            (mr, "mr", mr_domain, mr_spacing_dzyx),
            (ct, "ct", ct_domain, ct_spacing_dzyx),
        )
        if mode in {"appearance", "both"}:
            losses.extend(self._appearance(image, domain) for image, _, domain, _ in domains)
        if mode in {"geometry", "both"}:
            for image, identity, domain, spacing in domains:
                current, output = self._geometry(image, identity, domain, spacing)
                losses.append(current)
                registrations.append(output)
        merged = _merge_means(losses)
        total, selection_total = self._totals(merged)
        merged["curriculum_candidate_factor"] = total.new_tensor(
            self._curriculum_factors()["candidate"]
        )
        merged["curriculum_deformation_factor"] = total.new_tensor(
            self._curriculum_factors()["deformation"]
        )
        merged["selection_total"] = selection_total
        merged["total"] = total
        return TrainingResult(total, merged, registrations)

    def forward_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_phase,
        moving_phase,
        fixed_domain: Optional[torch.Tensor] = None,
        moving_domain: Optional[torch.Tensor] = None,
        spacing_dzyx: Optional[torch.Tensor] = None,
        include_synthetic: bool = True,
    ) -> TrainingResult:
        output = self.model(
            fixed,
            moving,
            fixed_phase=fixed_phase,
            moving_phase=moving_phase,
            fixed_domain=fixed_domain,
            moving_domain=moving_domain,
            spacing_dzyx=spacing_dzyx,
            retain_distributions=False,
        )
        losses = self.objective.pair_registration(output, fixed, moving)
        registrations = [output]
        if self._task_mode() in {"appearance", "both"}:
            losses.update(self._appearance(fixed, fixed_domain))
        if include_synthetic and self._task_mode() in {"geometry", "both"}:
            synthetic_losses, synthetic_output = self._geometry(
                fixed, fixed_phase, fixed_domain, spacing_dzyx
            )
            losses.update(synthetic_losses)
            registrations.append(synthetic_output)
        total, selection_total = self._totals(losses)
        losses["selection_total"] = selection_total
        losses["total"] = total
        return TrainingResult(total, losses, registrations)

    def forward_triplet(
        self,
        first: torch.Tensor,
        second: torch.Tensor,
        third: torch.Tensor,
        *,
        phases=("p", "c1", "c2"),
        domains=(None, None, None),
        spacing_dzyx: Optional[torch.Tensor] = None,
        include_synthetic: bool = True,
    ) -> TrainingResult:
        outputs = [
            self.model(
                fixed,
                moving,
                fixed_phase=phases[i],
                moving_phase=phases[j],
                fixed_domain=domains[i],
                moving_domain=domains[j],
                spacing_dzyx=spacing_dzyx,
                retain_distributions=False,
            )
            for fixed, moving, i, j in (
                (first, second, 0, 1),
                (second, third, 1, 2),
                (first, third, 0, 2),
            )
        ]
        losses = _merge_means(
            (
                self.objective.pair_registration(outputs[0], first, second),
                self.objective.pair_registration(outputs[1], second, third),
                self.objective.pair_registration(outputs[2], first, third),
            )
        )
        losses.update(relational_distribution_loss(*outputs))
        if self._task_mode() in {"appearance", "both"}:
            losses.update(self._appearance(first, domains[0]))
        if include_synthetic and self._task_mode() in {"geometry", "both"}:
            synthetic_losses, synthetic_output = self._geometry(
                first, phases[0], domains[0], spacing_dzyx
            )
            losses.update(synthetic_losses)
            outputs.append(synthetic_output)
        total, selection_total = self._totals(losses)
        losses["selection_total"] = selection_total
        losses["total"] = total
        return TrainingResult(total, losses, outputs)

    def forward(self, batch: Mapping[str, object], *, include_synthetic: bool = True) -> TrainingResult:
        kind = str(batch.get("kind", ""))
        if kind == "unpaired_domains":
            return self.forward_unpaired_domains(
                batch["mr"],
                batch["ct"],
                mr_domain=batch.get("mr_domain"),
                ct_domain=batch.get("ct_domain"),
                mr_spacing_dzyx=batch.get("mr_spacing_dzyx"),
                ct_spacing_dzyx=batch.get("ct_spacing_dzyx"),
            )
        if kind == "pair":
            return self.forward_pair(
                batch["fixed"],
                batch["moving"],
                fixed_phase=batch["fixed_phase"],
                moving_phase=batch["moving_phase"],
                fixed_domain=batch.get("fixed_domain"),
                moving_domain=batch.get("moving_domain"),
                spacing_dzyx=batch.get("spacing_dzyx"),
                include_synthetic=include_synthetic,
            )
        if kind == "triplet":
            return self.forward_triplet(
                batch["first"],
                batch["second"],
                batch["third"],
                phases=batch.get("phases", ("p", "c1", "c2")),
                domains=batch.get("domains", (None, None, None)),
                spacing_dzyx=batch.get("spacing_dzyx"),
                include_synthetic=include_synthetic,
            )
        raise ValueError("Training batch kind must be pair, triplet, or unpaired_domains.")

