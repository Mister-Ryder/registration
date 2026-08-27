"""Training graph for pairwise inference and triplet-only relational supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch
import torch.nn as nn

from ..config import AugmentationConfig, LossConfig, ModelConfig
from ..losses.objective import (
    PRACMObjective,
    decoupling_losses,
    relational_distribution_loss,
    synthetic_correspondence_losses,
)
from ..model.pracm_v4 import PRACM3D, RegistrationOutput3D
from ..phase import phase_indices
from .augmentation import make_synthetic_pair


@dataclass
class TrainingResult:
    loss: torch.Tensor
    losses: Dict[str, torch.Tensor]
    registrations: Sequence[RegistrationOutput3D]


def _merge_means(items: Sequence[Mapping[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    names = set().union(*(item.keys() for item in items))
    return {
        name: torch.stack([item[name] for item in items if name in item]).mean()
        for name in names
    }


class PRACMTrainingModule(nn.Module):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        loss_config: Optional[LossConfig] = None,
        augmentation_config: Optional[AugmentationConfig] = None,
    ) -> None:
        super().__init__()
        self.model = PRACM3D(model_config)
        self.loss_config = loss_config or LossConfig()
        self.augmentation_config = augmentation_config or AugmentationConfig()
        self.objective = PRACMObjective(self.loss_config, self.model)

    def _decoupling(
        self,
        image: torch.Tensor,
        encoded,
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

    def forward_pair(
        self,
        fixed: torch.Tensor,
        moving: torch.Tensor,
        *,
        fixed_phase,
        moving_phase,
        fixed_domain: Optional[torch.Tensor] = None,
        moving_domain: Optional[torch.Tensor] = None,
        include_synthetic: bool = True,
    ) -> TrainingResult:
        output = self.model(
            fixed,
            moving,
            fixed_phase=fixed_phase,
            moving_phase=moving_phase,
            fixed_domain=fixed_domain,
            moving_domain=moving_domain,
            retain_distributions=False,
        )
        losses = self.objective.pair_registration(output, fixed, moving)
        auxiliary = _merge_means(
            (
                self._decoupling(
                    fixed, output.fixed_encoded, fixed_phase, fixed_domain
                ),
                self._decoupling(
                    moving, output.moving_encoded, moving_phase, moving_domain
                ),
            )
        )
        losses.update(auxiliary)
        registrations = [output]
        if include_synthetic:
            synthetic = make_synthetic_pair(
                fixed, self.augmentation_config, domain=fixed_domain
            )
            synthetic_output = self.model(
                synthetic.fixed,
                synthetic.moving,
                fixed_phase=fixed_phase,
                moving_phase=fixed_phase,
                fixed_domain=synthetic.fixed_domain,
                moving_domain=synthetic.moving_domain,
                retain_distributions=True,
            )
            losses.update(
                synthetic_correspondence_losses(
                    synthetic_output,
                    synthetic.ground_truth_flow,
                    synthetic.fixed_domain,
                    hard_negative_margin=self.loss_config.hard_negative_margin,
                )
            )
            registrations.append(synthetic_output)
        total = self.objective.weighted_total(losses)
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
        include_synthetic: bool = True,
    ) -> TrainingResult:
        if not (first.shape == second.shape == third.shape):
            raise ValueError("Triplet tensors must occupy one common physical grid.")
        first_to_second = self.model(
            first,
            second,
            fixed_phase=phases[0],
            moving_phase=phases[1],
            fixed_domain=domains[0],
            moving_domain=domains[1],
            retain_distributions=False,
        )
        second_to_third = self.model(
            second,
            third,
            fixed_phase=phases[1],
            moving_phase=phases[2],
            fixed_domain=domains[1],
            moving_domain=domains[2],
            retain_distributions=False,
        )
        first_to_third = self.model(
            first,
            third,
            fixed_phase=phases[0],
            moving_phase=phases[2],
            fixed_domain=domains[0],
            moving_domain=domains[2],
            retain_distributions=False,
        )
        outputs = (first_to_second, second_to_third, first_to_third)
        pair_losses = _merge_means(
            (
                self.objective.pair_registration(first_to_second, first, second),
                self.objective.pair_registration(second_to_third, second, third),
                self.objective.pair_registration(first_to_third, first, third),
            )
        )
        pair_losses.update(
            relational_distribution_loss(
                first_to_second, second_to_third, first_to_third
            )
        )
        auxiliary = _merge_means(
            (
                self._decoupling(first, first_to_second.fixed_encoded, phases[0], domains[0]),
                self._decoupling(second, first_to_second.moving_encoded, phases[1], domains[1]),
                self._decoupling(third, second_to_third.moving_encoded, phases[2], domains[2]),
            )
        )
        pair_losses.update(auxiliary)
        registrations = list(outputs)
        if include_synthetic:
            images = (first, second, third)
            index = int(torch.randint(0, 3, ()).item())
            synthetic = make_synthetic_pair(
                images[index], self.augmentation_config, domain=domains[index]
            )
            synthetic_output = self.model(
                synthetic.fixed,
                synthetic.moving,
                fixed_phase=phases[index],
                moving_phase=phases[index],
                fixed_domain=synthetic.fixed_domain,
                moving_domain=synthetic.moving_domain,
                retain_distributions=True,
            )
            pair_losses.update(
                synthetic_correspondence_losses(
                    synthetic_output,
                    synthetic.ground_truth_flow,
                    synthetic.fixed_domain,
                    hard_negative_margin=self.loss_config.hard_negative_margin,
                )
            )
            registrations.append(synthetic_output)
        total = self.objective.weighted_total(pair_losses)
        pair_losses["total"] = total
        return TrainingResult(total, pair_losses, registrations)

    def forward_unpaired_domains(
        self,
        mr: torch.Tensor,
        ct: torch.Tensor,
        *,
        mr_domain: Optional[torch.Tensor] = None,
        ct_domain: Optional[torch.Tensor] = None,
    ) -> TrainingResult:
        """Use exact within-domain geometry; never invent an MR/CT warp target."""

        all_losses = []
        auxiliary = []
        registrations = []
        for image, identity, domain in (
            (mr, "mr", mr_domain),
            (ct, "ct", ct_domain),
        ):
            synthetic = make_synthetic_pair(
                image, self.augmentation_config, domain=domain
            )
            output = self.model(
                synthetic.fixed,
                synthetic.moving,
                fixed_phase=identity,
                moving_phase=identity,
                fixed_domain=synthetic.fixed_domain,
                moving_domain=synthetic.moving_domain,
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
            all_losses.append(losses)
            auxiliary.extend(
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
            registrations.append(output)
        merged = _merge_means(all_losses)
        merged.update(_merge_means(auxiliary))
        total = self.objective.weighted_total(merged)
        merged["total"] = total
        return TrainingResult(total, merged, registrations)

    def forward(self, batch: Mapping[str, object], *, include_synthetic: bool = True) -> TrainingResult:
        kind = str(batch.get("kind", ""))
        if kind == "pair":
            return self.forward_pair(
                batch["fixed"],
                batch["moving"],
                fixed_phase=batch["fixed_phase"],
                moving_phase=batch["moving_phase"],
                fixed_domain=batch.get("fixed_domain"),
                moving_domain=batch.get("moving_domain"),
                include_synthetic=include_synthetic,
            )
        if kind == "triplet":
            return self.forward_triplet(
                batch["first"],
                batch["second"],
                batch["third"],
                phases=batch.get("phases", ("p", "c1", "c2")),
                domains=batch.get("domains", (None, None, None)),
                include_synthetic=include_synthetic,
            )
        if kind == "unpaired_domains":
            return self.forward_unpaired_domains(
                batch["mr"],
                batch["ct"],
                mr_domain=batch.get("mr_domain"),
                ct_domain=batch.get("ct_domain"),
            )
        raise ValueError(
            "Training batch kind must be 'pair', 'triplet', or 'unpaired_domains'."
        )
