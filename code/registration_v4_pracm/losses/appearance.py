"""Dense appearance-invariance supervision with exact zero-displacement positives."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F

from ..model.encoder import EncodedPyramid


def _mask_at(mask: Optional[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(
            (reference.shape[0], 1, *reference.shape[-3:]),
            device=reference.device,
            dtype=torch.bool,
        )
    return F.interpolate(mask.float(), size=reference.shape[-3:], mode="nearest") > 0.5


def _sample_positions(
    first: torch.Tensor,
    second: torch.Tensor,
    support: torch.Tensor,
    samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    first_flat = first.flatten(2)
    second_flat = second.flatten(2)
    support_flat = support[:, 0].flatten(1)
    selected_first = []
    selected_second = []
    for batch_index in range(first.shape[0]):
        valid = support_flat[batch_index].nonzero(as_tuple=False).flatten()
        if valid.numel() < 2:
            raise ValueError("Appearance contrastive supervision has fewer than two valid positions.")
        count = min(int(samples), int(valid.numel()))
        if valid.numel() > count:
            order = torch.randperm(valid.numel(), device=valid.device)[:count]
            valid = valid[order]
        selected_first.append(first_flat[batch_index, :, valid].transpose(0, 1))
        selected_second.append(second_flat[batch_index, :, valid].transpose(0, 1))
    # Formal training uses batch=1.  Enforce a common count for defensive reuse.
    count = min(value.shape[0] for value in selected_first)
    return (
        torch.stack([value[:count] for value in selected_first]),
        torch.stack([value[:count] for value in selected_second]),
    )


def appearance_invariance_losses(
    first: EncodedPyramid,
    second: EncodedPyramid,
    *,
    domain: Optional[torch.Tensor],
    levels: Sequence[int],
    temperature: float,
    samples: int,
    variance_floor: float,
) -> Dict[str, torch.Tensor]:
    """Same-location positives, spatial negatives, and an explicit anti-collapse term."""

    contrastive_terms = []
    variance_terms = []
    positive_terms = []
    negative_terms = []
    for level in levels:
        first_feature = first.structural[level]
        second_feature = second.structural[level]
        support = _mask_at(domain, first_feature)
        first_selected, second_selected = _sample_positions(
            first_feature, second_feature, support, samples
        )
        first_selected = F.normalize(first_selected.float(), dim=-1, eps=1e-6)
        second_selected = F.normalize(second_selected.float(), dim=-1, eps=1e-6)
        logits = torch.matmul(first_selected, second_selected.transpose(1, 2)) / temperature
        target = torch.arange(logits.shape[1], device=logits.device)[None].expand(logits.shape[0], -1)
        contrastive_terms.append(
            0.5
            * (
                F.cross_entropy(logits, target)
                + F.cross_entropy(logits.transpose(1, 2), target)
            )
        )
        positive = logits.diagonal(dim1=1, dim2=2) * temperature
        off_diagonal = ~torch.eye(
            logits.shape[1], device=logits.device, dtype=torch.bool
        )[None]
        negative = (logits * temperature)[off_diagonal.expand_as(logits)]
        positive_terms.append(positive.mean())
        negative_terms.append(negative.mean())
        first_std = first_selected.std(dim=1, unbiased=False)
        second_std = second_selected.std(dim=1, unbiased=False)
        variance_terms.append(
            0.5
            * (
                F.relu(variance_floor - first_std).mean()
                + F.relu(variance_floor - second_std).mean()
            )
        )
    return {
        "appearance_invariance": torch.stack(contrastive_terms).mean(),
        "appearance_variance": torch.stack(variance_terms).mean(),
        "diagnostic_appearance_positive": torch.stack(positive_terms).mean().detach(),
        "diagnostic_appearance_negative": torch.stack(negative_terms).mean().detach(),
    }

