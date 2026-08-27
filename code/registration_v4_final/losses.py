"""Memory-bounded anatomy-aware correspondence losses for full-resolution DSIR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ContrastiveDiagnostics:
    loss: torch.Tensor
    variance_penalty: torch.Tensor
    positive_cosine: torch.Tensor
    top1_accuracy: torch.Tensor
    sampled_locations: int


def _selected_features(
    descriptor: torch.Tensor,
    support: torch.Tensor,
    *,
    samples: int,
    generator: Optional[torch.Generator],
) -> tuple[torch.Tensor, torch.Tensor]:
    if descriptor.ndim != 5 or support.shape != (
        descriptor.shape[0],
        1,
        *descriptor.shape[-3:],
    ):
        raise ValueError("Descriptor/support must be [B,C,D,H,W] and [B,1,D,H,W].")
    if descriptor.shape[0] != 1:
        raise ValueError("The formal full-resolution trainer uses batch size one per GPU.")
    locations = torch.nonzero(support[0, 0].reshape(-1), as_tuple=False).flatten()
    if locations.numel() < 2:
        raise ValueError("Fewer than two foreground correspondence locations.")
    count = min(int(samples), int(locations.numel()))
    order = torch.randperm(locations.numel(), device=locations.device, generator=generator)[:count]
    selected = locations[order]
    features = descriptor[0].flatten(1)[:, selected].transpose(0, 1).float()
    return F.normalize(features, dim=1, eps=1e-6), selected


def _directional_chunked_ce(
    query: torch.Tensor,
    keys: torch.Tensor,
    *,
    temperature: float,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = query.shape[0]
    total_loss = query.new_zeros(())
    correct = query.new_zeros(())
    for start in range(0, count, int(chunk_size)):
        end = min(start + int(chunk_size), count)
        logits = query[start:end] @ keys.transpose(0, 1) / float(temperature)
        labels = torch.arange(start, end, device=query.device)
        total_loss = total_loss + F.cross_entropy(logits, labels, reduction="sum")
        correct = correct + (logits.argmax(dim=1) == labels).float().sum()
    return total_loss / count, correct / count


def anatomy_correspondence_loss(
    first_descriptor: torch.Tensor,
    second_descriptor: torch.Tensor,
    support: torch.Tensor,
    *,
    samples: int,
    temperature: float,
    chunk_size: int,
    variance_floor: float,
    generator: Optional[torch.Generator] = None,
) -> ContrastiveDiagnostics:
    """Symmetric same-anatomy InfoNCE with explicit anti-collapse control."""

    if first_descriptor.shape != second_descriptor.shape:
        raise ValueError("Descriptor views must have identical shape.")
    first, selected = _selected_features(
        first_descriptor, support, samples=samples, generator=generator
    )
    second = F.normalize(
        second_descriptor[0].flatten(1)[:, selected].transpose(0, 1).float(),
        dim=1,
        eps=1e-6,
    )
    forward, forward_top1 = _directional_chunked_ce(
        first, second, temperature=temperature, chunk_size=chunk_size
    )
    backward, backward_top1 = _directional_chunked_ce(
        second, first, temperature=temperature, chunk_size=chunk_size
    )
    first_std = first.std(dim=0, unbiased=False)
    second_std = second.std(dim=0, unbiased=False)
    variance = 0.5 * (
        F.relu(float(variance_floor) - first_std).mean()
        + F.relu(float(variance_floor) - second_std).mean()
    )
    return ContrastiveDiagnostics(
        loss=0.5 * (forward + backward),
        variance_penalty=variance,
        positive_cosine=(first * second).sum(dim=1).mean(),
        top1_accuracy=0.5 * (forward_top1 + backward_top1),
        sampled_locations=int(first.shape[0]),
    )


__all__ = ["ContrastiveDiagnostics", "anatomy_correspondence_loss"]

