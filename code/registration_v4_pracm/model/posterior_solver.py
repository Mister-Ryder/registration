"""Unrolled uncertainty-conditioned spatial posterior solver for PRA-CM V4-Full."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class PosteriorSolverOutput:
    flow: torch.Tensor
    variance: torch.Tensor
    data_flow: torch.Tensor
    correction_magnitude: torch.Tensor


class UncertaintyConditionedPosteriorSolver3D:
    """MAP-like fixed-point refinement over top-k hypotheses and spatial consensus.

    The solver has no free deformation-regression head.  Its data term is the
    explicit candidate posterior; posterior entropy only controls how strongly
    neighbouring voxels contribute to resolving ambiguity.
    """

    def __init__(
        self,
        *,
        iterations: int,
        data_sigma: float,
        spatial_weight: float,
    ) -> None:
        self.iterations = int(iterations)
        self.data_sigma = float(data_sigma)
        self.spatial_weight = float(spatial_weight)

    @staticmethod
    def _smooth(value: torch.Tensor) -> torch.Tensor:
        return F.avg_pool3d(
            F.pad(value, (1, 1, 1, 1, 1, 1), mode="replicate"),
            kernel_size=3,
            stride=1,
        )

    def __call__(
        self,
        base_flow: torch.Tensor,
        residual_hypotheses: torch.Tensor,
        probabilities: torch.Tensor,
        entropy: torch.Tensor,
    ) -> PosteriorSolverOutput:
        if residual_hypotheses.ndim != 6 or residual_hypotheses.shape[2] != 3:
            raise ValueError("Posterior hypotheses must be [B,K,3,D,H,W].")
        if probabilities.shape != (
            residual_hypotheses.shape[0],
            residual_hypotheses.shape[1],
            *residual_hypotheses.shape[-3:],
        ):
            raise ValueError("Posterior top-k probabilities do not match hypotheses.")
        hypotheses = base_flow[:, None].float() + residual_hypotheses.float()
        probability = probabilities.float()
        probability = probability / probability.sum(dim=1, keepdim=True).clamp_min(1e-8)
        estimate = hypotheses[:, 0]
        sigma2 = self.data_sigma ** 2
        ambiguity = entropy.float().clamp(0, 1)
        data_flow = estimate
        for _ in range(self.iterations):
            distance2 = (hypotheses - estimate[:, None]).square().sum(dim=2)
            responsibilities = torch.softmax(
                probability.clamp_min(1e-8).log() - 0.5 * distance2 / sigma2,
                dim=1,
            )
            data_flow = (responsibilities[:, :, None] * hypotheses).sum(dim=1)
            spatial = self._smooth(estimate)
            spatial_strength = self.spatial_weight * ambiguity
            estimate = (
                data_flow + spatial_strength * spatial
            ) / (1 + spatial_strength)
        centred = hypotheses - estimate[:, None]
        variance = (
            responsibilities[:, :, None] * centred.square()
        ).sum(dim=1)
        correction = (estimate - data_flow).square().sum(dim=1, keepdim=True).sqrt()
        return PosteriorSolverOutput(
            estimate.to(base_flow.dtype),
            variance.to(base_flow.dtype),
            data_flow.to(base_flow.dtype),
            correction.to(base_flow.dtype),
        )

