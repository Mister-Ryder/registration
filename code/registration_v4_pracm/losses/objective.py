"""Losses tied directly to PRA-CM correspondence and response mechanisms."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from ..config import LossConfig
from ..model.pracm_v4 import PRACM3D, RegistrationOutput3D
from ..ops.spatial import compose_flows, jacobian_determinant, resize_flow, warp


def masked_mean(value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return value.mean()
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(1)
    if weight.shape != value.shape:
        weight = weight.expand_as(value)
    return (value * weight).sum() / weight.sum().clamp_min(1)


def local_ncc_loss(
    fixed: torch.Tensor,
    warped: torch.Tensor,
    mask: torch.Tensor,
    *,
    window: int = 7,
) -> torch.Tensor:
    padding = window // 2

    def mean(value):
        return F.avg_pool3d(
            F.pad(value, (padding,) * 6, mode="replicate"), window, stride=1
        )

    weight = mask.to(fixed.dtype)
    local_weight = mean(weight).clamp_min(1e-6)
    fixed_mean = mean(fixed * weight) / local_weight
    moving_mean = mean(warped * weight) / local_weight
    fixed_zero = fixed - fixed_mean
    moving_zero = warped - moving_mean
    covariance = mean(weight * fixed_zero * moving_zero) / local_weight
    fixed_var = mean(weight * fixed_zero.square()) / local_weight
    moving_var = mean(weight * moving_zero.square()) / local_weight
    ncc = covariance / (fixed_var * moving_var + 1e-5).sqrt()
    evidence = weight * (local_weight > 1e-3).to(weight.dtype)
    return masked_mean(1 - ncc.clamp(-1, 1), evidence)


def _first_derivatives(flow: torch.Tensor):
    return (
        flow[:, :, 1:] - flow[:, :, :-1],
        flow[:, :, :, 1:] - flow[:, :, :, :-1],
        flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1],
    )


def smoothness_losses(flow: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    first = _first_derivatives(flow)
    first_loss = sum(value.abs().mean() for value in first) / 3
    second_parts = []
    for value, dim in zip(first, (2, 3, 4)):
        if value.shape[dim] > 1:
            second_parts.append(torch.diff(value, dim=dim).abs().mean())
    second_loss = torch.stack(second_parts).mean() if second_parts else flow.sum() * 0
    return first_loss, second_loss


def _resize_mask(mask: Optional[torch.Tensor], size, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones((reference.shape[0], 1, *size), device=reference.device, dtype=torch.bool)
    return F.interpolate(mask.float(), size=size, mode="nearest") > 0.5


def structural_alignment_loss(
    output: RegistrationOutput3D,
    *,
    level: int,
) -> torch.Tensor:
    fixed = output.fixed_encoded.structural[level]
    moving = output.moving_encoded.structural[level]
    flow = resize_flow(output.flow, fixed.shape[-3:])
    warped, valid = warp(moving, flow, return_valid=True)
    similarity = (fixed * F.normalize(warped, dim=1, eps=1e-6)).sum(dim=1, keepdim=True)
    confidence = F.interpolate(
        output.confidence.detach(), size=fixed.shape[-3:], mode="trilinear", align_corners=True
    )
    endpoint = F.interpolate(
        output.endpoint_valid.float(), size=fixed.shape[-3:], mode="nearest"
    ) > 0.5
    weight = (
        (0.25 + 0.75 * confidence)
        * valid.to(confidence.dtype)
        * endpoint.to(confidence.dtype)
    )
    return (weight * (1 - similarity)).sum() / weight.sum().clamp_min(1)


def _candidate_corner_loss(
    probabilities: torch.Tensor,
    offsets: torch.Tensor,
    residual: torch.Tensor,
    mask: torch.Tensor,
    candidate_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Trilinear soft-label NLL for a continuous residual displacement."""

    radius = int(offsets.abs().max().item())
    side = 2 * radius + 1
    lower = torch.floor(residual.detach())
    fraction = residual.detach() - lower
    total = probabilities.new_zeros(probabilities.shape[0], 1, *probabilities.shape[-3:])
    support = torch.zeros_like(total)
    for bit_z in (0, 1):
        for bit_y in (0, 1):
            for bit_x in (0, 1):
                bits = residual.new_tensor([bit_z, bit_y, bit_x]).view(1, 3, 1, 1, 1)
                corner = lower + bits
                weight_axis = torch.where(bits.bool(), fraction, 1 - fraction)
                weight = weight_axis.prod(dim=1, keepdim=True)
                inside = (corner.abs() <= radius).all(dim=1, keepdim=True)
                index = (
                    (corner[:, 0] + radius) * side * side
                    + (corner[:, 1] + radius) * side
                    + (corner[:, 2] + radius)
                ).long().clamp(0, probabilities.shape[1] - 1)
                selected = torch.gather(probabilities, 1, index.unsqueeze(1)).clamp_min(1e-8)
                selected_valid = (
                    torch.ones_like(inside)
                    if candidate_valid is None
                    else torch.gather(candidate_valid, 1, index.unsqueeze(1))
                )
                effective = weight * inside * selected_valid.to(weight.dtype)
                total = total - effective * selected.log()
                support = support + effective
    effective_mask = mask & (support > 1e-6)
    return masked_mean(total / support.clamp_min(1e-6), effective_mask)



def _masked_correlation(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected_first = first[mask]
    selected_second = second[mask]
    if selected_first.numel() < 2:
        return first.sum() * 0
    selected_first = selected_first.float() - selected_first.float().mean()
    selected_second = selected_second.float() - selected_second.float().mean()
    denominator = (
        selected_first.square().mean() * selected_second.square().mean()
    ).sqrt().clamp_min(1e-8)
    return (selected_first * selected_second).mean() / denominator

def synthetic_correspondence_losses(
    output: RegistrationOutput3D,
    ground_truth_flow: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    hard_negative_margin: float,
) -> Dict[str, torch.Tensor]:
    candidate_terms = []
    hard_negative_terms = []
    calibration_terms = []
    top1_terms = []
    top5_terms = []
    support_terms = []
    expected_error_terms = []
    entropy_error_terms = []
    for level_output in output.levels:
        gt = resize_flow(ground_truth_flow, level_output.flow.shape[-3:])
        mask = _resize_mask(valid_mask, gt.shape[-3:], gt)
        for iteration in level_output.iterations:
            distribution = iteration.distribution
            if (
                distribution.probabilities is None
                or distribution.logits is None
                or distribution.valid is None
            ):
                raise RuntimeError("Synthetic training requires retained correlation volumes.")
            residual = gt - iteration.base_flow.detach()
            candidate_terms.append(
                _candidate_corner_loss(
                    distribution.probabilities,
                    distribution.offsets_dzyx,
                    residual,
                    mask,
                    distribution.valid,
                )
            )
            offsets = distribution.offsets_dzyx.view(1, -1, 3, 1, 1, 1)
            distance = (offsets - residual.detach()[:, None]).square().sum(dim=2).sqrt()
            nearest = distance.argmin(dim=1, keepdim=True)
            positive = torch.gather(distribution.logits, 1, nearest)
            target_valid = torch.gather(distribution.valid, 1, nearest)
            target_in_search = (
                residual.detach().abs()
                <= float(distribution.offsets_dzyx.abs().max().item())
            ).all(dim=1, keepdim=True)

            diagnostic_mask = mask & target_in_search
            top1 = distribution.probabilities.argmax(dim=1, keepdim=True) == nearest
            topk = distribution.probabilities.topk(
                min(5, distribution.probabilities.shape[1]), dim=1
            ).indices
            top5 = (topk == nearest).any(dim=1, keepdim=True)
            top1_terms.append(
                masked_mean(top1.float(), diagnostic_mask & target_valid)
            )
            top5_terms.append(
                masked_mean(top5.float(), diagnostic_mask & target_valid)
            )
            support_terms.append(
                masked_mean(target_valid.float(), diagnostic_mask)
            )
            expected = iteration.base_flow + distribution.mode_residual
            expected_error = (expected - gt).square().sum(dim=1, keepdim=True).sqrt()
            expected_error_terms.append(
                masked_mean(expected_error, diagnostic_mask & target_valid)
            )
            entropy_error_terms.append(
                _masked_correlation(
                    distribution.entropy,
                    expected_error,
                    diagnostic_mask & target_valid,
                )
            )

            negative_valid = distribution.valid & (distance > 1.5)
            negative_logits = distribution.logits.masked_fill(~negative_valid, -1e4)
            hard = negative_logits.max(dim=1, keepdim=True).values
            has_negative = negative_valid.any(dim=1, keepdim=True)
            positive = torch.where(target_valid, positive, torch.zeros_like(positive))
            hard = torch.where(has_negative, hard, torch.zeros_like(hard))
            hard_negative_terms.append(
                masked_mean(
                    F.relu(hard_negative_margin - positive + hard),
                    mask & target_valid & target_in_search & has_negative,
                )
            )
            predicted = iteration.base_flow + distribution.mode_residual
            squared_error = (predicted - gt).square()
            variance = distribution.variance_residual + 0.25
            calibration = 0.5 * (squared_error / variance + variance.log())
            calibration_terms.append(
                masked_mean(calibration, mask.expand_as(calibration))
            )

    flow_error = F.smooth_l1_loss(output.flow, ground_truth_flow, reduction="none")
    flow_loss = masked_mean(flow_error, valid_mask.expand_as(flow_error))
    level = output.levels[-1].level
    fixed = output.fixed_encoded.structural[level]
    moving = output.moving_encoded.structural[level]
    gt_level = resize_flow(ground_truth_flow, fixed.shape[-3:])
    positive = warp(moving, gt_level)
    descriptor_mask = _resize_mask(valid_mask, fixed.shape[-3:], fixed)
    contrastive = masked_mean(
        1 - (fixed * F.normalize(positive, dim=1, eps=1e-6)).sum(dim=1, keepdim=True),
        descriptor_mask,
    )
    zero = output.flow.sum() * 0
    return {
        "synthetic_flow": flow_loss,
        "synthetic_candidate": torch.stack(candidate_terms).mean() if candidate_terms else zero,
        "synthetic_contrastive": contrastive
        + (torch.stack(hard_negative_terms).mean() if hard_negative_terms else zero),
        "uncertainty_calibration": (
            torch.stack(calibration_terms).mean() if calibration_terms else zero
        ),
        "posterior_solver_consistency": flow_loss,
        "diagnostic_candidate_top1": torch.stack(top1_terms).mean().detach(),
        "diagnostic_candidate_top5": torch.stack(top5_terms).mean().detach(),
        "diagnostic_candidate_support_recall": torch.stack(support_terms).mean().detach(),
        "diagnostic_candidate_expected_epe": torch.stack(expected_error_terms).mean().detach(),
        "diagnostic_entropy_error_correlation": torch.stack(
            entropy_error_terms
        ).mean().detach(),
    }


def decoupling_losses(
    model: PRACM3D,
    image: torch.Tensor,
    encoded,
    phase_id: torch.Tensor,
    domain: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    reconstruction, response_logits, structural_logits = model.auxiliary_predictions(
        encoded, image.shape[-3:], domain=domain
    )
    mask = torch.ones_like(image, dtype=torch.bool) if domain is None else domain.bool()
    reconstruction_loss = masked_mean((reconstruction - image).abs(), mask)
    response_phase = F.cross_entropy(response_logits, phase_id)
    structural_phase = F.cross_entropy(structural_logits, phase_id)
    level = model.finest_auxiliary_level
    structural = encoded.structural[level]
    response = encoded.response[level]
    level_mask = _resize_mask(domain, structural.shape[-3:], structural).to(structural.dtype)
    count = level_mask.sum(dim=(-3, -2, -1), keepdim=True).clamp_min(1)
    structural_mean = (structural * level_mask).sum(
        dim=(-3, -2, -1), keepdim=True
    ) / count
    response_mean = (response * level_mask).sum(
        dim=(-3, -2, -1), keepdim=True
    ) / count
    structural = (structural - structural_mean) * level_mask
    response = (response - response_mean) * level_mask
    cross_covariance = torch.einsum(
        "bcdhw,brdhw->bcr", structural, response
    ) / count.view(count.shape[0], 1, 1)
    orthogonality = cross_covariance.square().mean()
    return {
        "response_reconstruction": reconstruction_loss,
        "response_phase": response_phase,
        "structural_phase_adversarial": structural_phase,
        "structural_response_orthogonality": orthogonality,
    }


def relational_distribution_loss(
    first_to_second: RegistrationOutput3D,
    second_to_third: RegistrationOutput3D,
    first_to_third: RegistrationOutput3D,
) -> Dict[str, torch.Tensor]:
    first_flow = first_to_second.flow.float()
    second_flow = second_to_third.flow.float()
    direct_mean = first_to_third.flow.float()
    first_variance = first_to_second.variance.float()
    second_variance = second_to_third.variance.float()
    direct_variance = first_to_third.variance.float()
    composed_mean, composition_valid = compose_flows(
        first_flow,
        second_flow,
        return_valid=True,
    )
    # First-order uncertainty propagation through
    # x -> x + f12(x) -> x + f12(x) + f23(x+f12(x)).  A plain variance sum
    # ignores the local Jacobian of the second mapping and is incorrect near
    # expansion/contraction.  We retain a diagonal covariance but propagate it
    # with J^2 before adding the sampled second-edge variance.
    gradients = torch.gradient(second_flow, dim=(2, 3, 4))
    jacobian = torch.stack(gradients, dim=2)
    identity = torch.eye(3, device=jacobian.device, dtype=jacobian.dtype).view(
        1, 3, 3, 1, 1, 1
    )
    mapping_jacobian = jacobian + identity
    sampled_jacobian = warp(
        mapping_jacobian.flatten(1, 2),
        first_flow,
        padding_mode="zeros",
    ).view_as(mapping_jacobian)
    propagated_first_variance = (
        sampled_jacobian.square() * first_variance[:, None]
    ).sum(dim=2)
    composed_variance = propagated_first_variance + warp(
        second_variance,
        first_flow,
        padding_mode="zeros",
    )
    valid = (
        composition_valid
        & first_to_second.endpoint_valid
        & first_to_third.endpoint_valid
    )
    valid &= warp(
        second_to_third.endpoint_valid.float(),
        first_flow,
        mode="nearest",
        padding_mode="zeros",
    ) > 0.5
    vp = direct_variance + 0.25
    vq = composed_variance + 0.25
    difference = direct_mean - composed_mean
    kl_pq = 0.5 * ((vq / vp).log() + (vp + difference.square()) / vq - 1)
    kl_qp = 0.5 * ((vp / vq).log() + (vq + difference.square()) / vp - 1)
    moments = masked_mean(0.5 * (kl_pq + kl_qp), valid.expand_as(difference))
    second_entropy = warp(
        second_to_third.entropy.float(),
        first_flow,
        padding_mode="zeros",
    )
    composed_entropy = 1 - (1 - first_to_second.entropy.float()) * (1 - second_entropy)
    entropy = masked_mean(
        (first_to_third.entropy.float() - composed_entropy).abs(), valid
    )
    return {"relational_moments": moments, "relational_entropy": entropy}


class PRACMObjective:
    def __init__(self, config: LossConfig, model: PRACM3D) -> None:
        self.config = config
        self.model = model

    def pair_registration(
        self,
        output: RegistrationOutput3D,
        fixed: torch.Tensor,
        moving: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        structural = structural_alignment_loss(
            output, level=self.model.finest_auxiliary_level
        )
        warped = warp(moving, output.flow)
        response_supported_domain = (
            output.endpoint_valid.to(output.expected_gate.dtype)
            * output.expected_gate.detach()
        )
        photometric = local_ncc_loss(fixed, warped, response_supported_domain)
        smooth_first, smooth_second = smoothness_losses(output.flow)
        jacobian = masked_mean(
            F.relu(
                self.config.jacobian_margin
                - jacobian_determinant(output.flow.float())
            ),
            output.endpoint_valid,
        )
        gate_support = masked_mean(
            F.relu(self.config.minimum_gate_support - output.expected_gate),
            output.endpoint_valid,
        )
        return {
            "structural": structural,
            "photometric": photometric,
            "smooth_first": smooth_first,
            "smooth_second": smooth_second,
            "jacobian": jacobian,
            "response_gate_support": gate_support,
        }

    def weighted_total(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = next(iter(losses.values())) * 0
        for name, value in losses.items():
            if name in {field for field in self.config.__dataclass_fields__}:
                total = total + float(getattr(self.config, name)) * value
        return total
