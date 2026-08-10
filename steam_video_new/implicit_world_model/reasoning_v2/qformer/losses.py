"""Minimal losses for separately trained QF1 and QF2."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import SLOT_NAMES


def masked_slot_distillation_loss(
    predicted: Tensor,
    target: Tensor,
    validity: Tensor,
) -> Tensor:
    """Cosine reconstruction over valid feature slots only."""

    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target slots must share shape [B, 4, D]")
    if validity.shape != predicted.shape[:2] or validity.dtype is not torch.bool:
        raise ValueError("slot validity must be boolean with shape [B, 4]")
    if not validity.any():
        raise ValueError("slot distillation requires at least one valid target")
    per_slot = 1.0 - F.cosine_similarity(predicted, target.detach(), dim=-1)
    weights = validity.to(per_slot.dtype)
    return (per_slot * weights).sum() / weights.sum()


def masked_heterogeneous_slot_distillation_loss(
    predicted: dict[str, Tensor],
    target: dict[str, Tensor],
    validity: Tensor,
) -> Tensor:
    """One masked objective that reconstructs each frozen raw slot embedding."""

    if set(predicted) != set(SLOT_NAMES) or set(target) != set(SLOT_NAMES):
        raise ValueError(f"slot mappings must contain exactly {SLOT_NAMES}")
    if validity.ndim != 2 or validity.shape[1] != len(SLOT_NAMES):
        raise ValueError("slot validity must have shape [B, 4]")
    if validity.dtype is not torch.bool or not validity.any():
        raise ValueError("slot validity must be boolean with at least one target")
    losses: list[Tensor] = []
    weights: list[Tensor] = []
    for index, name in enumerate(SLOT_NAMES):
        if predicted[name].shape != target[name].shape or predicted[name].ndim != 2:
            raise ValueError(f"{name} prediction and target shapes disagree")
        if predicted[name].shape[0] != validity.shape[0]:
            raise ValueError(f"{name} batch size disagrees with validity")
        losses.append(
            (1.0 - F.cosine_similarity(predicted[name], target[name].detach(), dim=-1))
            * validity[:, index].to(predicted[name].dtype)
        )
        weights.append(validity[:, index].to(predicted[name].dtype))
    return torch.stack(losses, dim=1).sum() / torch.stack(weights, dim=1).sum()


def trusted_multi_positive_retrieval_loss(
    scores: Tensor,
    positive_mask: Tensor,
    trusted_negative_mask: Tensor,
) -> Tensor:
    """Make every positive outrank a shared set of trusted negatives.

    Unlabeled same-video nodes must be false in both masks and therefore never
    enter the denominator.
    """

    if scores.ndim != 2:
        raise ValueError("retrieval scores must have shape [B, N]")
    if positive_mask.shape != scores.shape or trusted_negative_mask.shape != scores.shape:
        raise ValueError("retrieval masks must match scores")
    if positive_mask.dtype is not torch.bool or trusted_negative_mask.dtype is not torch.bool:
        raise ValueError("retrieval masks must be boolean")
    if (positive_mask & trusted_negative_mask).any():
        raise ValueError("a candidate cannot be both positive and negative")
    if (~positive_mask.any(dim=1)).any():
        raise ValueError("every retrieval sample requires a positive")
    if (~trusted_negative_mask.any(dim=1)).any():
        raise ValueError("every retrieval sample requires a trusted negative")

    negative_scores = scores.masked_fill(~trusted_negative_mask, -torch.inf)
    negative_lse = torch.logsumexp(negative_scores, dim=1, keepdim=True)
    per_candidate = F.softplus(negative_lse - scores)
    weights = positive_mask.to(scores.dtype)
    return (per_candidate * weights).sum() / weights.sum()


def multi_positive_marginal_loss(
    scores: Tensor,
    positive_mask: Tensor,
    valid_mask: Tensor,
) -> Tensor:
    """BLIP-style marginal objective retained as a matched loss ablation."""

    if scores.shape != positive_mask.shape or scores.shape != valid_mask.shape:
        raise ValueError("marginal masks must match retrieval scores")
    if positive_mask.dtype is not torch.bool or valid_mask.dtype is not torch.bool:
        raise ValueError("marginal masks must be boolean")
    if (positive_mask & ~valid_mask).any():
        raise ValueError("all positives must be valid")
    if (~positive_mask.any(dim=1)).any():
        raise ValueError("every sample requires a positive")
    positive_lse = torch.logsumexp(scores.masked_fill(~positive_mask, -torch.inf), dim=1)
    valid_lse = torch.logsumexp(scores.masked_fill(~valid_mask, -torch.inf), dim=1)
    return (valid_lse - positive_lse).mean()
