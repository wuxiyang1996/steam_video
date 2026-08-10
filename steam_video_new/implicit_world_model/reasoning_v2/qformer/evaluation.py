"""Retrieval metrics for fixed candidate pools and multi-clue positives."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def retrieval_metrics(
    scores: np.ndarray,
    positive_mask: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ks: Sequence[int] = (1, 4, 8),
) -> dict[str, float]:
    if scores.ndim != 2 or positive_mask.shape != scores.shape or valid_mask.shape != scores.shape:
        raise ValueError("retrieval arrays must share shape [B, N]")
    if positive_mask.dtype != np.bool_ or valid_mask.dtype != np.bool_:
        raise ValueError("retrieval masks must be boolean")
    if np.any(positive_mask & ~valid_mask) or np.any(~positive_mask.any(axis=1)):
        raise ValueError("every sample requires valid positives")
    ranks: list[np.ndarray] = []
    recalls = {int(k): [] for k in ks}
    coverage = {int(k): [] for k in ks}
    reciprocal_ranks: list[float] = []
    for row_scores, row_positive, row_valid in zip(scores, positive_mask, valid_mask):
        valid_indices = np.flatnonzero(row_valid)
        ordered = valid_indices[np.argsort(-row_scores[valid_indices], kind="stable")]
        positive_indices = set(np.flatnonzero(row_positive).tolist())
        positive_ranks = np.asarray(
            [rank + 1 for rank, index in enumerate(ordered) if int(index) in positive_indices]
        )
        ranks.append(positive_ranks)
        reciprocal_ranks.append(1.0 / float(positive_ranks.min()))
        for k in ks:
            retrieved = set(ordered[: int(k)].tolist())
            hit_count = len(retrieved & positive_indices)
            recalls[int(k)].append(hit_count / len(positive_indices))
            coverage[int(k)].append(float(hit_count == len(positive_indices)))
    result = {"mrr": float(np.mean(reciprocal_ranks))}
    for k in ks:
        result[f"recall@{k}"] = float(np.mean(recalls[int(k)]))
        result[f"all_clue_coverage@{k}"] = float(np.mean(coverage[int(k)]))
    return result


def clue_group_metrics(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    positive_group_masks: Sequence[np.ndarray],
    *,
    ks: Sequence[int] = (1, 4, 8),
) -> dict[str, float]:
    """Measure whether Top-K hits each clue interval's interchangeable nodes."""

    if scores.ndim != 2 or valid_mask.shape != scores.shape:
        raise ValueError("group metric scores and validity must share shape [B, N]")
    if valid_mask.dtype != np.bool_ or len(positive_group_masks) != scores.shape[0]:
        raise ValueError("group metric validity or batch size is invalid")
    recalls = {int(k): [] for k in ks}
    coverage = {int(k): [] for k in ks}
    for row_scores, row_valid, groups in zip(scores, valid_mask, positive_group_masks):
        groups = np.asarray(groups)
        if groups.ndim != 2 or groups.shape[1] != scores.shape[1] or groups.dtype != np.bool_:
            raise ValueError("each positive group mask must have shape [G, N]")
        if not len(groups) or np.any(~groups.any(axis=1)) or np.any(groups & ~row_valid):
            raise ValueError("every clue group requires valid candidate nodes")
        valid_indices = np.flatnonzero(row_valid)
        ordered = valid_indices[np.argsort(-row_scores[valid_indices], kind="stable")]
        for k in ks:
            retrieved = np.zeros(scores.shape[1], dtype=np.bool_)
            retrieved[ordered[: int(k)]] = True
            hit = np.any(groups & retrieved[None, :], axis=1)
            recalls[int(k)].append(float(hit.mean()))
            coverage[int(k)].append(float(hit.all()))
    result: dict[str, float] = {}
    for k in ks:
        result[f"clue_group_recall@{k}"] = float(np.mean(recalls[int(k)]))
        result[f"all_clue_group_coverage@{k}"] = float(np.mean(coverage[int(k)]))
    return result


def permutation_consistency(
    original_scores: np.ndarray,
    permuted_scores: np.ndarray,
    permutation: np.ndarray,
) -> dict[str, float]:
    if original_scores.shape != permuted_scores.shape or permutation.shape != original_scores.shape:
        raise ValueError("permutation arrays must share shape")
    restored = np.take_along_axis(permuted_scores, np.argsort(permutation, axis=1), axis=1)
    difference = np.abs(original_scores - restored)
    return {
        "max_absolute_error": float(difference.max(initial=0.0)),
        "mean_absolute_error": float(difference.mean()) if difference.size else 0.0,
    }
