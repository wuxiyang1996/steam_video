"""Metrics: candidate accuracy (proxy EM), margin curves, type breakdown."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


def candidate_accuracy(logits: torch.Tensor) -> float:
    # gold is index 0
    pred = logits.argmax(dim=-1)
    return (pred == 0).float().mean().item()


def summarize_margins(margins: torch.Tensor) -> dict[str, float]:
    # [B,K]
    out = {}
    means = margins.mean(dim=0)
    for i, v in enumerate(means.tolist()):
        out[f"m_{i}"] = float(v)
    out["m_last"] = float(means[-1].item())
    out["m_gain"] = float((means[-1] - means[0]).item()) if means.numel() > 1 else 0.0
    mono = (margins[:, 1:] > margins[:, :-1]).float().mean().item() if margins.size(1) > 1 else 0.0
    out["mono_rate"] = float(mono)
    return out


def type_breakdown(logits: torch.Tensor, qtypes: list[str]) -> dict[str, float]:
    pred = logits.argmax(dim=-1)
    correct = (pred == 0)
    buckets: dict[str, list[float]] = defaultdict(list)
    for c, t in zip(correct.tolist(), qtypes):
        buckets[str(t)].append(float(c))
    return {t: sum(v) / max(len(v), 1) for t, v in buckets.items()}


def pack_results(
    name: str,
    logits: torch.Tensor,
    margins: torch.Tensor,
    qtypes: list[str],
) -> dict[str, Any]:
    return {
        "model": name,
        "acc": candidate_accuracy(logits),
        "n": int(logits.size(0)),
        **summarize_margins(margins),
        "by_type": type_breakdown(logits, qtypes),
    }
