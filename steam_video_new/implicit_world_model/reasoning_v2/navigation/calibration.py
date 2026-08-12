"""Hard-negative calibration gate for L1.5 proposals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import NavigationGraph


@dataclass(frozen=True)
class ProposalCalibrationReport:
    labeled_count: int
    positive_count: int
    trusted_negative_count: int
    admitted_positive_count: int
    admitted_negative_count: int
    recall: float | None
    precision: float | None
    blockers: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.blockers


def evaluate_proposals(
    graph: NavigationGraph,
    labels: Mapping[frozenset[str], str],
    *,
    minimum_precision: float = 0.9,
    minimum_recall: float = 0.5,
) -> ProposalCalibrationReport:
    """Evaluate after graph freeze using labels ``positive``/``hard_negative``."""

    allowed = {"positive", "hard_negative"}
    if set(labels.values()) - allowed:
        raise ValueError("proposal labels must be positive or hard_negative")
    admitted = {frozenset((row.src, row.dst)) for row in graph.proposals}
    positives = {pair for pair, label in labels.items() if label == "positive"}
    negatives = {pair for pair, label in labels.items() if label == "hard_negative"}
    true_positive = len(admitted & positives)
    false_positive = len(admitted & negatives)
    recall = true_positive / len(positives) if positives else None
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else None
    )
    blockers: list[str] = []
    if not positives:
        blockers.append("positive_labels_missing")
    if not negatives:
        blockers.append("trusted_hard_negatives_missing")
    if precision is not None and precision < minimum_precision:
        blockers.append("proposal_precision_below_gate")
    if recall is not None and recall < minimum_recall:
        blockers.append("proposal_recall_below_gate")
    return ProposalCalibrationReport(
        labeled_count=len(labels),
        positive_count=len(positives),
        trusted_negative_count=len(negatives),
        admitted_positive_count=true_positive,
        admitted_negative_count=false_positive,
        recall=recall,
        precision=precision,
        blockers=tuple(blockers),
    )
