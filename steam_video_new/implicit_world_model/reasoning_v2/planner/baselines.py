"""Categorical planner baselines."""

from __future__ import annotations

from typing import Sequence

from .contracts import (
    JointActionTree,
    PreferenceStatus,
    ReasoningPathForest,
    TrajectoryPreferenceDecision,
)


class IncomparablePlannerBaseline:
    model_name = "incomparable-planner-baseline/v0.1"

    def choose(
        self,
        forest: ReasoningPathForest,
        candidates: Sequence[JointActionTree],
    ) -> TrajectoryPreferenceDecision:
        del forest
        return TrajectoryPreferenceDecision(
            status=PreferenceStatus.INCOMPARABLE,
            frontier_candidate_ids=tuple(row.tree_id for row in candidates),
            selected_candidate_id=None,
            rationale="no grounded categorical preference available",
        )
