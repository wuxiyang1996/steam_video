"""Categorical activation policy for the optional GTSAM belief backup."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from memory_graph.navigation import GraphReadAction, NavigationActionType
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    BeliefSnapshot,
)

from .measurement import MeasurementOutcome, RelationMeasurement, VerifierDecision


class CorrectionMode(str, Enum):
    IWM_BELIEF_ONLY = "iwm_belief_only"
    IWM_WITH_GTSAM_BACKUP = "iwm_with_gtsam_backup"
    GTSAM_ALWAYS = "gtsam_always"


@dataclass(frozen=True)
class BackupTriggerDecision:
    activate: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"activate": self.activate, "reasons": list(self.reasons)}


class CategoricalBackupTriggerPolicy:
    """Activate GTSAM only from explicit categorical correction conditions."""

    name = "categorical_gtsam_backup_trigger/v0.1"

    def evaluate(
        self,
        *,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        decision: VerifierDecision,
        verification_history: tuple[RelationMeasurement, ...] = (),
    ) -> BackupTriggerDecision:
        reasons: list[str] = []
        if belief.contradictions:
            reasons.append("contradiction_detected")
        if decision.outcome is MeasurementOutcome.REJECTS:
            reasons.append("contradiction_detected")
            if action.action_type is NavigationActionType.INSPECT_STATE_CHANGE:
                reasons.append("state_history_conflict")
        previous = {
            measurement.outcome
            for measurement in verification_history
            if measurement.variable_id
            == f"relation::{decision.edge_id}::{decision.relation}"
            and measurement.outcome is not MeasurementOutcome.INCONCLUSIVE
        }
        if previous and decision.outcome not in previous and (
            decision.outcome is not MeasurementOutcome.INCONCLUSIVE
        ):
            reasons.append("multiple_competing_hypotheses")
        if (
            action.action_type is NavigationActionType.TRACK_ENTITY
            and decision.outcome is MeasurementOutcome.INCONCLUSIVE
        ):
            reasons.append("identity_ambiguous")
        return BackupTriggerDecision(bool(reasons), tuple(dict.fromkeys(reasons)))


class AlwaysActivatePolicy:
    name = "gtsam_always/v0.1"

    def evaluate(self, **_: object) -> BackupTriggerDecision:
        return BackupTriggerDecision(True, ("gtsam_always",))
