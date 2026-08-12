"""Optional GTSAM real-evidence correction adapter for multi-trajectory IWM.

The adapter is action-aware and is called only after a real graph read.  It
shares one evidence/factor state across competing trajectories, while
projecting the categorical correction back into each trajectory's cursor
belief.  Numeric marginals never cross this boundary.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    Answerability,
    GraphReadExecution,
    RelationGrounding,
)

from factor_graph.gtsam_backend import GTSAM_AVAILABLE
from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend

from .contracts import (
    ActionKind,
    AnswerabilityState,
    CursorBeliefState,
    LegalGraphAction,
    RetainedEvidenceGraph,
)
from .multi_trajectory import shared_action_key


_IDENTITY = {
    "same_entity",
    "same_object",
    "same_instance_candidate",
    "reappears_candidate",
}
_STATE = {"state_transition", "transition_support"}
_COUNTER = {"contradicts"}


class GTSAMMultiTrajectoryBeliefUpdater:
    """Maintain one persistent shared factor state behind categorical beliefs."""

    name = "gtsam_multi_trajectory_backup/v0.1"

    def __init__(
        self,
        overlay: CausalTemporalOverlay,
        *,
        mode: str = "backup",
        verifier: Any | None = None,
    ) -> None:
        self.overlay = overlay
        self.mode = mode
        self.verifier = verifier
        self._backend: GTSAMExecutedReadBeliefBackend | None = None
        self._snapshot: Any | None = None
        self._last_execution_key: tuple[Any, ...] | None = None
        self._last_corrected_snapshot: Any | None = None
        self.audit_records: list[dict[str, Any]] = []

    @property
    def available(self) -> bool:
        return GTSAM_AVAILABLE

    def update_after_action(
        self,
        trajectory_id: str,
        previous_belief: CursorBeliefState,
        structurally_updated_belief: CursorBeliefState,
        action: LegalGraphAction,
        observation: MemoryNode,
        graph: RetainedEvidenceGraph,
    ) -> CursorBeliefState:
        del graph
        if not GTSAM_AVAILABLE:
            raise RuntimeError(
                "GTSAM backup is unavailable; install factor_graph/requirements-gtsam.txt. "
                "The IWM planner was not replaced or silently changed."
            )
        execution_key = (
            tuple(previous_belief.acquired_evidence),
            shared_action_key(action),
            observation.node_id,
        )
        reused_shared_measurement = execution_key == self._last_execution_key
        backend_audit: dict[str, Any] | None = None
        if reused_shared_measurement:
            corrected = self._last_corrected_snapshot
        else:
            self._ensure_initialized(previous_belief)
            assert self._backend is not None and self._snapshot is not None
            graph_action = _graph_read_action(action)
            result = self._backend.update_from_execution(
                self._snapshot,
                graph_action,
                GraphReadExecution(
                    observations=(observation,),
                    skill_invocation={
                        "status": "executed",
                        "source": "full_graph_iwm_real_read",
                        "imagined": False,
                    },
                ),
                self.overlay,
            )
            corrected = result.belief
            self._snapshot = corrected
            self._last_execution_key = execution_key
            self._last_corrected_snapshot = corrected
            backend_audit = result.audit_record
        self.audit_records.append(
            {
                "trajectory_id": trajectory_id,
                "observation_id": observation.node_id,
                "action_kind": action.kind.value,
                "backend": self.name,
                "mode": self.mode,
                "real_observation_only": True,
                "numeric_solver_state_exposed": False,
                "shared_measurement_reused": reused_shared_measurement,
                "backend_audit": backend_audit,
            }
        )
        return _project_snapshot(structurally_updated_belief, corrected)

    def _ensure_initialized(self, belief: CursorBeliefState) -> None:
        if self._backend is not None:
            return
        self._backend = GTSAMExecutedReadBeliefBackend(
            mode=self.mode,
            verifier=self.verifier,
        )
        self._snapshot = self._backend.initialize(
            belief.question,
            self.overlay,
            seed_evidence=belief.acquired_evidence,
            missing_roles=belief.missing_roles,
            graph_read_budget=belief.remaining_reads,
        )


def _graph_read_action(action: LegalGraphAction) -> GraphReadAction:
    relation = str(action.relation or "")
    if action.kind is ActionKind.START_AT:
        action_type = NavigationActionType.SEMANTIC
    elif action.kind is ActionKind.TEMPORAL_FORWARD:
        action_type = NavigationActionType.TEMPORAL_FORWARD
    elif action.kind is ActionKind.TEMPORAL_BACKWARD:
        action_type = NavigationActionType.TEMPORAL_BACK
    elif relation in _IDENTITY:
        action_type = NavigationActionType.TRACK_ENTITY
    elif relation in _STATE:
        action_type = NavigationActionType.INSPECT_STATE_CHANGE
    elif relation in _COUNTER:
        action_type = NavigationActionType.SEARCH_COUNTEREVIDENCE
    else:
        action_type = NavigationActionType.FOLLOW_DEPENDENCY
    return GraphReadAction(
        action_type=action_type,
        source_id=action.source_id,
        target_ids=(action.target_id,) if action.target_id is not None else (),
        relation=action.relation,
        rationale="executed full-graph reasoning read",
    )


def _project_snapshot(
    structural: CursorBeliefState,
    snapshot: Any,
) -> CursorBeliefState:
    accepted = tuple(
        state.edge_id
        for state in snapshot.relation_states
        if state.grounding is RelationGrounding.VERIFIED
    )
    rejected = tuple(
        state.edge_id
        for state in snapshot.relation_states
        if state.grounding is RelationGrounding.CONTRADICTED
    )
    unresolved = tuple(
        state.edge_id
        for state in snapshot.relation_states
        if state.grounding
        in {RelationGrounding.PARTIAL, RelationGrounding.ENDPOINTS_OBSERVED}
    )
    acquired = tuple(snapshot.acquired_evidence)
    bindings = tuple(
        (role, node_id)
        for role, node_id in structural.grounded_role_evidence
        if node_id in set(acquired)
    )
    return replace(
        structural,
        belief_id=f"{structural.belief_id}:gtsam-corrected",
        acquired_evidence=acquired,
        accepted_relations=accepted,
        rejected_relations=rejected,
        unresolved_relations=unresolved,
        missing_roles=tuple(snapshot.missing_roles),
        grounded_role_evidence=bindings,
        contradictions=tuple(snapshot.contradictions),
        answerability=(
            AnswerabilityState.READY
            if snapshot.answerability is Answerability.READY
            else AnswerabilityState.ABSTAIN
            if snapshot.answerability is Answerability.ABSTAIN
            else AnswerabilityState.NOT_READY
        ),
        remaining_reads=int(snapshot.remaining_graph_reads),
        step=max(structural.step, int(snapshot.step)),
    )
