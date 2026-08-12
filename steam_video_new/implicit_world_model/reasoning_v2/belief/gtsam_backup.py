"""Optional GTSAM correction after a grounded read.

This adapter deliberately sits outside the IWM and planner.  It receives one
shared physical measurement after execution, keeps all numeric solver state
private, and projects only a categorical correction into each reasoning path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from factor_graph.gtsam_backend import GTSAM_AVAILABLE
from factor_graph.navigation_backend import GTSAMExecutedReadBeliefBackend
from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode, RelationBelief
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    Answerability as BackendAnswerability,
    GraphReadExecution,
)

from ..navigation.contracts import NavigationAction, ProposalKind
from ..world_model.contracts import (
    Answerability,
    BeliefState,
    GroundedBeliefEffect,
    RealObservation,
)


@dataclass(frozen=True)
class _SharedCorrection:
    target_id: str
    missing_roles: tuple[str, ...]
    contradictions: tuple[str, ...]
    answerability: Answerability


class GTSAMRealReadCorrector:
    """Maintain one persistent shared factor state behind categorical paths."""

    model_name = "gtsam-real-read-backup/v2"

    def __init__(
        self,
        overlay: CausalTemporalOverlay,
        *,
        mode: str = "backup",
        verifier: Any | None = None,
        read_budget: int = 8,
    ) -> None:
        if read_budget <= 0:
            raise ValueError("GTSAM backup requires a positive read budget")
        self.overlay = overlay
        self.mode = mode
        self.verifier = verifier
        self.read_budget = read_budget
        self._backend: GTSAMExecutedReadBeliefBackend | None = None
        self._snapshot: Any | None = None
        self.audit_records: list[dict[str, object]] = []

    @property
    def available(self) -> bool:
        return GTSAM_AVAILABLE

    def correct_batch(
        self,
        *,
        path_hypotheses: Mapping[str, str],
        beliefs: Mapping[str, BeliefState],
        acquired_ids_before: tuple[str, ...],
        observation: RealObservation,
    ) -> tuple[GroundedBeliefEffect, ...]:
        if not GTSAM_AVAILABLE:
            raise RuntimeError(
                "GTSAM backup is unavailable; install "
                "factor_graph/requirements-gtsam.txt. The IWM/planner path "
                "remains usable without this optional corrector."
            )
        if set(path_hypotheses) != set(beliefs):
            raise ValueError("GTSAM correction path coverage mismatch")
        if not beliefs:
            return ()
        self._ensure_initialized(beliefs, acquired_ids_before)
        assert self._backend is not None and self._snapshot is not None
        node = self._node(observation.target_id)
        graph_action = _graph_action(observation.action, self.overlay)
        result = self._backend.update_from_execution(
            self._snapshot,
            graph_action,
            GraphReadExecution(
                observations=(node,),
                skill_invocation={
                    "status": "executed",
                    "source": "reasoning_v2_real_read",
                    "imagined": False,
                },
            ),
            self.overlay,
        )
        self._snapshot = result.belief
        correction = _categorical_correction(observation.target_id, result.belief)
        self.audit_records.append(
            {
                "target_id": observation.target_id,
                "path_count": len(beliefs),
                "backend": self.model_name,
                "mode": self.mode,
                "real_observation_only": True,
                "shared_measurement_count": 1,
                "numeric_solver_state_exposed": False,
                "backend_audit": result.audit_record,
            }
        )
        return tuple(
            GroundedBeliefEffect(
                path_id=path_id,
                belief_after=_project(belief, correction),
                direct_same_target=True,
                verified=True,
                rationale="categorical projection of one shared grounded read",
            )
            for path_id, belief in beliefs.items()
        )

    def _ensure_initialized(
        self,
        beliefs: Mapping[str, BeliefState],
        acquired_ids_before: tuple[str, ...],
    ) -> None:
        if self._backend is not None:
            return
        questions = {belief.question for belief in beliefs.values()}
        if len(questions) != 1:
            raise ValueError("one GTSAM session cannot mix questions")
        missing_roles = tuple(
            dict.fromkeys(
                role for belief in beliefs.values() for role in belief.missing_roles
            )
        )
        self._backend = GTSAMExecutedReadBeliefBackend(
            mode=self.mode,
            verifier=self.verifier,
        )
        self._snapshot = self._backend.initialize(
            next(iter(questions)),
            self.overlay,
            seed_evidence=acquired_ids_before,
            missing_roles=missing_roles,
            graph_read_budget=self.read_budget,
        )

    def _node(self, node_id: str) -> MemoryNode:
        nodes = self.overlay.l1_observations + self.overlay.atomic_events
        try:
            return next(node for node in nodes if node.node_id == node_id)
        except StopIteration as exc:
            raise ValueError(
                "GTSAM backup can only correct reads grounded in its overlay: "
                f"{node_id}"
            ) from exc


def _graph_action(
    action: NavigationAction,
    overlay: CausalTemporalOverlay,
) -> GraphReadAction:
    relation = _structural_relation(action, overlay)
    action_type = NavigationActionType.SEMANTIC
    if relation is not None:
        action_type = (
            NavigationActionType.TEMPORAL_FORWARD
            if _moves_forward(action, overlay)
            else NavigationActionType.TEMPORAL_BACK
        )
    return GraphReadAction(
        action_type=action_type,
        source_id=action.source_id,
        target_ids=(str(action.target_id),),
        relation=relation,
        rationale="executed reasoning-v2 evidence read",
    )


def _structural_relation(
    action: NavigationAction,
    overlay: CausalTemporalOverlay,
) -> str | None:
    if action.proposal_kind not in {
        ProposalKind.TEMPORAL,
        ProposalKind.EVENT_CONTINUATION_CANDIDATE,
    }:
        return None
    edge = next(
        (
            row
            for row in overlay.relations + overlay.l1_structural_relations
            if row.edge_id == action.proposal_id
        ),
        None,
    )
    if edge is None:
        return None
    return _temporal_label(edge)


def _temporal_label(edge: RelationBelief) -> str | None:
    labels = tuple(
        label
        for label in ("temporal_next", "before", "overlaps", "during")
        if label in edge.relation_probabilities
    )
    return labels[0] if len(labels) == 1 else None


def _moves_forward(
    action: NavigationAction,
    overlay: CausalTemporalOverlay,
) -> bool:
    node_by_id = {
        node.node_id: node for node in overlay.l1_observations + overlay.atomic_events
    }
    if action.source_id not in node_by_id or action.target_id not in node_by_id:
        return True
    return (
        node_by_id[str(action.target_id)].time_span.start_s
        >= node_by_id[action.source_id].time_span.start_s
    )


def _categorical_correction(target_id: str, snapshot: Any) -> _SharedCorrection:
    answerability = (
        Answerability.READY
        if snapshot.answerability is BackendAnswerability.READY
        else Answerability.ABSTAIN
        if snapshot.answerability is BackendAnswerability.ABSTAIN
        else Answerability.NOT_READY
    )
    return _SharedCorrection(
        target_id=target_id,
        missing_roles=tuple(snapshot.missing_roles),
        contradictions=tuple(snapshot.contradictions),
        answerability=answerability,
    )


def _project(belief: BeliefState, correction: _SharedCorrection) -> BeliefState:
    missing = tuple(
        role for role in belief.missing_roles if role in correction.missing_roles
    )
    answerability = (
        correction.answerability
        if correction.answerability is not Answerability.READY or not missing
        else Answerability.NOT_READY
    )
    return BeliefState(
        question=belief.question,
        required_roles=belief.required_roles,
        missing_roles=missing,
        contradictions=tuple(
            dict.fromkeys((*belief.contradictions, *correction.contradictions))
        ),
        answerability=answerability,
        grounded_role_evidence=belief.grounded_role_evidence,
    )
