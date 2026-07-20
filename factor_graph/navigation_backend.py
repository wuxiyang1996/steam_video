"""Persistent executed-read belief backend for closed-loop GTSAM navigation.

Only grounded categorical verifier decisions enter the measurement boundary.
Numeric marginals remain internal and are projected to categorical planner
state; they are never action rewards or LLM outputs.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode, RelationBelief
from steam_video_new.implicit_world_model.l15_graph_navigator.belief import (
    _infer_missing_roles,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    BeliefUpdateResult,
    GraphReadExecution,
    RelationGrounding,
    RelationState,
    UncertaintyChange,
    UncertaintyLevel,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.factor_graph import (
    FactorGraphBeliefBackend,
)

from .categorical import BeliefLabel
from .measurement import (
    MeasurementCalibrationRegistry,
    MeasurementOutcome,
    RelationMeasurement,
    measurement_from_execution,
)
from .post_read_verifier import PostReadCategoricalVerifier
from .session import GTSAMBeliefSession


_ACTION_ROLE = {
    NavigationActionType.SEMANTIC: "semantic",
    NavigationActionType.TEMPORAL_BACK: "temporal",
    NavigationActionType.TEMPORAL_FORWARD: "temporal",
    NavigationActionType.TRACK_ENTITY: "identity",
    NavigationActionType.INSPECT_STATE_CHANGE: "state_transition",
    NavigationActionType.FOLLOW_DEPENDENCY: "dependency",
    NavigationActionType.CANDIDATE_CAUSE: "dependency",
    NavigationActionType.EFFECT: "dependency",
    NavigationActionType.FIND_BRIDGE: "bridge",
    NavigationActionType.SEARCH_COUNTEREVIDENCE: "counterevidence",
    NavigationActionType.VERIFY: "verification",
}
_NON_RELATION_RESOLVERS = {
    NavigationActionType.SEMANTIC,
    NavigationActionType.FIND_BRIDGE,
    NavigationActionType.SEARCH_COUNTEREVIDENCE,
}
_TEMPORAL = {"before", "temporal_next", "overlaps", "during"}
_IDENTITY = {
    "same_entity",
    "same_object",
    "same_instance_candidate",
    "reappears_candidate",
}
_DEPENDENCY = {
    "state_transition",
    "transition_support",
    "response_candidate",
    "observation_support",
    "explains",
    "enables",
}
_ROLE_RELATIONS = {
    "temporal": _TEMPORAL,
    "identity": _IDENTITY,
    "state_transition": {"state_transition"},
    "dependency": _DEPENDENCY,
    "bridge": _TEMPORAL | _IDENTITY | _DEPENDENCY,
    "counterevidence": {"contradicts"},
    "verification": _IDENTITY | _DEPENDENCY,
}


class GTSAMExecutedReadBeliefBackend:
    """Own persistent navigation belief independently of persisted verifiers."""

    name = "gtsam_executed_read_navigation/v0.2"
    allowed_modes = {"correct", "verifier_only", "frozen", "shuffled"}

    def __init__(self, *, mode: str = "correct", verifier: Any | None = None) -> None:
        if mode not in self.allowed_modes:
            raise ValueError(f"unknown GTSAM navigation mode: {mode}")
        self.mode = mode
        self.verifier = verifier or PostReadCategoricalVerifier()
        self.calibration = MeasurementCalibrationRegistry.pilot()
        self.session: GTSAMBeliefSession | None = None
        self.overlay: CausalTemporalOverlay | None = None
        self._templates: tuple[RelationState, ...] = ()
        self._initial_missing_roles: tuple[str, ...] = ()
        self._resolved_non_relation_roles: set[str] = set()
        self._direct_measurements: list[tuple[RelationMeasurement, str]] = []
        self._affected_variables: set[str] = set()
        self._initial_inference: Any = None

    def initialize(
        self,
        question: str,
        overlay: CausalTemporalOverlay,
        *,
        seed_evidence: tuple[str, ...] = (),
        missing_roles: tuple[str, ...] | None = None,
        graph_read_budget: int = 8,
    ) -> BeliefSnapshot:
        # Reuse only the validated structural conversion. All persisted
        # verifier state, posterior values, blocks, and priorities are removed.
        structural = FactorGraphBeliefBackend().initialize(
            question,
            overlay,
            seed_evidence=seed_evidence,
            missing_roles=missing_roles,
            graph_read_budget=graph_read_budget,
        )
        self.overlay = overlay
        self.session = GTSAMBeliefSession(
            overlay,
            calibration=self.calibration,
            activate_persisted_verified_measurements=False,
        )
        self._initial_inference = self.session.snapshot()
        self._initial_missing_roles = (
            _infer_missing_roles(question)
            if missing_roles is None
            else tuple(dict.fromkeys(missing_roles))
        )
        self._templates = tuple(
            replace(
                state,
                posterior_probabilities=(),
                verified_relations=(),
                factor_sources=tuple(
                    source
                    for source in state.factor_sources
                    if source not in {
                        "hard_verified_relation",
                        "persisted_hard_verifier_measurement",
                    }
                ),
                calibration_status="gtsam_prior_without_persisted_verifiers",
                grounding=RelationGrounding.UNSEEN,
            )
            for state in structural.relation_states
        )
        return self._build_belief(
            question=question,
            acquired=tuple(dict.fromkeys(seed_evidence)),
            frontier=tuple(dict.fromkeys(seed_evidence))[-1:],
            remaining_graph_reads=graph_read_budget,
            step=0,
            belief_id=f"{overlay.overlay_id}:gtsam-belief:0",
        )

    def update(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        observations: list[MemoryNode],
        overlay: CausalTemporalOverlay,
    ) -> BeliefUpdateResult:
        raise RuntimeError(
            "GTSAM executed-read backend requires update_from_execution; "
            "use ClosedLoopNavigator"
        )

    def update_from_execution(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        execution: GraphReadExecution,
        overlay: CausalTemporalOverlay,
    ) -> BeliefUpdateResult:
        self._require_session(overlay)
        observed_ids = _validated_observation_ids(execution.observations, overlay)
        acquired = tuple(dict.fromkeys(belief.acquired_evidence + observed_ids))
        next_step = belief.step + 1
        edge = _edge_for_action(action, overlay)
        audit: dict[str, object]
        if edge is None or execution.skill_invocation.get("status") != "executed":
            role = _ACTION_ROLE.get(action.action_type)
            if observed_ids and action.action_type in _NON_RELATION_RESOLVERS and role:
                self._resolved_non_relation_roles.add(role)
            audit = {
                "backend": self.name,
                "mode": self.mode,
                "measurement_status": "not_applicable",
                "reason": "executed action did not cover one typed relation",
            }
        else:
            audit = self._verify_and_record(
                belief=belief,
                action=action,
                execution=execution,
                overlay=overlay,
                edge=edge,
            )
        corrected = self._build_belief(
            question=belief.question,
            acquired=acquired,
            frontier=observed_ids or belief.frontier,
            remaining_graph_reads=max(0, belief.remaining_graph_reads - 1),
            step=next_step,
            belief_id=_belief_id(overlay.overlay_id, next_step, action),
        )
        return BeliefUpdateResult(
            belief=corrected,
            delta=_realized_delta(belief, corrected),
            audit_record=audit,
        )

    def _verify_and_record(
        self,
        *,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        execution: GraphReadExecution,
        overlay: CausalTemporalOverlay,
        edge: RelationBelief,
    ) -> dict[str, object]:
        if self.session is None:
            raise RuntimeError("GTSAM backend is not initialized")
        decision = self.verifier.verify(
            action=action,
            execution=execution,
            overlay=overlay,
            edge_id=edge.edge_id,
            acquired_observation_ids=belief.acquired_evidence,
        )
        previously_grounded = tuple(
            dict.fromkeys(
                ref
                for node in overlay.atomic_events + overlay.l1_observations
                if node.node_id in set(belief.acquired_evidence)
                for ref in node.source_segments
            )
        )
        before = self._inference()
        measurement = measurement_from_execution(
            action=action,
            execution=execution,
            overlay=overlay,
            decision=decision,
            calibration_version=self.calibration.version,
            previously_grounded_evidence_refs=previously_grounded,
        )
        injected: RelationMeasurement | None = None
        factor_activated = False
        if self.mode == "correct":
            self.session.journal.append(measurement)
            self._direct_measurements.append((measurement, _ACTION_ROLE[action.action_type]))
            factor_activated = decision.outcome is not MeasurementOutcome.INCONCLUSIVE
        elif self.mode == "verifier_only":
            self._direct_measurements.append((measurement, _ACTION_ROLE[action.action_type]))
        elif self.mode == "shuffled":
            injected = _shuffle_measurement_target(measurement, overlay)
            if injected is not None:
                self.session.journal.append(injected)
                factor_activated = injected.outcome is not MeasurementOutcome.INCONCLUSIVE
        after = self._inference()
        changed = _categorical_changes(before, after)
        if self.mode in {"correct", "shuffled"}:
            self._affected_variables.update(changed)
        if injected is not None and factor_activated:
            self._affected_variables.add(injected.variable_id)
        direct_variable = measurement.variable_id
        return {
            "backend": self.name,
            "mode": self.mode,
            "measurement_status": {
                "correct": "appended",
                "verifier_only": "verifier_direct_only",
                "frozen": "frozen",
                "shuffled": "shuffled" if injected is not None else "shuffle_unavailable",
            }[self.mode],
            "factor_activated": factor_activated,
            "verifier_decision": decision.to_dict(),
            "measurement": measurement.to_dict(),
            "injected_measurement": injected.to_dict() if injected is not None else None,
            "changed_variables": list(changed),
            "direct_changed_variables": [name for name in changed if name == direct_variable],
            "propagated_changed_variables": [name for name in changed if name != direct_variable],
        }

    def _build_belief(
        self,
        *,
        question: str,
        acquired: tuple[str, ...],
        frontier: tuple[str, ...],
        remaining_graph_reads: int,
        step: int,
        belief_id: str,
    ) -> BeliefSnapshot:
        if self.session is None or self.overlay is None:
            raise RuntimeError("GTSAM backend is not initialized")
        inference = self._inference()
        acquired_set = set(acquired)
        outcomes = _outcomes_by_variable(self._direct_measurements)
        direct_roles = _resolved_roles(self._direct_measurements, inference, self.mode)
        resolved_roles = self._resolved_non_relation_roles | direct_roles
        missing = tuple(
            role for role in self._initial_missing_roles if role not in resolved_roles
        )
        states: list[RelationState] = []
        blocked: set[str] = set()
        contradictions: set[str] = set()
        affected = self._affected_variables | set(outcomes)
        for template in self._templates:
            edge = _edge_by_id(self.overlay, template.edge_id)
            relation_labels = {
                relation: inference.categorical[f"relation::{template.edge_id}::{relation}"]
                for relation, _ in template.relation_probabilities
            }
            verified = tuple(
                sorted(
                    relation
                    for relation, _ in template.relation_probabilities
                    if _is_accepted(
                        f"relation::{template.edge_id}::{relation}",
                        relation_labels[relation],
                        outcomes,
                        affected,
                        self.mode,
                    )
                )
            )
            rejected = any(
                _is_rejected(
                    f"relation::{template.edge_id}::{relation}",
                    relation_labels[relation],
                    outcomes,
                    affected,
                    self.mode,
                )
                for relation, _ in template.relation_probabilities
            )
            conflicting = any(
                len(outcomes.get(f"relation::{template.edge_id}::{relation}", set()))
                > 1
                for relation, _ in template.relation_probabilities
            )
            observed_count = int(template.src in acquired_set) + int(template.dst in acquired_set)
            if rejected or conflicting:
                grounding = RelationGrounding.CONTRADICTED
                blocked.add(template.edge_id)
            elif verified or edge.status.value == "deterministic":
                grounding = RelationGrounding.VERIFIED
            elif observed_count == 0:
                grounding = RelationGrounding.UNSEEN
            elif observed_count == 1:
                grounding = RelationGrounding.PARTIAL
            else:
                grounding = RelationGrounding.ENDPOINTS_OBSERVED
            if conflicting:
                contradictions.add(template.edge_id)
            if (
                "contradicts" in verified
                and f"relation::{template.edge_id}::contradicts" in outcomes
            ):
                contradictions.add(template.edge_id)
            factor_sources = set(template.factor_sources)
            for relation, _ in template.relation_probabilities:
                variable = f"relation::{template.edge_id}::{relation}"
                if variable in affected:
                    factor_sources.add("executed_graph_read_belief")
            states.append(
                replace(
                    template,
                    posterior_probabilities=tuple(
                        (relation, inference.posteriors[f"relation::{template.edge_id}::{relation}"])
                        for relation, _ in template.relation_probabilities
                    ),
                    verified_relations=verified,
                    factor_sources=tuple(sorted(factor_sources)),
                    calibration_status=self.calibration.version,
                    grounding=grounding,
                )
            )
        priority = _priority_edges(tuple(states), missing, blocked, affected, inference)
        answerability = (
            Answerability.READY
            if acquired and not missing and not contradictions
            else Answerability.NOT_READY
        )
        return BeliefSnapshot(
            belief_id=belief_id,
            backend_name=f"{self.name}:{self.mode}",
            backend_ref=(
                f"gtsam:{inference.variable_count}v:{inference.factor_count}f:"
                f"active_measurements={_active_measurement_count(self.session)}"
            ),
            question=question,
            acquired_evidence=acquired,
            frontier=frontier,
            missing_roles=missing,
            contradictions=tuple(sorted(contradictions)),
            relation_states=tuple(states),
            priority_edge_ids=priority,
            blocked_edge_ids=tuple(sorted(blocked)),
            uncertainty=_uncertainty_level(missing, acquired),
            answerability=answerability,
            remaining_graph_reads=remaining_graph_reads,
            step=step,
        )

    def _inference(self) -> Any:
        if self.session is None:
            raise RuntimeError("GTSAM backend is not initialized")
        return self._initial_inference if self.mode in {"frozen", "verifier_only"} else self.session.snapshot()

    def _require_session(self, overlay: CausalTemporalOverlay) -> None:
        if self.session is None or self.overlay is None:
            raise RuntimeError("GTSAM backend must be initialized before update")
        if overlay.overlay_id != self.overlay.overlay_id:
            raise ValueError("GTSAM backend cannot switch overlays within one session")


def _validated_observation_ids(
    observations: tuple[MemoryNode, ...],
    overlay: CausalTemporalOverlay,
) -> tuple[str, ...]:
    known = {
        node.node_id for node in overlay.atomic_events + overlay.l1_observations
    }
    return tuple(dict.fromkeys(node.node_id for node in observations if node.node_id in known))


def _edge_for_action(
    action: GraphReadAction,
    overlay: CausalTemporalOverlay,
) -> RelationBelief | None:
    if action.source_id is None or action.relation is None:
        return None
    candidates = [
        edge
        for edge in overlay.relations + overlay.l1_structural_relations
        if action.relation in edge.relation_probabilities
        and action.source_id in {edge.src, edge.dst}
        and any(target in {edge.src, edge.dst} for target in action.target_ids)
    ]
    if len(candidates) > 1:
        raise ValueError(
            "graph action ambiguously covers multiple typed edges; action must carry an edge id"
        )
    return candidates[0] if candidates else None


def _edge_by_id(overlay: CausalTemporalOverlay, edge_id: str) -> RelationBelief:
    return next(
        edge
        for edge in overlay.relations + overlay.l1_structural_relations
        if edge.edge_id == edge_id
    )


def _outcomes_by_variable(
    measurements: list[tuple[RelationMeasurement, str]],
) -> dict[str, set[MeasurementOutcome]]:
    outcomes: dict[str, set[MeasurementOutcome]] = {}
    for measurement, _ in measurements:
        if measurement.outcome is MeasurementOutcome.INCONCLUSIVE:
            continue
        outcomes.setdefault(measurement.variable_id, set()).add(measurement.outcome)
    return outcomes


def _resolved_roles(
    measurements: list[tuple[RelationMeasurement, str]],
    inference: Any,
    mode: str,
) -> set[str]:
    outcomes = _outcomes_by_variable(measurements)
    resolved: set[str] = set()
    for measurement, role in measurements:
        variable_outcomes = outcomes.get(measurement.variable_id, set())
        accepted = variable_outcomes == {MeasurementOutcome.SUPPORTS}
        if mode == "correct":
            accepted = accepted and inference.categorical[measurement.variable_id] is BeliefLabel.ACCEPTED
        if accepted:
            resolved.add(role)
    return resolved


def _is_accepted(
    variable: str,
    label: BeliefLabel,
    outcomes: dict[str, set[MeasurementOutcome]],
    affected: set[str],
    mode: str,
) -> bool:
    direct = outcomes.get(variable, set())
    if direct:
        return direct == {MeasurementOutcome.SUPPORTS} and (
            mode == "verifier_only" or label is BeliefLabel.ACCEPTED
        )
    return mode in {"correct", "shuffled"} and variable in affected and label is BeliefLabel.ACCEPTED


def _is_rejected(
    variable: str,
    label: BeliefLabel,
    outcomes: dict[str, set[MeasurementOutcome]],
    affected: set[str],
    mode: str,
) -> bool:
    direct = outcomes.get(variable, set())
    if not direct:
        return mode in {"correct", "shuffled"} and variable in affected and label is BeliefLabel.REJECTED
    return direct == {MeasurementOutcome.REJECTS} and (
        mode == "verifier_only" or label is BeliefLabel.REJECTED
    )


def _priority_edges(
    states: tuple[RelationState, ...],
    missing_roles: tuple[str, ...],
    blocked: set[str],
    affected_variables: set[str],
    inference: Any,
) -> tuple[str, ...]:
    required = set().union(*(_ROLE_RELATIONS.get(role, set()) for role in missing_roles)) if missing_roles else set()
    priority: list[str] = []
    for state in states:
        if state.edge_id in blocked:
            continue
        relations = {relation for relation, _ in state.relation_probabilities}
        if required and not relations & required:
            continue
        variables = {
            f"relation::{state.edge_id}::{relation}" for relation in relations
        }
        affected_uncertain = any(
            variable in affected_variables
            and inference.categorical[variable] is BeliefLabel.UNCERTAIN
            for variable in variables
        )
        if state.grounding in {RelationGrounding.PARTIAL, RelationGrounding.ENDPOINTS_OBSERVED} or affected_uncertain:
            priority.append(state.edge_id)
    return tuple(dict.fromkeys(priority))


def _shuffle_measurement_target(
    measurement: RelationMeasurement,
    overlay: CausalTemporalOverlay,
) -> RelationMeasurement | None:
    variables = sorted(
        (f"relation::{edge.edge_id}::{relation}", edge.edge_id, relation)
        for edge in overlay.relations + overlay.l1_structural_relations
        for relation in edge.relation_probabilities
    )
    alternatives = [row for row in variables if row[0] != measurement.variable_id]
    if not alternatives:
        return None
    digest = hashlib.sha256(measurement.measurement_id.encode("utf-8")).digest()
    variable_id, edge_id, relation = alternatives[
        int.from_bytes(digest[:4], "big") % len(alternatives)
    ]
    return replace(
        measurement,
        measurement_id=f"{measurement.measurement_id}:shuffled",
        edge_id=edge_id,
        relation=relation,
        variable_id=variable_id,
        reasons=measurement.reasons + ("destructive control: measurement target shuffled",),
    )


def _categorical_changes(before: Any, after: Any) -> tuple[str, ...]:
    return tuple(
        sorted(
            variable
            for variable in before.categorical
            if before.categorical[variable] is not after.categorical[variable]
        )
    )


def _active_measurement_count(session: GTSAMBeliefSession) -> int:
    return sum(
        measurement.outcome is not MeasurementOutcome.INCONCLUSIVE
        for measurement in session.journal.measurements
    )


def _belief_id(overlay_id: str, step: int, action: GraphReadAction) -> str:
    raw = "\x1f".join(
        (action.action_type.value, action.source_id or "", *action.target_ids, action.relation or "")
    )
    return f"{overlay_id}:gtsam-belief:{step}:{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _uncertainty_level(
    missing_roles: tuple[str, ...],
    acquired: tuple[str, ...],
) -> UncertaintyLevel:
    if not acquired or len(missing_roles) > 1:
        return UncertaintyLevel.HIGH
    if missing_roles:
        return UncertaintyLevel.MEDIUM
    return UncertaintyLevel.LOW


def _realized_delta(before: BeliefSnapshot, after: BeliefSnapshot) -> BeliefDeltaDescriptor:
    before_states = {state.edge_id: state for state in before.relation_states}
    relation_updates = tuple(
        state.edge_id
        for state in after.relation_states
        if before_states.get(state.edge_id) != state
    )
    order = {UncertaintyLevel.LOW: 0, UncertaintyLevel.MEDIUM: 1, UncertaintyLevel.HIGH: 2}
    uncertainty_change = (
        UncertaintyChange.DECREASE
        if order[after.uncertainty] < order[before.uncertainty]
        else UncertaintyChange.INCREASE
        if order[after.uncertainty] > order[before.uncertainty]
        else UncertaintyChange.UNCHANGED
    )
    return BeliefDeltaDescriptor(
        resolved_roles=tuple(role for role in before.missing_roles if role not in after.missing_roles),
        relation_updates=relation_updates,
        contradiction_updates=tuple(
            edge_id for edge_id in after.contradictions if edge_id not in before.contradictions
        ),
        uncertainty_change=uncertainty_change,
        answerability_after=after.answerability,
        predicted_only=False,
    )
