"""Hybrid discrete factor-graph backend for exploration-time belief maintenance.

Numeric probabilities are internal belief marginals. They are never action
rewards and never enter the preference-only planner as utility values.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import math
from typing import Iterable

from memory_graph.navigation import GraphReadAction, NavigationActionType
from memory_graph.types import CausalTemporalOverlay, MemoryNode, RelationBelief

from .belief import (
    _action_can_resolve,
    _compare_uncertainty,
    _infer_missing_roles,
    _next_belief_id,
    _uncertainty_level,
    _verified_relations,
)
from .contracts import (
    Answerability,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    BeliefUpdateResult,
    RelationGrounding,
    RelationState,
)
from .continuous import ContinuousBeliefSmoother, ObservedIntervalSmoother


GTSAM_AVAILABLE = importlib.util.find_spec("gtsam") is not None
_EPSILON = 1e-4
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


@dataclass(frozen=True)
class BinaryVariable:
    variable_id: str
    prior_true: float


@dataclass(frozen=True)
class UnaryFactor:
    factor_id: str
    variable_id: str
    likelihood_false: float
    likelihood_true: float
    source: str

    @property
    def variables(self) -> tuple[str, ...]:
        return (self.variable_id,)


@dataclass(frozen=True)
class PairwiseFactor:
    factor_id: str
    left_variable_id: str
    right_variable_id: str
    potential: tuple[tuple[float, float], tuple[float, float]]
    source: str

    @property
    def variables(self) -> tuple[str, ...]:
        return self.left_variable_id, self.right_variable_id


@dataclass(frozen=True)
class FactorInferenceResult:
    posteriors: dict[str, float]
    factor_sources: dict[str, tuple[str, ...]]
    variable_count: int
    factor_count: int
    iterations: int


class BinaryFactorGraph:
    """Small loopy sum-product engine for binary relation hypotheses."""

    def __init__(self) -> None:
        self.variables: dict[str, BinaryVariable] = {}
        self.factors: list[UnaryFactor | PairwiseFactor] = []

    def add_variable(self, variable_id: str, prior_true: float) -> None:
        self.variables[variable_id] = BinaryVariable(
            variable_id,
            _clamp_probability(prior_true),
        )

    def add_unary(
        self,
        factor_id: str,
        variable_id: str,
        likelihood_false: float,
        likelihood_true: float,
        source: str,
    ) -> None:
        self.factors.append(
            UnaryFactor(
                factor_id,
                variable_id,
                max(_EPSILON, likelihood_false),
                max(_EPSILON, likelihood_true),
                source,
            )
        )

    def add_pairwise(
        self,
        factor_id: str,
        left: str,
        right: str,
        potential: tuple[tuple[float, float], tuple[float, float]],
        source: str,
    ) -> None:
        self.factors.append(PairwiseFactor(factor_id, left, right, potential, source))

    def infer(self, *, iterations: int = 8) -> FactorInferenceResult:
        if iterations <= 0:
            raise ValueError("factor graph iterations must be positive")
        neighbors: dict[str, list[int]] = {name: [] for name in self.variables}
        sources: dict[str, set[str]] = {name: {"relation_prior"} for name in self.variables}
        for index, factor in enumerate(self.factors):
            for variable_id in factor.variables:
                neighbors[variable_id].append(index)
                sources[variable_id].add(factor.source)

        factor_to_variable = {
            (index, variable_id): (0.5, 0.5)
            for index, factor in enumerate(self.factors)
            for variable_id in factor.variables
        }
        variable_to_factor: dict[tuple[str, int], tuple[float, float]] = {}
        for _ in range(iterations):
            for variable_id, variable in self.variables.items():
                prior = (1.0 - variable.prior_true, variable.prior_true)
                for target_factor in neighbors[variable_id]:
                    message = [prior[0], prior[1]]
                    for source_factor in neighbors[variable_id]:
                        if source_factor == target_factor:
                            continue
                        incoming = factor_to_variable[(source_factor, variable_id)]
                        message[0] *= incoming[0]
                        message[1] *= incoming[1]
                    variable_to_factor[(variable_id, target_factor)] = _normalize(message)

            updated: dict[tuple[int, str], tuple[float, float]] = {}
            for index, factor in enumerate(self.factors):
                if isinstance(factor, UnaryFactor):
                    updated[(index, factor.variable_id)] = _normalize(
                        (factor.likelihood_false, factor.likelihood_true)
                    )
                    continue
                left_message = variable_to_factor[(factor.left_variable_id, index)]
                right_message = variable_to_factor[(factor.right_variable_id, index)]
                left_out = (
                    factor.potential[0][0] * right_message[0]
                    + factor.potential[0][1] * right_message[1],
                    factor.potential[1][0] * right_message[0]
                    + factor.potential[1][1] * right_message[1],
                )
                right_out = (
                    factor.potential[0][0] * left_message[0]
                    + factor.potential[1][0] * left_message[1],
                    factor.potential[0][1] * left_message[0]
                    + factor.potential[1][1] * left_message[1],
                )
                updated[(index, factor.left_variable_id)] = _normalize(left_out)
                updated[(index, factor.right_variable_id)] = _normalize(right_out)
            factor_to_variable = updated

        posteriors: dict[str, float] = {}
        for variable_id, variable in self.variables.items():
            marginal = [1.0 - variable.prior_true, variable.prior_true]
            for factor_index in neighbors[variable_id]:
                incoming = factor_to_variable[(factor_index, variable_id)]
                marginal[0] *= incoming[0]
                marginal[1] *= incoming[1]
            posteriors[variable_id] = _normalize(marginal)[1]
        return FactorInferenceResult(
            posteriors=posteriors,
            factor_sources={
                name: tuple(sorted(values)) for name, values in sources.items()
            },
            variable_count=len(self.variables),
            factor_count=len(self.factors),
            iterations=iterations,
        )


class FactorGraphBeliefBackend:
    """Maintain exploration belief with global factor propagation and smoothing."""

    name = "hybrid_factor_graph/v0.1"

    def __init__(
        self,
        *,
        inference_iterations: int = 8,
        continuous_smoother: ContinuousBeliefSmoother | None = None,
    ) -> None:
        if inference_iterations <= 0:
            raise ValueError("inference_iterations must be positive")
        self.inference_iterations = inference_iterations
        self.continuous_smoother = continuous_smoother or ObservedIntervalSmoother()

    def initialize(
        self,
        question: str,
        overlay: CausalTemporalOverlay,
        *,
        seed_evidence: tuple[str, ...] = (),
        missing_roles: tuple[str, ...] | None = None,
        graph_read_budget: int = 8,
    ) -> BeliefSnapshot:
        _validate_evidence(seed_evidence, overlay)
        if graph_read_budget < 0:
            raise ValueError("graph_read_budget must be non-negative")
        acquired = tuple(dict.fromkeys(seed_evidence))
        roles = (
            _infer_missing_roles(question)
            if missing_roles is None
            else tuple(dict.fromkeys(missing_roles))
        )
        return self._snapshot(
            overlay=overlay,
            question=question,
            acquired=acquired,
            frontier=acquired[-1:],
            missing_roles=roles,
            remaining_graph_reads=graph_read_budget,
            step=0,
            belief_id=f"{overlay.overlay_id}:factor-belief:0",
        )

    def update(
        self,
        belief: BeliefSnapshot,
        action: GraphReadAction,
        observations: list[MemoryNode],
        overlay: CausalTemporalOverlay,
    ) -> BeliefUpdateResult:
        known = _known_node_ids(overlay)
        observed_ids = tuple(
            dict.fromkeys(node.node_id for node in observations if node.node_id in known)
        )
        acquired = tuple(dict.fromkeys(belief.acquired_evidence + observed_ids))
        action_role = _action_role(action.action_type)
        resolved_roles = (
            (action_role,)
            if observed_ids
            and action_role in belief.missing_roles
            and _action_can_resolve(action, overlay)
            else ()
        )
        missing_roles = tuple(
            role for role in belief.missing_roles if role not in set(resolved_roles)
        )
        next_step = belief.step + 1
        next_belief = self._snapshot(
            overlay=overlay,
            question=belief.question,
            acquired=acquired,
            frontier=observed_ids or belief.frontier,
            missing_roles=missing_roles,
            remaining_graph_reads=max(
                0,
                belief.remaining_graph_reads
                - (0 if action.action_type is NavigationActionType.STOP else 1),
            ),
            step=next_step,
            belief_id=_next_belief_id(overlay.overlay_id, next_step, action),
        )
        previous_states = {state.edge_id: state for state in belief.relation_states}
        relation_updates = tuple(
            state.edge_id
            for state in next_belief.relation_states
            if state.edge_id not in previous_states
            or state.grounding != previous_states[state.edge_id].grounding
            or state.posterior_probabilities
            != previous_states[state.edge_id].posterior_probabilities
        )
        new_contradictions = tuple(
            edge_id
            for edge_id in next_belief.contradictions
            if edge_id not in set(belief.contradictions)
        )
        return BeliefUpdateResult(
            belief=next_belief,
            delta=BeliefDeltaDescriptor(
                resolved_roles=resolved_roles,
                relation_updates=relation_updates,
                contradiction_updates=new_contradictions,
                uncertainty_change=_compare_uncertainty(
                    belief.uncertainty,
                    next_belief.uncertainty,
                ),
                answerability_after=next_belief.answerability,
                predicted_only=False,
            ),
        )

    def _snapshot(
        self,
        *,
        overlay: CausalTemporalOverlay,
        question: str,
        acquired: tuple[str, ...],
        frontier: tuple[str, ...],
        missing_roles: tuple[str, ...],
        remaining_graph_reads: int,
        step: int,
        belief_id: str,
    ) -> BeliefSnapshot:
        acquired_set = set(acquired)
        inference = _build_graph(overlay, acquired_set).infer(
            iterations=self.inference_iterations
        )
        conflicts, blocked, conflict_sources = _global_conflicts(
            overlay,
            acquired_set,
            inference.posteriors,
        )
        relation_states = tuple(
            _factor_relation_state(
                edge,
                acquired_set,
                inference,
                blocked,
                conflict_sources,
            )
            for edge in overlay.relations + overlay.l1_structural_relations
        )
        priority = _priority_edges(relation_states, missing_roles, blocked)
        continuous = self.continuous_smoother.update(
            overlay,
            frozenset(acquired_set),
        )
        answerability = (
            Answerability.READY
            if acquired and not missing_roles and not conflicts
            else Answerability.NOT_READY
        )
        backend_ref = (
            f"sum-product:{inference.variable_count}v:{inference.factor_count}f:"
            f"{inference.iterations}i:continuous={continuous.backend_name}:"
            f"{continuous.backend_ref}"
        )
        return BeliefSnapshot(
            belief_id=belief_id,
            backend_name=self.name,
            backend_ref=backend_ref,
            question=question,
            acquired_evidence=acquired,
            frontier=frontier,
            missing_roles=missing_roles,
            contradictions=tuple(sorted(conflicts)),
            relation_states=relation_states,
            priority_edge_ids=priority,
            blocked_edge_ids=tuple(sorted(blocked)),
            uncertainty=_uncertainty_level(missing_roles, acquired),
            answerability=answerability,
            remaining_graph_reads=remaining_graph_reads,
            step=step,
        )


def _build_graph(
    overlay: CausalTemporalOverlay,
    acquired: set[str],
) -> BinaryFactorGraph:
    graph = BinaryFactorGraph()
    edges = overlay.relations + overlay.l1_structural_relations
    variables_by_pair: dict[tuple[str, str], dict[str, list[str]]] = {}
    for edge in edges:
        by_relation: dict[str, str] = {}
        pair = tuple(sorted((edge.src, edge.dst)))
        for relation, prior in edge.relation_probabilities.items():
            variable_id = _variable_id(edge.edge_id, relation)
            graph.add_variable(variable_id, prior)
            by_relation[relation] = variable_id
            variables_by_pair.setdefault(pair, {}).setdefault(relation, []).append(
                variable_id
            )
            if edge.status.value == "deterministic":
                graph.add_unary(
                    f"deterministic:{variable_id}",
                    variable_id,
                    _EPSILON,
                    1.0,
                    "deterministic_relation",
                )
            if relation in _verified_relations(edge):
                graph.add_unary(
                    f"verified:{variable_id}",
                    variable_id,
                    _EPSILON,
                    1.0,
                    "hard_verified_relation",
                )
            if (
                relation == "contradicts"
                and {edge.src, edge.dst} <= acquired
                and _relation_is_admitted(edge, relation)
            ):
                graph.add_unary(
                    f"observed-contradiction:{variable_id}",
                    variable_id,
                    _EPSILON,
                    1.0,
                    "grounded_contradiction",
                )
            if relation in {"explains", "enables"} and not _temporally_valid(
                edge, overlay
            ):
                graph.add_unary(
                    f"invalid-direction:{variable_id}",
                    variable_id,
                    1.0,
                    _EPSILON,
                    "temporal_direction_constraint",
                )
        temporal_variables = [
            variable_id
            for relation, variable_id in by_relation.items()
            if relation in {"before", "overlaps", "during"}
        ]
        _add_mutex_group(graph, temporal_variables, f"temporal-mutex:{edge.edge_id}")

    for pair, by_relation in variables_by_pair.items():
        contradicts = by_relation.get("contradicts", [])
        positives = [
            variable_id
            for relation, variables in by_relation.items()
            if relation != "contradicts"
            for variable_id in variables
        ]
        for contradiction in contradicts:
            for positive in positives:
                graph.add_pairwise(
                    f"contradiction-mutex:{contradiction}:{positive}",
                    contradiction,
                    positive,
                    _mutex_potential(),
                    "contradiction_incompatibility",
                )
        identity = [
            variable_id
            for relation in _IDENTITY
            for variable_id in by_relation.get(relation, [])
        ]
        for transition in by_relation.get("state_transition", []):
            for identity_variable in identity:
                graph.add_pairwise(
                    f"state-implies-identity:{transition}:{identity_variable}",
                    transition,
                    identity_variable,
                    _implication_potential(),
                    "state_requires_identity",
                )
        temporal_before = (
            by_relation.get("before", []) + by_relation.get("temporal_next", [])
        )
        for causal_relation in ("explains", "enables"):
            for causal in by_relation.get(causal_relation, []):
                for before in temporal_before:
                    graph.add_pairwise(
                        f"causal-implies-before:{causal}:{before}",
                        causal,
                        before,
                        _implication_potential(),
                        "causal_requires_temporal_precedence",
                    )
    return graph


def _factor_relation_state(
    edge: RelationBelief,
    acquired: set[str],
    inference: FactorInferenceResult,
    blocked: set[str],
    conflict_sources: dict[str, set[str]],
) -> RelationState:
    observed_count = int(edge.src in acquired) + int(edge.dst in acquired)
    verified = _verified_relations(edge)
    if edge.edge_id in blocked:
        grounding = RelationGrounding.CONTRADICTED
    elif observed_count == 0:
        grounding = RelationGrounding.UNSEEN
    elif observed_count == 1:
        grounding = RelationGrounding.PARTIAL
    elif (
        "contradicts" in edge.relation_probabilities
        and _relation_is_admitted(edge, "contradicts")
    ):
        grounding = RelationGrounding.CONTRADICTED
    elif verified or edge.status.value == "deterministic":
        grounding = RelationGrounding.VERIFIED
    else:
        grounding = RelationGrounding.ENDPOINTS_OBSERVED
    posterior = {
        relation: inference.posteriors[_variable_id(edge.edge_id, relation)]
        for relation in edge.relation_probabilities
    }
    factor_sources = {
        source
        for relation in edge.relation_probabilities
        for source in inference.factor_sources[_variable_id(edge.edge_id, relation)]
    }
    factor_sources.update(conflict_sources.get(edge.edge_id, set()))
    return RelationState(
        edge_id=edge.edge_id,
        src=edge.src,
        dst=edge.dst,
        relation_probabilities=tuple(sorted(edge.relation_probabilities.items())),
        posterior_probabilities=tuple(sorted(posterior.items())),
        correlation_features=tuple(sorted(edge.features.items())),
        verified_relations=verified,
        factor_sources=tuple(sorted(factor_sources)),
        calibration_status=edge.status.value,
        grounding=grounding,
    )


def _global_conflicts(
    overlay: CausalTemporalOverlay,
    acquired: set[str],
    posteriors: dict[str, float],
) -> tuple[set[str], set[str], dict[str, set[str]]]:
    edges = overlay.relations + overlay.l1_structural_relations
    conflicts: set[str] = set()
    blocked: set[str] = set()
    sources: dict[str, set[str]] = {}
    grounded_contradictions = [
        edge
        for edge in edges
        if "contradicts" in edge.relation_probabilities
        and {edge.src, edge.dst} <= acquired
        and _relation_is_admitted(edge, "contradicts")
        and posteriors[_variable_id(edge.edge_id, "contradicts")] >= 0.5
    ]
    for edge in grounded_contradictions:
        conflicts.add(edge.edge_id)
        blocked.add(edge.edge_id)
        sources.setdefault(edge.edge_id, set()).add("grounded_contradiction")

    verified_identity = [
        edge
        for edge in edges
        if set(_verified_relations(edge)) & _IDENTITY
        and {edge.src, edge.dst} <= acquired
    ]
    components = _identity_components(verified_identity)
    for contradiction in grounded_contradictions:
        if components.get(contradiction.src) != components.get(contradiction.dst):
            continue
        component = components.get(contradiction.src)
        if component is None:
            continue
        for identity_edge in verified_identity:
            if components.get(identity_edge.src) != component:
                continue
            conflicts.add(identity_edge.edge_id)
            blocked.add(identity_edge.edge_id)
            sources.setdefault(identity_edge.edge_id, set()).add(
                "identity_component_contradiction"
            )

    temporal_edges = [
        edge
        for edge in edges
        if set(edge.relation_probabilities) & {"before", "temporal_next"}
        and {edge.src, edge.dst} <= acquired
        and (
            edge.status.value == "deterministic"
            or bool(set(_verified_relations(edge)) & {"before", "temporal_next"})
        )
    ]
    for edge_id in _temporal_cycle_edge_ids(temporal_edges):
        conflicts.add(edge_id)
        blocked.add(edge_id)
        sources.setdefault(edge_id, set()).add("temporal_cycle")
    return conflicts, blocked, sources


def _priority_edges(
    relation_states: tuple[RelationState, ...],
    missing_roles: tuple[str, ...],
    blocked: set[str],
) -> tuple[str, ...]:
    required_relations = set().union(
        *(_ROLE_RELATIONS.get(role, set()) for role in missing_roles)
    ) if missing_roles else set()
    priority = []
    for state in relation_states:
        if state.edge_id in blocked:
            priority.append(state.edge_id)
            continue
        names = set(dict(state.relation_probabilities))
        if required_relations and not names & required_relations:
            continue
        if state.grounding in {
            RelationGrounding.PARTIAL,
            RelationGrounding.ENDPOINTS_OBSERVED,
        }:
            priority.append(state.edge_id)
    return tuple(dict.fromkeys(priority))


def _identity_components(edges: list[RelationBelief]) -> dict[str, str]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for edge in edges:
        union(edge.src, edge.dst)
    return {node: find(node) for node in parent}


def _temporal_cycle_edge_ids(edges: list[RelationBelief]) -> set[str]:
    adjacency: dict[str, list[tuple[str, str]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.src, []).append((edge.dst, edge.edge_id))
    visiting: set[str] = set()
    visited: set[str] = set()
    path_nodes: list[str] = []
    path_edges: list[str] = []
    cycle_edges: set[str] = set()

    def visit(node: str) -> None:
        visiting.add(node)
        path_nodes.append(node)
        for target, edge_id in adjacency.get(node, []):
            if target in visiting:
                cycle_start = path_nodes.index(target)
                cycle_edges.add(edge_id)
                cycle_edges.update(path_edges[cycle_start:])
                continue
            if target in visited:
                continue
            path_edges.append(edge_id)
            visit(target)
            path_edges.pop()
        path_nodes.pop()
        visiting.remove(node)
        visited.add(node)

    for node in tuple(adjacency):
        if node not in visited:
            visit(node)
    return cycle_edges


def _relation_is_admitted(edge: RelationBelief, relation: str) -> bool:
    return edge.status.value == "deterministic" or relation in _verified_relations(edge)


def _add_mutex_group(
    graph: BinaryFactorGraph,
    variables: list[str],
    prefix: str,
) -> None:
    for index, left in enumerate(variables):
        for right in variables[index + 1 :]:
            graph.add_pairwise(
                f"{prefix}:{left}:{right}",
                left,
                right,
                _mutex_potential(),
                "mutually_exclusive_relations",
            )


def _mutex_potential() -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 1.0), (1.0, _EPSILON))


def _implication_potential() -> tuple[tuple[float, float], tuple[float, float]]:
    return ((1.0, 1.0), (_EPSILON, 1.0))


def _temporally_valid(
    edge: RelationBelief,
    overlay: CausalTemporalOverlay,
) -> bool:
    by_id = {node.node_id: node for node in overlay.atomic_events}
    src, dst = by_id.get(edge.src), by_id.get(edge.dst)
    if src is None or dst is None:
        return False
    return src.time_span.start_s <= dst.time_span.start_s


def _validate_evidence(
    evidence: Iterable[str],
    overlay: CausalTemporalOverlay,
) -> None:
    unknown = set(evidence) - _known_node_ids(overlay)
    if unknown:
        raise ValueError(f"seed_evidence references unknown nodes: {sorted(unknown)}")


def _known_node_ids(overlay: CausalTemporalOverlay) -> set[str]:
    return {
        node.node_id for node in overlay.atomic_events + overlay.l1_observations
    }


def _action_role(action_type: NavigationActionType) -> str | None:
    return {
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
    }.get(action_type)


def _variable_id(edge_id: str, relation: str) -> str:
    return f"relation::{edge_id}::{relation}"


def _normalize(values: Iterable[float]) -> tuple[float, float]:
    first, second = (max(_EPSILON, float(value)) for value in values)
    total = first + second
    return first / total, second / total


def _clamp_probability(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("factor prior must be finite")
    return max(_EPSILON, min(1.0 - _EPSILON, float(value)))
