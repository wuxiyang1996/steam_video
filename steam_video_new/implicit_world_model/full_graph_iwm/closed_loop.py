"""Real-read closed loop and hidden CG-Bench coverage scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import hashlib
import json
from collections import deque
from time import monotonic
from typing import Any, Protocol, Sequence

from .action_compiler import GraphActionCompiler, execute_graph_action
from .contracts import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    FrontierChange,
    FullGraphPlanDecision,
    LegalGraphAction,
    ProgressChange,
    RetainedEvidenceGraph,
)


TERMINAL_KINDS = {ActionKind.STOP, ActionKind.ANSWER, ActionKind.ABSTAIN}


class GraphPlanner(Protocol):
    def plan(
        self, belief: CursorBeliefState, graph: RetainedEvidenceGraph
    ) -> FullGraphPlanDecision: ...


class RealBeliefUpdater(Protocol):
    def update(self, belief: CursorBeliefState, observation: Any) -> CursorBeliefState: ...


@dataclass(frozen=True)
class ClueInterval:
    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s:
            raise ValueError("invalid clue interval")


class HiddenClueCoverageEvaluator:
    """Score real reads without exposing clue locations to the planner."""

    def __init__(self, clues: Sequence[ClueInterval]) -> None:
        if not clues:
            raise ValueError("at least one clue interval is required")
        self.clues = tuple(clues)
        self.covered_indices: set[int] = set()

    def score_read(
        self,
        action: LegalGraphAction,
        graph: RetainedEvidenceGraph,
    ) -> tuple[CategoricalBeliefDelta, EvidenceOutcome, tuple[int, ...]]:
        if action.target_id is None:
            matched: set[int] = set()
        else:
            node = graph.node_by_id[action.target_id]
            matched = {
                index
                for index, clue in enumerate(self.clues)
                if _overlaps(
                    (node.time_span.start_s, node.time_span.end_s),
                    (clue.start_s, clue.end_s),
                )
            }
        newly_covered = matched - self.covered_indices
        self.covered_indices.update(matched)
        complete = len(self.covered_indices) == len(self.clues)
        return (
            CategoricalBeliefDelta(
                progress=(
                    ProgressChange.ADVANCED
                    if newly_covered
                    else ProgressChange.UNCHANGED
                ),
                answerability_after=(
                    AnswerabilityState.READY
                    if complete
                    else AnswerabilityState.NOT_READY
                ),
                frontier_change=FrontierChange.UNCHANGED,
                predicted_only=False,
            ),
            EvidenceOutcome.SUPPORT if newly_covered else EvidenceOutcome.INCONCLUSIVE,
            tuple(sorted(newly_covered)),
        )

    @property
    def complete(self) -> bool:
        return len(self.covered_indices) == len(self.clues)


def run_real_read_closed_loop(
    *,
    case_id: str,
    question: str,
    graph: RetainedEvidenceGraph,
    clue_intervals: Sequence[ClueInterval],
    planner: GraphPlanner,
    read_budget: int,
    arm: str,
    max_decisions: int | None = None,
    initial_missing_roles: Sequence[str] = (),
    belief_updater: RealBeliefUpdater | None = None,
) -> dict[str, Any]:
    """Plan, execute one real read, expose it, and replan.

    Hidden clue overlap creates evaluation/training labels after execution.  It
    is never copied into ``CursorBeliefState`` and therefore cannot influence a
    later planner decision.
    """

    if read_budget < 1:
        raise ValueError("read_budget must be positive")
    evaluator = HiddenClueCoverageEvaluator(clue_intervals)
    belief = CursorBeliefState(
        belief_id=f"belief:{case_id}:{arm}:initial",
        question=question,
        missing_roles=tuple(initial_missing_roles),
        remaining_reads=read_budget,
    )
    decision_limit = max_decisions or max(4, read_budget * 3 + 2)
    steps: list[dict[str, Any]] = []
    seen_states: set[tuple[str | None, tuple[str, ...], str]] = set()
    started = monotonic()
    termination = "decision_limit"
    for _ in range(decision_limit):
        decision = planner.plan(belief, graph)
        action = decision.selected_action
        state_key = (
            belief.current_node_id,
            belief.acquired_evidence,
            action.action_id,
        )
        if state_key in seen_states:
            termination = "repeated_nonprogress_state"
            break
        seen_states.add(state_key)
        step: dict[str, Any] = {
            "belief_before": _belief_audit(belief),
            "decision": _jsonable(decision),
            "selected_action": _jsonable(action),
        }
        if action.kind in TERMINAL_KINDS:
            termination = action.kind.value
            steps.append(step)
            break
        execution = execute_graph_action(action, belief, graph)
        belief = execution.updated_belief
        belief_before_correction = belief
        if belief_updater is not None and execution.observation is not None:
            belief = belief_updater.update(belief, execution.observation)
        realized, outcome, newly_covered = evaluator.score_read(action, graph)
        step.update(
            {
                "real_observation": (
                    _node_observation(execution.observation)
                    if execution.observation is not None
                    else None
                ),
                "belief_after_execution_before_correction": _belief_audit(
                    belief_before_correction
                ),
                "belief_correction": {
                    "applied": belief_updater is not None,
                    "source": "executed_real_observation_only",
                    "hidden_evaluator_feedback": False,
                },
                "realized_label_evaluator_only": {
                    "observation_outcome": outcome.value,
                    "belief_delta": _jsonable(realized),
                    "newly_covered_clue_indices": list(newly_covered),
                    "fed_back_to_planner": False,
                },
                "belief_after_real_read": _belief_audit(belief),
            }
        )
        steps.append(step)
        if belief.remaining_reads == 0:
            termination = "read_budget_exhausted"
            break
    elapsed = monotonic() - started
    return _finalize_run(
        case_id=case_id,
        arm=arm,
        graph=graph,
        belief=belief,
        evaluator=evaluator,
        steps=steps,
        termination=termination,
        elapsed_s=elapsed,
    )


def run_oracle_clue_ceiling(
    *,
    case_id: str,
    question: str,
    graph: RetainedEvidenceGraph,
    clue_intervals: Sequence[ClueInterval],
    read_budget: int,
) -> dict[str, Any]:
    """Evaluator-only upper bound; hidden clue locations select the read."""

    evaluator = HiddenClueCoverageEvaluator(clue_intervals)
    belief = CursorBeliefState(
        belief_id=f"belief:{case_id}:oracle:initial",
        question=question,
        remaining_reads=read_budget,
    )
    steps: list[dict[str, Any]] = []
    started = monotonic()
    termination = "oracle_no_covering_action"
    # Navigation may require cursor-only backtracks between real reads.  Bound
    # decisions separately from the evidence-read budget so the oracle remains
    # finite without incorrectly treating every graph hop as a read.
    decision_limit = max(4, read_budget * 3 + 2)
    for _ in range(decision_limit):
        actions = GraphActionCompiler().compile(belief, graph)
        action = _oracle_action(actions, graph, evaluator)
        if action is None:
            break
        before = belief
        execution = execute_graph_action(action, belief, graph)
        belief = execution.updated_belief
        realized, outcome, newly_covered = evaluator.score_read(action, graph)
        steps.append(
            {
                "belief_before": _belief_audit(before),
                "selected_action": _jsonable(action),
                "decision": {
                    "planning_status": "oracle_hidden_alignment_evaluator_only",
                    "top_k_applied": False,
                },
                "real_observation": _node_observation(execution.observation),
                "realized_label_evaluator_only": {
                    "observation_outcome": outcome.value,
                    "belief_delta": _jsonable(realized),
                    "newly_covered_clue_indices": list(newly_covered),
                    "fed_back_to_planner": False,
                },
                "belief_after_real_read": _belief_audit(belief),
            }
        )
        if evaluator.complete:
            termination = "oracle_coverage_complete"
            break
        if belief.remaining_reads == 0:
            termination = "read_budget_exhausted"
            break
    return _finalize_run(
        case_id=case_id,
        arm="oracle_clue_ceiling",
        graph=graph,
        belief=belief,
        evaluator=evaluator,
        steps=steps,
        termination=termination,
        elapsed_s=monotonic() - started,
    )


def graph_fingerprint(graph: RetainedEvidenceGraph) -> str:
    payload = {
        "graph_id": graph.graph_id,
        "capacity": graph.capacity,
        "nodes": [
            {
                "node_id": node.node_id,
                "start_s": node.time_span.start_s,
                "end_s": node.time_span.end_s,
            }
            for node in graph.nodes
        ],
        "temporal_edges": [_jsonable(edge) for edge in graph.temporal_edges],
        "correlation_edges": [edge.to_dict() for edge in graph.correlation_edges],
        "candidate_edges": [edge.to_dict() for edge in graph.candidate_edges],
        "verified_relations": [edge.to_dict() for edge in graph.verified_relations],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def action_divergence(
    reference: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    left = _action_sequence(reference)
    right = _action_sequence(candidate)
    first = next(
        (
            index
            for index in range(max(len(left), len(right)))
            if (left[index] if index < len(left) else None)
            != (right[index] if index < len(right) else None)
        ),
        None,
    )
    return {
        "reference_arm": reference["arm"],
        "candidate_arm": candidate["arm"],
        "action_diverged": first is not None,
        "first_divergence_step": first,
        "reference_action_ids": left,
        "candidate_action_ids": right,
    }


def _finalize_run(
    *,
    case_id: str,
    arm: str,
    graph: RetainedEvidenceGraph,
    belief: CursorBeliefState,
    evaluator: HiddenClueCoverageEvaluator,
    steps: list[dict[str, Any]],
    termination: str,
    elapsed_s: float,
) -> dict[str, Any]:
    reads = sum(step.get("real_observation") is not None for step in steps)
    covered = len(evaluator.covered_indices)
    clue_count = len(evaluator.clues)
    delayed = evaluator.complete and clue_count > 1 and reads > 1
    return {
        "schema_version": "steam-full-graph-iwm-closed-loop-run/v0.1",
        "case_id": case_id,
        "arm": arm,
        "graph_fingerprint": graph_fingerprint(graph),
        "steps": steps,
        "termination": termination,
        "metrics": {
            "clue_coverage_complete": evaluator.complete,
            "covered_clue_count": covered,
            "clue_count": clue_count,
            "clue_recall": covered / clue_count,
            "real_read_count": reads,
            "read_efficiency": covered / reads if reads else None,
            "abstained": termination == ActionKind.ABSTAIN.value,
            "delayed_reasoning_success": delayed,
            "latency_s": elapsed_s,
            "answer_accuracy": None,
            "answer_accuracy_status": "not_evaluated_without_terminal_answer_head",
        },
        "final_visible_belief": _belief_audit(belief),
        "hidden_evaluator_feedback_to_planner": False,
    }


def _oracle_action(
    actions: Sequence[LegalGraphAction],
    graph: RetainedEvidenceGraph,
    evaluator: HiddenClueCoverageEvaluator,
) -> LegalGraphAction | None:
    uncovered = [
        clue
        for index, clue in enumerate(evaluator.clues)
        if index not in evaluator.covered_indices
    ]
    priority = {
        ActionKind.START_AT: 0,
        ActionKind.FOLLOW_CORRELATION: 1,
        ActionKind.TEMPORAL_FORWARD: 2,
        ActionKind.TEMPORAL_BACKWARD: 3,
    }
    covering_nodes = {
        node.node_id
        for node in graph.nodes
        if any(
            _overlaps(
                (node.time_span.start_s, node.time_span.end_s),
                (clue.start_s, clue.end_s),
            )
            for clue in uncovered
        )
    }
    if not covering_nodes:
        return None
    adjacency = _navigation_adjacency(graph)
    candidates: list[tuple[int, bool, int, str, LegalGraphAction]] = []
    for action in actions:
        if action.kind in TERMINAL_KINDS or action.target_id is None:
            continue
        distance = _shortest_hops(action.target_id, covering_nodes, adjacency)
        if distance is not None:
            candidates.append(
                (
                    distance,
                    not action.reads_evidence,
                    priority.get(action.kind, 99),
                    action.action_id,
                    action,
                )
            )
    return min(candidates, default=None, key=lambda item: item[:-1])[-1] if candidates else None


def _navigation_adjacency(graph: RetainedEvidenceGraph) -> dict[str, set[str]]:
    """Build the evaluator-only directed reachability graph.

    Temporal edges are traversable in both directions by the action compiler;
    soft and categorical L1.5 edges retain their declared directionality.
    """

    adjacency = {node.node_id: set() for node in graph.nodes}
    for edge in graph.temporal_edges:
        adjacency[edge.src].add(edge.dst)
        adjacency[edge.dst].add(edge.src)
    for edge in graph.correlation_edges:
        if edge.evidence_refs and edge.src_to_dst_affinity > 0.0:
            adjacency[edge.src].add(edge.dst)
        if edge.evidence_refs and edge.dst_to_src_affinity > 0.0:
            adjacency[edge.dst].add(edge.src)
    for edge in graph.candidate_edges:
        if edge.permits(edge.src, edge.dst):
            adjacency[edge.src].add(edge.dst)
        if edge.permits(edge.dst, edge.src):
            adjacency[edge.dst].add(edge.src)
    return adjacency


def _shortest_hops(
    source: str,
    targets: set[str],
    adjacency: dict[str, set[str]],
) -> int | None:
    if source in targets:
        return 0
    queue: deque[tuple[str, int]] = deque([(source, 0)])
    visited = {source}
    while queue:
        node_id, distance = queue.popleft()
        for neighbor in adjacency.get(node_id, ()):
            if neighbor in visited:
                continue
            if neighbor in targets:
                return distance + 1
            visited.add(neighbor)
            queue.append((neighbor, distance + 1))
    return None


def _action_sequence(run: dict[str, Any]) -> list[str]:
    return [
        str(step["selected_action"]["action_id"])
        for step in run.get("steps") or []
        if step.get("real_observation") is not None
    ]


def _belief_audit(belief: CursorBeliefState) -> dict[str, Any]:
    return {
        "belief_id": belief.belief_id,
        "current_node_id": belief.current_node_id,
        "acquired_evidence": list(belief.acquired_evidence),
        "imagined_evidence": list(belief.imagined_evidence),
        "missing_roles": list(belief.missing_roles),
        "contradictions": list(belief.contradictions),
        "answerability": belief.answerability.value,
        "remaining_reads": belief.remaining_reads,
        "step": belief.step,
    }


def _node_observation(node: Any) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": node.node_id,
        "node_type": node.node_type,
        "time_span": {
            "start_s": node.time_span.start_s,
            "end_s": node.time_span.end_s,
        },
        "evidence_value": node.text or None,
        "embedding_ref": (
            _jsonable(node.embedding_ref) if node.embedding_ref is not None else None
        ),
        "provenance": node.provenance,
    }


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _jsonable(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
