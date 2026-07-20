"""Matched-checkpoint navigation ablation over locked evidence case sets."""

from __future__ import annotations

from collections import deque
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from memory_graph.navigation_ablation import (
    _add_native_l1_adjacency,
    _event_adjacency,
    _lexical_score,
    _verified_dependency_adjacency,
)

from .belief import FactorizedBeliefBackend
from .contracts import BeliefBackend
from .contracts import Answerability, BeliefDeltaDescriptor, BeliefUpdateResult
from .factor_graph import FactorGraphBeliefBackend
from .overlay_io import load_overlay_artifact, overlay_from_dict
from .planner import ClosedLoopNavigator, PersistedGraphReadExecutor, PreferenceOnlyPlanner
from .preference_data import require_valid_navigation_case_set
from .world_model import RuleBasedObservationBeliefModel, RuleBasedTrajectoryPreferenceModel


MATCHED_STRATEGIES = (
    "semantic_only",
    "event_only",
    "native_l1_candidate",
    "verified_dependency",
    "factorized_direct_preference",
    "factor_graph_direct_preference",
    "factor_graph_rule_world_model_lookahead",
    "factor_graph_frozen_posterior_lookahead",
    "factor_graph_shuffled_relations",
    "oracle",
)


def evaluate_matched_navigation(
    case_set: dict[str, Any],
    *,
    case_root: Path,
    allow_ai_provisional: bool = False,
) -> dict[str, Any]:
    """Evaluate every policy from the same seed checkpoint and read-call budget."""

    require_valid_navigation_case_set(case_set)
    status = str(case_set.get("annotation_status") or "")
    if status != "human_locked" and not (
        status == "ai_provisional" and allow_ai_provisional
    ):
        raise ValueError(
            "matched ablation requires a human_locked case set; "
            "AI-provisional evaluation must be explicitly enabled"
        )
    rows: dict[str, list[dict[str, Any]]] = {
        strategy: [] for strategy in MATCHED_STRATEGIES
    }
    for case in case_set["cases"]:
        overlay_path = _resolve_path(case_root, str(case["overlay_path"]))
        loaded = load_overlay_artifact(overlay_path)
        overlay = loaded.overlay
        if overlay.overlay_id != case["overlay_id"]:
            raise ValueError(
                f"case {case['case_id']} overlay_id mismatch: "
                f"{case['overlay_id']} != {overlay.overlay_id}"
            )
        known_events = {node.node_id for node in overlay.atomic_events}
        seeds = tuple(str(value) for value in case["seed_event_ids"])
        gold = {str(value) for value in case["gold_event_ids"]}
        unknown = (set(seeds) | gold) - known_events
        if unknown:
            raise ValueError(f"case {case['case_id']} references unknown events: {sorted(unknown)}")
        for strategy in MATCHED_STRATEGIES:
            if strategy in {
                "semantic_only",
                "event_only",
                "native_l1_candidate",
                "verified_dependency",
            }:
                acquired, first_action, calls = _run_static_strategy(
                    strategy,
                    case=case,
                    overlay=overlay,
                    case_root=case_root,
                )
                contradictions: tuple[str, ...] = ()
            elif strategy == "oracle":
                unseen_gold = sorted(gold - set(seeds))
                reads = unseen_gold[: int(case["graph_read_budget"])]
                acquired = tuple(dict.fromkeys(seeds + tuple(reads)))
                first_action = (
                    {"action_type": "oracle", "target_ids": reads[:1]}
                    if reads
                    else {"action_type": "stop", "target_ids": []}
                )
                calls = len(reads)
                contradictions = ()
            else:
                planning_overlay = (
                    _shuffled_relation_overlay(overlay, str(case["case_id"]))
                    if strategy == "factor_graph_shuffled_relations"
                    else overlay
                )
                acquired, first_action, calls, contradictions = _run_planner_strategy(
                    strategy,
                    case=case,
                    overlay=planning_overlay,
                )
            hits = gold & set(acquired)
            accepted = (
                _first_action_accepted(first_action, case["acceptable_first_actions"])
                if strategy not in {
                    "semantic_only",
                    "event_only",
                    "native_l1_candidate",
                    "verified_dependency",
                    "oracle",
                }
                else None
            )
            rows[strategy].append(
                {
                    "case_id": case["case_id"],
                    "tags": list(case.get("tags") or []),
                    "acquired_event_ids": list(acquired),
                    "gold_event_ids": sorted(gold),
                    "hit_count": len(hits),
                    "evidence_recall": len(hits) / len(gold),
                    "answerable": hits == gold and not contradictions,
                    "graph_read_calls": calls,
                    "first_action": first_action,
                    "first_action_accepted": accepted,
                    "contradictions": list(contradictions),
                }
            )
    summaries = {
        strategy: {
            **_summary(strategy_rows),
            "by_tag": _summaries_by_tag(strategy_rows),
        }
        for strategy, strategy_rows in rows.items()
    }
    diagnostics = _diagnostic_gates(summaries, status)
    return {
        "schema_version": "steam-matched-navigation-ablation/v0.1",
        "case_set_id": case_set["case_set_id"],
        "case_set_annotation_status": status,
        "case_count": len(case_set["cases"]),
        "strategies": summaries,
        "diagnostic_gates": diagnostics,
        "cases": rows,
        "semantics": {
            "matched_checkpoint": "all policies begin with the case seed_event_ids",
            "matched_budget": "graph_read_budget counts real retrieval calls, not imagined rollouts",
            "answerable": "all locked gold events acquired and no unresolved factor contradiction",
            "preference": "ordinal four-way comparison only; no numeric reward or utility",
            "world_model": "current lookahead row uses the transparent categorical rule baseline",
            "diagnostics": "frozen-posterior and deterministically shuffled-relation controls must not be reported as learned policies",
        },
    }


def _run_static_strategy(
    strategy: str,
    *,
    case: dict[str, Any],
    overlay: Any,
    case_root: Path,
) -> tuple[tuple[str, ...], dict[str, Any], int]:
    events = {node.node_id: node.to_dict() for node in overlay.atomic_events}
    question = str(case["question"])
    scores = _case_relevance_scores(case, case_root, overlay, events, question)
    ranked = sorted(
        events,
        key=lambda event_id: (
            -scores[event_id],
            float((events[event_id].get("time_span") or {}).get("start_s") or 0.0),
            event_id,
        ),
    )
    acquired = list(dict.fromkeys(str(value) for value in case["seed_event_ids"]))
    reads: list[str] = []
    budget = int(case["graph_read_budget"])
    if strategy == "semantic_only":
        for event_id in ranked:
            if event_id not in acquired:
                reads.append(event_id)
                acquired.append(event_id)
            if len(reads) >= budget:
                break
    else:
        overlay_dict = overlay.to_dict()
        adjacency = _event_adjacency(overlay_dict, verified_only=False)
        preferred: dict[str, set[str]] = {}
        if strategy == "native_l1_candidate":
            _add_native_l1_adjacency(preferred, overlay_dict, events)
        elif strategy == "verified_dependency":
            preferred = _verified_dependency_adjacency(overlay_dict)
        for node_id, neighbors in preferred.items():
            adjacency.setdefault(node_id, set()).update(neighbors)
        queue = deque(acquired)
        if not queue and ranked:
            first = ranked[0]
            acquired.append(first)
            reads.append(first)
            queue.append(first)
        while queue and len(reads) < budget:
            source = queue.popleft()
            neighbors = sorted(
                adjacency.get(source, ()),
                key=lambda event_id: (
                    0 if event_id in preferred.get(source, set()) else 1,
                    -scores.get(event_id, 0.0),
                    event_id,
                ),
            )
            for target in neighbors:
                if target not in events or target in acquired:
                    continue
                acquired.append(target)
                reads.append(target)
                queue.append(target)
                if len(reads) >= budget:
                    break
        for event_id in ranked:
            if len(reads) >= budget:
                break
            if event_id not in acquired:
                acquired.append(event_id)
                reads.append(event_id)
    first_action = {
        "action_type": strategy,
        "source_id": acquired[0] if acquired else None,
        "target_ids": reads[:1],
        "relation": None,
    }
    return tuple(acquired), first_action, len(reads)


def _run_planner_strategy(
    strategy: str,
    *,
    case: dict[str, Any],
    overlay: Any,
) -> tuple[tuple[str, ...], dict[str, Any], int, tuple[str, ...]]:
    backend: BeliefBackend
    if strategy == "factorized_direct_preference":
        backend = FactorizedBeliefBackend()
        horizon = 1
    elif strategy == "factor_graph_direct_preference":
        backend = FactorGraphBeliefBackend()
        horizon = 1
    elif strategy == "factor_graph_frozen_posterior_lookahead":
        backend = _FrozenPosteriorBackend()
        horizon = 2
    else:
        backend = FactorGraphBeliefBackend()
        horizon = 2
    belief = backend.initialize(
        str(case["question"]),
        overlay,
        seed_evidence=tuple(str(value) for value in case["seed_event_ids"]),
        missing_roles=tuple(str(value) for value in case["missing_roles"]),
        graph_read_budget=int(case["graph_read_budget"]),
    )
    planner = PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        RuleBasedTrajectoryPreferenceModel(),
        horizon=horizon,
    )
    run = ClosedLoopNavigator(
        backend,
        planner,
        PersistedGraphReadExecutor(),
    ).run(belief, overlay, max_steps=int(case["graph_read_budget"]) + 1)
    first_action = (
        _action_dict(run.steps[0].decision.selected_action)
        if run.steps
        else {"action_type": "stop", "source_id": None, "target_ids": [], "relation": None}
    )
    calls = sum(
        step.decision.selected_action.action_type.value != "stop" for step in run.steps
    )
    return (
        run.final_belief.acquired_evidence,
        first_action,
        calls,
        run.final_belief.contradictions,
    )


def _first_action_accepted(
    actual: dict[str, Any],
    expected_rows: list[dict[str, Any]],
) -> bool:
    for expected in expected_rows:
        if actual.get("action_type") != expected.get("action_type"):
            continue
        if "source_id" in expected and actual.get("source_id") != expected.get("source_id"):
            continue
        if "target_ids" in expected and set(actual.get("target_ids") or []) != set(
            expected.get("target_ids") or []
        ):
            continue
        if "relation" in expected and actual.get("relation") != expected.get("relation"):
            continue
        return True
    return False


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    accepted = [row["first_action_accepted"] for row in rows if row["first_action_accepted"] is not None]
    return {
        "case_count": count,
        "answer_accuracy": sum(bool(row["answerable"]) for row in rows) / count if count else None,
        "mean_evidence_recall": sum(float(row["evidence_recall"]) for row in rows) / count if count else None,
        "mean_graph_read_calls": sum(int(row["graph_read_calls"]) for row in rows) / count if count else None,
        "first_action_accuracy": sum(bool(value) for value in accepted) / len(accepted) if accepted else None,
    }


def _summaries_by_tag(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    tags = sorted({str(tag) for row in rows for tag in row.get("tags") or []})
    return {
        tag: _summary([row for row in rows if tag in (row.get("tags") or [])])
        for tag in tags
    }


def _diagnostic_gates(
    summaries: dict[str, dict[str, Any]],
    annotation_status: str,
) -> dict[str, Any]:
    direct = summaries["factor_graph_direct_preference"]["answer_accuracy"]
    lookahead = summaries["factor_graph_rule_world_model_lookahead"]["answer_accuracy"]
    frozen = summaries["factor_graph_frozen_posterior_lookahead"]["answer_accuracy"]
    shuffled = summaries["factor_graph_shuffled_relations"]["answer_accuracy"]
    gates = {
        "lookahead_beats_direct": lookahead is not None and direct is not None and lookahead > direct,
        "relation_shuffling_hurts": lookahead is not None and shuffled is not None and lookahead > shuffled,
        "posterior_correction_helps": lookahead is not None and frozen is not None and lookahead > frozen,
    }
    return {
        "annotation_status": annotation_status,
        "formal_result": annotation_status == "human_locked",
        "answer_accuracy_deltas": {
            "lookahead_minus_direct": lookahead - direct,
            "lookahead_minus_shuffled": lookahead - shuffled,
            "lookahead_minus_frozen_posterior": lookahead - frozen,
        },
        "gates": gates,
        "engineering_status": (
            "provisional_only"
            if annotation_status != "human_locked"
            else "go" if all(gates.values()) else "no_go"
        ),
    }


def _action_dict(action: Any) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "source_id": action.source_id,
        "target_ids": list(action.target_ids),
        "relation": action.relation,
    }


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _case_relevance_scores(
    case: dict[str, Any],
    case_root: Path,
    overlay: Any,
    events: dict[str, dict[str, Any]],
    question: str,
) -> dict[str, float]:
    query_ref = case.get("query_embedding_ref")
    if not isinstance(query_ref, dict):
        return {
            event_id: float(_lexical_score(question, str(event.get("text") or "")))
            for event_id, event in events.items()
        }
    import numpy as np

    query_path = _resolve_path(case_root, str(query_ref["path"]))
    _verify_checksum(query_path, str(query_ref.get("checksum") or ""))
    query_matrix = np.load(query_path)
    query = query_matrix[int(query_ref["row_index"])]
    if len(query) != 2048:
        raise ValueError(f"query embedding dimension must be 2048: {query_path}")
    matrix_cache: dict[str, Any] = {}
    scores: dict[str, float] = {}
    for node in overlay.atomic_events:
        ref = node.embedding_ref
        if ref is None or ref.model != "Qwen/Qwen3-VL-Embedding-2B":
            raise ValueError(f"event {node.node_id} lacks the required Qwen embedding")
        matrix_path = str(Path(ref.path).expanduser().resolve())
        if matrix_path not in matrix_cache:
            _verify_checksum(Path(matrix_path), ref.checksum or "")
            matrix_cache[matrix_path] = np.load(matrix_path)
        vector = matrix_cache[matrix_path][ref.row_index]
        if len(vector) != len(query):
            raise ValueError(f"embedding dimension mismatch for {node.node_id}")
        scores[node.node_id] = float(np.dot(query, vector))
    return scores


def _verify_checksum(path: Path, expected: str) -> None:
    if not expected:
        return
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(f"embedding checksum mismatch for {path}")


class _FrozenPosteriorBackend:
    """Diagnostic backend that acquires evidence but freezes graph posterior state."""

    name = "frozen_factor_posterior_ablation/v0.1"

    def __init__(self) -> None:
        self.base = FactorGraphBeliefBackend()

    def initialize(self, *args: Any, **kwargs: Any) -> Any:
        initial = self.base.initialize(*args, **kwargs)
        return replace(initial, backend_name=self.name, backend_ref="frozen_after_initialization")

    def update(self, belief: Any, action: Any, observations: Any, overlay: Any) -> BeliefUpdateResult:
        computed = self.base.update(belief, action, observations, overlay)
        candidate = computed.belief
        answerability = (
            Answerability.READY
            if candidate.acquired_evidence and not candidate.missing_roles and not belief.contradictions
            else Answerability.NOT_READY
        )
        frozen = replace(
            candidate,
            backend_name=self.name,
            backend_ref="frozen_after_initialization",
            contradictions=belief.contradictions,
            relation_states=belief.relation_states,
            priority_edge_ids=belief.priority_edge_ids,
            blocked_edge_ids=belief.blocked_edge_ids,
            answerability=answerability,
        )
        return BeliefUpdateResult(
            belief=frozen,
            delta=BeliefDeltaDescriptor(
                resolved_roles=computed.delta.resolved_roles,
                relation_updates=(),
                contradiction_updates=(),
                uncertainty_change=computed.delta.uncertainty_change,
                answerability_after=answerability,
                predicted_only=False,
            ),
        )


def _shuffled_relation_overlay(overlay: Any, case_id: str) -> Any:
    payload = overlay.to_dict()
    relations = payload.get("relations") or []
    if len(relations) < 2:
        return overlay
    keys = ("relation_probabilities", "status", "direction_confidence", "provenance", "warrant")
    factor_payloads = [{key: row.get(key) for key in keys} for row in relations]
    offset = 1 + int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % (len(relations) - 1)
    rotated = factor_payloads[offset:] + factor_payloads[:offset]
    for row, replacement in zip(relations, rotated):
        for key, value in replacement.items():
            if value is None:
                row.pop(key, None)
            else:
                row[key] = value
    payload["overlay_id"] = f"{overlay.overlay_id}:shuffled:{offset}"
    return overlay_from_dict(payload)
