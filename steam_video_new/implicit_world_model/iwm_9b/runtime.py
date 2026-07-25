"""Structured 9B adapters for the persistent multi-trajectory runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from memory_graph.types import MemoryNode

from ..full_graph_iwm.contracts import (
    AnswerabilityState,
    CategoricalBeliefDelta,
    ContradictionChange,
    EvidenceOutcome,
    FrontierChange,
    ImaginedTransition,
    PreferenceLabel,
    PredictedObservation,
    ProgressChange,
    RetainedEvidenceGraph,
)
from ..full_graph_iwm.multi_trajectory import TrajectoryPool
from ..full_graph_iwm.multi_trajectory_rollout import (
    HypothesisExpansionRequest,
    HypothesisPathPair,
    HypothesisPathPreference,
    ImaginedHypothesisPath,
)


class StructuredJSONClient(Protocol):
    """Minimal serving boundary for a local/vLLM or remote structured model."""

    model: str

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]: ...


class Structured9BMultiTrajectoryModel:
    """Use rich transition patches and categorical partial preference.

    The same base model may serve both modes through separate LoRA adapters.
    Imagined structured patches are attached only to ``ImaginedTransition`` and
    are never applied to persistent belief.
    """

    def __init__(
        self,
        client: StructuredJSONClient,
        *,
        transition_batch_size: int = 32,
        comparison_batch_size: int = 32,
    ) -> None:
        if transition_batch_size < 1 or comparison_batch_size < 1:
            raise ValueError("9B runtime batch sizes must be positive")
        self.client = client
        self.model_name = str(getattr(client, "model", "Qwen/Qwen3.5-9B"))
        self.transition_batch_size = transition_batch_size
        self.comparison_batch_size = comparison_batch_size
        self.audits: list[dict[str, Any]] = []
        self.repair_count = 0

    def predict_batch(
        self,
        requests: Sequence[HypothesisExpansionRequest],
        graph: RetainedEvidenceGraph,
    ) -> Sequence[ImaginedTransition]:
        predictions: list[ImaginedTransition] = []
        model_requests = tuple(row for row in requests if row.action.reads_evidence)
        for start in range(0, len(model_requests), self.transition_batch_size):
            batch = tuple(model_requests[start : start + self.transition_batch_size])
            predictions.extend(self._predict_batch(batch, graph))
        prediction_by_request = {
            request.request_id: prediction
            for request, prediction in zip(model_requests, predictions)
        }
        ordered = tuple(
            prediction_by_request[request.request_id]
            if request.action.reads_evidence
            else _neutral_no_read_transition(request)
            for request in requests
        )
        self.audits.append(
            {
                "operation": "structured_transition",
                "request_count": len(requests),
                "model_prediction_count": len(predictions),
                "neutral_no_read_count": len(requests) - len(model_requests),
                "prediction_count": len(ordered),
                "complete_coverage": len(ordered) == len(requests),
                "cumulative_repair_count": self.repair_count,
                "numeric_reward_present": False,
                "top_k_applied": False,
            }
        )
        return ordered

    def compare_batch(
        self,
        pool: TrajectoryPool,
        pairs: Sequence[HypothesisPathPair],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> Sequence[HypothesisPathPreference]:
        if include_imagined_transitions:
            _validate_joint_coverage(
                pool,
                tuple(path for pair in pairs for path in (pair.left, pair.right)),
            )
        result: list[HypothesisPathPreference] = []
        for start in range(0, len(pairs), self.comparison_batch_size):
            batch = tuple(pairs[start : start + self.comparison_batch_size])
            aliases = {
                _alias("comparison", index): pair for index, pair in enumerate(batch)
            }
            payload = {
                "question": _question(pool),
                "persistent_trajectory_pool": _pool_view(pool),
                "comparisons": {
                    alias: {
                        "left": _path_view(
                            pair.left,
                            graph,
                            include_imagined_transitions=include_imagined_transitions,
                        ),
                        "right": _path_view(
                            pair.right,
                            graph,
                            include_imagined_transitions=include_imagined_transitions,
                        ),
                    }
                    for alias, pair in aliases.items()
                },
                "allowed_labels": [
                    "prefer_left",
                    "prefer_right",
                    "tie",
                    "incomparable",
                ],
                "contract": _preference_contract(),
            }
            rows = None
            for attempt in range(2):
                response = self.client.complete_json(
                    task=(
                        "<PLAN_COMPARE> Compare every supplied joint-chain pair."
                        if attempt == 0
                        else (
                            "Repair the Planner JSON. Return every exact comparison "
                            "alias once, with no extra aliases or numeric values."
                        )
                    ),
                    payload=payload,
                )
                try:
                    _reject_numeric_output(response)
                    rows = _strict_alias_rows(response, "comparisons", aliases)
                    break
                except ValueError:
                    if attempt == 1:
                        raise
                    self.repair_count += 1
            assert rows is not None
            for alias, pair in aliases.items():
                row = rows[alias]
                if set(row) != {"label", "rationale"}:
                    raise ValueError("9B Planner comparison row schema is invalid")
                result.append(
                    HypothesisPathPreference(
                        comparison_id=pair.comparison_id,
                        left_id=pair.left.path_id,
                        right_id=pair.right.path_id,
                        label=PreferenceLabel(str(row["label"])),
                        rationale=_string(row["rationale"], "rationale"),
                    )
                )
        self.audits.append(
            {
                "operation": "structured_pairwise_preference",
                "pair_count": len(pairs),
                "result_count": len(result),
                "complete_coverage": len(result) == len(pairs),
                "imagined_transitions_visible": include_imagined_transitions,
                "cumulative_repair_count": self.repair_count,
                "numeric_reward_present": False,
                "top_k_applied": False,
            }
        )
        return tuple(result)

    def select_frontier(
        self,
        pool: TrajectoryPool,
        paths: Sequence[ImaginedHypothesisPath],
        graph: RetainedEvidenceGraph,
        *,
        include_imagined_transitions: bool,
    ) -> tuple[str, ...]:
        if include_imagined_transitions:
            _validate_joint_coverage(pool, paths)
        ordered = tuple(sorted(paths, key=lambda row: row.path_id))
        aliases = {_alias("chain", index): path for index, path in enumerate(ordered)}
        payload = {
            "question": _question(pool),
            "persistent_trajectory_pool": _pool_view(pool),
            "candidate_joint_chains": {
                alias: _path_view(
                    path,
                    graph,
                    include_imagined_transitions=include_imagined_transitions,
                )
                for alias, path in aliases.items()
            },
            "allowed_status": ["unique", "tie", "incomparable"],
            "allowed_preference_output": {
                "status": "one allowed status",
                "preferred": "complete frontier of exact chain aliases",
                "rationale": "short categorical comparison rationale",
            },
            "planning_objective": {
                "compare_delayed_hypothesis_conditioned_effects": True,
                "prefer_grounded_progress_toward_missing_answer_evidence": True,
                "do_not_require_the_first_hop_to_finish_the_answer": True,
                "terminal_no_read_paths_do_not_add_evidence": True,
            },
            "contract": _preference_contract(),
        }
        response = None
        for attempt in range(2):
            candidate = self.client.complete_json(
                task=(
                    (
                        "<PLAN_COMPARE> Compare every complete joint reasoning chain "
                        "and return the full categorical frontier."
                    )
                    if attempt == 0
                    else (
                        "Repair the Planner frontier JSON. Use exactly status, "
                        "preferred and rationale; preserve ties and incomparability."
                    )
                ),
                payload=payload,
            )
            try:
                _reject_numeric_output(candidate)
                if not isinstance(candidate, dict) or set(candidate) != {
                    "status",
                    "preferred",
                    "rationale",
                }:
                    raise ValueError("9B Planner frontier schema is invalid")
                response = candidate
                break
            except ValueError:
                if attempt == 1:
                    raise
                self.repair_count += 1
        assert response is not None
        status = _string(response["status"], "status")
        preferred = _strings(response["preferred"], "preferred")
        _string(response["rationale"], "rationale")
        if status not in {"unique", "tie", "incomparable"}:
            raise ValueError("9B Planner frontier status is invalid")
        if len(preferred) != len(set(preferred)) or not set(preferred) <= set(aliases):
            raise ValueError("9B Planner returned an invalid chain alias")
        if status == "unique" and len(preferred) != 1:
            raise ValueError("unique frontier requires one preferred chain")
        if status == "tie" and len(preferred) < 2:
            raise ValueError("tie frontier requires multiple preferred chains")
        if status == "incomparable" and len(preferred) < 2:
            raise ValueError(
                "incomparable frontier must preserve multiple undominated chains"
            )
        self.audits.append(
            {
                "operation": "structured_setwise_preference",
                "path_count": len(paths),
                "complete_coverage": True,
                "imagined_transitions_visible": include_imagined_transitions,
                "cumulative_repair_count": self.repair_count,
                "numeric_reward_present": False,
                "top_k_applied": False,
            }
        )
        return tuple(aliases[alias].path_id for alias in preferred)

    def _predict_batch(
        self,
        requests: tuple[HypothesisExpansionRequest, ...],
        graph: RetainedEvidenceGraph,
    ) -> tuple[ImaginedTransition, ...]:
        aliases = {
            _alias("prediction", index): request
            for index, request in enumerate(requests)
        }
        payload = {
            "requests": {
                alias: _transition_input(request, graph)
                for alias, request in aliases.items()
            },
            "required_output": {
                "only_key": "predictions",
                "exact_prediction_aliases": list(aliases),
                "predictions": {
                    "each_alias": {
                        "observation_patch": {
                            "event_or_state": "short structured descriptor",
                            "entity_bindings": "object of categorical bindings",
                            "temporal_binding": "categorical temporal relation",
                            "evidence_role": "categorical evidence role",
                            "alternatives": "string list",
                        },
                        "belief_patch": {
                            "supported_claims": "string list",
                            "contradicted_claims": "string list",
                            "newly_bound_variables": "string list",
                            "opened_dependencies": "string list",
                            "resolved_dependencies": "string list",
                        },
                        "categorical_audit": {
                            "observation_outcome": [
                                row.value for row in EvidenceOutcome
                            ],
                            "progress": [row.value for row in ProgressChange],
                            "answerability_after": [
                                row.value for row in AnswerabilityState
                            ],
                            "frontier_change": [row.value for row in FrontierChange],
                            "contradiction_change": [
                                row.value for row in ContradictionChange
                            ],
                            "resolved_roles": "current role strings",
                            "opened_roles": "categorical role strings",
                            "relation_updates": "categorical relation strings",
                        },
                    }
                },
            },
            "contract": {
                "one_prediction_per_request": True,
                "hypothesis_conditioned": True,
                "predicted_only": True,
                "do_not_select_an_action": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            response = self.client.complete_json(
                task=(
                    (
                        "<IWM_TRANSITION> Predict the structured future observation "
                        "and hypothesis-conditioned belief patch for every legal "
                        "action. Cover every exact prediction alias."
                    )
                    if attempt == 0
                    else (
                        "Repair the IWM JSON. Return every exact prediction alias "
                        "once, with the complete requested schema, no extra aliases "
                        "and no numeric values."
                    )
                ),
                payload=payload,
            )
            try:
                _reject_numeric_output(response)
                rows = _strict_alias_rows(response, "predictions", aliases)
                return tuple(
                    _parse_transition(request, rows[alias], graph)
                    for alias, request in aliases.items()
                )
            except ValueError:
                if attempt == 1:
                    raise
                self.repair_count += 1
        raise RuntimeError("unreachable structured IWM repair state")


def _parse_transition(
    request: HypothesisExpansionRequest,
    row: Any,
    graph: RetainedEvidenceGraph,
) -> ImaginedTransition:
    if not isinstance(row, dict) or set(row) != {
        "observation_patch",
        "belief_patch",
        "categorical_audit",
    }:
        raise ValueError("9B structured transition row schema is invalid")
    observation = _strict_object(
        row["observation_patch"],
        {
            "event_or_state",
            "entity_bindings",
            "temporal_binding",
            "evidence_role",
            "alternatives",
        },
        "observation_patch",
    )
    _string(observation["event_or_state"], "event_or_state")
    observation["entity_bindings"] = _entity_bindings(observation["entity_bindings"])
    _string(observation["temporal_binding"], "temporal_binding")
    _string(observation["evidence_role"], "evidence_role")
    _strings(observation["alternatives"], "alternatives")

    belief_patch = _strict_object(
        row["belief_patch"],
        {
            "supported_claims",
            "contradicted_claims",
            "newly_bound_variables",
            "opened_dependencies",
            "resolved_dependencies",
        },
        "belief_patch",
    )
    for field, value in belief_patch.items():
        _strings(value, field)

    audit = _strict_object(
        row["categorical_audit"],
        {
            "observation_outcome",
            "progress",
            "answerability_after",
            "frontier_change",
            "contradiction_change",
            "resolved_roles",
            "opened_roles",
            "relation_updates",
        },
        "categorical_audit",
    )
    resolved = _strings(audit["resolved_roles"], "resolved_roles")
    known_roles = set(request.belief.required_roles or request.belief.missing_roles)
    if not set(resolved) <= known_roles:
        raise ValueError("9B transition resolved an unknown role")
    currently_missing = set(request.belief.missing_roles)
    resolved = tuple(role for role in resolved if role in currently_missing)
    opened = _strings(audit["opened_roles"], "opened_roles")
    relation_updates = _strings(audit["relation_updates"], "relation_updates")
    structured_patch = {
        "observation_patch": observation,
        "belief_patch": belief_patch,
        "hypothesis": request.hypothesis,
        "predicted_only": True,
    }
    return ImaginedTransition(
        action=request.action,
        observation=PredictedObservation(
            target_id=request.action.target_id,
            outcome=EvidenceOutcome(
                _categorical(audit["observation_outcome"], "observation_outcome")
            ),
            descriptor=_target_descriptor(request, graph),
        ),
        belief_delta=CategoricalBeliefDelta(
            progress=ProgressChange(_categorical(audit["progress"], "progress")),
            answerability_after=AnswerabilityState(
                _categorical(audit["answerability_after"], "answerability_after")
            ),
            frontier_change=FrontierChange(
                _categorical(audit["frontier_change"], "frontier_change")
            ),
            contradiction_change=ContradictionChange(
                _categorical(audit["contradiction_change"], "contradiction_change")
            ),
            resolved_roles=resolved,
            opened_roles=opened,
            relation_updates=relation_updates,
        ),
        structured_patch=structured_patch,
    )


def _neutral_no_read_transition(
    request: HypothesisExpansionRequest,
) -> ImaginedTransition:
    return ImaginedTransition(
        action=request.action,
        observation=PredictedObservation(
            target_id=request.action.target_id,
            outcome=EvidenceOutcome.INCONCLUSIVE,
        ),
        belief_delta=CategoricalBeliefDelta(
            progress=ProgressChange.UNCHANGED,
            answerability_after=request.belief.answerability,
            frontier_change=FrontierChange.UNCHANGED,
            contradiction_change=ContradictionChange.UNCHANGED,
        ),
    )


def _transition_input(
    request: HypothesisExpansionRequest,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    acquired = set(request.belief.acquired_evidence) - set(
        request.belief.imagined_evidence
    )
    return {
        "question": request.belief.question,
        "hypothesis": request.hypothesis,
        "persistent_path_state": {
            "current_node_id": request.belief.current_node_id,
            "missing_roles": list(request.belief.missing_roles),
            "grounded_role_evidence": [
                list(row) for row in request.belief.grounded_role_evidence
            ],
            "contradictions": list(request.belief.contradictions),
            "answerability": request.belief.answerability.value,
        },
        "acquired_real_evidence": [
            _real_node_view(node) for node in graph.nodes if node.node_id in acquired
        ],
        "action": _action_view(request, graph),
        "imagined_prefix": [
            _transition_view(row, graph) for row in request.imagined_history
        ],
    }


def _path_view(
    path: ImaginedHypothesisPath,
    graph: RetainedEvidenceGraph,
    *,
    include_imagined_transitions: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "action_sequence": [
            {
                "kind": row.action.kind.value,
                "relation": row.action.relation,
                "target_semantic_key": list(_target_descriptor_from_action(row, graph)),
            }
            for row in path.transitions
        ],
    }
    if include_imagined_transitions:
        result["hypothesis_conditioned_outcomes"] = [
            {
                "hypothesis": outcome.hypothesis,
                "imagined_transitions": [
                    _transition_view(row, graph) for row in outcome.transitions
                ],
            }
            for outcome in path.conditioned_outcomes
        ]
    else:
        result["competing_hypotheses"] = [
            outcome.hypothesis for outcome in path.conditioned_outcomes
        ]
    return result


def _transition_view(
    transition: ImaginedTransition,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    delta = transition.belief_delta
    return {
        "target_semantic_key": list(_target_descriptor_from_action(transition, graph)),
        "structured_belief_event_patch": transition.structured_patch,
        "categorical_audit": {
            "observation_outcome": transition.observation.outcome.value,
            "progress": delta.progress.value,
            "answerability_after": delta.answerability_after.value,
            "frontier_change": delta.frontier_change.value,
            "contradiction_change": delta.contradiction_change.value,
            "resolved_roles": list(delta.resolved_roles),
            "opened_roles": list(delta.opened_roles),
            "relation_updates": list(delta.relation_updates),
        },
    }


def _action_view(
    request: HypothesisExpansionRequest,
    graph: RetainedEvidenceGraph,
) -> dict[str, Any]:
    action = request.action
    target = graph.node_by_id.get(action.target_id or "")
    return {
        "kind": action.kind.value,
        "relation": action.relation,
        "target": (
            {
                "semantic_key": _semantic_key(target),
                "node_type": target.node_type,
                "structural_tags": list(target.metadata.get("structural_tags") or []),
            }
            if target is not None
            else None
        ),
    }


def _real_node_view(node: MemoryNode) -> dict[str, Any]:
    return {
        "semantic_key": _semantic_key(node),
        "evidence_value": str(node.text or node.metadata.get("predicate") or ""),
        "provenance": {
            "producer": str(node.provenance.get("producer") or ""),
            "layer": str(node.metadata.get("layer") or ""),
        },
    }


def _pool_view(pool: TrajectoryPool) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis": row.hypothesis,
            "status": row.status.value,
            "missing_roles": list(row.belief.missing_roles),
            "contradictions": list(row.belief.contradictions),
            "answerability": row.belief.answerability.value,
        }
        for row in pool.expandable
    ]


def _validate_joint_coverage(
    pool: TrajectoryPool,
    paths: Sequence[ImaginedHypothesisPath],
) -> None:
    expected = {row.trajectory_id for row in pool.expandable}
    for path in paths:
        observed = {row.trajectory_id for row in path.conditioned_outcomes}
        if observed != expected:
            raise ValueError(
                "9B Planner requires every joint chain to cover every active hypothesis"
            )


def _strict_alias_rows(
    response: Any,
    key: str,
    aliases: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if isinstance(response, dict) and set(response) == set(aliases):
        rows = response
    elif isinstance(response, dict) and set(response) == {key}:
        rows = response[key]
    else:
        raise ValueError(f"9B response must contain only {key}")
    if not isinstance(rows, dict) or set(rows) != set(aliases):
        raise ValueError(f"9B {key} coverage does not match the request")
    if any(not isinstance(value, dict) for value in rows.values()):
        raise ValueError(f"9B {key} rows must be objects")
    return rows


def _strict_object(
    value: Any,
    keys: set[str],
    field: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{field} schema is invalid")
    return value


def _strings(value: Any, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {
            "",
            "none",
            "null",
            "n/a",
            "not_applicable",
        }:
            return ()
        rows = tuple(part.strip() for part in normalized.split(",") if part.strip())
        if len(rows) != len(set(rows)):
            raise ValueError(f"{field} must not contain duplicates")
        return rows
    if not isinstance(value, list) or any(not isinstance(row, str) for row in value):
        raise ValueError(f"{field} must be a string list")
    rows = tuple(value)
    if len(rows) != len(set(rows)):
        raise ValueError(f"{field} must not contain duplicates")
    return rows


def _entity_bindings(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {
            "",
            "none",
            "null",
            "n/a",
            "not_applicable",
            "unbound",
        }:
            return {}
        pairs = [row.strip() for row in normalized.split(",") if row.strip()]
        if any(":" not in row for row in pairs):
            return {"binding_description": normalized}
        return {
            key.strip(): item.strip()
            for key, item in (row.split(":", 1) for row in pairs)
            if key.strip() and item.strip()
        }
    if not isinstance(value, dict):
        raise ValueError("entity_bindings must be an object")
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, (str, list)):
            raise ValueError("entity_bindings values must be strings or string lists")
        if isinstance(item, list) and any(not isinstance(row, str) for row in item):
            raise ValueError("entity_bindings list values must contain strings")
    return value


def _string(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _categorical(value: Any, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        return value[0]
    raise ValueError(f"{field} must be one categorical string")


def _reject_numeric_output(value: Any) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise ValueError("9B output contains a forbidden numeric value")
    if isinstance(value, Mapping):
        for child in value.values():
            _reject_numeric_output(child)
        return
    if isinstance(value, Sequence):
        for child in value:
            _reject_numeric_output(child)
        return
    raise ValueError("9B output contains an unsupported value type")


def _target_descriptor(
    request: HypothesisExpansionRequest,
    graph: RetainedEvidenceGraph,
) -> tuple[str, ...]:
    return _target_descriptor_from_action(request, graph)


def _target_descriptor_from_action(
    value: Any,
    graph: RetainedEvidenceGraph,
) -> tuple[str, ...]:
    action = value.action
    if not action.reads_evidence or action.target_id is None:
        return ()
    node = graph.node_by_id.get(action.target_id)
    if node is None:
        raise ValueError("9B action target is absent from retained graph")
    return (_semantic_key(node),)


def _semantic_key(node: MemoryNode) -> str:
    return str(node.metadata.get("predicate") or node.text or node.node_id)


def _question(pool: TrajectoryPool) -> str:
    if not pool.expandable:
        raise ValueError("9B Planner requires an expandable trajectory pool")
    questions = {row.belief.question for row in pool.expandable}
    if len(questions) != 1:
        raise ValueError("9B Planner pool contains inconsistent questions")
    return next(iter(questions))


def _preference_contract() -> dict[str, Any]:
    return {
        "compare_complete_joint_chains": True,
        "preserve_ties_and_incomparability": True,
        "every_chain_covers_every_active_hypothesis": True,
        "no_heuristic_top_k": True,
        "no_numeric_reward_score_probability_confidence_or_utility": True,
    }


def _alias(prefix: str, index: int) -> str:
    if index < 0:
        raise ValueError("alias index must be non-negative")
    value = index
    letters = ""
    while True:
        value, remainder = divmod(value, 26)
        letters = chr(ord("a") + remainder) + letters
        if value == 0:
            return f"{prefix}_{letters}"
        value -= 1
