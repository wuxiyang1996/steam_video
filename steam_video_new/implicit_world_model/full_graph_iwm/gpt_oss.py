"""Strict batched GPT-OSS adapters for full retained-graph navigation."""

from __future__ import annotations

import hashlib
from typing import Any, Protocol, Sequence

from .contracts import (
    AnswerabilityState,
    BatchedCategoricalWorldModel,
    BatchedTrajectoryPreferenceModel,
    CategoricalBeliefDelta,
    ContradictionChange,
    CursorBeliefState,
    EvidenceOutcome,
    FrontierChange,
    IWMRequest,
    ImaginedTransition,
    PreferenceLabel,
    PredictedObservation,
    ProgressChange,
    TrajectoryPair,
    TrajectoryPreference,
)
from .model_input import graph_input_to_categorical_payload


class CategoricalJSONClient(Protocol):
    model: str

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class GPTOSSFullGraphWorldModel(BatchedCategoricalWorldModel):
    """Predict all supplied legal actions in one categorical request."""

    def __init__(self, client: CategoricalJSONClient) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def predict_batch(
        self,
        requests: Sequence[IWMRequest],
    ) -> Sequence[ImaginedTransition]:
        if not requests:
            return ()
        context_ids = [_context_id(request) for request in requests]
        contexts = {
            context_id: graph_input_to_categorical_payload(request.graph_input)
            for context_id, request in zip(context_ids, requests)
        }
        choices = {
            request.action.action_id: _alphabetic_alias("choice", index)
            for index, request in enumerate(requests)
        }
        payload = {
            "requests": [
                {
                    "choice": choices[request.action.action_id],
                    "action": {
                        "kind": request.action.kind.value,
                        "source_id": request.action.source_id,
                        "target_id": request.action.target_id,
                        "edge_id": request.action.edge_id,
                        "relation": request.action.relation,
                    },
                    "belief_id": request.belief.belief_id,
                    "context_id": context_id,
                    "parent_action_ids": list(request.parent_action_ids),
                    "imagined_history": [
                        _transition_payload(value) for value in request.imagined_history
                    ],
                }
                for context_id, request in zip(context_ids, requests)
            ],
            "contexts": contexts,
            "allowed_output": {
                "outcome": [value.value for value in EvidenceOutcome],
                "progress": [value.value for value in ProgressChange],
                "answerability_after": [value.value for value in AnswerabilityState],
                "frontier_change": [value.value for value in FrontierChange],
                "contradiction_change": [value.value for value in ContradictionChange],
                "resolved_roles": "exact strings copied from missing_roles",
                "opened_roles": "categorical role strings",
                "relation_updates": "edge IDs copied from the graph",
            },
            "required_output": {
                "top_level": "one JSON object",
                "only_key": "predictions",
                "prediction_fields": [
                    "choice",
                    "outcome",
                    "progress",
                    "answerability_after",
                    "frontier_change",
                    "contradiction_change",
                    "resolved_roles",
                    "opened_roles",
                    "relation_updates",
                ],
                "array_fields_even_when_empty": [
                    "resolved_roles",
                    "opened_roles",
                    "relation_updates",
                ],
            },
            "required_contract": {
                "one_prediction_per_action": True,
                "predictions_are_imagined": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        request_by_choice = {
            choices[request.action.action_id]: request for request in requests
        }
        for attempt in range(2):
            task = (
                (
                    "For every legal reasoning action, predict a categorical observation "
                    "descriptor and categorical future-belief delta. Return one JSON "
                    "object with the sole key predictions. Do not choose an action."
                )
                if attempt == 0
                else (
                    "Repair the response format. Return exactly one top-level JSON "
                    "object with the sole key predictions and exactly one row per "
                    "requested action; do not return prose, a bare list, or numbers."
                )
            )
            try:
                result = self.client.complete_json(task=task, payload=payload)
                _reject_numeric_output(result)
                rows = result.get("predictions")
                if not isinstance(rows, list):
                    raise ValueError("GPT-OSS batch output requires a predictions list")
                candidate = {
                    str(row.get("choice")): row for row in rows if isinstance(row, dict)
                }
                if set(candidate) != set(request_by_choice):
                    raise ValueError(
                        "GPT-OSS prediction coverage does not match legal actions"
                    )
                by_action = {
                    request_by_choice[choice].action.action_id: row
                    for choice, row in candidate.items()
                }
                return _parse_world_predictions(requests, by_action)
            except ValueError as exc:
                if "forbidden numeric" in str(exc) or attempt == 1:
                    raise
        raise RuntimeError("unreachable GPT-OSS transition repair state")


class GPTOSSFullGraphPreferenceModel(BatchedTrajectoryPreferenceModel):
    """Compare complete imagined trajectories, emitting ordinal labels only."""

    def __init__(self, client: CategoricalJSONClient) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def compare_batch(
        self,
        pairs: Sequence[TrajectoryPair],
        belief: CursorBeliefState,
    ) -> Sequence[TrajectoryPreference]:
        if not pairs:
            return ()
        trajectory_by_id = {
            trajectory.trajectory_id: trajectory
            for pair in pairs
            for trajectory in (pair.left, pair.right)
        }
        trajectory_aliases = {
            trajectory_id: _alphabetic_alias("path", index)
            for index, trajectory_id in enumerate(trajectory_by_id)
        }
        trajectories = {
            trajectory_aliases[trajectory_id]: [
                _transition_payload(value) for value in trajectory.transitions
            ]
            for trajectory_id, trajectory in trajectory_by_id.items()
        }
        comparison_aliases = tuple(
            _alphabetic_alias("comparison", index) for index in range(len(pairs))
        )
        payload = {
            "belief": {
                "question": belief.question,
                "answerability": belief.answerability.value,
                "missing_roles": list(belief.missing_roles),
                "contradictions": list(belief.contradictions),
                "accepted_relations": list(belief.accepted_relations),
                "rejected_relations": list(belief.rejected_relations),
                "unresolved_relations": list(belief.unresolved_relations),
            },
            "pairs": [
                {
                    "comparison": comparison,
                    "left": trajectory_aliases[pair.left.trajectory_id],
                    "right": trajectory_aliases[pair.right.trajectory_id],
                }
                for comparison, pair in zip(comparison_aliases, pairs)
            ],
            "trajectories": trajectories,
            "allowed_labels": [value.value for value in PreferenceLabel],
            "required_output": {
                "top_level": "one JSON object",
                "only_key": "comparisons",
                "comparison_fields": [
                    "comparison",
                    "label",
                    "rationale",
                ],
            },
            "required_contract": {
                "one_comparison_per_pair": True,
                "preserve_pair_order": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        task = (
            "Compare every pair of complete imagined reasoning trajectories under "
            "the same belief. Return one JSON object whose comparisons list uses "
            "only ordinal pairwise preferences."
        )
        rows: list[dict[str, Any]] | None = None
        for attempt in range(2):
            current_task = (
                task
                if attempt == 0
                else (
                    "Repair the response format. Return exactly one top-level JSON "
                    "object with the sole key comparisons and exactly one row per pair; "
                    "do not return a bare list, Markdown, prose, or numeric values."
                )
            )
            try:
                result = self.client.complete_json(task=current_task, payload=payload)
                _reject_numeric_output(result)
                candidate_rows = result.get("comparisons")
                if not isinstance(candidate_rows, list) or len(candidate_rows) != len(
                    pairs
                ):
                    raise ValueError(
                        "GPT-OSS preference coverage does not match trajectory pairs"
                    )
                if any(
                    not isinstance(row, dict) or row.get("comparison") != comparison
                    for comparison, row in zip(comparison_aliases, candidate_rows)
                ):
                    raise ValueError(
                        "GPT-OSS comparison IDs do not match the requested pairs"
                    )
                for row in candidate_rows:
                    PreferenceLabel(str(row.get("label") or "incomparable"))
                rows = candidate_rows
                break
            except ValueError as exc:
                if "forbidden numeric" in str(exc) or attempt == 1:
                    raise
        assert rows is not None
        comparisons: list[TrajectoryPreference] = []
        for pair, row in zip(pairs, rows):
            comparisons.append(
                TrajectoryPreference(
                    left_id=pair.left.trajectory_id,
                    right_id=pair.right.trajectory_id,
                    label=PreferenceLabel(str(row.get("label") or "incomparable")),
                    rationale=str(row.get("rationale") or "categorical comparison"),
                )
            )
        return tuple(comparisons)


def _parse_world_predictions(
    requests: Sequence[IWMRequest],
    by_action: dict[str, dict[str, Any]],
) -> tuple[ImaginedTransition, ...]:
    predictions: list[ImaginedTransition] = []
    for request in requests:
        row = by_action[request.action.action_id]
        resolved = _strings(row.get("resolved_roles"), "resolved_roles")
        if not set(resolved).issubset(request.belief.missing_roles):
            raise ValueError("resolved_roles must be copied from current missing_roles")
        relation_updates = _strings(row.get("relation_updates"), "relation_updates")
        known_edges = {
            edge.edge_id for edge in request.graph_input.correlation_edges
        } | {edge.edge_id for edge in request.graph_input.temporal_edges}
        if not set(relation_updates).issubset(known_edges):
            raise ValueError("relation_updates contains an unknown edge")
        predictions.append(
            ImaginedTransition(
                action=request.action,
                observation=PredictedObservation(
                    target_id=request.action.target_id,
                    outcome=EvidenceOutcome(str(row.get("outcome") or "inconclusive")),
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=ProgressChange(str(row.get("progress") or "unchanged")),
                    answerability_after=AnswerabilityState(
                        str(row.get("answerability_after") or "not_ready")
                    ),
                    frontier_change=FrontierChange(
                        str(row.get("frontier_change") or "unchanged")
                    ),
                    contradiction_change=ContradictionChange(
                        str(row.get("contradiction_change") or "unchanged")
                    ),
                    resolved_roles=resolved,
                    opened_roles=_strings(row.get("opened_roles"), "opened_roles"),
                    relation_updates=relation_updates,
                ),
            )
        )
    return tuple(predictions)


def _transition_payload(transition: ImaginedTransition) -> dict[str, Any]:
    return {
        "action_id": transition.action.action_id,
        "action_kind": transition.action.kind.value,
        "target_id": transition.action.target_id,
        "relation": transition.action.relation,
        "observation_outcome": transition.observation.outcome.value,
        "belief_delta": {
            "progress": transition.belief_delta.progress.value,
            "answerability_after": transition.belief_delta.answerability_after.value,
            "frontier_change": transition.belief_delta.frontier_change.value,
            "contradiction_change": transition.belief_delta.contradiction_change.value,
            "resolved_roles": list(transition.belief_delta.resolved_roles),
            "opened_roles": list(transition.belief_delta.opened_roles),
            "relation_updates": list(transition.belief_delta.relation_updates),
        },
        "predicted_only": True,
    }


def _context_id(request: IWMRequest) -> str:
    value = "\x1f".join(
        (
            request.belief.belief_id,
            request.graph_input.current_node_id or "virtual_root",
            *request.graph_input.acquired_evidence,
            *(action.action_id for action in request.graph_input.legal_actions),
        )
    )
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]
    return f"context:{digest}"


def _alphabetic_alias(prefix: str, index: int) -> str:
    if index < 0:
        raise ValueError("categorical alias index must be non-negative")
    letters: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("a") + remainder))
        if value == 0:
            break
        value -= 1
    return f"{prefix}_{''.join(reversed(letters))}"


def _strings(value: object, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of categorical strings")
    return tuple(value)


def _reject_numeric_output(value: object, path: str = "output") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        raise ValueError(f"GPT-OSS emitted forbidden numeric value at {path}")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_numeric_output(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_numeric_output(item, f"{path}.{key}")
        return
    raise ValueError(f"GPT-OSS emitted unsupported value at {path}")
