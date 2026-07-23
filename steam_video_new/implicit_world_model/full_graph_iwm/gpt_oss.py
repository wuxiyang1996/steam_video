"""Strict batched GPT-OSS adapters for full retained-graph navigation."""

from __future__ import annotations

from dataclasses import replace
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
    LegalGraphAction,
    PreferenceLabel,
    PredictedObservation,
    ProgressChange,
    RetainedEvidenceGraph,
    TrajectoryPair,
    TrajectoryPrediction,
    TrajectoryPreference,
)
from memory_graph.types import MemoryNode
from .model_input import graph_input_to_categorical_payload


class CategoricalJSONClient(Protocol):
    model: str

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class GPTOSSQuestionBeliefInitializer:
    """Create categorical evidence roles missing from the initial question state."""

    def __init__(self, client: CategoricalJSONClient) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def initialize(self, question: str) -> tuple[str, ...]:
        payload = {
            "question": question,
            "required_output": {
                "only_keys": ["missing_roles", "rationale"],
                "missing_roles": (
                    "short categorical evidence-role strings needed to answer the question"
                ),
            },
            "required_contract": {
                "roles_are_not_answers": True,
                "roles_are_not_graph_nodes": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    (
                        "Decompose the question into the minimal categorical evidence "
                        "roles that must be grounded before it is answerable."
                    )
                    if attempt == 0
                    else (
                        "Repair the initial belief schema. Return only missing_roles as "
                        "a non-empty list of unique categorical strings and rationale."
                    )
                ),
                payload=payload,
            )
            try:
                if set(result) != {"missing_roles", "rationale"}:
                    raise ValueError("belief initializer output fields do not match schema")
                roles = _strings(result.get("missing_roles"), "missing_roles")
                if not roles or len(roles) != len(set(roles)) or any(not role.strip() for role in roles):
                    raise ValueError("belief initializer requires unique non-empty roles")
                return roles
            except ValueError:
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable belief initializer repair state")


class GPTOSSRealEvidenceBeliefUpdater:
    """Correct persistent belief from one executed, real L1 observation."""

    def __init__(self, client: CategoricalJSONClient) -> None:
        self.client = client
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def update(
        self,
        belief: CursorBeliefState,
        observation: MemoryNode,
    ) -> CursorBeliefState:
        return self._update(belief, observation, hypothesis=None)

    def update_for_trajectory(
        self,
        trajectory_id: str,
        hypothesis: str,
        previous_belief: CursorBeliefState,
        structurally_updated_belief: CursorBeliefState,
        action: LegalGraphAction,
        observation: MemoryNode,
        graph: RetainedEvidenceGraph,
    ) -> CursorBeliefState:
        """Correct one persistent trajectory using its explicit hypothesis."""

        del trajectory_id, previous_belief, action, graph
        return self._update(
            structurally_updated_belief,
            observation,
            hypothesis=hypothesis,
        )

    def _update(
        self,
        belief: CursorBeliefState,
        observation: MemoryNode,
        *,
        hypothesis: str | None,
    ) -> CursorBeliefState:
        payload = {
            "question": belief.question,
            "trajectory_hypothesis": hypothesis,
            "belief_before": {
                "required_roles": list(belief.required_roles),
                "missing_roles": list(belief.missing_roles),
                "grounded_role_evidence": [
                    {"role": role, "node_id": node_id}
                    for role, node_id in belief.grounded_role_evidence
                ],
                "contradictions": list(belief.contradictions),
                "answerability": belief.answerability.value,
            },
            "executed_real_observation": {
                "node_id": observation.node_id,
                "text": observation.text,
                "predicate": observation.metadata.get("predicate"),
                "action_kind": observation.metadata.get("action_kind"),
                "participants": observation.metadata.get("participants") or [],
                "states": observation.metadata.get("states") or [],
                "state_change": observation.metadata.get("state_change"),
            },
            "allowed_contradiction_change": [
                value.value for value in ContradictionChange
            ],
            "required_output": {
                "only_keys": [
                    "resolved_roles",
                    "opened_roles",
                    "contradiction_change",
                    "rationale",
                ],
                "resolved_roles": "exact strings copied from missing_roles",
                "opened_roles": "categorical evidence roles, empty unless newly exposed",
            },
            "required_contract": {
                "observation_is_real_not_imagined": True,
                "correction_is_conditioned_on_trajectory_hypothesis": (
                    hypothesis is not None
                ),
                "no_hidden_clue_or_answer_label": True,
                "answerability_is_computed_by_belief_backend": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        result = self.client.complete_json(
            task=(
                "Update the persistent categorical belief using only the executed real "
                "observation. Resolve a role only when the observation directly grounds it."
            ),
            payload=payload,
        )
        expected = {
            "resolved_roles",
            "opened_roles",
            "contradiction_change",
            "rationale",
        }
        if set(result) != expected:
            raise ValueError("real belief update fields do not match schema")
        resolved = _strings(result.get("resolved_roles"), "resolved_roles")
        if not set(resolved).issubset(belief.missing_roles):
            raise ValueError("real belief update resolved unknown roles")
        opened = _strings(result.get("opened_roles"), "opened_roles")
        missing = [role for role in belief.missing_roles if role not in set(resolved)]
        missing.extend(role for role in opened if role not in missing)
        required = list(belief.required_roles or belief.missing_roles)
        required.extend(role for role in opened if role not in required)
        bindings = dict(belief.grounded_role_evidence)
        for role in resolved:
            bindings[role] = observation.node_id
        contradiction_change = ContradictionChange(
            str(result.get("contradiction_change") or "")
        )
        contradictions = belief.contradictions
        if contradiction_change is ContradictionChange.RESOLVED:
            contradictions = ()
        elif contradiction_change is ContradictionChange.OPENED:
            contradictions = tuple(
                dict.fromkeys((*contradictions, "real_evidence_contradiction"))
            )
        cited_nodes = {
            node_id
            for node_id in belief.acquired_evidence
            if node_id not in set(belief.imagined_evidence)
        }
        minimum_distinct_evidence = min(2, len(required))
        lineage_complete = set(required).issubset(bindings)
        answerability = AnswerabilityState.NOT_READY
        if (
            not missing
            and not contradictions
            and lineage_complete
            and len(cited_nodes) >= minimum_distinct_evidence
        ):
            answerability = AnswerabilityState.READY
        return replace(
            belief,
            belief_id=f"{belief.belief_id}:real-correction",
            required_roles=tuple(required),
            missing_roles=tuple(missing),
            grounded_role_evidence=tuple(
                (role, bindings[role]) for role in required if role in bindings
            ),
            contradictions=contradictions,
            answerability=answerability,
        )


class GPTOSSFullGraphWorldModel(BatchedCategoricalWorldModel):
    """Predict every supplied legal action in bounded transport batches."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        batch_size: int = 48,
        max_contexts_per_batch: int = 8,
    ) -> None:
        if batch_size < 1 or max_contexts_per_batch < 1:
            raise ValueError("world-model batch limits must be positive")
        self.client = client
        self.batch_size = batch_size
        self.max_contexts_per_batch = max_contexts_per_batch
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))
        self.normalization_audits: list[dict[str, Any]] = []

    def predict_batch(
        self,
        requests: Sequence[IWMRequest],
    ) -> Sequence[ImaginedTransition]:
        if not requests:
            return ()
        predictions: list[ImaginedTransition] = []
        for batch in self._transport_batches(requests):
            predictions.extend(self._predict_one_batch(batch))
        return tuple(predictions)

    def _transport_batches(
        self,
        requests: Sequence[IWMRequest],
    ) -> tuple[tuple[IWMRequest, ...], ...]:
        batches: list[tuple[IWMRequest, ...]] = []
        current: list[IWMRequest] = []
        context_ids: set[str] = set()
        for request in requests:
            context_id = _context_id(request)
            adds_context = context_id not in context_ids
            if current and (
                len(current) >= self.batch_size
                or (adds_context and len(context_ids) >= self.max_contexts_per_batch)
            ):
                batches.append(tuple(current))
                current = []
                context_ids = set()
            current.append(request)
            context_ids.add(context_id)
        if current:
            batches.append(tuple(current))
        return tuple(batches)

    def _predict_one_batch(
        self,
        requests: Sequence[IWMRequest],
    ) -> tuple[ImaginedTransition, ...]:
        context_ids = [_context_id(request) for request in requests]
        contexts = {
            context_id: graph_input_to_categorical_payload(request.graph_input)
            for context_id, request in zip(context_ids, requests)
        }
        choices = tuple(
            _alphabetic_alias("choice", index) for index in range(len(requests))
        )
        payload = {
            "requests": [
                {
                    "choice": choice,
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
                for choice, context_id, request in zip(choices, context_ids, requests)
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
                "semantic_keys_are_grounded_address_summaries": True,
                "use_target_semantic_key_structural_tags_and_edge_channel": True,
                "compare_each_target_against_current_missing_roles": True,
                "resolved_roles_must_be_exact_current_role_strings": True,
                "do_not_collapse_distinct_targets_without_reason": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        request_by_choice = dict(zip(choices, requests))
        for attempt in range(2):
            task = (
                (
                    "For every legal reasoning action, predict a categorical outcome "
                    "and categorical future-belief delta. Each target's semantic_key is "
                    "already bound by the backend: use it to predict which "
                    "exact current missing_roles the read may resolve, while keeping the "
                    "observation imagined-only. Distinguish targets when their keys imply "
                    "different evidence roles. Return one JSON object with the sole key "
                    "predictions. Do not choose an action."
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
                ordered_rows = tuple(candidate[choice] for choice in choices)
                return _parse_world_predictions(
                    requests,
                    ordered_rows,
                    normalization_audits=self.normalization_audits,
                )
            except ValueError as exc:
                if "forbidden numeric" in str(exc) or attempt == 1:
                    raise
        raise RuntimeError("unreachable GPT-OSS transition repair state")


class GPTOSSFullGraphPreferenceModel(BatchedTrajectoryPreferenceModel):
    """Compare complete imagined trajectories, emitting ordinal labels only."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        batch_size: int = 64,
    ) -> None:
        if batch_size < 1:
            raise ValueError("preference batch_size must be positive")
        self.client = client
        self.batch_size = batch_size
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))

    def compare_batch(
        self,
        pairs: Sequence[TrajectoryPair],
        belief: CursorBeliefState,
    ) -> Sequence[TrajectoryPreference]:
        if not pairs:
            return ()
        comparisons: list[TrajectoryPreference] = []
        for start in range(0, len(pairs), self.batch_size):
            comparisons.extend(
                GPTOSSFullGraphSetwisePreferenceModel(self.client)._compare_one_batch(
                    pairs[start : start + self.batch_size],
                    belief,
                )
            )
        return tuple(comparisons)


class GPTOSSFullGraphSetwisePreferenceModel:
    """Choose categorical trajectories with complete-coverage bounded tournaments.

    Every candidate is judged; batching is a transport constraint, not a heuristic
    candidate-pruning step. Ties and incomparable groups are preserved.
    """

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        batch_size: int = 24,
        group_batch_size: int = 8,
    ) -> None:
        if batch_size < 2 or group_batch_size < 1:
            raise ValueError("setwise tournament batch size must be at least two")
        self.client = client
        self.batch_size = batch_size
        self.group_batch_size = group_batch_size
        self.model_name = str(getattr(client, "model", "openai/gpt-oss-120b"))
        self.tournament_audits: list[dict[str, Any]] = []

    def select(
        self,
        trajectories: Sequence[TrajectoryPrediction],
        belief: CursorBeliefState,
    ) -> tuple[str, ...]:
        if not trajectories:
            return ()
        survivors = tuple(trajectories)
        rounds: list[dict[str, Any]] = []
        while len(survivors) > self.batch_size:
            next_survivors: list[TrajectoryPrediction] = []
            group_rows: list[dict[str, int]] = []
            groups = tuple(
                survivors[start : start + self.batch_size]
                for start in range(0, len(survivors), self.batch_size)
            )
            judged_groups = tuple(group for group in groups if len(group) > 1)
            decisions: list[tuple[str, ...]] = []
            for start in range(0, len(judged_groups), self.group_batch_size):
                batch = judged_groups[start : start + self.group_batch_size]
                decisions.extend(
                    self._select_group_batch(batch, belief)
                    if len(batch) > 1
                    else (self._select_one(batch[0], belief),)
                )
            decision_index = 0
            for group in groups:
                if len(group) == 1:
                    preferred = {group[0].trajectory_id}
                else:
                    preferred = set(decisions[decision_index])
                    decision_index += 1
                kept = (
                    tuple(row for row in group if row.trajectory_id in preferred)
                    if preferred
                    else tuple(group)
                )
                next_survivors.extend(kept)
                group_rows.append(
                    {
                        "input_count": len(group),
                        "survivor_count": len(kept),
                    }
                )
            rounds.append(
                {
                    "input_count": len(survivors),
                    "survivor_count": len(next_survivors),
                    "group_count": len(group_rows),
                    "groups": group_rows,
                }
            )
            if len(next_survivors) == len(survivors):
                self.tournament_audits.append(
                    {
                        "protocol": "complete_coverage_categorical_tournament",
                        "rounds": rounds,
                        "termination": "no_categorical_reduction_preserve_tie",
                    }
                )
                return tuple(row.trajectory_id for row in survivors)
            survivors = tuple(next_survivors)
        preferred = self._select_one(survivors, belief)
        self.tournament_audits.append(
            {
                "protocol": "complete_coverage_categorical_tournament",
                "rounds": rounds,
                "final_input_count": len(survivors),
                "final_preferred_count": len(preferred),
                "termination": "final_setwise_decision",
            }
        )
        return preferred

    def _select_group_batch(
        self,
        groups: Sequence[Sequence[TrajectoryPrediction]],
        belief: CursorBeliefState,
    ) -> tuple[tuple[str, ...], ...]:
        """Transport independent complete-set decisions in one model request."""

        group_aliases = tuple(
            _alphabetic_alias("group", index) for index in range(len(groups))
        )
        trajectory_aliases: list[dict[str, str]] = []
        group_payload: dict[str, Any] = {}
        for group_alias, group in zip(group_aliases, groups):
            aliases = {
                trajectory.trajectory_id: _alphabetic_alias(
                    f"{group_alias}_trajectory", index
                )
                for index, trajectory in enumerate(group)
            }
            trajectory_aliases.append(aliases)
            group_payload[group_alias] = {
                "trajectories": {
                    aliases[trajectory.trajectory_id]: [
                        _anonymous_preference_transition_payload(transition)
                        for transition in trajectory.transitions
                    ]
                    for trajectory in group
                }
            }
        payload = {
            "belief": {
                "question": belief.question,
                "answerability": belief.answerability.value,
                "missing_roles": list(belief.missing_roles),
                "contradictions": list(belief.contradictions),
                "acquired_evidence": list(belief.acquired_evidence),
            },
            "independent_groups": group_payload,
            "allowed_status": ["unique", "tie", "incomparable"],
            "required_output": {
                "only_keys": ["decisions"],
                "decisions": {
                    group_alias: {
                        "status": "unique, tie, or incomparable",
                        "preferred": [
                            "zero or more exact aliases from this group; always a JSON list"
                        ],
                        "rationale": "short categorical rationale",
                    }
                    for group_alias in group_aliases
                },
            },
            "required_contract": {
                "groups_are_independent": True,
                "every_candidate_in_every_group_must_be_judged": True,
                "categorical_preference_only": True,
                "no_action_or_graph_identifier_shortcut": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    "Independently compare every complete candidate set. Return exactly "
                    "one categorical decision for every supplied group; preserve ties and "
                    "incomparability."
                    if attempt == 0
                    else "Repair the output so decisions exactly cover every group schema."
                ),
                payload=payload,
            )
            try:
                if set(result) != {"decisions"} or not isinstance(
                    result.get("decisions"), dict
                ):
                    raise ValueError("grouped setwise output fields do not match schema")
                decision_payload = result["decisions"]
                if set(decision_payload) != set(group_aliases):
                    raise ValueError("grouped setwise decisions do not cover every group")
                parsed: list[tuple[str, ...]] = []
                for group_alias, aliases in zip(group_aliases, trajectory_aliases):
                    row = decision_payload[group_alias]
                    if not isinstance(row, dict) or set(row) != {
                        "status",
                        "preferred",
                        "rationale",
                    }:
                        raise ValueError("grouped setwise decision fields do not match schema")
                    status = str(row.get("status") or "")
                    preferred = _strings(row.get("preferred"), "preferred")
                    known = set(aliases.values())
                    if not set(preferred) <= known or len(preferred) != len(set(preferred)):
                        raise ValueError("grouped setwise decision has unknown aliases")
                    if status == "unique" and len(preferred) != 1:
                        raise ValueError("unique grouped preference requires one trajectory")
                    if status == "tie" and len(preferred) < 2:
                        raise ValueError("tie grouped preference requires multiple trajectories")
                    if status == "incomparable" and preferred:
                        raise ValueError("incomparable grouped preference must be empty")
                    if status not in {"unique", "tie", "incomparable"}:
                        raise ValueError("unknown grouped setwise status")
                    by_alias = {
                        alias: trajectory_id for trajectory_id, alias in aliases.items()
                    }
                    parsed.append(tuple(by_alias[alias] for alias in preferred))
                return tuple(parsed)
            except ValueError:
                if attempt == 1:
                    raise
        raise RuntimeError("unreachable grouped setwise repair state")

    def _select_one(
        self,
        trajectories: Sequence[TrajectoryPrediction],
        belief: CursorBeliefState,
    ) -> tuple[str, ...]:
        aliases = {
            trajectory.trajectory_id: _alphabetic_alias("trajectory", index)
            for index, trajectory in enumerate(trajectories)
        }
        payload = {
            "belief": {
                "question": belief.question,
                "answerability": belief.answerability.value,
                "missing_roles": list(belief.missing_roles),
                "contradictions": list(belief.contradictions),
                "acquired_evidence": list(belief.acquired_evidence),
            },
            "trajectories": {
                aliases[trajectory.trajectory_id]: [
                    _anonymous_preference_transition_payload(transition)
                    for transition in trajectory.transitions
                ]
                for trajectory in trajectories
            },
            "allowed_status": ["unique", "tie", "incomparable"],
            "required_output": {
                "only_keys": ["status", "preferred", "rationale"],
                "preferred": "list of exact trajectory aliases, empty only for incomparable",
            },
            "required_contract": {
                "complete_candidate_set_present": True,
                "categorical_preference_only": True,
                "no_action_or_graph_identifier_shortcut": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        status = ""
        preferred: tuple[str, ...] = ()
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    (
                        "Compare the complete set of imagined reasoning trajectories and "
                        "return the categorically preferred trajectory aliases. Use unique "
                        "only when one trajectory is strictly preferred; preserve ties or "
                        "incomparability instead of inventing a score."
                    )
                    if attempt == 0
                    else (
                        "Repair the categorical setwise result. Return status unique with "
                        "exactly one preferred alias, status tie with multiple preferred "
                        "aliases, or status incomparable with an empty preferred list."
                    )
                ),
                payload=payload,
            )
            try:
                if set(result) != {"status", "preferred", "rationale"}:
                    raise ValueError("setwise preference output fields do not match schema")
                status = str(result.get("status") or "")
                preferred = _strings(result.get("preferred"), "preferred")
                known = set(aliases.values())
                if not set(preferred) <= known or len(preferred) != len(set(preferred)):
                    raise ValueError("setwise preference contains unknown/duplicate aliases")
                if status == "unique" and len(preferred) != 1:
                    raise ValueError("unique setwise preference requires one trajectory")
                if status == "tie" and len(preferred) < 2:
                    raise ValueError("tie setwise preference requires multiple trajectories")
                if status == "incomparable" and preferred:
                    raise ValueError("incomparable setwise preference must be empty")
                if status not in {"unique", "tie", "incomparable"}:
                    raise ValueError("unknown setwise preference status")
                break
            except ValueError:
                if attempt == 1:
                    raise
        by_alias = {alias: trajectory_id for trajectory_id, alias in aliases.items()}
        return tuple(by_alias[alias] for alias in preferred)

    def _compare_one_batch(
        self,
        pairs: Sequence[TrajectoryPair],
        belief: CursorBeliefState,
    ) -> tuple[TrajectoryPreference, ...]:
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
                _anonymous_preference_transition_payload(value)
                for value in trajectory.transitions
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
    rows: Sequence[dict[str, Any]],
    *,
    normalization_audits: list[dict[str, Any]] | None = None,
) -> tuple[ImaginedTransition, ...]:
    predictions: list[ImaginedTransition] = []
    if len(rows) != len(requests):
        raise ValueError("world prediction row count mismatch")
    for request, row in zip(requests, rows):
        if not request.action.reads_evidence:
            predictions.append(
                ImaginedTransition(
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
            )
            continue
        raw_resolved = _strings(row.get("resolved_roles"), "resolved_roles")
        allowed_roles = set(request.belief.missing_roles)
        resolved = tuple(role for role in raw_resolved if role in allowed_roles)
        raw_relation_updates = _strings(
            row.get("relation_updates"), "relation_updates"
        )
        known_edges = (
            {edge.edge_id for edge in request.graph_input.correlation_edges}
            | {edge.edge_id for edge in request.graph_input.candidate_edges}
            | {edge.edge_id for edge in request.graph_input.temporal_edges}
            | {edge.edge_id for edge in request.graph_input.verified_relations}
        )
        relation_updates = tuple(
            edge_id for edge_id in raw_relation_updates if edge_id in known_edges
        )
        dropped_roles = tuple(
            role for role in raw_resolved if role not in allowed_roles
        )
        dropped_edges = tuple(
            edge_id for edge_id in raw_relation_updates if edge_id not in known_edges
        )
        if normalization_audits is not None and (dropped_roles or dropped_edges):
            normalization_audits.append(
                {
                    "action_id": request.action.action_id,
                    "dropped_resolved_roles": list(dropped_roles),
                    "dropped_relation_updates": list(dropped_edges),
                    "policy": "allowlist_intersection_no_semantic_mapping",
                }
            )
        predictions.append(
            ImaginedTransition(
                action=request.action,
                observation=PredictedObservation(
                    target_id=request.action.target_id,
                    outcome=EvidenceOutcome(str(row.get("outcome") or "inconclusive")),
                    descriptor=_target_descriptor(request),
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


def _target_descriptor(request: IWMRequest) -> tuple[str, ...]:
    """Bind an imagined observation to its action target, never to a model row."""

    target_id = request.action.target_id
    if target_id is None:
        return ()
    matches = tuple(
        view.key.semantic_key
        for view in request.graph_input.nodes
        if view.key.node_id == target_id
    )
    if len(matches) != 1:
        raise ValueError("action target must bind to exactly one visible node key")
    return matches


def _transition_payload(transition: ImaginedTransition) -> dict[str, Any]:
    return {
        "action_id": transition.action.action_id,
        "action_kind": transition.action.kind.value,
        "target_id": transition.action.target_id,
        "relation": transition.action.relation,
        "observation_outcome": transition.observation.outcome.value,
        "observation_descriptor": list(transition.observation.descriptor),
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


def _anonymous_preference_transition_payload(
    transition: ImaginedTransition,
) -> dict[str, Any]:
    """Hide graph/action identifiers from the trajectory comparator.

    The IWM sees the graph and action in order to predict their consequences.
    The preference model is intentionally restricted to those consequences, so
    it cannot implement a lexical action-ID or timestamp shortcut.
    """

    return {
        "observation_outcome": transition.observation.outcome.value,
        "observation_descriptor": list(transition.observation.descriptor),
        "belief_delta": {
            "progress": transition.belief_delta.progress.value,
            "answerability_after": transition.belief_delta.answerability_after.value,
            "frontier_change": transition.belief_delta.frontier_change.value,
            "contradiction_change": (
                transition.belief_delta.contradiction_change.value
            ),
            "resolved_roles": list(transition.belief_delta.resolved_roles),
            "opened_roles": list(transition.belief_delta.opened_roles),
            "relation_update_kinds": (
                ["present"] if transition.belief_delta.relation_updates else []
            ),
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
    # Some OpenAI-compatible providers serialize a singleton categorical array
    # as a scalar despite an explicit JSON contract.  Canonicalize that transport
    # variation here; downstream role/edge allowlists still enforce semantics.
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.lower() in {"", "none", "null", "n/a", "not_applicable"}:
            return ()
        return (normalized,)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be categorical strings or null")
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
