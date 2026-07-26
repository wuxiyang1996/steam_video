"""Categorical direct-policy baseline with no imagined world-model transition."""

from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Any

from .action_compiler import GraphActionCompiler
from .contracts import (
    ActionKind,
    AnswerabilityState,
    CategoricalBeliefDelta,
    CursorBeliefState,
    EvidenceOutcome,
    FullGraphPlanDecision,
    ImaginedTransition,
    LegalGraphAction,
    PreferenceLabel,
    PredictedObservation,
    ProgressChange,
    RetainedEvidenceGraph,
    TrajectoryPrediction,
    TrajectoryPreference,
)
from .gpt_oss import CategoricalJSONClient
from .model_input import build_iwm_graph_input, graph_input_to_categorical_payload


class GPTOSSReactiveGraphPlanner:
    """Compare current legal actions directly, without imagining future belief."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        action_compiler: GraphActionCompiler | None = None,
        max_action_pairs: int | None = None,
        preference_batch_size: int = 64,
        setwise: bool = False,
        execute_stable_ties: bool = False,
    ) -> None:
        if max_action_pairs is not None and max_action_pairs < 1:
            raise ValueError("max_action_pairs must be positive when supplied")
        self.client = client
        self.action_compiler = action_compiler or GraphActionCompiler()
        self.max_action_pairs = max_action_pairs
        if preference_batch_size < 1:
            raise ValueError("preference_batch_size must be positive")
        self.preference_batch_size = preference_batch_size
        self.setwise = setwise
        self.stable_tie_execution_requested = execute_stable_ties
        self.execute_stable_ties = False

    def plan(
        self,
        belief: CursorBeliefState,
        graph: RetainedEvidenceGraph,
    ) -> FullGraphPlanDecision:
        actions = self.action_compiler.compile(belief, graph)
        trajectories = tuple(_null_trajectory(action) for action in actions)
        if self.setwise:
            return self._plan_setwise(belief, graph, actions, trajectories)
        pairs = tuple(combinations(trajectories, 2))
        if self.max_action_pairs is not None and len(pairs) > self.max_action_pairs:
            return _resource_abstention(actions, trajectories)

        action_alias = {
            action.action_id: _alphabetic_alias("choice", index)
            for index, action in enumerate(actions)
        }
        base_payload = {
            "current_state": graph_input_to_categorical_payload(
                build_iwm_graph_input(belief, graph, actions)
            ),
            "actions": {
                action_alias[action.action_id]: {
                    "kind": action.kind.value,
                    "source_id": action.source_id,
                    "target_id": action.target_id,
                    "edge_id": action.edge_id,
                    "relation": action.relation,
                    "reads_evidence": action.reads_evidence,
                }
                for action in actions
            },
            "allowed_labels": [value.value for value in PreferenceLabel],
            "required_contract": {
                "current_real_belief_only": True,
                "no_imagined_transition": True,
                "one_categorical_comparison_per_pair": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        rows: list[dict[str, Any]] = []
        for start in range(0, len(pairs), self.preference_batch_size):
            pair_batch = pairs[start : start + self.preference_batch_size]
            comparison_aliases = tuple(
                _alphabetic_alias("comparison", index)
                for index in range(len(pair_batch))
            )
            payload = {
                **base_payload,
                "pairs": [
                    {
                        "comparison": comparison,
                        "left": action_alias[left.first_action.action_id],
                        "right": action_alias[right.first_action.action_id],
                    }
                    for comparison, (left, right) in zip(comparison_aliases, pair_batch)
                ],
            }
            batch_rows: list[dict[str, Any]] | None = None
            for attempt in range(2):
                task = (
                    (
                        "Compare each pair of currently legal reasoning actions using "
                        "only the current real belief and graph view. Do not predict a "
                        "future observation or belief transition."
                    )
                    if attempt == 0
                    else (
                        "Repair the response. Return one JSON object with the sole key "
                        "comparisons and exactly one categorical row per requested pair."
                    )
                )
                result = self.client.complete_json(task=task, payload=payload)
                candidate = result.get("comparisons")
                try:
                    if set(result) != {"comparisons"}:
                        raise ValueError(
                            "reactive output requires the sole key comparisons"
                        )
                    if not isinstance(candidate, list) or len(candidate) != len(
                        pair_batch
                    ):
                        raise ValueError("reactive comparison coverage mismatch")
                    if any(
                        not isinstance(row, dict) or row.get("comparison") != comparison
                        for comparison, row in zip(comparison_aliases, candidate)
                    ):
                        raise ValueError("reactive comparison order mismatch")
                    for row in candidate:
                        PreferenceLabel(str(row.get("label") or "incomparable"))
                    batch_rows = candidate
                    break
                except ValueError:
                    if attempt == 1:
                        raise
            assert batch_rows is not None
            rows.extend(batch_rows)

        preferences = tuple(
            TrajectoryPreference(
                left_id=left.trajectory_id,
                right_id=right.trajectory_id,
                label=PreferenceLabel(str(row.get("label") or "incomparable")),
                rationale=str(row.get("rationale") or "reactive categorical policy"),
            )
            for (left, right), row in zip(pairs, rows)
        )
        dominated: set[str] = set()
        for (left, right), preference in zip(pairs, preferences):
            if preference.label is PreferenceLabel.PREFER_LEFT:
                dominated.add(right.trajectory_id)
            elif preference.label is PreferenceLabel.PREFER_RIGHT:
                dominated.add(left.trajectory_id)
        undominated = tuple(
            trajectory.trajectory_id
            for trajectory in trajectories
            if trajectory.trajectory_id not in dominated
        )
        surviving = [
            trajectory
            for trajectory in trajectories
            if trajectory.trajectory_id in set(undominated)
        ]
        if len(surviving) == 1:
            selected = surviving[0].first_action
            status = "reactive_selected_unique_undominated_action"
        else:
            selected = next(
                action for action in actions if action.kind is ActionKind.ABSTAIN
            )
            status = (
                "reactive_abstain_preference_cycle"
                if not surviving
                else "reactive_abstain_non_unique_partial_order"
            )
        return FullGraphPlanDecision(
            selected_action=selected,
            planning_status=status,
            trajectories=trajectories,
            preferences=preferences,
            undominated_trajectory_ids=undominated,
            legal_action_count=len(actions),
            top_k_applied=False,
        )

    def _plan_setwise(
        self,
        belief: CursorBeliefState,
        graph: RetainedEvidenceGraph,
        actions: tuple[LegalGraphAction, ...],
        trajectories: tuple[TrajectoryPrediction, ...],
    ) -> FullGraphPlanDecision:
        aliases = {
            action.action_id: _alphabetic_alias("choice", index)
            for index, action in enumerate(actions)
        }
        payload = {
            "current_state": graph_input_to_categorical_payload(
                build_iwm_graph_input(belief, graph, actions)
            ),
            "actions": {
                aliases[action.action_id]: {
                    "kind": action.kind.value,
                    "source_id": action.source_id,
                    "target_id": action.target_id,
                    "edge_id": action.edge_id,
                    "relation": action.relation,
                    "reads_evidence": action.reads_evidence,
                }
                for action in actions
            },
            "allowed_status": ["unique", "tie", "incomparable"],
            "required_output": {
                "only_keys": ["status", "preferred", "rationale"],
                "preferred": "list of exact choice aliases",
            },
            "required_contract": {
                "complete_action_set_present": True,
                "current_real_belief_only": True,
                "no_imagined_transition": True,
                "categorical_preference_only": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
            },
        }
        by_alias = {
            alias: action
            for action, alias in ((a, aliases[a.action_id]) for a in actions)
        }
        status = ""
        preferred: list[str] = []
        for attempt in range(2):
            result = self.client.complete_json(
                task=(
                    (
                        "Compare the complete set of currently legal reasoning actions "
                        "using only the current real belief. Return a unique preferred "
                        "alias only when strictly justified; otherwise preserve ties or "
                        "incomparability."
                    )
                    if attempt == 0
                    else (
                        "Repair the categorical result: unique requires exactly one "
                        "alias, tie requires at least two aliases, and incomparable "
                        "requires an empty preferred list."
                    )
                ),
                payload=payload,
            )
            try:
                if set(result) != {"status", "preferred", "rationale"}:
                    raise ValueError("reactive setwise output fields do not match schema")
                status = str(result.get("status") or "")
                candidate = result.get("preferred")
                if not isinstance(candidate, list) or any(
                    not isinstance(value, str) for value in candidate
                ):
                    raise ValueError("reactive setwise preferred must be a string list")
                preferred = candidate
                if len(preferred) != len(set(preferred)):
                    raise ValueError("reactive setwise aliases must be unique")
                if not set(preferred) <= set(by_alias):
                    raise ValueError(
                        "reactive setwise output contains an unknown alias"
                    )
                if not (
                    (status == "unique" and len(preferred) == 1)
                    or (status == "tie" and len(preferred) >= 2)
                    or (status == "incomparable" and not preferred)
                ):
                    raise ValueError("reactive setwise status/preferred mismatch")
                break
            except ValueError:
                if attempt == 1:
                    raise
        if status == "unique":
            selected = by_alias[preferred[0]]
            planning_status = "reactive_setwise_selected_unique_action"
        elif status == "tie":
            selected = next(
                action for action in actions if action.kind is ActionKind.ABSTAIN
            )
            planning_status = "reactive_setwise_abstain_tied_actions"
        elif status == "incomparable":
            selected = next(action for action in actions if action.kind is ActionKind.ABSTAIN)
            planning_status = "reactive_setwise_abstain_incomparable"
        trajectory_by_action = {
            trajectory.first_action.action_id: trajectory for trajectory in trajectories
        }
        return FullGraphPlanDecision(
            selected_action=selected,
            planning_status=planning_status,
            trajectories=trajectories,
            preferences=(),
            undominated_trajectory_ids=tuple(
                trajectory_by_action[by_alias[alias].action_id].trajectory_id
                for alias in preferred
            ),
            legal_action_count=len(actions),
            top_k_applied=False,
        )


def _null_trajectory(action: LegalGraphAction) -> TrajectoryPrediction:
    digest = hashlib.sha256(action.action_id.encode("utf-8")).hexdigest()[:20]
    return TrajectoryPrediction(
        trajectory_id=f"reactive:{digest}",
        transitions=(
            ImaginedTransition(
                action=action,
                observation=PredictedObservation(
                    target_id=action.target_id,
                    outcome=EvidenceOutcome.INCONCLUSIVE,
                ),
                belief_delta=CategoricalBeliefDelta(
                    progress=ProgressChange.UNCHANGED,
                    answerability_after=AnswerabilityState.NOT_READY,
                ),
            ),
        ),
    )


def _resource_abstention(
    actions: tuple[LegalGraphAction, ...],
    trajectories: tuple[TrajectoryPrediction, ...],
) -> FullGraphPlanDecision:
    selected = next(action for action in actions if action.kind is ActionKind.ABSTAIN)
    return FullGraphPlanDecision(
        selected_action=selected,
        planning_status="reactive_abstain_exhaustive_comparison_budget_exceeded",
        trajectories=trajectories,
        preferences=(),
        undominated_trajectory_ids=tuple(
            trajectory.trajectory_id for trajectory in trajectories
        ),
        legal_action_count=len(actions),
        top_k_applied=False,
    )


def _alphabetic_alias(prefix: str, index: int) -> str:
    letters: list[str] = []
    value = index
    while True:
        value, remainder = divmod(value, 26)
        letters.append(chr(ord("a") + remainder))
        if value == 0:
            break
        value -= 1
    return f"{prefix}_{''.join(reversed(letters))}"
