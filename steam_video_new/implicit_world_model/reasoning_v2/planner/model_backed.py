"""Model-backed categorical selection over complete joint action trees."""

from __future__ import annotations

from typing import Any, Sequence

from ..world_model.model_backed import CategoricalJSONClient
from .contracts import (
    JointActionTree,
    PreferenceStatus,
    ReasoningPathForest,
    TrajectoryPreferenceDecision,
)


class ModelBackedJointTreePlanner:
    """Choose one shared first read only after seeing all path outcomes."""

    def __init__(
        self,
        client: CategoricalJSONClient,
        *,
        include_delayed_steps: bool = True,
    ) -> None:
        self.client = client
        self.model_name = str(client.model)
        self.include_delayed_steps = include_delayed_steps

    def choose(
        self,
        forest: ReasoningPathForest,
        candidates: Sequence[JointActionTree],
    ) -> TrajectoryPreferenceDecision:
        if not candidates:
            return TrajectoryPreferenceDecision(
                PreferenceStatus.ABSTAIN,
                (),
                None,
                "no executable joint action tree",
            )
        aliases = {f"tree_{index}": tree for index, tree in enumerate(candidates)}
        task = (
            "Compare the complete joint action trees for multi-path reasoning. "
            "Each tree contains all hypothesis-conditioned futures under one "
            "shared first evidence read. Return a categorical partial preference; "
            "do not emit scores or use candidate order as a tie-break."
        )
        payload = {
            "forest": _forest_payload(forest),
            "joint_action_trees": [
                _tree_payload(
                    row,
                    external_tree_id=alias,
                    include_delayed=self.include_delayed_steps,
                )
                for alias, row in aliases.items()
            ],
            "allowed_status": [value.value for value in PreferenceStatus],
            "decision_contract": {
                "only_top_level_keys": [
                    "status",
                    "frontier_tree_ids",
                    "selected_tree_id",
                    "rationale",
                ],
                "frontier_tree_ids": "non-empty subset of joint action tree IDs",
                "selected_tree_id": (
                    "one frontier ID only when status is select; otherwise null"
                ),
                "rationale": "short categorical comparison",
            },
        }
        request_payload = payload
        for attempt in range(3):
            result = self.client.complete_json(
                task=(
                    task
                    if attempt == 0
                    else "Repair the joint-tree preference JSON to exactly match decision_contract."
                ),
                payload=request_payload,
            )
            try:
                expected_keys = {
                    "status",
                    "frontier_tree_ids",
                    "selected_tree_id",
                    "rationale",
                }
                if set(result) != expected_keys:
                    raise ValueError("joint-tree planner fields do not match schema")
                status = _status(result.get("status"))
                frontier_aliases = _frontier(
                    result.get("frontier_tree_ids"), tuple(aliases)
                )
                selected_raw = result.get("selected_tree_id")
                selected_alias = selected_raw if isinstance(selected_raw, str) else None
                if status is PreferenceStatus.SELECT:
                    if selected_alias not in frontier_aliases:
                        raise ValueError(
                            "selected_tree_id must belong to frontier_tree_ids"
                        )
                elif selected_alias is not None:
                    raise ValueError("non-select planner decision cannot choose a tree")
                frontier = tuple(aliases[alias].tree_id for alias in frontier_aliases)
                selected = (
                    aliases[selected_alias].tree_id
                    if selected_alias is not None
                    else None
                )
                return TrajectoryPreferenceDecision(
                    status=status,
                    frontier_candidate_ids=frontier,
                    selected_candidate_id=selected,
                    rationale=str(result.get("rationale") or ""),
                )
            except ValueError as exc:
                if attempt == 2:
                    raise
                request_payload = {
                    **payload,
                    "repair_feedback": {
                        "validation_error": str(exc),
                        "invalid_response": result,
                    },
                }
        raise RuntimeError("unreachable joint-tree preference schema repair")


def _forest_payload(forest: ReasoningPathForest) -> dict[str, Any]:
    return {
        "step": str(forest.step),
        "remaining_reads": str(forest.shared_evidence.remaining_reads),
        "acquired_evidence_ids": list(forest.shared_evidence.acquired_ids),
        "paths": [
            {
                "path_id": path.path_id,
                "hypothesis": path.hypothesis,
                "status": path.status.value,
                "cursor_id": path.cursor_id,
                "missing_roles": list(path.belief.missing_roles),
                "contradictions": list(path.belief.contradictions),
                "answerability": path.belief.answerability.value,
                "action_history": list(path.action_history),
            }
            for path in forest.plannable
        ],
    }


def _tree_payload(
    tree: JointActionTree,
    *,
    external_tree_id: str,
    include_delayed: bool,
) -> dict[str, Any]:
    return {
        "tree_id": external_tree_id,
        "shared_first_read": {
            "target_id": tree.first_action.target_id,
            "source_id": tree.first_action.source_id,
            "proposal_kind": (
                tree.first_action.proposal_kind.value
                if tree.first_action.proposal_kind is not None
                else None
            ),
        },
        "covered_hypotheses": list(tree.covered_hypotheses),
        "hypothesis_conditioned_futures": [
            {
                "path_id": trajectory.root_path_id,
                "hypothesis": trajectory.hypothesis,
                "steps": [
                    {
                        "target_id": step.action.target_id,
                        "observation_kind": step.observation.descriptor.kind.value,
                        "event_family": step.observation.descriptor.event_family,
                        "progress": step.effect.progress.value,
                        "answerability_after": step.effect.answerability_after.value,
                        "contradiction_change": step.effect.contradiction_change.value,
                        "frontier_change": step.effect.frontier_change.value,
                        "resolved_roles": list(step.effect.resolved_roles),
                        "opened_roles": list(step.effect.opened_roles),
                    }
                    for step in (
                        trajectory.steps if include_delayed else trajectory.steps[:1]
                    )
                ],
            }
            for trajectory in tree.trajectories
        ],
    }


def _status(value: Any) -> PreferenceStatus:
    try:
        return PreferenceStatus(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid planner status: {value}") from exc


def _frontier(
    value: Any,
    known_ids: tuple[str, ...],
) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) for item in value)
    ):
        raise ValueError("frontier_tree_ids must be a non-empty string list")
    frontier = tuple(dict.fromkeys(value))
    known = set(known_ids)
    if not set(frontier).issubset(known):
        raise ValueError("frontier_tree_ids contains an unknown joint tree")
    return frontier
