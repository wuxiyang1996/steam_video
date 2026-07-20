"""Generate real graph-read sibling branches from one immutable checkpoint."""

from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import json
from pathlib import Path
from typing import Any

from memory_graph.navigation import NavigationActionType, propose_navigation_actions
from memory_graph.types import CausalTemporalOverlay

from .artifacts import (
    action_to_dict,
    belief_to_dict,
    transition_to_dict,
)
from .contracts import (
    BeliefBackend,
    BeliefDeltaDescriptor,
    BeliefSnapshot,
    GraphReadExecutor,
    ObservationBeliefWorldModel,
    PredictedTransition,
    TrajectoryPrediction,
    TrajectoryPreferenceModel,
    UncertaintyChange,
)


def generate_sibling_artifact(
    belief: BeliefSnapshot,
    overlay: CausalTemporalOverlay,
    *,
    backend: BeliefBackend,
    executor: GraphReadExecutor,
    world_model: ObservationBeliefWorldModel,
    preference_model: TrajectoryPreferenceModel,
    include_stop: bool = True,
) -> dict[str, Any]:
    """Execute every branch against persisted graph state, never imagined text."""

    actions = sorted(
        propose_navigation_actions(belief.navigation_view(), overlay),
        key=lambda action: (
            action.action_type is NavigationActionType.STOP,
            action.action_type.value,
            action.source_id or "",
            action.target_ids,
            action.relation or "",
        ),
    )
    if not include_stop:
        actions = [
            action
            for action in actions
            if action.action_type is not NavigationActionType.STOP
        ]

    branches: list[dict[str, Any]] = []
    trajectories: list[TrajectoryPrediction] = []
    for index, action in enumerate(actions):
        branch_id = f"branch:{index}"
        predicted = world_model.predict(belief, action, overlay)
        if action.action_type is NavigationActionType.STOP:
            observations = ()
            after = belief
            delta = BeliefDeltaDescriptor(
                uncertainty_change=UncertaintyChange.UNCHANGED,
                answerability_after=belief.answerability,
                predicted_only=False,
            )
            invocation: dict[str, object] = {
                "skill_id": "preference_stop",
                "args": {"action_type": "stop"},
                "outputs": {"real_observation_ids": []},
                "status": "executed",
            }
        else:
            execution = executor.execute(belief, action, overlay)
            observations = execution.observations
            update = backend.update(belief, action, list(observations), overlay)
            after = update.belief
            delta = update.delta
            invocation = execution.skill_invocation
        real_transition = PredictedTransition(
            action=action,
            observation=replace(
                predicted.observation,
                target_ids=tuple(node.node_id for node in observations),
                predicted_only=False,
            ),
            belief_delta=delta,
        )
        trajectory = TrajectoryPrediction(branch_id, (real_transition,))
        trajectories.append(trajectory)
        branches.append(
            {
                "branch_id": branch_id,
                "belief_before_id": belief.belief_id,
                "action": action_to_dict(action),
                "skill_invocation": invocation,
                "real_observation_ids": [node.node_id for node in observations],
                "belief_after": belief_to_dict(after),
                "realized_transition": transition_to_dict(real_transition),
            }
        )

    labels = [
        preference_model.compare(left, right, belief)
        for left, right in combinations(trajectories, 2)
    ]
    artifact = {
        "schema_version": "steam-preference-siblings/v0.1",
        "overlay_id": overlay.overlay_id,
        "example_id": overlay.example_id,
        "checkpoint": belief_to_dict(belief),
        "branches": branches,
        "pairwise_preferences": [
            {
                "left_branch_id": label.left_id,
                "right_branch_id": label.right_id,
                "label": label.label.value,
                "rationale": label.rationale,
                "label_source": "rule_based_provisional/v0.1",
            }
            for label in labels
        ],
        "annotation_status": "requires_independent_annotation",
        "allowed_labels": [
            "prefer_left",
            "tie",
            "prefer_right",
            "incomparable",
        ],
        "output_contract": "ordinal_preference_only",
    }
    require_valid_sibling_artifact(artifact)
    return artifact


def validate_sibling_artifact(payload: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover - shared project dependency
        raise RuntimeError("jsonschema is required for sibling artifacts") from exc
    schema = json.loads(
        (Path(__file__).resolve().parent / "sibling_trajectory.schema.json").read_text(
            encoding="utf-8"
        )
    )
    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: "
        f"{error.message}"
        for error in errors
    ]


def require_valid_sibling_artifact(payload: dict[str, Any]) -> None:
    errors = validate_sibling_artifact(payload)
    if errors:
        raise ValueError("invalid sibling artifact: " + "; ".join(errors[:5]))
