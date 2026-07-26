"""Small model-backed v2 smoke with matched categorical intervention arms."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Sequence

from steam_video_new.implicit_world_model.full_graph_iwm.transition_cache import (
    PersistentCategoricalResponseCacheClient,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    OpenAICompatibleCategoricalClient,
)

from ..evidence import load_navigation_artifact
from ..navigation import ModelBackedEntryLocalizer
from ..planner import (
    ModelBackedJointTreePlanner,
    PersistentMultiPathPlanner,
    PreferenceStatus,
    TrajectoryPreferenceDecision,
    apply_real_read,
    initialize_forest,
)
from ..world_model import (
    AddressOnlyObservationBaseline,
    Answerability,
    BeliefEffect,
    BeliefState,
    ConservativeEffectBaseline,
    ModelBackedHypothesisEffectModel,
    ModelBackedObservationWorldModel,
    ModelBackedRealEffectCorrector,
)


DEFAULT_CASE_ID = "cgcase:03ae94a7899404983244"


class _OracleFirstReadPlanner:
    model_name = "evaluator-oracle-first-read"

    def __init__(self, expected_targets: tuple[str, ...]) -> None:
        self.expected_targets = set(expected_targets)

    def choose(self, forest, candidates):
        del forest
        selected = next(
            (
                tree
                for tree in candidates
                if tree.first_action.target_id in self.expected_targets
            ),
            None,
        )
        if selected is None:
            return TrajectoryPreferenceDecision(
                PreferenceStatus.ABSTAIN,
                tuple(tree.tree_id for tree in candidates),
                None,
                "expected clue is absent from the matched localized frontier",
            )
        return TrajectoryPreferenceDecision(
            PreferenceStatus.SELECT,
            (selected.tree_id,),
            selected.tree_id,
            "hidden evaluator oracle for headroom only",
        )


class _ShuffledEffectModel:
    def __init__(self, delegate: ModelBackedHypothesisEffectModel) -> None:
        self.delegate = delegate
        self.model_name = f"shuffled:{delegate.model_name}"

    def predict_effect_batch(self, requests) -> Sequence[BeliefEffect]:
        predictions = tuple(self.delegate.predict_effect_batch(requests))
        if len(predictions) < 2:
            return predictions
        rotated = predictions[1:] + predictions[:1]
        return tuple(
            replace(effect, request_id=request.request_id)
            for request, effect in zip(requests, rotated, strict=True)
        )


def run_smoke(
    *,
    client: Any,
    navigation_dataset_path: Path,
    gate_path: Path,
    graph_root: Path,
    case_id: str,
    output_path: Path,
) -> dict[str, Any]:
    case = _case(navigation_dataset_path, case_id)
    gate = _case(gate_path, case_id)
    graph_path = graph_root / str(case["video_id"]) / "l1_l15_navigation_graph.json"
    memory, navigation = load_navigation_artifact(graph_path)
    question = str(case["planner_input"]["question"])
    hypotheses = tuple(str(value) for value in case["planner_input"]["choices"])
    expected_targets = tuple(
        str(value) for value in (gate.get("retained_l1_node_ids_by_clue") or [[]])[0]
    )
    localizer = ModelBackedEntryLocalizer(client, maximum_anchors=2)
    entry_ids = localizer.localize(
        question=question,
        missing_roles=("first_item_identity", "answer_attributes"),
        memory=memory,
    )
    navigation = replace(navigation, entry_node_ids=entry_ids)
    belief = BeliefState(
        question,
        required_roles=("first_item_identity", "answer_attributes"),
        missing_roles=("first_item_identity", "answer_attributes"),
        answerability=Answerability.NOT_READY,
    )

    def fresh_forest():
        return initialize_forest(
            forest_id=f"forest:{case_id}",
            belief=belief,
            hypotheses=hypotheses,
            read_budget=2,
        )

    observation_model = ModelBackedObservationWorldModel(client)
    effect_model = ModelBackedHypothesisEffectModel(client)
    iwm_planner = PersistentMultiPathPlanner(
        observation_model,
        effect_model,
        ModelBackedJointTreePlanner(client, include_delayed_steps=True),
        horizon=2,
    )
    arms = {
        "iwm": iwm_planner,
        "no_wm": PersistentMultiPathPlanner(
            AddressOnlyObservationBaseline(),
            ConservativeEffectBaseline(),
            ModelBackedJointTreePlanner(client, include_delayed_steps=True),
            horizon=2,
        ),
        "shuffled_iwm": PersistentMultiPathPlanner(
            observation_model,
            _ShuffledEffectModel(effect_model),
            ModelBackedJointTreePlanner(client, include_delayed_steps=True),
            horizon=2,
        ),
        "immediate_only": PersistentMultiPathPlanner(
            observation_model,
            effect_model,
            ModelBackedJointTreePlanner(client, include_delayed_steps=False),
            horizon=2,
        ),
        "oracle": PersistentMultiPathPlanner(
            AddressOnlyObservationBaseline(),
            ConservativeEffectBaseline(),
            _OracleFirstReadPlanner(expected_targets),
            horizon=2,
        ),
    }
    arm_rows = {}
    decisions = {}
    for name, planner in arms.items():
        decision = planner.plan(fresh_forest(), memory, navigation)
        decisions[name] = decision
        arm_rows[name] = _decision_row(decision, expected_targets, hypotheses)

    first = decisions["iwm"]
    after_read = apply_real_read(
        fresh_forest(),
        first,
        memory,
        corrector=ModelBackedRealEffectCorrector(client),
    )
    replanned = iwm_planner.plan(after_read, memory, navigation)
    beliefs_by_hypothesis: dict[str, set[tuple[Any, ...]]] = {}
    for path in after_read.paths:
        beliefs_by_hypothesis.setdefault(path.hypothesis, set()).add(
            (
                path.belief.missing_roles,
                path.belief.contradictions,
                path.belief.answerability.value,
            )
        )
    signatures = {
        signature
        for hypothesis_signatures in beliefs_by_hypothesis.values()
        for signature in hypothesis_signatures
    }
    selected = {name: row["selected_target_id"] for name, row in arm_rows.items()}
    gates = {
        "model_closed_loop_executed": first.selected_action is not None,
        "replan_produced_candidates": bool(replanned.candidates),
        "joint_hypothesis_coverage": all(
            row["joint_hypothesis_coverage"] for row in arm_rows.values()
        )
        and all(row["candidate_tree_count"] > 0 for row in arm_rows.values()),
        "real_belief_hypotheses_diverged": len(signatures) > 1,
        "localized_entry_contains_first_clue": bool(
            set(entry_ids) & set(expected_targets)
        ),
        "iwm_first_action_matches_first_clue": arm_rows["iwm"][
            "matches_expected_first_clue"
        ],
        "action_diverges_from_no_wm": selected["iwm"] != selected["no_wm"],
        "shuffling_changes_action": selected["iwm"] != selected["shuffled_iwm"],
        "delayed_advantage_over_immediate": (
            arm_rows["iwm"]["matches_expected_first_clue"]
            and not arm_rows["immediate_only"]["matches_expected_first_clue"]
        ),
        "oracle_headroom_available": arm_rows["oracle"]["matches_expected_first_clue"],
        "answer_accuracy_available": False,
        "transition_calibration_available": False,
        "multi_video_fixed_cohort": False,
    }
    training_export_eligible = all(gates.values())
    result = {
        "schema_version": "steam-reasoning-v2-model-smoke/v0.1",
        "case_id": case_id,
        "video_id": case["video_id"],
        "model": str(client.model),
        "scope": "single_case_infrastructure_smoke_not_training_evidence",
        "entry_localization": localizer.audits[-1],
        "expected_first_clue_node_ids_hidden_evaluator_only": list(expected_targets),
        "arms": arm_rows,
        "replan": _decision_row(replanned, expected_targets, hypotheses),
        "gates": gates,
        "training_export_eligible": training_export_eligible,
        "training_performed": False,
        "response_audits": list(getattr(client, "response_audits", ())),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blocked = output_path.with_name("training_export.blocked.json")
    blocked.write_text(
        json.dumps(
            {
                "training_export_eligible": training_export_eligible,
                "training_performed": False,
                "blocking_gates": [
                    name for name, passed in gates.items() if not passed
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _decision_row(decision, expected_targets, hypotheses):
    selected = (
        decision.selected_action.target_id
        if decision.selected_action is not None
        else None
    )
    required = set(hypotheses)
    return {
        "status": decision.preference.status.value,
        "selected_target_id": selected,
        "matches_expected_first_clue": selected in set(expected_targets),
        "candidate_tree_count": len(decision.candidates),
        "observation_prediction_count": decision.observation_prediction_count,
        "effect_prediction_count": decision.effect_prediction_count,
        "joint_hypothesis_coverage": bool(decision.candidates)
        and all(
            set(tree.covered_hypotheses) == required for tree in decision.candidates
        ),
        "top_k_applied": decision.top_k_applied,
        "rationale": decision.preference.rationale,
    }


def _case(path: Path, case_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("records") or payload.get("cases") or ()
    try:
        return next(row for row in rows if row.get("case_id") == case_id)
    except StopIteration as exc:
        raise ValueError(f"case is absent from {path}: {case_id}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--keys-py", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--navigation-dataset", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    delegate = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=240,
        max_tokens=16000,
        reasoning_effort="low",
    )
    client = PersistentCategoricalResponseCacheClient(
        delegate,
        args.output.with_suffix(args.output.suffix + ".responses.json"),
        mode="record",
    )
    result = run_smoke(
        client=client,
        navigation_dataset_path=args.navigation_dataset,
        gate_path=args.gate,
        graph_root=args.graph_root,
        case_id=args.case_id,
        output_path=args.output,
    )
    print(json.dumps(result["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
