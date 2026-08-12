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
from .complete_graph import (
    audit_clue_grounded_values,
    build_entry_frontiers,
    shortest_path_distance,
    shortest_path_next_hops,
    with_entry_frontier,
)


DEFAULT_CASE_ID = "cgcase:03ae94a7899404983244"


class _OracleRoutePlanner:
    model_name = "evaluator-oracle-route"

    def __init__(self, first_targets, later_targets, navigation) -> None:
        self.first_targets = tuple(first_targets)
        self.later_targets = tuple(later_targets)
        self.navigation = navigation

    def choose(self, forest, candidates):
        expected = self.first_targets
        if forest.step > 0:
            expected = shortest_path_next_hops(
                self.navigation,
                source_ids=forest.shared_evidence.acquired_ids[-1:],
                goal_ids=self.later_targets,
            )
        selected = next(
            (
                tree
                for tree in candidates
                if tree.first_action.target_id in set(expected)
            ),
            None,
        )
        if selected is None:
            return TrajectoryPreferenceDecision(
                PreferenceStatus.ABSTAIN,
                tuple(tree.tree_id for tree in candidates),
                None,
                "evaluator route target is absent from the legal frontier",
            )
        return TrajectoryPreferenceDecision(
            PreferenceStatus.SELECT,
            (selected.tree_id,),
            selected.tree_id,
            "hidden evaluator shortest-route oracle for headroom only",
        )


class _ShuffledEffectModel:
    def __init__(self, delegate: ModelBackedHypothesisEffectModel) -> None:
        self.delegate = delegate
        self.model_name = f"shuffled:{delegate.model_name}"
        self.batch_count = 0
        self.changed_effect_count = 0

    def predict_effect_batch(self, requests) -> Sequence[BeliefEffect]:
        predictions = tuple(self.delegate.predict_effect_batch(requests))
        if len(predictions) < 2:
            return predictions
        self.batch_count += 1
        rotated = predictions[1:] + predictions[:1]
        shuffled = tuple(
            replace(effect, request_id=request.request_id)
            for request, effect in zip(requests, rotated, strict=True)
        )
        self.changed_effect_count += sum(
            _effect_signature(before) != _effect_signature(after)
            for before, after in zip(predictions, shuffled, strict=True)
        )
        return shuffled


def run_smoke(
    *,
    client: Any,
    navigation_dataset_path: Path,
    gate_path: Path,
    graph_root: Path,
    case_id: str,
    output_path: Path,
    model_batch_size: int = 8,
    read_budget: int = 2,
    terminal_targets_path: Path | None = None,
) -> dict[str, Any]:
    if read_budget < 2:
        raise ValueError("complete-graph smoke requires at least two real reads")
    case = _case(navigation_dataset_path, case_id)
    gate = _case(gate_path, case_id)
    graph_path = graph_root / str(case["video_id"]) / "l1_l15_navigation_graph.json"
    memory, navigation = load_navigation_artifact(graph_path)
    question = str(case["planner_input"]["question"])
    hypotheses = tuple(str(value) for value in case["planner_input"]["choices"])
    clue_groups = tuple(
        tuple(str(value) for value in group)
        for group in (gate.get("retained_l1_node_ids_by_clue") or ())
    )
    expected_targets = clue_groups[0] if clue_groups else ()
    later_targets = clue_groups[1] if len(clue_groups) > 1 else ()
    expected_answer = None
    if terminal_targets_path is not None:
        expected_answer = (
            str(
                _case(terminal_targets_path, case_id).get("answer_text") or ""
            ).strip()
            or None
        )
    localizer = ModelBackedEntryLocalizer(client, maximum_anchors=2)
    frontiers = build_entry_frontiers(
        question=question,
        missing_roles=("first_item_identity", "answer_attributes"),
        memory=memory,
        expected_first_clue_ids=expected_targets,
        localizer=localizer,
    )
    belief = BeliefState(
        question,
        required_roles=("first_item_identity", "answer_attributes"),
        missing_roles=("first_item_identity", "answer_attributes"),
        answerability=Answerability.NOT_READY,
    )

    observation_model = ModelBackedObservationWorldModel(
        client, batch_size=model_batch_size
    )
    effect_model = ModelBackedHypothesisEffectModel(
        client, batch_size=model_batch_size
    )
    clue_value_audit = audit_clue_grounded_values(memory, clue_groups)
    protocols = {}
    audit_start = len(getattr(client, "response_audits", ()))
    for frontier in frontiers:
        protocol_navigation = with_entry_frontier(navigation, frontier)
        protocols[frontier.protocol.value] = _run_protocol(
            client=client,
            case_id=case_id,
            memory=memory,
            navigation=protocol_navigation,
            frontier=frontier,
            belief=belief,
            hypotheses=hypotheses,
            expected_targets=expected_targets,
            later_targets=later_targets,
            observation_model=observation_model,
            effect_model=effect_model,
            expected_answer=expected_answer,
            read_budget=read_budget,
        )
    aggregate_metrics = {
        "complete_graph_preserved_both_protocols": all(
            row["complete_graph"]["preserved"] for row in protocols.values()
        ),
        "clue_grounded_values_enriched": clue_value_audit[
            "all_clue_groups_have_enriched_value"
        ],
        "oracle_entry_closed_loop_executed": protocols["oracle_entry"]["gates"][
            "iwm_matched_budget_closed_loop_executed"
        ],
        "oracle_entry_replan_moves_toward_later_clue_shortest_path": protocols[
            "oracle_entry"
        ]["gates"]["iwm_replan_moves_toward_later_clue_shortest_path"],
        "oracle_entry_route_budget_feasible": protocols["oracle_entry"]["gates"][
            "oracle_route_budget_feasible"
        ],
        "oracle_entry_later_clue_reached_strict_interval": protocols[
            "oracle_entry"
        ]["gates"]["iwm_later_clue_reached_strict_interval"],
        "oracle_entry_real_beliefs_diverged": protocols["oracle_entry"]["gates"][
            "iwm_real_beliefs_diverged"
        ],
        "learned_entry_contains_first_clue_strict_interval": protocols[
            "learned_entry"
        ]["gates"]["entry_contains_first_clue_strict_interval"],
        "learned_entry_closed_loop_executed": protocols["learned_entry"]["gates"][
            "iwm_matched_budget_closed_loop_executed"
        ],
        "learned_entry_iwm_first_action_matches_first_clue_strict_interval": protocols[
            "learned_entry"
        ]["gates"]["iwm_first_action_matches_first_clue_strict_interval"],
        "oracle_entry_iwm_answer_correct": protocols["oracle_entry"]["gates"][
            "iwm_answer_correct"
        ],
        "learned_entry_iwm_answer_correct": protocols["learned_entry"]["gates"][
            "iwm_answer_correct"
        ],
        "answer_accuracy_available": expected_answer is not None,
        "transition_calibration_available": False,
        "multi_video_fixed_cohort": False,
    }
    aggregate_gates = {
        name: aggregate_metrics[name]
        for name in (
            "complete_graph_preserved_both_protocols",
            "clue_grounded_values_enriched",
            "oracle_entry_closed_loop_executed",
            "oracle_entry_route_budget_feasible",
            "oracle_entry_later_clue_reached_strict_interval",
            "oracle_entry_iwm_answer_correct",
            "learned_entry_closed_loop_executed",
            "learned_entry_iwm_answer_correct",
            "answer_accuracy_available",
            "transition_calibration_available",
            "multi_video_fixed_cohort",
        )
    }
    metric_families = _metric_families(
        protocols,
        expected_answer=expected_answer,
        read_budget=read_budget,
    )
    training_export_eligible = all(aggregate_gates.values())
    result = {
        "schema_version": "steam-reasoning-v2-complete-graph-smoke/v0.3",
        "case_id": case_id,
        "video_id": case["video_id"],
        "model": str(client.model),
        "model_batch_size": model_batch_size,
        "matched_read_budget": read_budget,
        "scope": "dual-entry-complete-graph-smoke_not_training_evidence",
        "complete_graph": {
            "node_count": len(navigation.node_ids),
            "proposal_count": len(navigation.proposals),
            "question_independent": bool(memory.metadata.get("question_independent")),
            "unread_evidence_values_exposed": False,
        },
        "expected_first_clue_node_ids_hidden_evaluator_only": list(expected_targets),
        "later_clue_node_ids_hidden_evaluator_only": list(later_targets),
        "expected_answer_hidden_evaluator_only": expected_answer,
        "clue_value_audit_hidden_evaluator_only": clue_value_audit,
        "protocols": protocols,
        "metrics": aggregate_metrics,
        "metric_families": metric_families,
        "gates": aggregate_gates,
        "training_export_eligible": training_export_eligible,
        "training_performed": False,
        "response_audits": list(getattr(client, "response_audits", ()))[audit_start:],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    blocked = output_path.with_suffix(
        output_path.suffix + ".training_export.blocked.json"
    )
    blocked.write_text(
        json.dumps(
            {
                "training_export_eligible": training_export_eligible,
                "training_performed": False,
                "blocking_gates": [
                    name for name, passed in aggregate_gates.items() if not passed
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _run_protocol(
    *,
    client,
    case_id,
    memory,
    navigation,
    frontier,
    belief,
    hypotheses,
    expected_targets,
    later_targets,
    observation_model,
    effect_model,
    expected_answer,
    read_budget,
):
    def fresh_forest():
        return initialize_forest(
            forest_id=f"forest:{case_id}:{frontier.protocol.value}",
            belief=belief,
            hypotheses=hypotheses,
            read_budget=read_budget,
        )

    arms = {
        "iwm": PersistentMultiPathPlanner(
            observation_model,
            effect_model,
            ModelBackedJointTreePlanner(client, include_delayed_steps=True),
            horizon=2,
        ),
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
            _OracleRoutePlanner(expected_targets, later_targets, navigation),
            horizon=2,
        ),
    }
    corrector = ModelBackedRealEffectCorrector(client)
    arm_rows = {
        name: _run_closed_loop_arm(
            planner=planner,
            forest=fresh_forest(),
            memory=memory,
            navigation=navigation,
            corrector=corrector,
            hypotheses=hypotheses,
            first_targets=expected_targets,
            later_targets=later_targets,
            read_budget=read_budget,
        )
        for name, planner in arms.items()
    }
    for row in arm_rows.values():
        row["first_unique_answer_read"] = _first_answer_read(
            row["answer_candidates_after_each_real_read"]
        )
        row["first_correct_answer_read_hidden_evaluator_only"] = (
            _first_answer_read(
                row["answer_candidates_after_each_real_read"],
                expected_answer=expected_answer,
            )
            if expected_answer is not None
            else None
        )
    iwm = arm_rows["iwm"]
    route_distance = shortest_path_distance(
        navigation,
        source_ids=frontier.node_ids,
        goal_ids=later_targets,
    )
    minimum_reads_to_later_clue = (
        route_distance + 1 if route_distance is not None else None
    )
    sequences = {
        name: tuple(row["executed_target_ids"]) for name, row in arm_rows.items()
    }
    return {
        "entry": frontier.audit,
        "complete_graph": {
            "node_count": len(navigation.node_ids),
            "proposal_count": len(navigation.proposals),
            "preserved": bool(navigation.metadata.get("complete_graph_preserved")),
            "only_entry_frontier_changed": True,
        },
        "arms": arm_rows,
        "route_budget_audit_hidden_evaluator_only": {
            "entry_to_later_clue_edge_distance": route_distance,
            "minimum_reads_including_entry": minimum_reads_to_later_clue,
            "matched_read_budget": read_budget,
            "budget_feasible": minimum_reads_to_later_clue is not None
            and minimum_reads_to_later_clue <= read_budget,
        },
        "gates": {
            "entry_contains_first_clue_strict_interval": bool(
                set(frontier.node_ids) & set(expected_targets)
            ),
            "iwm_first_action_matches_first_clue_strict_interval": iwm["rounds"][0][
                "matches_expected_target"
            ],
            "iwm_matched_budget_closed_loop_executed": len(
                iwm["executed_target_ids"]
            )
            == read_budget,
            "iwm_replan_moves_toward_later_clue_shortest_path": iwm["rounds"][1][
                "matches_expected_target"
            ],
            "iwm_later_clue_reached_strict_interval": bool(
                set(iwm["executed_target_ids"]) & set(later_targets)
            ),
            "iwm_real_beliefs_diverged": iwm["real_beliefs_diverged"],
            "iwm_first_real_read_changes_belief": bool(
                iwm["belief_changed_after_each_real_read"]
                and iwm["belief_changed_after_each_real_read"][0]
            ),
            "iwm_answer_available": len(iwm["final_answer_candidates"]) == 1,
            "iwm_answer_correct": (
                expected_answer is not None
                and iwm["final_answer_candidates"] == [expected_answer]
            ),
            "iwm_action_sequence_diverges_from_no_wm": sequences["iwm"]
            != sequences["no_wm"],
            "shuffling_changes_action_sequence": sequences["iwm"]
            != sequences["shuffled_iwm"],
            "delayed_changes_action_sequence": sequences["iwm"]
            != sequences["immediate_only"],
            "oracle_route_budget_feasible": minimum_reads_to_later_clue is not None
            and minimum_reads_to_later_clue <= read_budget,
            "oracle_later_clue_reached_strict_interval": bool(
                set(arm_rows["oracle"]["executed_target_ids"])
                & set(later_targets)
            ),
        },
    }


def _run_closed_loop_arm(
    *,
    planner,
    forest,
    memory,
    navigation,
    corrector,
    hypotheses,
    first_targets,
    later_targets,
    read_budget,
):
    rows = []
    executed = []
    any_divergence = False
    belief_summaries = []
    belief_changes = []
    answer_candidates = []
    for round_index in range(read_budget):
        if round_index == 0:
            expected = first_targets
        else:
            expected = shortest_path_next_hops(
                navigation,
                source_ids=tuple(executed[-1:]),
                goal_ids=later_targets,
            )
        decision = planner.plan(forest, memory, navigation)
        row = _decision_row(decision, expected, hypotheses)
        rows.append(row)
        if decision.selected_action is None:
            break
        executed.append(str(decision.selected_action.target_id))
        before_summary = _belief_summary(forest)
        forest = apply_real_read(
            forest,
            decision,
            memory,
            corrector=corrector,
        )
        any_divergence = any_divergence or _beliefs_diverged(forest)
        after_summary = _belief_summary(forest)
        belief_summaries.append(after_summary)
        belief_changes.append(after_summary != before_summary)
        answer_candidates.append(_answer_candidates(forest))
    while len(rows) < read_budget:
        rows.append(_missing_round_row())
    return {
        "rounds": rows,
        "executed_target_ids": executed,
        "executed_read_count": len(executed),
        "remaining_read_budget": forest.shared_evidence.remaining_reads,
        "real_beliefs_diverged": any_divergence,
        "belief_after_each_real_read": belief_summaries,
        "belief_changed_after_each_real_read": belief_changes,
        "answer_candidates_after_each_real_read": answer_candidates,
        "final_answer_candidates": answer_candidates[-1] if answer_candidates else [],
        "top_k_applied": forest.top_k_applied,
        "intervention_audit": _intervention_audit(planner),
    }


def _beliefs_diverged(forest):
    by_hypothesis: dict[str, set[tuple[Any, ...]]] = {}
    for path in forest.paths:
        by_hypothesis.setdefault(path.hypothesis, set()).add(
            (
                path.belief.missing_roles,
                path.belief.contradictions,
                path.belief.answerability.value,
            )
        )
    signatures = {
        signature
        for hypothesis_signatures in by_hypothesis.values()
        for signature in hypothesis_signatures
    }
    return len(signatures) > 1


def _answer_candidates(forest):
    return sorted(
        {
            path.hypothesis
            for path in forest.paths
            if path.status.value == "active"
            and path.belief.answerability is Answerability.READY
            and not path.belief.contradictions
        }
    )


def _first_answer_read(answer_rows, *, expected_answer=None):
    for read_index, candidates in enumerate(answer_rows, start=1):
        if expected_answer is None and len(candidates) == 1:
            return read_index
        if expected_answer is not None and candidates == [expected_answer]:
            return read_index
    return None


def _metric_families(protocols, *, expected_answer, read_budget):
    def arm(protocol, name):
        return protocols[protocol]["arms"][name]

    return {
        "answer_accuracy": {
            "available": expected_answer is not None,
            "oracle_entry_iwm_correct": protocols["oracle_entry"]["gates"][
                "iwm_answer_correct"
            ],
            "learned_entry_iwm_correct": protocols["learned_entry"]["gates"][
                "iwm_answer_correct"
            ],
        },
        "read_efficiency": {
            "matched_read_budget": read_budget,
            "oracle_entry_iwm_first_correct_answer_read": arm(
                "oracle_entry", "iwm"
            )["first_correct_answer_read_hidden_evaluator_only"],
            "learned_entry_iwm_first_correct_answer_read": arm(
                "learned_entry", "iwm"
            )["first_correct_answer_read_hidden_evaluator_only"],
        },
        "action_divergence": {
            protocol: {
                comparison: (
                    arm(protocol, "iwm")["executed_target_ids"]
                    != arm(protocol, comparison)["executed_target_ids"]
                )
                for comparison in ("no_wm", "shuffled_iwm", "immediate_only")
            }
            for protocol in ("oracle_entry", "learned_entry")
        },
        "strict_localization": {
            "oracle_entry_contains_first_interval": protocols["oracle_entry"][
                "gates"
            ]["entry_contains_first_clue_strict_interval"],
            "learned_entry_contains_first_interval": protocols["learned_entry"][
                "gates"
            ]["entry_contains_first_clue_strict_interval"],
        },
    }


def _belief_summary(forest):
    by_hypothesis: dict[str, set[tuple[Any, ...]]] = {}
    for path in forest.paths:
        by_hypothesis.setdefault(path.hypothesis, set()).add(
            (
                path.belief.missing_roles,
                path.belief.contradictions,
                path.belief.answerability.value,
            )
        )
    return {
        hypothesis: [
            {
                "missing_roles": list(signature[0]),
                "contradictions": list(signature[1]),
                "answerability": signature[2],
            }
            for signature in sorted(signatures, key=repr)
        ]
        for hypothesis, signatures in sorted(by_hypothesis.items())
    }


def _intervention_audit(planner):
    effect_model = planner.effect_model
    if not isinstance(effect_model, _ShuffledEffectModel):
        return None
    return {
        "kind": "within_batch_effect_rotation_preserving_marginals",
        "batch_count": effect_model.batch_count,
        "changed_effect_count": effect_model.changed_effect_count,
        "effective": effect_model.changed_effect_count > 0,
    }


def _effect_signature(effect):
    return (
        effect.progress,
        effect.answerability_after,
        effect.contradiction_change,
        effect.frontier_change,
        effect.resolved_roles,
        effect.opened_roles,
    )


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
        "matches_expected_target": selected in set(expected_targets),
        "expected_target_ids_hidden_evaluator_only": list(expected_targets),
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


def _missing_round_row():
    return {
        "status": "not_executed",
        "selected_target_id": None,
        "matches_expected_target": False,
        "expected_target_ids_hidden_evaluator_only": [],
        "candidate_tree_count": 0,
        "observation_prediction_count": 0,
        "effect_prediction_count": 0,
        "joint_hypothesis_coverage": False,
        "top_k_applied": False,
        "rationale": "previous round did not select an executable action",
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
    parser.add_argument("--model-batch-size", type=int, default=8)
    parser.add_argument("--read-budget", type=int, default=2)
    parser.add_argument("--terminal-targets", type=Path)
    parser.add_argument("--cache-mode", choices=("record", "replay"), default="record")
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
        mode=args.cache_mode,
    )
    result = run_smoke(
        client=client,
        navigation_dataset_path=args.navigation_dataset,
        gate_path=args.gate,
        graph_root=args.graph_root,
        case_id=args.case_id,
        output_path=args.output,
        model_batch_size=args.model_batch_size,
        read_budget=args.read_budget,
        terminal_targets_path=args.terminal_targets,
    )
    print(json.dumps(result["gates"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
