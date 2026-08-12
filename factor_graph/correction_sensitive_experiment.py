"""Multi-step correction-sensitive navigation controls over grounded fixtures."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.planner import (
    ClosedLoopNavigator,
    PersistedGraphReadExecutor,
    PreferenceOnlyPlanner,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.world_model import (
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)

from .navigation_backend import GTSAMExecutedReadBeliefBackend


MODES = ("correct", "verifier_only", "frozen", "shuffled")
EVENT_A = "event:atomic:video_skills_l1:00140"
EVENT_B = "event:atomic:video_skills_l1:00258"


def materialize_fixtures(source: Path, fixture_dir: Path) -> dict[str, Path]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    fixture_dir.mkdir(parents=True, exist_ok=True)
    outputs = {"support": source.resolve()}
    for variant in ("reject", "inconclusive"):
        derived = deepcopy(payload)
        derived["overlay_id"] = f"overlay:phase-e:correction-sensitive:{variant}:v1"
        derived.setdefault("metadata", {})["correction_sensitive_variant"] = variant
        derived["metadata"]["derived_from"] = str(source.resolve())
        identity = next(
            edge for edge in derived["relations"] if edge["edge_id"] == "phase-d:identity"
        )
        identity["relation_probabilities"]["same_entity"] = 0.55
        state = next(
            edge for edge in derived["relations"] if edge["edge_id"] == "phase-d:state"
        )
        state["relation_probabilities"]["state_transition"] = 0.55
        if variant == "reject":
            identity["provenance"]["participant_alignment"] = {
                "src": "l1-track:phase-d:mKqiGQrHtW8:man",
                "dst": "l1-track:evidence.entity_mention:ddaa4d819c",
            }
        else:
            identity["provenance"]["participant_alignment"] = {
                "src": "l1-track:phase-e:missing-src",
                "dst": "l1-track:phase-e:missing-dst",
            }
        path = fixture_dir / f"phase_e_correction_{variant}_overlay.json"
        path.write_text(
            json.dumps(derived, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        outputs[variant] = path.resolve()
    return outputs


def run_experiment(source: Path, fixture_dir: Path) -> dict[str, Any]:
    fixtures = materialize_fixtures(source, fixture_dir)
    cases = (
        {
            "case_id": "state-support-propagates-identity",
            "variant": "support",
            "question": "Who is the same person after the state change?",
            "missing_roles": ("identity",),
            "probe": GraphReadAction(
                NavigationActionType.INSPECT_STATE_CHANGE,
                source_id=EVENT_A,
                target_ids=(EVENT_B,),
                relation="state_transition",
                rationale="fixed grounded state probe",
            ),
            "expected_outcome": "supports",
        },
        {
            "case_id": "identity-reject-propagates-state-rejection",
            "variant": "reject",
            "question": "What state transition is valid for the same entity?",
            "missing_roles": ("state_transition",),
            "probe": GraphReadAction(
                NavigationActionType.TRACK_ENTITY,
                source_id=EVENT_A,
                target_ids=(EVENT_B,),
                relation="same_entity",
                rationale="fixed grounded mismatched-identity probe",
            ),
            "expected_outcome": "rejects",
        },
        {
            "case_id": "identity-inconclusive-negative-control",
            "variant": "inconclusive",
            "question": "What state transition is valid for the same entity?",
            "missing_roles": ("state_transition",),
            "probe": GraphReadAction(
                NavigationActionType.TRACK_ENTITY,
                source_id=EVENT_A,
                target_ids=(EVENT_B,),
                relation="same_entity",
                rationale="fixed grounded inconclusive-identity probe",
            ),
            "expected_outcome": "inconclusive",
        },
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        overlay = load_overlay_artifact(fixtures[case["variant"]]).overlay
        arms = {
            mode: _run_arm(case, overlay, mode)
            for mode in MODES
        }
        rows.append(
            {
                "case_id": case["case_id"],
                "variant": case["variant"],
                "expected_outcome": case["expected_outcome"],
                "arms": arms,
                "separate_diagnostics": {
                    "correct_vs_verifier_only_next_action_differs": (
                        arms["correct"]["next_action"]
                        != arms["verifier_only"]["next_action"]
                    ),
                    "correct_vs_frozen_next_action_differs": (
                        arms["correct"]["next_action"] != arms["frozen"]["next_action"]
                    ),
                    "correct_vs_shuffled_next_action_differs": (
                        arms["correct"]["next_action"] != arms["shuffled"]["next_action"]
                    ),
                    "correct_probe_propagated_change_count": len(
                        arms["correct"]["probe_audit"].get("propagated_changed_variables") or []
                    ),
                    "correct_probe_factor_activated": arms["correct"]["probe_audit"].get(
                        "factor_activated"
                    ),
                },
            }
        )
    return {
        "schema_version": "steam-gtsam-correction-sensitive-navigation/v0.2",
        "status": "derived_grounded_mechanism_pilot_not_gold",
        "production_ready": False,
        "llm_numeric_output": False,
        "source_fixture": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "cases": rows,
        "summary": {
            "case_count": len(rows),
            "support_propagated_belief_change": (
                rows[0]["separate_diagnostics"][
                    "correct_probe_propagated_change_count"
                ] > 0
            ),
            "reject_propagated_belief_change": (
                rows[1]["separate_diagnostics"][
                    "correct_probe_propagated_change_count"
                ] > 0
            ),
            "support_action_divergence": rows[0]["separate_diagnostics"][
                "correct_vs_verifier_only_next_action_differs"
            ],
            "reject_action_divergence": rows[1]["separate_diagnostics"][
                "correct_vs_verifier_only_next_action_differs"
            ],
            "inconclusive_is_negative_control": (
                not rows[2]["separate_diagnostics"][
                    "correct_vs_verifier_only_next_action_differs"
                ]
                and not rows[2]["separate_diagnostics"]["correct_probe_factor_activated"]
            ),
        },
        "limitations": [
            "Cases are transparently derived from grounded Phase D evidence, not independent human gold.",
            "Pilot measurement likelihoods are not calibrated production probabilities.",
            "This isolates mechanism sensitivity; it is not an answer-accuracy benchmark.",
            "Belief correction and action divergence are reported separately; neither is inferred from the other.",
        ],
    }


def _run_arm(case: dict[str, Any], overlay: Any, mode: str) -> dict[str, Any]:
    backend = GTSAMExecutedReadBeliefBackend(mode=mode)
    initial = backend.initialize(
        str(case["question"]),
        overlay,
        seed_evidence=(EVENT_A,),
        missing_roles=tuple(case["missing_roles"]),
        graph_read_budget=3,
    )
    executor = PersistedGraphReadExecutor()
    probe = case["probe"]
    execution = executor.execute(initial, probe, overlay)
    update = backend.update_from_execution(initial, probe, execution, overlay)
    planner = PreferenceOnlyPlanner(
        RuleBasedObservationBeliefModel(),
        RuleBasedTrajectoryPreferenceModel(),
        horizon=2,
    )
    continuation = ClosedLoopNavigator(backend, planner, executor).run(
        update.belief,
        overlay,
        max_steps=2,
    )
    next_action = (
        _action(continuation.steps[0].decision.selected_action)
        if continuation.steps
        else None
    )
    return {
        "mode": mode,
        "probe_action": _action(probe),
        "probe_outcome": (
            (update.audit_record or {}).get("verifier_decision") or {}
        ).get("outcome"),
        "probe_audit": update.audit_record,
        "belief_after_probe": {
            "missing_roles": list(update.belief.missing_roles),
            "answerability": update.belief.answerability.value,
            "priority_edge_ids": list(update.belief.priority_edge_ids),
            "blocked_edge_ids": list(update.belief.blocked_edge_ids),
        },
        "next_action": next_action,
        "continuation_actions": [
            _action(step.decision.selected_action) for step in continuation.steps
        ],
        "final_belief": {
            "missing_roles": list(continuation.final_belief.missing_roles),
            "answerability": continuation.final_belief.answerability.value,
            "contradictions": list(continuation.final_belief.contradictions),
        },
    }


def _action(action: GraphReadAction) -> dict[str, Any]:
    return {
        "action_type": action.action_type.value,
        "source_id": action.source_id,
        "target_ids": list(action.target_ids),
        "relation": action.relation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--fixture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_experiment(args.source, args.fixture_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
