"""Engineering finalization pilot for the optional categorical GTSAM backup."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from memory_graph.navigation import GraphReadAction, NavigationActionType
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.planner import (
    PersistedGraphReadExecutor,
)

from .navigation_backend import GTSAMExecutedReadBeliefBackend


EVENT_A = "event:atomic:video_skills_l1:00140"
EVENT_B = "event:atomic:video_skills_l1:00258"


def run_experiment(fixtures: dict[str, Path]) -> dict[str, Any]:
    definitions = {
        "support": GraphReadAction(
            NavigationActionType.INSPECT_STATE_CHANGE,
            EVENT_A,
            (EVENT_B,),
            "state_transition",
        ),
        "reject": GraphReadAction(
            NavigationActionType.TRACK_ENTITY,
            EVENT_A,
            (EVENT_B,),
            "same_entity",
        ),
        "inconclusive": GraphReadAction(
            NavigationActionType.TRACK_ENTITY,
            EVENT_A,
            (EVENT_B,),
            "same_entity",
        ),
    }
    rows: list[dict[str, Any]] = []
    restart_equal = False
    for name in ("support", "reject", "inconclusive"):
        path = fixtures[name].expanduser().resolve()
        overlay = load_overlay_artifact(path).overlay
        backend = GTSAMExecutedReadBeliefBackend(mode="backup")
        initial = backend.initialize(
            "Which categorical correction is warranted?",
            overlay,
            seed_evidence=(EVENT_A,),
            missing_roles=(
                "state_transition" if name == "support" else "identity",
            ),
            graph_read_budget=2,
        )
        action = definitions[name]
        execution = PersistedGraphReadExecutor().execute(initial, action, overlay)
        update = backend.update_from_execution(initial, action, execution, overlay)
        audit = update.audit_record or {}
        checkpoint = backend.export_checkpoint(update.belief)
        restored_backend = GTSAMExecutedReadBeliefBackend(mode="backup")
        restored = restored_backend.restore_checkpoint(checkpoint, overlay)
        case_restart_equal = _categorical_belief(restored) == _categorical_belief(
            update.belief
        )
        restart_equal = case_restart_equal if not rows else restart_equal and case_restart_equal
        rows.append(
            {
                "case": name,
                "fixture": str(path),
                "fixture_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "verifier_outcome": (audit.get("verifier_decision") or {}).get(
                    "outcome"
                ),
                "measurement_status": audit.get("measurement_status"),
                "backup_trigger": audit.get("backup_trigger"),
                "factor_activated": audit.get("factor_activated"),
                "changed_variables": audit.get("changed_variables"),
                "categorical_belief_after": _categorical_belief(update.belief),
                "checkpoint_restart_equal": case_restart_equal,
                "checkpoint_numeric_solver_state_exposed": checkpoint[
                    "numeric_solver_state_exposed"
                ],
            }
        )
    by_name = {row["case"]: row for row in rows}
    gates = {
        "support_does_not_activate_backup": (
            by_name["support"]["measurement_status"] == "backup_not_triggered"
            and by_name["support"]["factor_activated"] is False
            and by_name["support"]["categorical_belief_after"]["missing_roles"]
            == []
        ),
        "reject_activates_backup_factor": (
            by_name["reject"]["measurement_status"] == "backup_activated"
            and by_name["reject"]["factor_activated"] is True
        ),
        "inconclusive_trigger_adds_no_factor": (
            by_name["inconclusive"]["measurement_status"] == "backup_activated"
            and by_name["inconclusive"]["factor_activated"] is False
        ),
        "checkpoint_restart_categorical_parity": restart_equal,
        "numeric_solver_state_not_exported": all(
            row["checkpoint_numeric_solver_state_exposed"] is False for row in rows
        ),
    }
    return {
        "schema_version": "steam-gtsam-backup-finalization/v0.1",
        "status": "engineering_complete_not_production_calibrated",
        "backup_mode": "iwm_with_gtsam_backup",
        "llm_numeric_output": False,
        "pilot_calibration_only": True,
        "cases": rows,
        "gates": gates,
        "backup_engineering_complete": all(gates.values()),
        "production_calibrated": False,
        "navigation_benefit_proven": False,
    }


def _categorical_belief(belief: Any) -> dict[str, Any]:
    return {
        "acquired_evidence": list(belief.acquired_evidence),
        "missing_roles": list(belief.missing_roles),
        "contradictions": list(belief.contradictions),
        "blocked_edge_ids": list(belief.blocked_edge_ids),
        "answerability": belief.answerability.value,
        "uncertainty": belief.uncertainty.value,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support", required=True, type=Path)
    parser.add_argument("--reject", required=True, type=Path)
    parser.add_argument("--inconclusive", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = run_experiment(
        {
            "support": args.support,
            "reject": args.reject,
            "inconclusive": args.inconclusive,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
