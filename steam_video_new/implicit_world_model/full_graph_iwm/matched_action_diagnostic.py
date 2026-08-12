"""Evaluate IWM predictions on identical initial-belief actions across arms."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .multi_trajectory_cgbench import _pooled_transition_calibration


MATCHED_ACTION_SCHEMA = "steam-iwm-matched-action-diagnostic/v0.1"
MODEL_ARMS = {
    "world_model_guided",
    "shuffled_world_model_prediction",
    "immediate_effect_only",
}


def analyze_matched_initial_actions(
    artifacts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Match predictions at step zero, before arm-specific reads can diverge."""

    runs = [
        run
        for artifact in artifacts
        for run in artifact.get("runs") or ()
        if isinstance(run, Mapping)
    ]
    truth: dict[tuple[str, str], dict[str, str]] = {}
    truth_conflicts: list[dict[str, Any]] = []
    for run in runs:
        steps = run.get("steps") or ()
        labels = run.get("realized_labels_evaluator_only") or ()
        if not steps or not labels:
            continue
        step = steps[0]
        action = step.get("decision", {}).get("selected_action") or {}
        key = (str(run.get("case_id") or ""), _action_key(action))
        label = next(
            (
                row
                for row in labels
                if str(row.get("observation_id") or "")
                == str(step.get("observation_id") or "")
            ),
            None,
        )
        if label is None:
            continue
        value = {
            "realized_observation_outcome": str(label.get("observation_outcome")),
            "realized_progress": str(label.get("belief_delta", {}).get("progress")),
            "realized_answerability": str(
                label.get("belief_delta", {}).get("answerability_after")
            ),
        }
        if key in truth and truth[key] != value:
            truth_conflicts.append(
                {"case_id": key[0], "action_key": key[1], "labels": [truth[key], value]}
            )
        else:
            truth[key] = value

    arm_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    conflicts: list[dict[str, Any]] = []
    candidate_counts: dict[str, int] = defaultdict(int)
    matched_counts: dict[str, int] = defaultdict(int)
    for run in runs:
        arm = str(run.get("arm") or "")
        if arm not in MODEL_ARMS or not (run.get("steps") or ()):
            continue
        case_id = str(run.get("case_id") or "")
        grouped = _initial_predictions(run["steps"][0])
        candidate_counts[arm] += len(grouped)
        for action_key, rows in grouped.items():
            realized = truth.get((case_id, action_key))
            if realized is None:
                continue
            matched_counts[arm] += 1
            fields = {
                "predicted_observation_outcome": {
                    str(row.get("observation", {}).get("outcome")) for row in rows
                },
                "predicted_progress": {
                    str(row.get("belief_delta", {}).get("progress")) for row in rows
                },
                "predicted_answerability": {
                    str(row.get("belief_delta", {}).get("answerability_after"))
                    for row in rows
                },
            }
            unstable = {
                name: sorted(values)
                for name, values in fields.items()
                if len(values) != 1
            }
            if unstable:
                conflicts.append(
                    {
                        "case_id": case_id,
                        "arm": arm,
                        "action_key": action_key,
                        "hypothesis_conditioned_conflicts": unstable,
                        "excluded_from_action_unit_calibration": True,
                    }
                )
                continue
            arm_rows[arm].append(
                {
                    **realized,
                    **{name: next(iter(values)) for name, values in fields.items()},
                }
            )

    scheduler = _scheduler_summary(runs)
    correction = _correction_summary(runs)
    arms = {}
    for arm in sorted(MODEL_ARMS):
        rows = arm_rows[arm]
        arms[arm] = {
            "candidate_initial_action_count": candidate_counts[arm],
            "matched_executed_action_count": matched_counts[arm],
            "hypothesis_invariant_action_count": len(rows),
            "excluded_hypothesis_dependent_count": matched_counts[arm] - len(rows),
            "calibration": _pooled_transition_calibration(rows),
        }
    return {
        "schema_version": MATCHED_ACTION_SCHEMA,
        "artifact_count": len(artifacts),
        "case_count": len({str(run.get("case_id") or "") for run in runs}),
        "truth_action_count": len(truth),
        "truth_conflicts": truth_conflicts,
        "arms": arms,
        "hypothesis_dependent_prediction_conflicts": conflicts,
        "scheduler_reliance": scheduler,
        "correction_safety": correction,
        "contracts": {
            "initial_belief_only": True,
            "same_case_same_action_only": True,
            "one_action_unit_not_hypothesis_replication": True,
            "hypothesis_dependent_observation_predictions_excluded": True,
            "hidden_labels_fed_back_to_planner": False,
        },
        "training_performed": False,
    }


def _initial_predictions(step: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for path in step.get("decision", {}).get("imagined_paths") or ():
        outcomes = path.get("conditioned_outcomes") or ()
        for outcome in outcomes:
            transitions = outcome.get("transitions") or ()
            if not transitions:
                continue
            transition = transitions[0]
            key = _action_key(transition.get("action") or {})
            hypothesis_key = "\x1f".join(
                (
                    str(outcome.get("trajectory_id") or ""),
                    str(outcome.get("hypothesis") or ""),
                )
            )
            grouped[key].setdefault(hypothesis_key, transition)
    return {key: list(rows.values()) for key, rows in grouped.items()}


def _action_key(action: Mapping[str, Any]) -> str:
    return "\x1f".join(
        str(action.get(key) or "")
        for key in ("kind", "source_id", "target_id", "edge_id", "relation")
    )


def _scheduler_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result = {}
    for arm in sorted(MODEL_ARMS | {"no_world_model"}):
        steps = [
            step
            for run in runs
            if run.get("arm") == arm
            for step in run.get("steps") or ()
        ]
        scheduled = sum(
            bool(
                step.get("decision", {})
                .get("preference_audit", {})
                .get("evidence_scheduler_used")
            )
            for step in steps
        )
        result[arm] = {
            "step_count": len(steps),
            "scheduler_step_count": scheduled,
            "scheduler_reliance_rate": scheduled / len(steps) if steps else None,
        }
    return result


def _correction_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    non_oracle = [run for run in runs if run.get("arm") != "oracle_clue_ceiling"]
    false_eliminations = [
        run
        for run in non_oracle
        if run.get("metrics", {}).get("false_correct_hypothesis_elimination") is True
    ]
    all_inconclusive = [
        run
        for run in false_eliminations
        if run.get("realized_labels_evaluator_only")
        and all(
            row.get("observation_outcome") == "inconclusive"
            for row in run.get("realized_labels_evaluator_only") or ()
        )
    ]
    return {
        "run_count": len(non_oracle),
        "false_correct_hypothesis_elimination_count": len(false_eliminations),
        "false_elimination_with_all_inconclusive_reads_count": len(all_inconclusive),
    }


def _read_artifact(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_matched_initial_actions(
        [_read_artifact(path) for path in args.artifact]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
