"""Combine case-isolated matched pilots and export unreviewed executed traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

from .multi_trajectory_cgbench import MULTI_ARMS, _aggregate
from .multi_trajectory_cohort import _slug


def analyze_pilot(
    *,
    integrity_path: Path,
    hidden_cohort_gate_path: Path,
    cases_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    integrity = _read(integrity_path)
    hidden_gate = _read(hidden_cohort_gate_path)
    case_ids = list(integrity.get("integrity_pass_case_ids") or ())
    delayed_ids = {
        str(row["case_id"])
        for row in hidden_gate.get("cases") or ()
        if row.get("structural_delayed_candidate") is True
    }
    case_artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for case_id in case_ids:
        path = cases_dir / f"{_slug(case_id)}.json"
        if not path.is_file():
            missing.append(case_id)
            continue
        artifact = _read(path)
        if artifact.get("selected_case_ids") != [case_id]:
            raise ValueError(f"case artifact identity mismatch: {path}")
        case_artifacts.append(artifact)
    if missing:
        raise ValueError("missing completed case artifacts: " + ", ".join(missing))

    model = _consistent_model_name(case_artifacts)

    runs = [run for artifact in case_artifacts for run in artifact.get("runs") or ()]
    errors = [
        row for artifact in case_artifacts for row in artifact.get("errors") or ()
    ]
    method_failures = [
        row
        for artifact in case_artifacts
        for row in artifact.get("method_failures") or ()
    ]
    if errors:
        raise ValueError("completed pilot contains runtime arm errors")
    expected_run_count = len(case_ids) * len(MULTI_ARMS)
    if len(runs) != expected_run_count:
        raise ValueError("completed pilot does not contain every matched arm")
    by_case_arm = {(str(run["case_id"]), str(run["arm"])): run for run in runs}
    if len(by_case_arm) != expected_run_count:
        raise ValueError("pilot contains duplicate case-arm runs")

    delayed_case_ids = [case_id for case_id in case_ids if case_id in delayed_ids]
    control_case_ids = [case_id for case_id in case_ids if case_id not in delayed_ids]
    delayed_runs = [run for run in runs if str(run["case_id"]) in delayed_ids]
    control_runs = [run for run in runs if str(run["case_id"]) not in delayed_ids]
    divergences = [
        row
        for artifact in case_artifacts
        for row in artifact.get("action_divergence") or ()
    ]
    wm_runs = [run for run in runs if run.get("arm") == "world_model_guided"]
    localized_preflight = [
        row
        for artifact in case_artifacts
        for row in artifact.get("localized_entry_preflight") or ()
    ]
    if not localized_preflight:
        localized_preflight = [
            {
                "case_id": str(run["case_id"]),
                "clue_coverage_complete": bool(
                    run.get("metrics", {}).get("clue_coverage_complete")
                ),
                "derived_from_legacy_oracle_run": True,
                "evaluator_only": True,
                "fed_back_to_planner": False,
            }
            for run in runs
            if run.get("arm") == "oracle_clue_ceiling"
        ]
    localized_evaluable_ids = {
        str(row["case_id"])
        for row in localized_preflight
        if row.get("clue_coverage_complete") is True
    }
    localized_evaluable_runs = [
        run
        for run in runs
        if str(run["case_id"]) in localized_evaluable_ids
    ]
    method_failure_case_ids = sorted({str(row["case_id"]) for row in method_failures})
    integrity_exclusion_count = len(integrity.get("excluded_cases") or ())
    summary = {
        "schema_version": "steam-matched-pilot-analysis/v0.2",
        "model": model,
        "requested_case_count": int(
            integrity.get("requested_case_count") or len(case_ids)
        ),
        "integrity_pass_case_count": len(case_ids),
        "integrity_exclusion_count": integrity_exclusion_count,
        "completed_case_count": len(case_artifacts),
        "matched_arm_count": len(MULTI_ARMS),
        "completed_case_arm_count": len(runs),
        "runtime_error_count": len(errors),
        "method_failure_count": len(method_failures),
        "method_failure_case_ids": method_failure_case_ids,
        "delayed_case_ids": delayed_case_ids,
        "control_case_ids": control_case_ids,
        "metrics_by_arm": _aggregate(runs, MULTI_ARMS),
        "metrics_by_arm_localized_frontier_complete": _aggregate(
            localized_evaluable_runs, MULTI_ARMS
        ),
        "localized_entry_preflight": {
            "evaluable_case_ids": sorted(localized_evaluable_ids),
            "insufficient_case_ids": sorted(set(case_ids) - localized_evaluable_ids),
            "evaluator_only": True,
            "fed_back_to_planner": False,
        },
        "delayed_metrics_by_arm": _aggregate(delayed_runs, MULTI_ARMS),
        "control_metrics_by_arm": _aggregate(control_runs, MULTI_ARMS),
        "action_divergence": _divergence_summary(divergences),
        "world_model_behavior": {
            "cases_with_real_read": sum(
                int(run["metrics"].get("real_read_count") or 0) > 0 for run in wm_runs
            ),
            "cases_with_clue_hit": sum(
                float(run["metrics"].get("clue_recall") or 0.0) > 0 for run in wm_runs
            ),
            "cases_answered_without_abstention": sum(
                not bool(run["metrics"].get("answer_abstained")) for run in wm_runs
            ),
            "cases_with_complete_joint_chain_retention": sum(
                run["metrics"].get("joint_chain_outcome_retention_rate") == 1.0
                for run in wm_runs
            ),
        },
        "scientific_gate": {
            "case_artifacts_complete": True,
            "runtime_errors_zero": not errors,
            "method_failures_zero": not method_failures,
            "matched_arms_complete": len(runs) == expected_run_count,
            "infrastructure_complete": (
                not errors and not method_failures and len(runs) == expected_run_count
            ),
            "world_model_beats_no_world_model_accuracy": _strictly_greater(
                _metric(runs, "world_model_guided", "answer_correct"),
                _metric(runs, "no_world_model", "answer_correct"),
            ),
            "world_model_beats_immediate_only_delayed_success": _strictly_greater(
                _metric(
                    delayed_runs,
                    "world_model_guided",
                    "delayed_reasoning_success",
                ),
                _metric(
                    delayed_runs,
                    "immediate_effect_only",
                    "delayed_reasoning_success",
                ),
            ),
            "scientific_validation_passed": False,
        },
        "limitations": [
            f"{integrity_exclusion_count} requested cases were excluded by the frozen integrity audit",
            "method-failure cases are retained and reported fail-closed",
            "the pilot is too small for significance claims",
            "exported traces are unreviewed and prohibited from training",
        ],
        "training_performed": False,
    }
    candidates = {
        "schema_version": "steam-executed-iwm-trace-candidates/v0.2",
        "model": model,
        "review_status": "unreviewed",
        "training_allowed": False,
        "hidden_evaluator_labels_included": False,
        "records": _executed_candidates(wm_runs, cases_dir),
        "transition_over_crediting_candidates": (
            _transition_over_crediting_candidates(wm_runs, cases_dir)
        ),
        "over_crediting_contract": {
            "uses_hidden_clues_or_answers": False,
            "reference": "post-read corrected hypothesis belief",
            "labels_are_categorical": True,
            "requires_independent_review_before_training": True,
        },
        "training_performed": False,
    }
    return summary, candidates


def _consistent_model_name(case_artifacts: Iterable[dict[str, Any]]) -> str:
    models = {
        str(artifact.get("model") or "").strip()
        for artifact in case_artifacts
        if str(artifact.get("model") or "").strip()
    }
    if not models:
        raise ValueError("completed pilot artifacts do not declare a model")
    if len(models) != 1:
        raise ValueError(
            "completed pilot mixes model identities: " + ", ".join(sorted(models))
        )
    return next(iter(models))


def _divergence_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    arms = sorted({str(row["candidate_arm"]) for row in rows})
    for arm in arms:
        selected = [row for row in rows if row["candidate_arm"] == arm]
        result[arm] = {
            "case_count": len(selected),
            "action_divergence_rate": (
                fmean(bool(row.get("action_diverged")) for row in selected)
                if selected
                else None
            ),
            "first_step_divergence_count": sum(
                row.get("first_divergence_step") == 0 for row in selected
            ),
        }
    return result


def _metric(
    runs: list[dict[str, Any]],
    arm: str,
    key: str,
) -> float | None:
    values = [
        float(run["metrics"][key])
        for run in runs
        if run.get("arm") == arm and run["metrics"].get(key) is not None
    ]
    return fmean(values) if values else None


def _strictly_greater(left: float | None, right: float | None) -> bool:
    return left is not None and right is not None and left > right


def _executed_candidates(
    wm_runs: Iterable[dict[str, Any]],
    cases_dir: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for run in wm_runs:
        case_id = str(run["case_id"])
        source = cases_dir / f"{_slug(case_id)}.json"
        for step_index, step in enumerate(run.get("steps") or ()):
            observation_id = step.get("observation_id")
            if not observation_id:
                continue
            decision = step["decision"]
            record_id = hashlib.sha256(
                f"{case_id}\x1f{step_index}\x1f{observation_id}".encode()
            ).hexdigest()[:20]
            records.append(
                {
                    "record_id": f"executed_trace:{record_id}",
                    "case_id": case_id,
                    "step_index": step_index,
                    "source_artifact": str(source.resolve()),
                    "input": {
                        "trajectory_pool_before": step["pool_before"],
                        "candidate_joint_chains": decision.get("imagined_paths") or [],
                        "planning_horizon": decision.get("planning_horizon"),
                    },
                    "teacher_output": {
                        "preferred_path_ids": decision.get("preferred_path_ids") or [],
                        "selected_action": decision.get("selected_action"),
                        "planning_status": decision.get("planning_status"),
                    },
                    "executed_real_observation_id": observation_id,
                    "corrected_trajectory_pool_after": step["pool_after"],
                    "review_status": "unreviewed",
                    "training_allowed": False,
                }
            )
    return records


def _transition_over_crediting_candidates(
    wm_runs: Iterable[dict[str, Any]],
    cases_dir: Path,
) -> list[dict[str, Any]]:
    """Compare selected imagined deltas with correction, without hidden GT."""

    records: list[dict[str, Any]] = []
    for run in wm_runs:
        case_id = str(run["case_id"])
        source = cases_dir / f"{_slug(case_id)}.json"
        for step_index, step in enumerate(run.get("steps") or ()):
            if not step.get("observation_id"):
                continue
            decision = step.get("decision") or {}
            selected = decision.get("selected_action") or {}
            before = {
                str(row["trajectory_id"]): row
                for row in (step.get("pool_before") or {}).get("trajectories") or ()
            }
            after = {
                str(row["trajectory_id"]): row
                for row in (step.get("pool_after") or {}).get("trajectories") or ()
            }
            seen: set[tuple[str, str, tuple[str, ...]]] = set()
            for path in decision.get("imagined_paths") or ():
                transitions = path.get("transitions") or ()
                if not transitions or not _same_action(
                    transitions[0].get("action") or {}, selected
                ):
                    continue
                for outcome in path.get("conditioned_outcomes") or ():
                    trajectory_id = str(outcome.get("trajectory_id") or "")
                    predicted_rows = outcome.get("transitions") or ()
                    if not trajectory_id or not predicted_rows:
                        continue
                    predicted = predicted_rows[0].get("belief_delta") or {}
                    resolved_roles = tuple(
                        str(row) for row in predicted.get("resolved_roles") or ()
                    )
                    key = (
                        trajectory_id,
                        str(predicted.get("answerability_after") or ""),
                        resolved_roles,
                    )
                    if key in seen or trajectory_id not in before or trajectory_id not in after:
                        continue
                    seen.add(key)
                    before_belief = before[trajectory_id].get("belief") or {}
                    after_belief = after[trajectory_id].get("belief") or {}
                    remaining_after = set(after_belief.get("missing_roles") or ())
                    unsupported_resolutions = sorted(
                        role for role in resolved_roles if role in remaining_after
                    )
                    predicted_ready = (
                        predicted.get("answerability_after") == "ready"
                    )
                    corrected_ready = after_belief.get("answerability") == "ready"
                    before_signature = (
                        tuple(before_belief.get("missing_roles") or ()),
                        tuple(before_belief.get("contradictions") or ()),
                        before_belief.get("answerability"),
                    )
                    after_signature = (
                        tuple(after_belief.get("missing_roles") or ()),
                        tuple(after_belief.get("contradictions") or ()),
                        after_belief.get("answerability"),
                    )
                    predicted_advanced = predicted.get("progress") == "advanced"
                    if (
                        (predicted_ready and not corrected_ready)
                        or unsupported_resolutions
                        or (predicted_advanced and before_signature == after_signature)
                    ):
                        label = "over_crediting"
                    elif predicted.get("progress") == "unchanged" and (
                        before_signature != after_signature
                    ):
                        label = "under_crediting"
                    elif predicted_ready == corrected_ready and not unsupported_resolutions:
                        label = "consistent"
                    else:
                        label = "inconclusive"
                    record_id = hashlib.sha256(
                        f"{case_id}\x1f{step_index}\x1f{trajectory_id}".encode()
                    ).hexdigest()[:20]
                    records.append(
                        {
                            "record_id": f"transition_audit:{record_id}",
                            "case_id": case_id,
                            "step_index": step_index,
                            "trajectory_id": trajectory_id,
                            "source_artifact": str(source.resolve()),
                            "executed_real_observation_id": step["observation_id"],
                            "predicted_belief_delta": predicted,
                            "corrected_belief_before": before_belief,
                            "corrected_belief_after": after_belief,
                            "provisional_label": label,
                            "unsupported_predicted_resolved_roles": (
                                unsupported_resolutions
                            ),
                            "hidden_evaluator_labels_included": False,
                            "review_status": "unreviewed",
                            "training_allowed": False,
                        }
                    )
    return records


def _same_action(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left.get("kind"),
        left.get("source_id"),
        left.get("target_id"),
        left.get("edge_id"),
    ) == (
        right.get("kind"),
        right.get("source_id"),
        right.get("target_id"),
        right.get("edge_id"),
    )


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--integrity", required=True, type=Path)
    parser.add_argument("--hidden-cohort-gate", required=True, type=Path)
    parser.add_argument("--cases-dir", required=True, type=Path)
    parser.add_argument("--analysis-output", required=True, type=Path)
    parser.add_argument("--candidates-output", required=True, type=Path)
    args = parser.parse_args(argv)
    analysis, candidates = analyze_pilot(
        integrity_path=args.integrity,
        hidden_cohort_gate_path=args.hidden_cohort_gate,
        cases_dir=args.cases_dir,
    )
    _write(args.analysis_output, analysis)
    _write(args.candidates_output, candidates)
    print(
        json.dumps(
            {
                "completed_case_count": analysis["completed_case_count"],
                "runtime_error_count": analysis["runtime_error_count"],
                "candidate_count": len(candidates["records"]),
                "scientific_validation_passed": analysis["scientific_gate"][
                    "scientific_validation_passed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
