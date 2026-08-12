"""Analyze zero-shot IWM wiring, imagined transitions, and post-read correction."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Mapping, Sequence

from .multi_trajectory_cgbench import _pooled_transition_calibration


DIAGNOSTIC_SCHEMA = "steam-zero-shot-iwm-diagnostic/v0.1"


def analyze_zero_shot_artifacts(
    artifacts: Sequence[dict[str, Any]],
    hidden_key: dict[str, Any],
) -> dict[str, Any]:
    hidden = {str(row["case_id"]): row for row in hidden_key.get("cases") or ()}
    runs = [
        run
        for artifact in artifacts
        for run in artifact.get("runs") or ()
        if isinstance(run, dict)
    ]
    wm_runs = [run for run in runs if run.get("arm") == "world_model_guided"]
    transition_rows = [
        row
        for run in wm_runs
        for row in (
            run.get("empirical_audit", {})
            .get("transition_calibration", {})
            .get("rows", ())
        )
    ]
    postread_rows = [
        row
        for run in wm_runs
        for row in _postread_rows(run, hidden.get(str(run.get("case_id")), {}))
    ]
    imagined_rows = [
        row
        for run in wm_runs
        for step in run.get("steps") or ()
        for row in _conditioned_transitions(step)
        if row.get("action", {}).get("reads_evidence") is True
    ]
    violation_counts: Counter[str] = Counter()
    violating_rows = 0
    for row in imagined_rows:
        violations = _obvious_transition_violations(row)
        if violations:
            violating_rows += 1
            violation_counts.update(violations)

    divergences = [
        row
        for artifact in artifacts
        for row in artifact.get("action_divergence") or ()
        if row.get("candidate_arm") == "no_world_model"
        or row.get("arm") == "no_world_model"
    ]
    different_first = sum(
        tuple(row.get("reference_action_ids") or (None,))[0]
        != tuple(row.get("candidate_action_ids") or (None,))[0]
        for row in divergences
    )
    visibility = [
        row
        for artifact in artifacts
        for row in artifact.get("input_visibility_audits") or ()
    ]
    return {
        "schema_version": DIAGNOSTIC_SCHEMA,
        "artifact_count": len(artifacts),
        "world_model_case_count": len(wm_runs),
        "runtime_integrity": {
            "error_count": sum(len(row.get("errors") or ()) for row in artifacts),
            "method_failure_count": sum(
                len(row.get("method_failures") or ()) for row in artifacts
            ),
            "joint_chain_lossless_rate": _mean(
                run.get("empirical_audit", {})
                .get("joint_chain_coverage", {})
                .get("outcome_retention_rate")
                for run in wm_runs
            ),
            "post_read_belief_divergence_rate": _mean(
                run.get("empirical_audit", {})
                .get("post_read_belief_divergence", {})
                .get("divergence_rate")
                for run in wm_runs
            ),
            "observed_first_action_divergence_count": different_first,
            "visibility_leakage_count": sum(
                row.get("leakage_detected") is True for row in visibility
            ),
        },
        "pre_read_transition": {
            "row_count": len(transition_rows),
            "calibration": _pooled_transition_calibration(transition_rows),
            "imagined_conditioned_transition_count": len(imagined_rows),
            "obvious_semantic_contract_violation_count": violating_rows,
            "obvious_semantic_contract_violation_rate": (
                violating_rows / len(imagined_rows) if imagined_rows else None
            ),
            "violation_types": dict(sorted(violation_counts.items())),
            "note": (
                "Historical artifacts predate strict projected-state validation; "
                "the count is diagnostic and does not rewrite those traces."
            ),
        },
        "post_read_oracle_observation": _postread_summary(postread_rows),
        "qa": {
            "answer_accuracy": _mean(
                run.get("metrics", {}).get("answer_correct") for run in wm_runs
            ),
            "clue_recall": _mean(
                run.get("metrics", {}).get("clue_recall") for run in wm_runs
            ),
            "abstain_rate": _mean(
                run.get("metrics", {}).get("answer_abstained") for run in wm_runs
            ),
        },
        "input_visibility_audits": visibility,
        "training_performed": False,
    }


def _conditioned_transitions(step: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for path in step.get("decision", {}).get("imagined_paths") or ():
        outcomes = path.get("conditioned_outcomes") or ()
        if outcomes:
            for outcome in outcomes:
                yield from outcome.get("transitions") or ()
        else:
            yield from path.get("transitions") or ()


def _obvious_transition_violations(row: dict[str, Any]) -> tuple[str, ...]:
    observation = row.get("observation") or {}
    delta = row.get("belief_delta") or {}
    outcome = str(observation.get("outcome") or "")
    violations: list[str] = []
    if set(delta.get("resolved_roles") or ()) & set(delta.get("opened_roles") or ()):
        violations.append("role_resolved_and_opened")
    if outcome in {"empty", "inconclusive"}:
        if delta.get("resolved_roles"):
            violations.append("non_evidence_resolves_roles")
        if delta.get("progress") == "advanced":
            violations.append("non_evidence_advances")
        if delta.get("contradiction_change") == "resolved":
            violations.append("non_evidence_resolves_contradiction")
        if delta.get("answerability_after") == "ready":
            violations.append("non_evidence_claims_ready")
    return tuple(violations)


def _postread_rows(
    run: dict[str, Any],
    hidden: dict[str, Any],
) -> Iterable[dict[str, Any]]:
    answer = _normalize(hidden.get("answer_text"))
    realized = {
        str(row.get("observation_id")): row
        for row in run.get("realized_labels_evaluator_only") or ()
    }
    for index, step in enumerate(run.get("steps") or ()):
        trajectories = step.get("pool_before", {}).get("trajectories") or ()
        correct = next(
            (
                row
                for row in trajectories
                if _normalize(row.get("hypothesis")) == answer
            ),
            None,
        )
        if correct is None:
            continue
        trajectory_id = str(correct.get("trajectory_id"))
        assessment = next(
            (
                row
                for row in step.get("assessments") or ()
                if str(row.get("trajectory_id")) == trajectory_id
            ),
            {},
        )
        after = next(
            (
                row
                for row in step.get("pool_after", {}).get("trajectories") or ()
                if str(row.get("trajectory_id")) == trajectory_id
            ),
            {},
        )
        before_belief = correct.get("belief") or {}
        after_belief = after.get("belief") or {}
        observation_id = str(step.get("observation_id") or "")
        yield {
            "case_id": str(run.get("case_id")),
            "step_index": index,
            "observation_id": observation_id,
            "realized_outcome": (
                realized.get(observation_id, {}).get("observation_outcome")
            ),
            "correct_hypothesis_proposed_effect": assessment.get("effect"),
            "correct_hypothesis_effect": _effective_assessment(assessment),
            "effect_verification": assessment.get("verification"),
            "effect_target_binding": assessment.get("target_binding"),
            "effect_relation_scope": assessment.get("relation_scope"),
            "correct_hypothesis_status_after": after.get("status"),
            "missing_role_count_decreased": len(after_belief.get("missing_roles") or ())
            < len(before_belief.get("missing_roles") or ()),
            "answerability_after": after_belief.get("answerability"),
        }


def _postread_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    support = [row for row in rows if row.get("realized_outcome") == "support"]
    inconclusive = [
        row for row in rows if row.get("realized_outcome") == "inconclusive"
    ]
    eliminated = {"contradicted", "rejected", "terminated"}
    return {
        "scope": (
            "The model receives the executed real observation. This isolates "
            "evidence interpretation/correction and is never a navigation arm."
        ),
        "row_count": len(rows),
        "support_row_count": len(support),
        "correct_hypothesis_supported_or_complete_on_support_rate": _mean(
            row.get("correct_hypothesis_effect") in {"support", "complete"}
            for row in support
        ),
        "correct_hypothesis_survival_on_support_rate": _mean(
            row.get("correct_hypothesis_status_after") not in eliminated
            for row in support
        ),
        "correct_hypothesis_survival_on_inconclusive_rate": _mean(
            row.get("correct_hypothesis_status_after") not in eliminated
            for row in inconclusive
        ),
        "missing_role_reduction_on_support_rate": _mean(
            row.get("missing_role_count_decreased") for row in support
        ),
        "rows": list(rows),
        "hidden_feedback_to_planner": False,
    }


def _effective_assessment(assessment: Mapping[str, Any]) -> str | None:
    effect = assessment.get("effect")
    if effect == "inconclusive":
        return "inconclusive"
    if not assessment:
        return None
    if not {
        "verification",
        "target_binding",
        "relation_scope",
    }.issubset(assessment):
        return str(effect) if effect is not None else None
    if (
        assessment.get("verification") == "verified"
        and assessment.get("target_binding") == "same_target"
        and assessment.get("relation_scope") == "direct"
    ):
        return str(effect)
    return "inconclusive"


def _normalize(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _mean(values: Iterable[Any]) -> float | None:
    selected = [float(value) for value in values if value is not None]
    return fmean(selected) if selected else None


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_zero_shot_artifacts(
        [_read(path) for path in args.artifact],
        _read(args.hidden_key),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps({key: report[key] for key in ("runtime_integrity", "qa")}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
