"""Evaluate held-out categorical IWM transitions before Planner training.

The model prediction file contains labels only.  All numeric metrics and gate
decisions are computed by this evaluator and are never model outputs or reward
targets.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import TRANSITION_TASK, validate_sft_record


CALIBRATION_SCHEMA = "steam-iwm-transition-calibration/v0.1"
FIELDS = (
    "observation_outcome",
    "progress",
    "required_clue_coverage_after",
    "answerability_after",
)


def evaluate_transition_predictions(
    references: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    split: str = "validation",
    minimum_video_count: int = 5,
    minimum_outcome_balanced_accuracy: float = 0.60,
    minimum_progress_balanced_accuracy: float = 0.60,
    minimum_delayed_positive_recall: float = 0.50,
    minimum_delayed_positive_unit_recall: float = 0.50,
    maximum_semantic_hard_negative_support_rate: float = 0.20,
    maximum_ready_false_discovery_rate: float = 0.10,
) -> dict[str, Any]:
    """Return an auditable, evaluator-only transition calibration report."""

    selected: dict[str, Mapping[str, Any]] = {}
    for row in references:
        validate_sft_record(row)
        if row["task"] == TRANSITION_TASK and row["split"] == split:
            selected[str(row["record_id"])] = row
    prediction_by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: set[str] = set()
    for row in predictions:
        record_id = str(row.get("record_id") or "")
        prediction = row.get("prediction")
        if not record_id or not isinstance(prediction, Mapping):
            raise ValueError("each prediction requires record_id and prediction")
        if record_id in prediction_by_id:
            duplicates.add(record_id)
        prediction_by_id[record_id] = prediction
    if duplicates:
        raise ValueError("duplicate prediction record_ids: " + ", ".join(sorted(duplicates)))
    unknown = sorted(set(prediction_by_id) - set(selected))
    missing = sorted(set(selected) - set(prediction_by_id))

    field_pairs: dict[str, list[tuple[str, str]]] = defaultdict(list)
    delayed_pairs: list[tuple[str, str]] = []
    delayed_unit_predictions: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    hard_negative_pairs: list[tuple[str, str]] = []
    for record_id, reference in selected.items():
        prediction = prediction_by_id.get(record_id)
        if prediction is None:
            continue
        categorical = reference["target"]["categorical_audit"]
        for field in FIELDS:
            predicted = prediction.get(field)
            if not isinstance(predicted, str) or not predicted:
                raise ValueError(f"{record_id} prediction is missing categorical {field}")
            expected = categorical[field]
            if expected.get("supervised") is True:
                field_pairs[field].append((str(expected["value"]), predicted))
        outcome_pair = field_pairs["observation_outcome"][-1]
        slices = reference.get("data_slices") or {}
        if slices.get("delayed_positive") is True:
            delayed_pairs.append(outcome_pair)
            transition_input = reference["input"]
            delayed_unit_predictions[
                (
                    str(reference["video_id"]),
                    str(transition_input.get("question") or ""),
                    str((transition_input.get("action") or {}).get("target_id") or ""),
                )
            ].append(outcome_pair[1])
        if slices.get("semantic_neighbor_hard_negative") is True:
            hard_negative_pairs.append(outcome_pair)

    metrics = {field: _categorical_metrics(pairs) for field, pairs in field_pairs.items()}
    delayed_recall = _positive_recall(delayed_pairs, "support")
    delayed_unit_recall = (
        sum(all(value == "support" for value in values) for values in delayed_unit_predictions.values())
        / len(delayed_unit_predictions)
        if delayed_unit_predictions
        else None
    )
    hard_negative_support_rate = _prediction_rate(hard_negative_pairs, "support")
    ready_fdr = _false_discovery_rate(field_pairs["answerability_after"], "ready")
    coverage = len(selected) - len(missing)
    coverage_fraction = coverage / len(selected) if selected else 0.0
    video_count = len({str(row["video_id"]) for row in selected.values()})
    checks = {
        "held_out_records_present": bool(selected),
        "minimum_video_count": video_count >= minimum_video_count,
        "prediction_coverage_complete": not missing,
        "no_unknown_prediction_ids": not unknown,
        "outcome_balanced_accuracy": metrics.get("observation_outcome", {}).get(
            "balanced_accuracy", 0.0
        )
        >= minimum_outcome_balanced_accuracy,
        "progress_balanced_accuracy": metrics.get("progress", {}).get(
            "balanced_accuracy", 0.0
        )
        >= minimum_progress_balanced_accuracy,
        "delayed_positive_recall": delayed_recall is not None
        and delayed_recall >= minimum_delayed_positive_recall,
        "delayed_positive_unit_recall": delayed_unit_recall is not None
        and delayed_unit_recall >= minimum_delayed_positive_unit_recall,
        "semantic_hard_negative_support_rate": hard_negative_support_rate is not None
        and hard_negative_support_rate <= maximum_semantic_hard_negative_support_rate,
        "ready_false_discovery_rate": ready_fdr is None
        or ready_fdr <= maximum_ready_false_discovery_rate,
    }
    return {
        "schema_version": CALIBRATION_SCHEMA,
        "split": split,
        "reference_record_count": len(selected),
        "prediction_record_count": len(prediction_by_id),
        "evaluated_record_count": coverage,
        "prediction_coverage": coverage_fraction,
        "video_count": video_count,
        "missing_record_ids": missing,
        "unknown_record_ids": unknown,
        "categorical_metrics": metrics,
        "slice_metrics": {
            "delayed_positive_count": len(delayed_pairs),
            "delayed_positive_recall": delayed_recall,
            "delayed_positive_unit_count": len(delayed_unit_predictions),
            "delayed_positive_unit_recall": delayed_unit_recall,
            "delayed_positive_unit_requires_all_hypotheses": True,
            "semantic_hard_negative_count": len(hard_negative_pairs),
            "semantic_hard_negative_support_rate": hard_negative_support_rate,
            "ready_false_discovery_rate": ready_fdr,
        },
        "thresholds": {
            "minimum_video_count": minimum_video_count,
            "minimum_outcome_balanced_accuracy": minimum_outcome_balanced_accuracy,
            "minimum_progress_balanced_accuracy": minimum_progress_balanced_accuracy,
            "minimum_delayed_positive_recall": minimum_delayed_positive_recall,
            "minimum_delayed_positive_unit_recall": (
                minimum_delayed_positive_unit_recall
            ),
            "maximum_semantic_hard_negative_support_rate": (
                maximum_semantic_hard_negative_support_rate
            ),
            "maximum_ready_false_discovery_rate": maximum_ready_false_discovery_rate,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "blockers": [name for name, passed in checks.items() if not passed],
        "model_output_contract": "categorical_labels_only",
        "numeric_metrics_are_evaluator_only": True,
        "numeric_reward_present": False,
        "planner_training_authorized": all(checks.values()),
        "training_performed": False,
    }


def _categorical_metrics(pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    labels = sorted({value for pair in pairs for value in pair})
    confusion = {
        expected: {predicted: 0 for predicted in labels} for expected in labels
    }
    correct = 0
    recalls: list[float] = []
    for expected, predicted in pairs:
        confusion[expected][predicted] += 1
        correct += int(expected == predicted)
    for expected in labels:
        total = sum(confusion[expected].values())
        if total:
            recalls.append(confusion[expected][expected] / total)
    return {
        "count": len(pairs),
        "labels": labels,
        "accuracy": correct / len(pairs) if pairs else 0.0,
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else 0.0,
        "confusion_matrix": confusion,
    }


def _positive_recall(
    pairs: Sequence[tuple[str, str]], positive: str
) -> float | None:
    positives = [predicted for expected, predicted in pairs if expected == positive]
    if not positives:
        return None
    return sum(predicted == positive for predicted in positives) / len(positives)


def _prediction_rate(
    pairs: Sequence[tuple[str, str]], predicted_label: str
) -> float | None:
    if not pairs:
        return None
    return sum(predicted == predicted_label for _, predicted in pairs) / len(pairs)


def _false_discovery_rate(
    pairs: Sequence[tuple[str, str]], positive: str
) -> float | None:
    predicted_positive = [expected for expected, predicted in pairs if predicted == positive]
    if not predicted_positive:
        return None
    return sum(expected != positive for expected in predicted_positive) / len(
        predicted_positive
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            rows.append(value)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)
    report = evaluate_transition_predictions(
        _read_jsonl(args.records),
        _read_jsonl(args.predictions),
        split=args.split,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
