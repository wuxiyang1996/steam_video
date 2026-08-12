"""Evaluate compact observation descriptors separately from belief effects."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schemas import OBSERVATION_TASK, validate_sft_record


def evaluate_observation_predictions(
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    minimum_video_count: int = 5,
    minimum_event_accuracy: float = 0.5,
    minimum_full_descriptor_accuracy: float = 0.25,
) -> dict[str, Any]:
    references = {
        str(row["record_id"]): row
        for row in records
        if row.get("task") == OBSERVATION_TASK
        and row.get("split") == "validation"
        and row.get("representation_eligible") is True
    }
    predicted = {str(row["record_id"]): row for row in predictions}
    unknown = sorted(set(predicted) - set(references))
    missing = sorted(set(references) - set(predicted))
    counts = defaultdict(int)
    slice_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    numeric_outputs = 0
    for record_id, reference in references.items():
        counts["count"] += 1
        row = predicted.get(record_id) or {}
        numeric_outputs += int(row.get("model_output_contains_numeric_scalar") is True)
        prediction = row.get("prediction", {}).get("observation_descriptor")
        if not isinstance(prediction, Mapping):
            continue
        counts["valid"] += 1
        target = reference["target"]["observation_descriptor"]["value"]
        event_match = _normalize_text(prediction.get("event")) == _normalize_text(
            target.get("event")
        )
        entity_match = _entity_facts(prediction.get("entities")) == _entity_facts(
            target.get("entities")
        )
        states_match = _stable(prediction.get("states")) == _stable(
            target.get("states")
        )
        state_change_match = _stable(prediction.get("state_change")) == _stable(
            target.get("state_change")
        )
        full_match = event_match and entity_match and states_match and state_change_match
        counts["event"] += int(event_match)
        counts["entities"] += int(entity_match)
        counts["states"] += int(states_match)
        counts["state_change"] += int(state_change_match)
        counts["full"] += int(full_match)
        channel = str(reference["input"]["edge_context"]["channel"])
        slice_counts[f"edge:{channel}"][0] += 1
        slice_counts[f"edge:{channel}"][1] += int(full_match)
        delayed = bool(
            reference.get("evaluation_labels", {}).get("delayed_clue_acquisition")
        )
        slice_counts[f"delayed:{str(delayed).lower()}"][0] += 1
        slice_counts[f"delayed:{str(delayed).lower()}"][1] += int(full_match)
    total = counts["count"]
    videos = {
        str(row["video_id"])
        for row in references.values()
    }
    metrics = {
        "count": total,
        "valid_coverage": _ratio(counts["valid"], total),
        "event_accuracy": _ratio(counts["event"], total),
        "entity_fact_set_accuracy": _ratio(counts["entities"], total),
        "states_accuracy": _ratio(counts["states"], total),
        "state_change_accuracy": _ratio(counts["state_change"], total),
        "full_descriptor_accuracy": _ratio(counts["full"], total),
    }
    checks = {
        "held_out_records_present": total > 0,
        "minimum_video_count": len(videos) >= minimum_video_count,
        "prediction_coverage_complete": not missing,
        "no_unknown_prediction_ids": not unknown,
        "valid_generation_coverage_complete": counts["valid"] == total,
        "no_numeric_model_outputs": numeric_outputs == 0,
        "event_accuracy": metrics["event_accuracy"] >= minimum_event_accuracy,
        "full_descriptor_accuracy": metrics["full_descriptor_accuracy"]
        >= minimum_full_descriptor_accuracy,
    }
    return {
        "schema_version": "steam-iwm-observation-calibration/v0.1",
        "split": "validation",
        "reference_record_count": total,
        "prediction_record_count": len(predictions),
        "video_count": len(videos),
        "missing_record_ids": missing,
        "unknown_record_ids": unknown,
        "numeric_model_output_count": numeric_outputs,
        "metrics": metrics,
        "slice_metrics": {
            name: {
                "count": values[0],
                "full_descriptor_accuracy": _ratio(values[1], values[0]),
            }
            for name, values in sorted(slice_counts.items())
        },
        "thresholds": {
            "minimum_video_count": minimum_video_count,
            "minimum_event_accuracy": minimum_event_accuracy,
            "minimum_full_descriptor_accuracy": minimum_full_descriptor_accuracy,
        },
        "checks": checks,
        "passed": all(checks.values()),
        "blockers": [name for name, passed in checks.items() if not passed],
        "clue_acquisition_scored_as_model_target": False,
        "hypothesis_effect_training_authorized": False,
        "planner_training_authorized": False,
        "numeric_metrics_are_evaluator_only": True,
        "training_performed": False,
    }


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _entity_facts(value: Any) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not isinstance(value, list):
        return ()
    rows = []
    for entity in value:
        if not isinstance(entity, Mapping):
            continue
        rows.append(
            tuple(
                sorted(
                    (str(key), _normalize_text(child))
                    for key, child in entity.items()
                    if key in {"role", "entity_type", "surface", "visual_signature"}
                    and child not in (None, "", [], {})
                )
            )
        )
    return tuple(sorted(rows))


def _stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    records = _read_jsonl(args.records)
    for row in records:
        validate_sft_record(row)
    report = evaluate_observation_predictions(records, _read_jsonl(args.predictions))
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
