"""Audit atomic-event temporal and entity/state grounding deterministically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def audit_overlay(overlay: dict[str, Any]) -> dict[str, Any]:
    l1 = {
        str(node["node_id"]): node
        for node in overlay.get("l1_observations") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    events = [
        node
        for node in overlay.get("atomic_events") or []
        if isinstance(node, dict)
    ]
    grounded_spans = exact_span_reuse = participants = grounded_participants = 0
    states = linked_states = 0
    issues: list[dict[str, Any]] = []

    for event in events:
        event_id = str(event.get("node_id"))
        span = event.get("time_span") or {}
        refs = [str(value) for value in event.get("source_segments") or []]
        evidence = [l1[ref] for ref in refs if ref in l1]
        if evidence and any(_contains(item.get("time_span") or {}, span) for item in evidence):
            grounded_spans += 1
        else:
            issues.append({"event_id": event_id, "issue": "ungrounded_time_span"})
        if evidence and any(_same_span(item.get("time_span") or {}, span) for item in evidence):
            exact_span_reuse += 1

        metadata = event.get("metadata") or {}
        event_participants = metadata.get("participants") or []
        mention_ids = {
            str(value.get("mention_id"))
            for value in event_participants
            if isinstance(value, dict) and value.get("mention_id")
        }
        for participant in event_participants:
            if not isinstance(participant, dict):
                continue
            participants += 1
            grounding = {
                str(value) for value in participant.get("grounding_refs") or []
            }
            if grounding and grounding.issubset(set(refs)):
                grounded_participants += 1
            else:
                issues.append(
                    {
                        "event_id": event_id,
                        "mention_id": participant.get("mention_id"),
                        "issue": "participant_missing_or_invalid_grounding",
                    }
                )
        for state in metadata.get("states") or []:
            if not isinstance(state, dict):
                continue
            states += 1
            if str(state.get("mention_id")) in mention_ids:
                linked_states += 1
            else:
                issues.append(
                    {
                        "event_id": event_id,
                        "mention_id": state.get("mention_id"),
                        "issue": "state_unlinked_to_participant",
                    }
                )

    order_pairs = determinate_pairs = trusted_determinate_pairs = tied_pairs = 0
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            order_pairs += 1
            left_span = left.get("time_span") or {}
            right_span = right.get("time_span") or {}
            if float(left_span.get("end_s", 0)) <= float(right_span.get("start_s", 0)):
                determinate_pairs += 1
            elif float(right_span.get("end_s", 0)) <= float(left_span.get("start_s", 0)):
                determinate_pairs += 1
            elif _same_span(left_span, right_span):
                tied_pairs += 1
            left_l1 = [
                l1[str(ref)]
                for ref in left.get("source_segments") or []
                if str(ref) in l1
            ]
            right_l1 = [
                l1[str(ref)]
                for ref in right.get("source_segments") or []
                if str(ref) in l1
            ]
            if left_l1 and right_l1:
                left_end = max(
                    float((node.get("time_span") or {}).get("end_s", 0))
                    for node in left_l1
                )
                right_start = min(
                    float((node.get("time_span") or {}).get("start_s", 0))
                    for node in right_l1
                )
                right_end = max(
                    float((node.get("time_span") or {}).get("end_s", 0))
                    for node in right_l1
                )
                left_start = min(
                    float((node.get("time_span") or {}).get("start_s", 0))
                    for node in left_l1
                )
                if left_end <= right_start or right_end <= left_start:
                    trusted_determinate_pairs += 1

    return {
        "video_id": overlay.get("video_id"),
        "trust_status": (overlay.get("metadata") or {}).get("trust_status"),
        "event_count": len(events),
        "l1_count": len(l1),
        "events_per_l1": len(events) / len(l1) if l1 else None,
        "time_span_grounding_rate": grounded_spans / len(events) if events else None,
        "exact_l1_span_reuse_rate": exact_span_reuse / len(events) if events else None,
        "llm_refined_span_rate": (
            1.0 - exact_span_reuse / len(events) if events else None
        ),
        "temporal_order_determinate_pair_rate": (
            determinate_pairs / order_pairs if order_pairs else None
        ),
        "trusted_l1_temporal_order_pair_rate": (
            trusted_determinate_pairs / order_pairs if order_pairs else None
        ),
        "identical_span_pair_rate": tied_pairs / order_pairs if order_pairs else None,
        "participant_grounding_rate": (
            grounded_participants / participants if participants else None
        ),
        "state_link_rate": linked_states / states if states else None,
        "participant_count": participants,
        "state_count": states,
        "issues": issues,
    }


def _contains(outer: dict[str, Any], inner: dict[str, Any]) -> bool:
    try:
        return (
            float(outer["start_s"]) <= float(inner["start_s"]) + 1e-6
            and float(outer["end_s"]) >= float(inner["end_s"]) - 1e-6
        )
    except (KeyError, TypeError, ValueError):
        return False


def _same_span(left: dict[str, Any], right: dict[str, Any]) -> bool:
    try:
        return (
            abs(float(left["start_s"]) - float(right["start_s"])) <= 1e-6
            and abs(float(left["end_s"]) - float(right["end_s"])) <= 1e-6
        )
    except (KeyError, TypeError, ValueError):
        return False


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "events_per_l1",
        "time_span_grounding_rate",
        "exact_l1_span_reuse_rate",
        "llm_refined_span_rate",
        "temporal_order_determinate_pair_rate",
        "trusted_l1_temporal_order_pair_rate",
        "identical_span_pair_rate",
        "participant_grounding_rate",
        "state_link_rate",
    )
    means: dict[str, float | None] = {}
    for name in metric_names:
        values = [float(row[name]) for row in reports if row.get(name) is not None]
        means[name] = sum(values) / len(values) if values else None
    return {
        "video_count": len(reports),
        **means,
        "issue_count": sum(len(row["issues"]) for row in reports),
        "trusted_video_count": sum(row.get("trust_status") == "trusted" for row in reports),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    reports = [
        audit_overlay(_load(path))
        for path in sorted(args.run_dir.glob("*/causal_temporal_overlay.json"))
    ]
    result = {
        "schema_version": "steam-atomic-overlay-audit/v0.1",
        "summary": aggregate(reports),
        "videos": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
