"""Independent evaluation and annotation packets for video-only visual L1."""

from __future__ import annotations

from typing import Any

from .types import MemoryNode


def summarize_video_l1(nodes: list[MemoryNode]) -> dict[str, Any]:
    """Report structural coverage without pretending it measures correctness."""

    participants = [
        participant
        for node in nodes
        for participant in node.metadata.get("participants") or []
        if isinstance(participant, dict)
    ]
    states = [
        state
        for node in nodes
        for state in node.metadata.get("states") or []
        if isinstance(state, dict)
    ]
    track_counts: dict[str, int] = {}
    for participant in participants:
        track_id = str(participant.get("mention_id") or "")
        if track_id:
            track_counts[track_id] = track_counts.get(track_id, 0) + 1
    return {
        "node_count": len(nodes),
        "participant_count": len(participants),
        "state_assertion_count": len(states),
        "state_change_count": sum(
            isinstance(node.metadata.get("state_change"), dict) for node in nodes
        ),
        "entity_track_count": len(track_counts),
        "reused_entity_track_count": sum(count > 1 for count in track_counts.values()),
        "frame_grounded_node_rate": (
            sum(_has_frame_grounding(node) for node in nodes) / len(nodes)
            if nodes
            else None
        ),
        "mean_event_duration_s": (
            sum(node.time_span.end_s - node.time_span.start_s for node in nodes)
            / len(nodes)
            if nodes
            else None
        ),
        "correctness_status": "requires_independent_human_labels",
    }


def build_video_l1_annotation_packet(nodes: list[MemoryNode]) -> dict[str, Any]:
    """Create a human-label template for localization, tracks, and states."""

    return {
        "protocol_version": "video-only-l1-human-audit/v0.1",
        "instructions": {
            "event": "Judge direct visibility and correct the temporal interval.",
            "track": "Judge only whether two mentions depict the same physical entity.",
            "state": "Judge whether the stated attribute/value is directly visible.",
        },
        "events": [
            {
                "node_id": node.node_id,
                "predicate": node.text,
                "predicted_time_span": {
                    "start_s": node.time_span.start_s,
                    "end_s": node.time_span.end_s,
                },
                "visible": None,
                "gold_time_span": None,
            }
            for node in nodes
        ],
        "track_judgments": _track_pairs(nodes),
        "state_judgments": [
            {
                "node_id": node.node_id,
                "mention_id": state.get("mention_id"),
                "attribute": state.get("attribute"),
                "value": state.get("value"),
                "correct": None,
            }
            for node in nodes
            for state in node.metadata.get("states") or []
            if isinstance(state, dict)
        ],
    }


def evaluate_video_l1_human_audit(
    nodes: list[MemoryNode],
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate explicit human labels; incomplete labels remain visibly incomplete."""

    by_id = {node.node_id: node for node in nodes}
    event_rows = audit.get("events")
    track_rows = audit.get("track_judgments")
    state_rows = audit.get("state_judgments")
    if not all(isinstance(value, list) for value in (event_rows, track_rows, state_rows)):
        raise ValueError("video L1 audit requires event, track, and state lists")

    visible_labels: list[bool] = []
    temporal_ious: list[float] = []
    for row in event_rows:
        if not isinstance(row, dict) or row.get("node_id") not in by_id:
            continue
        if isinstance(row.get("visible"), bool):
            visible_labels.append(bool(row["visible"]))
        gold = row.get("gold_time_span")
        if row.get("visible") is True and isinstance(gold, dict):
            node = by_id[str(row["node_id"])]
            temporal_ious.append(
                interval_iou(
                    node.time_span.start_s,
                    node.time_span.end_s,
                    float(gold["start_s"]),
                    float(gold["end_s"]),
                )
            )

    track_labels = [
        bool(row["correct"])
        for row in track_rows
        if isinstance(row, dict) and isinstance(row.get("correct"), bool)
    ]
    state_labels = [
        bool(row["correct"])
        for row in state_rows
        if isinstance(row, dict) and isinstance(row.get("correct"), bool)
    ]
    expected_event_ids = set(by_id)
    labeled_event_ids = {
        str(row.get("node_id"))
        for row in event_rows
        if isinstance(row, dict) and isinstance(row.get("visible"), bool)
    }
    complete = (
        labeled_event_ids == expected_event_ids
        and all(isinstance(row.get("correct"), bool) for row in track_rows)
        and all(isinstance(row.get("correct"), bool) for row in state_rows)
    )
    return {
        "labels_source": "independent_human",
        "complete": complete,
        "visible_event_precision": _mean_bool(visible_labels),
        "mean_temporal_iou": (
            sum(temporal_ious) / len(temporal_ious) if temporal_ious else None
        ),
        "temporal_iou_at_0_5": (
            sum(value >= 0.5 for value in temporal_ious) / len(temporal_ious)
            if temporal_ious
            else None
        ),
        "entity_track_precision": _mean_bool(track_labels),
        "visible_state_precision": _mean_bool(state_labels),
        "counts": {
            "event_labels": len(visible_labels),
            "temporal_labels": len(temporal_ious),
            "track_labels": len(track_labels),
            "state_labels": len(state_labels),
        },
    }


def interval_iou(
    predicted_start: float,
    predicted_end: float,
    gold_start: float,
    gold_end: float,
) -> float:
    if predicted_end <= predicted_start or gold_end <= gold_start:
        return 0.0
    intersection = max(
        0.0,
        min(predicted_end, gold_end) - max(predicted_start, gold_start),
    )
    union = max(predicted_end, gold_end) - min(predicted_start, gold_start)
    return intersection / union if union > 0 else 0.0


def _track_pairs(nodes: list[MemoryNode]) -> list[dict[str, Any]]:
    mentions: dict[str, list[tuple[MemoryNode, dict[str, Any]]]] = {}
    for node in nodes:
        for participant in node.metadata.get("participants") or []:
            if not isinstance(participant, dict):
                continue
            track_id = str(participant.get("mention_id") or "")
            if track_id:
                mentions.setdefault(track_id, []).append((node, participant))
    rows: list[dict[str, Any]] = []
    for track_id, values in sorted(mentions.items()):
        ordered = sorted(values, key=lambda value: value[0].time_span.start_s)
        for (src_node, src), (dst_node, dst) in zip(ordered, ordered[1:]):
            rows.append(
                {
                    "track_id": track_id,
                    "src_node_id": src_node.node_id,
                    "dst_node_id": dst_node.node_id,
                    "src_surface": src.get("surface"),
                    "dst_surface": dst.get("surface"),
                    "correct": None,
                }
            )
    return rows


def _has_frame_grounding(node: MemoryNode) -> bool:
    localization = node.metadata.get("localization")
    return bool(
        isinstance(localization, dict)
        and localization.get("evidence_frames")
        and localization.get("frame_records")
    )


def _mean_bool(values: list[bool]) -> float | None:
    return sum(values) / len(values) if values else None
