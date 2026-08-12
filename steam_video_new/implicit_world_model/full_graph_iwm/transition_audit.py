"""Build a blinded review packet for runtime transition calibration failures."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


AUDIT_SCHEMA = "steam-iwm-transition-audit/v0.1"
REVIEW_LABELS = {"over_crediting", "under_crediting", "consistent", "inconclusive"}


def build_transition_audit_packet(
    candidates: dict[str, Any],
    dataset: dict[str, Any],
    *,
    packet_id: str,
    evidence_by_id: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert evaluator-free executed comparisons into an auditable packet."""

    if candidates.get("schema_version") != "steam-executed-iwm-trace-candidates/v0.2":
        raise ValueError("unsupported executed candidate schema")
    if candidates.get("training_allowed") is not False:
        raise ValueError("source candidates must be prohibited from training")
    rows = candidates.get("transition_over_crediting_candidates")
    if not isinstance(rows, list):
        raise ValueError("source candidates have no transition audit rows")
    case_to_video = {
        str(row["case_id"]): str(row["video_id"])
        for row in dataset.get("cases") or ()
        if row.get("case_id") and row.get("video_id")
    }
    case_to_split = {
        str(row["case_id"]): str(row.get("split") or "unknown")
        for row in dataset.get("cases") or ()
        if row.get("case_id")
    }
    records: list[dict[str, Any]] = []
    for row in rows:
        case_id = str(row.get("case_id") or "")
        if case_id not in case_to_video:
            raise ValueError(f"transition audit case is absent from dataset: {case_id}")
        provisional = str(row.get("provisional_label") or "")
        if provisional not in REVIEW_LABELS:
            raise ValueError("transition audit row has an invalid provisional label")
        if row.get("hidden_evaluator_labels_included") is not False:
            raise ValueError("hidden evaluator label leaked into transition audit")
        observation_id = str(row["executed_real_observation_id"])
        observation = evidence_by_id.get(observation_id)
        if observation is None:
            raise ValueError(
                f"executed real observation is absent from L1: {observation_id}"
            )
        if str(observation.get("video_id") or "") != case_to_video[case_id]:
            raise ValueError("executed observation video does not match audit case")
        records.append(
            {
                "record_id": str(row["record_id"]),
                "case_id": case_id,
                "video_id": case_to_video[case_id],
                "split": case_to_split[case_id],
                "step_index": int(row["step_index"]),
                "trajectory_id": str(row["trajectory_id"]),
                "executed_real_observation_id": observation_id,
                "executed_real_observation": _compact_observation(observation),
                "predicted_belief_delta": row["predicted_belief_delta"],
                "corrected_belief_before": row["corrected_belief_before"],
                "corrected_belief_after": row["corrected_belief_after"],
                "automatic_failure_slice": provisional,
                "unsupported_predicted_resolved_roles": list(
                    row.get("unsupported_predicted_resolved_roles") or ()
                ),
                "review": {
                    "decision": None,
                    "rationale": None,
                    "annotator": None,
                },
                "review_status": "unreviewed",
                "training_allowed": False,
            }
        )
    records.sort(key=lambda row: row["record_id"])
    packet = {
        "schema_version": AUDIT_SCHEMA,
        "packet_id": packet_id,
        "annotation_status": "unreviewed",
        "source_model": str(candidates.get("model") or "unknown"),
        "source_candidate_schema": str(candidates["schema_version"]),
        "source_candidates_sha256": hashlib.sha256(
            json.dumps(
                candidates,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "allowed_review_decisions": sorted(REVIEW_LABELS),
        "contract": {
            "reference_is_post_read_corrected_hypothesis_belief": True,
            "hidden_clues_or_answers_included": False,
            "numeric_reward_score_probability_confidence_or_utility": False,
            "automatic_failure_slice_is_not_a_review_label": True,
            "training_requires_independent_locked_review": True,
        },
        "records": records,
        "training_allowed": False,
        "training_performed": False,
    }
    validate_transition_audit_packet(packet)
    report = transition_audit_readiness(packet)
    return packet, report


def validate_transition_audit_packet(packet: dict[str, Any]) -> None:
    if packet.get("schema_version") != AUDIT_SCHEMA:
        raise ValueError("unsupported transition audit packet schema")
    if packet.get("annotation_status") not in {"unreviewed", "human_locked"}:
        raise ValueError("invalid transition audit annotation status")
    if not str(packet.get("source_model") or "").strip():
        raise ValueError("transition audit source model is missing")
    if packet.get("source_candidate_schema") != (
        "steam-executed-iwm-trace-candidates/v0.2"
    ):
        raise ValueError("transition audit source candidate schema is invalid")
    digest = str(packet.get("source_candidates_sha256") or "")
    if len(digest) != 64 or any(row not in "0123456789abcdef" for row in digest):
        raise ValueError("transition audit source checksum is invalid")
    if packet.get("training_allowed") is not False:
        raise ValueError("transition audit packet cannot directly allow training")
    records = packet.get("records")
    if not isinstance(records, list):
        raise ValueError("transition audit records must be a list")
    ids: set[str] = set()
    for row in records:
        required = {
            "record_id",
            "case_id",
            "video_id",
            "split",
            "step_index",
            "trajectory_id",
            "executed_real_observation_id",
            "executed_real_observation",
            "predicted_belief_delta",
            "corrected_belief_before",
            "corrected_belief_after",
            "automatic_failure_slice",
            "unsupported_predicted_resolved_roles",
            "review",
            "review_status",
            "training_allowed",
        }
        if set(row) != required:
            raise ValueError("transition audit record schema is invalid")
        record_id = str(row["record_id"])
        if not record_id or record_id in ids:
            raise ValueError("transition audit record IDs must be non-empty and unique")
        ids.add(record_id)
        if row["automatic_failure_slice"] not in REVIEW_LABELS:
            raise ValueError("invalid automatic transition failure slice")
        observation = row["executed_real_observation"]
        if not isinstance(observation, dict):
            raise ValueError("executed real observation must be an object")
        if observation.get("node_id") != row["executed_real_observation_id"]:
            raise ValueError("executed observation identity mismatch")
        if observation.get("video_id") != row["video_id"]:
            raise ValueError("executed observation video mismatch")
        if observation.get("uses_hidden_supervision") is not False:
            raise ValueError("executed observation uses hidden supervision")
        review = row["review"]
        if not isinstance(review, dict) or set(review) != {
            "decision",
            "rationale",
            "annotator",
        }:
            raise ValueError("transition review fields are invalid")
        if row["review_status"] == "unreviewed":
            if any(review.values()):
                raise ValueError("unreviewed transition contains review labels")
        elif row["review_status"] == "human_locked":
            if review["decision"] not in REVIEW_LABELS:
                raise ValueError("locked transition review decision is invalid")
            if not str(review["rationale"] or "").strip() or not str(
                review["annotator"] or ""
            ).strip():
                raise ValueError("locked transition review is incomplete")
        else:
            raise ValueError("invalid transition record review status")
        if row["training_allowed"] is not False:
            raise ValueError("individual transition audit rows cannot allow training")


def transition_audit_readiness(packet: dict[str, Any]) -> dict[str, Any]:
    validate_transition_audit_packet(packet)
    rows = packet["records"]
    labels = Counter(str(row["automatic_failure_slice"]) for row in rows)
    reviewed = Counter(str(row["review_status"]) for row in rows)
    case_count = len({str(row["case_id"]) for row in rows})
    video_count = len({str(row["video_id"]) for row in rows})
    splits = Counter(str(row["split"]) for row in rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    train_case_count = len({str(row["case_id"]) for row in train_rows})
    train_video_count = len({str(row["video_id"]) for row in train_rows})
    return {
        "schema_version": "steam-iwm-transition-audit-readiness/v0.1",
        "packet_id": packet["packet_id"],
        "counts": {
            "record_count": len(rows),
            "case_count": case_count,
            "video_count": video_count,
            "split_counts": dict(sorted(splits.items())),
            "train_case_count": train_case_count,
            "train_video_count": train_video_count,
            "automatic_failure_slices": dict(sorted(labels.items())),
            "review_statuses": dict(sorted(reviewed.items())),
        },
        "readiness_dimensions": {
            "train_split_present": bool(train_rows),
            "at_least_thirty_train_cases": train_case_count >= 30,
            "at_least_twenty_train_videos": train_video_count >= 20,
            "all_four_failure_slices_present": REVIEW_LABELS <= set(labels),
            "at_least_twenty_over_crediting_candidates": (
                labels["over_crediting"] >= 20
            ),
            "all_records_independently_locked": bool(rows)
            and reviewed["human_locked"] == len(rows),
        },
        "training_ready": False,
        "training_blockers": [
            "automatic slices are diagnostics, not independent labels",
            "packet-level training_allowed remains false",
            "SFT export requires a separate human-locked adapter and split audit",
        ],
        "training_performed": False,
    }


def load_l1_evidence_catalog(
    dataset: dict[str, Any], graph_root: str | Path
) -> dict[str, dict[str, Any]]:
    root = Path(graph_root).expanduser().resolve()
    videos = {
        str(row["video_id"])
        for row in dataset.get("cases") or ()
        if row.get("video_id")
    }
    result: dict[str, dict[str, Any]] = {}
    for video_id in sorted(videos):
        path = root / video_id / "causal_temporal_overlay.json"
        if not path.is_file():
            continue
        overlay = _read(path)
        for node in overlay.get("l1_observations") or ():
            node_id = str(node.get("node_id") or "")
            if node_id:
                result[node_id] = node
    return result


def _compact_observation(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    participants = []
    for row in metadata.get("participants") or ():
        if not isinstance(row, dict):
            continue
        participants.append(
            {
                key: row[key]
                for key in (
                    "role",
                    "entity_type",
                    "surface",
                    "visual_signature",
                    "mention_id",
                    "track_status",
                )
                if row.get(key) not in (None, "")
            }
        )
    return {
        "node_id": str(node["node_id"]),
        "video_id": str(node["video_id"]),
        "time_span": node["time_span"],
        "text": str(node.get("text") or ""),
        "predicate": str(metadata.get("predicate") or ""),
        "action_kind": str(metadata.get("action_kind") or "unknown"),
        "participants": participants,
        "states": metadata.get("states") or [],
        "state_change": metadata.get("state_change"),
        "uses_hidden_supervision": bool(
            (node.get("provenance") or {}).get("uses_hidden_supervision")
        ),
    }


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
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--packet-id", required=True)
    parser.add_argument("--packet-output", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    args = parser.parse_args(argv)
    dataset = _read(args.dataset)
    packet, report = build_transition_audit_packet(
        _read(args.candidates),
        dataset,
        packet_id=args.packet_id,
        evidence_by_id=load_l1_evidence_catalog(dataset, args.graph_root),
    )
    _write(args.packet_output, packet)
    _write(args.report_output, report)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
