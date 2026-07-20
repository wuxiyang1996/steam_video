"""Blinded ordinal annotation and train-record export for real sibling reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


ALLOWED_PREFERENCE_LABELS = (
    "prefer_left",
    "tie",
    "prefer_right",
    "incomparable",
)
_PACKAGE_DIR = Path(__file__).resolve().parent


def validate_navigation_case_set(payload: dict[str, Any]) -> list[str]:
    errors = _schema_errors(payload, "navigation_case_set.schema.json")
    case_ids = [str(case.get("case_id") or "") for case in payload.get("cases", [])]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        errors.append(f"duplicate case_id values: {duplicates}")
    checksum = payload.get("locked_sha256")
    if payload.get("annotation_status") == "human_locked" and not checksum:
        errors.append("human_locked case sets require locked_sha256")
    if checksum and checksum != _content_checksum(payload):
        errors.append("navigation case set locked_sha256 does not match content")
    return errors


def require_valid_navigation_case_set(payload: dict[str, Any]) -> None:
    errors = validate_navigation_case_set(payload)
    if errors:
        raise ValueError("invalid navigation case set: " + "; ".join(errors[:8]))


def lock_navigation_case_set(
    payload: dict[str, Any],
    *,
    annotation_status: str,
    annotator: str,
) -> dict[str, Any]:
    if annotation_status not in {"ai_provisional", "human_locked"}:
        raise ValueError("annotation_status must be ai_provisional or human_locked")
    locked = json.loads(json.dumps(payload))
    locked["annotation_status"] = annotation_status
    locked["annotator"] = annotator
    locked["locked_sha256"] = None
    errors = validate_navigation_case_set(locked)
    errors = [error for error in errors if "locked_sha256" not in error]
    if errors:
        raise ValueError("cannot lock navigation case set: " + "; ".join(errors[:8]))
    locked["locked_sha256"] = _content_checksum(locked)
    require_valid_navigation_case_set(locked)
    return locked


def build_preference_annotation_packet(
    sibling_artifacts: Iterable[tuple[str, dict[str, Any]]],
    *,
    packet_id: str,
    max_comparisons_per_case: int | None = None,
) -> dict[str, Any]:
    """Remove rule labels and expose only real categorical branch outcomes."""

    comparisons: list[dict[str, Any]] = []
    for case_id, artifact in sibling_artifacts:
        branches = {
            str(branch["branch_id"]): branch for branch in artifact.get("branches", [])
        }
        question = str((artifact.get("checkpoint") or {}).get("question") or "")
        checkpoint = artifact.get("checkpoint") or {}
        checkpoint_summary = {
            "acquired_evidence": list(checkpoint.get("acquired_evidence") or []),
            "frontier": list(checkpoint.get("frontier") or []),
            "missing_roles": list(checkpoint.get("missing_roles") or []),
            "contradictions": list(checkpoint.get("contradictions") or []),
            "uncertainty": checkpoint.get("uncertainty"),
            "answerability": checkpoint.get("answerability"),
            "remaining_graph_reads": checkpoint.get("remaining_graph_reads"),
        }
        pair_rows = list(artifact.get("pairwise_preferences") or [])
        if max_comparisons_per_case is not None:
            if max_comparisons_per_case <= 0:
                raise ValueError("max_comparisons_per_case must be positive")
            pair_rows = _select_informative_pairs(
                pair_rows,
                branches,
                limit=max_comparisons_per_case,
            )
        for index, row in enumerate(pair_rows):
            left_id = str(row.get("left_branch_id") or "")
            right_id = str(row.get("right_branch_id") or "")
            if left_id not in branches or right_id not in branches:
                raise ValueError(f"unknown sibling branch in {case_id}: {left_id}, {right_id}")
            comparisons.append(
                {
                    "comparison_id": f"{case_id}:comparison:{index:05d}",
                    "case_id": case_id,
                    "video_id": str(artifact.get("video_id") or artifact.get("example_id") or "unknown"),
                    "question": question,
                    "checkpoint_summary": checkpoint_summary,
                    "left": _blinded_branch(branches[left_id]),
                    "right": _blinded_branch(branches[right_id]),
                    "label": None,
                    "rationale": "",
                }
            )
    packet = {
        "schema_version": "steam-trajectory-preference-annotation/v0.1",
        "packet_id": packet_id,
        "annotation_status": "unlabeled",
        "annotator": None,
        "locked_sha256": None,
        "allowed_labels": list(ALLOWED_PREFERENCE_LABELS),
        "instructions": [
            "Judge the complete observed branch outcome, not only the first action name.",
            "Use prefer_left, tie, prefer_right, or incomparable; never invent a numeric reward.",
            "Prefer evidence-grounded role resolution and answerability without unresolved contradiction.",
            "Use incomparable when branches make different useful progress without a justified ordering.",
            "This packet hides rule-based provisional labels and relation posterior probabilities.",
        ],
        "comparisons": comparisons,
    }
    errors = validate_preference_annotation_packet(packet, require_labels=False)
    if errors:
        raise ValueError("invalid generated preference packet: " + "; ".join(errors[:8]))
    return packet


def validate_preference_annotation_packet(
    payload: dict[str, Any],
    *,
    require_labels: bool | None = None,
) -> list[str]:
    errors = _schema_errors(payload, "preference_annotation.schema.json")
    should_require = (
        payload.get("annotation_status") in {"ai_provisional", "human_locked"}
        if require_labels is None
        else require_labels
    )
    ids = [str(row.get("comparison_id") or "") for row in payload.get("comparisons", [])]
    if len(ids) != len(set(ids)):
        errors.append("comparison_id values must be unique")
    if should_require:
        for index, row in enumerate(payload.get("comparisons", [])):
            if row.get("label") not in ALLOWED_PREFERENCE_LABELS:
                errors.append(f"comparisons.{index}.label is not complete")
            if not str(row.get("rationale") or "").strip():
                errors.append(f"comparisons.{index}.rationale is empty")
    checksum = payload.get("locked_sha256")
    if payload.get("annotation_status") == "human_locked" and not checksum:
        errors.append("human_locked packets require locked_sha256")
    if checksum and checksum != _content_checksum(payload):
        errors.append("preference packet locked_sha256 does not match content")
    return errors


def lock_annotation_packet(
    payload: dict[str, Any],
    *,
    annotation_status: str,
    annotator: str,
) -> dict[str, Any]:
    if annotation_status not in {"ai_provisional", "human_locked"}:
        raise ValueError("annotation_status must be ai_provisional or human_locked")
    locked = json.loads(json.dumps(payload))
    locked["annotation_status"] = annotation_status
    locked["annotator"] = annotator
    locked["locked_sha256"] = None
    errors = validate_preference_annotation_packet(locked, require_labels=True)
    errors = [error for error in errors if "locked_sha256" not in error]
    if errors:
        raise ValueError("cannot lock preference packet: " + "; ".join(errors[:8]))
    locked["locked_sha256"] = _content_checksum(locked)
    errors = validate_preference_annotation_packet(locked, require_labels=True)
    if errors:
        raise ValueError("invalid locked preference packet: " + "; ".join(errors[:8]))
    return locked


def export_training_records(
    packet: dict[str, Any],
    *,
    allow_ai_provisional: bool = False,
) -> list[dict[str, Any]]:
    errors = validate_preference_annotation_packet(packet, require_labels=True)
    if errors:
        raise ValueError("invalid annotated packet: " + "; ".join(errors[:8]))
    status = str(packet.get("annotation_status") or "")
    if status != "human_locked" and not (
        status == "ai_provisional" and allow_ai_provisional
    ):
        raise ValueError(
            "training export requires human_locked annotations; "
            "pass allow_ai_provisional only for explicitly provisional experiments"
        )
    records: list[dict[str, Any]] = []
    seen_transitions: set[str] = set()
    for comparison in packet["comparisons"]:
        for side in ("left", "right"):
            branch = comparison[side]
            fingerprint = hashlib.sha256(
                json.dumps(branch, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()
            if fingerprint in seen_transitions:
                continue
            seen_transitions.add(fingerprint)
            records.append(
                {
                    "task": "observation_belief_transition",
                    "case_id": comparison["case_id"],
                    "video_id": comparison["video_id"],
                    "question": comparison["question"],
                    "checkpoint": comparison["checkpoint_summary"],
                    "action": branch["action"],
                    "target": {
                        "observation_descriptor": branch["observation_descriptor"],
                        "belief_delta": branch["belief_delta"],
                    },
                    "label_source": status,
                }
            )
        records.append(
            {
                "task": "trajectory_pairwise_preference",
                "comparison_id": comparison["comparison_id"],
                "case_id": comparison["case_id"],
                "video_id": comparison["video_id"],
                "question": comparison["question"],
                "checkpoint": comparison["checkpoint_summary"],
                "left": comparison["left"],
                "right": comparison["right"],
                "target": {"label": comparison["label"]},
                "rationale": comparison["rationale"],
                "label_source": status,
            }
        )
    _reject_numeric_reward_fields(records)
    return records


def _blinded_branch(branch: dict[str, Any]) -> dict[str, Any]:
    transition = branch.get("realized_transition") or {}
    return {
        "action": branch.get("action") or {},
        "real_observation_ids": list(branch.get("real_observation_ids") or []),
        "observation_descriptor": transition.get("observation_descriptor") or {},
        "belief_delta": transition.get("belief_delta") or {},
    }


def _select_informative_pairs(
    rows: list[dict[str, Any]],
    branches: dict[str, dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    """Select categorical disagreements without reading provisional labels."""

    def rank(row: dict[str, Any]) -> tuple[object, ...]:
        left_id = str(row.get("left_branch_id") or "")
        right_id = str(row.get("right_branch_id") or "")
        left = _branch_signature(branches[left_id])
        right = _branch_signature(branches[right_id])
        return (
            -(left[1] != right[1]),
            -(left[2] != right[2]),
            -(left[3] != right[3]),
            -(left[0] != right[0]),
            -(left[4] != right[4]),
            left_id,
            right_id,
        )

    ranked = sorted(rows, key=rank)
    selected: list[dict[str, Any]] = []
    branch_counts: dict[str, int] = {}
    action_pairs: set[tuple[str, str]] = set()
    for diversify in (True, False):
        for row in ranked:
            if row in selected:
                continue
            left_id = str(row.get("left_branch_id") or "")
            right_id = str(row.get("right_branch_id") or "")
            left_action = _branch_signature(branches[left_id])[0]
            right_action = _branch_signature(branches[right_id])[0]
            action_pair = tuple(sorted((left_action, right_action)))
            if diversify and action_pair in action_pairs:
                continue
            if branch_counts.get(left_id, 0) >= 3 or branch_counts.get(right_id, 0) >= 3:
                continue
            selected.append(row)
            action_pairs.add(action_pair)
            branch_counts[left_id] = branch_counts.get(left_id, 0) + 1
            branch_counts[right_id] = branch_counts.get(right_id, 0) + 1
            if len(selected) >= limit:
                return selected
    return selected


def _branch_signature(branch: dict[str, Any]) -> tuple[str, tuple[str, ...], str, tuple[str, ...], str]:
    transition = branch.get("realized_transition") or {}
    delta = transition.get("belief_delta") or {}
    action = branch.get("action") or {}
    return (
        str(action.get("action_type") or ""),
        tuple(sorted(str(value) for value in delta.get("resolved_roles") or [])),
        str(delta.get("answerability_after") or ""),
        tuple(sorted(str(value) for value in delta.get("contradiction_updates") or [])),
        str(delta.get("uncertainty_change") or ""),
    )


def _schema_errors(payload: dict[str, Any], schema_name: str) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for workflow artifacts") from exc
    schema = json.loads((_PACKAGE_DIR / schema_name).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)
    return [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    ]


def _content_checksum(payload: dict[str, Any]) -> str:
    normalized = dict(payload)
    normalized["locked_sha256"] = None
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_numeric_reward_fields(payload: object) -> None:
    forbidden = {"reward", "utility", "q_value", "score", "delayed_utility"}
    if isinstance(payload, dict):
        overlap = forbidden & {str(key).casefold() for key in payload}
        if overlap:
            raise ValueError(f"numeric reward fields are forbidden: {sorted(overlap)}")
        for value in payload.values():
            _reject_numeric_reward_fields(value)
    elif isinstance(payload, list):
        for value in payload:
            _reject_numeric_reward_fields(value)
