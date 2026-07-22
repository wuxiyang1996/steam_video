"""Failure slicing and targeted sampling for blind transition review."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from typing import Any

from .executed_transitions import validate_executed_transition_dataset
from .preference_data import validate_navigation_case_set
from .transition_review import (
    build_transition_review_packet,
    validate_transition_review_packet,
)


FAILURE_SLICES = (
    "identity_unconfirmed",
    "missing_same_attribute_before_after",
    "counterevidence_related_not_refuting",
    "relation_definition_unclear",
    "evidence_retrieval_insufficient",
)

TARGET_STRATA = (
    "strict_state_delta_candidate",
    "identity_hard_negative_candidate",
    "counterevidence_support_or_refute",
    "empty_or_reject_control",
    "inconclusive_control",
    "delayed_two_hop",
)

DEFAULT_TARGET_QUOTAS = {
    "strict_state_delta_candidate": 6,
    "identity_hard_negative_candidate": 8,
    "counterevidence_support_or_refute": 10,
    "empty_or_reject_control": 8,
    "inconclusive_control": 8,
    "delayed_two_hop": 10,
}


def inspect_inconclusive_failure_slices(packet: dict[str, Any]) -> dict[str, Any]:
    """Partition provisional inconclusive rows using visible categorical facts only."""

    errors = validate_transition_review_packet(packet)
    if errors:
        raise ValueError("invalid transition review packet: " + "; ".join(errors[:8]))
    if packet.get("annotation_status") not in {"ai_provisional", "human_locked"}:
        raise ValueError("failure slicing requires a locked reviewed packet")
    rows = []
    for item in packet.get("items") or []:
        if (item.get("annotation") or {}).get("relation_outcome") != "inconclusive":
            continue
        primary, signals = _failure_slice(item)
        rows.append(
            {
                "item_id": item["item_id"],
                "primary_slice": primary,
                "action_type": (item.get("action") or {}).get("action_type"),
                "relation": (item.get("action") or {}).get("relation"),
                "visible_signals": signals,
                "evidence_refs": (item.get("annotation") or {}).get("evidence_refs") or [],
                "rationale": (item.get("annotation") or {}).get("rationale") or "",
            }
        )
    counts = Counter(row["primary_slice"] for row in rows)
    return {
        "schema_version": "steam-transition-failure-slices/v0.1",
        "packet_id": packet["packet_id"],
        "source_annotation_status": packet["annotation_status"],
        "partition_contract": "exclusive_primary_slice_from_visible_categorical_evidence",
        "inconclusive_item_count": len(rows),
        "slice_counts": {name: counts[name] for name in FAILURE_SLICES},
        "partition_complete": len(rows) == sum(counts.values()),
        "items": rows,
        "next_collection_requirements": [
            "same grounded identity plus explicit same-attribute changed values",
            "different-person and stable-attribute-conflict identity hard negatives",
            "directly supporting and directly refuting counterevidence outcomes",
            "executed empty, reject, and inconclusive controls",
            "delayed cases whose first read only opens a later decisive read",
        ],
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
    }


def build_targeted_transition_gathering(
    dataset: dict[str, Any],
    case_set: dict[str, Any],
    *,
    packet_id: str,
    quotas: dict[str, int] | None = None,
    consistency_duplicates: int = 6,
    excluded_video_ids: set[str] | frozenset[str] = frozenset(),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Select a diverse, executed candidate set and produce an outcome-blind packet."""

    dataset_errors = validate_executed_transition_dataset(dataset)
    if dataset_errors:
        raise ValueError("invalid executed transition dataset: " + "; ".join(dataset_errors[:8]))
    case_errors = validate_navigation_case_set(case_set)
    if case_errors:
        raise ValueError("invalid navigation case set: " + "; ".join(case_errors[:8]))
    requested = dict(DEFAULT_TARGET_QUOTAS if quotas is None else quotas)
    if set(requested) - set(TARGET_STRATA):
        raise ValueError("targeted gathering contains an unknown stratum")
    if any(int(value) < 0 for value in requested.values()):
        raise ValueError("targeted gathering quotas must be non-negative")

    cases = {str(case["case_id"]): case for case in case_set.get("cases") or []}
    excluded_video_ids = {str(value) for value in excluded_video_ids}
    excluded_record_ids: set[str] = set()
    available: dict[str, list[dict[str, Any]]] = {name: [] for name in TARGET_STRATA}
    for record in dataset.get("records") or []:
        if str(record.get("video_id") or "") in excluded_video_ids:
            excluded_record_ids.add(str(record.get("record_id") or ""))
            continue
        action = record.get("action") or {}
        execution = record.get("execution") or {}
        verifier = execution.get("verifier_measurement") or {}
        relation = action.get("relation")
        outcome = verifier.get("categorical_outcome")
        case = cases.get(str(record.get("case_id"))) or {}
        tags = set(str(value) for value in case.get("tags") or [])
        if relation == "state_transition" and outcome == "supports":
            available["strict_state_delta_candidate"].append(record)
        if relation == "same_entity" and outcome == "inconclusive":
            available["identity_hard_negative_candidate"].append(record)
        if relation == "contradicts" and execution.get("status") == "executed":
            available["counterevidence_support_or_refute"].append(record)
        if execution.get("status") != "executed" or execution.get("observation_outcome") == "empty":
            available["empty_or_reject_control"].append(record)
        if outcome == "inconclusive" and execution.get("status") == "executed":
            available["inconclusive_control"].append(record)
        if "delayed_two_hop" in tags and (
            record.get("action_provenance") or {}
        ).get("reviewed_accepted") is True:
            available["delayed_two_hop"].append(record)

    selected: dict[str, str] = {}
    selected_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for stratum in TARGET_STRATA:
        candidates = [
            record for record in available[stratum]
            if str(record["record_id"]) not in selected
        ]
        for record in _diverse_sample(candidates, int(requested.get(stratum, 0))):
            record_id = str(record["record_id"])
            selected[record_id] = stratum
            counts[stratum] += 1
            selected_rows.append(
                {
                    "record_id": record_id,
                    "case_id": record["case_id"],
                    "video_id": record["video_id"],
                    "target_stratum": stratum,
                    "candidate_basis": _candidate_basis(stratum),
                }
            )
    if not selected:
        raise ValueError("targeted gathering found no executable candidates")

    packet, hidden_key = build_transition_review_packet(
        dataset,
        case_set,
        packet_id=packet_id,
        native_controls_per_action=0,
        consistency_duplicates=consistency_duplicates,
        selected_record_strata=selected,
    )
    video_counts = Counter(row["video_id"] for row in selected_rows)
    deficits = {
        name: max(0, int(requested.get(name, 0)) - counts[name])
        for name in TARGET_STRATA
    }
    report = {
        "schema_version": "steam-targeted-transition-gathering/v0.1",
        "packet_id": packet_id,
        "source_dataset_id": dataset["dataset_id"],
        "requested_quotas": requested,
        "available_candidates": {name: len(available[name]) for name in TARGET_STRATA},
        "selected_counts": {name: counts[name] for name in TARGET_STRATA},
        "quota_deficits": deficits,
        "selected_record_count": len(selected_rows),
        "selected_video_count": len(video_counts),
        "selected_video_counts": dict(sorted(video_counts.items())),
        "excluded_video_ids": sorted(excluded_video_ids),
        "excluded_record_count": len(excluded_record_ids),
        "public_packet_item_count": len(packet["items"]),
        "consistency_duplicate_count": consistency_duplicates,
        "counterevidence_outcome_coverage": "pending_independent_human_review",
        "selection_is_gold": False,
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
        "blockers": [
            "target strata are acquisition rules rather than gold labels",
            "counterevidence support versus refute remains a human review target",
            *(
                ["requested targeted quotas have source-data deficits"]
                if any(deficits.values()) else []
            ),
        ],
    }
    manifest = {
        "schema_version": "steam-targeted-transition-manifest/v0.1",
        "packet_id": packet_id,
        "warning": "Keep hidden from reviewers; target strata are not gold labels.",
        "records": selected_rows,
    }
    return packet, hidden_key, {"report": report, "manifest": manifest}


def _failure_slice(item: dict[str, Any]) -> tuple[str, list[str]]:
    action = item.get("action") or {}
    relation = action.get("relation")
    action_type = action.get("action_type")
    evidence_count = len((item.get("executed_result") or {}).get("evidence_refs") or [])
    if action_type == "search_counterevidence" or relation == "contradicts":
        return "counterevidence_related_not_refuting", [
            "counterevidence operation executed",
            "visible result does not directly negate source claim",
        ]
    if relation == "transition_support":
        return "relation_definition_unclear", [
            "transition_support has no strict same-attribute contract"
        ]
    if relation == "state_transition":
        common_attributes = _common_state_attributes(item)
        if not common_attributes:
            return "missing_same_attribute_before_after", [
                "no common explicit state attribute across endpoints"
            ]
        return "identity_unconfirmed", [
            "explicit common state attribute is visible",
            "shared grounded identity is not established",
        ]
    if relation == "same_entity" and evidence_count < 2:
        return "evidence_retrieval_insufficient", [
            "fewer than two executed evidence citations",
            "identity relation remains unresolved",
        ]
    return "identity_unconfirmed", [
        "identity relation remains unresolved after executed read"
    ]


def _common_state_attributes(item: dict[str, Any]) -> set[str]:
    action = item.get("action") or {}
    endpoint_ids = [action.get("source_id"), *(action.get("target_ids") or [])]
    nodes = {
        str(node.get("node_id")): node
        for node in (item.get("visible_context") or {}).get("nodes") or []
    }
    attribute_sets = []
    for endpoint in endpoint_ids[:2]:
        node = nodes.get(str(endpoint)) or {}
        attribute_sets.append(
            {str(state.get("attribute")) for state in node.get("states") or [] if state.get("attribute")}
        )
    return set.intersection(*attribute_sets) if len(attribute_sets) == 2 else set()


def _diverse_sample(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_video[str(record.get("video_id") or "missing")].append(record)
    for video_id in by_video:
        by_video[video_id].sort(key=lambda row: _stable_key(row, "within-video"))
    selected = []
    video_ids = sorted(by_video, key=lambda value: hashlib.sha256(value.encode()).hexdigest())
    while len(selected) < limit:
        advanced = False
        for video_id in video_ids:
            if by_video[video_id] and len(selected) < limit:
                selected.append(by_video[video_id].pop(0))
                advanced = True
        if not advanced:
            break
    return selected


def _stable_key(record: dict[str, Any], salt: str) -> str:
    return hashlib.sha256(
        f"{salt}|{record.get('record_id')}".encode("utf-8")
    ).hexdigest()


def _candidate_basis(stratum: str) -> list[str]:
    return {
        "strict_state_delta_candidate": [
            "executed relation read",
            "categorical verifier support candidate",
            "human must confirm identity and same-attribute delta",
        ],
        "identity_hard_negative_candidate": [
            "executed same-entity read",
            "categorical verifier remained inconclusive",
            "human must distinguish reject from insufficient evidence",
        ],
        "counterevidence_support_or_refute": [
            "executed counterevidence search",
            "human must label direct support, direct refutation, or inconclusive",
        ],
        "empty_or_reject_control": [
            "executed result is empty or execution was insufficient",
        ],
        "inconclusive_control": [
            "executed relation verifier remained inconclusive",
        ],
        "delayed_two_hop": [
            "case was collected for delayed two-hop reasoning",
            "human must verify that the first read opens later decisive evidence",
        ],
    }[stratum]
