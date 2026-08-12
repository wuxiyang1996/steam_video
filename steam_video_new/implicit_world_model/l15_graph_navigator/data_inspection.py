"""Inspect gathered executed transitions without training a model."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any

from .executed_transitions import validate_executed_transition_dataset
from .preference_data import validate_navigation_case_set


def inspect_transition_gathering(
    dataset: dict[str, Any],
    case_set: dict[str, Any],
) -> dict[str, Any]:
    dataset_errors = validate_executed_transition_dataset(dataset)
    case_errors = validate_navigation_case_set(case_set)
    records_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset.get("records") or []:
        records_by_case[str(record.get("case_id") or "")].append(record)

    action_counts: Counter[str] = Counter()
    verifier_cross: dict[str, Counter[str]] = defaultdict(Counter)
    resolved_roles: Counter[str] = Counter()
    action_origins: Counter[str] = Counter()
    action_legalities: Counter[str] = Counter()
    grounded_count = 0
    unreviewed_target_count = 0
    for record in dataset.get("records") or []:
        action_counts[str((record.get("action") or {}).get("action_type") or "missing")] += 1
        execution = record.get("execution") or {}
        provenance = record.get("action_provenance") or {}
        action_origins[str(provenance.get("origin") or "legacy_unspecified")] += 1
        action_legalities[str(provenance.get("legality") or "legacy_unspecified")] += 1
        grounded_count += int(execution.get("grounded") is True)
        verifier = execution.get("verifier_measurement") or {}
        verifier_cross[str(verifier.get("source") or "missing")][
            str(verifier.get("categorical_outcome") or "missing")
        ] += 1
        for role in ((record.get("target") or {}).get("belief_delta") or {}).get(
            "resolved_roles"
        ) or []:
            resolved_roles[str(role)] += 1
        unreviewed_target_count += int(record.get("review_decision") is None)

    case_coverage: list[dict[str, Any]] = []
    role_cases: dict[str, Counter[str]] = defaultdict(Counter)
    for case in case_set.get("cases") or []:
        case_id = str(case.get("case_id") or "")
        case_records = records_by_case.get(case_id, [])
        proposed = case.get("acceptable_first_actions") or []
        data_actions = [
            action for action in proposed if action.get("action_type") != "stop"
        ]
        executed_signatures = {
            _action_signature(record.get("action") or {}) for record in case_records
        }
        executed_types = {
            str((record.get("action") or {}).get("action_type") or "")
            for record in case_records
        }
        exact = bool(data_actions) and any(
            _action_signature(action) in executed_signatures for action in data_actions
        )
        type_covered = bool(data_actions) and any(
            str(action.get("action_type") or "") in executed_types
            for action in data_actions
        )
        exact_grounded = any(
            _action_signature(record.get("action") or {})
            in {_action_signature(action) for action in data_actions}
            and (record.get("execution") or {}).get("grounded") is True
            for record in case_records
        )
        role = str((case.get("missing_roles") or ["missing"])[0])
        if data_actions:
            role_cases[role]["data_case"] += 1
            role_cases[role]["exact_action"] += int(exact)
            role_cases[role]["action_type"] += int(type_covered)
            role_cases[role]["exact_grounded"] += int(exact_grounded)
        else:
            role_cases[role]["non_read_case"] += 1
        case_coverage.append(
            {
                "case_id": case_id,
                "role": role,
                "record_count": len(case_records),
                "proposed_data_action_count": len(data_actions),
                "exact_proposed_action_executed": exact,
                "proposed_action_type_executed": type_covered,
                "exact_proposed_action_grounded": exact_grounded,
            }
        )

    reviewed_action_types = {
        str(action.get("action_type") or "")
        for case in case_set.get("cases") or []
        for action in case.get("acceptable_first_actions") or []
        if action.get("action_type") != "stop"
    }
    unrepresented_optional_action_families = [
        action
        for action in (
            "inspect_state_change",
            "search_counterevidence",
            "find_bridge",
        )
        if action_counts[action] == 0
    ]
    missing_action_families = [
        action
        for action in unrepresented_optional_action_families
        if action in reviewed_action_types
    ]
    blockers = []
    if dataset_errors or case_errors:
        blockers.append("schema validation errors remain")
    if dataset.get("annotation_status") != "human_locked":
        blockers.append("executed transition targets are not independently human locked")
    if unreviewed_target_count:
        blockers.append("executed transition targets still require record-level review")
    if missing_action_families:
        blockers.append(
            "missing executed action families: " + ", ".join(missing_action_families)
        )
    if any(
        row["data_case"] and row["exact_grounded"] < row["data_case"]
        for row in role_cases.values()
    ):
        blockers.append("not every reviewed case has its proposed action grounded")
    if resolved_roles.get("state_transition", 0) == 0:
        blockers.append("no executed record resolves the state_transition role")
    if resolved_roles.get("counterevidence", 0) == 0:
        blockers.append("no executed record resolves the counterevidence role")

    return {
        "schema_version": "steam-transition-gathering-inspection/v0.1",
        "dataset_id": dataset.get("dataset_id"),
        "source_case_set_id": case_set.get("case_set_id"),
        "dataset_valid": not dataset_errors,
        "case_set_valid": not case_errors,
        "validation_errors": [*dataset_errors, *case_errors],
        "case_count": len(case_set.get("cases") or []),
        "record_count": len(dataset.get("records") or []),
        "grounded_record_count": grounded_count,
        "unreviewed_target_count": unreviewed_target_count,
        "action_counts": dict(sorted(action_counts.items())),
        "action_origin_counts": dict(sorted(action_origins.items())),
        "action_legality_counts": dict(sorted(action_legalities.items())),
        "missing_action_families": missing_action_families,
        "unrepresented_optional_action_families": unrepresented_optional_action_families,
        "resolved_role_counts": dict(sorted(resolved_roles.items())),
        "verifier_source_outcomes": {
            source: dict(sorted(counts.items()))
            for source, counts in sorted(verifier_cross.items())
        },
        "role_case_coverage": {
            role: dict(sorted(counts.items()))
            for role, counts in sorted(role_cases.items())
        },
        "case_coverage": case_coverage,
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
        "blockers": blockers,
    }


def _action_signature(action: dict[str, Any]) -> str:
    return json.dumps(
        {
            "action_type": action.get("action_type"),
            "source_id": action.get("source_id"),
            "target_ids": list(action.get("target_ids") or []),
            "relation": action.get("relation"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
