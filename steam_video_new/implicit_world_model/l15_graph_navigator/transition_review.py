"""Build and review outcome-blinded executed-transition packets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .executed_transitions import validate_executed_transition_dataset
from .preference_data import validate_navigation_case_set


_SCHEMA = Path(__file__).resolve().parent / "transition_review_packet.schema.json"
_VALIDITY = {"valid", "invalid", "inconclusive"}
_OUTCOMES = {"supports", "rejects", "inconclusive", "not_applicable"}
_DECISIONS = {"accept", "reject"}
_FORBIDDEN_PUBLIC_KEYS = {
    "action_provenance",
    "legality",
    "native_candidate",
    "reviewed_accepted",
    "review_status",
    "edge_admitted",
    "belief_update_audit",
    "verifier_measurement",
    "target",
    "target_source",
    "resolved_roles",
    "relation_updates",
    "sampling_stratum",
    "source_record_id",
    "gtsam",
}


def build_transition_review_packet(
    dataset: dict[str, Any],
    case_set: dict[str, Any],
    *,
    packet_id: str,
    native_controls_per_action: int = 2,
    consistency_duplicates: int = 6,
    selected_record_strata: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    errors = validate_executed_transition_dataset(dataset)
    if errors:
        raise ValueError("invalid executed transition dataset: " + "; ".join(errors[:8]))
    if dataset.get("annotation_status") != "unreviewed":
        raise ValueError("transition review must start from an unreviewed dataset")
    case_errors = validate_navigation_case_set(case_set)
    if case_errors:
        raise ValueError("invalid source case set: " + "; ".join(case_errors[:8]))
    if case_set.get("case_set_id") != (dataset.get("source_case_set") or {}).get(
        "case_set_id"
    ):
        raise ValueError("transition dataset and case set do not match")
    if native_controls_per_action < 0 or consistency_duplicates < 0:
        raise ValueError("sampling limits must be non-negative")

    records = list(dataset.get("records") or [])
    checkpoints = {
        str(row["checkpoint_id"]): row["checkpoint"]
        for row in dataset.get("checkpoints") or []
    }
    cases = {str(case["case_id"]): case for case in case_set.get("cases") or []}
    overlay_cache: dict[Path, dict[str, Any]] = {}
    reviewed = [
        record
        for record in records
        if (record.get("action_provenance") or {}).get("reviewed_accepted") is True
    ]
    if selected_record_strata is not None:
        known_ids = {str(record["record_id"]) for record in records}
        missing = sorted(set(selected_record_strata) - known_ids)
        if missing:
            raise ValueError(
                "selected transition record is absent from dataset: " + missing[0]
            )
        reviewed = [
            record
            for record in records
            if str(record["record_id"]) in selected_record_strata
        ]
    native_by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        provenance = record.get("action_provenance") or {}
        if provenance.get("origin") != "native_candidate":
            continue
        action_type = str((record.get("action") or {}).get("action_type") or "missing")
        native_by_action[action_type].append(record)
    controls = [
        record
        for action_type in sorted(native_by_action)
        for record in _stable_sample(
            native_by_action[action_type],
            native_controls_per_action,
            salt=f"{packet_id}:native:{action_type}",
        )
    ]
    selected: list[tuple[dict[str, Any], str, str | None]] = []
    seen: set[str] = set()
    for record, stratum in [
        *((
            record,
            (
                "targeted:" + selected_record_strata[str(record["record_id"])]
                if selected_record_strata is not None
                else "reviewed_action"
            ),
        ) for record in reviewed),
        *((record, "native_control") for record in controls),
    ]:
        record_id = str(record["record_id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        selected.append((record, stratum, None))
    duplicate_sources = _stable_sample(
        reviewed,
        min(consistency_duplicates, len(reviewed)),
        salt=f"{packet_id}:duplicates",
    )
    for source in duplicate_sources:
        selected.append(
            (
                source,
                "consistency_duplicate",
                "duplicate:" + _digest(str(source["record_id"]))[:16],
            )
        )

    item_rows: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    occurrence: Counter[str] = Counter()
    for record, stratum, duplicate_group in selected:
        record_id = str(record["record_id"])
        occurrence[record_id] += 1
        item_id = "transition-review:" + _digest(
            f"{packet_id}\0{record_id}\0{occurrence[record_id]}"
        )[:24]
        case = cases[str(record["case_id"])]
        overlay_path = Path(str(case["overlay_path"])).expanduser().resolve()
        overlay = overlay_cache.get(overlay_path)
        if overlay is None:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            overlay_cache[overlay_path] = overlay
        item = _public_item(
            record,
            item_id=item_id,
            checkpoint=checkpoints[str(record["checkpoint_ref"])],
            overlay=overlay,
        )
        hidden = {
            "item_id": item_id,
            "source_record_id": record_id,
            "source_case_id": record["case_id"],
            "sampling_stratum": stratum,
            "duplicate_group": duplicate_group,
            "action_provenance": record.get("action_provenance"),
            "stored_verifier_measurement": (
                record.get("execution") or {}
            ).get("verifier_measurement"),
            "stored_target": record.get("target"),
            "overlay_path": str(overlay_path),
            "overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        }
        order = _digest(f"{packet_id}:order:{item_id}")
        item_rows.append((order, item, hidden))
    item_rows.sort(key=lambda row: row[0])

    packet = {
        "schema_version": "steam-transition-review-packet/v0.1",
        "packet_id": packet_id,
        "annotation_status": "unreviewed",
        "labels_source": None,
        "annotator": None,
        "locked_sha256": None,
        "source_dataset_sha256": _checksum(dataset),
        "source_case_set_sha256": _checksum(case_set),
        "output_contract": "categorical_only_no_numeric_judgments",
        "instructions": [
            "Judge only visible executed evidence; hidden provenance and stored targets are unavailable.",
            "Observation validity asks whether the listed real observation and citations are grounded.",
            "Action execution validity asks whether this operation was actually completed as described.",
            "Belief delta validity asks whether the visible result warrants a categorical belief update; inconclusive is valid when evidence cannot decide.",
            "A completed empty counterevidence search is a valid operation result but does not establish a contradiction.",
            "State support requires one grounded identity with an explicit same-attribute before/after value change.",
            "Return only categorical judgments, evidence citations, and text rationale; never return numeric reward, score, confidence, probability, or utility.",
        ],
        "items": [row[1] for row in item_rows],
    }
    validation = validate_transition_review_packet(packet)
    if validation:
        raise ValueError("invalid transition review packet: " + "; ".join(validation[:8]))
    hidden_key = {
        "schema_version": "steam-transition-review-hidden-key/v0.1",
        "packet_sha256": _checksum(packet),
        "warning": "Do not expose until provisional review is locked.",
        "source_dataset_id": dataset["dataset_id"],
        "source_dataset_sha256": _checksum(dataset),
        "items": [row[2] for row in item_rows],
    }
    return packet, hidden_key


def validate_transition_review_packet(payload: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for transition review") from exc
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: list(error.path),
        )
    ]
    ids = [str(item.get("item_id") or "") for item in payload.get("items") or []]
    if len(ids) != len(set(ids)):
        errors.append("transition review item_id values must be unique")
    leaks = _forbidden_paths(payload.get("items") or [], path="items")
    if leaks:
        errors.append("public transition review packet leaks hidden field: " + leaks[0])
    status = payload.get("annotation_status")
    if status in {"ai_provisional", "human_locked"}:
        expected_source = (
            "model_provisional" if status == "ai_provisional" else "independent_human"
        )
        if payload.get("labels_source") != expected_source:
            errors.append(f"{status} packet requires labels_source={expected_source}")
        if not str(payload.get("annotator") or "").strip():
            errors.append("locked transition review packet requires annotator")
        for index, item in enumerate(payload.get("items") or []):
            annotation = item.get("annotation") or {}
            if not _annotation_complete(annotation):
                errors.append(f"items.{index}.annotation is incomplete")
            citations = {str(value) for value in annotation.get("evidence_refs") or []}
            if citations - _visible_refs(item):
                errors.append(f"items.{index} cites evidence absent from public packet")
            numeric = _numeric_paths(annotation, path=f"items.{index}.annotation")
            if numeric:
                errors.append("numeric transition annotation forbidden: " + numeric[0])
    checksum = payload.get("locked_sha256")
    if status in {"ai_provisional", "human_locked"} and not checksum:
        errors.append("locked transition review packet requires locked_sha256")
    if checksum and checksum != _checksum(payload):
        errors.append("transition review locked_sha256 does not match content")
    return errors


def apply_transition_review(
    packet: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    errors = validate_transition_review_packet(packet)
    if errors or packet.get("annotation_status") != "unreviewed":
        raise ValueError("transition review requires a valid unreviewed packet")
    if review.get("packet_id") != packet.get("packet_id"):
        raise ValueError("transition review packet_id mismatch")
    labels_source = review.get("labels_source")
    if labels_source not in {"model_provisional", "independent_human"}:
        raise ValueError(
            "transition review labels_source must be model_provisional or independent_human"
        )
    annotator = str(review.get("annotator") or "").strip()
    if not annotator:
        raise ValueError("transition review requires annotator provenance")
    allowed_root = {
        "schema_version", "packet_id", "labels_source", "annotator", "protocol", "decisions"
    }
    if set(review) - allowed_root:
        raise ValueError("transition review contains unexpected root fields")
    allowed = {
        "item_id", "review_decision", "observation_validity",
        "action_execution_validity", "belief_delta_validity", "relation_outcome",
        "evidence_refs", "rationale",
    }
    decisions: dict[str, dict[str, Any]] = {}
    for row in review.get("decisions") or []:
        item_id = str(row.get("item_id") or "")
        if not item_id or item_id in decisions or set(row) - allowed:
            raise ValueError(f"invalid transition review decision: {item_id!r}")
        numeric = _numeric_paths(row, path=f"decisions.{item_id}")
        if numeric:
            raise ValueError("numeric transition annotation forbidden: " + numeric[0])
        decisions[item_id] = row
    expected = {str(item["item_id"]) for item in packet["items"]}
    if set(decisions) != expected:
        raise ValueError("transition review coverage mismatch")
    locked = json.loads(json.dumps(packet))
    locked["annotation_status"] = (
        "human_locked" if labels_source == "independent_human" else "ai_provisional"
    )
    locked["labels_source"] = labels_source
    locked["annotator"] = annotator
    for item in locked["items"]:
        item["annotation"] = {
            key: decisions[item["item_id"]].get(key)
            for key in (
                "review_decision", "observation_validity",
                "action_execution_validity", "belief_delta_validity",
                "relation_outcome", "evidence_refs", "rationale",
            )
        }
    locked["locked_sha256"] = None
    prelock = [
        error for error in validate_transition_review_packet(locked)
        if "locked_sha256" not in error
    ]
    if prelock:
        raise ValueError("cannot lock transition review: " + "; ".join(prelock[:8]))
    locked["locked_sha256"] = _checksum(locked)
    final = validate_transition_review_packet(locked)
    if final:
        raise ValueError("invalid locked transition review: " + "; ".join(final[:8]))
    return locked


def inspect_transition_review(
    packet: dict[str, Any], hidden_key: dict[str, Any] | None = None
) -> dict[str, Any]:
    errors = validate_transition_review_packet(packet)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for item in packet.get("items") or []:
        annotation = item.get("annotation") or {}
        for field in (
            "review_decision", "observation_validity", "action_execution_validity",
            "belief_delta_validity", "relation_outcome",
        ):
            if annotation.get(field):
                counts[field][str(annotation[field])] += 1
    duplicate_agreement: dict[str, Any] = {
        "evaluated_group_count": 0,
        "exact_agreement_group_count": 0,
        "all_groups_exact": None,
    }
    if hidden_key is not None and packet.get("annotation_status") != "unreviewed":
        if hidden_key.get("packet_sha256") != _checksum(_unlocked_copy(packet)):
            # The key binds the public unreviewed packet; annotations are added later.
            public_ids = {str(item["item_id"]) for item in packet["items"]}
            key_ids = {str(item["item_id"]) for item in hidden_key.get("items") or []}
            if public_ids != key_ids:
                errors.append("hidden key does not match transition packet items")
        annotations = {
            str(item["item_id"]): item.get("annotation") or {}
            for item in packet["items"]
        }
        groups: dict[str, list[str]] = defaultdict(list)
        sources: dict[str, list[str]] = defaultdict(list)
        for row in hidden_key.get("items") or []:
            item_id = str(row.get("item_id") or "")
            source = str(row.get("source_record_id") or "")
            sources[source].append(item_id)
            if row.get("duplicate_group"):
                groups[str(row["duplicate_group"])].append(item_id)
        exact = 0
        evaluated = 0
        for source, item_ids in sources.items():
            if len(item_ids) < 2:
                continue
            evaluated += 1
            projected = {
                json.dumps(annotations[item_id], sort_keys=True, ensure_ascii=False)
                for item_id in item_ids
            }
            exact += int(len(projected) == 1)
        duplicate_agreement = {
            "evaluated_group_count": evaluated,
            "exact_agreement_group_count": exact,
            "all_groups_exact": exact == evaluated if evaluated else None,
        }
    blockers = []
    if errors:
        blockers.append("schema or hidden-key validation errors remain")
    if packet.get("annotation_status") != "human_locked":
        blockers.append("review remains model-provisional rather than independent-human locked")
    blockers.append("training is intentionally disabled for this review stage")
    return {
        "schema_version": "steam-transition-review-inspection/v0.1",
        "packet_id": packet.get("packet_id"),
        "annotation_status": packet.get("annotation_status"),
        "valid": not errors,
        "validation_errors": errors,
        "item_count": len(packet.get("items") or []),
        "categorical_counts": {
            field: dict(sorted(counter.items())) for field, counter in sorted(counts.items())
        },
        "duplicate_consistency": duplicate_agreement,
        "numeric_annotation_output_present": bool(
            _numeric_paths(
                [item.get("annotation") for item in packet.get("items") or []],
                path="items.annotation",
            )
        ) if packet.get("annotation_status") != "unreviewed" else False,
        "formal_eligible": packet.get("annotation_status") == "human_locked",
        "training_ready": False,
        "training_performed": False,
        "blockers": blockers,
    }


def _public_item(
    record: dict[str, Any],
    *,
    item_id: str,
    checkpoint: dict[str, Any],
    overlay: dict[str, Any],
) -> dict[str, Any]:
    execution = record.get("execution") or {}
    requested_ids = {
        str(value)
        for value in (
            [record.get("action", {}).get("source_id")]
            + list((record.get("action") or {}).get("target_ids") or [])
            + list(execution.get("real_observation_ids") or [])
        )
        if value
    }
    overlay_nodes = {
        str(node.get("node_id")): node
        for node in [
            *(overlay.get("atomic_events") or []),
            *(overlay.get("l1_observations") or []),
        ]
        if isinstance(node, dict) and node.get("node_id")
    }
    visible_nodes = [
        _public_overlay_node(overlay_nodes[node_id])
        for node_id in sorted(requested_ids)
        if node_id in overlay_nodes
    ]
    return {
        "item_id": item_id,
        "question": record["question"],
        "belief_before": {
            "missing_roles": checkpoint.get("missing_roles") or [],
            "contradictions": checkpoint.get("contradictions") or [],
            "answerability": checkpoint.get("answerability") or "not_ready",
        },
        "action": record["action"],
        "visible_context": {
            "nodes": visible_nodes,
        },
        "executed_result": {
            "status": execution.get("status"),
            "real_observation_ids": execution.get("real_observation_ids") or [],
            "evidence_refs": execution.get("evidence_refs") or [],
            "observation_outcome": execution.get("observation_outcome"),
            "grounded_empty_result": execution.get("grounded_empty_result") is True,
        },
        "annotation": {
            "review_decision": None,
            "observation_validity": None,
            "action_execution_validity": None,
            "belief_delta_validity": None,
            "relation_outcome": None,
            "evidence_refs": [],
            "rationale": "",
        },
    }


def _public_overlay_node(node: dict[str, Any]) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    embedding = node.get("embedding_ref")
    public_embedding = None
    if isinstance(embedding, dict):
        public_embedding = {
            key: embedding.get(key)
            for key in (
                "model", "dimension", "dtype", "normalized", "row_index", "checksum"
            )
        }
    return {
        "node_id": node.get("node_id"),
        "node_kind": node.get("node_type"),
        "description": node.get("text"),
        "evidence_refs": list(
            dict.fromkeys(
                str(value)
                for value in [
                    *(node.get("source_segments") or []),
                    *(metadata.get("evidence_refs") or []),
                ]
            )
        ),
        "participants": [
            {
                key: participant.get(key)
                for key in ("mention_id", "role", "entity_type", "surface", "grounding_refs")
            }
            for participant in metadata.get("participants") or []
            if isinstance(participant, dict)
        ],
        "states": [
            {
                key: state.get(key)
                for key in (
                    "mention_id", "attribute", "value", "polarity",
                    "source_l1_node_id", "subject_source_l1_node_id", "subject_binding",
                )
            }
            for state in metadata.get("states") or []
            if isinstance(state, dict)
        ],
        "embedding_ref": public_embedding,
    }
def _stable_sample(records: list[dict[str, Any]], limit: int, *, salt: str) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda record: _digest(f"{salt}\0{record['record_id']}")
    )[:limit]


def _annotation_complete(annotation: dict[str, Any]) -> bool:
    return (
        annotation.get("review_decision") in _DECISIONS
        and annotation.get("observation_validity") in _VALIDITY
        and annotation.get("action_execution_validity") in _VALIDITY
        and annotation.get("belief_delta_validity") in _VALIDITY
        and annotation.get("relation_outcome") in _OUTCOMES
        and bool(annotation.get("evidence_refs"))
        and bool(str(annotation.get("rationale") or "").strip())
    )


def _visible_refs(item: dict[str, Any]) -> set[str]:
    refs = set(str(value) for value in (item.get("executed_result") or {}).get("evidence_refs") or [])
    refs.update(str(value) for value in (item.get("executed_result") or {}).get("real_observation_ids") or [])
    for node in (item.get("visible_context") or {}).get("nodes") or []:
        refs.add(str(node.get("node_id") or ""))
        refs.update(str(value) for value in node.get("evidence_refs") or [])
    refs.discard("")
    return refs


def _forbidden_paths(value: Any, *, path: str) -> list[str]:
    if isinstance(value, dict):
        return [
            child
            for key, nested in value.items()
            for child in (
                [f"{path}.{key}"] if str(key).casefold() in _FORBIDDEN_PUBLIC_KEYS else []
            ) + _forbidden_paths(nested, path=f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            child for index, nested in enumerate(value)
            for child in _forbidden_paths(nested, path=f"{path}[{index}]")
        ]
    return []


def _numeric_paths(value: Any, *, path: str) -> list[str]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [path]
    if isinstance(value, dict):
        return [
            nested_path for key, nested in value.items()
            for nested_path in _numeric_paths(nested, path=f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            nested_path for index, nested in enumerate(value)
            for nested_path in _numeric_paths(nested, path=f"{path}[{index}]")
        ]
    return [path]


def _unlocked_copy(packet: dict[str, Any]) -> dict[str, Any]:
    clone = json.loads(json.dumps(packet))
    clone["annotation_status"] = "unreviewed"
    clone["labels_source"] = None
    clone["annotator"] = None
    clone["locked_sha256"] = None
    for item in clone.get("items") or []:
        item["annotation"] = {
            "review_decision": None,
            "observation_validity": None,
            "action_execution_validity": None,
            "belief_delta_validity": None,
            "relation_outcome": None,
            "evidence_refs": [],
            "rationale": "",
        }
    return clone


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checksum(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload))
    if "locked_sha256" in clone:
        clone["locked_sha256"] = None
    return _digest(json.dumps(clone, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
