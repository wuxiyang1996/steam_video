"""Build, review, and inspect outcome-blinded evidence packets."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

from .balanced_cases import (
    lock_balanced_review_queue,
    validate_balanced_review_queue,
)


_SCHEMA = Path(__file__).resolve().parent / "balanced_evidence_packet.schema.json"
_OUTCOMES = {"supports", "rejects", "inconclusive", "not_applicable"}
_DECISIONS = {"accept", "reject"}
_DELAYED = {"yes", "no", "not_applicable"}
_FORBIDDEN_PUBLIC_MARKERS = {
    "hard_verifier",
    "offline_verifier_challenge",
    "identity_support",
    "identity_reject",
    "identity_inconclusive",
    "state_support",
    "state_reject",
    "state_inconclusive",
    "counterevidence_useful",
    "counterevidence_empty",
    "relation_probabilities",
    "direction_confidence",
}


def build_balanced_evidence_packet(
    queue: dict[str, Any],
    *,
    packet_id: str,
    context_limit: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a public evidence packet and a separate hidden provenance key."""

    errors = validate_balanced_review_queue(queue)
    if errors:
        raise ValueError("invalid balanced review queue: " + "; ".join(errors[:8]))
    if queue.get("annotation_status") != "unreviewed":
        raise ValueError("evidence packets must be built from an unreviewed queue")
    if context_limit < 0:
        raise ValueError("context_limit must be non-negative")

    items: list[dict[str, Any]] = []
    key_items: list[dict[str, Any]] = []
    embedding_key: dict[str, dict[str, Any]] = {}
    overlay_cache: dict[Path, dict[str, Any]] = {}
    for row in queue.get("reviews") or []:
        case = row["case"]
        overlay_path = Path(case["overlay_path"]).expanduser().resolve()
        overlay = overlay_cache.get(overlay_path)
        if overlay is None:
            overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
            overlay_cache[overlay_path] = overlay
        item, hidden = _build_item(
            row,
            overlay,
            overlay_path=overlay_path,
            context_limit=context_limit,
            embedding_key=embedding_key,
        )
        items.append(item)
        key_items.append(hidden)

    packet = {
        "schema_version": "steam-balanced-evidence-packet/v0.1",
        "packet_id": packet_id,
        "annotation_status": "unreviewed",
        "labels_source": None,
        "annotator": None,
        "locked_sha256": None,
        "source_review_queue_sha256": _checksum(queue),
        "output_contract": "categorical_only_no_numeric_judgments",
        "instructions": [
            "Judge only the visible evidence in this packet.",
            "supports requires direct visible support; failed support is inconclusive unless visible evidence contradicts the relation.",
            "Identity requires compatible type and attributes without simultaneous-distinct or impossible-motion evidence.",
            "State transition requires the same accepted identity, the same attribute, and explicit before/after values.",
            "Use not_applicable only when the requested verifier does not apply to the proposed operation.",
            "Return categorical judgments and a textual rationale; never return reward, score, confidence, probability, or utility numbers.",
        ],
        "items": items,
    }
    errors = validate_balanced_evidence_packet(packet)
    if errors:
        raise ValueError("invalid evidence packet: " + "; ".join(errors[:8]))
    key = {
        "schema_version": "steam-balanced-evidence-key/v0.1",
        "packet_sha256": _checksum(packet),
        "warning": "Keep hidden from annotators until the packet is locked.",
        "items": key_items,
        "embedding_artifacts": sorted(
            embedding_key.values(), key=lambda value: value["artifact_id"]
        ),
    }
    return packet, key


def validate_balanced_evidence_packet(payload: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for evidence packets") from exc
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: list(error.path),
        )
    ]
    item_ids = [str(item.get("item_id") or "") for item in payload.get("items") or []]
    if len(item_ids) != len(set(item_ids)):
        errors.append("item_id values must be unique")
    public_text = json.dumps(payload.get("items") or [], sort_keys=True).casefold()
    for marker in sorted(_FORBIDDEN_PUBLIC_MARKERS):
        if marker in public_text:
            errors.append(f"public packet leaks forbidden marker: {marker}")
    forbidden_key_paths = _forbidden_key_paths(
        payload.get("items") or [], path="items"
    )
    if forbidden_key_paths:
        errors.append("public packet exposes forbidden field: " + forbidden_key_paths[0])
    for index, item in enumerate(payload.get("items") or []):
        query = item.get("query") or {}
        evidence = item.get("evidence") or {}
        if not isinstance(query.get("proposed_action"), dict):
            errors.append(f"items.{index} requires a proposed action")
        if not evidence.get("proposed_endpoints"):
            errors.append(f"items.{index} requires visible proposed endpoints")
    status = payload.get("annotation_status")
    if status in {"ai_provisional", "human_locked"}:
        if not str(payload.get("annotator") or "").strip():
            errors.append("locked evidence packet requires annotator")
        expected_source = (
            "model_provisional" if status == "ai_provisional" else "independent_human"
        )
        if payload.get("labels_source") != expected_source:
            errors.append(f"{status} packet requires labels_source={expected_source}")
        for index, item in enumerate(payload.get("items") or []):
            annotation = item.get("annotation") or {}
            if not _annotation_complete(annotation):
                errors.append(f"items.{index}.annotation is incomplete")
            if annotation.get("review_decision") == "accept" and not (
                annotation.get("evidence_chain_valid")
                and annotation.get("first_action_valid")
                and annotation.get("evidence_refs")
            ):
                errors.append(f"items.{index} accepted without valid cited evidence")
            cited = {str(value) for value in annotation.get("evidence_refs") or []}
            unknown_citations = cited - _visible_evidence_refs(item)
            if unknown_citations:
                errors.append(
                    f"items.{index} cites evidence absent from packet: "
                    f"{sorted(unknown_citations)}"
                )
            if (
                annotation.get("review_decision") == "accept"
                and item.get("stratum") in {"identity", "state"}
                and annotation.get("verifier_outcome") == "not_applicable"
            ):
                errors.append(f"items.{index} accepted relation without verifier outcome")
            numeric_paths = _numeric_paths(annotation, path=f"items.{index}.annotation")
            if numeric_paths:
                errors.append("numeric annotation output forbidden: " + numeric_paths[0])
    checksum = payload.get("locked_sha256")
    if status in {"ai_provisional", "human_locked"} and not checksum:
        errors.append("locked evidence packet requires locked_sha256")
    if checksum and checksum != _checksum(payload):
        errors.append("evidence packet locked_sha256 does not match content")
    return errors


def lock_balanced_evidence_packet(
    payload: dict[str, Any],
    *,
    annotation_status: str,
    annotator: str,
) -> dict[str, Any]:
    if annotation_status not in {"ai_provisional", "human_locked"}:
        raise ValueError("annotation_status must be ai_provisional or human_locked")
    locked = json.loads(json.dumps(payload))
    locked["annotation_status"] = annotation_status
    locked["labels_source"] = (
        "model_provisional"
        if annotation_status == "ai_provisional"
        else "independent_human"
    )
    locked["annotator"] = annotator
    locked["locked_sha256"] = None
    errors = [
        error
        for error in validate_balanced_evidence_packet(locked)
        if "locked_sha256" not in error
    ]
    if errors:
        raise ValueError("cannot lock evidence packet: " + "; ".join(errors[:8]))
    locked["locked_sha256"] = _checksum(locked)
    errors = validate_balanced_evidence_packet(locked)
    if errors:
        raise ValueError("invalid locked evidence packet: " + "; ".join(errors[:8]))
    return locked


def apply_evidence_review(
    packet: dict[str, Any],
    review: dict[str, Any],
) -> dict[str, Any]:
    """Apply a complete external categorical review and lock its provenance."""

    packet_errors = validate_balanced_evidence_packet(packet)
    if packet_errors:
        raise ValueError("invalid evidence packet: " + "; ".join(packet_errors[:8]))
    if packet.get("annotation_status") != "unreviewed":
        raise ValueError("evidence review input must be unreviewed")
    if review.get("packet_id") != packet.get("packet_id"):
        raise ValueError("evidence review packet_id mismatch")
    if review.get("labels_source") != "model_provisional":
        raise ValueError("evidence review labels_source must be model_provisional")
    annotator = str(review.get("annotator") or "").strip()
    if not annotator:
        raise ValueError("evidence review requires annotator provenance")
    allowed_root = {
        "schema_version", "packet_id", "labels_source", "annotator",
        "protocol", "decisions",
    }
    unexpected_root = set(review) - allowed_root
    if unexpected_root:
        raise ValueError(f"unexpected evidence review fields: {sorted(unexpected_root)}")
    decisions: dict[str, dict[str, Any]] = {}
    allowed_decision = {
        "item_id", "review_decision", "verifier_outcome", "evidence_chain_valid",
        "first_action_valid", "delayed_effect", "evidence_refs", "rationale",
    }
    for row in review.get("decisions") or []:
        item_id = str(row.get("item_id") or "")
        if not item_id or item_id in decisions:
            raise ValueError(f"duplicate or empty reviewed item_id: {item_id!r}")
        unexpected = set(row) - allowed_decision
        if unexpected:
            raise ValueError(
                f"unexpected annotation fields for {item_id}: {sorted(unexpected)}"
            )
        numeric_paths = _numeric_paths(row, path=f"decisions.{item_id}")
        if numeric_paths:
            raise ValueError("numeric annotation output forbidden: " + numeric_paths[0])
        decisions[item_id] = row
    expected = {str(item["item_id"]) for item in packet["items"]}
    if set(decisions) != expected:
        raise ValueError(
            "evidence review coverage mismatch; "
            f"missing={sorted(expected - set(decisions))}, "
            f"unknown={sorted(set(decisions) - expected)}"
        )
    filled = json.loads(json.dumps(packet))
    for item in filled["items"]:
        decision = decisions[str(item["item_id"])]
        item["annotation"] = {
            key: decision.get(key)
            for key in (
                "review_decision",
                "verifier_outcome",
                "evidence_chain_valid",
                "first_action_valid",
                "delayed_effect",
                "evidence_refs",
                "rationale",
            )
        }
    return lock_balanced_evidence_packet(
        filled,
        annotation_status="ai_provisional",
        annotator=annotator,
    )


def import_evidence_annotations(
    queue: dict[str, Any],
    packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    queue_errors = validate_balanced_review_queue(queue)
    packet_errors = validate_balanced_evidence_packet(packet)
    if queue_errors:
        raise ValueError("invalid review queue: " + "; ".join(queue_errors[:8]))
    if packet_errors:
        raise ValueError("invalid evidence packet: " + "; ".join(packet_errors[:8]))
    if packet.get("annotation_status") not in {"ai_provisional", "human_locked"}:
        raise ValueError("evidence packet must be locked before import")
    if packet.get("source_review_queue_sha256") != _checksum(queue):
        raise ValueError("evidence packet does not match the source review queue")
    annotations = {
        str(item["item_id"]): item["annotation"] for item in packet["items"]
    }
    expected = {str(row["review_id"]) for row in queue.get("reviews") or []}
    if set(annotations) != expected:
        raise ValueError("evidence packet and review queue item coverage differ")

    filled = json.loads(json.dumps(queue))
    for row in filled["reviews"]:
        annotation = annotations[str(row["review_id"])]
        for field in (
            "review_decision",
            "verifier_outcome",
            "evidence_chain_valid",
            "first_action_valid",
            "delayed_effect",
            "rationale",
        ):
            row[field] = annotation[field]
    status = str(packet["annotation_status"])
    locked = lock_balanced_review_queue(
        filled,
        annotation_status=status,
        annotator=str(packet["annotator"]),
    )
    report = inspect_balanced_evidence_packet(packet)
    report.update(
        {
            "review_queue_status": locked["annotation_status"],
            "review_queue_locked_sha256": locked["locked_sha256"],
            "formal_gate_eligible": status == "human_locked",
            "training_performed": False,
        }
    )
    return locked, report


def inspect_balanced_evidence_packet(payload: dict[str, Any]) -> dict[str, Any]:
    errors = validate_balanced_evidence_packet(payload)
    strata: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    endpoint_count = context_count = embedding_count = retrieved_count = 0
    empty_endpoint_items: list[str] = []
    for item in payload.get("items") or []:
        strata[str(item.get("stratum") or "missing")] += 1
        evidence = item.get("evidence") or {}
        endpoints = evidence.get("proposed_endpoints") or []
        endpoint_count += len(endpoints)
        context_count += len(evidence.get("bounded_context") or [])
        embedding_count += sum(
            node.get("embedding_ref") is not None
            for node in [*endpoints, *(evidence.get("bounded_context") or [])]
        )
        retrieved_count += len(evidence.get("retrieved_relation_candidates") or [])
        if not endpoints:
            empty_endpoint_items.append(str(item.get("item_id") or ""))
        annotation = item.get("annotation") or {}
        if annotation.get("verifier_outcome"):
            outcomes[str(annotation["verifier_outcome"])] += 1
        if annotation.get("review_decision"):
            decisions[str(annotation["review_decision"])] += 1
    return {
        "schema_version": "steam-balanced-evidence-inspection/v0.1",
        "packet_id": payload.get("packet_id"),
        "annotation_status": payload.get("annotation_status"),
        "valid": not errors,
        "validation_errors": errors,
        "item_count": len(payload.get("items") or []),
        "stratum_counts": dict(sorted(strata.items())),
        "review_decision_counts": dict(sorted(decisions.items())),
        "verifier_outcome_counts": dict(sorted(outcomes.items())),
        "endpoint_node_count": endpoint_count,
        "bounded_context_node_count": context_count,
        "embedding_ref_count": embedding_count,
        "retrieved_relation_candidate_count": retrieved_count,
        "empty_endpoint_item_ids": empty_endpoint_items,
        "outcome_marker_leakage": False if not any(
            "leaks forbidden marker" in error for error in errors
        ) else True,
        "numeric_annotation_output_present": bool(
            _numeric_paths(
                [item.get("annotation") for item in payload.get("items") or []],
                path="items.annotation",
            )
        ) if payload.get("annotation_status") != "unreviewed" else False,
        "formal_gate_eligible": payload.get("annotation_status") == "human_locked",
        "training_performed": False,
    }


def _build_item(
    row: dict[str, Any],
    overlay: dict[str, Any],
    *,
    overlay_path: Path,
    context_limit: int,
    embedding_key: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    case = row["case"]
    action = case["acceptable_first_actions"][0]
    all_nodes = [
        node
        for node in [
            *(overlay.get("atomic_events") or []),
            *(overlay.get("l1_observations") or []),
        ]
        if isinstance(node, dict) and node.get("node_id")
    ]
    by_id = {str(node["node_id"]): node for node in all_nodes}
    proposed_ids = list(
        dict.fromkeys(
            str(value)
            for value in [
                *(case.get("seed_event_ids") or []),
                *(action.get("target_ids") or []),
                *(case.get("gold_event_ids") or []),
            ]
        )
    )
    endpoints = [
        _public_node(by_id[node_id], embedding_key, overlay_path)
        for node_id in proposed_ids
        if node_id in by_id
    ]
    context_nodes = _bounded_context_nodes(
        all_nodes, set(proposed_ids), context_limit=context_limit
    )
    public_context = [
        _public_node(node, embedding_key, overlay_path) for node in context_nodes
    ]
    included = {
        str(node["node_id"]) for node in [*endpoints, *public_context]
    }
    relations = [
        relation
        for relation in [
            *(overlay.get("relations") or []),
            *(overlay.get("l1_structural_relations") or []),
        ]
        if isinstance(relation, dict)
    ]
    deterministic = _deterministic_connections(relations, included)
    retrieved = _retrieved_relation_candidates(relations, action, by_id, embedding_key, overlay_path)
    item = {
        "item_id": row["review_id"],
        "stratum": row["category"],
        "query": {
            "question": case["question"],
            "proposed_action": action,
            "graph_read_budget": case["graph_read_budget"],
        },
        "evidence": {
            "proposed_endpoints": endpoints,
            "bounded_context": public_context,
            "observed_deterministic_connections": deterministic,
            "retrieved_relation_candidates": retrieved,
        },
        "annotation": {
            "review_decision": None,
            "verifier_outcome": None,
            "evidence_chain_valid": None,
            "first_action_valid": None,
            "delayed_effect": None,
            "evidence_refs": [],
            "rationale": "",
        },
    }
    return item, {
        "item_id": row["review_id"],
        "overlay_path": str(overlay_path),
        "overlay_sha256": hashlib.sha256(overlay_path.read_bytes()).hexdigest(),
        "source_case_id": case["case_id"],
    }


def _public_node(
    node: dict[str, Any],
    embedding_key: dict[str, dict[str, Any]],
    overlay_path: Path,
) -> dict[str, Any]:
    metadata = node.get("metadata") or {}
    embedding = node.get("embedding_ref")
    public_embedding = None
    if isinstance(embedding, dict):
        raw_path = str(embedding.get("path") or "")
        resolved = Path(raw_path)
        if raw_path and not resolved.is_absolute():
            resolved = (overlay_path.parent / resolved).resolve()
        artifact_id = "embedding:" + hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:16]
        embedding_key.setdefault(
            artifact_id,
            {
                "artifact_id": artifact_id,
                "path": str(resolved),
                "checksum": embedding.get("checksum"),
            },
        )
        public_embedding = {
            "artifact_id": artifact_id,
            "model": embedding.get("model"),
            "dimension": embedding.get("dimension"),
            "dtype": embedding.get("dtype"),
            "normalized": embedding.get("normalized"),
            "row_index": embedding.get("row_index"),
            "checksum": embedding.get("checksum"),
        }
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "time_span": node.get("time_span"),
        "text": node.get("text"),
        "source_node_id": node.get("source_node_id"),
        "source_segments": node.get("source_segments") or [],
        "participants": [
            {
                key: participant.get(key)
                for key in (
                    "mention_id", "role", "entity_type", "surface", "grounding_refs"
                )
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
        "evidence_refs": metadata.get("evidence_refs") or [],
        "embedding_ref": public_embedding,
    }


def _bounded_context_nodes(
    nodes: list[dict[str, Any]],
    proposed_ids: set[str],
    *,
    context_limit: int,
) -> list[dict[str, Any]]:
    if context_limit == 0:
        return []
    atomic = sorted(
        [node for node in nodes if node.get("node_type") == "atomic_event"],
        key=lambda node: (
            float((node.get("time_span") or {}).get("start_s") or 0.0),
            str(node.get("node_id") or ""),
        ),
    )
    indices = [
        index for index, node in enumerate(atomic)
        if str(node.get("node_id") or "") in proposed_ids
    ]
    ranked: list[tuple[int, float, str, dict[str, Any]]] = []
    for index, node in enumerate(atomic):
        node_id = str(node.get("node_id") or "")
        if node_id in proposed_ids:
            continue
        distance = min((abs(index - focus) for focus in indices), default=len(atomic))
        start = float((node.get("time_span") or {}).get("start_s") or 0.0)
        ranked.append((distance, start, node_id, node))
    chosen = [row[3] for row in sorted(ranked)[:context_limit]]
    return sorted(
        chosen,
        key=lambda node: (
            float((node.get("time_span") or {}).get("start_s") or 0.0),
            str(node.get("node_id") or ""),
        ),
    )


def _deterministic_connections(
    relations: list[dict[str, Any]],
    included: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for relation in relations:
        if relation.get("status") != "deterministic":
            continue
        if relation.get("src") not in included or relation.get("dst") not in included:
            continue
        names = sorted(str(name) for name in (relation.get("relation_probabilities") or {}))
        for name in names:
            rows.append(
                {
                    "src": relation.get("src"),
                    "dst": relation.get("dst"),
                    "relation": name,
                    "evidence_refs": relation.get("evidence_refs") or [],
                }
            )
    return sorted(rows, key=lambda row: (str(row["src"]), str(row["dst"]), row["relation"]))


def _retrieved_relation_candidates(
    relations: list[dict[str, Any]],
    action: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    embedding_key: dict[str, dict[str, Any]],
    overlay_path: Path,
) -> list[dict[str, Any]]:
    requested = str(action.get("relation") or "")
    if not requested:
        return []
    rows = []
    for relation in relations:
        if requested not in (relation.get("relation_probabilities") or {}):
            continue
        src, dst = str(relation.get("src") or ""), str(relation.get("dst") or "")
        if src not in by_id or dst not in by_id:
            continue
        if action.get("action_type") != "search_counterevidence" and (
            src != action.get("source_id")
            or dst not in set(action.get("target_ids") or [])
        ):
            continue
        rows.append(
            {
                "relation": requested,
                "source": _public_node(by_id[src], embedding_key, overlay_path),
                "destination": _public_node(by_id[dst], embedding_key, overlay_path),
                "evidence_refs": relation.get("evidence_refs") or [],
                "status": "candidate_only_requires_review",
            }
        )
    return rows


def _annotation_complete(annotation: dict[str, Any]) -> bool:
    return (
        annotation.get("review_decision") in _DECISIONS
        and annotation.get("verifier_outcome") in _OUTCOMES
        and isinstance(annotation.get("evidence_chain_valid"), bool)
        and isinstance(annotation.get("first_action_valid"), bool)
        and annotation.get("delayed_effect") in _DELAYED
        and bool(annotation.get("evidence_refs"))
        and bool(str(annotation.get("rationale") or "").strip())
    )


def _visible_evidence_refs(item: dict[str, Any]) -> set[str]:
    evidence = item.get("evidence") or {}
    values: set[str] = set()
    for node in [
        *(evidence.get("proposed_endpoints") or []),
        *(evidence.get("bounded_context") or []),
    ]:
        values.add(str(node.get("node_id") or ""))
        values.update(str(ref) for ref in node.get("evidence_refs") or [])
    for relation in [
        *(evidence.get("observed_deterministic_connections") or []),
        *(evidence.get("retrieved_relation_candidates") or []),
    ]:
        values.update(str(ref) for ref in relation.get("evidence_refs") or [])
        for endpoint in (relation.get("source"), relation.get("destination")):
            if isinstance(endpoint, dict):
                values.add(str(endpoint.get("node_id") or ""))
                values.update(str(ref) for ref in endpoint.get("evidence_refs") or [])
    values.discard("")
    return values


def _numeric_paths(value: Any, *, path: str) -> list[str]:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return []
    if isinstance(value, (int, float)):
        return [path]
    if isinstance(value, dict):
        return [
            child_path
            for key, child in value.items()
            for child_path in _numeric_paths(child, path=f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            child_path
            for index, child in enumerate(value)
            for child_path in _numeric_paths(child, path=f"{path}[{index}]")
        ]
    return [path]


def _forbidden_key_paths(value: Any, *, path: str) -> list[str]:
    forbidden = {
        "confidence", "direction_confidence", "hard_verifier", "path",
        "probability", "relation_probabilities", "reward", "score", "utility",
        "vector", "embedding_vector",
    }
    if isinstance(value, dict):
        return [
            child_path
            for key, child in value.items()
            for child_path in (
                [f"{path}.{key}"] if str(key).casefold() in forbidden else []
            ) + _forbidden_key_paths(child, path=f"{path}.{key}")
        ]
    if isinstance(value, list):
        return [
            child_path
            for index, child in enumerate(value)
            for child_path in _forbidden_key_paths(child, path=f"{path}[{index}]")
        ]
    return []


def _checksum(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload))
    if "locked_sha256" in clone:
        clone["locked_sha256"] = None
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
