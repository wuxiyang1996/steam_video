"""Mine correction-sensitive cases without filling rare quotas with easy cases."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from memory_graph.navigation import NavigationActionType

from .case_miner import _mine_overlay, _video_splits
from .overlay_io import load_overlay_artifact
from .preference_data import lock_navigation_case_set, validate_navigation_case_set


BALANCED_CASE_CATEGORIES = (
    "delayed_two_hop",
    "identity_support",
    "identity_reject",
    "identity_inconclusive",
    "state_support",
    "state_reject",
    "state_inconclusive",
    "contradiction_open",
    "contradiction_resolve",
    "counterevidence_useful",
    "counterevidence_empty",
    "blocked_path_recovery",
    "ambiguity_abstention",
)
DEFAULT_BALANCED_QUOTAS = {
    "delayed_two_hop": 15,
    "identity_support": 10,
    "identity_reject": 10,
    "identity_inconclusive": 10,
    "state_support": 10,
    "state_reject": 10,
    "state_inconclusive": 10,
    "contradiction_open": 10,
    "contradiction_resolve": 10,
    "counterevidence_useful": 10,
    "counterevidence_empty": 10,
    "blocked_path_recovery": 10,
    "ambiguity_abstention": 10,
}
_IDENTITY = {"same_entity", "same_object", "same_instance_candidate", "reappears_candidate"}
_STATE = {"state_transition", "transition_support"}
_SCHEMA = Path(__file__).resolve().parent / "balanced_case_review.schema.json"


def mine_balanced_reasoning_cases(
    overlay_paths: Iterable[Path],
    *,
    case_set_id: str,
    quotas: dict[str, int] | None = None,
    per_video_category_limit: int = 2,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return draft cases, deficit report, and an outcome-blinded review queue."""

    requested = dict(DEFAULT_BALANCED_QUOTAS if quotas is None else quotas)
    unknown = set(requested) - set(BALANCED_CASE_CATEGORIES)
    if unknown:
        raise ValueError(f"unknown balanced categories: {sorted(unknown)}")
    if any(int(value) < 0 for value in requested.values()):
        raise ValueError("balanced quotas must be non-negative")
    if per_video_category_limit <= 0:
        raise ValueError("per_video_category_limit must be positive")
    selected_overlays, duplicate_count = _select_unique_overlays(overlay_paths)
    video_ids = sorted({overlay.video_id for _, overlay in selected_overlays})
    split_by_video = _video_splits(video_ids)
    available: dict[str, list[dict[str, Any]]] = {
        category: [] for category in BALANCED_CASE_CATEGORIES
    }
    for path, overlay in selected_overlays:
        mined = _mine_balanced_overlay(
            path, overlay, split_by_video[overlay.video_id]
        )
        for category, rows in mined.items():
            available[category].extend(rows)

    chosen: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for category in BALANCED_CASE_CATEGORIES:
        per_video: Counter[str] = Counter()
        for candidate in sorted(available[category], key=_row_order):
            if selected_counts[category] >= int(requested.get(category, 0)):
                break
            video_id = str(candidate["_video_id"])
            if per_video[video_id] >= per_video_category_limit:
                continue
            row = dict(candidate)
            row.pop("_video_id", None)
            chosen.append(row)
            selected_counts[category] += 1
            per_video[video_id] += 1
    deficits = {
        category: max(0, int(requested.get(category, 0)) - selected_counts[category])
        for category in BALANCED_CASE_CATEGORIES
    }
    case_set = {
        "schema_version": "steam-navigation-gold-cases/v0.1",
        "case_set_id": case_set_id,
        "annotation_status": "draft",
        "annotator": None,
        "locked_sha256": None,
        "cases": chosen,
    }
    case_errors = validate_navigation_case_set(case_set)
    if case_errors:
        raise ValueError("invalid balanced draft cases: " + "; ".join(case_errors[:8]))
    report = {
        "schema_version": "steam-balanced-case-mining-report/v0.1",
        "case_set_id": case_set_id,
        "available_video_count": len(video_ids),
        "available_overlay_count": len(selected_overlays),
        "duplicate_overlay_count": duplicate_count,
        "requested_quotas": requested,
        "available_candidates": {
            category: len(available[category]) for category in BALANCED_CASE_CATEGORIES
        },
        "selected_categories": dict(selected_counts),
        "quota_deficits": deficits,
        "cross_category_backfill": False,
        "formal_ready": False,
        "formal_blockers": [
            "all candidates require independent evidence/action review",
            "verifier outcome is a review target, not a mined gold label",
            *(
                [f"quota deficits remain: {deficits}"]
                if any(deficits.values())
                else []
            ),
        ],
    }
    queue = build_balanced_review_queue(case_set, report)
    return case_set, report, queue


def build_balanced_review_queue(
    case_set: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for case in case_set["cases"]:
        mined_category = next(
            tag for tag in case["tags"] if tag in BALANCED_CASE_CATEGORIES
        )
        stratum = _review_stratum(mined_category)
        rows.append(
            {
                "review_id": "review:" + hashlib.sha256(
                    f"{case_set['case_set_id']}|{case['case_id']}".encode("utf-8")
                ).hexdigest()[:20],
                "category": stratum,
                "case": _blind_review_case(case, stratum),
                "candidate_basis": _candidate_basis(stratum),
                "review_decision": None,
                "verifier_outcome": None,
                "evidence_chain_valid": None,
                "first_action_valid": None,
                "delayed_effect": None,
                "rationale": "",
            }
        )
    queue = {
        "schema_version": "steam-balanced-case-review/v0.1",
        "queue_id": f"{case_set['case_set_id']}:review",
        "annotation_status": "unreviewed",
        "annotator": None,
        "locked_sha256": None,
        "source_case_set_id": case_set["case_set_id"],
        "allowed_verifier_outcomes": [
            "supports", "rejects", "inconclusive", "not_applicable"
        ],
        "instructions": [
            "Judge visible evidence and the proposed action, not the category name.",
            "Rows, identifiers, questions, and tags hide the miner's expected categorical outcome.",
            "A failed support check is inconclusive unless evidence directly contradicts the relation.",
            "Confirm delayed effect only when the first hop opens evidence needed by a later hop.",
            "Never provide a numeric reward, score, probability, confidence, or utility.",
            "Edit the embedded draft case when its question, evidence chain, or legal action is wrong.",
        ],
        "reviews": sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{case_set['case_set_id']}|{row['review_id']}".encode("utf-8")
            ).hexdigest(),
        ),
    }
    errors = validate_balanced_review_queue(queue)
    if errors:
        raise ValueError("invalid balanced review queue: " + "; ".join(errors[:8]))
    return queue


def validate_balanced_review_queue(payload: dict[str, Any]) -> list[str]:
    try:
        import jsonschema
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("jsonschema is required for balanced review") from exc
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    errors = [
        f"{'.'.join(str(part) for part in error.absolute_path) or '<root>'}: {error.message}"
        for error in sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(payload),
            key=lambda error: list(error.path),
        )
    ]
    ids = [str(row.get("review_id") or "") for row in payload.get("reviews") or []]
    if len(ids) != len(set(ids)):
        errors.append("review_id values must be unique")
    status = payload.get("annotation_status")
    if status in {"ai_provisional", "human_locked"}:
        for index, row in enumerate(payload.get("reviews") or []):
            required = (
                row.get("review_decision") in {"accept", "reject"}
                and row.get("verifier_outcome")
                in {"supports", "rejects", "inconclusive", "not_applicable"}
                and isinstance(row.get("evidence_chain_valid"), bool)
                and isinstance(row.get("first_action_valid"), bool)
                and row.get("delayed_effect") in {"yes", "no", "not_applicable"}
                and bool(str(row.get("rationale") or "").strip())
            )
            if not required:
                errors.append(f"reviews.{index} is incomplete")
            if row.get("review_decision") == "accept" and not (
                row.get("evidence_chain_valid") and row.get("first_action_valid")
            ):
                errors.append(f"reviews.{index} accepted an invalid case")
            if (
                row.get("review_decision") == "accept"
                and row.get("category") in {"identity", "state"}
                and row.get("verifier_outcome") == "not_applicable"
            ):
                errors.append(
                    f"reviews.{index} accepted a relation case without a categorical outcome"
                )
    checksum = payload.get("locked_sha256")
    if status == "human_locked" and not checksum:
        errors.append("human_locked review queues require locked_sha256")
    if checksum and checksum != _checksum(payload):
        errors.append("balanced review queue locked_sha256 does not match content")
    return errors


def lock_balanced_review_queue(
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
    errors = [
        error for error in validate_balanced_review_queue(locked)
        if "locked_sha256" not in error
    ]
    if errors:
        raise ValueError("cannot lock balanced review queue: " + "; ".join(errors[:8]))
    locked["locked_sha256"] = _checksum(locked)
    errors = validate_balanced_review_queue(locked)
    if errors:
        raise ValueError("invalid locked balanced review queue: " + "; ".join(errors[:8]))
    return locked


def export_reviewed_balanced_case_set(
    queue: dict[str, Any],
    *,
    case_set_id: str,
) -> dict[str, Any]:
    errors = validate_balanced_review_queue(queue)
    if errors:
        raise ValueError("invalid balanced review queue: " + "; ".join(errors[:8]))
    status = str(queue.get("annotation_status") or "")
    if status not in {"ai_provisional", "human_locked"}:
        raise ValueError("balanced review queue must be locked before case export")
    accepted = [
        _finalize_reviewed_case(row)
        for row in queue["reviews"]
        if row["review_decision"] == "accept"
    ]
    if not accepted:
        raise ValueError("balanced review queue has no accepted cases")
    draft = {
        "schema_version": "steam-navigation-gold-cases/v0.1",
        "case_set_id": case_set_id,
        "annotation_status": "draft",
        "annotator": None,
        "locked_sha256": None,
        "cases": accepted,
    }
    return lock_navigation_case_set(
        draft,
        annotation_status=status,
        annotator=str(queue.get("annotator") or "unknown-reviewer"),
    )


def _mine_balanced_overlay(path: Path, overlay, split: str) -> dict[str, list[dict[str, Any]]]:
    result = {category: [] for category in BALANCED_CASE_CATEGORIES}
    events = {node.node_id: node for node in overlay.atomic_events}
    edges = [
        edge for edge in overlay.relations + overlay.l1_structural_relations
        if edge.src in events and edge.dst in events
    ]
    rejected = []
    supported = []
    ambiguous: dict[str, list[Any]] = defaultdict(list)
    has_contradiction = False
    for edge in edges:
        verifier_relations = set(
            ((edge.provenance or {}).get("hard_verifier") or {})
        )
        for relation in sorted(set(edge.relation_probabilities) | verifier_relations):
            verdict = _hard_verdict(edge, relation)
            if relation in _IDENTITY:
                category = f"identity_{_verdict_suffix(verdict)}"
                result[category].append(_edge_candidate(path, overlay, edge, relation, category, split))
            if relation in _STATE:
                category = f"state_{_verdict_suffix(verdict)}"
                result[category].append(_edge_candidate(path, overlay, edge, relation, category, split))
            if relation == "contradicts":
                has_contradiction = True
                for category in ("contradiction_open", "counterevidence_useful"):
                    result[category].append(_edge_candidate(path, overlay, edge, relation, category, split))
            if verdict is False:
                rejected.append((edge, relation))
            elif verdict is True:
                supported.append((edge, relation))
            else:
                ambiguous[edge.src].append((edge, relation))

    mined, _ = _mine_overlay(path, overlay, split)
    for row in mined["delayed_bridge"]:
        result["delayed_two_hop"].append(_retag(row, "delayed_two_hop"))
    if not has_contradiction and events:
        anchor = sorted(events)[0]
        result["counterevidence_empty"].append(
            _synthetic_diagnostic_case(
                path, overlay, split, "counterevidence_empty", anchor, (),
                NavigationActionType.SEARCH_COUNTEREVIDENCE, "contradicts"
            )
        )
    for edge, relation in rejected:
        alternatives = [
            candidate for candidate, _ in supported
            if candidate.edge_id != edge.edge_id
            and {candidate.src, candidate.dst} & {edge.src, edge.dst}
        ]
        if alternatives:
            result["blocked_path_recovery"].append(
                _synthetic_diagnostic_case(
                    path, overlay, split, "blocked_path_recovery", edge.src,
                    (edge.dst,), NavigationActionType.VERIFY, relation,
                )
            )
    contradiction_edges = [edge for edge in edges if "contradicts" in edge.relation_probabilities]
    for contradiction in contradiction_edges:
        if any(
            {edge.src, edge.dst} & {contradiction.src, contradiction.dst}
            for edge, _ in supported if edge.edge_id != contradiction.edge_id
        ):
            result["contradiction_resolve"].append(
                _synthetic_diagnostic_case(
                    path, overlay, split, "contradiction_resolve",
                    contradiction.src, (contradiction.dst,),
                    NavigationActionType.VERIFY, "contradicts",
                )
            )
    for source, rows in ambiguous.items():
        targets = tuple(sorted({edge.dst for edge, _ in rows if edge.dst != source}))
        if len(targets) >= 2:
            result["ambiguity_abstention"].append(
                _synthetic_diagnostic_case(
                    path, overlay, split, "ambiguity_abstention", source,
                    targets, NavigationActionType.STOP, None,
                )
            )
    for category in result:
        result[category] = list({row["case_id"]: row for row in result[category]}.values())
    return result


def _select_unique_overlays(
    overlay_paths: Iterable[Path],
) -> tuple[list[tuple[Path, Any]], int]:
    selected: dict[str, tuple[Path, Any]] = {}
    count = 0
    for raw_path in sorted(
        {Path(value).expanduser().resolve() for value in overlay_paths}
    ):
        loaded = load_overlay_artifact(raw_path)
        overlay = loaded.overlay
        count += 1
        key = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        current = selected.get(key)
        if current is None or str(raw_path) < str(current[0]):
            selected[key] = (raw_path, overlay)
    return sorted(selected.values(), key=lambda row: str(row[0])), count - len(selected)


def _edge_candidate(path, overlay, edge, relation: str, category: str, split: str) -> dict[str, Any]:
    action_type = (
        NavigationActionType.TRACK_ENTITY
        if relation in _IDENTITY
        else NavigationActionType.INSPECT_STATE_CHANGE
        if relation == "state_transition"
        else NavigationActionType.SEARCH_COUNTEREVIDENCE
        if relation == "contradicts"
        else NavigationActionType.FOLLOW_DEPENDENCY
    )
    case = _case(
        path, overlay, split, category, edge.src, (edge.dst,), action_type, relation
    )
    if relation not in edge.relation_probabilities:
        case["tags"].append("offline_verifier_challenge")
        case["notes"] = (
            "Hard-verifier diagnostic for a relation rejected before admission; "
            "it is not a legal online graph edge unless review explicitly restores it."
        )
    return case


def _synthetic_diagnostic_case(path, overlay, split, category, source, targets, action_type, relation):
    return _case(path, overlay, split, category, source, targets, action_type, relation)


def _case(path, overlay, split, category, source, targets, action_type, relation):
    digest = hashlib.sha256(
        f"{path}|{category}|{source}|{'|'.join(targets)}|{relation}".encode("utf-8")
    ).hexdigest()[:12]
    action = {
        "action_type": action_type.value,
        "source_id": source if action_type is not NavigationActionType.STOP else None,
        "target_ids": list(targets) if action_type is not NavigationActionType.STOP else [],
        "relation": relation if action_type is not NavigationActionType.STOP else None,
    }
    return {
        "case_id": f"{overlay.video_id}:{category}:{digest}",
        "overlay_path": str(path),
        "overlay_id": overlay.overlay_id,
        "question": f"Review the {category.replace('_', ' ')} reasoning path from {source}.",
        "seed_event_ids": [source],
        "missing_roles": [_role(category)],
        "graph_read_budget": 3 if category in {"delayed_two_hop", "blocked_path_recovery"} else 2,
        "gold_event_ids": list(dict.fromkeys((source, *targets))),
        "acceptable_first_actions": [action],
        "required_relation_types": [relation] if relation else [],
        "tags": [category, "correction_sensitive", split],
        "split": split,
        "notes": "Automatically mined diagnostic candidate; reviewer must verify evidence, outcome, and action.",
        "_video_id": overlay.video_id,
    }


def _retag(row: dict[str, Any], category: str) -> dict[str, Any]:
    value = dict(row)
    original = value["case_id"].replace(":delayed_bridge:", f":{category}:")
    suffix = hashlib.sha256(str(value["overlay_path"]).encode("utf-8")).hexdigest()[:8]
    value["case_id"] = f"{original}:{suffix}"
    value["tags"] = [category if tag == "delayed_bridge" else tag for tag in value["tags"]]
    return value


def _hard_verdict(edge, relation: str) -> bool | None:
    value = ((edge.provenance or {}).get("hard_verifier") or {}).get(relation)
    if not isinstance(value, dict) or "passed" not in value:
        return None
    return value.get("passed") is True


def _verdict_suffix(value: bool | None) -> str:
    return "support" if value is True else "reject" if value is False else "inconclusive"


def _role(category: str) -> str:
    if category.startswith("identity"):
        return "identity"
    if category.startswith("state"):
        return "state_transition"
    if category.startswith("counter") or category.startswith("contradiction"):
        return "counterevidence"
    return "bridge"


def _row_order(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row.get("split")), str(row.get("overlay_id")), str(row.get("case_id"))


def _review_stratum(category: str) -> str:
    if category.startswith("identity_"):
        return "identity"
    if category.startswith("state_"):
        return "state"
    if category.startswith("contradiction_"):
        return "contradiction"
    if category.startswith("counterevidence_"):
        return "counterevidence"
    return category


def _blind_review_case(case: dict[str, Any], stratum: str) -> dict[str, Any]:
    value = json.loads(json.dumps(case))
    digest = hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest()[:16]
    value["case_id"] = f"review-case:{digest}"
    source = str((value.get("seed_event_ids") or ["evidence"])[0])
    value["question"] = (
        f"Review the proposed {stratum.replace('_', ' ')} reasoning path from {source}."
    )
    value["tags"] = [
        f"review_stratum:{stratum}",
        *[
            tag
            for tag in value.get("tags") or []
            if tag not in BALANCED_CASE_CATEGORIES
            and tag != "offline_verifier_challenge"
        ],
    ]
    value["notes"] = (
        "Outcome-blinded diagnostic candidate; judge visible evidence and online action legality."
    )
    return value


def _finalize_reviewed_case(row: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(row["case"]))
    stratum = str(row["category"])
    outcome = str(row["verifier_outcome"])
    if stratum in {"identity", "state"}:
        suffix = {
            "supports": "support",
            "rejects": "reject",
            "inconclusive": "inconclusive",
        }.get(outcome)
        if suffix is None:
            raise ValueError(
                f"accepted {stratum} review requires a categorical relation outcome"
            )
        category = f"{stratum}_{suffix}"
    else:
        category = stratum
    value["tags"] = [
        category if tag == f"review_stratum:{stratum}" else tag
        for tag in value.get("tags") or []
    ]
    value["notes"] = "Independently reviewed and accepted from an outcome-blinded queue."
    return value


def _candidate_basis(category: str) -> list[str]:
    return [
        f"review_stratum:{category}",
        "candidate_only:not_gold",
        "requires_visible_evidence_review",
        "requires_categorical_verifier_outcome_review",
    ]


def _checksum(payload: dict[str, Any]) -> str:
    clone = json.loads(json.dumps(payload))
    clone["locked_sha256"] = None
    return hashlib.sha256(
        json.dumps(clone, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
