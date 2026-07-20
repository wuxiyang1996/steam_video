"""Mine stratified draft navigation cases without claiming automatic gold."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from memory_graph.navigation import NavigationActionType
from memory_graph.types import CausalTemporalOverlay, RelationBelief

from .belief import _verified_relations
from .overlay_io import load_overlay_artifact


CASE_CATEGORIES = (
    "temporal",
    "identity",
    "state_transition",
    "verified_dependency",
    "delayed_bridge",
    "counterevidence",
)
DEFAULT_QUOTAS = {
    "temporal": 8,
    "identity": 8,
    "state_transition": 8,
    "verified_dependency": 8,
    "delayed_bridge": 6,
    "counterevidence": 2,
}
_IDENTITY = {
    "same_entity",
    "same_object",
    "same_instance_candidate",
    "reappears_candidate",
}
_DEPENDENCY = {
    "state_transition",
    "transition_support",
    "response_candidate",
    "observation_support",
    "explains",
    "enables",
}


def mine_navigation_cases(
    overlay_paths: Iterable[Path],
    *,
    case_set_id: str,
    desired_count: int = 40,
    per_video_limit: int = 5,
    quotas: dict[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if desired_count <= 0 or per_video_limit <= 0:
        raise ValueError("desired_count and per_video_limit must be positive")
    selected_overlays, duplicates = _select_overlay_per_video(overlay_paths)
    split_by_video = _video_splits(sorted(selected_overlays))
    candidates: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CASE_CATEGORIES
    }
    relation_coverage: Counter[str] = Counter()
    for video_id, (path, overlay) in selected_overlays.items():
        mined, relation_counts = _mine_overlay(path, overlay, split_by_video[video_id])
        relation_coverage.update(relation_counts)
        for category, rows in mined.items():
            candidates[category].extend(rows)
    requested = dict(DEFAULT_QUOTAS if quotas is None else quotas)
    chosen: list[dict[str, Any]] = []
    video_counts: Counter[str] = Counter()
    chosen_ids: set[str] = set()
    selection_order = sorted(
        CASE_CATEGORIES,
        key=lambda category: (
            len(candidates[category]) / max(1, int(requested.get(category, 0))),
            category,
        ),
    )
    for category in selection_order:
        rows = sorted(candidates[category], key=_candidate_order)
        target = min(int(requested.get(category, 0)), desired_count - len(chosen))
        for original in rows:
            if target <= 0:
                break
            row = dict(original)
            video_id = str(row.pop("_video_id"))
            if video_counts[video_id] >= per_video_limit or row["case_id"] in chosen_ids:
                continue
            chosen.append(row)
            chosen_ids.add(row["case_id"])
            video_counts[video_id] += 1
            target -= 1
    if len(chosen) < desired_count:
        overflow = sorted(
            (
                row
                for category in CASE_CATEGORIES
                for row in candidates[category]
                if row["case_id"] not in chosen_ids
            ),
            key=_candidate_order,
        )
        for original in overflow:
            if len(chosen) >= desired_count:
                break
            row = dict(original)
            video_id = str(row.pop("_video_id"))
            if video_counts[video_id] >= per_video_limit:
                continue
            chosen.append(row)
            chosen_ids.add(row["case_id"])
            video_counts[video_id] += 1
    selected_counts = Counter(tag for case in chosen for tag in case["tags"] if tag in CASE_CATEGORIES)
    available_counts = {category: len(candidates[category]) for category in CASE_CATEGORIES}
    deficits = {
        category: max(0, int(requested.get(category, 0)) - selected_counts[category])
        for category in CASE_CATEGORIES
    }
    case_set = {
        "schema_version": "steam-navigation-gold-cases/v0.1",
        "case_set_id": case_set_id,
        "annotation_status": "draft",
        "annotator": None,
        "locked_sha256": None,
        "cases": chosen,
    }
    report = {
        "schema_version": "steam-navigation-case-mining-report/v0.1",
        "case_set_id": case_set_id,
        "requested_case_count": desired_count,
        "selected_case_count": len(chosen),
        "selected_video_count": len({case["overlay_id"] for case in chosen}),
        "available_overlay_count": len(selected_overlays),
        "duplicate_overlay_count": duplicates,
        "requested_quotas": requested,
        "available_candidates": available_counts,
        "selected_categories": dict(selected_counts),
        "quota_deficits": deficits,
        "relation_coverage": dict(relation_coverage),
        "split_case_counts": dict(Counter(case.get("split") for case in chosen)),
        "diagnostic_case_count": sum("delayed_preference" in case["tags"] for case in chosen),
        "formal_ready": False,
        "formal_blockers": [
            "all mined cases are draft hypotheses, not independent human gold",
            "acceptable actions and evidence chains require blinded human review",
            *(
                [f"quota deficits remain: {deficits}"]
                if any(deficits.values())
                else []
            ),
        ],
    }
    return case_set, report


def _select_overlay_per_video(
    overlay_paths: Iterable[Path],
) -> tuple[dict[str, tuple[Path, CausalTemporalOverlay]], int]:
    choices: dict[str, list[tuple[tuple[int, int, str], Path, CausalTemporalOverlay]]] = defaultdict(list)
    count = 0
    for raw_path in overlay_paths:
        path = raw_path.expanduser().resolve()
        loaded = load_overlay_artifact(path)
        overlay = loaded.overlay
        count += 1
        admitted = sum(
            len(_admitted_relations(edge))
            for edge in overlay.relations + overlay.l1_structural_relations
        )
        rare_diagnostic = sum(
            bool(_admitted_relations(edge) & (_DEPENDENCY | {"contradicts"}))
            for edge in overlay.relations
        )
        identity = sum(
            bool(_admitted_relations(edge) & _IDENTITY)
            for edge in overlay.relations
        )
        quality = (
            rare_diagnostic * 1_000_000 + identity * 1000 + admitted,
            len(overlay.atomic_events),
            str(path),
        )
        choices[overlay.video_id].append((quality, path, overlay))
    selected: dict[str, tuple[Path, CausalTemporalOverlay]] = {}
    for video_id, rows in choices.items():
        ordered = sorted(rows)
        if ordered:
            selected[video_id] = ordered[-1][1], ordered[-1][2]
    return selected, count - len(selected)


def _mine_overlay(
    path: Path,
    overlay: CausalTemporalOverlay,
    split: str,
) -> tuple[dict[str, list[dict[str, Any]]], Counter[str]]:
    result: dict[str, list[dict[str, Any]]] = {category: [] for category in CASE_CATEGORIES}
    events = {node.node_id: node for node in overlay.atomic_events}
    admitted_edges: list[tuple[RelationBelief, str]] = []
    coverage: Counter[str] = Counter()
    for edge in overlay.relations + overlay.l1_structural_relations:
        if edge.src not in events or edge.dst not in events:
            continue
        for relation in sorted(_admitted_relations(edge)):
            admitted_edges.append((edge, relation))
            coverage[relation] += 1
            category = _relation_category(relation)
            if category is None:
                continue
            result[category].append(
                _edge_case(path, overlay, events, edge, relation, category, split)
            )
    direct_pairs = {
        (edge.src, edge.dst)
        for edge, relation in admitted_edges
        if relation not in {"before"}
    }
    outgoing: dict[str, list[tuple[RelationBelief, str]]] = defaultdict(list)
    for edge, relation in admitted_edges:
        if relation in {"temporal_next"} | _IDENTITY | _DEPENDENCY:
            outgoing[edge.src].append((edge, relation))
    for src, first_rows in outgoing.items():
        for first_edge, first_relation in first_rows:
            middle = first_edge.dst
            for second_edge, second_relation in outgoing.get(middle, []):
                dst = second_edge.dst
                if src == dst or (src, dst) in direct_pairs:
                    continue
                result["delayed_bridge"].append(
                    _bridge_case(
                        path,
                        overlay,
                        events,
                        first_edge,
                        first_relation,
                        second_edge,
                        second_relation,
                        split,
                    )
                )
    for rows in result.values():
        deduped = {row["case_id"]: row for row in rows}
        rows[:] = list(deduped.values())
    return result, coverage


def _edge_case(
    path: Path,
    overlay: CausalTemporalOverlay,
    events: dict[str, Any],
    edge: RelationBelief,
    relation: str,
    category: str,
    split: str,
) -> dict[str, Any]:
    source_text = _short_text(events[edge.src].text or edge.src)
    target_text = _short_text(events[edge.dst].text or edge.dst)
    question, role = _question_and_role(category, source_text, target_text)
    action_type = _action_type(relation)
    return {
        "case_id": _case_id(overlay.video_id, category, edge.edge_id, relation),
        "overlay_path": str(path),
        "overlay_id": overlay.overlay_id,
        "question": question,
        "seed_event_ids": [edge.src],
        "missing_roles": [role],
        "graph_read_budget": 2,
        "gold_event_ids": [edge.src, edge.dst],
        "acceptable_first_actions": [
            {
                "action_type": action_type.value,
                "source_id": edge.src,
                "target_ids": [edge.dst],
            }
        ],
        "required_relation_types": [relation],
        "tags": [category, "one_step_control", split],
        "split": split,
        "notes": "Automatically mined draft; independently verify the question, evidence chain, and acceptable action.",
        "_video_id": overlay.video_id,
    }


def _bridge_case(
    path: Path,
    overlay: CausalTemporalOverlay,
    events: dict[str, Any],
    first_edge: RelationBelief,
    first_relation: str,
    second_edge: RelationBelief,
    second_relation: str,
    split: str,
) -> dict[str, Any]:
    source_text = _short_text(events[first_edge.src].text or first_edge.src)
    target_text = _short_text(events[second_edge.dst].text or second_edge.dst)
    action_type = _action_type(first_relation)
    if first_relation == "temporal_next":
        question = (
            f"What happened immediately after '{source_text}', and which intermediate "
            f"evidence leads to '{target_text}'?"
        )
    elif first_relation in _IDENTITY:
        question = (
            f"Following the same entity from '{source_text}', which intermediate "
            f"observation leads to '{target_text}'?"
        )
    else:
        question = f"Starting from '{source_text}', what evidence chain leads to '{target_text}'?"
    return {
        "case_id": _case_id(
            overlay.video_id,
            "delayed_bridge",
            first_edge.edge_id,
            second_edge.edge_id,
        ),
        "overlay_path": str(path),
        "overlay_id": overlay.overlay_id,
        "question": question,
        "seed_event_ids": [first_edge.src],
        "missing_roles": ["bridge"],
        "graph_read_budget": 3,
        "gold_event_ids": [first_edge.src, first_edge.dst, second_edge.dst],
        "acceptable_first_actions": [
            {
                "action_type": action_type.value,
                "source_id": first_edge.src,
                "target_ids": [first_edge.dst],
            }
        ],
        "required_relation_types": list(dict.fromkeys((first_relation, second_relation))),
        "tags": ["delayed_bridge", "delayed_preference", split],
        "split": split,
        "notes": "Automatically mined two-hop draft with no admitted direct non-before edge; human review must confirm that the bridge is genuinely required.",
        "_video_id": overlay.video_id,
    }


def _admitted_relations(edge: RelationBelief) -> set[str]:
    names = set(_verified_relations(edge))
    if edge.status.value == "deterministic":
        names.update(edge.relation_probabilities)
    provenance = edge.provenance or {}
    accepted = provenance.get("accepted_identity_track")
    if accepted:
        names.update(set(edge.relation_probabilities) & (_IDENTITY | _DEPENDENCY))
    return names


def _relation_category(relation: str) -> str | None:
    if relation == "temporal_next":
        return "temporal"
    if relation in _IDENTITY:
        return "identity"
    if relation == "state_transition":
        return "state_transition"
    if relation in _DEPENDENCY:
        return "verified_dependency"
    if relation == "contradicts":
        return "counterevidence"
    return None


def _action_type(relation: str) -> NavigationActionType:
    if relation == "temporal_next":
        return NavigationActionType.TEMPORAL_FORWARD
    if relation in _IDENTITY:
        return NavigationActionType.TRACK_ENTITY
    if relation == "state_transition":
        return NavigationActionType.INSPECT_STATE_CHANGE
    if relation in {"explains", "enables"}:
        return NavigationActionType.EFFECT
    if relation == "contradicts":
        return NavigationActionType.SEARCH_COUNTEREVIDENCE
    return NavigationActionType.FOLLOW_DEPENDENCY


def _question_and_role(
    category: str,
    source_text: str,
    target_text: str,
) -> tuple[str, str]:
    return {
        "temporal": (f"What happened immediately after '{source_text}'?", "temporal"),
        "identity": (f"Does '{target_text}' contain the same entity as '{source_text}'?", "identity"),
        "state_transition": (f"How did the tracked entity reach '{target_text}' after '{source_text}'?", "state_transition"),
        "verified_dependency": (f"Does the verified event '{target_text}' depend on '{source_text}'?", "dependency"),
        "counterevidence": (f"Does '{target_text}' contradict '{source_text}'?", "counterevidence"),
    }[category]


def _video_splits(video_ids: list[str]) -> dict[str, str]:
    ordered = sorted(
        video_ids,
        key=lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest(),
    )
    if len(ordered) < 3:
        return {video_id: "dev" for video_id in ordered}
    test_count = max(1, round(len(ordered) * 0.2))
    dev_count = max(1, round(len(ordered) * 0.2))
    result = {video_id: "train" for video_id in ordered}
    for video_id in ordered[:test_count]:
        result[video_id] = "test"
    for video_id in ordered[test_count : test_count + dev_count]:
        result[video_id] = "dev"
    return result


def _candidate_order(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        0 if "delayed_preference" in row["tags"] else 1,
        row["split"],
        row["overlay_id"],
        row["case_id"],
    )


def _case_id(video_id: str, category: str, *values: str) -> str:
    digest = hashlib.sha256("|".join(values).encode("utf-8")).hexdigest()[:12]
    return f"{video_id}:{category}:{digest}"


def _short_text(value: str, limit: int = 120) -> str:
    normalized = " ".join(value.split()).replace("'", "’")
    return normalized if len(normalized) <= limit else normalized[: limit - 1] + "…"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--glob", default="**/causal_temporal_overlay.json")
    parser.add_argument("--case-set-id", required=True)
    parser.add_argument("--desired-count", type=int, default=40)
    parser.add_argument("--per-video-limit", type=int, default=5)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.overlay_root.expanduser().resolve().glob(args.glob))
    cases, report = mine_navigation_cases(
        paths,
        case_set_id=args.case_set_id,
        desired_count=args.desired_count,
        per_video_limit=args.per_video_limit,
    )
    for path, payload in ((args.output, cases), (args.report, report)):
        destination = path.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
