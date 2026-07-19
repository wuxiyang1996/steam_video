"""Compare four bounded graph-navigation strategies on fixed evidence targets."""

from __future__ import annotations

import argparse
import json
import re
from collections import deque
from pathlib import Path
from typing import Any


STRATEGIES = (
    "semantic_only",
    "event_only",
    "native_l1_candidate",
    "verified_dependency",
)
TEMPORAL_RELATIONS = frozenset({"temporal_next", "before", "overlaps", "during"})
DEPENDENCY_RELATIONS = frozenset({"transition_support", "state_transition"})
CANDIDATE_L1_RELATIONS = frozenset(
    {
        "same_instance_candidate",
        "reappears_candidate",
        "observation_support",
        "same_entity",
        "same_object",
    }
)


def evaluate_navigation_ablation(
    overlay: dict[str, Any],
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    events = {
        str(node["node_id"]): node
        for node in overlay.get("atomic_events") or []
        if isinstance(node, dict) and node.get("node_id")
    }
    results: dict[str, list[dict[str, Any]]] = {name: [] for name in STRATEGIES}
    for case in cases:
        question = str(case.get("question") or "")
        gold = {str(value) for value in case.get("gold_event_ids") or []}
        unknown = gold - set(events)
        if not question or not gold or unknown:
            raise ValueError(
                f"case {case.get('case_id')} requires a question and known gold_event_ids; "
                f"unknown={sorted(unknown)}"
            )
        budget = int(case.get("graph_read_budget") or 4)
        if budget < 1:
            raise ValueError("graph_read_budget must be positive")
        for strategy in STRATEGIES:
            acquired = _run_strategy(
                strategy,
                question=question,
                budget=budget,
                events=events,
                overlay=overlay,
            )
            hits = gold & set(acquired)
            results[strategy].append(
                {
                    "case_id": case.get("case_id"),
                    "acquired_event_ids": acquired,
                    "graph_reads": len(acquired),
                    "gold_count": len(gold),
                    "hit_count": len(hits),
                    "evidence_recall": len(hits) / len(gold),
                    "answerable": hits == gold,
                }
            )
    summary = {
        strategy: _summarize(rows) for strategy, rows in results.items()
    }
    return {
        "schema_version": "steam-navigation-ablation/v0.1",
        "case_count": len(cases),
        "strategies": summary,
        "cases": results,
        "semantics": {
            "answerable": "all independently specified gold evidence events acquired",
            "graph_reads": "persisted event reads; imagined observations are excluded",
            "native_l1_candidate": "candidate/native L1 edges may choose reads but are not answer evidence",
            "verified_dependency": "only hard-verified or accepted-track-derived dependency edges",
        },
    }


def _run_strategy(
    strategy: str,
    *,
    question: str,
    budget: int,
    events: dict[str, dict[str, Any]],
    overlay: dict[str, Any],
) -> list[str]:
    ranked = sorted(
        events,
        key=lambda event_id: (
            -_lexical_score(question, _event_text(events[event_id])),
            float((events[event_id].get("time_span") or {}).get("start_s") or 0.0),
            event_id,
        ),
    )
    if strategy == "semantic_only":
        return ranked[:budget]
    acquired = [ranked[0]] if ranked else []
    adjacency = _event_adjacency(overlay, verified_only=strategy == "verified_dependency")
    if strategy == "native_l1_candidate":
        _add_native_l1_adjacency(adjacency, overlay, events)
    queue = deque(acquired)
    while queue and len(acquired) < budget:
        current = queue.popleft()
        neighbors = sorted(
            adjacency.get(current, ()),
            key=lambda event_id: (
                -_lexical_score(question, _event_text(events[event_id])),
                event_id,
            ),
        )
        for neighbor in neighbors:
            if neighbor in acquired or neighbor not in events:
                continue
            acquired.append(neighbor)
            queue.append(neighbor)
            if len(acquired) >= budget:
                break
    for event_id in ranked:
        if len(acquired) >= budget:
            break
        if event_id not in acquired:
            acquired.append(event_id)
    return acquired


def _event_adjacency(
    overlay: dict[str, Any],
    *,
    verified_only: bool,
) -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {}
    for edge in overlay.get("relations") or []:
        if not isinstance(edge, dict):
            continue
        names = set((edge.get("relation_probabilities") or {}).keys())
        allowed = bool(names & TEMPORAL_RELATIONS)
        if verified_only and names & DEPENDENCY_RELATIONS:
            allowed = allowed or _dependency_verified(edge)
        if not verified_only:
            allowed = allowed and not bool(names & DEPENDENCY_RELATIONS)
        if not allowed:
            continue
        src, dst = str(edge.get("src") or ""), str(edge.get("dst") or "")
        adjacency.setdefault(src, set()).add(dst)
        adjacency.setdefault(dst, set()).add(src)
    return adjacency


def _dependency_verified(edge: dict[str, Any]) -> bool:
    provenance = edge.get("provenance") or {}
    if provenance.get("accepted_identity_track"):
        return True
    verifier = provenance.get("hard_verifier")
    return isinstance(verifier, dict) and any(
        isinstance(value, dict) and value.get("passed") is True
        for name, value in verifier.items()
        if name in DEPENDENCY_RELATIONS
    )


def _add_native_l1_adjacency(
    adjacency: dict[str, set[str]],
    overlay: dict[str, Any],
    events: dict[str, dict[str, Any]],
) -> None:
    event_by_l1: dict[str, set[str]] = {}
    for event_id, event in events.items():
        for ref in event.get("source_segments") or []:
            event_by_l1.setdefault(str(ref), set()).add(event_id)
    for edge in overlay.get("l1_structural_relations") or []:
        if not isinstance(edge, dict):
            continue
        names = set((edge.get("relation_probabilities") or {}).keys())
        if not names & CANDIDATE_L1_RELATIONS:
            continue
        src_events = event_by_l1.get(str(edge.get("src") or ""), set())
        dst_events = event_by_l1.get(str(edge.get("dst") or ""), set())
        for src in src_events:
            for dst in dst_events:
                if src == dst:
                    continue
                adjacency.setdefault(src, set()).add(dst)
                adjacency.setdefault(dst, set()).add(src)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "case_count": count,
        "answer_accuracy": sum(bool(row["answerable"]) for row in rows) / count
        if count
        else None,
        "mean_evidence_recall": sum(float(row["evidence_recall"]) for row in rows)
        / count
        if count
        else None,
        "mean_graph_reads": sum(int(row["graph_reads"]) for row in rows) / count
        if count
        else None,
    }


def _event_text(event: dict[str, Any]) -> str:
    return str((event.get("metadata") or {}).get("predicate") or event.get("text") or "")


def _lexical_score(question: str, text: str) -> int:
    return len(_tokens(question) & _tokens(text))


def _tokens(value: str) -> set[str]:
    stop = {"a", "an", "the", "is", "was", "what", "why", "how", "did", "to"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.casefold())
        if len(token) > 1 and token not in stop
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
    cases_payload = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = cases_payload.get("cases") if isinstance(cases_payload, dict) else cases_payload
    if not isinstance(cases, list):
        raise ValueError("cases JSON must be an array or an object with a cases array")
    report = evaluate_navigation_ablation(overlay, cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["strategies"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
