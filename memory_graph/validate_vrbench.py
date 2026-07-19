#!/usr/bin/env python3
"""Validate VRBench temporal and multi-hop coverage with Video_Skills.

The adapter's hidden ``reasoning_process`` timestamps are evaluation targets,
not causal gold labels.  This module evaluates only temporal coverage, ordering,
and intermediate-step (bridge) coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

from .adapter import canonical_to_memory_nodes
from .reliability import audit_l1_nodes


HIDDEN_SOURCE_TYPES = frozenset(
    {
        "official_answer",
        "qa_answer",
        "reasoning_process",
        "reasoning_process_step",
        "video_summary",
    }
)
DISCOVERED_NODE_TYPES = frozenset({"observation", "event", "dialogue_span"})
DEFAULT_WORKSPACE = Path("/fs/gamma-projects/vlm-robot")
TARGET_REASONING_TYPES = (
    "Event Attribution",
    "Logical Linkage",
    "Implicit Inference",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate VRBench temporal/multi-hop L1 coverage. Hidden reasoning_process "
            "spans are evaluation targets, never causal gold."
        )
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_WORKSPACE / "datasets"))
    parser.add_argument("--video-skills-root", default=str(DEFAULT_WORKSPACE / "Video_Skills"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--phase", choices=("smoke", "locked"), default="smoke")
    parser.add_argument(
        "--reasoning-type",
        action="append",
        dest="reasoning_types",
        help="Keep this VRBench reasoning_type; repeat to select multiple types.",
    )
    parser.add_argument("--mode", choices=("expert_demo", "video_only"), required=True)
    parser.add_argument(
        "--canonical-dir",
        help="Directory containing pre-generated Video_Skills canonical JSON files.",
    )
    return parser


def _add_video_skills_import(root: Path) -> None:
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Video_Skills root does not exist: {resolved}")
    value = str(resolved)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_items(
    adapter_class: Any,
    dataset_root: Path,
    *,
    reasoning_types: list[str] | None,
    limit: int | None,
) -> list[Any]:
    if limit is not None and limit < 1:
        raise ValueError("--limit must be a positive integer")
    wanted = {value.casefold() for value in reasoning_types or []}
    adapter = adapter_class(dataset_root, split="eval")
    selected = []
    for item in adapter.iter_items():
        reasoning_type = str((item.metadata or {}).get("reasoning_type") or "")
        if wanted and reasoning_type.casefold() not in wanted:
            continue
        selected.append(item)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def _reasoning_targets(item: Any) -> list[dict[str, Any]]:
    """Use adapter-parsed hidden annotation spans without exposing them as inputs."""
    targets = []
    for segment in item.annotation_segments:
        if segment.get("source_type") != "reasoning_process_step":
            continue
        span = _normalized_span(segment.get("time_span"))
        if span is None:
            continue
        targets.append(
            {
                "target_id": str(segment.get("segment_id") or f"step:{len(targets) + 1}"),
                "step_index": len(targets),
                "time_span": span,
                "text": segment.get("text"),
            }
        )
    return targets


def _normalized_span(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        start = float(value["start_s"])
        end = float(value["end_s"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (math.isfinite(start) and math.isfinite(end)):
        return None
    if end < start:
        start, end = end, start
    return {"start_s": start, "end_s": end}


def _canonical_candidates(root: Path, example_id: str, video_id: str, question_id: str) -> Iterable[Path]:
    safe_example_id = example_id.replace(":", "_").replace("/", "_")
    names = (
        f"{example_id}.json",
        f"{safe_example_id}.json",
        f"{question_id}.json",
        "canonical_example.json",
        "canonical.json",
    )
    for name in names[:2]:
        yield root / name
    for parent in (root / video_id / question_id, root / video_id, root / safe_example_id):
        for name in names[2:]:
            yield parent / name


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"canonical JSON must contain an object: {path}")
    return payload


class CanonicalStore:
    def __init__(self, root: Path | None) -> None:
        self.root = root.expanduser().resolve() if root else None
        self._index: dict[str, Path] | None = None
        if self.root is not None and not self.root.is_dir():
            raise FileNotFoundError(f"--canonical-dir does not exist: {self.root}")

    def get(self, item: Any) -> tuple[Path, dict[str, Any]] | None:
        if self.root is None:
            return None
        question_id = str((item.question or {}).get("question_id") or "")
        for candidate in _canonical_candidates(
            self.root, str(item.example_id), str(item.video_id), question_id
        ):
            if candidate.is_file():
                payload = _load_json_object(candidate)
                if str(payload.get("example_id") or item.example_id) == str(item.example_id):
                    return candidate, payload
        if self._index is None:
            self._index = self._build_index()
        path = self._index.get(str(item.example_id))
        return (path, _load_json_object(path)) if path else None

    def _build_index(self) -> dict[str, Path]:
        assert self.root is not None
        index: dict[str, Path] = {}
        for path in sorted(self.root.rglob("*.json")):
            try:
                payload = _load_json_object(path)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
                continue
            example_id = payload.get("example_id")
            if example_id:
                index.setdefault(str(example_id), path)
        return index


def _extract_l1_graph(canonical: dict[str, Any]) -> dict[str, Any]:
    metadata = canonical.get("metadata") or {}
    graph = metadata.get("clue_memory_graph")
    if isinstance(graph, dict):
        return graph
    evidence_index = canonical.get("evidence_index")
    if isinstance(evidence_index, dict):
        return evidence_index
    raise ValueError(
        f"canonical {canonical.get('example_id', '<unknown>')} has no "
        "metadata.clue_memory_graph or evidence_index"
    )


def _is_discovered_l1_node(node: dict[str, Any]) -> bool:
    if node.get("node_type") not in DISCOVERED_NODE_TYPES:
        return False
    if _normalized_span(node.get("time_span")) is None:
        return False
    if str(node.get("source_type") or "") in HIDDEN_SOURCE_TYPES:
        return False
    if node.get("discovery_status") == "provided_supervision":
        return False
    visibility = node.get("visibility") or {}
    if visibility.get("hidden_supervision") or visibility.get("visible_to_agent") is False:
        return False
    return True


def _canonical_discovered_nodes(canonical: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    graph, adapted = canonical_to_memory_nodes(canonical)
    nodes = [
        {
            "node_id": node.node_id,
            "node_type": node.node_type,
            "source_type": node.metadata.get("source_type"),
            "discovery_status": node.metadata.get("discovery_status"),
            "time_span": {
                "start_s": node.time_span.start_s,
                "end_s": node.time_span.end_s,
            },
            "text": node.text,
        }
        for node in adapted
        if not _memory_node_uses_hidden_supervision(node)
    ]
    nodes.sort(key=_node_sort_key)
    return graph, nodes


def _memory_node_uses_hidden_supervision(node: Any) -> bool:
    visibility = node.metadata.get("visibility")
    if isinstance(visibility, dict):
        if visibility.get("hidden_supervision") or visibility.get("visible_to_agent") is False:
            return True
    source_type = str(node.metadata.get("source_type") or "")
    return (
        source_type in HIDDEN_SOURCE_TYPES
        or node.metadata.get("discovery_status") == "provided_supervision"
    )


def _provisional_annotation_graph(item: Any, targets: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Construct an expert-demo-only temporal graph from adapter annotations."""
    nodes = [
        {
            "node_id": target["target_id"],
            "node_type": "event",
            "source_type": "reasoning_process_step",
            "discovery_status": "provided_supervision",
            "time_span": target["time_span"],
            "text": target.get("text"),
        }
        for target in targets
    ]
    nodes.sort(key=_node_sort_key)
    edges = [
        {
            "edge_id": f"temporal:{left['node_id']}->{right['node_id']}",
            "src": left["node_id"],
            "dst": right["node_id"],
            "edge_type": "temporal_next",
        }
        for left, right in zip(nodes, nodes[1:])
    ]
    return (
        {
            "graph_id": f"provisional_vrbench_temporal:{item.example_id}",
            "graph_kind": "provisional_annotation_temporal_graph",
            "nodes": nodes,
            "edges": edges,
            "causal_labels": False,
        },
        nodes,
    )


def _node_sort_key(node: dict[str, Any]) -> tuple[float, float, str]:
    span = node["time_span"]
    return (float(span["start_s"]), float(span["end_s"]), str(node["node_id"]))


def _overlap_s(left: dict[str, float], right: dict[str, float]) -> float:
    return max(0.0, min(left["end_s"], right["end_s"]) - max(left["start_s"], right["start_s"]))


def _match_targets(
    targets: list[dict[str, Any]], nodes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[int | None]]:
    results = []
    matched_indices: list[int | None] = []
    for target in targets:
        overlaps = [_overlap_s(target["time_span"], node["time_span"]) for node in nodes]
        best_index = max(range(len(nodes)), key=lambda index: overlaps[index], default=None)
        best_overlap = overlaps[best_index] if best_index is not None else 0.0
        covered = best_overlap > 0.0
        matched_indices.append(best_index if covered else None)
        duration = target["time_span"]["end_s"] - target["time_span"]["start_s"]
        results.append(
            {
                **target,
                "covered": covered,
                "best_overlap_s": round(best_overlap, 3),
                "overlap_ratio": round(best_overlap / duration, 3) if duration > 0 else None,
                "matched_node_id": nodes[best_index]["node_id"] if covered and best_index is not None else None,
            }
        )
    return results, matched_indices


def _order_consistency(matched_indices: list[int | None]) -> tuple[float | None, int, int]:
    comparable = 0
    consistent = 0
    for left in range(len(matched_indices)):
        if matched_indices[left] is None:
            continue
        for right in range(left + 1, len(matched_indices)):
            if matched_indices[right] is None:
                continue
            comparable += 1
            if matched_indices[left] <= matched_indices[right]:
                consistent += 1
    rate = consistent / comparable if comparable else None
    return rate, consistent, comparable


def _evaluate(
    item: Any,
    graph: dict[str, Any],
    nodes: list[dict[str, Any]],
    source: str,
    *,
    l1_reliability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    targets = _reasoning_targets(item)
    matches, matched_indices = _match_targets(targets, nodes)
    covered_count = sum(bool(row["covered"]) for row in matches)
    recall = covered_count / len(matches) if matches else None
    order_rate, order_hits, order_pairs = _order_consistency(matched_indices)
    bridge_rows = matches[1:-1] if len(matches) >= 3 else []
    bridge_hits = sum(bool(row["covered"]) for row in bridge_rows)
    bridge_coverage = bridge_hits / len(bridge_rows) if bridge_rows else None
    temporal_valid = all(
        node["time_span"]["end_s"] >= node["time_span"]["start_s"] for node in nodes
    )
    gates = {
        "has_timestamped_reasoning_chain": {
            "pass": len(targets) >= 2,
            "value": len(targets),
            "threshold": ">=2",
        },
        "reasoning_step_span_coverage": {
            "pass": recall is not None and recall >= 0.8,
            "value": _rounded(recall),
            "threshold": ">=0.8",
        },
        "reasoning_step_order_consistency": {
            "pass": order_rate is not None and order_rate >= 0.8,
            "value": _rounded(order_rate),
            "threshold": ">=0.8",
        },
        "multi_hop_bridge_coverage": {
            "pass": bridge_coverage is not None and bridge_coverage >= 0.8,
            "value": _rounded(bridge_coverage),
            "threshold": ">=0.8; interior steps only",
        },
        "valid_temporal_spans": {
            "pass": temporal_valid,
            "value": temporal_valid,
            "threshold": True,
        },
    }
    return {
        "example_id": item.example_id,
        "video_id": item.video_id,
        "question_id": (item.question or {}).get("question_id"),
        "question": (item.question or {}).get("question_text"),
        "reasoning_type": (item.metadata or {}).get("reasoning_type"),
        "graph_source": source,
        "l1_reliability": l1_reliability,
        "graph_id": graph.get("graph_id") or graph.get("index_id"),
        "l1_node_count": len(nodes),
        "timestamped_reasoning_step_count": len(targets),
        "covered_reasoning_step_count": covered_count,
        "reasoning_step_span_coverage": _rounded(recall),
        "order_consistent_pair_count": order_hits,
        "order_comparable_pair_count": order_pairs,
        "reasoning_step_order_consistency": _rounded(order_rate),
        "bridge_step_count": len(bridge_rows),
        "covered_bridge_step_count": bridge_hits,
        "multi_hop_bridge_coverage": _rounded(bridge_coverage),
        "per_reasoning_step": matches,
        "gates": gates,
        "overall_gate": all(gate["pass"] for gate in gates.values()),
    }


def _rounded(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _aggregate(examples: list[dict[str, Any]]) -> dict[str, Any]:
    target_total = sum(row["timestamped_reasoning_step_count"] for row in examples)
    target_hits = sum(row["covered_reasoning_step_count"] for row in examples)
    order_pairs = sum(row["order_comparable_pair_count"] for row in examples)
    order_hits = sum(row["order_consistent_pair_count"] for row in examples)
    bridge_total = sum(row["bridge_step_count"] for row in examples)
    bridge_hits = sum(row["covered_bridge_step_count"] for row in examples)
    coverage = target_hits / target_total if target_total else None
    order = order_hits / order_pairs if order_pairs else None
    bridge = bridge_hits / bridge_total if bridge_total else None
    gates = {
        "reasoning_step_span_coverage": {
            "pass": coverage is not None and coverage >= 0.8,
            "value": _rounded(coverage),
            "threshold": ">=0.8",
        },
        "reasoning_step_order_consistency": {
            "pass": order is not None and order >= 0.8,
            "value": _rounded(order),
            "threshold": ">=0.8",
        },
        "multi_hop_bridge_coverage": {
            "pass": bridge is not None and bridge >= 0.8,
            "value": _rounded(bridge),
            "threshold": ">=0.8; interior steps only",
        },
    }
    return {
        "example_count": len(examples),
        "timestamped_reasoning_step_count": target_total,
        "reasoning_step_span_coverage": _rounded(coverage),
        "reasoning_step_order_consistency": _rounded(order),
        "multi_hop_bridge_coverage": _rounded(bridge),
        "example_gate_pass_count": sum(bool(row["overall_gate"]) for row in examples),
        "gates": gates,
        "overall_gate": bool(examples) and all(gate["pass"] for gate in gates.values()),
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "locked" and args.mode != "video_only":
        raise SystemExit("error: locked VRBench runs require --mode video_only")
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    _add_video_skills_import(Path(args.video_skills_root))
    from dataset_clip_wrapper.adapters.vrbench import VRBenchAdapter

    selected_reasoning_types = args.reasoning_types or list(TARGET_REASONING_TYPES)
    effective_limit = args.limit if args.limit is not None else (20 if args.phase == "smoke" else 100)
    items = _load_items(
        VRBenchAdapter,
        dataset_root,
        reasoning_types=selected_reasoning_types,
        limit=effective_limit,
    )
    if not items:
        raise ValueError("no VRBench examples matched --reasoning-type/--limit")

    canonical_store = CanonicalStore(Path(args.canonical_dir) if args.canonical_dir else None)
    resolved: list[tuple[Any, tuple[Path, dict[str, Any]] | None]] = [
        (item, canonical_store.get(item)) for item in items
    ]
    missing = [str(item.example_id) for item, canonical in resolved if canonical is None]
    if args.mode == "video_only" and missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise SystemExit(
            "error: video_only validation requires pre-generated perception canonical JSON for every "
            f"selected example; missing {len(missing)}: {preview}{suffix}. Run Video_Skills "
            "video-only perception/graph generation first, then pass --canonical-dir."
        )

    examples = []
    canonical_paths: dict[str, str] = {}
    for item, canonical_entry in resolved:
        if canonical_entry is not None:
            path, canonical = canonical_entry
            graph, nodes = _canonical_discovered_nodes(canonical)
            source = "canonical_discovered_l1"
            canonical_paths[str(item.example_id)] = str(path)
            _, adapted_nodes = canonical_to_memory_nodes(canonical)
            observation_end = graph.get("observation_end_s")
            l1_report = audit_l1_nodes(
                adapted_nodes,
                observation_end_s=(
                    float(observation_end) if observation_end is not None else None
                ),
            ).to_dict()
        else:
            targets = _reasoning_targets(item)
            graph, nodes = _provisional_annotation_graph(item, targets)
            source = "expert_demo_provisional_annotation_temporal_graph"
            l1_report = None
        examples.append(
            _evaluate(
                item,
                graph,
                nodes,
                source,
                l1_reliability=l1_report,
            )
        )

    aggregate = _aggregate(examples)
    aggregate["trusted_gate"] = (
        aggregate["overall_gate"] if args.mode == "video_only" else False
    )
    aggregate["gate_interpretation"] = (
        "video_only_transfer"
        if args.mode == "video_only"
        else "provisional expert-text wiring only"
    )
    report = {
        "schema_version": "steam-vrbench-temporal-validation/v0.1",
        "dataset": "vrbench",
        "mode": args.mode,
        "phase": args.phase,
        "locked_run": args.phase == "locked",
        "limit": effective_limit,
        "reasoning_types": selected_reasoning_types,
        "conclusion_scope": (
            "provisional_structure_debug"
            if args.mode == "expert_demo"
            else "video_only_temporal_and_multihop_transfer"
        ),
        "semantics": {
            "reasoning_process": "hidden timestamp supervision used only for evaluation",
            "causal_gold": False,
            "bridge": "covered interior reasoning step in a chain of at least three timestamped steps",
            "canonical_node_policy": (
                "timestamped observation/event/dialogue_span nodes excluding hidden or provided supervision"
            ),
        },
        "canonical_dir": str(canonical_store.root) if canonical_store.root else None,
        "canonical_paths": canonical_paths,
        "summary": aggregate,
        "examples": examples,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
