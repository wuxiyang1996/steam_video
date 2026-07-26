#!/usr/bin/env python3
"""Export video-only L1/L2/repair trajectories as expert-demo candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


GOLD_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "gold_label",
    "gold_eval_only",
    "correct",
    "correct_answer",
    "correct_eval_only",
    "official_answer",
}
ACCEPTED = {"accepted_strong", "accepted_bridge"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_source_example(report: dict[str, Any]) -> dict[str, Any]:
    source = report.get("source_path")
    example_id = str(report.get("example_id") or "")
    if not source or not example_id:
        return {}
    for row in _read_jsonl(Path(str(source))):
        if str(row.get("example_id") or "") == example_id:
            return row
    return {}


def _drop_gold_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            key: _drop_gold_keys(value)
            for key, value in payload.items()
            if str(key) not in GOLD_KEYS
        }
    if isinstance(payload, list):
        return [_drop_gold_keys(item) for item in payload]
    return payload


def _contains_gold_key(payload: Any) -> bool:
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) in GOLD_KEYS:
                return True
            if _contains_gold_key(value):
                return True
    elif isinstance(payload, list):
        return any(_contains_gold_key(item) for item in payload)
    return False


def _node_text(node: dict[str, Any]) -> str:
    return str(node.get("text") or node.get("summary") or node.get("label") or node.get("description") or "").strip()


def _node_lookup(graph: dict[str, Any], repair_subgraph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and node.get("node_id"):
            lookup[str(node["node_id"])] = node
    for node in repair_subgraph.get("nodes") or []:
        if isinstance(node, dict) and node.get("node_id"):
            lookup.setdefault(str(node["node_id"]), node)
    return lookup


def _collect_demo_refs(report: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    l2 = report.get("L2_status") if isinstance(report.get("L2_status"), dict) else {}
    refs.extend(str(ref) for ref in l2.get("support_refs") or [] if ref)
    trajectory = report.get("l2_trajectory") if isinstance(report.get("l2_trajectory"), dict) else {}
    for round_row in trajectory.get("rounds") or []:
        if not isinstance(round_row, dict):
            continue
        verifier = round_row.get("verifier_signal") if isinstance(round_row.get("verifier_signal"), dict) else {}
        pack = verifier.get("verified_evidence_pack") if isinstance(verifier.get("verified_evidence_pack"), dict) else {}
        refs.extend(str(ref) for ref in pack.get("support_refs") or [] if ref)
        obs = round_row.get("observation_summary") if isinstance(round_row.get("observation_summary"), dict) else {}
        refs.extend(str(ref) for ref in obs.get("support_refs") or [] if ref)
    repair = report.get("repair_report") if isinstance(report.get("repair_report"), dict) else {}
    selector = repair.get("option_evidence_selector") if isinstance(repair.get("option_evidence_selector"), dict) else {}
    for pack in selector.get("option_packs") or []:
        if not isinstance(pack, dict):
            continue
        refs.extend(str(ref) for ref in pack.get("positive_refs") or [] if ref)
        refs.extend(str(ref) for ref in pack.get("negative_refs") or [] if ref)
    repair_subgraph = report.get("repair_subgraph") if isinstance(report.get("repair_subgraph"), dict) else {}
    for node in repair_subgraph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("node_id"):
            refs.append(str(node["node_id"]))
        for pack in node.get("option_evidence_packs") or []:
            if isinstance(pack, dict):
                refs.extend(str(ref) for ref in pack.get("positive_refs") or [] if ref)
                refs.extend(str(ref) for ref in pack.get("negative_refs") or [] if ref)
    return list(dict.fromkeys(refs))


def _hidden_supervision_stub(example: dict[str, Any]) -> dict[str, Any]:
    hidden = example.get("hidden_supervision") if isinstance(example.get("hidden_supervision"), dict) else {}
    return {
        "available_for_training": bool(hidden.get("available_for_training", True)),
        "available_for_inference": False,
        "sources": list(hidden.get("sources") or []),
        "note": "Hidden supervision is recorded for split/eval bookkeeping only and is not included in visible_demo_inputs.",
    }


def _demo_type(report: dict[str, Any]) -> str:
    status = str(report.get("final_acceptance_status") or "")
    if status == "accepted_strong" and report.get("final_repair_applied"):
        return "repair_strong"
    if status == "accepted_strong":
        return "direct_strong"
    if status == "accepted_bridge":
        return "bridge_verified"
    if status == "needs_more_evidence":
        return "abstain_needs_more_evidence"
    return "rejected_or_invalid"


def _heuristic_acceptance(report: dict[str, Any]) -> bool:
    status = str(report.get("final_acceptance_status") or "")
    if status not in ACCEPTED:
        return False
    if status == "accepted_bridge":
        return False
    if not report.get("final_repair_applied"):
        return False
    selector = ((report.get("repair_report") or {}).get("option_evidence_selector") or {})
    return not bool(selector.get("selector_backend"))


def _support_ref_count(report: dict[str, Any]) -> int:
    l2 = report.get("L2_status") if isinstance(report.get("L2_status"), dict) else {}
    return int(l2.get("support_ref_count") or 0)


def _quality_flags(report: dict[str, Any], *, min_support_refs: int, visible_inputs: dict[str, Any]) -> dict[str, Any]:
    status = str(report.get("final_acceptance_status") or "")
    accepted = status in ACCEPTED
    strict = bool((report.get("strict_vlm_perception") or {}).get("qwen_only"))
    high_l1 = ((report.get("L1_quality") or {}).get("grade")) == "high"
    support_refs = _support_ref_count(report)
    trajectory_complete = bool(report.get("l2_trajectory_complete"))
    repair_complete = (not report.get("final_repair_applied")) or bool(report.get("repair_subgraph_complete"))
    heuristic = _heuristic_acceptance(report)
    no_gold_visible = not _contains_gold_key(visible_inputs)
    training_candidate = (
        accepted
        and strict
        and high_l1
        and trajectory_complete
        and repair_complete
        and not heuristic
        and no_gold_visible
        and (status == "accepted_bridge" or support_refs >= min_support_refs)
    )
    abstain_candidate = (
        status == "needs_more_evidence"
        and strict
        and high_l1
        and trajectory_complete
        and repair_complete
        and not heuristic
        and no_gold_visible
    )
    return {
        "training_candidate": training_candidate,
        "abstain_candidate": abstain_candidate,
        "accepted": accepted,
        "strict_vlm_perception": strict,
        "high_l1": high_l1,
        "l2_trajectory_complete": trajectory_complete,
        "repair_subgraph_complete": repair_complete,
        "heuristic_final_acceptance": heuristic,
        "no_gold_keys_in_visible_inputs": no_gold_visible,
        "support_ref_count": support_refs,
        "min_support_refs": min_support_refs,
    }


def _visible_inputs(example: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    question = _drop_gold_keys(example.get("question") or report.get("question") or {})
    video = example.get("video") or {}
    return {
        "mode": "video_only",
        "visible_to_agent": ["video", "question", "automatic_clips", "automatic_segments", "l1_clue_memory_graph"],
        "video": _drop_gold_keys(video),
        "question": question,
    }


def _compact_l1_evidence(
    graph: dict[str, Any],
    repair_subgraph: dict[str, Any],
    refs: list[str],
    *,
    max_nodes: int,
) -> list[dict[str, Any]]:
    lookup = _node_lookup(graph, repair_subgraph)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(ref: str, role: str) -> None:
        if ref in seen or len(selected) >= max_nodes:
            return
        node = lookup.get(ref)
        if not isinstance(node, dict):
            return
        text = _node_text(node)
        if not text and node.get("node_type") not in {"l2_gap_diagnosis", "repair_plan", "option_verifier"}:
            return
        selected.append(
            {
                "ref": ref,
                "role": role,
                "node_type": node.get("node_type") or node.get("type"),
                "source_type": node.get("source_type"),
                "producer": node.get("producer"),
                "time_span": node.get("time_span") or node.get("source_span") or {},
                "text": text[:600],
            }
        )
        seen.add(ref)

    for ref in refs:
        add_node(ref, "used_by_l2_or_repair")
    if len(selected) < max_nodes:
        priority_types = {"repair_semantic_observation", "observation", "event", "state", "place", "visual_social_cue"}
        for node in graph.get("nodes") or []:
            if not isinstance(node, dict) or node.get("node_type") not in priority_types:
                continue
            node_id = str(node.get("node_id") or "")
            if node_id:
                add_node(node_id, "compact_context")
            if len(selected) >= max_nodes:
                break
    return selected


def _l1_snapshot(
    example: dict[str, Any],
    report: dict[str, Any],
    *,
    include_graph: bool,
    training_view: str,
    max_l1_nodes: int,
) -> dict[str, Any]:
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    repair_subgraph = report.get("repair_subgraph") if isinstance(report.get("repair_subgraph"), dict) else {}
    quality = report.get("L1_quality") or {}
    refs = _collect_demo_refs(report)
    snapshot = {
        "graph_id": graph.get("graph_id") or f"clue_memory:{report.get('example_id')}",
        "training_view": training_view,
        "quality": quality,
        "counts": {
            "nodes": len(graph.get("nodes") or []),
            "edges": len(graph.get("edges") or []),
        },
        "used_ref_count": len(refs),
    }
    if training_view == "compact":
        snapshot["compact_evidence_nodes"] = _compact_l1_evidence(
            graph,
            repair_subgraph,
            refs,
            max_nodes=max_l1_nodes,
        )
        snapshot["compact_policy"] = {
            "max_l1_nodes": max_l1_nodes,
            "selection": "support_refs + repair_refs + semantic/context fill",
        }
    elif include_graph:
        snapshot["graph"] = _drop_gold_keys(graph)
    return snapshot


def _l2_demo(report: dict[str, Any]) -> dict[str, Any]:
    l2_status = _drop_gold_keys(report.get("L2_status") or {})
    return {
        "final_acceptance_status": report.get("final_acceptance_status"),
        "final_repair_applied": bool(report.get("final_repair_applied")),
        "final_repair_needed": bool(report.get("final_repair_needed")),
        "verifier_reason": report.get("verifier_reason"),
        "l2_status": l2_status,
        "trajectory": _drop_gold_keys(report.get("l2_trajectory") or {}),
        "repair_report": _drop_gold_keys(report.get("repair_report") or {}),
        "repair_subgraph": _drop_gold_keys(report.get("repair_subgraph") or {}),
    }


def _build_demo(
    report: dict[str, Any],
    *,
    include_graph: bool,
    min_support_refs: int,
    training_view: str,
    max_l1_nodes: int,
) -> dict[str, Any]:
    example = _load_source_example(report)
    demo_type = _demo_type(report)
    visible_inputs = _visible_inputs(example, report)
    flags = _quality_flags(report, min_support_refs=min_support_refs, visible_inputs=visible_inputs)
    demo_id = f"expert_demo:{report.get('dataset')}:{report.get('example_id')}:{demo_type}"
    return {
        "schema_version": "video-skills-relaunch/expert-demo-export-v0.1",
        "demo_id": demo_id,
        "demo_type": demo_type,
        "dataset": report.get("dataset"),
        "example_id": report.get("example_id"),
        "video_regime": report.get("video_regime"),
        "task_family": report.get("task_family"),
        "source_path": report.get("source_path"),
        "export_policy": {
            "input_mode": "video_only",
            "purpose": "expert_trajectory_gathering",
            "gold_fields_removed_from_visible_inputs": sorted(GOLD_KEYS),
            "forbidden_training_inputs": [
                "official_answer",
                "gold_eval_only",
                "clue_intervals",
                "reasoning_process",
                "hidden_supervision",
            ],
            "include_full_l1_graph": include_graph,
            "training_view": training_view,
            "max_l1_nodes": max_l1_nodes,
        },
        "visible_demo_inputs": visible_inputs,
        "hidden_supervision": _hidden_supervision_stub(example),
        "l1": _l1_snapshot(
            example,
            report,
            include_graph=include_graph,
            training_view=training_view,
            max_l1_nodes=max_l1_nodes,
        ),
        "l2": _l2_demo(report),
        "quality_flags": flags,
    }


def _summarize(demos: list[dict[str, Any]], final_summary: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    by_dataset: dict[str, int] = {}
    final_status: dict[str, int] = {}
    for demo in demos:
        by_type[demo["demo_type"]] = by_type.get(demo["demo_type"], 0) + 1
        by_dataset[str(demo.get("dataset"))] = by_dataset.get(str(demo.get("dataset")), 0) + 1
        status = str((demo.get("l2") or {}).get("final_acceptance_status") or "missing")
        final_status[status] = final_status.get(status, 0) + 1
    return {
        "schema_version": "video-skills-relaunch/expert-demo-quality-v0.1",
        "examples": len(demos),
        "demo_type_counts": by_type,
        "dataset_counts": by_dataset,
        "final_acceptance_status_counts": final_status,
        "training_candidate_count": sum(1 for demo in demos if (demo.get("quality_flags") or {}).get("training_candidate")),
        "abstain_candidate_count": sum(1 for demo in demos if (demo.get("quality_flags") or {}).get("abstain_candidate")),
        "strict_vlm_perception_all": all((demo.get("quality_flags") or {}).get("strict_vlm_perception") for demo in demos),
        "high_l1_all": all((demo.get("quality_flags") or {}).get("high_l1") for demo in demos),
        "heuristic_final_acceptance_count": sum(1 for demo in demos if (demo.get("quality_flags") or {}).get("heuristic_final_acceptance")),
        "visible_gold_key_leak_count": sum(1 for demo in demos if not (demo.get("quality_flags") or {}).get("no_gold_keys_in_visible_inputs")),
        "compact_evidence_node_count": sum(len(((demo.get("l1") or {}).get("compact_evidence_nodes") or [])) for demo in demos),
        "training_views": sorted({str((demo.get("l1") or {}).get("training_view") or "full") for demo in demos}),
        "source_final_report_summary": final_summary,
    }


def build_export(
    final_report: dict[str, Any],
    *,
    include_graph: bool,
    include_abstain: bool,
    min_support_refs: int,
    training_view: str = "full",
    max_l1_nodes: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = final_report.get("reports") or []
    demos = []
    for report in rows:
        status = str(report.get("final_acceptance_status") or "")
        if not include_abstain and status not in ACCEPTED:
            continue
        demos.append(
            _build_demo(
                report,
                include_graph=include_graph,
                min_support_refs=min_support_refs,
                training_view=training_view,
                max_l1_nodes=max_l1_nodes,
            )
        )
    return demos, _summarize(demos, final_report.get("summary") or {})


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export video-only L1/L2/repair trajectories as expert-demo candidates.")
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--quality-report-output", type=Path, required=True)
    parser.add_argument("--no-full-l1-graph", action="store_true", help="Export graph counts/quality only, not full L1 graph payload.")
    parser.add_argument("--training-view", default="full", choices=["full", "compact"])
    parser.add_argument("--max-l1-nodes", type=int, default=80)
    parser.add_argument("--accepted-only", action="store_true", help="Drop needs_more_evidence abstain demos.")
    parser.add_argument("--min-support-refs", type=int, default=2)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    final_report = _read_json(args.final_report)
    demos, quality = build_export(
        final_report,
        include_graph=(not args.no_full_l1_graph and args.training_view == "full"),
        include_abstain=not args.accepted_only,
        min_support_refs=args.min_support_refs,
        training_view=args.training_view,
        max_l1_nodes=args.max_l1_nodes,
    )
    _write_jsonl(args.output_jsonl, demos)
    _write_json(args.quality_report_output, quality)
    print(json.dumps(quality, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
