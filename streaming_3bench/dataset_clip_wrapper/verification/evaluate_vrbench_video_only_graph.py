#!/usr/bin/env python3
"""Evaluate VRBench video-only graph coverage against hidden timestamp targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.cluster_paths import DEFAULT_DATASET_ROOT, DEFAULT_KEYS_PY, output_path
from dataset_clip_wrapper.runners.llm_pipeline import build_llm_enriched_example
from dataset_clip_wrapper.schemas import (
    BackboneConfig,
    ClipRetrievalConfig,
    ClipSchemaConfig,
    GraphComposerConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)


def _overlap_s(left: dict[str, Any] | None, right: dict[str, Any] | None) -> float:
    if not left or not right:
        return 0.0
    start = max(float(left.get("start_s", 0.0)), float(right.get("start_s", 0.0)))
    end = min(float(left.get("end_s", 0.0)), float(right.get("end_s", 0.0)))
    return max(0.0, end - start)


def _span_duration_s(span: dict[str, Any] | None) -> float:
    if not span:
        return 0.0
    return max(0.0, float(span.get("end_s", 0.0)) - float(span.get("start_s", 0.0)))


def _target_steps(item) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for segment in item.annotation_segments:
        if segment.get("source_type") != "reasoning_process_step":
            continue
        span = segment.get("time_span")
        if not span:
            continue
        targets.append(
            {
                "target_id": segment.get("segment_id"),
                "time_span": span,
                "text": segment.get("text"),
            }
        )
    return targets


def _clip_schema_spans(example: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        schema["time_span"]
        for schema in example.get("metadata", {}).get("clip_schemas", [])
        if schema.get("time_span")
    ]


def _discovered_evidence_nodes(example: dict[str, Any]) -> list[dict[str, Any]]:
    graph = example.get("metadata", {}).get("clue_memory_graph", {})
    nodes = []
    for node in graph.get("nodes", []):
        if node.get("node_type") not in {"observation", "event", "dialogue_span"}:
            continue
        if not node.get("time_span"):
            continue
        nodes.append(node)
    return nodes


def _coverage(target: dict[str, Any], spans: list[dict[str, Any]], *, min_overlap_s: float) -> dict[str, Any]:
    overlaps = [_overlap_s(target.get("time_span"), span) for span in spans]
    best = max(overlaps, default=0.0)
    duration = _span_duration_s(target.get("time_span"))
    return {
        "covered": best >= min_overlap_s,
        "best_overlap_s": round(best, 3),
        "target_duration_s": round(duration, 3),
        "overlap_ratio": round(best / duration, 3) if duration > 0 else None,
    }


def _evaluate_item(item, config: WrapperConfig, *, min_overlap_s: float) -> dict[str, Any]:
    example = build_llm_enriched_example(item, config=config)
    targets = _target_steps(item)
    schema_spans = _clip_schema_spans(example)
    discovered_nodes = _discovered_evidence_nodes(example)
    discovered_spans = [node["time_span"] for node in discovered_nodes]

    per_target = []
    for target in targets:
        clip_cov = _coverage(target, schema_spans, min_overlap_s=min_overlap_s)
        evidence_cov = _coverage(target, discovered_spans, min_overlap_s=min_overlap_s)
        per_target.append(
            {
                **target,
                "clip_schema_coverage": clip_cov,
                "discovered_evidence_coverage": evidence_cov,
            }
        )

    target_count = len(targets)
    clip_hits = sum(1 for row in per_target if row["clip_schema_coverage"]["covered"])
    evidence_hits = sum(1 for row in per_target if row["discovered_evidence_coverage"]["covered"])
    clue_graph = example.get("metadata", {}).get("clue_memory_graph", {})
    return {
        "example_id": item.example_id,
        "video_id": item.video_id,
        "video_path": str(item.video_path) if item.video_path else None,
        "duration_s": example.get("video", {}).get("duration_s"),
        "question": item.question.get("question_text"),
        "target_step_count": target_count,
        "clip_schema_count": len(schema_spans),
        "discovered_evidence_node_count": len(discovered_nodes),
        "clue_node_count": len(clue_graph.get("nodes", [])),
        "clue_edge_count": len(clue_graph.get("edges", [])),
        "clip_schema_target_recall": round(clip_hits / target_count, 3) if target_count else None,
        "discovered_evidence_target_recall": round(evidence_hits / target_count, 3) if target_count else None,
        "perception": example.get("metadata", {}).get("perception", {}),
        "per_target": per_target,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure VRBench video-only graph coverage against hidden reasoning_process timestamps."
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--output", default=str(output_path("vrbench_video_only_graph_quality.json")))
    parser.add_argument("--keys-py", default=DEFAULT_KEYS_PY)
    parser.add_argument("--clip-schema-backend", default="video_tools", choices=["video_tools", "qwen"])
    parser.add_argument("--clip-schema-max-clips", type=int, default=8)
    parser.add_argument("--clip-schema-frames", type=int, default=4)
    parser.add_argument("--retrieval-topk", type=int, default=4)
    parser.add_argument("--min-overlap-s", type=float, default=1.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    adapter = get_adapter("vrbench", Path(args.dataset_root), split=args.split)
    config = WrapperConfig(
        dataset_root=args.dataset_root,
        dataset="vrbench",
        regime=VideoRegime.LONG,
        mode=RuntimeMode.VIDEO_ONLY,
        split=args.split,
        limit=args.limit,
        backbone=BackboneConfig(keys_py_path=args.keys_py),
        retrieval=ClipRetrievalConfig(enabled=True, topk=args.retrieval_topk, mode="sequential"),
        clip_schema=ClipSchemaConfig(
            backend=args.clip_schema_backend,  # type: ignore[arg-type]
            keys_py_path=args.keys_py,
            max_clips=args.clip_schema_max_clips,
            request_frames=args.clip_schema_frames,
        ),
        graph_composer=GraphComposerConfig(
            keys_py_path=args.keys_py,
            use_llm_planner=False,
        ),
        run_clip_schema=True,
        run_graph_compose=True,
    )

    examples = []
    for item in adapter.iter_items(limit=args.limit):
        examples.append(_evaluate_item(item, config, min_overlap_s=args.min_overlap_s))

    total_targets = sum(row["target_step_count"] for row in examples)
    total_clip_hits = sum(
        1
        for row in examples
        for target in row["per_target"]
        if target["clip_schema_coverage"]["covered"]
    )
    total_evidence_hits = sum(
        1
        for row in examples
        for target in row["per_target"]
        if target["discovered_evidence_coverage"]["covered"]
    )
    report = {
        "dataset": "vrbench",
        "mode": "video_only",
        "metric": "hidden_reasoning_process_timestamp_coverage",
        "clip_schema_backend": args.clip_schema_backend,
        "clip_schema_max_clips": args.clip_schema_max_clips,
        "retrieval_topk": args.retrieval_topk,
        "min_overlap_s": args.min_overlap_s,
        "example_count": len(examples),
        "target_step_count": total_targets,
        "clip_schema_target_recall": round(total_clip_hits / total_targets, 3) if total_targets else None,
        "discovered_evidence_target_recall": round(total_evidence_hits / total_targets, 3) if total_targets else None,
        "examples": examples,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "examples"}, indent=2))
    print(f"output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
