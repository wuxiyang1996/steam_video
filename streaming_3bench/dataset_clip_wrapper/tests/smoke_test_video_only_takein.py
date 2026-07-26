#!/usr/bin/env python3
"""Offline video-only take-in smoke test for all supported video datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.runners.llm_pipeline import build_llm_enriched_example
from dataset_clip_wrapper.schemas import (
    BackboneConfig,
    ClipSchemaConfig,
    GraphComposerConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)

HIDDEN_SOURCE_TYPES = {
    "segment_description",
    "inference_shot",
    "key_relationship",
    "clue_interval",
    "clue_clip",
    "reasoning_process_step",
    "video_summary",
    "qa_answer",
}

DATASET_REGIMES = {
    "video_holmes": VideoRegime.SHORT,
    "cg_bench": VideoRegime.LONG,
    "vrbench": VideoRegime.LONG,
    "siv_bench": VideoRegime.SHORT,
    "ovo_bench": VideoRegime.STREAMING,
    "videomme": VideoRegime.SHORT,
}


def _runtime_evidence_nodes(clue_graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        node
        for node in clue_graph.get("nodes", [])
        if node.get("node_type") in {"observation", "event", "dialogue_span"}
    ]


def _check_dataset(dataset: str, dataset_root: str) -> dict[str, Any]:
    config = WrapperConfig(
        dataset_root=dataset_root,
        dataset=dataset,  # type: ignore[arg-type]
        regime=DATASET_REGIMES[dataset],
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
        backbone=BackboneConfig(name="annotation_only"),
        clip_schema=ClipSchemaConfig(
            backend="video_tools",
            max_clips=1,
            request_frames=3,
        ),
        graph_composer=GraphComposerConfig(use_llm_planner=False),
        run_clip_schema=True,
        run_graph_compose=True,
    )
    adapter = get_adapter(dataset, Path(dataset_root), split="train")
    item = next(adapter.iter_items(limit=1))
    example = build_llm_enriched_example(item, config=config)
    video = example.get("video", {})
    metadata = example.get("metadata", {})
    clip_schemas = metadata.get("clip_schemas") or []
    clue_graph = metadata.get("clue_memory_graph") or {}
    evidence_index = example.get("evidence_index") or {}

    errors: list[str] = []
    primary_path = video.get("primary_path")
    if not primary_path or not Path(primary_path).exists():
        errors.append("missing readable primary video path")
    if float(video.get("duration_s") or 0.0) <= 0.0:
        errors.append("missing positive duration_s")
    if (example.get("available_inputs") or {}).get("mode") != RuntimeMode.VIDEO_ONLY.value:
        errors.append("example is not video_only")
    if video.get("segments"):
        errors.append("video_only exposed dataset annotation segments")
    if not clip_schemas:
        errors.append("no clip schemas produced")
    for schema in clip_schemas:
        if schema.get("producer") != "video_tool_perception_backend":
            errors.append(f"unexpected clip-schema producer: {schema.get('producer')}")
        if schema.get("tool_error"):
            errors.append(f"clip schema tool_error: {schema.get('tool_error')}")
        if int(schema.get("sampled_frame_count") or 0) <= 0:
            errors.append("clip schema sampled no frames")
        if not schema.get("observable_facts"):
            errors.append("clip schema has no observable facts")
    if not evidence_index.get("nodes"):
        errors.append("evidence_index has no nodes")
    if not clue_graph.get("nodes"):
        errors.append("clue_memory_graph has no nodes")
    runtime_nodes = _runtime_evidence_nodes(clue_graph)
    if not runtime_nodes:
        errors.append("clue_memory_graph has no runtime evidence nodes")
    leaked_nodes = [
        node.get("source_type")
        for node in clue_graph.get("nodes", [])
        if node.get("source_type") in HIDDEN_SOURCE_TYPES
    ]
    leaked_candidates = [
        evidence.get("source_type")
        for evidence in example.get("evidence_candidates", [])
        if evidence.get("source_type") in HIDDEN_SOURCE_TYPES
    ]
    if leaked_nodes:
        errors.append(f"hidden source nodes leaked: {sorted(set(leaked_nodes))}")
    if leaked_candidates:
        errors.append(f"hidden evidence candidates leaked: {sorted(set(leaked_candidates))}")

    return {
        "dataset": dataset,
        "example_id": example.get("example_id"),
        "video_path": primary_path,
        "duration_s": video.get("duration_s"),
        "clip_schema_count": len(clip_schemas),
        "sampled_frame_counts": [schema.get("sampled_frame_count", 0) for schema in clip_schemas],
        "evidence_index_nodes": len(evidence_index.get("nodes", [])),
        "evidence_index_edges": len(evidence_index.get("edges", [])),
        "clue_nodes": len(clue_graph.get("nodes", [])),
        "clue_edges": len(clue_graph.get("edges", [])),
        "runtime_evidence_nodes": len(runtime_nodes),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    report = [_check_dataset(dataset, dataset_root) for dataset in DATASET_REGIMES]
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
