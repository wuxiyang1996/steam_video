#!/usr/bin/env python3
"""Offline smoke test for pipeline-emitted coarse/fine video-reference graphs."""

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


CASES = {
    "video_holmes": VideoRegime.SHORT,
    "cg_bench": VideoRegime.LONG,
    "vrbench": VideoRegime.LONG,
    "siv_bench": VideoRegime.SHORT,
    "ovo_bench": VideoRegime.STREAMING,
    "videomme": VideoRegime.SHORT,
}


def _has_no_gaps(nodes: list[dict[str, Any]]) -> bool:
    spans = [node.get("time_span") for node in nodes if node.get("time_span")]
    spans = sorted(spans, key=lambda span: span["start_s"])
    if not spans:
        return False
    if abs(float(spans[0]["start_s"])) > 1e-6:
        return False
    for left, right in zip(spans, spans[1:]):
        if float(right["start_s"]) > float(left["end_s"]) + 1e-6:
            return False
    return True


def _check_dataset(dataset: str, regime: VideoRegime) -> dict[str, Any]:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    adapter = get_adapter(dataset, Path(dataset_root), split="train")
    item = next(adapter.iter_items(limit=1))
    config = WrapperConfig(
        dataset_root=dataset_root,
        dataset=dataset,  # type: ignore[arg-type]
        regime=regime,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
        backbone=BackboneConfig(name="annotation_only"),
        clip_schema=ClipSchemaConfig(backend="video_tools", max_clips=2, request_frames=2),
        graph_composer=GraphComposerConfig(use_llm_planner=False),
        run_clip_schema=True,
        run_graph_compose=False,
    )
    example = build_llm_enriched_example(item, config=config)
    graph = example.get("metadata", {}).get("coarse_fine_graph") or {}
    coarse = graph.get("coarse_graph") or {}
    fine = graph.get("fine_graph") or {}
    links = graph.get("coarse_to_fine_links") or []
    coarse_nodes = [node for node in coarse.get("nodes", []) if node.get("node_type") == "clip"]
    fine_clip_nodes = [node for node in fine.get("nodes", []) if node.get("node_type") == "clip"]
    fine_obs_nodes = [node for node in fine.get("nodes", []) if node.get("node_type") == "observation"]

    errors: list[str] = []
    if not graph:
        errors.append("missing metadata.coarse_fine_graph")
    if graph.get("purpose") != "video_clip_reference_layer":
        errors.append("unexpected graph purpose")
    if not fine_clip_nodes:
        errors.append("missing fine clip references")
    if not fine_obs_nodes:
        errors.append("missing fine observation references from clip schemas")
    if regime == VideoRegime.LONG:
        if not coarse_nodes:
            errors.append("long video missing full coarse graph")
        if coarse.get("coverage") != "full_video":
            errors.append("long video coarse coverage is not full_video")
        if fine.get("coverage") != "retrieved_neighborhoods":
            errors.append("long video fine coverage is not retrieved_neighborhoods")
        if not links:
            errors.append("long video missing fine-to-coarse refines links")
        if not _has_no_gaps(coarse_nodes):
            errors.append("coarse graph has temporal gaps")
    elif regime == VideoRegime.STREAMING:
        if coarse_nodes:
            errors.append("streaming fixed-window graph should not require coarse graph")
        if fine.get("coverage") != "full_video":
            errors.append("streaming fine coverage is not full_video")
        if not _has_no_gaps(fine_clip_nodes):
            errors.append("streaming fine graph has temporal gaps")
    else:
        if coarse_nodes:
            errors.append("short video should not require coarse graph")
        if fine.get("coverage") != "full_video":
            errors.append("short video fine coverage is not full_video")
        if not _has_no_gaps(fine_clip_nodes):
            errors.append("short-video fine graph has temporal gaps")

    return {
        "dataset": dataset,
        "example_id": example.get("example_id"),
        "duration_s": example.get("video", {}).get("duration_s"),
        "strategy": graph.get("strategy"),
        "coarse_coverage": coarse.get("coverage"),
        "fine_coverage": fine.get("coverage"),
        "coarse_clip_nodes": len(coarse_nodes),
        "fine_clip_nodes": len(fine_clip_nodes),
        "fine_observation_nodes": len(fine_obs_nodes),
        "selected_coarse_indices": graph.get("selected_coarse_indices", []),
        "coarse_to_fine_links": len(links),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    report = [_check_dataset(dataset, regime) for dataset, regime in CASES.items()]
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
