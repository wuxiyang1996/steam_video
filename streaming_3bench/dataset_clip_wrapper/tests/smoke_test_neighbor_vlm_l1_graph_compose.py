#!/usr/bin/env python3
"""Offline smoke test for target-clip + neighbor VLM L1 graph composition."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.l1_clue_graph.graph_composer import GraphComposer
from dataset_clip_wrapper.pipeline import build_canonical_example
from dataset_clip_wrapper.schemas import GraphComposerConfig, RuntimeMode, VideoRegime, WrapperConfig


class FakeNeighborVlmClient:
    def chat_json(self, messages: list[dict[str, Any]], *, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = json.loads(messages[-1]["content"])
        target = payload["target_clip"]
        target_clip_id = target["clip_id"]
        neighbor_ids = [clip["clip_id"] for clip in payload.get("neighbor_clips") or []]
        edges = []
        if target_clip_id == "clip_b" and "clip_a" in neighbor_ids:
            edges.append(
                {
                    "src_clip_id": "clip_a",
                    "dst_clip_id": "clip_b",
                    "edge_type": "reappears",
                    "text": "Both clips show a similar metal fence.",
                    "confidence": 0.82,
                }
            )
        target_nodes = [] if target_clip_id == "clip_a" else [
            {
                "node_id": "target_fence",
                "node_type": "observation",
                "text": f"{target_clip_id} shows a metal fence clue.",
                "modality": "visual",
                "confidence": 0.8,
            }
        ]
        return {
            "target_nodes": target_nodes,
            "neighbor_edges": edges,
            "notes": "fake local neighbor graph",
        }


def main() -> int:
    config = WrapperConfig(
        dataset_root="/mnt/is_data/xwu/video_skills/data/datasets",
        dataset="video_holmes",
        regime=VideoRegime.SHORT,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
    )
    adapter = get_adapter("video_holmes", Path(config.dataset_root), split="train")
    raw = next(adapter.iter_items(limit=1))
    example = build_canonical_example(raw, config=config)
    clips = example["video"]["derived_clips"][:2]
    fake_schemas = [
        {
            "clip_id": "clip_a",
            "time_span": clips[0]["source_span"],
            "scene_description": "A dark metal fence appears.",
            "observable_facts": [{"text": "A metal fence is visible.", "modality": "visual"}],
            "salient_objects": [{"surface_form": "metal fence", "attributes": ["dark"]}],
        },
        {
            "clip_id": "clip_b",
            "time_span": clips[1]["source_span"],
            "scene_description": "A similar metal fence appears again.",
            "observable_facts": [{"text": "A similar metal fence is visible.", "modality": "visual"}],
            "salient_objects": [{"surface_form": "metal fence", "attributes": ["similar"]}],
        },
    ]

    composer = GraphComposer(
        GraphComposerConfig(composer_mode="neighbor_vlm_l1", use_llm_planner=True),
        FakeNeighborVlmClient(),  # type: ignore[arg-type]
    )
    composed = composer.compose_from_clip_schemas(
        example_id=example["example_id"],
        video_id=raw.video_id,
        clip_policy=example["evidence_index"]["clip_policy"],
        clip_schemas=fake_schemas,
        segments=example["video"]["segments"],
        mode=config.mode,
        duration_s=float(example["video"]["duration_s"]),
        observation_end_s=example["evidence_index"]["clip_policy"].get("observation_end_s"),
    )
    edge_types = {edge.get("edge_type") for edge in composed["graph"].get("edges", [])}
    producers = {node.get("producer") for node in composed["graph"].get("nodes", [])}
    report = {
        "composer_mode": composed.get("composer_mode"),
        "used_deterministic_fallback": composed.get("used_deterministic_fallback"),
        "nodes": len(composed["graph"].get("nodes", [])),
        "edges": len(composed["graph"].get("edges", [])),
        "has_neighbor_nodes": "neighbor_vlm_l1_graph_composer" in producers,
        "has_schema_anchor": "neighbor_vlm_l1_schema_anchor" in producers,
        "has_reappears": "reappears" in edge_types,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["composer_mode"] == "neighbor_vlm_l1" and report["has_reappears"] and report["has_schema_anchor"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
