#!/usr/bin/env python3
"""Offline smoke test for VLM-first L1 graph composition using a fake client."""

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


class FakeVlmClient:
    def chat_json(self, messages: list[dict[str, Any]], *, response_format: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "nodes": [
                {
                    "node_id": "n_fence_early",
                    "node_type": "observation",
                    "clip_id": "clip_a",
                    "time_span": {"start_s": 10.0, "end_s": 14.0},
                    "text": "A man leaves a place with a distinctive metal fence.",
                    "modality": "visual",
                    "confidence": 0.8,
                },
                {
                    "node_id": "n_fence_later",
                    "node_type": "observation",
                    "clip_id": "clip_b",
                    "time_span": {"start_s": 80.0, "end_s": 84.0},
                    "text": "A similar metal fence appears again later.",
                    "modality": "visual",
                    "confidence": 0.8,
                },
            ],
            "edges": [
                {
                    "src": "n_fence_early",
                    "dst": "n_fence_later",
                    "edge_type": "reappears",
                    "evidence_refs": ["n_fence_early", "n_fence_later"],
                    "text": "The later fence visually resembles the earlier fence.",
                }
            ],
            "notes": "fake VLM L1 graph for smoke testing",
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
            "scene_description": "A man leaves a place with a metal fence.",
            "observable_facts": [],
        },
        {
            "clip_id": "clip_b",
            "time_span": clips[1]["source_span"],
            "scene_description": "A similar metal fence appears later.",
            "observable_facts": [],
        },
    ]

    composer = GraphComposer(
        GraphComposerConfig(composer_mode="vlm_l1", use_llm_planner=True),
        FakeVlmClient(),  # type: ignore[arg-type]
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
    report = {
        "composer_mode": composed.get("composer_mode"),
        "used_deterministic_fallback": composed.get("used_deterministic_fallback"),
        "nodes": len(composed["graph"].get("nodes", [])),
        "edges": len(composed["graph"].get("edges", [])),
        "has_reappears": "reappears" in edge_types,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["composer_mode"] == "vlm_l1" and report["has_reappears"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
