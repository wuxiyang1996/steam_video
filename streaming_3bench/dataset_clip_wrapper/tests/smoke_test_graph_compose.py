#!/usr/bin/env python3
"""Offline smoke test for deterministic graph composition (no API calls)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.l1_clue_graph.graph_composer import GraphComposer
from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient
from dataset_clip_wrapper.pipeline import build_canonical_example
from dataset_clip_wrapper.schemas import GraphComposerConfig, RuntimeMode, VideoRegime, WrapperConfig


def main() -> int:
    config = WrapperConfig(
        dataset_root="/mnt/is_data/xwu/video_skills/data/datasets",
        dataset="video_holmes",
        regime=VideoRegime.SHORT,
        mode=RuntimeMode.EXPERT_DEMO,
        split="train",
        limit=1,
    )
    adapter = get_adapter("video_holmes", Path(config.dataset_root), split="train")
    raw = next(adapter.iter_items(limit=1))
    example = build_canonical_example(raw, config=config)
    first_clip = example["video"]["derived_clips"][0]
    fake_schemas = [
        {
            "clip_id": first_clip["clip_id"],
            "time_span": first_clip["source_span"],
            "granularity": "fine",
            "scene_description": "A person speaks near a fence.",
            "observable_facts": [{"text": "A person stands by a metal fence.", "modality": "visual"}],
            "dialogue_spans": [],
            "entity_mentions": [{"surface_form": "person", "entity_type": "person"}],
            "events": [
                {
                    "description": "person stands by fence",
                    "time_span": first_clip["source_span"],
                }
            ],
        }
    ]

    composer = GraphComposer(
        GraphComposerConfig(use_llm_planner=False),
        OpenRouterClient(model="offline", api_key="offline"),
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
    report = {
        "nodes": len(composed["graph"]["nodes"]),
        "edges": len(composed["graph"]["edges"]),
        "trace_steps": len(composed["execution_trace"]),
        "passed": len(composed["graph"]["nodes"]) > 0,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
