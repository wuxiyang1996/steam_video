#!/usr/bin/env python3
"""Smoke test for expert-demo export boundaries."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from dataset_clip_wrapper.expert_demos.export_expert_demos import build_export


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_export_expert_demo_boundaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        source = Path(tmp) / "source.jsonl"
        example = {
            "example_id": "demo:1",
            "dataset": "toy",
            "video": {"video_id": "v1", "path": "video.mp4"},
            "question": {
                "question_text": "What color is the cup?",
                "options": [{"label": "A", "text": "red"}],
                "answer": {"label": "A", "text": "red"},
            },
            "hidden_supervision": {
                "available_for_training": True,
                "available_for_inference": False,
                "sources": ["official_answer"],
            },
            "metadata": {
                "clue_memory_graph": {
                    "graph_id": "g1",
                    "nodes": [{"node_id": "n1", "text": "A red cup is visible."}],
                    "edges": [],
                }
            },
        }
        _write_jsonl(source, [example])
        final_report = {
            "summary": {"examples": 1},
            "reports": [
                {
                    "dataset": "toy",
                    "example_id": "demo:1",
                    "source_path": str(source),
                    "video_regime": "short",
                    "task_family": "toy_qa",
                    "L1_quality": {"grade": "high"},
                    "L2_status": {
                        "acceptance_status": "accepted_strong",
                        "final_answer": {"label": "A", "text": "red"},
                        "support_ref_count": 2,
                        "gold_eval_only": {"label": "A"},
                    },
                    "strict_vlm_perception": {"qwen_only": True},
                    "final_acceptance_status": "accepted_strong",
                    "final_repair_applied": False,
                    "final_repair_needed": False,
                    "l2_trajectory": {"rounds": [{"round_type": "initial_l2_reasoning"}]},
                    "l2_trajectory_complete": True,
                    "repair_subgraph_complete": True,
                }
            ],
        }
        demos, quality = build_export(
            final_report,
            include_graph=True,
            include_abstain=True,
            min_support_refs=2,
            training_view="compact",
            max_l1_nodes=8,
        )
    assert quality["training_candidate_count"] == 1
    assert demos[0]["demo_type"] == "direct_strong"
    visible = json.dumps(demos[0]["visible_demo_inputs"])
    l2 = json.dumps(demos[0]["l2"])
    assert "answer" not in demos[0]["visible_demo_inputs"]["question"]
    assert "gold_eval_only" not in l2
    assert "official_answer" in demos[0]["hidden_supervision"]["sources"]
    assert demos[0]["l1"]["training_view"] == "compact"
    assert demos[0]["l1"]["compact_policy"]["max_l1_nodes"] == 8


if __name__ == "__main__":
    test_export_expert_demo_boundaries()
    print("expert demo export smoke test passed")
