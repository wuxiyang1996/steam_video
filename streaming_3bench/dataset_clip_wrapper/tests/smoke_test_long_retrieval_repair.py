#!/usr/bin/env python3
"""Offline smoke test for multi-step long-video retrieval repair."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.run_repair_protocol import _select_rerouted_repair_spans


def main() -> int:
    example = {
        "example_id": "synthetic:long:vehicle",
        "dataset": "cg_bench",
        "video": {"video_id": "synthetic", "duration_s": 180.0},
        "question": {
            "question_text": "What color is the vehicle driving from the left in the animation?",
            "options": [
                {"label": "A", "text": "Gray"},
                {"label": "B", "text": "White"},
            ],
        },
        "metadata": {
            "clip_schemas": [],
            "coarse_clip_schemas": [
                {
                    "time_span": {"start_s": 0.0, "end_s": 30.0},
                    "scene_description": "A kitchen scene with people talking.",
                    "observable_facts": [],
                },
                {
                    "time_span": {"start_s": 28.0, "end_s": 58.0},
                    "scene_description": "A live-action room with no vehicle and no animation visible.",
                    "observable_facts": [],
                },
                {
                    "time_span": {"start_s": 56.0, "end_s": 86.0},
                    "scene_description": "A hallway scene without the target event.",
                    "observable_facts": [],
                },
                {
                    "time_span": {"start_s": 84.0, "end_s": 114.0},
                    "scene_description": "A simple animation shows a white vehicle driving from the left.",
                    "observable_facts": [
                        {"modality": "visual", "text": "The white vehicle moves from the left side."}
                    ],
                },
            ],
        },
    }
    row = {
        "repair_hints": {
            "missing_requirements": [],
            "commonsense_repair": {"missing_requirements": ["discriminative_visual_evidence"]},
        }
    }
    prior_schemas = [
        {
            "time_span": {"start_s": 30.0, "end_s": 38.0},
            "scene_description": "The frames show no vehicle and no animation.",
        }
    ]
    spans, meta = _select_rerouted_repair_spans(
        example,
        row,
        gaps=["discriminative_visual_evidence"],
        clue_spec={
            "planner_backend": "smoke",
            "visual_target": "white vehicle driving from the left in animation",
            "coarse_search_queries": [
                {"role": "target_retrieval", "query": "white vehicle driving from left animation"}
            ],
        },
        prior_schemas=prior_schemas,
        max_repair_clips=6,
        reroute_topk=3,
        reroute_topk_per_query=2,
        api_key=None,
        args=SimpleNamespace(
            dry_run=True,
            disable_llm_reroute_selector=True,
            allow_lexical_fallback=True,
            clue_planner_model="openai/gpt-oss-120b",
            clue_selector_max_tokens=1600,
            clue_planner_timeout_s=180,
            coarse_summary_prompt_chars=260,
            reroute_topk=3,
        ),
    )
    report = {
        "mode": meta.get("mode"),
        "negative_coarse_indices": meta.get("negative_coarse_indices"),
        "selected_coarse_indices": meta.get("selected_coarse_indices"),
        "retrieval_round_count": len(meta.get("retrieval_rounds") or []),
        "span_parent_indices": [span.parent_index for span in spans],
        "passed": (
            meta.get("mode") == "reroute"
            and 1 in (meta.get("negative_coarse_indices") or [])
            and 1 not in (meta.get("selected_coarse_indices") or [])
            and 3 in (meta.get("selected_coarse_indices") or [])
            and bool(spans)
        ),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
