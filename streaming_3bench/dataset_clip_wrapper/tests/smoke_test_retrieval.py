#!/usr/bin/env python3
"""Offline tests for M3-style retrieve-gated hierarchical segmentation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.perception.clip_policy import segment_coarse_index, segment_perception_clips, segment_video
from dataset_clip_wrapper.l1_clue_graph.clip_retrieval import retrieve_coarse_clips
from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.runners.llm_pipeline import _resolve_perception_spans
from dataset_clip_wrapper.schemas import (
    ClipPolicyConfig,
    ClipRetrievalConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)


def main() -> int:
    long_policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=2741.0)
    coarse = segment_coarse_index(2741.0, long_policy)
    full = segment_video(2741.0, long_policy, fine_expansion="all")
    retrieval = retrieve_coarse_clips(
        coarse_spans=coarse,
        query_text="Where did the person hide the key before leaving the kitchen?",
        segments=[],
        topk=2,
    )
    perception = segment_perception_clips(
        2741.0,
        long_policy,
        selected_coarse_indices=retrieval["selected_coarse_indices"],
    )
    fine_only = [s for s in perception if s.granularity == "fine"]

    cg_config = WrapperConfig(
        dataset_root="/mnt/is_data/xwu/video_skills/data/datasets",
        dataset="cg_bench",
        regime=VideoRegime.LONG,
        mode=RuntimeMode.EXPERT_DEMO,
        split="train",
        limit=1,
    )
    cg = next(iter_canonical_examples(cg_config))
    duration_s = float(cg["video"]["duration_s"])
    policy = cg_config.resolved_clip_policy(duration_s)
    _, perception_meta = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=policy,
        regime=VideoRegime.LONG,
        retrieval_config=ClipRetrievalConfig(topk=2),
        question_text=cg["question"].get("question_text", ""),
        visible_segments=cg["video"]["segments"],
        mode=RuntimeMode.EXPERT_DEMO,
    )

    stream_config = WrapperConfig(
        dataset_root="/mnt/is_data/xwu/video_skills/data/datasets",
        dataset="cg_bench",
        regime=VideoRegime.STREAMING,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
    )
    stream = next(iter_canonical_examples(stream_config))
    coarse_visual = _coarse_visual_summary_retrieval()

    report = {
        "synthetic_2741s": {
            "coarse_count": len(coarse),
            "full_hierarchical_count": len(full),
            "retrieved_coarse": retrieval["selected_coarse_indices"],
            "gated_fine_count": len(fine_only),
            "passed": len(coarse) < len(full) and len(fine_only) < 50,
        },
        "cg_bench_sample": {
            "index_clip_count": cg["metadata"]["index_clip_count"],
            "coarse_clip_count": cg["metadata"]["coarse_clip_count"],
            "fine_clip_count": cg["metadata"]["fine_clip_count"],
            "perception_fine_count": perception_meta.get("perception_clip_count"),
            "retrieved_coarse": perception_meta.get("retrieval", {}).get("selected_coarse_indices"),
            "passed": cg["metadata"]["fine_clip_count"] == 0 and cg["metadata"]["coarse_clip_count"] > 0,
        },
        "cg_bench_streaming": {
            "clip_count": stream["metadata"]["clip_count"],
            "coarse_clip_count": stream["metadata"]["coarse_clip_count"],
            "fine_clip_count": stream["metadata"]["fine_clip_count"],
            "strategy": stream["evidence_index"]["clip_policy"]["strategy"],
            "passed": stream["metadata"]["fine_clip_count"] == 0 and stream["metadata"]["coarse_clip_count"] > 0 and stream["metadata"]["clip_count"] < 200,
        },
        "coarse_visual_summary_retrieval": coarse_visual,
    }
    report["all_passed"] = (
        report["synthetic_2741s"]["passed"]
        and report["cg_bench_sample"]["passed"]
        and report["cg_bench_streaming"]["passed"]
        and report["coarse_visual_summary_retrieval"]["passed"]
    )
    print(json.dumps(report, indent=2))
    return 0 if report["all_passed"] else 2


def _coarse_visual_summary_retrieval() -> dict[str, object]:
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=180.0)
    coarse = segment_coarse_index(180.0, policy, regime=VideoRegime.LONG)
    target = coarse[3]
    center_span = {
        "start_s": (target.start_s + target.end_s) / 2.0 - 1.0,
        "end_s": (target.start_s + target.end_s) / 2.0 + 1.0,
    }
    retrieval = retrieve_coarse_clips(
        coarse_spans=coarse,
        query_text="what color is the vehicle driving from the left in the animation",
        segments=[
            {
                "segment_id": "coarse_schema:test:0003",
                "source_type": "coarse_visual_summary",
                "time_span": center_span,
                "text": "A white vehicle drives from the left in a simple animation.",
            }
        ],
        topk=2,
    )
    selected = retrieval["selected_coarse_indices"]
    return {
        "selected": selected,
        "fallback_reason": retrieval.get("fallback_reason"),
        "passed": bool(selected and selected[0] == 3 and retrieval.get("fallback_reason") is None),
    }


if __name__ == "__main__":
    raise SystemExit(main())
