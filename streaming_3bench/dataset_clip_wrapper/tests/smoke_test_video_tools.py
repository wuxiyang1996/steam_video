#!/usr/bin/env python3
"""Offline smoke test for the local video-tools clip-schema backend."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.perception.clip_policy import segment_video
from dataset_clip_wrapper.pipeline import _clip_id
from dataset_clip_wrapper.schemas import ClipPolicyConfig, VideoRegime
from dataset_clip_wrapper.perception.video_tool_backend import VideoToolConfig, VideoToolPerceptionBackend


def main() -> int:
    dataset_root = Path("/mnt/is_data/xwu/video_skills/data/datasets")
    adapter = get_adapter("video_holmes", dataset_root, split="train")
    item = next(adapter.iter_items(limit=1))
    if not item.video_path or not item.video_path.exists():
        print(json.dumps({"passed": False, "error": "missing_video_path"}, indent=2))
        return 2

    duration_s = float(item.duration_s or 0.0)
    policy = ClipPolicyConfig.dataset_default("video_holmes", VideoRegime.SHORT)
    span = segment_video(duration_s, policy, regime=VideoRegime.SHORT)[0]
    backend = VideoToolPerceptionBackend(VideoToolConfig(request_frames=3))
    schema = backend.build_clip_schema(
        clip_id=_clip_id(item.video_id, span.clip_index, span.granularity),
        clip=span,
        video_path=item.video_path,
    )
    report = {
        "passed": schema.get("producer") == "video_tool_perception_backend"
        and schema.get("sampled_frame_count", 0) > 0
        and "tool_error" not in schema,
        "clip_id": schema.get("clip_id"),
        "sampled_frame_count": schema.get("sampled_frame_count", 0),
        "observable_fact_count": len(schema.get("observable_facts", [])),
        "event_count": len(schema.get("events", [])),
        "tool_count": len(schema.get("tool_results", [])),
        "tool_error": schema.get("tool_error"),
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
