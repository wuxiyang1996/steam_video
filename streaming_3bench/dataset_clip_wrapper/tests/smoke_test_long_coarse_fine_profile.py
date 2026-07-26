#!/usr/bin/env python3
"""Smoke test the long-video coarse→fine graph profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.dataset_graph_presets import (
    LONG_COARSE_FINE_DATASETS,
    apply_profile_defaults,
    clip_policy_for,
    regime_for_dataset,
    retrieval_for,
)
from dataset_clip_wrapper.runners.llm_pipeline import _question_retrieval_query, _resolve_perception_spans
from dataset_clip_wrapper.pipeline import build_canonical_example
from dataset_clip_wrapper.schemas import (
    BackboneConfig,
    BenchmarkProfile,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)


def _check_dataset(dataset: str) -> dict[str, object]:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    profile = BenchmarkProfile.LONG_COARSE_FINE
    regime = regime_for_dataset(dataset, profile)  # type: ignore[arg-type]
    policy = clip_policy_for(dataset, regime)  # type: ignore[arg-type]
    retrieval = retrieval_for(regime)
    apply_profile_defaults(
        dataset=dataset,  # type: ignore[arg-type]
        regime=regime,
        profile=profile,
        clip_policy=policy,
        retrieval=retrieval,
    )
    adapter = get_adapter(dataset, Path(dataset_root), split="train")
    item = next(adapter.iter_items(limit=1))
    config = WrapperConfig(
        dataset_root=dataset_root,
        dataset=dataset,  # type: ignore[arg-type]
        regime=regime,
        benchmark_profile=profile,
        mode=RuntimeMode.VIDEO_ONLY,
        clip_policy=policy,
        retrieval=retrieval,
        split="train",
        limit=1,
        backbone=BackboneConfig(name="annotation_only"),
    )
    example = build_canonical_example(item, config=config)
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    resolved_policy = config.resolved_clip_policy(duration_s)
    query = _question_retrieval_query(item.question)
    spans, perception = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=resolved_policy,
        regime=regime,
        retrieval_config=retrieval,
        question_text=query,
        visible_segments=(example.get("video") or {}).get("segments") or [],
        mode=RuntimeMode.VIDEO_ONLY,
    )

    clip_policy = (example.get("evidence_index") or {}).get("clip_policy") or {}
    retrieved = (perception.get("retrieval") or {}).get("selected_coarse_indices") or []
    errors: list[str] = []
    if regime != VideoRegime.LONG:
        errors.append(f"profile did not resolve long regime: {regime}")
    if example.get("task_family") != "long_video_coarse_to_fine_qa":
        errors.append(f"unexpected task_family: {example.get('task_family')}")
    if clip_policy.get("strategy") != "hierarchical":
        errors.append(f"unexpected clip strategy: {clip_policy.get('strategy')}")
    if clip_policy.get("index_fine_expansion") != "retrieval_gated":
        errors.append(f"unexpected fine expansion: {clip_policy.get('index_fine_expansion')}")
    if not retrieval.query_in_video_only:
        errors.append("video_only long profile must use question/options for retrieval")
    if retrieval.topk < 3:
        errors.append(f"long profile topk too small: {retrieval.topk}")
    if not retrieved:
        errors.append("retrieval selected no coarse windows")
    if perception.get("perception_clip_count", 0) >= max(1, (example.get("metadata") or {}).get("clip_count", 1)):
        errors.append("retrieval-gated perception did not reduce fine clips")
    if " " not in query:
        errors.append("question retrieval query appears empty or option-free")

    return {
        "dataset": dataset,
        "example_id": example.get("example_id"),
        "profile": profile.value,
        "regime": regime.value,
        "task_family": example.get("task_family"),
        "coarse_clip_count": (example.get("metadata") or {}).get("coarse_clip_count"),
        "canonical_clip_count": (example.get("metadata") or {}).get("clip_count"),
        "retrieval_topk": retrieval.topk,
        "retrieved_coarse": retrieved,
        "perception_fine_count": perception.get("perception_clip_count"),
        "query_preview": query[:160],
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    report = [_check_dataset(dataset) for dataset in LONG_COARSE_FINE_DATASETS]
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
