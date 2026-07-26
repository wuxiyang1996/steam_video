#!/usr/bin/env python3
"""Smoke test the offline short-video multi-hop benchmark profile."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.adapters import get_adapter
from dataset_clip_wrapper.dataset_graph_presets import SHORT_MULTI_HOP_DATASETS, regime_for_dataset
from dataset_clip_wrapper.pipeline import build_canonical_example
from dataset_clip_wrapper.schemas import (
    BackboneConfig,
    BenchmarkProfile,
    ClipRetrievalConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)


def _check_dataset(dataset: str) -> dict[str, object]:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    profile = BenchmarkProfile.SHORT_MULTI_HOP
    regime = regime_for_dataset(dataset, profile)  # type: ignore[arg-type]
    adapter = get_adapter(dataset, Path(dataset_root), split="train")
    item = next(adapter.iter_items(limit=1))
    config = WrapperConfig(
        dataset_root=dataset_root,
        dataset=dataset,  # type: ignore[arg-type]
        regime=regime,
        benchmark_profile=profile,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
        retrieval=ClipRetrievalConfig(enabled=False, topk=1),
        backbone=BackboneConfig(name="annotation_only"),
    )
    example = build_canonical_example(item, config=config)
    clip_policy = (example.get("evidence_index") or {}).get("clip_policy") or {}
    metadata = example.get("metadata") or {}

    errors: list[str] = []
    if dataset == "siv_bench":
        errors.append("siv_bench must not be part of short_multi_hop")
    if regime != VideoRegime.SHORT:
        errors.append(f"profile did not resolve short regime: {regime}")
    if metadata.get("video_regime") != "short":
        errors.append(f"metadata video_regime is not short: {metadata.get('video_regime')}")
    if clip_policy.get("online"):
        errors.append("short_multi_hop must be offline, not streaming online")
    if clip_policy.get("observation_end_s") is not None:
        errors.append("short_multi_hop must not impose observation_end_s")
    if example.get("task_family") != "short_video_multi_hop_qa":
        errors.append(f"unexpected task_family: {example.get('task_family')}")
    if not (example.get("video") or {}).get("derived_clips"):
        errors.append("no derived clips")

    return {
        "dataset": dataset,
        "example_id": example.get("example_id"),
        "profile": profile.value,
        "regime": regime.value,
        "task_family": example.get("task_family"),
        "online": clip_policy.get("online"),
        "clip_count": len((example.get("video") or {}).get("derived_clips") or []),
        "passed": not errors,
        "errors": errors,
    }


def main() -> int:
    report = [_check_dataset(dataset) for dataset in SHORT_MULTI_HOP_DATASETS]
    siv_replaced = "siv_bench" not in SHORT_MULTI_HOP_DATASETS
    report.append({
        "dataset": "siv_bench",
        "profile": BenchmarkProfile.SHORT_MULTI_HOP.value,
        "included": "siv_bench" in SHORT_MULTI_HOP_DATASETS,
        "passed": siv_replaced,
        "errors": [] if siv_replaced else ["siv_bench is still included"],
    })
    print(json.dumps(report, indent=2))
    return 0 if all(row["passed"] for row in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
