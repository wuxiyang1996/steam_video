#!/usr/bin/env python3
"""Smoke test split-aware training manifest helpers."""

from __future__ import annotations

from types import SimpleNamespace

from dataset_clip_wrapper.manifests.build_training_manifests import (
    _group_leakage_count,
    _manifest_row,
    _split_groups,
)
from dataset_clip_wrapper.schemas import BenchmarkProfile


def test_split_groups_are_video_isolated() -> None:
    groups = ["video_holmes:v1", "video_holmes:v2", "cg_bench:v3", "cg_bench:v4"]
    split_by_group = _split_groups(groups, train_ratio=0.5, dev_ratio=0.25, seed="smoke")
    manifests = {"train": [], "dev": [], "test": []}
    for group, split in split_by_group.items():
        manifests[split].append({"split_group_key": group})
    assert _group_leakage_count(manifests) == 0
    assert set(split_by_group) == set(groups)


def test_manifest_row_strips_gold_question_fields() -> None:
    item = SimpleNamespace(
        dataset="video_holmes",
        example_id="ex1",
        video_id="video1",
        video_path=None,
        duration_s=10.0,
        task_family="toy",
        question={
            "question_id": "q1",
            "question_text": "What happens?",
            "answer": {"label": "A"},
            "options": [{"label": "A", "text": "Runs"}],
        },
        hidden_supervision_sources=["official_answer"],
        raw_source_refs=[],
    )
    row = _manifest_row(
        item,
        split="train",
        source_split="train",
        benchmark_profile=BenchmarkProfile.DEFAULT,
        seed="smoke",
    )
    assert row["split_group_key"] == "video_holmes:video1"
    assert "answer" not in row["question"]
    assert row["hidden_supervision"]["available_for_inference"] is False
    assert row["visible_runtime_mode"] == "video_only"


if __name__ == "__main__":
    test_split_groups_are_video_isolated()
    test_manifest_row_strips_gold_question_fields()
    print("training manifest smoke test passed")
