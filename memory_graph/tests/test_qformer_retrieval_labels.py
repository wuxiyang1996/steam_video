from __future__ import annotations

import numpy as np
from types import SimpleNamespace

from steam_video_new.implicit_world_model.reasoning_v2.qformer import (
    SLOT_NAMES,
    FeatureRow,
    FourSlotFeatureStore,
    write_feature_store,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.compile_retrieval_labels import (
    compile_retrieval_labels,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.train_qf2 import (
    _build_split_examples,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.train_gated_residual import (
    _partition_train_records,
    _replace_with_visual_hard_negatives,
)


def _store(tmp_path) -> FourSlotFeatureStore:
    rows = (
        FeatureRow(0, "node:a:0", "video:a", "hash:0"),
        FeatureRow(1, "node:a:1", "video:a", "hash:1"),
        FeatureRow(2, "node:b:0", "video:b", "hash:2"),
    )
    arrays = {
        name: np.ones((len(rows), 3), dtype=np.float32)
        for name in SLOT_NAMES
    }
    manifest = write_feature_store(
        tmp_path,
        arrays=arrays,
        validity=np.ones((len(rows), len(SLOT_NAMES)), dtype=np.bool_),
        rows=rows,
        encoders={name: f"fixture/{name}" for name in SLOT_NAMES},
        source_contract="fixture-safe-pre-read/v1",
        boundary_audit_version="fixture-boundary/v1",
    )
    return FourSlotFeatureStore(manifest)


def test_compile_labels_uses_clue_overlap_without_answer_leakage(tmp_path) -> None:
    navigation = {
        "cases": [
            {
                "case_id": "case:1",
                "video_id": "video:a",
                "split": "validation",
                "planner_input": {"question": "What happened next?"},
            }
        ],
        "qformer_node_spans": {
            "node:a:0": {"start_s": 0.0, "end_s": 2.0},
            "node:a:1": {"start_s": 2.0, "end_s": 4.0},
            "node:b:0": {"start_s": 0.0, "end_s": 2.0},
        },
    }
    hidden_targets = {
        "cases": [
            {
                "case_id": "case:1",
                "video_id": "video:a",
                "clue_intervals": [{"start_s": 2.5, "end_s": 3.0}],
                "answer": "must never be copied",
            }
        ]
    }

    result = compile_retrieval_labels(
        feature_store=_store(tmp_path),
        navigation_dataset=navigation,
        hidden_targets=hidden_targets,
    )

    assert result["record_count"] == 1
    assert result["answer_fields_present"] is False
    assert "must never be copied" not in str(result)
    record = result["records"][0]
    assert "answer" not in record
    assert record["positive_node_ids"] == ["node:a:1"]
    assert record["positive_node_groups"] == [
        {"clue_index": 0, "node_ids": ["node:a:1"]}
    ]
    assert record["clue_group_count"] == 1
    assert record["candidate_node_ids"] == ["node:a:0", "node:a:1"]
    assert record["trusted_same_video_negative_node_ids"] == []
    assert "remaining_unlabeled_nodes_are_ignore" in record["same_video_nonpositive_semantics"]

    smaller_margin = compile_retrieval_labels(
        feature_store=_store(tmp_path / "smaller-margin"),
        navigation_dataset=navigation,
        hidden_targets=hidden_targets,
        same_video_negative_margin_s=0.25,
    )
    assert smaller_margin["schema_version"] == "steam-qformer-retrieval-labels/v0.3"
    assert smaller_margin["records"][0]["trusted_same_video_negative_node_ids"] == [
        "node:a:0"
    ]


def test_compile_labels_can_require_every_clue_group(tmp_path) -> None:
    navigation = {
        "cases": [
            {
                "case_id": "case:partial",
                "video_id": "video:a",
                "split": "train",
                "planner_input": {"question": "What happened?"},
            }
        ],
        "qformer_node_spans": {
            "node:a:0": {"start_s": 0.0, "end_s": 2.0},
            "node:a:1": {"start_s": 2.0, "end_s": 4.0},
        },
    }
    hidden_targets = {
        "cases": [
            {
                "case_id": "case:partial",
                "video_id": "video:a",
                "clue_intervals": [
                    {"start_s": 0.5, "end_s": 1.0},
                    {"start_s": 8.0, "end_s": 9.0},
                ],
            }
        ]
    }

    unfiltered = compile_retrieval_labels(
        feature_store=_store(tmp_path / "unfiltered"),
        navigation_dataset=navigation,
        hidden_targets=hidden_targets,
    )
    assert unfiltered["records"][0]["clue_group_count"] == 2
    assert unfiltered["records"][0]["uncovered_clue_group_count"] == 1

    filtered = compile_retrieval_labels(
        feature_store=_store(tmp_path / "filtered"),
        navigation_dataset=navigation,
        hidden_targets=hidden_targets,
        require_all_clue_groups=True,
    )
    assert filtered["record_count"] == 0
    assert filtered["require_all_clue_groups"] is True
    assert filtered["skips"]["uncovered_clue_groups"] == 1


def test_qf2_sampler_uses_only_cross_video_trusted_negatives() -> None:
    rows = (
        FeatureRow(0, "node:a:positive", "video:a", "hash:0"),
        FeatureRow(1, "node:a:unlabeled", "video:a", "hash:1"),
        FeatureRow(2, "node:b:positive", "video:b", "hash:2"),
        FeatureRow(3, "node:b:other", "video:b", "hash:3"),
    )

    class Cache:
        manifest = SimpleNamespace(rows=rows)

        def __contains__(self, node_id):
            return any(row.node_id == node_id for row in rows)

    records = [
        {
            "case_id": "case:a",
            "video_id": "video:a",
            "split": "train",
            "positive_node_ids": ["node:a:positive"],
        },
        {
            "case_id": "case:b",
            "video_id": "video:b",
            "split": "train",
            "positive_node_ids": ["node:b:positive"],
        },
    ]
    examples = _build_split_examples(
        records, Cache(), split="train", trusted_negatives=2, seed=9
    )
    case_a = next(row for row in examples if row["case_id"] == "case:a")
    negatives = {
        node_id
        for node_id, negative in zip(
            case_a["candidate_node_ids"], case_a["negative_mask"]
        )
        if negative
    }
    assert negatives == {"node:b:positive", "node:b:other"}
    assert "node:a:unlabeled" not in case_a["candidate_node_ids"]


def test_gated_fit_selection_partition_is_video_disjoint_and_deterministic() -> None:
    records = [
        {"case_id": f"case:{index}", "video_id": f"video:{index // 2}", "split": "train"}
        for index in range(8)
    ]
    fit, selection = _partition_train_records(records, fraction=0.25, seed=17)
    fit_again, selection_again = _partition_train_records(
        records, fraction=0.25, seed=17
    )
    assert fit == fit_again
    assert selection == selection_again
    assert {row["video_id"] for row in fit}.isdisjoint(
        {row["video_id"] for row in selection}
    )
    assert {row["split"] for row in fit} == {"fit"}
    assert {row["split"] for row in selection} == {"selection"}


def test_qf2_sampler_can_mix_only_explicit_same_video_negatives() -> None:
    rows = (
        FeatureRow(0, "node:a:positive", "video:a", "hash:0"),
        FeatureRow(1, "node:a:trusted", "video:a", "hash:1"),
        FeatureRow(2, "node:a:ignored", "video:a", "hash:2"),
        FeatureRow(3, "node:b:other", "video:b", "hash:3"),
    )

    class Cache:
        manifest = SimpleNamespace(rows=rows)

        def __contains__(self, node_id):
            return any(row.node_id == node_id for row in rows)

    records = [
        {
            "case_id": "case:a",
            "video_id": "video:a",
            "split": "train",
            "positive_node_ids": ["node:a:positive"],
            "trusted_same_video_negative_node_ids": ["node:a:trusted"],
        },
        {
            "case_id": "case:b",
            "video_id": "video:b",
            "split": "train",
            "positive_node_ids": ["node:b:other"],
            "trusted_same_video_negative_node_ids": [],
        },
    ]
    examples = _build_split_examples(
        records,
        Cache(),
        split="train",
        trusted_negatives=2,
        same_video_negatives=1,
        seed=9,
    )
    case_a = next(row for row in examples if row["case_id"] == "case:a")
    negatives = {
        node_id
        for node_id, is_negative in zip(
            case_a["candidate_node_ids"], case_a["negative_mask"]
        )
        if is_negative
    }
    assert negatives == {"node:a:trusted", "node:b:other"}
    assert "node:a:ignored" not in case_a["candidate_node_ids"]


def test_visual_hard_miner_chooses_highest_anchor_safe_node_only() -> None:
    example = {
        "case_id": "case:a",
        "candidate_node_ids": ["positive", "cross:1", "cross:2"],
        "positive_mask": [True, False, False],
        "negative_mask": [False, True, True],
    }
    records = [
        {
            "case_id": "case:a",
            "trusted_same_video_negative_node_ids": ["safe:low", "safe:high"],
        }
    ]

    class Features:
        def __contains__(self, node_id):
            return node_id in {"safe:low", "safe:high"}

        def node(self, node_id):
            visual = {
                "safe:low": np.asarray([0.0, 1.0], dtype=np.float32),
                "safe:high": np.asarray([1.0, 0.0], dtype=np.float32),
            }[node_id]
            return {"validity": np.ones(4, dtype=np.bool_), "features": {"visual": visual}}

    class Questions:
        def case(self, case_id):
            return np.asarray([1.0, 0.0], dtype=np.float32)

    _replace_with_visual_hard_negatives(
        [example], records, feature_store=Features(), question_store=Questions(), count=1
    )
    assert example["candidate_node_ids"] == ["positive", "safe:high", "cross:1"]
    assert "safe:low" not in example["candidate_node_ids"]
