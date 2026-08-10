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
    assert record["candidate_node_ids"] == ["node:a:0", "node:a:1"]
    assert record["same_video_nonpositive_semantics"] == "ignore_unlabeled_not_negative"


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
