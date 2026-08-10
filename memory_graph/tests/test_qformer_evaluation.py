from __future__ import annotations

import json

import numpy as np
import pytest

from steam_video_new.implicit_world_model.reasoning_v2.qformer import (
    SLOT_NAMES,
    FeatureRow,
    write_feature_store,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.evaluate_embedding_baselines import (
    evaluate_embedding_baselines,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.compare_gated_reports import (
    paired_case_bootstrap,
)

from steam_video_new.implicit_world_model.reasoning_v2.qformer.evaluation import (
    clue_group_metrics,
    permutation_consistency,
    retrieval_metrics,
)
from steam_video_new.implicit_world_model.reasoning_v2.qformer.feature_store import (
    sha256_file,
)


def test_retrieval_metrics_report_partial_and_complete_clue_coverage() -> None:
    scores = np.asarray([[4.0, 3.0, 2.0, 1.0], [4.0, 1.0, 3.0, 2.0]])
    positives = np.asarray([[1, 1, 0, 0], [0, 0, 1, 0]], dtype=np.bool_)
    valid = np.ones_like(positives)
    metrics = retrieval_metrics(scores, positives, valid, ks=(1, 2))
    assert metrics["recall@1"] == 0.25
    assert metrics["all_clue_coverage@1"] == 0.0
    assert metrics["recall@2"] == 1.0
    assert metrics["all_clue_coverage@2"] == 1.0


def test_permutation_consistency_restores_node_order() -> None:
    original = np.asarray([[0.1, 0.2, 0.3]])
    permutation = np.asarray([[2, 0, 1]])
    permuted = np.take_along_axis(original, permutation, axis=1)
    report = permutation_consistency(original, permuted, permutation)
    assert report["max_absolute_error"] == 0.0


def test_clue_group_metrics_accept_interchangeable_nodes() -> None:
    scores = np.asarray([[0.8, 0.9, 0.85, 0.1]], dtype=np.float32)
    valid = np.ones_like(scores, dtype=np.bool_)
    groups = [
        np.asarray(
            [
                [True, True, False, False],
                [False, False, True, False],
            ],
            dtype=np.bool_,
        )
    ]
    metrics = clue_group_metrics(scores, valid, groups, ks=(1, 2))
    assert metrics["clue_group_recall@1"] == 0.5
    assert metrics["all_clue_group_coverage@1"] == 0.0
    assert metrics["clue_group_recall@2"] == 1.0
    assert metrics["all_clue_group_coverage@2"] == 1.0


def test_embedding_baselines_use_full_same_video_pool(tmp_path) -> None:
    rows = tuple(
        FeatureRow(index, f"node:{index}", "video:a", f"hash:{index}")
        for index in range(3)
    )
    base = np.asarray([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    feature_manifest = write_feature_store(
        tmp_path / "features",
        arrays={name: base.copy() for name in SLOT_NAMES},
        validity=np.ones((3, 4), dtype=np.bool_),
        rows=rows,
        encoders={name: "fixture/shared" for name in SLOT_NAMES},
        source_contract="fixture-safe/v1",
        boundary_audit_version="fixture-audit/v1",
    )
    labels = {
        "answer_fields_present": False,
        "records": [{
            "case_id": "case:1",
            "video_id": "video:a",
            "split": "validation",
            "positive_node_ids": ["node:0"],
            "candidate_node_ids": ["node:0", "node:1", "node:2"],
        }],
    }
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps(labels), encoding="utf-8")
    question_dir = tmp_path / "questions"
    question_dir.mkdir()
    matrix_path = question_dir / "question_embeddings.npy"
    np.save(matrix_path, np.asarray([[1.0, 0.0]], dtype=np.float32))
    question_manifest = question_dir / "manifest.json"
    question_manifest.write_text(
        json.dumps({
            "schema_version": "steam-qformer-question-embeddings/v0.1",
            "labels_checksum": sha256_file(labels_path),
            "encoder": "fixture/shared",
            "dimension": 2,
            "dtype": "float32",
            "matrix": matrix_path.name,
            "matrix_checksum": sha256_file(matrix_path),
            "rows": [{"row_index": 0, "case_id": "case:1"}],
            "answer_fields_present": False,
        }),
        encoding="utf-8",
    )
    report = evaluate_embedding_baselines(
        feature_manifest=feature_manifest,
        labels_path=labels_path,
        question_manifest=question_manifest,
    )
    assert report["question_count"] == 1
    assert report["metrics"]["b0_caption_question_cosine"]["recall@1"] == 1.0


def test_paired_bootstrap_reports_case_aligned_positive_delta() -> None:
    def report(offset: float):
        rows = []
        for index, value in enumerate((0.1, 0.3, 0.5)):
            rows.append(
                {
                    "case_id": f"case:{index}",
                    "node_union_metrics": {"mrr": value + offset, "recall@8": value + offset},
                    "clue_group_metrics": {
                        "clue_group_recall@4": value + offset,
                        "clue_group_recall@8": value + offset,
                        "all_clue_group_coverage@8": value + offset,
                    },
                }
            )
        return {"heldout_ranking": {"per_case": rows}}

    comparison = paired_case_bootstrap(report(0.0), report(0.1), samples=500, seed=4)
    assert comparison["mrr"]["mean_delta"] == pytest.approx(0.1)
    assert comparison["mrr"]["bootstrap_probability_positive"] == 1.0
