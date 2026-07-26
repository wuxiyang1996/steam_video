from __future__ import annotations

import numpy as np

from steam_video_new.implicit_world_model.cgbench_grounded_navigation.quarantine import (
    quarantine_failed_grounding_cases,
)


def _transition(transition_id: str, status: str, row_index: int | None):
    return {
        "transition_id": transition_id,
        "real_observation": {
            "descriptor_status": status,
            "descriptor": {} if status == "grounded_qwen_vl_read" else None,
        },
        "target": {
            "observation_descriptor": {
                "embedding_ref": (
                    {"row_index": row_index} if row_index is not None else None
                )
            }
        },
    }


def test_quarantine_removes_whole_failed_cases_and_reindexes_embeddings(
    tmp_path,
) -> None:
    dataset = {
        "dataset_id": "test",
        "cases": [
            {
                "case_id": "case:keep",
                "executed_transitions": [
                    _transition("transition:keep", "grounded_qwen_vl_read", 1)
                ],
            },
            {
                "case_id": "case:drop",
                "executed_transitions": [
                    _transition("transition:drop-good", "grounded_qwen_vl_read", 0),
                    _transition("transition:drop-failed", "grounding_failed", None),
                ],
            },
        ],
    }
    hidden = {"cases": [{"case_id": "case:keep"}, {"case_id": "case:drop"}]}
    manifest = {
        "rows": [
            {"row_index": 0, "transition_id": "transition:drop-good"},
            {"row_index": 1, "transition_id": "transition:keep"},
        ]
    }
    matrix = np.stack(
        (np.zeros(2048, dtype=np.float32), np.ones(2048, dtype=np.float32))
    )

    result, hidden_result, new_manifest, report = quarantine_failed_grounding_cases(
        dataset,
        hidden,
        manifest,
        matrix,
        matrix_output=tmp_path / "quarantined.npy",
    )

    assert [row["case_id"] for row in result["cases"]] == ["case:keep"]
    assert [row["case_id"] for row in hidden_result["cases"]] == ["case:keep"]
    assert new_manifest["row_count"] == 1
    assert new_manifest["rows"] == [
        {"row_index": 0, "transition_id": "transition:keep"}
    ]
    assert report["answer_or_clue_label_used_for_exclusion"] is False
    assert report["retained_transition_count"] == 1
    assert np.load(tmp_path / "quarantined.npy").shape == (1, 2048)
