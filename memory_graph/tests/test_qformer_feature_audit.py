from __future__ import annotations

import json

from steam_video_new.implicit_world_model.reasoning_v2.qformer.audit_features import (
    audit_overlays,
)


def _overlay(tmp_path, node):
    path = tmp_path / f"{node['node_id']}.json"
    path.write_text(json.dumps({"l1_observations": [node]}))
    return path


def test_audit_does_not_treat_text_embedding_as_visual(tmp_path) -> None:
    node = {
        "node_id": "a",
        "video_id": "v",
        "text": "bounded caption",
        "time_span": {"start_s": 0, "end_s": 1},
        "provenance": {"uses_hidden_supervision": False},
        "embedding_ref": {"path": "combined.npy", "row_index": 0},
        "metadata": {
            "participants": [{"entity_type": "person"}],
            "visibility": {"hidden_supervision": False},
        },
    }
    report = audit_overlays([_overlay(tmp_path, node)])
    assert report["coverage"]["combined_text_embedding"] == 1.0
    assert report["coverage"]["explicit_visual_embedding"] == 0.0
    assert report["qf1_training_ready"] is False
    assert "independent_visual_embeddings_missing" in report["blockers"]


def test_audit_accepts_explicit_visual_contract(tmp_path) -> None:
    node = {
        "node_id": "a",
        "video_id": "v",
        "text": "bounded caption",
        "time_span": {"start_s": 0, "end_s": 1},
        "provenance": {"uses_hidden_supervision": False},
        "embedding_ref": {
            "path": "visual.npy",
            "row_index": 0,
            "input_modality": "video",
        },
        "metadata": {
            "states": [{"attribute": "open", "value": "yes"}],
            "visibility": {"hidden_supervision": False},
        },
    }
    report = audit_overlays([_overlay(tmp_path, node)])
    assert report["qf1_training_ready"] is True
    assert report["coverage"]["complete_four_slot_ready"] == 1.0

