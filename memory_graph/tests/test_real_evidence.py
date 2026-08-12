from __future__ import annotations

import json

import pytest

from memory_graph.types import MemoryNode, TimeSpan
from steam_video_new.implicit_world_model.full_graph_iwm import (
    ActionKind,
    ArtifactBackedRealEvidenceReader,
    LegalGraphAction,
    RetainedEvidenceGraph,
    build_node_reread_artifact,
)


def _observation() -> MemoryNode:
    return MemoryNode(
        node_id="l1:event",
        video_id="video:test",
        time_span=TimeSpan(2.0, 3.0),
        provenance={"producer": "test"},
        text="person interacts with object",
        metadata={"predicate": "person interacts with object"},
    )


def _graph() -> RetainedEvidenceGraph:
    return RetainedEvidenceGraph(
        graph_id="graph:test",
        nodes=(_observation(),),
        temporal_edges=(),
        correlation_edges=(),
        capacity=1,
    )


def test_locked_reread_enriches_but_cannot_replace_grounding(tmp_path) -> None:
    path = tmp_path / "rereads.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "steam-grounded-real-evidence-reread/v0.1",
                "records": [
                    {
                        "node_id": "l1:event",
                        "video_id": "video:test",
                        "time_span": {"start_s": 2.0, "end_s": 3.0},
                        "descriptor": "person places the red cup on the table",
                        "structured_observation": {
                            "action_kind": "placement",
                            "target": "red cup",
                        },
                        "review_status": "human_locked",
                        "model": "qwen-vl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    reader = ArtifactBackedRealEvidenceReader(path)
    original = _observation()
    action = LegalGraphAction(
        "read:event", ActionKind.START_AT, target_id="l1:event", reads_evidence=True
    )

    enriched = reader.read(original, action, _graph())

    assert enriched.node_id == original.node_id
    assert enriched.time_span == original.time_span
    assert enriched.text == "person places the red cup on the table"
    assert enriched.metadata["target"] == "red cup"
    assert enriched.metadata["visual_reread"]["hidden_answer_used"] is False


def test_unreviewed_reread_fails_closed(tmp_path) -> None:
    path = tmp_path / "rereads.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "steam-grounded-real-evidence-reread/v0.1",
                "records": [
                    {
                        "node_id": "l1:event",
                        "review_status": "unreviewed",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not independently locked"):
        ArtifactBackedRealEvidenceReader(path)


def test_node_reread_builder_is_blinded_and_unreviewed(tmp_path) -> None:
    class _Client:
        model = "qwen-vl-test"

        def perceive(self, prompt, *, image_urls=None, system=""):
            assert "question" not in prompt.lower()
            assert "answer" not in prompt.lower()
            assert image_urls == ["data:image/jpeg;base64,test"]
            assert "visible evidence" in system
            return {
                "descriptor": "person places the red cup on the table",
                "action_kind": "placement",
                "participants": ["person", "red cup"],
                "visible_states": ["cup on table"],
                "state_change": "cup moves from hand to table",
            }

    def _sampler(path, *, windows, frames_per_window):
        assert path == (tmp_path / "video.mp4").resolve()
        assert windows[0]["start_s"] == 2.0
        assert frames_per_window == 4
        return ["data:image/jpeg;base64,test"], [
            {"frame_index": 0, "time_s": 2.5, "purpose": "executed_l1_read"}
        ]

    artifact = build_node_reread_artifact(
        graph=_graph(),
        node_ids=("l1:event",),
        video_path=tmp_path / "video.mp4",
        client=_Client(),
        frames_per_window=4,
        frame_sampler=_sampler,
    )

    assert artifact["hidden_question_or_answer_used"] is False
    assert artifact["training_allowed"] is False
    assert artifact["records"][0]["review_status"] == "unreviewed"
    assert artifact["records"][0]["structured_observation"]["action_kind"] == (
        "placement"
    )
