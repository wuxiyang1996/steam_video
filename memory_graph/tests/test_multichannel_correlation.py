from __future__ import annotations

import pytest

from memory_graph.multichannel_correlation import (
    NavigationPairSignal,
    build_multichannel_pair_audit,
)
from memory_graph.soft_correlation import build_soft_semantic_correlations
from memory_graph.types import MemoryNode, TimeSpan


def _node(index: int, *, value: str) -> MemoryNode:
    return MemoryNode(
        node_id=f"n{index}",
        video_id="video:test",
        time_span=TimeSpan(index * 10.0, index * 10.0 + 2.0),
        provenance={"producer": "test"},
        node_type="observation",
        text=f"event {index}",
        metadata={
            "participants": [
                {
                    "mention_id": "candidate-track:1",
                    "entity_type": "object",
                    "track_status": "global_track_candidate",
                }
            ],
            "states": [
                {
                    "mention_id": "candidate-track:1",
                    "attribute": "open_state",
                    "value": value,
                }
            ],
        },
    )


def test_multichannel_audit_adds_grounded_descriptors_without_admitting_them() -> None:
    nodes = (_node(0, value="closed"), _node(1, value="open"))
    semantic = build_soft_semantic_correlations(
        nodes,
        {"n0": (1.0, 0.0), "n1": (0.9, 0.1)},
        score_source="qwen-test",
    )

    audit = build_multichannel_pair_audit(nodes, semantic)

    assert audit["pair_count"] == 1
    assert audit["grounded_descriptors_admit_edges"] is False
    assert audit["learned_edge_selector_present"] is False
    pair = audit["pairs"][0]
    by_channel = {row["channel"]: row for row in pair["signals"]}
    assert by_channel["entity_correspondence"]["status"] == "candidate"
    assert by_channel["entity_correspondence"]["category"] == (
        "shared_cross_window_track_id"
    )
    assert by_channel["change"]["category"] == (
        "same_candidate_track_same_attribute_different_value"
    )
    assert by_channel["change"]["direction"] == "src_to_dst"
    assert "change" not in pair["legal_hop_sources"]
    assert pair["verified_identity_claim"] is False
    assert pair["verified_state_transition_claim"] is False
    assert pair["causal_claim"] is False
    assert pair["planner_preference"] is False


def test_multichannel_schema_rejects_causal_or_probability_claims() -> None:
    with pytest.raises(ValueError, match="causality or probability"):
        NavigationPairSignal(
            channel="change",
            status="candidate",
            category="causal_probability",
            source="test",
            evidence_refs=("n0", "n1"),
        )


def test_structured_descriptor_without_semantic_edge_is_not_a_legal_hop() -> None:
    nodes = (_node(0, value="closed"), _node(1, value="open"))
    semantic = build_soft_semantic_correlations(
        nodes,
        {"n0": (1.0, 0.0), "n1": (-1.0, 0.0)},
        score_source="qwen-test",
    )

    audit = build_multichannel_pair_audit(nodes, semantic)

    pair = audit["pairs"][0]
    assert pair["legal_hop_sources"] == []
    assert {row["channel"] for row in pair["signals"]} == {
        "entity_correspondence",
        "change",
    }
