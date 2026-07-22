from __future__ import annotations

from memory_graph.soft_correlation import (
    SoftCorrelationAdmissionPolicy,
    build_soft_semantic_correlations,
    calibrate_soft_correlation_admission,
)
from memory_graph.types import MemoryNode, TimeSpan


def _node(index: int, *, visual_signature: str = "") -> MemoryNode:
    return MemoryNode(
        node_id=f"l1:{index}",
        video_id="video:test",
        time_span=TimeSpan(float(index), float(index + 1)),
        provenance={"producer": "test"},
        node_type="semantic_observation",
        text=f"semantic event {index}",
        metadata={"visual_signature": visual_signature},
    )


def test_soft_correlation_scores_all_non_temporal_pairs_but_emits_sparse_edges() -> (
    None
):
    nodes = [_node(index) for index in range(5)]
    result = build_soft_semantic_correlations(
        nodes,
        {
            "l1:0": (1.0, 0.0, 0.0),
            "l1:1": (0.0, 1.0, 0.0),
            "l1:2": (0.99, 0.1, 0.0),
            "l1:3": (0.5, 0.5, 0.707),
            "l1:4": (-1.0, 0.0, 0.0),
        },
        score_source="qwen3-vl-embedding-test",
        excluded_pairs=(("l1:0", "l1:1"), ("l1:1", "l1:2")),
    )

    assert result.evaluated_pair_count == 8
    assert result.excluded_temporal_pair_count == 2
    by_pair = {(edge.src, edge.dst): edge for edge in result.edges}
    assert ("l1:0", "l1:2") in by_pair
    assert all(pair not in by_pair for pair in {("l1:0", "l1:1"), ("l1:1", "l1:2")})
    assert by_pair[("l1:0", "l1:2")].semantic_similarity > 0.99
    assert (
        by_pair[("l1:0", "l1:2")]
        .to_dict()["score_semantics"]
        .endswith("not_probability")
    )
    assert result.audit_dict()["top_k_applied"] is False


def test_repeated_legacy_visual_signature_does_not_create_a_clique() -> None:
    nodes = [
        _node(index, visual_signature="hands with red nail polish")
        for index in range(5)
    ]
    embeddings = {
        node.node_id: tuple(
            1.0 if row == column else 0.0 for column in range(len(nodes))
        )
        for row, node in enumerate(nodes)
    }

    result = build_soft_semantic_correlations(
        nodes,
        embeddings,
        score_source="qwen3-vl-embedding-test",
    )

    assert result.evaluated_pair_count == 10
    assert result.edges == ()
    assert result.audit_dict()["edge_density"] == 0.0


def test_near_identical_embeddings_form_a_recurrence_chain_not_a_clique() -> None:
    nodes = [_node(index) for index in range(5)]
    result = build_soft_semantic_correlations(
        nodes,
        {node.node_id: (1.0, 0.0) for node in nodes},
        score_source="qwen3-vl-embedding-test",
    )

    assert [(edge.src, edge.dst) for edge in result.edges] == [
        ("l1:0", "l1:1"),
        ("l1:1", "l1:2"),
        ("l1:2", "l1:3"),
        ("l1:3", "l1:4"),
    ]
    assert all(edge.channel == "semantic_recurrence" for edge in result.edges)
    assert result.audit_dict()["semantic_equivalence_class_count"] == 1


def test_pair_audit_covers_every_pair_and_keeps_semantic_edges_bidirectional() -> None:
    nodes = [_node(index) for index in range(5)]
    result = build_soft_semantic_correlations(
        nodes,
        {
            "l1:0": (1.0, 0.0, 0.0),
            "l1:1": (0.0, 1.0, 0.0),
            "l1:2": (0.99, 0.1, 0.0),
            "l1:3": (0.5, 0.5, 0.707),
            "l1:4": (-1.0, 0.0, 0.0),
        },
        score_source="qwen3-vl-embedding-test",
        excluded_pairs=(("l1:0", "l1:1"),),
    )

    assert len(result.pair_audits) == 10
    assert sum(row.excluded_temporal_pair for row in result.pair_audits) == 1
    assert all(
        edge.src_to_dst_affinity == edge.dst_to_src_affinity
        for edge in result.edges
    )
    audit = result.pair_audit_dict()
    assert audit["contains_question_or_answer"] is False
    assert audit["summary"]["pair_audit_row_count"] == 10
    assert audit["summary"]["top_k_applied"] is False


def test_global_admission_threshold_rejects_without_applying_top_k() -> None:
    nodes = [_node(index) for index in range(5)]
    result = build_soft_semantic_correlations(
        nodes,
        {
            "l1:0": (1.0, 0.0, 0.0),
            "l1:1": (0.0, 1.0, 0.0),
            "l1:2": (0.99, 0.1, 0.0),
            "l1:3": (0.5, 0.5, 0.707),
            "l1:4": (-1.0, 0.0, 0.0),
        },
        score_source="qwen3-vl-embedding-test",
        admission_policy=SoftCorrelationAdmissionPolicy(
            policy_id="test-global-threshold",
            minimum_semantic_similarity=0.9999,
            calibration_source="unit_test",
        ),
    )

    assert result.edges == ()
    assert any(
        "below_global_similarity_threshold" in row.admission_reasons
        for row in result.pair_audits
    )
    assert result.audit_dict()["admission_policy"]["top_k_applied"] is False


def test_positive_only_calibration_does_not_fabricate_precision() -> None:
    policy, report = calibrate_soft_correlation_admission(
        (0.9, 0.8, 0.7, 0.2),
        target_positive_coverage=0.75,
        calibration_source="unit_test_positive_bridges",
    )

    assert policy.minimum_semantic_similarity == 0.7
    assert report["selected_positive_coverage"] == 0.75
    assert report["selected_precision"] is None
    assert report["precision_status"] == "unavailable_no_trusted_negative_labels"
    assert report["unmatched_pairs_treated_as_negative"] is False
