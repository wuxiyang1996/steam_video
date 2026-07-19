from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from memory_graph.adapter import canonical_to_memory_nodes
from memory_graph.atomic_events import parse_atomic_events
from memory_graph.baseline_manifest import freeze_baseline, verify_baseline
from memory_graph.calibration import calibrate
from memory_graph.cli import main as cli_main
from memory_graph.contracts import (
    AtomicEvent,
    EntityLinkJudgment,
    EntityMention,
    L1HumanAudit,
    StateAssertion,
)
from memory_graph.event_adapter import atomic_events_to_graph_nodes
from memory_graph.graph_builder import LinearRelationScorer, build_memory_graph
from memory_graph.identity_tracks import build_identity_tracks
from memory_graph.identity_reread import (
    apply_identity_reread_packet,
    apply_identity_reread_to_artifact,
    prepare_identity_reread_packet,
)
from memory_graph.identity_verifier import verify_identity_candidates
from memory_graph.l1_relation_audit import (
    create_locked_audit_split,
    evaluate_l1_relation_packet,
    prepare_l1_relation_packet,
)
from memory_graph.l1_structuralizer import structuralize_video_skills_l1
from memory_graph.l1_structural_edges import materialize_l1_structural_relations
from memory_graph.mechanism_candidates import (
    generate_mechanism_candidates,
    generate_video_skills_l1_edge_candidates,
)
from memory_graph.navigation import (
    NavigationActionType,
    NavigationBeliefState,
    RuleBasedDependencyWorldModel,
    execute_real_graph_read,
    plan_next_read,
    propose_navigation_actions,
    update_belief_after_read,
)
from memory_graph.navigation_ablation import evaluate_navigation_ablation
from memory_graph.openrouter_validation import (
    _computed_audit_summary,
    _parse_teacher_relations,
)
from memory_graph.pipeline import (
    _apply_visual_rereads,
    _select_l1_inputs,
    _verify_proposals,
)
from memory_graph.reliability import audit_l1_nodes
from memory_graph.retry_overlay_audit import (
    event_graph_from_overlay,
    update_run_summary,
)
from memory_graph.schema_validation import validate_overlay_artifact
from memory_graph.selectstream_policy import (
    merge_preserves_causal_witness,
    plan_bounded_memory,
)
from memory_graph.state_relations import derive_state_transition_candidates
from memory_graph.types import (
    CausalWitness,
    CausalTemporalOverlay,
    ConfidenceComponents,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_EMBEDDING_MODEL,
    MechanismKind,
    MemoryNode,
    RelationBelief,
    RelationStatus,
    TimeSpan,
    VisualVerification,
)
from memory_graph.verifiers import verify_relation
from memory_graph.visual_verifier import _evidence_grounding_problems
from memory_graph.video_l1 import (
    PayloadVideoL1Provider,
    VideoL1AtomicEventExtractor,
    _assign_track_ids,
    _event_to_l1_node,
    _parse_coarse_response,
    _parse_fine_response,
)
from memory_graph.video_l1_evaluation import (
    build_video_l1_annotation_packet,
    evaluate_video_l1_human_audit,
    summarize_video_l1,
)
from memory_graph.video_skills_l1 import (
    VideoSkillsL1AtomicEventExtractor,
    audit_video_skills_l1,
)


class MemoryGraphTest(unittest.TestCase):
    def test_adapter_prefers_semantic_event_nodes_over_raw_clips(self) -> None:
        canonical = {
            "schema_version": "video-skills-relaunch/v0.1",
            "example_id": "example:1",
            "dataset": "video_holmes",
            "video": {"video_id": "video-1", "primary_path": "/tmp/video.mp4"},
            "available_inputs": {"mode": "video_only"},
            "evidence_candidates": [],
            "evidence_index": {
                "index_id": "index:1",
                "clip_policy": {"strategy": "fixed_window"},
                "nodes": [
                    {
                        "node_id": "clip:1",
                        "node_type": "clip",
                        "time_span": {"start_s": 0, "end_s": 5},
                    },
                    {
                        "node_id": "event:1",
                        "node_type": "event",
                        "text": "A person places a key on the table.",
                        "time_span": {"start_s": 1, "end_s": 3},
                        "provenance": {"created_by": "fixture"},
                    },
                ],
                "edges": [],
            },
            "metadata": {"video_regime": "short"},
        }

        graph, nodes = canonical_to_memory_nodes(canonical)

        self.assertEqual(graph["graph_id"], "clue_memory:example:1")
        self.assertEqual([node.node_id for node in nodes], ["memory:event:1"])
        self.assertEqual(nodes[0].source_node_id, "event:1")
        self.assertEqual(nodes[0].node_type, "observation")
        self.assertEqual(nodes[0].text, "A person places a key on the table.")

    def test_video_skills_l1_gate_requires_materialized_high_grade_graph(self) -> None:
        missing = audit_video_skills_l1({"metadata": {}})
        canonical = {
            "metadata": {
                "clip_schemas": [{"clip_id": "clip:1"}],
                "graph_compose": {
                    "used_deterministic_fallback": False,
                    "execution_trace": [],
                },
                "clue_memory_graph": {
                    "schema_version": "video-skills-relaunch/v0.1",
                    "graph_id": "clue_memory:example:gate",
                    "example_id": "example:gate",
                    "dataset": "video_holmes",
                    "video_regime": "short",
                    "input_mode": "video_only",
                    "clip_policy": {"strategy": "fixed_window"},
                    "nodes": [
                        {
                            "node_id": "event:1",
                            "node_type": "event",
                            "text": "A person opens a door.",
                            "clip_id": "clip:1",
                            "time_span": {"start_s": 0.0, "end_s": 2.0},
                            "producer": "neighbor_vlm_l1_graph_composer",
                            "visibility": {
                                "mode": "video_only",
                                "visible_to_agent": True,
                                "hidden_supervision": False,
                            },
                        },
                        {
                            "node_id": "event:2",
                            "node_type": "event",
                            "text": "The person enters the room.",
                            "clip_id": "clip:1",
                            "time_span": {"start_s": 2.0, "end_s": 4.0},
                            "producer": "neighbor_vlm_l1_graph_composer",
                            "visibility": {
                                "mode": "video_only",
                                "visible_to_agent": True,
                                "hidden_supervision": False,
                            },
                        },
                    ],
                    "edges": [
                        {
                            "edge_id": "edge:1",
                            "src": "event:1",
                            "dst": "event:2",
                            "edge_type": "supports_observation",
                            "confidence": 0.8,
                        }
                    ],
                },
            }
        }

        accepted = audit_video_skills_l1(canonical)
        self_edge_fail = audit_video_skills_l1(
            {
                "metadata": {
                    **canonical["metadata"],
                    "clue_memory_graph": {
                        **canonical["metadata"]["clue_memory_graph"],
                        "edges": [
                            {
                                "edge_id": "edge:self",
                                "src": "event:1",
                                "dst": "event:1",
                                "edge_type": "supports_observation",
                            }
                        ],
                    },
                }
            }
        )

        self.assertEqual(missing.status, "fail")
        self.assertEqual(accepted.status, "pass")
        self.assertEqual(accepted.grade, "high")
        self.assertEqual(self_edge_fail.status, "fail")
        self.assertTrue(
            any("self-edges" in issue for issue in self_edge_fail.issues)
        )

    def test_video_skills_l1_gate_rejects_missing_ids_and_nonfinite_values(self) -> None:
        canonical = {
            "metadata": {
                "clip_schemas": [{"clip_id": "clip:1"}],
                "graph_compose": {
                    "used_deterministic_fallback": False,
                    "execution_trace": [],
                },
                "clue_memory_graph": {
                    "schema_version": "video-skills-relaunch/v0.1",
                    "graph_id": "clue_memory:invalid",
                    "example_id": "example:invalid",
                    "dataset": "video_holmes",
                    "video_regime": "short",
                    "input_mode": "video_only",
                    "clip_policy": {"strategy": "fixed_window"},
                    "nodes": [
                        {
                            "node_type": "event",
                            "clip_id": "clip:1",
                            "text": "A visible event.",
                            "time_span": {"start_s": 0.0, "end_s": float("inf")},
                            "confidence": float("nan"),
                            "producer": "neighbor_vlm_l1_graph_composer",
                        }
                    ],
                    "edges": [
                        {
                            "src": "missing:1",
                            "dst": "missing:2",
                            "edge_type": "supports_observation",
                        }
                    ],
                },
            }
        }

        report = audit_video_skills_l1(canonical)

        self.assertEqual(report.status, "fail")
        self.assertEqual(report.metrics["missing_node_ids"], 1)
        self.assertEqual(report.metrics["missing_edge_ids"], 1)
        self.assertEqual(report.metrics["invalid_timestamps"], 1)
        self.assertEqual(report.metrics["invalid_confidences"], 1)

    def test_relation_belief_rejects_nonfinite_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "probability must be in"):
            RelationBelief(
                edge_id="edge:nan",
                src="event:1",
                dst="event:2",
                relation_probabilities={"same_entity": float("nan")},
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=0.5,
            )

    def test_video_skills_l1_passthrough_does_not_regenerate_observations(self) -> None:
        node = _grounded_node("memory:event:door")
        node.metadata["source_node_type"] = "event"
        node.text = "A person opens a door."

        events = VideoSkillsL1AtomicEventExtractor().extract([node])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].predicate, node.text)
        self.assertTrue(events[0].provenance["video_skills_l1_passthrough"])

    def test_video_skills_l1_structuralizer_projects_entity_and_state_subgraphs(
        self,
    ) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "event:approach",
                    "node_type": "event",
                    "clip_id": "clip:1",
                    "text": "The car approaches the obstacle.",
                },
                {
                    "node_id": "event:stop",
                    "node_type": "event",
                    "clip_id": "clip:2",
                    "text": "The car stops at the obstacle.",
                },
                {
                    "node_id": "entity:car:1",
                    "node_type": "entity_mention",
                    "mention_id": "clip:1:entity:000",
                    "entity_type": "object",
                    "instance_id": "car:shared",
                    "evidence_refs": ["clip:1"],
                    "clip_id": "clip:1",
                    "text": "car",
                    "confidence": 0.95,
                },
                {
                    "node_id": "entity:car:2",
                    "node_type": "entity_mention",
                    "mention_id": "clip:2:entity:000",
                    "entity_type": "object",
                    "instance_id": "car:shared",
                    "evidence_refs": ["clip:2"],
                    "clip_id": "clip:2",
                    "text": "the car",
                    "confidence": 0.94,
                },
                {
                    "node_id": "state:moving",
                    "node_type": "state",
                    "clip_id": "clip:1",
                    "text": "The car is moving.",
                    "confidence": 0.93,
                },
                {
                    "node_id": "state:stopped",
                    "node_type": "state",
                    "clip_id": "clip:2",
                    "text": "The car is stopped.",
                    "confidence": 0.96,
                },
            ],
            "edges": [
                {
                    "edge_id": "edge:same-car",
                    "src": "entity:car:1",
                    "dst": "entity:car:2",
                    "edge_type": "same_object",
                    "identity_verified": True,
                },
                {
                    "edge_id": "edge:state",
                    "src": "event:approach",
                    "dst": "event:stop",
                    "edge_type": "state_change",
                },
            ],
        }
        nodes = [
            MemoryNode(
                node_id="memory:event:approach",
                video_id="video-1",
                time_span=TimeSpan(0, 1),
                provenance={},
                node_type="observation",
                text="The car approaches the obstacle.",
                source_node_id="event:approach",
                source_segments=["event:approach"],
                metadata={"source_node_type": "event", "confidence": 0.9},
            ),
            MemoryNode(
                node_id="memory:event:stop",
                video_id="video-1",
                time_span=TimeSpan(2, 3),
                provenance={},
                node_type="observation",
                text="The car stops at the obstacle.",
                source_node_id="event:stop",
                source_segments=["event:stop"],
                metadata={"source_node_type": "event", "confidence": 0.9},
            ),
        ]

        enriched, report = structuralize_video_skills_l1(graph, nodes)
        events = VideoSkillsL1AtomicEventExtractor().extract(enriched)
        event_nodes, _ = atomic_events_to_graph_nodes(events, l1_nodes=enriched)
        mention_id = events[0].participants[0].mention_id
        belief = _belief(
            event_nodes[0],
            event_nodes[1],
            "transition_support",
            participant_alignment=[
                {
                    "src_mention_id": mention_id,
                    "dst_mention_id": mention_id,
                }
            ],
            evidence_quotes={
                "src": events[0].predicate,
                "dst": events[1].predicate,
            },
            dependency_basis="state_continuity",
        )

        self.assertEqual(report.events_with_participants, 2)
        self.assertEqual(report.events_with_states, 2)
        self.assertEqual(
            events[0].participants[0].mention_id,
            events[1].participants[0].mention_id,
        )
        self.assertEqual(events[0].states[0].attribute, "motion_state")
        self.assertEqual(events[1].states[0].attribute, "motion_state")
        self.assertTrue(verify_relation(belief, event_nodes[0], event_nodes[1]).passed)

    def test_identity_tracks_reject_component_and_motion_conflicts(self) -> None:
        nodes = {
            "person:a": {
                "node_id": "person:a",
                "entity_type": "person",
                "attributes": {"clothing": "red coat"},
            },
            "person:b": {
                "node_id": "person:b",
                "entity_type": "person",
                "attributes": {"clothing": "red coat"},
            },
            "person:c": {
                "node_id": "person:c",
                "entity_type": "person",
                "attributes": {"clothing": "blue coat"},
            },
            "car:a": {
                "node_id": "car:a",
                "entity_type": "vehicle",
                "instance_id": "car-1",
                "time_span": {"start_s": 0, "end_s": 2},
            },
            "car:b": {
                "node_id": "car:b",
                "entity_type": "vehicle",
                "instance_id": "car-2",
                "time_span": {"start_s": 1, "end_s": 3},
            },
            "walker:a": {
                "node_id": "walker:a",
                "entity_type": "person",
                "position_m": [0, 0],
                "max_speed_mps": 2,
                "time_span": {"start_s": 0, "end_s": 1},
            },
            "walker:b": {
                "node_id": "walker:b",
                "entity_type": "person",
                "position_m": [100, 0],
                "max_speed_mps": 2,
                "time_span": {"start_s": 2, "end_s": 3},
            },
            "object:a": {"node_id": "object:a", "entity_type": "object"},
        }
        for node_id, node in nodes.items():
            node.update(
                {
                    "node_type": "entity_mention",
                    "mention_id": f"mention:{node_id}",
                    "evidence_refs": [f"clip:{node_id}"],
                }
            )
        edges = [
            {
                "edge_id": "accept:ab",
                "src": "person:a",
                "dst": "person:b",
                "edge_type": "same_entity",
                "identity_verified": True,
            },
            {
                "edge_id": "reject:component",
                "src": "person:b",
                "dst": "person:c",
                "edge_type": "same_entity",
                "identity_verified": True,
            },
            {
                "edge_id": "reject:simultaneous",
                "src": "car:a",
                "dst": "car:b",
                "edge_type": "same_object",
                "identity_verified": True,
            },
            {
                "edge_id": "reject:motion",
                "src": "walker:a",
                "dst": "walker:b",
                "edge_type": "same_entity",
                "identity_verified": True,
            },
            {
                "edge_id": "reject:type",
                "src": "person:a",
                "dst": "object:a",
                "edge_type": "same_entity",
                "identity_verified": True,
            },
        ]

        tracks, report = build_identity_tracks(nodes, edges)
        reasons = {
            row["edge_id"]: " ".join(row["reasons"])
            for row in report.rejected_edges
        }

        self.assertEqual(tracks["person:a"], tracks["person:b"])
        self.assertNotEqual(tracks["person:a"], tracks["person:c"])
        self.assertIn("stable attribute conflict", reasons["reject:component"])
        self.assertIn("simultaneous distinct", reasons["reject:simultaneous"])
        self.assertIn("impossible displacement", reasons["reject:motion"])
        self.assertIn("entity type conflict", reasons["reject:type"])

    def test_unverified_identity_hint_does_not_create_track(self) -> None:
        nodes = {
            "entity:1": {
                "node_id": "entity:1",
                "node_type": "entity_mention",
                "mention_id": "mention:1",
                "entity_type": "person",
                "evidence_refs": ["clip:1"],
            },
            "entity:2": {
                "node_id": "entity:2",
                "node_type": "entity_mention",
                "mention_id": "mention:2",
                "entity_type": "person",
                "evidence_refs": ["clip:2"],
            },
        }
        tracks, report = build_identity_tracks(
            nodes,
            [
                {
                    "edge_id": "hint:1",
                    "src": "entity:1",
                    "dst": "entity:2",
                    "edge_type": "reappears",
                }
            ],
        )

        self.assertNotEqual(tracks["entity:1"], tracks["entity:2"])
        self.assertEqual(report.accepted_edge_ids, ())
        self.assertIn("lacks explicit verifier", report.rejected_edges[0]["reasons"][0])

    def test_identity_attributes_reject_only_explicit_contradictions(self) -> None:
        nodes = {
            "projector:a": {
                "node_id": "projector:a",
                "node_type": "entity_mention",
                "mention_id": "mention:projector:a",
                "entity_type": "object",
                "attributes": {
                    "color": "black and silver",
                    "material": "wood/metal",
                    "role": "projecting device",
                    "size": "large in frame",
                },
                "evidence_refs": ["clip:a"],
            },
            "projector:b": {
                "node_id": "projector:b",
                "node_type": "entity_mention",
                "mention_id": "mention:projector:b",
                "entity_type": "object",
                "attributes": {
                    "color": "dark gray/black",
                    "material": "metal",
                    "role": "light source",
                    "size": "close-up",
                },
                "evidence_refs": ["clip:b"],
            },
        }
        edges = [
            {
                "edge_id": "compatible-description",
                "src": "projector:a",
                "dst": "projector:b",
                "edge_type": "same_object",
                "confidence": 0.9,
            }
        ]

        verified, report = verify_identity_candidates(nodes, edges)

        self.assertFalse(verified[0].get("identity_verified", False))
        self.assertEqual(len(report.rejected), 0)
        self.assertEqual(len(report.targeted_reread_queue), 1)

    def test_identity_verifier_requires_instance_or_grounded_reread(self) -> None:
        nodes = {
            "box:1": {
                "node_id": "box:1",
                "node_type": "entity_mention",
                "mention_id": "mention:box:1",
                "entity_type": "object",
                "attributes": {"color": "red", "material": "wood"},
                "evidence_refs": ["clip:1"],
            },
            "box:2": {
                "node_id": "box:2",
                "node_type": "entity_mention",
                "mention_id": "mention:box:2",
                "entity_type": "object",
                "attributes": {"color": "red", "material": "wood"},
                "evidence_refs": ["clip:2"],
            },
        }
        edges = [
            {
                "edge_id": "candidate",
                "src": "box:1",
                "dst": "box:2",
                "edge_type": "same_object",
                "confidence": 0.9,
            }
        ]

        verified, report = verify_identity_candidates(nodes, edges)
        self.assertFalse(verified[0].get("identity_verified", False))
        self.assertEqual(len(report.targeted_reread_queue), 1)

        edges[0]["targeted_reread"] = {
            "passed": True,
            "src_evidence_ref": "frame:1",
            "dst_evidence_ref": "frame:2",
            "matched_attributes": ["wood grain"],
            "conflicts": [],
            "annotator": "visual-reviewer",
            "labels_source": "raw_video_verifier",
        }
        verified, report = verify_identity_candidates(nodes, edges)
        self.assertTrue(verified[0]["identity_verified"])
        self.assertEqual(report.accepted[0]["method"], "targeted_raw_video_reread")

        untrusted = [dict(edges[0])]
        untrusted[0]["targeted_reread"] = {
            **untrusted[0]["targeted_reread"],
            "labels_source": "graph_llm",
        }
        verified, report = verify_identity_candidates(nodes, untrusted)
        self.assertFalse(verified[0].get("identity_verified", False))
        self.assertEqual(len(report.targeted_reread_queue), 1)

    def test_identity_reread_packet_requires_grounded_visual_decision(self) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "box:1",
                    "node_type": "entity_mention",
                    "mention_id": "mention:box:1",
                    "entity_type": "object",
                    "attributes": {"color": "red"},
                    "clip_id": "clip:1",
                    "time_span": {"start_s": 0, "end_s": 1},
                    "text": "red box",
                    "evidence_refs": ["clip:1"],
                },
                {
                    "node_id": "box:2",
                    "node_type": "entity_mention",
                    "mention_id": "mention:box:2",
                    "entity_type": "object",
                    "attributes": {"color": "red"},
                    "clip_id": "clip:2",
                    "time_span": {"start_s": 2, "end_s": 3},
                    "text": "red box",
                    "evidence_refs": ["clip:2"],
                },
            ],
            "edges": [
                {
                    "edge_id": "same-box",
                    "src": "box:1",
                    "dst": "box:2",
                    "edge_type": "same_object",
                    "confidence": 0.9,
                }
            ],
        }
        packet = prepare_identity_reread_packet(
            graph,
            video_path="/datasets/video.mp4",
        )
        self.assertEqual(len(packet["items"]), 1)
        self.assertEqual(packet["items"][0]["edge_ids"], ["same-box"])
        self.assertEqual(packet["source_video_path"], "/datasets/video.mp4")
        packet["annotator"] = "visual-reviewer"
        packet["labels_source"] = "raw_video_verifier"
        packet["items"][0]["annotation"] = {
            "passed": True,
            "src_evidence_ref": "frame:clip1:10",
            "dst_evidence_ref": "frame:clip2:20",
            "matched_attributes": ["same red corner mark"],
            "conflicts": [],
            "reason": "Distinctive mark is visible in both frames.",
        }
        applied = apply_identity_reread_packet(graph, packet)
        verified, report = verify_identity_candidates(
            {node["node_id"]: node for node in applied["nodes"]},
            applied["edges"],
        )
        self.assertTrue(verified[0]["identity_verified"])
        self.assertEqual(report.accepted[0]["method"], "targeted_raw_video_reread")

        l1_nodes = [_grounded_node("memory:box:1"), _grounded_node("memory:box:2")]
        for memory_node, source_id in zip(l1_nodes, ("box:1", "box:2")):
            memory_node.source_node_id = source_id
        relations, _ = materialize_l1_structural_relations(applied, l1_nodes)
        self.assertEqual(
            relations[0].relation_probabilities,
            {"same_object": 0.9},
        )
        self.assertEqual(
            relations[0].provenance["admission_tier"],
            "verified_identity",
        )

        stale_packet = dict(packet)
        stale_packet["graph_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "does not match"):
            apply_identity_reread_packet(graph, stale_packet)

        canonical = {"metadata": {"clue_memory_graph": graph}, "question": {}}
        applied_canonical = apply_identity_reread_to_artifact(canonical, packet)
        self.assertTrue(
            applied_canonical["metadata"]["clue_memory_graph"]["edges"][0][
                "targeted_reread"
            ]["passed"]
        )
        self.assertNotIn("targeted_reread", graph["edges"][0])

    def test_native_l1_relations_remain_navigation_only(self) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "event:car:1",
                    "node_type": "event",
                    "clip_id": "clip:1",
                },
                {
                    "node_id": "event:car:2",
                    "node_type": "event",
                    "clip_id": "clip:2",
                },
                {
                    "node_id": "event:car:3",
                    "node_type": "event",
                    "clip_id": "clip:3",
                },
            ],
            "edges": [
                {
                    "edge_id": "edge:same-car",
                    "src": "event:car:1",
                    "dst": "event:car:2",
                    "edge_type": "same_object",
                    "confidence": 0.91,
                },
                {
                    "edge_id": "edge:support",
                    "src": "event:car:1",
                    "dst": "event:car:3",
                    "edge_type": "supports_observation",
                    "confidence": 0.88,
                },
                {
                    "edge_id": "edge:state",
                    "src": "event:car:1",
                    "dst": "event:car:2",
                    "edge_type": "state_change",
                    "confidence": 0.87,
                },
                {
                    "edge_id": "edge:hint",
                    "src": "event:car:1",
                    "dst": "event:car:2",
                    "edge_type": "causal_hint",
                    "confidence": 0.99,
                },
            ],
        }
        l1_src = _grounded_node("memory:event:car:1")
        l1_src.metadata.update(
            {"source_node_type": "event", "clip_id": "clip:1"}
        )
        l1_src.source_node_id = "event:car:1"
        l1_dst = _grounded_node(
            "memory:event:car:2",
            start_s=3,
            end_s=5,
        )
        l1_dst.metadata.update(
            {"source_node_type": "event", "clip_id": "clip:2"}
        )
        l1_dst.source_node_id = "event:car:2"
        l1_support = _grounded_node(
            "memory:event:car:3",
            start_s=6,
            end_s=8,
        )
        l1_support.metadata.update(
            {"source_node_type": "event", "clip_id": "clip:3"}
        )
        l1_support.source_node_id = "event:car:3"

        relations, report = materialize_l1_structural_relations(
            graph,
            [l1_src, l1_dst, l1_support],
        )

        probs = {
            next(iter(relation.relation_probabilities)): relation
            for relation in relations
        }
        self.assertEqual(len(relations), 2)
        self.assertEqual(
            probs["same_instance_candidate"].relation_probabilities,
            {"same_instance_candidate": 0.91},
        )
        self.assertEqual(
            probs["observation_support"].relation_probabilities,
            {"observation_support": 0.88},
        )
        self.assertEqual(
            probs["same_instance_candidate"].provenance["source_edge_type"],
            "same_object",
        )
        self.assertEqual(
            probs["same_instance_candidate"].provenance["endpoint_resolution"]["src"],
            "direct_source_node",
        )
        self.assertTrue(relations[0].provenance["navigation_only"])
        self.assertEqual(
            report.relation_counts,
            {"observation_support": 1, "same_instance_candidate": 1},
        )
        self.assertIn("edge:state", report.skipped_state_change_ids)

    def test_native_l1_endpoint_resolution_rejects_clip_only_fallback(self) -> None:
        graph = {
            "nodes": [
                {
                    "node_id": "entity:car:1",
                    "node_type": "entity_mention",
                    "clip_id": "clip:1",
                    "text": "car",
                },
                {
                    "node_id": "event:car:1",
                    "node_type": "event",
                    "clip_id": "clip:1",
                    "text": "A car approaches.",
                },
                {
                    "node_id": "event:car:2",
                    "node_type": "event",
                    "clip_id": "clip:2",
                    "text": "A car stops.",
                },
            ],
            "edges": [
                {
                    "edge_id": "edge:same-car",
                    "src": "entity:car:1",
                    "dst": "event:car:2",
                    "edge_type": "same_object",
                    "confidence": 0.9,
                }
            ],
        }
        l1_event = _grounded_node("memory:event:car:1")
        l1_event.source_node_id = "event:car:1"
        l1_event.metadata.update(
            {"source_node_type": "event", "clip_id": "clip:1"}
        )
        l1_dst = _grounded_node("memory:event:car:2", start_s=3, end_s=5)
        l1_dst.source_node_id = "event:car:2"
        l1_dst.metadata.update(
            {"source_node_type": "event", "clip_id": "clip:2"}
        )

        relations, report = materialize_l1_structural_relations(
            graph,
            [l1_event, l1_dst],
        )

        self.assertEqual(relations, [])
        self.assertEqual(report.unresolved_edge_ids, ("edge:same-car",))

    def test_causal_edges_require_visual_when_causal_gate_closed(self) -> None:
        src = _grounded_node("event:src")
        dst = _grounded_node("event:dst", start_s=3, end_s=5)
        proposal = RelationBelief(
            edge_id="edge:causal",
            src=src.node_id,
            dst=dst.node_id,
            relation_probabilities={"explains": 0.9},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.9,
            evidence_refs=[src.node_id, dst.node_id],
            provenance={},
        )

        accepted, rejected, summary = _verify_proposals(
            [proposal],
            event_nodes=[src, dst],
            l1_nodes=[src, dst],
            minimum_probability=0.5,
            relation_thresholds={},
            enabled=False,
            require_visual_verification=True,
        )

        self.assertEqual(accepted, [])
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0]["relation"], "explains")
        self.assertTrue(summary["visual_verification_required"])

    def test_navigation_reads_native_l1_identity_without_event_causality(
        self,
    ) -> None:
        l1_src = _grounded_node("l1:event:car:1")
        l1_dst = _grounded_node(
            "l1:event:car:2",
            start_s=3,
            end_s=5,
        )
        relation = RelationBelief(
            edge_id="l1-structural:car",
            src=l1_src.node_id,
            dst=l1_dst.node_id,
            relation_probabilities={"same_entity": 0.9},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.5,
            evidence_refs=[l1_src.node_id, l1_dst.node_id],
            provenance={"navigation_only": True},
        )
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:l1-navigation",
            example_id="example:l1-navigation",
            video_id="video-1",
            l1_observations=[l1_src, l1_dst],
            atomic_events=[],
            relations=[],
            l1_structural_relations=[relation],
        )
        belief = NavigationBeliefState(
            question="Where did the car go?",
            acquired_evidence=(l1_src.node_id,),
            frontier=(l1_src.node_id,),
        )

        actions = propose_navigation_actions(belief, overlay)
        track = next(
            action
            for action in actions
            if action.action_type is NavigationActionType.TRACK_ENTITY
        )
        observations = execute_real_graph_read(track, overlay)

        self.assertEqual(track.target_ids, (l1_dst.node_id,))
        self.assertEqual([node.node_id for node in observations], [l1_dst.node_id])
        self.assertFalse(overlay.relations)

    def test_temporal_skeleton_uses_explicit_intervals(self) -> None:
        nodes = [
            _node("a", 0, 2),
            _node("b", 3, 5),
            _node("c", 4, 6),
            _node("d", 4.2, 4.8),
        ]

        graph = build_memory_graph(
            graph_id="memory_graph:1",
            example_id="example:1",
            video_id="video-1",
            nodes=nodes,
        )
        relations = {
            (edge.src, edge.dst, next(iter(edge.relation_probabilities)))
            for edge in graph.relations
        }

        self.assertIn(("a", "b", "temporal_next"), relations)
        self.assertIn(("a", "b", "before"), relations)
        self.assertIn(("b", "c", "overlaps"), relations)
        self.assertIn(("d", "c", "during"), relations)
        self.assertTrue(all(edge.status is RelationStatus.DETERMINISTIC for edge in graph.relations))

    def test_probabilistic_relations_require_explicit_scorer_weights(self) -> None:
        nodes = [_node("a", 0, 2), _node("b", 3, 5), _node("c", 8, 10)]
        scorer = LinearRelationScorer(
            weights={
                "same_entity": {"cosine_similarity": 2.0},
                "state_transition": {"cosine_similarity": 1.0, "temporal_proximity": 1.0},
                "explains": {"temporal_proximity": 1.0, "forward_order": 1.0},
                "enables": {"temporal_proximity": 0.5},
                "contradicts": {"cosine_similarity": -1.0},
            },
            biases={relation: -0.5 for relation in (
                "same_entity",
                "state_transition",
                "explains",
                "enables",
                "contradicts",
            )},
            calibrated=False,
        )
        graph = build_memory_graph(
            graph_id="memory_graph:1",
            example_id="example:1",
            video_id="video-1",
            nodes=nodes,
            embeddings=[
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
            ],
            relation_scorer=scorer,
            top_k_candidates=1,
        )

        candidates = [edge for edge in graph.relations if edge.status is RelationStatus.UNCALIBRATED_PRIOR]
        self.assertTrue(candidates)
        self.assertTrue(all("explains" in edge.relation_probabilities for edge in candidates))
        self.assertTrue(all(edge.provenance["producer"] == scorer.producer for edge in candidates))

    def test_schema_and_embedding_contract_use_qwen_2b(self) -> None:
        schema_path = Path(__file__).parents[1] / "memory_graph.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        embedding_schema = schema["$defs"]["embedding_ref"]["properties"]
        self.assertEqual(DEFAULT_EMBEDDING_MODEL, "Qwen/Qwen3-VL-Embedding-2B")
        self.assertEqual(DEFAULT_EMBEDDING_DIM, 2048)
        self.assertEqual(embedding_schema["model"]["const"], DEFAULT_EMBEDDING_MODEL)
        self.assertEqual(embedding_schema["dimension"]["const"], DEFAULT_EMBEDDING_DIM)

    def test_overlay_schema_validation_resolves_local_references(self) -> None:
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:1",
            example_id="example:1",
            video_id="video-1",
            l1_observations=[_grounded_node("memory:l1")],
            atomic_events=[
                MemoryNode(
                    node_id="atomic:1",
                    video_id="video-1",
                    time_span=TimeSpan(0.0, 1.0),
                    provenance={"producer": "test"},
                    node_type="atomic_event",
                    text="A visible event.",
                    source_segments=["memory:l1"],
                )
            ],
            relations=[],
            metadata={
                "layer_contract": "l1_observations_plus_l1_5_atomic_overlay"
            },
        )

        self.assertEqual(validate_overlay_artifact(overlay.to_dict()), [])

    def test_overlay_schema_accepts_legacy_build_report_without_candidates(self) -> None:
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:legacy",
            example_id="example:legacy",
            video_id="video-1",
            l1_observations=[_grounded_node("memory:legacy")],
            atomic_events=[],
            relations=[],
            metadata={
                "layer_contract": "l1_observations_plus_l1_5_atomic_overlay"
            },
        ).to_dict()
        overlay["build_report"] = {
            "l1_reliability": {},
            "verifier_summary": {},
            "rejected_relations": [],
        }

        self.assertEqual(validate_overlay_artifact(overlay), [])

    def test_video_l1_coarse_parser_rejects_ungrounded_timestamp(self) -> None:
        accepted, rejected = _parse_coarse_response(
            {
                "events": [
                    {
                        "predicate": "A hand touches the cup.",
                        "coarse_start_s": 4.0,
                        "coarse_end_s": 9.0,
                        "confidence": 0.9,
                    }
                ]
            },
            window={"start_s": 0.0, "end_s": 8.0, "purpose": "coarse_scan"},
            window_index=0,
            minimum_confidence=0.5,
            max_events=4,
        )

        self.assertEqual(accepted, [])
        self.assertIn("outside", rejected[0]["reason"])

    def test_video_l1_fine_localization_tracks_states_and_frame_evidence(self) -> None:
        records = [
            {"frame_index": 0, "time_s": 1.0, "purpose": "fine_localization"},
            {"frame_index": 1, "time_s": 1.5, "purpose": "fine_localization"},
            {"frame_index": 2, "time_s": 2.0, "purpose": "fine_localization"},
        ]
        candidate = {
            "predicate": "A hand lifts the red cup.",
            "action_kind": "state_change",
            "coarse_window": {"start_s": 0.0, "end_s": 3.0},
        }
        event = _parse_fine_response(
            {
                "observed": True,
                "predicate": "A hand lifts the red cup.",
                "confidence": 0.9,
                "visible_start_frame": 0,
                "visible_end_frame": 2,
                "evidence_frames": [0, 1, 2],
                "action_kind": "state_change",
                "participants": [
                    {
                        "role": "patient",
                        "entity_type": "object",
                        "surface": "red cup",
                        "visual_signature": "small red cup with white rim",
                        "confidence": 0.9,
                        "evidence_frames": [0, 2],
                    }
                ],
                "states": [
                    {
                        "participant_index": 0,
                        "attribute": "support",
                        "value": "held in hand",
                        "confidence": 0.8,
                        "evidence_frames": [2],
                    }
                ],
                "state_change": {
                    "participant_index": 0,
                    "attribute": "support",
                    "before": "on table",
                    "after": "held in hand",
                    "confidence": 0.9,
                    "evidence_frames": [0, 2],
                },
            },
            candidate=candidate,
            frame_records=records,
            minimum_confidence=0.5,
        )
        assert event is not None
        track_count = _assign_track_ids([event], max_gap_s=12.0)
        node = _event_to_l1_node(event, video_id="video-1", index=1)
        atomic = VideoL1AtomicEventExtractor().extract([node])

        self.assertEqual(track_count, 1)
        self.assertEqual(node.time_span, TimeSpan(1.0, 2.0))
        self.assertEqual(node.metadata["state_change"]["before"], "on table")
        self.assertEqual(
            node.metadata["participants"][0]["track_status"],
            "candidate_visual_signature",
        )
        self.assertEqual(atomic[0].participants[0].surface, "red cup")
        self.assertEqual(atomic[0].states[0].value, "held in hand")

    def test_video_l1_evaluation_separates_structure_from_human_correctness(self) -> None:
        node = _grounded_node("visual_l1:1")
        node.metadata.update(
            {
                "participants": [
                    {
                        "mention_id": "track:1",
                        "surface": "cup",
                    }
                ],
                "states": [
                    {
                        "mention_id": "track:1",
                        "attribute": "support",
                        "value": "on table",
                    }
                ],
                "localization": {
                    "evidence_frames": [0, 1],
                    "frame_records": [
                        {"frame_index": 0, "time_s": 0.0},
                        {"frame_index": 1, "time_s": 1.0},
                    ],
                },
            }
        )
        packet = build_video_l1_annotation_packet([node])
        packet["events"][0]["visible"] = True
        packet["events"][0]["gold_time_span"] = {
            "start_s": node.time_span.start_s,
            "end_s": node.time_span.end_s,
        }
        packet["state_judgments"][0]["correct"] = True
        report = evaluate_video_l1_human_audit([node], packet)

        self.assertEqual(
            summarize_video_l1([node])["correctness_status"],
            "requires_independent_human_labels",
        )
        self.assertTrue(report["complete"])
        self.assertEqual(report["mean_temporal_iou"], 1.0)
        self.assertEqual(report["visible_state_precision"], 1.0)

    def test_persisted_video_l1_can_be_replayed_without_vlm(self) -> None:
        node = _grounded_node("visual_l1:1")
        node.video_id = "video-1"
        payload = {
            "model": "Qwen/Qwen3.5-9B",
            "duration_s": 10.0,
            "coarse_window_count": 2,
            "coarse_candidate_count": 1,
            "entity_track_count": 0,
            "nodes": [node.to_dict()],
        }

        result = PayloadVideoL1Provider(payload).extract(
            video_path="/unneeded.mp4",
            video_id="video-1",
            observation_end_s=10.0,
        )

        self.assertEqual(len(result.nodes), 1)
        self.assertEqual(result.model, "Qwen/Qwen3.5-9B")

    def test_cli_builds_temporal_graph_without_optional_ml_dependencies(self) -> None:
        canonical = {
            "schema_version": "video-skills-relaunch/v0.1",
            "example_id": "example:cli",
            "dataset": "video_holmes",
            "video": {"video_id": "video-cli", "primary_path": "/tmp/video.mp4"},
            "available_inputs": {"mode": "video_only"},
            "evidence_candidates": [],
            "evidence_index": {
                "index_id": "index:cli",
                "clip_policy": {"strategy": "fixed_window"},
                "nodes": [
                    {
                        "node_id": "event:1",
                        "node_type": "event",
                        "text": "A person enters the room.",
                        "time_span": {"start_s": 0, "end_s": 2},
                        "clip_id": "clip:1",
                        "producer": "neighbor_vlm_l1_graph_composer",
                        "visibility": {
                            "mode": "video_only",
                            "visible_to_agent": True,
                            "hidden_supervision": False,
                        },
                    },
                    {
                        "node_id": "event:2",
                        "node_type": "event",
                        "text": "The person picks up a key.",
                        "time_span": {"start_s": 3, "end_s": 5},
                        "clip_id": "clip:2",
                        "producer": "neighbor_vlm_l1_graph_composer",
                        "visibility": {
                            "mode": "video_only",
                            "visible_to_agent": True,
                            "hidden_supervision": False,
                        },
                    },
                ],
                "edges": [],
            },
            "metadata": {
                "video_regime": "short",
                "clip_schemas": [
                    {"clip_id": "clip:1"},
                    {"clip_id": "clip:2"},
                ],
                "graph_compose": {
                    "used_deterministic_fallback": False,
                    "execution_trace": [],
                },
                "clue_memory_graph": {
                    "schema_version": "video-skills-relaunch/v0.1",
                    "graph_id": "clue_memory:example:cli",
                    "example_id": "example:cli",
                    "dataset": "video_holmes",
                    "video_id": "video-cli",
                    "video_regime": "short",
                    "input_mode": "video_only",
                    "clip_policy": {"strategy": "fixed_window"},
                    "observation_end_s": 5,
                    "nodes": [
                        {
                            "node_id": "event:1",
                            "node_type": "event",
                            "text": "A person enters the room.",
                            "time_span": {"start_s": 0, "end_s": 2},
                            "clip_id": "clip:1",
                            "producer": "neighbor_vlm_l1_graph_composer",
                            "visibility": {
                                "mode": "video_only",
                                "visible_to_agent": True,
                                "hidden_supervision": False,
                            },
                        },
                        {
                            "node_id": "event:2",
                            "node_type": "event",
                            "text": "The person picks up a key.",
                            "time_span": {"start_s": 3, "end_s": 5},
                            "clip_id": "clip:2",
                            "producer": "neighbor_vlm_l1_graph_composer",
                            "visibility": {
                                "mode": "video_only",
                                "visible_to_agent": True,
                                "hidden_supervision": False,
                            },
                        },
                    ],
                    "edges": [
                        {
                            "edge_id": "edge:1",
                            "src": "event:1",
                            "dst": "event:2",
                            "edge_type": "temporal_next",
                        }
                    ],
                },
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            canonical_path = Path(temp_dir) / "canonical.json"
            events_path = Path(temp_dir) / "atomic_events.json"
            output_path = Path(temp_dir) / "graph.json"
            canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
            events_path.write_text(
                json.dumps(
                    {
                        "events": [
                            {
                                "event_id": "atomic:1",
                                "video_id": "video-cli",
                                "time_span": {"start_s": 0, "end_s": 2},
                                "predicate": "A person enters the room.",
                                "evidence_refs": ["memory:event:1"],
                                "confidence": 0.9,
                            },
                            {
                                "event_id": "atomic:2",
                                "video_id": "video-cli",
                                "time_span": {"start_s": 3, "end_s": 5},
                                "predicate": "The person picks up a key.",
                                "evidence_refs": ["memory:event:2"],
                                "confidence": 0.9,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            exit_code = cli_main([
                "--canonical",
                str(canonical_path),
                "--output",
                str(output_path),
                "--atomic-events",
                str(events_path),
                "--input-mode",
                "video_only",
            ])

            output = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(output["schema_version"], "steam-causal-overlay/v0.2")
            self.assertEqual(output["metadata"]["l1_reliability"]["status"], "incomplete")
            self.assertEqual(len(output["l1_observations"]), 2)
            self.assertEqual(len(output["atomic_events"]), 2)
            self.assertEqual(validate_overlay_artifact(output), [])
            self.assertIn("candidate_relations", output["build_report"])
            self.assertIn("rejected_relations", output["build_report"])
            self.assertIn("verifier_summary", output["build_report"])
            self.assertIn(
                "identity_candidates", output["metadata"]["relation_layers"]
            )
            self.assertIn(
                "observation_support", output["metadata"]["relation_layers"]
            )
            self.assertNotEqual(
                output["metadata"]["node_event_assumption"],
                "one_memory_node_approximately_one_event",
            )
            relation_types = {
                next(iter(relation["relation_probabilities"]))
                for relation in output["relations"]
            }
            self.assertIn("temporal_next", relation_types)
            self.assertIn("before", relation_types)

    def test_audit_summary_does_not_treat_plausible_as_strictly_supported(self) -> None:
        summary = _computed_audit_summary(
            {
                "candidate_edge_audit": [
                    {"judgment": "supported"},
                    {"judgment": "plausible"},
                    {"judgment": "unsupported"},
                ]
            },
            expected_predictions=3,
        )

        self.assertAlmostEqual(summary["strict_precision"], 1 / 3)
        self.assertAlmostEqual(summary["supported_or_plausible_rate"], 2 / 3)
        self.assertTrue(summary["audit_complete"])

    def test_l1_reliability_gate_is_incomplete_without_human_labels(self) -> None:
        report = audit_l1_nodes([_grounded_node("memory:event:1")])

        self.assertEqual(report.status, "incomplete")
        self.assertTrue(report.metrics["provenance_completeness"].passed)
        self.assertIsNone(report.metrics["grounded_event_precision"].passed)

    def test_l1_reliability_gate_passes_only_with_complete_independent_labels(self) -> None:
        nodes = [
            _grounded_node("memory:event:1"),
            _grounded_node("memory:event:2", start_s=2, end_s=4),
        ]
        audit = L1HumanAudit(
            grounded_event_correct={node.node_id: True for node in nodes},
            compound_event={node.node_id: False for node in nodes},
            gold_key_event_ids=("gold:1", "gold:2"),
            covered_key_event_ids=("gold:1", "gold:2"),
            entity_link_judgments=(
                EntityLinkJudgment(
                    src="memory:event:1",
                    dst="memory:event:2",
                    correct=True,
                ),
            ),
            annotator="independent-reviewer",
            protocol_version="l1-audit/v0.1",
        )

        report = audit_l1_nodes(nodes, human_audit=audit, observation_end_s=5)

        self.assertEqual(report.status, "pass")
        self.assertTrue(all(metric.passed for metric in report.metrics.values()))

    def test_hidden_supervision_fails_l1_gate(self) -> None:
        node = _grounded_node("memory:event:1")
        node.metadata["visibility"] = "hidden"

        report = audit_l1_nodes([node])

        self.assertEqual(report.status, "fail")
        self.assertFalse(report.metrics["hidden_supervision_leakage"].passed)

    def test_atomic_event_parser_preserves_l1_grounding(self) -> None:
        node = _grounded_node("memory:event:1")
        events = parse_atomic_events(
            {
                "events": [
                    {
                        "event_id": "atomic:1",
                        "video_id": "video-1",
                        "time_span": {"start_s": 0.2, "end_s": 1.2},
                        "predicate": "A person places a key on a table.",
                        "evidence_refs": [node.node_id],
                        "confidence": 0.9,
                        "participants": [
                            {
                                "mention_id": "person:local",
                                "role": "agent",
                                "entity_type": "person",
                                "surface": "a person",
                                "confidence": 0.9,
                                "grounding_refs": [node.node_id],
                            }
                        ],
                        "states": [],
                    }
                ]
            },
            nodes=[node],
            model="fixture-model",
        )

        self.assertEqual(events[0].evidence_refs, (node.node_id,))
        self.assertEqual(events[0].provenance["layer"], "L1.5")

    def test_atomic_event_parser_rejects_causal_narrative(self) -> None:
        node = _grounded_node("memory:event:1")
        payload = {
            "events": [
                {
                    "event_id": "atomic:1",
                    "time_span": {"start_s": 0.2, "end_s": 1.2},
                    "predicate": "The person takes the key because they want to escape.",
                    "evidence_refs": [node.node_id],
                    "confidence": 0.9,
                    "participants": [],
                    "states": [],
                }
            ]
        }

        with self.assertRaisesRegex(ValueError, "failed observation checks"):
            parse_atomic_events(payload, nodes=[node], model="fixture-model")

    def test_one_l1_observation_can_ground_multiple_atomic_events(self) -> None:
        l1 = _grounded_node("memory:event:1", start_s=0, end_s=4)
        participant = EntityMention(
            "person:1", "agent", "person", "the person", 0.9, (l1.node_id,)
        )
        events = [
            AtomicEvent(
                f"atomic:{index}",
                "video-1",
                TimeSpan(start, end),
                predicate,
                (l1.node_id,),
                0.9,
                (participant,),
            )
            for index, start, end, predicate in (
                (1, 0.2, 1.0, "The person opens a box."),
                (2, 2.0, 3.0, "The person removes a key."),
            )
        ]

        nodes, index = atomic_events_to_graph_nodes(events, l1_nodes=[l1])

        self.assertEqual(len(nodes), 2)
        self.assertEqual(index.l1_to_events[l1.node_id], ("event:atomic:1", "event:atomic:2"))
        self.assertTrue(all(node.source_segments == [l1.node_id] for node in nodes))

    def test_dict_visibility_hidden_supervision_fails_l1_gate(self) -> None:
        node = _grounded_node("memory:event:1")
        node.metadata["visibility"] = {
            "visible_to_agent": True,
            "hidden_supervision": True,
        }

        report = audit_l1_nodes([node])

        self.assertEqual(report.status, "fail")
        self.assertFalse(report.metrics["hidden_supervision_leakage"].passed)

    def test_staged_input_excludes_hidden_and_non_segment_expert_nodes(self) -> None:
        visible = _grounded_node("memory:segment")
        visible.metadata.update(
            source_type="segment_description",
            visibility={"visible_to_agent": True, "hidden_supervision": False},
        )
        hidden = _grounded_node("memory:inference")
        hidden.metadata.update(
            source_type="inference_scene",
            visibility={"visible_to_agent": False, "hidden_supervision": True},
        )

        selected = _select_l1_inputs([visible, hidden], input_mode="expert_demo")

        self.assertEqual([node.node_id for node in selected], [visible.node_id])

    def test_regression_9klsIDsGLlw_wrong_entity_alignment_is_rejected(self) -> None:
        src = _atomic_node(
            "event:src",
            0,
            1,
            "A woman starts a timer.",
            mention_id="woman:1",
            surface="a woman",
        )
        dst = _atomic_node(
            "event:dst",
            2,
            3,
            "A man looks at the timer.",
            mention_id="man:1",
            surface="a man",
        )
        belief = _belief(
            src,
            dst,
            "same_entity",
            participant_alignment=[
                {"src_mention_id": "woman:1", "dst_mention_id": "man:1"}
            ],
        )

        result = verify_relation(belief, src, dst)

        self.assertFalse(result.passed)

    def test_valid_state_transition_is_accepted(self) -> None:
        src = _atomic_node(
            "event:src",
            0,
            1,
            "The door is closed.",
            mention_id="l1-track:door",
            entity_type="object",
            surface="the door",
            state=("position", "closed"),
        )
        dst = _atomic_node(
            "event:dst",
            2,
            3,
            "The door is open.",
            mention_id="l1-track:door",
            entity_type="object",
            surface="the door",
            state=("position", "open"),
        )
        belief = _belief(
            src,
            dst,
            "state_transition",
            participant_alignment=[
                {
                    "src_mention_id": "l1-track:door",
                    "dst_mention_id": "l1-track:door",
                }
            ],
        )

        self.assertTrue(verify_relation(belief, src, dst).passed)

    def test_state_transition_rejects_surface_match_without_accepted_track(self) -> None:
        src = _atomic_node(
            "event:closed",
            0,
            1,
            "The door is closed.",
            mention_id="door:before",
            entity_type="object",
            surface="the door",
            state=("position", "closed"),
        )
        dst = _atomic_node(
            "event:open",
            2,
            3,
            "The door is open.",
            mention_id="door:after",
            entity_type="object",
            surface="the door",
            state=("position", "open"),
        )
        belief = _belief(
            src,
            dst,
            "state_transition",
            participant_alignment=[
                {
                    "src_mention_id": "door:before",
                    "dst_mention_id": "door:after",
                }
            ],
        )

        result = verify_relation(belief, src, dst)

        self.assertFalse(result.passed)
        self.assertIn("accepted conflict-aware identity track", result.reasons[0])

    def test_state_transition_is_derived_only_on_accepted_track(self) -> None:
        before = _atomic_node(
            "event:closed-track",
            0,
            1,
            "The door is closed.",
            mention_id="l1-track:door",
            entity_type="object",
            surface="the door",
            state=("openness", "closed"),
        )
        after = _atomic_node(
            "event:open-track",
            2,
            3,
            "The door is open.",
            mention_id="l1-track:door",
            entity_type="object",
            surface="the door",
            state=("openness", "open"),
        )
        candidates = derive_state_transition_candidates([before, after])

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].provenance["accepted_identity_track"],
            "l1-track:door",
        )
        self.assertTrue(verify_relation(candidates[0], before, after).passed)

        after.metadata["participants"][0]["mention_id"] = "door:lookalike"
        after.metadata["states"][0]["mention_id"] = "door:lookalike"
        self.assertEqual(derive_state_transition_candidates([before, after]), [])

    def test_regression_5P_6Q2Q0NLk_timer_narrative_does_not_enable(self) -> None:
        src = _atomic_node(
            "event:src", 0, 1, "A timer starts.", mention_id="timer:1", surface="a timer"
        )
        dst = _atomic_node(
            "event:dst", 2, 3, "A person leaves.", mention_id="person:1", surface="a person"
        )
        belief = _belief(
            src,
            dst,
            "enables",
            evidence_quotes={"src": "A timer starts.", "dst": "A person leaves."},
            minimal_support_set=[src.node_id, dst.node_id],
            mechanism_detail="The first event merely happens before the second event.",
        )

        result = verify_relation(belief, src, dst)

        self.assertFalse(result.passed)

    def test_regression_wardrobe_duplication_mechanism_is_accepted(self) -> None:
        src = _atomic_node(
            "event:src",
            0,
            1,
            "The wardrobe door opens.",
            mention_id="wardrobe:1",
            entity_type="object",
            surface="the wardrobe",
        )
        dst = _atomic_node(
            "event:dst",
            2,
            3,
            "A duplicate exits the wardrobe.",
            mention_id="wardrobe:2",
            entity_type="object",
            surface="the wardrobe",
        )
        belief = _belief(
            src,
            dst,
            "enables",
            evidence_quotes={
                "src": "The wardrobe door opens.",
                "dst": "A duplicate exits the wardrobe.",
            },
            minimal_support_set=[src.node_id, dst.node_id],
            precondition=(
                "The open wardrobe supplies the observable passage required for "
                "the duplicate to exit."
            ),
        )

        self.assertTrue(verify_relation(belief, src, dst).passed)

    def test_causal_edge_rejects_llm_refined_order_inside_same_l1_span(self) -> None:
        src = _atomic_node(
            "event:src",
            1,
            2,
            "The wardrobe door opens.",
            mention_id="wardrobe:1",
            entity_type="object",
            surface="the wardrobe",
        )
        dst = _atomic_node(
            "event:dst",
            3,
            4,
            "A duplicate exits the wardrobe.",
            mention_id="wardrobe:2",
            entity_type="object",
            surface="the wardrobe",
        )
        src.source_segments = ["memory:coarse"]
        dst.source_segments = ["memory:coarse"]
        belief = _belief(
            src,
            dst,
            "enables",
            evidence_quotes={
                "src": "The wardrobe door opens.",
                "dst": "A duplicate exits the wardrobe.",
            },
            minimal_support_set=[src.node_id, dst.node_id],
            precondition="The open wardrobe is the visible passage required to exit.",
        )
        belief.evidence_refs = ["memory:coarse"]
        l1 = _grounded_node("memory:coarse", start_s=0, end_s=10)

        result = verify_relation(
            belief,
            src,
            dst,
            l1_by_id={l1.node_id: l1},
        )

        self.assertFalse(result.passed)
        self.assertTrue(any("unverified timing" in reason for reason in result.reasons))

    def test_regression_7IJb1V1mPfA_mirror_does_not_explain_ghost(self) -> None:
        src = _atomic_node(
            "event:mirror",
            0,
            1,
            "A woman looks into a mirror.",
            mention_id="woman:mirror",
            surface="a woman",
        )
        dst = _atomic_node(
            "event:ghost",
            2,
            3,
            "A ghost appears behind her.",
            mention_id="ghost:1",
            entity_type="person",
            surface="a ghost",
        )
        belief = _belief(
            src,
            dst,
            "explains",
            evidence_quotes={
                "src": "A woman looks into a mirror.",
                "dst": "A ghost appears behind her.",
            },
            minimal_support_set=[src.node_id, dst.node_id],
            mechanism_detail="The mirror scene happens before the ghost appears.",
        )

        self.assertFalse(verify_relation(belief, src, dst).passed)

    def test_causal_witness_allows_explicit_mediator_support(self) -> None:
        src = _atomic_node(
            "event:push",
            0,
            1,
            "A person pushes the door.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:enter",
            3,
            4,
            "The person enters through the open door.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
        )
        witness = CausalWitness(
            witness_id="witness:door",
            relation="enables",
            cause_event_id=src.node_id,
            effect_event_id=dst.node_id,
            mechanism_event_id="event:door-opens",
            mechanism=MechanismKind.STATE_BRIDGE,
            mechanism_detail="The push visibly changes the door to an open state.",
            evidence_refs=tuple(src.source_segments + dst.source_segments),
            minimal_support_set=(
                src.node_id,
                "event:door-opens",
                dst.node_id,
            ),
            evidence_quotes={
                "src": "A person pushes the door.",
                "dst": "The person enters through the open door.",
            },
            state_bridge="The door changes from closed to open.",
        )
        belief = _belief(
            src,
            dst,
            "enables",
            causal_witness=witness.to_dict(),
        )
        belief.warrant = (
            "A person pushes the door. The person enters through the open door."
        )

        self.assertTrue(verify_relation(belief, src, dst).passed)

    def test_mechanism_candidates_start_from_visible_state_delta(self) -> None:
        src = _atomic_node(
            "event:closed",
            0,
            1,
            "The door is closed.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
            state=("openness", "closed"),
        )
        dst = _atomic_node(
            "event:open",
            2,
            3,
            "The door is open.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
            state=("openness", "open"),
        )

        candidates = generate_mechanism_candidates([src, dst])

        self.assertEqual(len(candidates), 1)
        self.assertIn("state_bridge", candidates[0].mechanism_hints)
        self.assertEqual(candidates[0].before_state["value"], "closed")
        self.assertEqual(candidates[0].after_state["value"], "open")

    def test_video_skills_l1_edges_seed_recall_but_not_accepted_causality(
        self,
    ) -> None:
        src = _atomic_node(
            "event:a",
            0,
            1,
            "A person reaches toward a cup.",
            mention_id="person:a",
            surface="person",
        )
        dst = _atomic_node(
            "event:b",
            2,
            3,
            "The cup is lifted.",
            mention_id="cup:b",
            surface="cup",
            entity_type="object",
        )
        l1_src = _grounded_node("memory:source:a")
        l1_src.source_node_id = "source:a"
        l1_dst = _grounded_node("memory:source:b")
        l1_dst.source_node_id = "source:b"
        edge = {
            "edge_id": "l1-edge:1",
            "src": "source:a",
            "dst": "source:b",
            "edge_type": "causal_hint",
        }
        l1_src.metadata["source_l1_edges"] = [edge]
        l1_dst.metadata["source_l1_edges"] = [edge]

        candidates = generate_video_skills_l1_edge_candidates(
            [src, dst],
            l1_nodes=[l1_src, l1_dst],
            event_to_l1={
                src.node_id: (l1_src.node_id,),
                dst.node_id: (l1_dst.node_id,),
            },
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].mechanism_hints,
            ("l1_causal_hint_requires_verification",),
        )
        self.assertTrue(candidates[0].requires_temporal_refine)

    def test_visual_reread_is_recorded_and_can_be_required(self) -> None:
        src = _atomic_node(
            "event:push",
            0,
            1,
            "A person pushes the door.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:open",
            2,
            3,
            "The door opens.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
        )
        witness = CausalWitness(
            witness_id="witness:push-open",
            relation="explains",
            cause_event_id=src.node_id,
            effect_event_id=dst.node_id,
            mechanism=MechanismKind.CONTACT_TRANSFER,
            mechanism_detail="The visible push transfers motion to the door.",
            evidence_refs=tuple(src.source_segments + dst.source_segments),
            minimal_support_set=(src.node_id, dst.node_id),
            evidence_quotes={
                "src": "A person pushes the door.",
                "dst": "The door opens.",
            },
            confidence_components=ConfidenceComponents(temporal_grounding=0.9),
        )
        belief = _belief(
            src,
            dst,
            "explains",
            causal_witness=witness.to_dict(),
        )
        belief.warrant = "A person pushes the door. The door opens."

        class PassingVisualProvider:
            model = "fixture-vlm"

            def verify(self, **_: object) -> VisualVerification:
                return VisualVerification(
                    status="passed",
                    model=self.model,
                    checks={
                        "mechanism_visible": True,
                        "temporal_order_visible": True,
                    },
                )

        proposals, count = _apply_visual_rereads(
            [belief],
            event_nodes=[src, dst],
            video_path="/tmp/fixture.mp4",
            provider=PassingVisualProvider(),
        )
        accepted, rejected, _ = _verify_proposals(
            proposals,
            event_nodes=[src, dst],
            l1_nodes=[],
            minimum_probability=0.5,
            relation_thresholds={},
            enabled=True,
            require_visual_verification=True,
        )

        self.assertEqual(count, 1)
        self.assertEqual(proposals[0].provenance["visual_verification"]["status"], "passed")
        self.assertEqual(len(accepted), 1)
        self.assertEqual(rejected, [])

    def test_teacher_output_is_upgraded_to_structured_causal_witness(self) -> None:
        src = _atomic_node(
            "event:push",
            0,
            1,
            "A person pushes the door.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:open",
            2,
            3,
            "The door opens.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
        )
        payload = {
            "relations": [
                {
                    "src": src.node_id,
                    "dst": dst.node_id,
                    "probabilities": {
                        "same_entity": 0.8,
                        "state_transition": 0.6,
                        "explains": 0.9,
                        "enables": 0.4,
                        "contradicts": 0.0,
                    },
                    "direction_confidence": 0.95,
                    "warrant": "A person pushes the door. The door opens.",
                    "mention_alignment": [
                        {
                            "src_mention": "door:before",
                            "dst_mention": "door:after",
                        }
                    ],
                    "evidence_quotes": {
                        "src": "A person pushes the door.",
                        "dst": "The door opens.",
                    },
                    "minimal_support_set": [src.node_id, dst.node_id],
                    "mechanism": "contact_transfer",
                    "mechanism_detail": (
                        "The visible push transfers motion to the door."
                    ),
                    "alternative_explanations": [],
                }
            ]
        }

        relations = _parse_teacher_relations(
            payload,
            nodes=[src, dst],
            expected_pairs={(src.node_id, dst.node_id)},
            model="fixture-teacher",
        )

        witness = relations[0].provenance["causal_witness"]
        self.assertEqual(witness["relation"], "explains")
        self.assertEqual(witness["mechanism"], "contact_transfer")
        self.assertEqual(witness["cause_event_id"], src.node_id)

    def test_visual_evidence_indices_enforce_frame_roles_and_time_direction(self) -> None:
        records = [
            {"frame_index": 0, "time_s": 10.0, "purpose": "cause"},
            {"frame_index": 1, "time_s": 20.0, "purpose": "mechanism"},
            {"frame_index": 2, "time_s": 30.0, "purpose": "effect"},
        ]
        checks = {
            "cause_visible": True,
            "effect_visible": True,
            "entity_continuity": True,
            "mechanism_visible": True,
            "temporal_order_visible": True,
        }

        valid = _evidence_grounding_problems(
            records,
            evidence={
                "cause": (0,),
                "effect": (2,),
                "mechanism": (1,),
                "entity": (0, 2),
            },
            checks=checks,
        )
        reversed_roles = _evidence_grounding_problems(
            records,
            evidence={
                "cause": (2,),
                "effect": (0,),
                "mechanism": (1,),
                "entity": (0, 2),
            },
            checks=checks,
        )

        self.assertEqual(valid, [])
        self.assertTrue(any("cause-window" in value for value in reversed_roles))
        self.assertTrue(any("effect-window" in value for value in reversed_roles))

    def test_observable_precondition_cannot_restate_later_perception(self) -> None:
        src = _atomic_node(
            "event:door-closes",
            0,
            1,
            "The door closes.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:man-sees",
            2,
            3,
            "The man sees the closed door.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
        )
        witness = CausalWitness(
            witness_id="witness:tautology",
            relation="explains",
            cause_event_id=src.node_id,
            effect_event_id=dst.node_id,
            mechanism=MechanismKind.OBSERVABLE_PRECONDITION,
            mechanism_detail="The closed state is later visible to the man.",
            evidence_refs=tuple(src.source_segments + dst.source_segments),
            minimal_support_set=(src.node_id, dst.node_id),
            evidence_quotes={
                "src": "The door closes.",
                "dst": "The man sees the closed door.",
            },
            precondition="The door is closed before it is seen.",
        )
        belief = _belief(
            src,
            dst,
            "explains",
            causal_witness=witness.to_dict(),
        )
        belief.warrant = "The door closes. The man sees the closed door."

        result = verify_relation(belief, src, dst)

        self.assertFalse(result.passed)
        self.assertTrue(any("restates" in reason for reason in result.reasons))

    def test_transition_support_is_grounded_but_not_claimed_as_causal(self) -> None:
        src = _atomic_node(
            "event:paper-ground",
            0,
            1,
            "The crumpled paper lies on the grass.",
            mention_id="paper:before",
            surface="crumpled paper",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:paper-picked",
            2,
            3,
            "A boy picks up the crumpled paper.",
            mention_id="paper:after",
            surface="crumpled paper",
            entity_type="object",
        )
        belief = _belief(
            src,
            dst,
            "transition_support",
            participant_alignment=[
                {
                    "src_mention_id": "paper:before",
                    "dst_mention_id": "paper:after",
                }
            ],
            evidence_quotes={
                "src": "The crumpled paper lies on the grass.",
                "dst": "A boy picks up the crumpled paper.",
            },
            dependency_basis="entity_trajectory",
        )

        result = verify_relation(belief, src, dst)

        self.assertTrue(result.passed)
        self.assertEqual(result.relation, "transition_support")
        self.assertNotIn("explains", belief.relation_probabilities)

    def test_response_candidate_rejects_generic_temporal_adjacency(self) -> None:
        src = _atomic_node(
            "event:walk",
            0,
            1,
            "A person walks across the room.",
            mention_id="person:1",
            surface="a person",
        )
        dst = _atomic_node(
            "event:lamp",
            2,
            3,
            "A lamp is visible.",
            mention_id="lamp:1",
            surface="a lamp",
            entity_type="object",
        )
        belief = _belief(
            src,
            dst,
            "response_candidate",
            evidence_quotes={
                "src": "A person walks across the room.",
                "dst": "A lamp is visible.",
            },
            dependency_basis="none",
        )

        self.assertFalse(verify_relation(belief, src, dst).passed)

    def test_world_model_navigation_follows_dependency_then_reads_real_node(self) -> None:
        src = _atomic_node(
            "event:paper-ground",
            0,
            1,
            "The crumpled paper lies on the grass.",
            mention_id="paper:before",
            surface="crumpled paper",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:paper-picked",
            2,
            3,
            "A boy picks up the crumpled paper.",
            mention_id="paper:after",
            surface="crumpled paper",
            entity_type="object",
        )
        relation = _belief(
            src,
            dst,
            "transition_support",
            participant_alignment=[
                {
                    "src_mention_id": "paper:before",
                    "dst_mention_id": "paper:after",
                }
            ],
            evidence_quotes={
                "src": "The crumpled paper lies on the grass.",
                "dst": "A boy picks up the crumpled paper.",
            },
            dependency_basis="entity_trajectory",
        )
        l1_src = _grounded_node(src.source_segments[0])
        l1_dst = _grounded_node(dst.source_segments[0])
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:navigation",
            example_id="example:navigation",
            video_id="video-1",
            l1_observations=[l1_src, l1_dst],
            atomic_events=[src, dst],
            relations=[relation],
        )
        belief = NavigationBeliefState(
            question="Where did the paper go?",
            acquired_evidence=(src.node_id,),
            frontier=(src.node_id,),
        )

        actions = propose_navigation_actions(belief, overlay)
        follow = next(
            action
            for action in actions
            if action.action_type is NavigationActionType.FOLLOW_DEPENDENCY
        )
        selected, predictions = plan_next_read(
            belief,
            overlay,
            RuleBasedDependencyWorldModel(),
        )
        observations = execute_real_graph_read(follow, overlay)
        updated = update_belief_after_read(belief, follow, observations)

        self.assertEqual(follow.target_ids, (dst.node_id,))
        self.assertTrue(predictions)
        self.assertNotIn(dst.node_id, belief.acquired_evidence)
        self.assertIn(dst.node_id, updated.acquired_evidence)
        self.assertIsNotNone(selected)

    def test_selectstream_policy_protects_complete_causal_witness(self) -> None:
        src = _atomic_node(
            "event:push",
            0,
            1,
            "A person pushes the door.",
            mention_id="door:before",
            surface="the door",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:open",
            2,
            3,
            "The door opens.",
            mention_id="door:after",
            surface="the door",
            entity_type="object",
        )
        distractor = _atomic_node(
            "event:background",
            4,
            5,
            "A lamp is visible.",
            mention_id="lamp:1",
            surface="a lamp",
            entity_type="object",
        )
        witness = CausalWitness(
            witness_id="witness:protected",
            relation="explains",
            cause_event_id=src.node_id,
            effect_event_id=dst.node_id,
            mechanism=MechanismKind.CONTACT_TRANSFER,
            mechanism_detail="The visible push transfers motion to the door.",
            evidence_refs=tuple(src.source_segments + dst.source_segments),
            minimal_support_set=(src.node_id, dst.node_id),
            evidence_quotes={
                "src": "A person pushes the door.",
                "dst": "The door opens.",
            },
        )
        relation = _belief(
            src,
            dst,
            "explains",
            causal_witness=witness.to_dict(),
        )

        decision = plan_bounded_memory(
            [src, dst, distractor],
            [relation],
            capacity=2,
            semantic_relevance={distractor.node_id: 1.0},
        )

        self.assertEqual(set(decision.keep), {src.node_id, dst.node_id})
        self.assertEqual(decision.evict, (distractor.node_id,))
        self.assertFalse(
            merge_preserves_causal_witness(
                {src.node_id},
                relations=[relation],
            )
        )
        self.assertTrue(
            merge_preserves_causal_witness(
                {src.node_id, dst.node_id},
                relations=[relation],
            )
        )

    def test_selectstream_values_predictive_dependency_without_calling_it_causal(
        self,
    ) -> None:
        src = _atomic_node(
            "event:paper-ground",
            0,
            1,
            "The paper lies on grass.",
            mention_id="paper:1",
            surface="paper",
            entity_type="object",
        )
        dst = _atomic_node(
            "event:paper-picked",
            2,
            3,
            "A boy picks up the paper.",
            mention_id="paper:2",
            surface="paper",
            entity_type="object",
        )
        background = _atomic_node(
            "event:sky",
            4,
            5,
            "The sky is visible.",
            mention_id="sky:1",
            surface="sky",
            entity_type="object",
        )
        dependency = _belief(
            src,
            dst,
            "transition_support",
            participant_alignment=[
                {
                    "src_mention_id": "paper:1",
                    "dst_mention_id": "paper:2",
                }
            ],
            evidence_quotes={
                "src": "The paper lies on grass.",
                "dst": "A boy picks up the paper.",
            },
            dependency_basis="entity_trajectory",
        )

        decision = plan_bounded_memory(
            [src, dst, background],
            [dependency],
            capacity=1,
            semantic_relevance={background.node_id: 0.5},
        )

        self.assertNotIn(background.node_id, decision.keep)
        kept = decision.keep[0]
        self.assertGreater(
            decision.utility_components[kept]["predictive_dependency"],
            0.0,
        )
        self.assertNotIn("causal_witness", dependency.provenance)

    def test_navigation_ablation_keeps_candidate_and_verified_edges_separate(self) -> None:
        events = [
            _atomic_node(
                "event:seed",
                0,
                1,
                "A person sees a door.",
                mention_id="person:1",
                surface="a person",
            ),
            _atomic_node(
                "event:background",
                2,
                3,
                "The person walks slowly.",
                mention_id="person:1",
                surface="a person",
            ),
            _atomic_node(
                "event:native-target",
                4,
                5,
                "A key is behind the door.",
                mention_id="key:1",
                surface="a key",
                entity_type="object",
            ),
            _atomic_node(
                "event:verified-target",
                6,
                7,
                "The verified bridge reveals the exit.",
                mention_id="exit:1",
                surface="the exit",
                entity_type="object",
            ),
        ]
        l1_nodes = [_grounded_node(ref) for event in events for ref in event.source_segments]
        temporal = RelationBelief(
            edge_id="temporal:seed-background",
            src="event:seed",
            dst="event:background",
            relation_probabilities={"temporal_next": 1.0},
            status=RelationStatus.DETERMINISTIC,
            direction_confidence=1.0,
        )
        dependency = RelationBelief(
            edge_id="dependency:verified",
            src="event:seed",
            dst="event:verified-target",
            relation_probabilities={"transition_support": 0.9},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.9,
            provenance={
                "hard_verifier": {"transition_support": {"passed": True}}
            },
        )
        native = RelationBelief(
            edge_id="native:candidate",
            src=events[0].source_segments[0],
            dst=events[2].source_segments[0],
            relation_probabilities={"same_instance_candidate": 0.8},
            status=RelationStatus.UNCALIBRATED_PRIOR,
            direction_confidence=0.5,
            provenance={"navigation_only": True},
        )
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:ablation",
            example_id="example:ablation",
            video_id="video-1",
            l1_observations=l1_nodes,
            atomic_events=events,
            relations=[temporal, dependency],
            l1_structural_relations=[native],
        )
        report = evaluate_navigation_ablation(
            overlay.to_dict(),
            [
                {
                    "case_id": "native",
                    "question": "What door key does the person see?",
                    "gold_event_ids": ["event:native-target"],
                    "graph_read_budget": 2,
                },
                {
                    "case_id": "verified",
                    "question": "What verified exit does the person sees?",
                    "gold_event_ids": ["event:verified-target"],
                    "graph_read_budget": 2,
                },
            ],
        )

        native_rows = report["cases"]["native_l1_candidate"]
        verified_rows = report["cases"]["verified_dependency"]
        self.assertTrue(native_rows[0]["answerable"])
        self.assertFalse(native_rows[1]["answerable"])
        self.assertFalse(verified_rows[0]["answerable"])
        self.assertTrue(verified_rows[1]["answerable"])

    def test_calibration_requires_independent_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            sample = Path(temp_dir) / "sample"
            sample.mkdir()
            graph = {
                "relations": [
                    {
                        "src": "event:a",
                        "dst": "event:b",
                        "relation_probabilities": {"enables": 0.9},
                    },
                    {
                        "src": "event:c",
                        "dst": "event:d",
                        "relation_probabilities": {"enables": 0.2},
                    },
                ]
            }
            audit = {
                "candidate_edge_audit": [
                    {
                        "src": "event:a",
                        "dst": "event:b",
                        "relation": "enables",
                        "judgment": "supported",
                    },
                    {
                        "src": "event:c",
                        "dst": "event:d",
                        "relation": "enables",
                        "judgment": "unsupported",
                    },
                    {
                        "src": "event:e",
                        "dst": "event:f",
                        "relation": "enables",
                        "judgment": "plausible",
                    },
                ]
            }
            (sample / "causal_temporal_overlay.json").write_text(
                json.dumps(graph), encoding="utf-8"
            )
            (sample / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (sample / "independent_audit.json").write_text(
                json.dumps(audit), encoding="utf-8"
            )

            self_audit = calibrate(
                Path(temp_dir), "gpt_self_audit", min_samples_per_class=1
            )
            human = calibrate(
                Path(temp_dir), "independent_human", min_samples_per_class=1
            )

        self.assertFalse(self_audit["calibrated"])
        self.assertTrue(human["calibrated"])
        self.assertEqual(human["relations"]["enables"]["sample_counts"]["used"], 2)

    def test_baseline_manifest_hashes_artifacts_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            artifact = root / "artifact.json"
            artifact.write_text('{"ok": true}\n', encoding="utf-8")
            original = artifact.read_bytes()
            manifest = freeze_baseline(
                {"smoke": artifact},
                repository=Path(__file__).resolve().parents[2],
                note="test baseline",
            )
            verification = verify_baseline(
                manifest,
                repository=Path(__file__).resolve().parents[2],
            )
            self.assertTrue(verification["passed"])
            artifact.write_text('{"ok": false}\n', encoding="utf-8")
            drift = verify_baseline(
                manifest,
                repository=Path(__file__).resolve().parents[2],
            )
            self.assertFalse(drift["passed"])

        self.assertEqual(manifest["schema_version"], "steam-memory-graph-baseline/v0.1")
        self.assertEqual(manifest["artifacts"][0]["name"], "smoke")
        self.assertEqual(manifest["artifacts"][0]["size_bytes"], len(original))

    def test_l1_relation_audit_requires_independent_complete_labels(self) -> None:
        src = _grounded_node("memory:a")
        dst = _grounded_node("memory:b", start_s=3, end_s=4)
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:audit",
            example_id="example:audit",
            video_id="video-1",
            l1_observations=[src, dst],
            atomic_events=[],
            relations=[],
            l1_structural_relations=[
                RelationBelief(
                    edge_id="l1:identity",
                    src=src.node_id,
                    dst=dst.node_id,
                    relation_probabilities={"same_entity": 0.9},
                    status=RelationStatus.UNCALIBRATED_PRIOR,
                    direction_confidence=0.5,
                    evidence_refs=[src.node_id, dst.node_id],
                ),
                RelationBelief(
                    edge_id="l1:state",
                    src=src.node_id,
                    dst=dst.node_id,
                    relation_probabilities={"state_transition": 0.9},
                    status=RelationStatus.UNCALIBRATED_PRIOR,
                    direction_confidence=1.0,
                    evidence_refs=[src.node_id, dst.node_id],
                ),
            ],
        )
        packet, model_key = prepare_l1_relation_packet(overlay.to_dict())
        self.assertEqual(len(packet["items"]), 2)
        self.assertNotIn("probability", packet["items"][0])
        self.assertIn("probability", model_key["items"][0])
        with self.assertRaisesRegex(ValueError, "annotator"):
            evaluate_l1_relation_packet(packet)

        packet["annotator"] = "independent-reviewer"
        for item in packet["items"]:
            item["annotation"]["judgment"] = "supported"
        report = evaluate_l1_relation_packet(packet)

        self.assertTrue(report["acceptance_passed"])
        self.assertEqual(report["groups"]["identity"]["strict_precision"], 1.0)
        self.assertEqual(
            report["groups"]["state_transition"]["strict_precision"], 1.0
        )

        packet["labels_source"] = "model_provisional"
        provisional = evaluate_l1_relation_packet(packet)
        self.assertTrue(provisional["provisional_target_met"])
        self.assertFalse(provisional["acceptance_passed"])
        self.assertEqual(provisional["labels_source"], "model_provisional")

        packet["labels_source"] = "independent_human"
        packet["items"][0]["annotation"]["judgment"] = "unclear"
        conservative = evaluate_l1_relation_packet(packet)
        self.assertFalse(conservative["acceptance_passed"])
        self.assertEqual(
            conservative["groups"]["identity"]["conservative_precision"], 0.0
        )
        self.assertEqual(
            conservative["groups"]["identity"]["decision_coverage"], 0.0
        )

        split = create_locked_audit_split(
            packet,
            development_fraction=0.5,
            salt="locked-test-salt",
        )
        development = set(split["development_item_ids"])
        held_out = set(split["held_out_item_ids"])
        self.assertFalse(development & held_out)
        self.assertEqual(development | held_out, {"l1-edge:0001", "l1-edge:0002"})
        self.assertEqual(split["group_counts"]["identity"]["held_out"], 1)
        self.assertEqual(split["group_counts"]["state_transition"]["held_out"], 1)

    def test_persisted_overlay_can_be_restored_for_audit_only_retry(self) -> None:
        first = _atomic_node(
            "event:1", 0.0, 1.0, "first", mention_id="person:1", surface="man"
        )
        second = _atomic_node(
            "event:2", 1.0, 2.0, "second", mention_id="person:1", surface="man"
        )
        relation = _belief(first, second, "same_entity")
        overlay = CausalTemporalOverlay(
            overlay_id="overlay:1",
            example_id="example:1",
            video_id="video-1",
            l1_observations=[
                _grounded_node("l1:event:1", start_s=0.0, end_s=1.0),
                _grounded_node("l1:event:2", start_s=1.0, end_s=2.0),
            ],
            atomic_events=[first, second],
            relations=[relation],
        )

        restored = event_graph_from_overlay(overlay.to_dict())

        self.assertEqual(restored.graph_id, "overlay:1")
        self.assertEqual(len(restored.nodes), 2)
        self.assertEqual(
            restored.relations[0].relation_probabilities,
            {"same_entity": 0.9},
        )
        self.assertTrue(restored.metadata["restored_from_persisted_overlay"])

    def test_successful_audit_retry_updates_only_matching_summary_error(self) -> None:
        summary = {
            "samples": [{"video_id": "video-1", "audit_summary": None}],
            "errors": [
                {"video_id": "video-1", "stage": "graph_audit", "error": "bad JSON"},
                {"video_id": "video-2", "stage": "overlay_build", "error": "failed"},
            ],
            "video_count_failed": 2,
        }
        audit = {
            "summary": {"verdict": "pass"},
            "computed_summary": {"audit_complete": True, "strict_precision": 1.0},
            "temporal_consistency": {"passed": True, "issues": []},
        }

        updated = update_run_summary(summary, video_id="video-1", audit=audit)

        self.assertEqual(updated["video_count_failed"], 1)
        self.assertEqual(updated["errors"][0]["video_id"], "video-2")
        self.assertEqual(updated["samples"][0]["audit_summary"]["verdict"], "pass")
        self.assertTrue(
            updated["samples"][0]["computed_audit_summary"]["audit_complete"]
        )


def _atomic_node(
    node_id: str,
    start_s: float,
    end_s: float,
    text: str,
    *,
    mention_id: str,
    surface: str,
    entity_type: str = "person",
    state: tuple[str, str] | None = None,
) -> MemoryNode:
    states = (
        [
            {
                "mention_id": mention_id,
                "attribute": state[0],
                "value": state[1],
                "polarity": "positive",
            }
        ]
        if state
        else []
    )
    return MemoryNode(
        node_id=node_id,
        video_id="video-1",
        time_span=TimeSpan(start_s, end_s),
        provenance={"layer": "L1.5"},
        node_type="atomic_event",
        text=text,
        source_segments=[f"l1:{node_id}"],
        metadata={
            "predicate": text,
            "evidence_refs": [f"l1:{node_id}"],
            "participants": [
                {
                    "mention_id": mention_id,
                    "role": "agent",
                    "entity_type": entity_type,
                    "surface": surface,
                }
            ],
            "states": states,
        },
    )


def _belief(
    src: MemoryNode,
    dst: MemoryNode,
    relation: str,
    **support: object,
) -> RelationBelief:
    quotes = support.get("evidence_quotes")
    if isinstance(quotes, dict):
        warrant = f"{quotes.get('src', '')} therefore {quotes.get('dst', '')}"
    else:
        warrant = "structured relation fixture"
    return RelationBelief(
        edge_id=f"candidate:{src.node_id}->{dst.node_id}:{relation}",
        src=src.node_id,
        dst=dst.node_id,
        relation_probabilities={relation: 0.9},
        status=RelationStatus.UNCALIBRATED_PRIOR,
        direction_confidence=0.9,
        evidence_refs=src.source_segments + dst.source_segments,
        warrant=warrant,
        provenance=dict(support),
    )


def _node(node_id: str, start_s: float, end_s: float) -> MemoryNode:
    return MemoryNode(
        node_id=node_id,
        video_id="video-1",
        time_span=TimeSpan(start_s, end_s),
        provenance={"created_by": "fixture"},
        text=f"event {node_id}",
        source_segments=[f"source:{node_id}"],
    )


def _grounded_node(
    node_id: str,
    *,
    start_s: float = 0,
    end_s: float = 2,
) -> MemoryNode:
    source_id = node_id.removeprefix("memory:")
    return MemoryNode(
        node_id=node_id,
        video_id="video-1",
        time_span=TimeSpan(start_s, end_s),
        provenance={
            "source_graph_id": "clue-memory:fixture",
            "source_node_type": "observation",
            "adapter": "fixture",
        },
        node_type="observation",
        text="A person places a key on a table.",
        source_node_id=source_id,
        source_segments=[source_id],
    )


if __name__ == "__main__":
    unittest.main()
