"""Single L1 -> L1.5 causal-temporal overlay build pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Sequence

from .adapter import canonical_to_memory_nodes
from .atomic_events import parse_atomic_events
from .causal_witness import witness_from_provenance
from .contracts import AtomicEvent, L1HumanAudit
from .embedding import EmbeddingProvider, embed_memory_nodes
from .event_adapter import OverlayIndex, atomic_events_to_graph_nodes
from .graph_builder import RelationScorer, build_memory_graph
from .l1_structural_edges import (
    L1StructuralEdgeReport,
    materialize_l1_structural_relations,
)
from .l1_structuralizer import (
    L1StructuralizationReport,
    structuralize_video_skills_l1,
)
from .mechanism_candidates import (
    MechanismCandidate,
    generate_mechanism_candidates,
    generate_video_skills_l1_edge_candidates,
)
from .reliability import L1ReliabilityReport, audit_l1_nodes
from .selectstream_policy import plan_bounded_memory
from .state_relations import derive_state_transition_candidates
from .types import (
    CausalTemporalOverlay,
    MemoryGraph,
    MemoryNode,
    RelationBelief,
    RelationStatus,
)
from .visual_verifier import VisualRereadProvider
from .video_l1 import VideoL1ExtractionResult, VideoL1Provider
from .video_skills_l1 import VideoSkillsL1QualityReport, audit_video_skills_l1


InputMode = Literal["expert_demo", "video_only"]


class AtomicEventExtractor(Protocol):
    def extract(self, nodes: list[MemoryNode]) -> list[AtomicEvent]: ...


class RelationTeacher(Protocol):
    model: str

    def label_relations(
        self,
        nodes: list[MemoryNode],
        embeddings: Sequence[Sequence[float]],
        *,
        top_k: int = 3,
        mechanism_candidates: Sequence[MechanismCandidate] | None = None,
    ) -> list[RelationBelief]: ...


@dataclass(frozen=True)
class PayloadAtomicEventExtractor:
    """Use a precomputed JSON payload while retaining parser hard checks."""

    payload: dict[str, Any]
    model: str = "precomputed-independent-extractor"

    def extract(self, nodes: list[MemoryNode]) -> list[AtomicEvent]:
        return parse_atomic_events(self.payload, nodes=nodes, model=self.model)


@dataclass
class OverlayBuildResult:
    overlay: CausalTemporalOverlay
    l1_report: L1ReliabilityReport
    atomic_events: list[AtomicEvent]
    overlay_index: OverlayIndex
    candidate_relations: list[RelationBelief]
    rejected_relations: list[dict[str, Any]]
    verifier_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = self.overlay.to_dict()
        payload["build_report"] = {
            "l1_reliability": self.l1_report.to_dict(),
            "overlay_index": self.overlay_index.to_dict(),
            "candidate_relations": [
                relation.to_dict() for relation in self.candidate_relations
            ],
            "verifier_summary": self.verifier_summary,
            "rejected_relations": self.rejected_relations,
        }
        return payload


def overlay_to_event_graph(overlay: CausalTemporalOverlay) -> MemoryGraph:
    """Compatibility view for graph auditors that operate on event endpoints."""
    return MemoryGraph(
        graph_id=overlay.overlay_id,
        example_id=overlay.example_id,
        video_id=overlay.video_id,
        nodes=overlay.atomic_events,
        relations=overlay.relations,
        metadata={
            **overlay.metadata,
            "compatibility_view": "atomic_events_only",
        },
    )


def build_causal_temporal_overlay(
    canonical: dict[str, Any],
    *,
    input_mode: InputMode,
    event_extractor: AtomicEventExtractor,
    human_audit: L1HumanAudit | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    embedding_output_path: str | Path | None = None,
    relation_teacher: RelationTeacher | None = None,
    relation_scorer: RelationScorer | None = None,
    batch_size: int = 8,
    top_k_candidates: int = 4,
    max_before_neighbors: int = 4,
    min_relation_probability: float = 0.5,
    relation_thresholds: dict[str, float] | None = None,
    apply_hard_verifiers: bool = True,
    visual_reread_provider: VisualRereadProvider | None = None,
    require_visual_verification: bool = False,
    memory_capacity: int | None = None,
    allow_provisional_expert_demo: bool = False,
    video_l1_provider: VideoL1Provider | None = None,
    allow_visual_l1_replacement: bool = False,
) -> OverlayBuildResult:
    """Build an overlay whose relation endpoints are atomic events only."""
    if input_mode not in {"expert_demo", "video_only"}:
        raise ValueError("input_mode must be expert_demo or video_only")
    if relation_teacher is not None and relation_scorer is not None:
        raise ValueError("provide either relation_teacher or relation_scorer, not both")
    if not 0.0 <= min_relation_probability <= 1.0:
        raise ValueError("min_relation_probability must be in [0, 1]")

    if video_l1_provider is not None and not allow_visual_l1_replacement:
        raise ValueError(
            "full-video visual L1 replacement is disabled; use an accepted "
            "Video_Skills clue_memory_graph and targeted visual repair"
        )
    source_graph, canonical_l1_nodes = canonical_to_memory_nodes(
        canonical,
        require_materialized_l1=(
            input_mode == "video_only" and video_l1_provider is None
        ),
    )
    observation_end = source_graph.get("observation_end_s")
    video_l1_result: VideoL1ExtractionResult | None = None
    video_skills_l1_quality: VideoSkillsL1QualityReport | None = None
    l1_structuralization: L1StructuralizationReport | None = None
    l1_structural_edge_report: L1StructuralEdgeReport | None = None
    if video_l1_provider is not None:
        if input_mode != "video_only":
            raise ValueError("video_l1_provider requires input_mode='video_only'")
        video_path = _canonical_video_path(canonical)
        if not video_path:
            raise ValueError("video_l1_provider requires a canonical raw-video path")
        video = canonical.get("video")
        video_id = (
            str(video.get("video_id"))
            if isinstance(video, dict) and video.get("video_id")
            else str(source_graph.get("video_id") or "")
        )
        video_l1_result = video_l1_provider.extract(
            video_path=video_path,
            video_id=video_id,
            observation_end_s=(
                float(observation_end) if observation_end is not None else None
            ),
        )
        all_l1_nodes = list(video_l1_result.nodes)
    else:
        all_l1_nodes = canonical_l1_nodes
        if input_mode == "video_only":
            video_skills_l1_quality = audit_video_skills_l1(canonical)
            if video_skills_l1_quality.status != "pass":
                raise ValueError(
                    "Video_Skills L1 failed acceptance gate: "
                    + "; ".join(video_skills_l1_quality.issues)
                )
            all_l1_nodes, l1_structuralization = structuralize_video_skills_l1(
                source_graph,
                all_l1_nodes,
            )
    l1_nodes = _select_l1_inputs(all_l1_nodes, input_mode=input_mode)
    if not l1_nodes:
        raise ValueError(
            f"canonical example produced no eligible {input_mode} L1 observations"
        )
    l1_structural_relations: list[RelationBelief] = []
    if video_skills_l1_quality is not None:
        (
            l1_structural_relations,
            l1_structural_edge_report,
        ) = materialize_l1_structural_relations(source_graph, l1_nodes)
    l1_report = audit_l1_nodes(
        l1_nodes,
        human_audit=human_audit,
        observation_end_s=float(observation_end) if observation_end is not None else None,
    )

    if input_mode == "video_only":
        causal_allowed = l1_report.status == "pass"
        deterministic_gate_passed = all(
            metric.passed is not False
            for metric in l1_report.metrics.values()
            if metric.source == "deterministic"
        )
        semantic_allowed = deterministic_gate_passed and (
            video_skills_l1_quality is None
            or video_skills_l1_quality.status == "pass"
        )
        if causal_allowed:
            trust_status = "trusted"
        elif video_skills_l1_quality is not None:
            trust_status = "accepted_video_skills_l1_structural"
        elif video_l1_result is not None:
            trust_status = "machine_visual_l1_unvalidated"
        else:
            trust_status = l1_report.status
    else:
        causal_allowed = allow_provisional_expert_demo
        semantic_allowed = allow_provisional_expert_demo
        trust_status = "provisional_expert_text"

    atomic_events = event_extractor.extract(l1_nodes)
    if not atomic_events:
        raise ValueError("atomic event extractor produced no L1.5 events")
    event_nodes, overlay_index = atomic_events_to_graph_nodes(
        atomic_events,
        l1_nodes=l1_nodes,
    )
    mechanism_candidates = generate_mechanism_candidates(event_nodes)
    existing_candidate_pairs = {
        (candidate.src, candidate.dst) for candidate in mechanism_candidates
    }
    for candidate in generate_video_skills_l1_edge_candidates(
        event_nodes,
        l1_nodes=l1_nodes,
        event_to_l1=overlay_index.event_to_l1,
    ):
        if (candidate.src, candidate.dst) not in existing_candidate_pairs:
            mechanism_candidates.append(candidate)
            existing_candidate_pairs.add((candidate.src, candidate.dst))

    embeddings: Sequence[Sequence[float]] | None = None
    if embedding_provider is not None:
        if embedding_output_path is None:
            raise ValueError("embedding_output_path is required with embedding_provider")
        embeddings = embed_memory_nodes(
            event_nodes,
            embedding_provider,
            output_path=embedding_output_path,
            batch_size=batch_size,
        )

    effective_scorer = relation_scorer if semantic_allowed else None
    event_graph = build_memory_graph(
        graph_id=f"event_graph:{canonical.get('example_id')}",
        example_id=str(canonical.get("example_id") or source_graph.get("example_id") or ""),
        video_id=l1_nodes[0].video_id,
        nodes=event_nodes,
        embeddings=embeddings,
        relation_scorer=effective_scorer,
        top_k_candidates=top_k_candidates,
        max_before_neighbors=max_before_neighbors,
    )
    temporal_relations = [
        relation
        for relation in event_graph.relations
        if relation.status is RelationStatus.DETERMINISTIC
    ]
    proposals = [
        relation
        for relation in event_graph.relations
        if relation.status is not RelationStatus.DETERMINISTIC
    ]
    if semantic_allowed:
        existing_labels = {
            (relation.src, relation.dst, label)
            for relation in proposals
            for label in relation.relation_probabilities
        }
        for derived in derive_state_transition_candidates(event_nodes):
            key = (derived.src, derived.dst, "state_transition")
            if key not in existing_labels:
                proposals.append(derived)
                existing_labels.add(key)
    if relation_teacher is not None and semantic_allowed:
        if embeddings is None:
            raise ValueError("relation_teacher requires event embeddings")
        proposals.extend(
            relation_teacher.label_relations(
                event_nodes,
                embeddings,
                top_k=top_k_candidates,
                mechanism_candidates=mechanism_candidates,
            )
        )
    visual_reread_count = 0
    if visual_reread_provider is not None and semantic_allowed:
        video_path = _canonical_video_path(canonical)
        proposals, visual_reread_count = _apply_visual_rereads(
            proposals,
            event_nodes=event_nodes,
            video_path=video_path,
            provider=visual_reread_provider,
        )

    if require_visual_verification and visual_reread_provider is None:
        raise ValueError(
            "require_visual_verification=True but no visual_reread_provider "
            "was configured"
        )
    # Phase 0: without a passed causal gate, explains/enables require a
    # passed raw-video witness. Missing visual verification rejects them
    # instead of admitting hard-verifier-only causal edges.
    effective_require_visual = require_visual_verification or (not causal_allowed)
    accepted, rejected, verifier_summary = _verify_proposals(
        proposals,
        event_nodes=event_nodes,
        l1_nodes=l1_nodes,
        minimum_probability=min_relation_probability,
        relation_thresholds=relation_thresholds or {},
        enabled=apply_hard_verifiers,
        require_visual_verification=effective_require_visual,
    )
    memory_plan = (
        plan_bounded_memory(
            event_nodes,
            temporal_relations + accepted,
            capacity=memory_capacity,
        )
        if memory_capacity is not None
        else None
    )
    targeted_visual_reread_queue = []
    if l1_structuralization is not None:
        targeted_visual_reread_queue.extend(
            {
                "target_id": event_id,
                "target_type": "event",
                "reason": "unresolved_event_participant",
            }
            for event_id in l1_structuralization.unresolved_event_ids
        )
    if l1_structural_edge_report is not None:
        targeted_visual_reread_queue.extend(
            {
                "target_id": edge_id,
                "target_type": "l1_edge",
                "reason": "unresolved_structural_endpoint",
            }
            for edge_id in l1_structural_edge_report.unresolved_edge_ids
        )
    overlay = CausalTemporalOverlay(
        overlay_id=f"causal_temporal_overlay:{canonical.get('example_id')}",
        example_id=str(canonical.get("example_id") or source_graph.get("example_id") or ""),
        video_id=l1_nodes[0].video_id,
        l1_observations=l1_nodes,
        atomic_events=event_nodes,
        relations=temporal_relations + accepted,
        l1_structural_relations=l1_structural_relations,
        metadata={
            "layer_contract": "l1_observations_plus_l1_5_atomic_overlay",
            "node_event_assumption": "l1_observation_container_with_l1_5_atomic_overlay",
            "input_mode": input_mode,
            "trust_status": trust_status,
            "semantic_relations_allowed": semantic_allowed,
            "causal_relations_allowed": causal_allowed,
            "l1_reliability": l1_report.to_dict(),
            "video_skills_l1_quality": (
                video_skills_l1_quality.to_dict()
                if video_skills_l1_quality is not None
                else None
            ),
            "video_skills_l1_structuralization": (
                l1_structuralization.to_dict()
                if l1_structuralization is not None
                else None
            ),
            "video_skills_l1_structural_edges": (
                l1_structural_edge_report.to_dict()
                if l1_structural_edge_report is not None
                else None
            ),
            "targeted_visual_reread_queue": targeted_visual_reread_queue,
            "source_l1_graph_id": source_graph.get("graph_id"),
            "source_canonical_schema_version": canonical.get("schema_version"),
            "atomic_event_count": len(event_nodes),
            "l1_observation_count": len(l1_nodes),
            "excluded_supervision_node_count": (
                len(canonical_l1_nodes) - len(l1_nodes)
                if video_l1_result is None
                else len(canonical_l1_nodes)
            ),
            "events_per_l1_observation": len(event_nodes) / len(l1_nodes),
            "relation_teacher": getattr(relation_teacher, "model", None),
            "relation_teacher_errors": list(
                getattr(relation_teacher, "last_label_errors", [])
            ),
            "mechanism_candidate_count": len(mechanism_candidates),
            "candidate_generation": "mechanism_first_plus_embedding_recall",
            "relation_layers": {
                "observed_structure": [
                    "temporal_next",
                    "before",
                    "overlaps",
                    "during",
                    "same_entity",
                    "same_object",
                    "state_transition",
                ],
                "identity_candidates": [
                    "same_instance_candidate",
                    "reappears_candidate",
                ],
                "observation_support": ["observation_support"],
                "predictive_navigation": [
                    "transition_support",
                    "response_candidate",
                ],
                "verified_candidate_causality": ["explains", "enables"],
            },
            "visual_verification_enabled": visual_reread_provider is not None,
            "visual_verification_required": effective_require_visual,
            "visual_reread_count": visual_reread_count,
            "visual_reread_model": getattr(visual_reread_provider, "model", None),
            "raw_video_path": _canonical_video_path(canonical),
            "video_l1_extraction": (
                {
                    key: value
                    for key, value in video_l1_result.to_dict().items()
                    if key != "nodes"
                }
                if video_l1_result is not None
                else None
            ),
            "selectstream_memory_plan": (
                memory_plan.to_dict() if memory_plan is not None else None
            ),
            "hard_verifiers_enabled": apply_hard_verifiers,
            "relation_thresholds": relation_thresholds or {},
            "probabilistic_relation_status": (
                "blocked_by_l1_gate"
                if not causal_allowed
                else (
                    relation_scorer.status.value
                    if relation_scorer is not None
                    else "uncalibrated_prior"
                )
            ),
        },
    )
    return OverlayBuildResult(
        overlay=overlay,
        l1_report=l1_report,
        atomic_events=atomic_events,
        overlay_index=overlay_index,
        candidate_relations=proposals,
        rejected_relations=rejected,
        verifier_summary=verifier_summary,
    )


def _select_l1_inputs(
    nodes: list[MemoryNode],
    *,
    input_mode: InputMode,
) -> list[MemoryNode]:
    """Keep only evidence visible in the declared staged input regime."""
    selected: list[MemoryNode] = []
    hidden_source_types = {
        "gold_annotation",
        "qa_answer",
        "qa_explanation",
        "hidden_supervision",
        "inference_scene",
        "key_relationship",
        "reasoning_process_step",
    }
    for node in nodes:
        visibility = node.metadata.get("visibility")
        if isinstance(visibility, dict):
            if visibility.get("hidden_supervision") is True:
                continue
            if visibility.get("visible_to_agent") is False:
                continue
        source_type = str(node.metadata.get("source_type") or "").lower()
        if source_type in hidden_source_types:
            continue
        if node.metadata.get("discovery_status") == "provided_supervision":
            continue
        if input_mode == "expert_demo" and source_type != "segment_description":
            continue
        selected.append(node)
    return selected


def _verify_proposals(
    proposals: list[RelationBelief],
    *,
    event_nodes: list[MemoryNode],
    l1_nodes: list[MemoryNode],
    minimum_probability: float,
    relation_thresholds: dict[str, float],
    enabled: bool,
    require_visual_verification: bool = False,
) -> tuple[list[RelationBelief], list[dict[str, Any]], dict[str, Any]]:
    event_by_id = {node.node_id: node for node in event_nodes}
    l1_by_id = {node.node_id: node for node in l1_nodes}
    accepted: list[RelationBelief] = []
    rejected: list[dict[str, Any]] = []
    accepted_labels = 0
    proposed_labels = 0

    for proposal in proposals:
        kept: dict[str, float] = {}
        verifier_records: dict[str, Any] = {}
        for relation_name, probability in proposal.relation_probabilities.items():
            threshold = float(relation_thresholds.get(relation_name, minimum_probability))
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(f"invalid threshold for {relation_name}: {threshold}")
            if probability < threshold:
                continue
            proposed_labels += 1
            if (
                require_visual_verification
                and relation_name in {"explains", "enables"}
                and not _visual_verification_passed(proposal)
            ):
                rejected.append(
                    {
                        "edge_id": proposal.edge_id,
                        "src": proposal.src,
                        "dst": proposal.dst,
                        "relation": relation_name,
                        "probability": probability,
                        "reasons": [
                            "candidate-causal relation requires passed raw-video verification"
                        ],
                    }
                )
                continue
            if not enabled:
                kept[relation_name] = probability
                verifier_records[relation_name] = {
                    "passed": True,
                    "reasons": ["hard verification disabled"],
                }
                continue

            from .verifiers import verify_relation

            single = replace(
                proposal,
                relation_probabilities={relation_name: probability},
            )
            result = verify_relation(
                single,
                src=event_by_id[proposal.src],
                dst=event_by_id[proposal.dst],
                l1_by_id=l1_by_id,
            )
            verifier_records[relation_name] = result.to_dict()
            if result.passed:
                kept[relation_name] = probability
                accepted_labels += 1
            else:
                rejected.append(
                    {
                        "edge_id": proposal.edge_id,
                        "src": proposal.src,
                        "dst": proposal.dst,
                        "relation": relation_name,
                        "probability": probability,
                        "reasons": list(result.reasons),
                    }
                )
        if kept:
            accepted.append(
                replace(
                    proposal,
                    relation_probabilities=kept,
                    provenance={
                        **proposal.provenance,
                        "hard_verifier": verifier_records,
                    },
                )
            )

    return accepted, rejected, {
        "proposed_labels_at_threshold": proposed_labels,
        "accepted_labels": accepted_labels if enabled else proposed_labels,
        "rejected_labels": len(rejected),
        "acceptance_rate": (
            (accepted_labels if enabled else proposed_labels) / proposed_labels
            if proposed_labels
            else None
        ),
        "minimum_probability": minimum_probability,
        "relation_thresholds": relation_thresholds,
        "enabled": enabled,
        "visual_verification_required": require_visual_verification,
    }


def _canonical_video_path(canonical: dict[str, Any]) -> str | None:
    video = canonical.get("video")
    if not isinstance(video, dict):
        return None
    value = video.get("primary_path") or video.get("path")
    return str(value) if value else None


def _apply_visual_rereads(
    proposals: list[RelationBelief],
    *,
    event_nodes: list[MemoryNode],
    video_path: str | None,
    provider: VisualRereadProvider,
) -> tuple[list[RelationBelief], int]:
    event_by_id = {node.node_id: node for node in event_nodes}
    updated: list[RelationBelief] = []
    count = 0
    for proposal in proposals:
        witness = witness_from_provenance(proposal.provenance)
        if witness is None:
            updated.append(proposal)
            continue
        if not video_path:
            visual = {
                "status": "inconclusive",
                "model": getattr(provider, "model", None),
                "protocol_version": "causal-visual-reread/v0.1",
                "video_windows": [],
                "checks": {},
                "reasons": ["canonical example does not provide a raw-video path"],
            }
        else:
            try:
                result = provider.verify(
                    video_path=video_path,
                    belief=proposal,
                    src=event_by_id[proposal.src],
                    dst=event_by_id[proposal.dst],
                    witness=witness,
                )
                visual = result.to_dict()
                count += 1
            except Exception as exc:
                visual = {
                    "status": "inconclusive",
                    "model": getattr(provider, "model", None),
                    "protocol_version": "causal-visual-reread/v0.1",
                    "video_windows": [],
                    "checks": {},
                    "reasons": [f"{type(exc).__name__}: {exc}"],
                }
        updated.append(
            replace(
                proposal,
                provenance={
                    **proposal.provenance,
                    "visual_verification": visual,
                },
            )
        )
    return updated, count


def _visual_verification_passed(proposal: RelationBelief) -> bool:
    visual = proposal.provenance.get("visual_verification")
    return isinstance(visual, dict) and visual.get("status") == "passed"
