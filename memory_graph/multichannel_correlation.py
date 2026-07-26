"""Question-independent, non-causal descriptors for L1.5 navigation pairs.

This module deliberately does not learn which edge should be followed.  It
preserves grounded pair evidence next to the existing semantic affinity so an
action-conditioned IWM can later predict the consequence of traversing a hop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from typing import Any, Mapping, Sequence

from .soft_correlation import SoftCorrelationBuildResult
from .types import MemoryNode


MULTICHANNEL_PAIR_SCHEMA = "steam-l1.5-multichannel-pair-descriptor/v0.1"
MULTICHANNEL_AUDIT_SCHEMA = "steam-l1.5-multichannel-pair-audit/v0.1"

SUPPORTED_CHANNELS = frozenset(
    {
        "semantic",
        "semantic_recurrence",
        "entity_correspondence",
        "change",
        "contrast",
    }
)
SUPPORTED_DIRECTIONS = frozenset({"symmetric", "src_to_dst", "dst_to_src"})
SUPPORTED_STATUSES = frozenset({"observed", "candidate", "conflict", "inconclusive"})
SUPPORTED_CANDIDATE_EDGE_CHANNELS = frozenset(
    {
        "caption_bridge",
        "entity_candidate",
        "change_candidate",
        "contrast_candidate",
    }
)
CROSS_NODE_TRACK_STATUSES = frozenset(
    {
        "accepted_identity_track",
        "verified_cross_window_track",
        "global_track_candidate",
        "cross_window_track_candidate",
    }
)


@dataclass(frozen=True)
class NavigationPairSignal:
    """One independently sourced L1.5 signal; never an action utility."""

    channel: str
    status: str
    category: str
    source: str
    evidence_refs: tuple[str, ...]
    direction: str = "symmetric"
    affinity: float | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.channel not in SUPPORTED_CHANNELS:
            raise ValueError(f"unsupported L1.5 signal channel: {self.channel}")
        if self.direction not in SUPPORTED_DIRECTIONS:
            raise ValueError(f"unsupported L1.5 signal direction: {self.direction}")
        if self.status not in SUPPORTED_STATUSES:
            raise ValueError(f"unsupported L1.5 signal status: {self.status}")
        if not self.category or not self.source or not self.evidence_refs:
            raise ValueError("L1.5 signal requires category, source, and evidence refs")
        if self.affinity is not None and (
            not math.isfinite(self.affinity) or not 0.0 <= self.affinity <= 1.0
        ):
            raise ValueError("L1.5 affinity must be finite and in [0, 1]")
        forbidden = f"{self.channel} {self.category}".lower()
        if "causal" in forbidden or "probability" in forbidden:
            raise ValueError("L1.5 descriptors may not claim causality or probability")
        if self.channel in {"semantic", "semantic_recurrence"}:
            if self.affinity is None or self.direction != "symmetric":
                raise ValueError("semantic L1.5 signals require symmetric affinity")
        elif self.affinity is not None:
            raise ValueError("grounded categorical descriptors may not invent affinity")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "channel": self.channel,
            "status": self.status,
            "category": self.category,
            "direction": self.direction,
            "source": self.source,
            "evidence_refs": list(self.evidence_refs),
            "score_semantics": (
                "navigation_affinity_not_probability"
                if self.affinity is not None
                else "categorical_grounded_descriptor_not_verified_relation"
            ),
            "provenance": dict(self.provenance),
        }
        if self.affinity is not None:
            payload["affinity"] = self.affinity
        return payload


@dataclass(frozen=True)
class CandidateNavigationEdge:
    """Categorical, question-independent candidate hop with no numeric score."""

    edge_id: str
    src: str
    dst: str
    channel: str
    category: str
    direction: str
    evidence_refs: tuple[str, ...]
    source: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.edge_id or not self.src or not self.dst or self.src == self.dst:
            raise ValueError("candidate navigation edge identifiers are invalid")
        if self.channel not in SUPPORTED_CANDIDATE_EDGE_CHANNELS:
            raise ValueError(f"unsupported candidate edge channel: {self.channel}")
        if self.direction not in {"bidirectional", "src_to_dst", "dst_to_src"}:
            raise ValueError("candidate edge direction is invalid")
        if not self.category or not self.source or not self.evidence_refs:
            raise ValueError("candidate edge requires category, source, and evidence")
        forbidden = f"{self.channel} {self.category}".lower()
        if "causal" in forbidden or "probability" in forbidden:
            raise ValueError("candidate navigation edges cannot claim causality/probability")

    def permits(self, source: str, target: str) -> bool:
        if source == self.src and target == self.dst:
            return self.direction in {"bidirectional", "src_to_dst"}
        if source == self.dst and target == self.src:
            return self.direction in {"bidirectional", "dst_to_src"}
        return False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "steam-l1.5-categorical-candidate-edge/v0.1",
            "edge_id": self.edge_id,
            "src": self.src,
            "dst": self.dst,
            "channel": self.channel,
            "category": self.category,
            "direction": self.direction,
            "evidence_refs": list(self.evidence_refs),
            "source": self.source,
            "provenance": dict(self.provenance),
            "status": "candidate",
            "numeric_score_present": False,
            "verified_relation": False,
            "causal_claim": False,
            "planner_preference": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CandidateNavigationEdge":
        return cls(
            edge_id=str(payload.get("edge_id") or ""),
            src=str(payload.get("src") or ""),
            dst=str(payload.get("dst") or ""),
            channel=str(payload.get("channel") or ""),
            category=str(payload.get("category") or ""),
            direction=str(payload.get("direction") or ""),
            evidence_refs=tuple(str(value) for value in payload.get("evidence_refs") or []),
            source=str(payload.get("source") or ""),
            provenance=dict(payload.get("provenance") or {}),
        )


@dataclass(frozen=True)
class MultiChannelNavigationPair:
    """All available static signals for one unordered pair of frozen L1 nodes."""

    src: str
    dst: str
    temporal_gap_s: float
    signals: tuple[NavigationPairSignal, ...]
    legal_hop_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.src or not self.dst or self.src == self.dst:
            raise ValueError("L1.5 pair requires two different node IDs")
        if not math.isfinite(self.temporal_gap_s) or self.temporal_gap_s < 0.0:
            raise ValueError("L1.5 temporal gap must be finite and non-negative")
        if not set(self.legal_hop_sources) <= {
            "temporal_backbone",
            "semantic",
            "semantic_recurrence",
        }:
            raise ValueError("grounded descriptors cannot automatically admit a legal hop")

    @property
    def pair_id(self) -> str:
        digest = hashlib.sha256(
            f"{self.src}\x1f{self.dst}".encode("utf-8")
        ).hexdigest()[:20]
        return f"l15pair:{digest}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": MULTICHANNEL_PAIR_SCHEMA,
            "pair_id": self.pair_id,
            "src": self.src,
            "dst": self.dst,
            "temporal_gap_s": self.temporal_gap_s,
            "signals": [signal.to_dict() for signal in self.signals],
            "legal_hop_sources": list(self.legal_hop_sources),
            "question_independent": True,
            "verified_identity_claim": False,
            "verified_state_transition_claim": False,
            "causal_claim": False,
            "planner_preference": False,
        }


def build_multichannel_pair_audit(
    nodes: Sequence[MemoryNode],
    semantic_build: SoftCorrelationBuildResult,
    *,
    temporal_pairs: Sequence[tuple[str, str]] = (),
) -> dict[str, Any]:
    """Augment the complete semantic pair audit with grounded L1 descriptors.

    Candidate track IDs and structured state fields remain candidates.  They
    never admit an edge on their own and no text-keyword inference is used.
    """

    node_by_id = {node.node_id: node for node in nodes}
    if len(node_by_id) != len(nodes):
        raise ValueError("multichannel audit requires unique node IDs")
    semantic_ids = {
        node_id
        for row in semantic_build.pair_audits
        for node_id in (row.src, row.dst)
    }
    if semantic_ids != set(node_by_id):
        raise ValueError("semantic audit and multichannel nodes do not match")
    temporal = {_ordered_pair(*pair) for pair in temporal_pairs if pair[0] != pair[1]}
    edge_by_pair = {
        _ordered_pair(edge.src, edge.dst): edge for edge in semantic_build.edges
    }
    pairs: list[MultiChannelNavigationPair] = []
    for row in semantic_build.pair_audits:
        src, dst = _ordered_pair(row.src, row.dst)
        pair = (src, dst)
        edge = edge_by_pair.get(pair)
        signals: list[NavigationPairSignal] = []
        legal_sources: list[str] = []
        if pair in temporal:
            legal_sources.append("temporal_backbone")
        if edge is not None:
            signals.append(
                NavigationPairSignal(
                    channel=edge.channel,
                    status="observed",
                    category="embedding_navigation_affinity",
                    direction="symmetric",
                    affinity=edge.navigation_affinity,
                    source=edge.score_source,
                    evidence_refs=edge.evidence_refs,
                    provenance={"edge_id": edge.edge_id},
                )
            )
            legal_sources.append(edge.channel)
        signals.extend(_grounded_pair_signals(node_by_id[src], node_by_id[dst]))
        pairs.append(
            MultiChannelNavigationPair(
                src=src,
                dst=dst,
                temporal_gap_s=row.temporal_gap_s,
                signals=tuple(signals),
                legal_hop_sources=tuple(dict.fromkeys(legal_sources)),
            )
        )
    channel_counts: dict[str, int] = {}
    for pair in pairs:
        for signal in pair.signals:
            channel_counts[signal.channel] = channel_counts.get(signal.channel, 0) + 1
    return {
        "schema_version": MULTICHANNEL_AUDIT_SCHEMA,
        "l1_fingerprint": semantic_build.l1_fingerprint,
        "semantic_policy_fingerprint": semantic_build.admission_policy.fingerprint,
        "pair_count": len(pairs),
        "channel_signal_counts": dict(sorted(channel_counts.items())),
        "question_independent": True,
        "contains_question_or_answer": False,
        "grounded_descriptors_admit_edges": False,
        "learned_edge_selector_present": False,
        "iwm_training_performed": False,
        "pairs": [pair.to_dict() for pair in pairs],
    }


def _grounded_pair_signals(
    src: MemoryNode, dst: MemoryNode
) -> tuple[NavigationPairSignal, ...]:
    signals: list[NavigationPairSignal] = []
    src_mentions = _mentions(src)
    dst_mentions = _mentions(dst)
    shared_mentions = sorted(set(src_mentions) & set(dst_mentions))
    for mention_id in shared_mentions:
        src_type = src_mentions[mention_id].get("entity_type")
        dst_type = dst_mentions[mention_id].get("entity_type")
        if src_type and dst_type and src_type != dst_type:
            status = "conflict"
            category = "shared_candidate_track_with_entity_type_conflict"
        else:
            statuses = {
                src_mentions[mention_id].get("track_status"),
                dst_mentions[mention_id].get("track_status"),
            }
            accepted = statuses <= {
                "accepted_identity_track",
                "verified_cross_window_track",
            }
            status = "observed" if accepted else "candidate"
            category = "shared_cross_window_track_id"
        signals.append(
            NavigationPairSignal(
                channel="entity_correspondence",
                status=status,
                category=category,
                source="l1_structured_participant_metadata",
                evidence_refs=(src.node_id, dst.node_id),
                provenance={
                    "mention_id": mention_id,
                    "src_track_status": src_mentions[mention_id].get("track_status"),
                    "dst_track_status": dst_mentions[mention_id].get("track_status"),
                    "src_entity_type": src_type,
                    "dst_entity_type": dst_type,
                },
            )
        )

    src_states = _states(src)
    dst_states = _states(dst)
    for key in sorted(set(src_states) & set(dst_states)):
        src_value = src_states[key]
        dst_value = dst_states[key]
        if src_value == dst_value or key[0] not in shared_mentions:
            continue
        mention_id, attribute = key
        signals.append(
            NavigationPairSignal(
                channel="change",
                status="candidate",
                category="same_candidate_track_same_attribute_different_value",
                direction="src_to_dst",
                source="l1_structured_state_metadata",
                evidence_refs=(src.node_id, dst.node_id),
                provenance={
                    "mention_id": mention_id,
                    "attribute": attribute,
                    "src_value": src_value,
                    "dst_value": dst_value,
                    "accepted_identity_required_for_state_transition": True,
                },
            )
        )

    src_contrast_refs = _structured_refs(src, "contrast_refs")
    dst_contrast_refs = _structured_refs(dst, "contrast_refs")
    src_points_to_dst = dst.node_id in src_contrast_refs
    dst_points_to_src = src.node_id in dst_contrast_refs
    if src_points_to_dst or dst_points_to_src:
        direction = (
            "symmetric"
            if src_points_to_dst and dst_points_to_src
            else ("src_to_dst" if src_points_to_dst else "dst_to_src")
        )
        signals.append(
            NavigationPairSignal(
                channel="contrast",
                status="observed",
                category="explicit_structured_contrast_reference",
                direction=direction,
                source="l1_structured_contrast_metadata",
                evidence_refs=(src.node_id, dst.node_id),
                provenance={
                    "src_points_to_dst": src_points_to_dst,
                    "dst_points_to_src": dst_points_to_src,
                },
            )
        )
    return tuple(signals)


def _mentions(node: MemoryNode) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    participants = node.metadata.get("participants")
    if isinstance(participants, Sequence) and not isinstance(participants, (str, bytes)):
        for participant in participants:
            if not isinstance(participant, Mapping):
                continue
            mention_id = participant.get("mention_id")
            track_status = str(participant.get("track_status") or "")
            if mention_id and track_status in CROSS_NODE_TRACK_STATUSES:
                result[str(mention_id)] = participant
    return result


def _states(node: MemoryNode) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    states = node.metadata.get("states")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        return result
    for state in states:
        if not isinstance(state, Mapping):
            continue
        mention_id = state.get("mention_id") or state.get("entity_track_id")
        attribute = state.get("attribute")
        value = state.get("value")
        if mention_id and attribute and value is not None:
            result[(str(mention_id), str(attribute))] = str(value)
    return result


def _structured_refs(node: MemoryNode, field: str) -> set[str]:
    values = node.metadata.get(field)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        return set()
    return {str(value) for value in values}


def _ordered_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)
