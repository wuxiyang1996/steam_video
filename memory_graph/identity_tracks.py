"""Conflict-aware admission of verified L1 identity tracks."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any


IDENTITY_EDGE_TYPES = frozenset({"same_entity", "same_object", "reappears"})
STABLE_ATTRIBUTE_KEYS = frozenset(
    {"color", "clothing", "material", "shape", "size", "role"}
)


@dataclass(frozen=True)
class IdentityTrackReport:
    observation_count: int
    candidate_edge_count: int
    accepted_edge_ids: tuple[str, ...]
    rejected_edges: tuple[dict[str, Any], ...]
    track_count: int
    endpoint_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "candidate_edge_count": self.candidate_edge_count,
            "accepted_edge_ids": list(self.accepted_edge_ids),
            "rejected_edges": list(self.rejected_edges),
            "track_count": self.track_count,
            "endpoint_contract": self.endpoint_contract,
            "method": "verified_component_consistency/v1",
        }


def build_identity_tracks(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[dict[str, str], IdentityTrackReport]:
    """Merge only explicitly verified links whose full components are compatible."""

    parent = {node_id: node_id for node_id in nodes}
    members = {node_id: {node_id} for node_id in nodes}

    def root(node_id: str) -> str:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def merge(left: str, right: str) -> None:
        left_root, right_root = root(left), root(right)
        if left_root == right_root:
            return
        keep, drop = sorted((left_root, right_root))
        parent[drop] = keep
        members[keep] |= members.pop(drop)

    accepted: list[str] = []
    rejected: list[dict[str, Any]] = []
    candidates = [
        edge
        for edge in edges
        if str(edge.get("edge_type") or "") in IDENTITY_EDGE_TYPES
    ]
    endpoint_ids = {
        str(edge.get(endpoint) or "")
        for edge in candidates
        for endpoint in ("src", "dst")
        if edge.get(endpoint)
    }
    endpoint_nodes = [nodes[node_id] for node_id in endpoint_ids if node_id in nodes]
    for index, edge in enumerate(candidates):
        edge_id = str(edge.get("edge_id") or f"identity-edge:{index}")
        src, dst = str(edge.get("src") or ""), str(edge.get("dst") or "")
        reasons: list[str] = []
        if src not in nodes or dst not in nodes:
            reasons.append("unknown identity endpoint")
        else:
            for endpoint_name, endpoint_id in (("src", src), ("dst", dst)):
                contract_reasons = _identity_endpoint_contract(nodes[endpoint_id])
                reasons.extend(
                    f"{endpoint_name} {reason}" for reason in contract_reasons
                )
        if not _identity_verified(edge):
            reasons.append("identity link lacks explicit verifier acceptance")
        if not reasons:
            proposed = members[root(src)] | members[root(dst)]
            reasons.extend(_component_conflicts(proposed, nodes))
        if reasons:
            rejected.append(
                {
                    "edge_id": edge_id,
                    "src": src,
                    "dst": dst,
                    "reasons": reasons,
                }
            )
            continue
        merge(src, dst)
        accepted.append(edge_id)

    tracks = {node_id: f"track:{root(node_id)}" for node_id in nodes}
    return tracks, IdentityTrackReport(
        observation_count=len(nodes),
        candidate_edge_count=len(candidates),
        accepted_edge_ids=tuple(accepted),
        rejected_edges=tuple(rejected),
        track_count=len(set(tracks.values())),
        endpoint_contract={
            "candidate_endpoint_count": len(endpoint_ids),
            "resolved_endpoint_count": len(endpoint_nodes),
            "entity_mention_endpoint_count": sum(
                str(node.get("node_type") or "") in {"entity", "entity_mention"}
                for node in endpoint_nodes
            ),
            "mention_id_count": sum(bool(_mention_id(node)) for node in endpoint_nodes),
            "entity_type_count": sum(bool(_entity_type(node)) for node in endpoint_nodes),
            "attributes_count": sum(
                bool(node.get("attributes") or _payload(node).get("attributes"))
                for node in endpoint_nodes
            ),
            "evidence_refs_count": sum(
                bool(node.get("evidence_refs") or _payload(node).get("evidence_refs"))
                for node in endpoint_nodes
            ),
        },
    )


def _identity_endpoint_contract(node: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if str(node.get("node_type") or "") not in {"entity", "entity_mention"}:
        reasons.append("identity endpoint is not an entity mention")
    if not _mention_id(node):
        reasons.append("identity endpoint lacks mention_id")
    if not _entity_type(node):
        reasons.append("identity endpoint lacks entity_type")
    if not (node.get("evidence_refs") or _payload(node).get("evidence_refs")):
        reasons.append("identity endpoint lacks evidence_refs")
    return reasons


def _mention_id(node: dict[str, Any]) -> str:
    return str(node.get("mention_id") or _payload(node).get("mention_id") or "").strip()


def _identity_verified(edge: dict[str, Any]) -> bool:
    if edge.get("identity_verified") is True:
        return True
    payload = edge.get("payload")
    return isinstance(payload, dict) and payload.get("identity_verified") is True


def _component_conflicts(
    member_ids: set[str],
    nodes: dict[str, dict[str, Any]],
) -> list[str]:
    ordered = sorted(member_ids)
    reasons: list[str] = []
    for index, left_id in enumerate(ordered):
        for right_id in ordered[index + 1 :]:
            left, right = nodes[left_id], nodes[right_id]
            left_type, right_type = _entity_type(left), _entity_type(right)
            if left_type and right_type and left_type != right_type:
                reasons.append(
                    f"entity type conflict: {left_id}={left_type}, {right_id}={right_type}"
                )
            for key in sorted(STABLE_ATTRIBUTE_KEYS):
                left_value = _attribute(left, key)
                right_value = _attribute(right, key)
                if left_value and right_value and left_value != right_value:
                    reasons.append(
                        f"stable attribute conflict {key}: {left_id}={left_value}, "
                        f"{right_id}={right_value}"
                    )
            if _simultaneous_distinct(left, right):
                reasons.append(f"simultaneous distinct instances: {left_id}, {right_id}")
            if _impossible_motion(left, right):
                reasons.append(f"impossible displacement: {left_id}, {right_id}")
    return list(dict.fromkeys(reasons))


def _entity_type(node: dict[str, Any]) -> str:
    explicit = node.get("entity_type") or _payload(node).get("entity_type")
    return str(explicit or "").strip().casefold()


def _attribute(node: dict[str, Any], key: str) -> str:
    attributes = node.get("attributes")
    if not isinstance(attributes, dict):
        attributes = _payload(node).get("attributes")
    value = attributes.get(key) if isinstance(attributes, dict) else None
    return str(value or "").strip().casefold()


def _simultaneous_distinct(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_instance = left.get("instance_id") or _payload(left).get("instance_id")
    right_instance = right.get("instance_id") or _payload(right).get("instance_id")
    if not left_instance or not right_instance or left_instance == right_instance:
        return False
    left_span, right_span = left.get("time_span"), right.get("time_span")
    if not isinstance(left_span, dict) or not isinstance(right_span, dict):
        return False
    return max(float(left_span["start_s"]), float(right_span["start_s"])) <= min(
        float(left_span["end_s"]), float(right_span["end_s"])
    )


def _impossible_motion(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_position, right_position = _position(left), _position(right)
    left_span, right_span = left.get("time_span"), right.get("time_span")
    if left_position is None or right_position is None:
        return False
    if not isinstance(left_span, dict) or not isinstance(right_span, dict):
        return False
    gap = max(
        0.0,
        float(right_span["start_s"]) - float(left_span["end_s"]),
        float(left_span["start_s"]) - float(right_span["end_s"]),
    )
    max_speed = min(_max_speed(left), _max_speed(right))
    distance = hypot(
        left_position[0] - right_position[0],
        left_position[1] - right_position[1],
    )
    return distance > max_speed * max(gap, 0.25)


def _position(node: dict[str, Any]) -> tuple[float, float] | None:
    value = node.get("position_m") or _payload(node).get("position_m")
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _max_speed(node: dict[str, Any]) -> float:
    value = node.get("max_speed_mps") or _payload(node).get("max_speed_mps") or 12.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 12.0


def _payload(node: dict[str, Any]) -> dict[str, Any]:
    value = node.get("payload")
    return value if isinstance(value, dict) else {}
