"""Conservative admission of grounded identity candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .identity_tracks import (
    IDENTITY_EDGE_TYPES,
    _component_conflicts,
    _identity_endpoint_contract,
    _payload,
)


@dataclass(frozen=True)
class IdentityVerificationReport:
    candidate_count: int
    accepted: tuple[dict[str, Any], ...]
    rejected: tuple[dict[str, Any], ...]
    targeted_reread_queue: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "accepted": list(self.accepted),
            "rejected": list(self.rejected),
            "targeted_reread_queue": list(self.targeted_reread_queue),
            "accepted_count": len(self.accepted),
            "rejected_count": len(self.rejected),
            "targeted_reread_count": len(self.targeted_reread_queue),
            "method": "explicit_instance_or_targeted_reread/v1",
            "attribute_only_admission": False,
        }


def verify_identity_candidates(
    nodes: dict[str, dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], IdentityVerificationReport]:
    """Verify identity without treating text or matching attributes as proof."""

    output = [dict(edge) for edge in edges]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    reread: list[dict[str, Any]] = []
    candidate_count = 0
    for index, edge in enumerate(output):
        if str(edge.get("edge_type") or "") not in IDENTITY_EDGE_TYPES:
            continue
        candidate_count += 1
        edge_id = str(edge.get("edge_id") or f"identity-edge:{index}")
        src_id, dst_id = str(edge.get("src") or ""), str(edge.get("dst") or "")
        src, dst = nodes.get(src_id), nodes.get(dst_id)
        reasons: list[str] = []
        if src is None or dst is None:
            reasons.append("unknown identity endpoint")
        else:
            reasons.extend(f"src {reason}" for reason in _identity_endpoint_contract(src))
            reasons.extend(f"dst {reason}" for reason in _identity_endpoint_contract(dst))
            reasons.extend(_component_conflicts({src_id, dst_id}, nodes))
        if reasons:
            rejected.append(
                {"edge_id": edge_id, "src": src_id, "dst": dst_id, "reasons": reasons}
            )
            continue

        assert src is not None and dst is not None
        method = _positive_verification_method(edge, src, dst)
        if method is None:
            reread.append(
                {
                    "edge_id": edge_id,
                    "src": src_id,
                    "dst": dst_id,
                    "src_evidence_refs": list(src.get("evidence_refs") or []),
                    "dst_evidence_refs": list(dst.get("evidence_refs") or []),
                    "reason": "grounded compatible pair lacks positive identity proof",
                    "priority": _priority(edge),
                }
            )
            continue

        payload = dict(edge.get("payload") or {})
        payload["identity_verified"] = True
        payload["identity_verification"] = {
            "passed": True,
            "method": method,
            "src_mention_id": src.get("mention_id") or _payload(src).get("mention_id"),
            "dst_mention_id": dst.get("mention_id") or _payload(dst).get("mention_id"),
        }
        edge["payload"] = payload
        edge["identity_verified"] = True
        accepted.append(
            {
                "edge_id": edge_id,
                "src": src_id,
                "dst": dst_id,
                "method": method,
            }
        )
    reread.sort(key=lambda row: (-float(row["priority"]), str(row["edge_id"])))
    return output, IdentityVerificationReport(
        candidate_count=candidate_count,
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        targeted_reread_queue=tuple(reread),
    )


def _positive_verification_method(
    edge: dict[str, Any],
    src: dict[str, Any],
    dst: dict[str, Any],
) -> str | None:
    src_instance = str(src.get("instance_id") or _payload(src).get("instance_id") or "")
    dst_instance = str(dst.get("instance_id") or _payload(dst).get("instance_id") or "")
    if src_instance and src_instance == dst_instance:
        return "shared_explicit_instance_id"
    reread = edge.get("targeted_reread")
    if not isinstance(reread, dict):
        reread = (edge.get("payload") or {}).get("targeted_reread")
    if not isinstance(reread, dict) or reread.get("passed") is not True:
        return None
    if not reread.get("src_evidence_ref") or not reread.get("dst_evidence_ref"):
        return None
    matched = reread.get("matched_attributes")
    if not isinstance(matched, list) or not any(str(value).strip() for value in matched):
        return None
    if reread.get("conflicts"):
        return None
    return "targeted_raw_video_reread"


def _priority(edge: dict[str, Any]) -> float:
    value = edge.get("confidence", 0.5)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.5
