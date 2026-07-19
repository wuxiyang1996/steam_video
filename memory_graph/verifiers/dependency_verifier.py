"""Admission checks for non-causal predictive navigation dependencies."""

from __future__ import annotations

from ..types import MemoryNode, RelationBelief, RelationType
from .causal_verifier import _quote_grounding_problems
from .entity_state_verifier import (
    VerifierResult,
    _aligned_participants,
    _common_problems,
    _identity_matches,
    _participants,
    _support_value,
    _unique,
)


def verify_transition_support(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> VerifierResult:
    """Require grounded trajectory/state continuity without claiming causality."""

    relation = RelationType.TRANSITION_SUPPORT.value
    problems = _common_problems(belief, src, dst, relation)
    problems.extend(_temporal_problems(src, dst))
    basis = str(_support_value(belief, "dependency_basis") or "none")
    if basis not in {"entity_trajectory", "state_continuity"}:
        problems.append(
            "transition_support requires entity_trajectory or state_continuity basis"
        )
    pairs, pair_problems = _aligned_participants(belief, src, dst)
    problems.extend(pair_problems)
    if pairs and not list(_identity_matches(pairs)):
        problems.append("transition_support lacks a grounded same-entity alignment")
    warrant = (belief.warrant or "").strip()
    quote_problems, grounded = _quote_grounding_problems(
        belief,
        src,
        dst,
        warrant,
    )
    problems.extend(quote_problems)
    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)
    return VerifierResult(
        True,
        (
            f"grounded {basis} dependency supports navigation",
            f"endpoint evidence: {grounded[0]!r} -> {grounded[1]!r}",
        ),
        relation,
    )


def verify_response_candidate(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    *,
    max_gap_s: float = 15.0,
) -> VerifierResult:
    """Require a local visible action-response pair, without inferring intent."""

    relation = RelationType.RESPONSE_CANDIDATE.value
    problems = _common_problems(belief, src, dst, relation)
    problems.extend(_temporal_problems(src, dst))
    gap = dst.time_span.start_s - src.time_span.end_s
    if gap > max_gap_s:
        problems.append(f"response candidate gap exceeds {max_gap_s:g}s")
    basis = str(_support_value(belief, "dependency_basis") or "none")
    if basis != "observable_response":
        problems.append("response_candidate requires observable_response basis")
    _, src_problems = _participants(src)
    _, dst_problems = _participants(dst)
    problems.extend(src_problems)
    problems.extend(dst_problems)
    warrant = (belief.warrant or "").strip()
    quote_problems, grounded = _quote_grounding_problems(
        belief,
        src,
        dst,
        warrant,
    )
    problems.extend(quote_problems)
    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)
    return VerifierResult(
        True,
        (
            "temporally local visible action-response pair supports navigation",
            f"endpoint evidence: {grounded[0]!r} -> {grounded[1]!r}",
        ),
        relation,
    )


def verify_dependency_relation(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    relation: str | RelationType,
) -> VerifierResult:
    name = relation.value if isinstance(relation, RelationType) else str(relation)
    if name == RelationType.TRANSITION_SUPPORT.value:
        return verify_transition_support(belief, src, dst)
    if name == RelationType.RESPONSE_CANDIDATE.value:
        return verify_response_candidate(belief, src, dst)
    return VerifierResult(False, (f"unsupported dependency relation {name!r}",), name)


def _temporal_problems(src: MemoryNode, dst: MemoryNode) -> list[str]:
    if src.video_id != dst.video_id:
        return ["dependency endpoints must belong to the same video"]
    if src.time_span.start_s > dst.time_span.start_s:
        return ["dependency source must not start after destination"]
    return []
