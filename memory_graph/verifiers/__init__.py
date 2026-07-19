"""Deterministic admission checks for semantic memory-graph relations."""

from __future__ import annotations

from ..types import MemoryNode, RelationBelief, RelationType
from .causal_verifier import (
    CausalVerifier,
    verify_causal_relation,
    verify_enables,
    verify_explains,
)
from .dependency_verifier import (
    verify_dependency_relation,
    verify_response_candidate,
    verify_transition_support,
)
from .entity_state_verifier import (
    EntityStateVerifier,
    VerifierResult,
    verify_contradicts,
    verify_entity_state_relation,
    verify_same_entity,
    verify_state_transition,
)

__all__ = [
    "CausalVerifier",
    "EntityStateVerifier",
    "VerifierResult",
    "verify_causal_relation",
    "verify_contradicts",
    "verify_dependency_relation",
    "verify_enables",
    "verify_entity_state_relation",
    "verify_explains",
    "verify_relation",
    "verify_response_candidate",
    "verify_same_entity",
    "verify_state_transition",
    "verify_transition_support",
]


def verify_relation(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    relation: str | RelationType | None = None,
    *,
    l1_by_id: dict[str, MemoryNode] | None = None,
) -> VerifierResult:
    """Verify one supported semantic relation without model calls or randomness."""

    if relation is None:
        if len(belief.relation_probabilities) != 1:
            return VerifierResult(
                False,
                ("relation must be explicit when belief has multiple labels",),
                "unknown",
            )
        name = next(iter(belief.relation_probabilities))
    else:
        name = relation.value if isinstance(relation, RelationType) else str(relation)
    if name in {
        RelationType.SAME_ENTITY.value,
        RelationType.STATE_TRANSITION.value,
        RelationType.CONTRADICTS.value,
    }:
        return verify_entity_state_relation(belief, src, dst, name)
    if name in {RelationType.EXPLAINS.value, RelationType.ENABLES.value}:
        return verify_causal_relation(
            belief,
            src,
            dst,
            name,
            l1_by_id=l1_by_id,
        )
    if name in {
        RelationType.TRANSITION_SUPPORT.value,
        RelationType.RESPONSE_CANDIDATE.value,
    }:
        return verify_dependency_relation(belief, src, dst, name)
    return VerifierResult(False, (f"unsupported verifier relation {name!r}",), name)
