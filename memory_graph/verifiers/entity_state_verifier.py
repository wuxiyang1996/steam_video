"""Deterministic verification for entity and visible-state relations."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Iterable

from ..types import MemoryNode, RelationBelief, RelationType


@dataclass(frozen=True)
class VerifierResult:
    """A stable, serializable verifier decision."""

    passed: bool
    reasons: tuple[str, ...]
    relation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "relation": self.relation,
        }


@dataclass(frozen=True)
class _Participant:
    mention_id: str
    entity_type: str
    surface: str


@dataclass(frozen=True)
class _State:
    mention_id: str
    attribute: str
    value: str
    polarity: str


class EntityStateVerifier:
    """Object-oriented facade for entity/state relation verification."""

    def verify(
        self,
        belief: RelationBelief,
        src: MemoryNode,
        dst: MemoryNode,
        relation: str | RelationType,
    ) -> VerifierResult:
        return verify_entity_state_relation(belief, src, dst, relation)


def verify_same_entity(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> VerifierResult:
    """Require an aligned participant with equal normalized type and surface."""

    relation = RelationType.SAME_ENTITY.value
    problems = _common_problems(belief, src, dst, relation)
    pairs, pair_problems = _aligned_participants(belief, src, dst)
    problems.extend(pair_problems)
    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)
    matches = [
        (left, right)
        for left, right in pairs
        if _norm(left.entity_type) == _norm(right.entity_type)
        and _norm(left.surface) == _norm(right.surface)
    ]
    if not matches:
        return VerifierResult(
            False,
            ("no aligned participant has matching entity_type and surface",),
            relation,
        )
    left, right = matches[0]
    return VerifierResult(
        True,
        (
            f"participant {left.mention_id!r}->{right.mention_id!r} "
            f"matches type {left.entity_type!r} and surface {left.surface!r}",
        ),
        relation,
    )


def verify_state_transition(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> VerifierResult:
    """Require a value change on one attribute of the same entity."""

    relation = RelationType.STATE_TRANSITION.value
    problems = _common_problems(belief, src, dst, relation)
    pairs, pair_problems = _aligned_participants(belief, src, dst)
    problems.extend(pair_problems)
    src_states, src_state_problems = _states(src)
    dst_states, dst_state_problems = _states(dst)
    problems.extend(src_state_problems)
    problems.extend(dst_state_problems)
    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)

    for left, right in _identity_matches(pairs):
        for before in src_states:
            if before.mention_id != left.mention_id:
                continue
            for after in dst_states:
                if after.mention_id != right.mention_id:
                    continue
                if (
                    _norm(before.attribute) == _norm(after.attribute)
                    and _norm(before.value) != _norm(after.value)
                ):
                    return VerifierResult(
                        True,
                        (
                            f"same entity changes {before.attribute!r} from "
                            f"{before.value!r} to {after.value!r}",
                        ),
                        relation,
                    )
    return VerifierResult(
        False,
        ("no same-entity state has the same attribute and a changed value",),
        relation,
    )


def verify_contradicts(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> VerifierResult:
    """Require incompatible polarity or values for one same-entity attribute."""

    relation = RelationType.CONTRADICTS.value
    problems = _common_problems(belief, src, dst, relation)
    pairs, pair_problems = _aligned_participants(belief, src, dst)
    problems.extend(pair_problems)
    src_states, src_state_problems = _states(src)
    dst_states, dst_state_problems = _states(dst)
    problems.extend(src_state_problems)
    problems.extend(dst_state_problems)
    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)

    for left, right in _identity_matches(pairs):
        for first in src_states:
            if first.mention_id != left.mention_id:
                continue
            for second in dst_states:
                if second.mention_id != right.mention_id:
                    continue
                if _norm(first.attribute) != _norm(second.attribute):
                    continue
                opposite_polarity = first.polarity != second.polarity
                different_value = _norm(first.value) != _norm(second.value)
                if opposite_polarity or different_value:
                    conflict = (
                        "opposite polarity"
                        if opposite_polarity
                        else f"incompatible values {first.value!r} and {second.value!r}"
                    )
                    return VerifierResult(
                        True,
                        (f"same-entity attribute {first.attribute!r} has {conflict}",),
                        relation,
                    )
    return VerifierResult(
        False,
        ("no same-entity attribute has conflicting polarity or value",),
        relation,
    )


def verify_entity_state_relation(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    relation: str | RelationType,
) -> VerifierResult:
    """Dispatch one deterministic entity/state relation verifier."""

    name = relation.value if isinstance(relation, RelationType) else str(relation)
    verifiers = {
        RelationType.SAME_ENTITY.value: verify_same_entity,
        RelationType.STATE_TRANSITION.value: verify_state_transition,
        RelationType.CONTRADICTS.value: verify_contradicts,
    }
    if name not in verifiers:
        return VerifierResult(False, (f"unsupported entity/state relation {name!r}",), name)
    return verifiers[name](belief, src, dst)


def _common_problems(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    relation: str,
) -> list[str]:
    problems: list[str] = []
    if belief.src != src.node_id or belief.dst != dst.node_id:
        problems.append("belief endpoints do not match the supplied src/dst nodes")
    if src.node_type != "atomic_event" or dst.node_type != "atomic_event":
        problems.append("both nodes must have node_type 'atomic_event'")
    if relation not in belief.relation_probabilities:
        problems.append(f"belief does not contain relation {relation!r}")
    return problems


def _participants(node: MemoryNode) -> tuple[list[_Participant], list[str]]:
    raw = node.metadata.get("participants")
    if not isinstance(raw, (list, tuple)) or not raw:
        return [], [f"node {node.node_id!r} has no structured participants"]
    participants: list[_Participant] = []
    problems: list[str] = []
    for index, value in enumerate(raw):
        mention_id = _field(value, "mention_id")
        entity_type = _field(value, "entity_type")
        surface = _field(value, "surface")
        if not mention_id or not entity_type or not surface:
            problems.append(
                f"node {node.node_id!r} participant {index} lacks mention_id/type/surface"
            )
            continue
        participants.append(_Participant(mention_id, entity_type, surface))
    return participants, problems


def _states(node: MemoryNode) -> tuple[list[_State], list[str]]:
    raw = node.metadata.get("states")
    if not isinstance(raw, (list, tuple)) or not raw:
        return [], [f"node {node.node_id!r} has no structured states"]
    states: list[_State] = []
    problems: list[str] = []
    for index, value in enumerate(raw):
        mention_id = _field(value, "mention_id")
        attribute = _field(value, "attribute")
        state_value = _field(value, "value")
        polarity = _field(value, "polarity") or "positive"
        if not mention_id or not attribute or not state_value:
            problems.append(
                f"node {node.node_id!r} state {index} lacks mention_id/attribute/value"
            )
            continue
        if polarity not in {"positive", "negative"}:
            problems.append(f"node {node.node_id!r} state {index} has invalid polarity")
            continue
        states.append(_State(mention_id, attribute, state_value, polarity))
    return states, problems


def _aligned_participants(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> tuple[list[tuple[_Participant, _Participant]], list[str]]:
    src_participants, problems = _participants(src)
    dst_participants, dst_problems = _participants(dst)
    problems.extend(dst_problems)
    if problems:
        return [], problems

    src_by_id = {value.mention_id: value for value in src_participants}
    dst_by_id = {value.mention_id: value for value in dst_participants}
    alignment = _support_value(belief, "participant_alignment")
    if alignment is None:
        # Exact normalized type+surface is itself a deterministic alignment.
        return [
            (left, right)
            for left in src_participants
            for right in dst_participants
            if _norm(left.entity_type) == _norm(right.entity_type)
            and _norm(left.surface) == _norm(right.surface)
        ], []

    ids = _alignment_ids(alignment)
    if not ids:
        return [], ["participant_alignment is present but has no src/dst mention pair"]
    pairs: list[tuple[_Participant, _Participant]] = []
    for src_id, dst_id in ids:
        if src_id not in src_by_id or dst_id not in dst_by_id:
            problems.append(
                f"participant_alignment references unknown pair {src_id!r}->{dst_id!r}"
            )
            continue
        pairs.append((src_by_id[src_id], dst_by_id[dst_id]))
    return pairs, problems


def _identity_matches(
    pairs: Iterable[tuple[_Participant, _Participant]],
) -> list[tuple[_Participant, _Participant]]:
    return [
        (left, right)
        for left, right in pairs
        if _norm(left.entity_type) == _norm(right.entity_type)
        and _norm(left.surface) == _norm(right.surface)
    ]


def _alignment_ids(value: Any) -> list[tuple[str, str]]:
    if isinstance(value, dict):
        if "src" in value and "dst" in value:
            return [(str(value["src"]), str(value["dst"]))]
        if "src_mention_id" in value and "dst_mention_id" in value:
            return [
                (str(value["src_mention_id"]), str(value["dst_mention_id"]))
            ]
        return [(str(src), str(dst)) for src, dst in value.items()]
    if isinstance(value, (list, tuple)):
        pairs: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, dict) and "src" in item and "dst" in item:
                pairs.append((str(item["src"]), str(item["dst"])))
            elif (
                isinstance(item, dict)
                and "src_mention_id" in item
                and "dst_mention_id" in item
            ):
                pairs.append(
                    (str(item["src_mention_id"]), str(item["dst_mention_id"]))
                )
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
        return pairs
    return []


def _support_value(belief: RelationBelief, key: str) -> Any:
    if key in belief.provenance:
        return belief.provenance[key]
    verification = belief.provenance.get("verification")
    if isinstance(verification, dict):
        return verification.get(key)
    return None


def _field(value: Any, name: str) -> str:
    if isinstance(value, dict):
        raw = value.get(name)
    else:
        raw = getattr(value, name, None)
    return str(raw).strip() if raw is not None else ""


def _norm(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
