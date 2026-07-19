"""Conservative deterministic verification for candidate-causal relations."""

from __future__ import annotations

from typing import Any, Iterable

from ..causal_witness import witness_from_provenance
from ..types import MechanismKind, MemoryNode, RelationBelief, RelationType
from .entity_state_verifier import VerifierResult, _common_problems, _norm, _support_value


class CausalVerifier:
    """Object-oriented facade for candidate-causal verification."""

    def verify(
        self,
        belief: RelationBelief,
        src: MemoryNode,
        dst: MemoryNode,
        relation: str | RelationType,
    ) -> VerifierResult:
        return verify_causal_relation(belief, src, dst, relation)


def verify_explains(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    *,
    l1_by_id: dict[str, MemoryNode] | None = None,
) -> VerifierResult:
    """Verify grounded direction plus an explicit state bridge or mechanism."""

    return _verify_causal(
        belief,
        src,
        dst,
        relation=RelationType.EXPLAINS.value,
        support_keys=("state_bridge", "mechanism_detail"),
        l1_by_id=l1_by_id,
    )


def verify_enables(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    *,
    l1_by_id: dict[str, MemoryNode] | None = None,
) -> VerifierResult:
    """Verify grounded direction plus an explicit precondition or mechanism."""

    return _verify_causal(
        belief,
        src,
        dst,
        relation=RelationType.ENABLES.value,
        support_keys=("precondition", "mechanism_detail"),
        l1_by_id=l1_by_id,
    )


def verify_causal_relation(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    relation: str | RelationType,
    *,
    l1_by_id: dict[str, MemoryNode] | None = None,
) -> VerifierResult:
    """Dispatch one candidate-causal verifier."""

    name = relation.value if isinstance(relation, RelationType) else str(relation)
    if name == RelationType.EXPLAINS.value:
        return verify_explains(belief, src, dst, l1_by_id=l1_by_id)
    if name == RelationType.ENABLES.value:
        return verify_enables(belief, src, dst, l1_by_id=l1_by_id)
    return VerifierResult(False, (f"unsupported causal relation {name!r}",), name)


def _verify_causal(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    *,
    relation: str,
    support_keys: tuple[str, ...],
    l1_by_id: dict[str, MemoryNode] | None,
) -> VerifierResult:
    problems = _common_problems(belief, src, dst, relation)
    witness = witness_from_provenance(belief.provenance)
    if witness is not None:
        if witness.cause_event_id != src.node_id or witness.effect_event_id != dst.node_id:
            problems.append("causal witness endpoints do not match the relation")
        if witness.relation != relation:
            problems.append("causal witness relation type does not match the proposal")
        if (
            witness.mechanism is MechanismKind.OBSERVABLE_PRECONDITION
            and _is_perception_only_effect(dst)
        ):
            problems.append(
                "observable precondition merely restates a later perception of the state"
            )
    if src.video_id != dst.video_id:
        problems.append("candidate-causal nodes must belong to the same video")
    if src.time_span.end_s > dst.time_span.start_s + 1e-6:
        problems.append("source must end before the destination begins")
    problems.extend(_l1_temporal_direction_problems(belief, src, dst, l1_by_id))

    warrant = (belief.warrant or "").strip()
    if not warrant:
        problems.append("a non-empty warrant is required")

    problems.extend(_evidence_ref_problems(belief, src, dst))
    problems.extend(_minimal_support_problems(belief, src, dst))
    quote_problems, grounded_quotes = _quote_grounding_problems(belief, src, dst, warrant)
    problems.extend(quote_problems)

    support_name, support = _first_explicit_support(
        belief,
        support_keys,
        witness=witness,
    )
    if support is None:
        choices = " or ".join(support_keys)
        problems.append(f"{relation} requires an explicit {choices}")
    elif not _substantive_support(support):
        problems.append(
            f"{support_name} is only temporal, shared-entity, or narrative setup"
        )

    if problems:
        return VerifierResult(False, tuple(_unique(problems)), relation)
    return VerifierResult(
        True,
        (
            f"source precedes destination and both endpoint quotes are grounded",
            f"explicit {support_name} supplies the causal bridge",
            f"grounded quotes: {grounded_quotes[0]!r} -> {grounded_quotes[1]!r}",
        ),
        relation,
    )


def _evidence_ref_problems(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> list[str]:
    if not belief.evidence_refs:
        return ["candidate-causal relation must cite evidence_refs"]
    cited = set(belief.evidence_refs)
    src_refs = _node_evidence_refs(src)
    dst_refs = _node_evidence_refs(dst)
    problems: list[str] = []
    if not cited.intersection(src_refs):
        problems.append("evidence_refs do not ground the source node")
    if not cited.intersection(dst_refs):
        problems.append("evidence_refs do not ground the destination node")
    return problems


def _l1_temporal_direction_problems(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    l1_by_id: dict[str, MemoryNode] | None,
) -> list[str]:
    if not l1_by_id:
        return []
    src_evidence = [l1_by_id[ref] for ref in src.source_segments if ref in l1_by_id]
    dst_evidence = [l1_by_id[ref] for ref in dst.source_segments if ref in l1_by_id]
    if not src_evidence or not dst_evidence:
        return ["causal endpoints lack resolvable L1 temporal evidence"]
    src_end = max(node.time_span.end_s for node in src_evidence)
    dst_start = min(node.time_span.start_s for node in dst_evidence)
    if src_end > dst_start + 1e-6:
        visual = belief.provenance.get("visual_verification")
        if isinstance(visual, dict) and visual.get("status") == "passed":
            checks = visual.get("checks")
            if isinstance(checks, dict) and checks.get("temporal_order_visible") is True:
                return []
        return [
            "causal direction relies on unverified timing refined inside a coarse L1 span"
        ]
    return []


def _node_evidence_refs(node: MemoryNode) -> set[str]:
    refs = {node.node_id, *node.source_segments}
    metadata_refs = node.metadata.get("evidence_refs")
    if isinstance(metadata_refs, (list, tuple, set)):
        refs.update(str(value) for value in metadata_refs)
    return refs


def _minimal_support_problems(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> list[str]:
    witness = witness_from_provenance(belief.provenance)
    raw: Any = (
        list(witness.minimal_support_set)
        if witness is not None
        else _support_value(belief, "minimal_support_set")
    )
    if not isinstance(raw, (list, tuple, set)):
        return ["minimal_support_set must contain the two event endpoint IDs"]
    support = {str(value) for value in raw}
    expected = {src.node_id, dst.node_id}
    if not expected.issubset(support):
        return [
            "minimal_support_set must include the source and destination event IDs"
        ]
    if witness is not None and witness.mechanism_event_id:
        if witness.mechanism_event_id not in support:
            return ["minimal_support_set must include the mechanism event ID"]
    return []


def _quote_grounding_problems(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    warrant: str,
) -> tuple[list[str], tuple[str, str]]:
    quote_map = _evidence_quotes(belief, src, dst)
    problems: list[str] = []
    grounded: list[str] = []
    for side, node in (("src", src), ("dst", dst)):
        quotes = quote_map[side]
        if not quotes:
            problems.append(f"evidence_quotes must include a {side} quote")
            grounded.append("")
            continue
        matching = [
            quote
            for quote in quotes
            if _quote_in_node(quote, node) and _norm(quote) in _norm(warrant)
        ]
        if not matching:
            problems.append(
                f"{side} evidence quote must occur in both the node text and warrant"
            )
            grounded.append("")
            continue
        grounded.append(matching[0])
    return problems, (grounded[0], grounded[1])


def _evidence_quotes(
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {"src": [], "dst": []}
    witness = witness_from_provenance(belief.provenance)
    raw: Any = (
        witness.evidence_quotes
        if witness is not None
        else _support_value(belief, "evidence_quotes")
    )
    if isinstance(raw, dict):
        for side, node in (("src", src), ("dst", dst)):
            value = raw.get(side, raw.get(node.node_id))
            result[side].extend(_strings(value))
    elif isinstance(raw, (list, tuple)):
        if len(raw) == 2 and all(isinstance(item, str) for item in raw):
            result["src"].extend(_strings(raw[0]))
            result["dst"].extend(_strings(raw[1]))
        for item in raw:
            if not isinstance(item, dict):
                continue
            side = str(item.get("side") or "")
            node_id = str(item.get("node_id") or "")
            target = side if side in result else (
                "src" if node_id == src.node_id else "dst" if node_id == dst.node_id else ""
            )
            if target:
                result[target].extend(_strings(item.get("quote") or item.get("quotes")))

    # Also accept concise direct fields for producers that do not emit a quote map.
    result["src"].extend(_strings(_support_value(belief, "src_quote")))
    result["dst"].extend(_strings(_support_value(belief, "dst_quote")))
    return {side: _unique(quotes) for side, quotes in result.items()}


def _quote_in_node(quote: str, node: MemoryNode) -> bool:
    needle = _norm(quote)
    return bool(needle) and any(needle in _norm(text) for text in _node_texts(node))


def _node_texts(node: MemoryNode) -> list[str]:
    texts: list[str] = []
    if node.text:
        texts.append(node.text)
    predicate = node.metadata.get("predicate")
    if isinstance(predicate, str):
        texts.append(predicate)
    return texts


def _is_perception_only_effect(node: MemoryNode) -> bool:
    text = _norm(" ".join(_node_texts(node)))
    perception_terms = (
        " see ",
        " sees ",
        " saw ",
        " look ",
        " looks ",
        " notice ",
        " notices ",
        " observe ",
        " observes ",
        " watch ",
        " watches ",
    )
    padded = f" {text} "
    return any(term in padded for term in perception_terms)


def _first_explicit_support(
    belief: RelationBelief,
    keys: tuple[str, ...],
    *,
    witness: Any = None,
) -> tuple[str, Any | None]:
    for key in keys:
        if witness is not None:
            value = getattr(witness, key, None)
            if value is not None:
                return key, value
            if key == "mechanism_detail" and witness.mechanism_detail:
                return key, witness.mechanism_detail
        value = _support_value(belief, key)
        if value is not None:
            return key, value
    return keys[0], None


def _substantive_support(value: Any) -> bool:
    fragments = _support_fragments(value)
    if not fragments:
        return False
    normalized = " ".join(_norm(fragment) for fragment in fragments)
    if len(normalized.split()) < 3:
        return False
    generic_only = (
        "temporal",
        "happens before",
        "occurs before",
        "comes before",
        "earlier than",
        "later than",
        "same entity",
        "same person",
        "same object",
        "shared entity",
        "narrative setup",
        "introduces the",
        "sets up the story",
    )
    return not any(phrase in normalized for phrase in generic_only)


def _support_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        return [
            str(item).strip()
            for item in value.values()
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
    if isinstance(value, (list, tuple)):
        return [
            str(item).strip()
            for item in value
            if isinstance(item, (str, int, float)) and str(item).strip()
        ]
    return []


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(values))
