"""Helpers for upgrading relation-teacher output to structured witnesses."""

from __future__ import annotations

from typing import Any

from .types import CausalWitness, ConfidenceComponents, MechanismKind


def witness_from_teacher_row(
    row: dict[str, Any],
    *,
    src: str,
    dst: str,
    evidence_refs: list[str],
) -> CausalWitness | None:
    """Create a witness only when the teacher supplied a real mechanism."""

    probabilities = row.get("probabilities") or {}
    causal_scores = {
        relation: float(probabilities.get(relation, 0.0))
        for relation in ("explains", "enables")
    }
    relation = max(causal_scores, key=causal_scores.get)
    if causal_scores[relation] < 0.5:
        return None

    try:
        mechanism = MechanismKind(str(row.get("mechanism") or "none"))
    except ValueError:
        return None
    if mechanism is MechanismKind.NONE:
        return None

    mechanism_detail = str(row.get("mechanism_detail") or "").strip()
    support = tuple(str(value) for value in row.get("minimal_support_set") or [])
    quotes = row.get("evidence_quotes") or {}
    state_delta = row.get("state_delta")
    alignment = row.get("mention_alignment") or []
    first_alignment = next(
        (value for value in alignment if isinstance(value, dict)),
        None,
    )
    before_state, after_state = _states_from_delta(state_delta)
    confidence = ConfidenceComponents(
        temporal_grounding=_probability(row.get("direction_confidence")),
        entity_continuity=_probability(row.get("entity_continuity")),
        state_delta_grounding=_probability(row.get("state_delta_confidence")),
        mechanism_visibility=_probability(row.get("mechanism_visibility")),
        effect_grounding=_probability(row.get("effect_grounding")),
        alternative_cause_penalty=_probability(row.get("alternative_cause_penalty")),
    )
    payload = {
        "witness_id": f"witness:{src}->{dst}:{relation}",
        "relation": relation,
        "cause_event_id": src,
        "effect_event_id": dst,
        "mechanism": mechanism,
        "mechanism_detail": mechanism_detail,
        "evidence_refs": tuple(evidence_refs),
        "minimal_support_set": support,
        "evidence_quotes": {
            "src": str(quotes.get("src") or ""),
            "dst": str(quotes.get("dst") or ""),
            **(
                {"bridge": str(quotes["bridge"])}
                if quotes.get("bridge")
                else {}
            ),
        },
        "mechanism_event_id": (
            str(row["mechanism_event_id"]) if row.get("mechanism_event_id") else None
        ),
        "affected_entity": (
            {
                "src_mention_id": str(first_alignment.get("src_mention") or ""),
                "dst_mention_id": str(first_alignment.get("dst_mention") or ""),
            }
            if first_alignment
            else None
        ),
        "before_state": before_state,
        "after_state": after_state,
        "precondition": (
            str(row["precondition"]) if row.get("precondition") else None
        ),
        "state_bridge": (
            str(row["state_bridge"])
            if row.get("state_bridge")
            else str(state_delta) if state_delta else None
        ),
        "confidence_components": confidence,
        "alternative_explanations": tuple(
            str(value) for value in row.get("alternative_explanations") or []
        ),
    }
    try:
        return CausalWitness(**payload)
    except (TypeError, ValueError):
        return None


def witness_from_provenance(provenance: dict[str, Any]) -> CausalWitness | None:
    raw = provenance.get("causal_witness")
    if not isinstance(raw, dict):
        verification = provenance.get("verification")
        if isinstance(verification, dict):
            raw = verification.get("causal_witness")
    if not isinstance(raw, dict):
        return None
    try:
        return CausalWitness.from_dict(raw)
    except (TypeError, ValueError):
        return None


def _states_from_delta(
    value: Any,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if not isinstance(value, dict):
        return None, None
    attribute = str(value.get("attribute") or "")
    src_mention = str(value.get("src_mention") or "")
    dst_mention = str(value.get("dst_mention") or "")
    before = value.get("before")
    after = value.get("after")
    if not attribute or before is None or after is None:
        return None, None
    return (
        {
            "mention_id": src_mention,
            "attribute": attribute,
            "value": str(before),
        },
        {
            "mention_id": dst_mention,
            "attribute": attribute,
            "value": str(after),
        },
    )


def _probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return None
