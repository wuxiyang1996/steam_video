"""Categorical global entry localization over safe L1 address keys."""

from __future__ import annotations

from typing import Any, Protocol

from ..evidence.contracts import EvidenceMemory


class _CategoricalClient(Protocol):
    model: str

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> dict[str, Any]: ...


class ModelBackedEntryLocalizer:
    """Inspect every safe address once and return a bounded categorical frontier."""

    def __init__(self, client: _CategoricalClient, *, maximum_anchors: int = 8) -> None:
        if maximum_anchors < 1:
            raise ValueError("maximum_anchors must be positive")
        self.client = client
        self.maximum_anchors = maximum_anchors
        self.model_name = str(client.model)
        self.audits: list[dict[str, Any]] = []

    def localize(
        self,
        *,
        question: str,
        missing_roles: tuple[str, ...],
        memory: EvidenceMemory,
    ) -> tuple[str, ...]:
        aliases = {
            f"address_{index}": record.address.node_id
            for index, record in enumerate(memory.records)
        }
        base_payload = {
            "question": question,
            "missing_roles": list(missing_roles),
            "candidate_addresses": {
                alias: {
                    "event_family": record.address.event_family,
                    "semantic_key": record.address.semantic_key,
                    "structural_tags": list(record.address.structural_tags),
                    "embedding_available": record.address.embedding_ref is not None,
                }
                for alias, record in zip(
                    aliases,
                    memory.records,
                    strict=True,
                )
            },
            "allowed_output": {
                "status": ["located", "inconclusive"],
                "preferred": "distinct candidate address aliases",
                "rationale": "short categorical explanation",
            },
            "maximum_distinct_addresses": str(self.maximum_anchors),
        }
        result = None
        repair_count = 0
        request_payload = base_payload
        for attempt in range(2):
            candidate = self.client.complete_json(
                task=(
                    "Choose a minimal categorical frontier of entry addresses for the "
                    "question. Inspect every supplied safe semantic address. Select "
                    "representative starts for distinct plausible event sequences; "
                    "later evidence is reached by graph navigation. Return exactly "
                    "status, preferred, and rationale. Do not rank, score, or apply "
                    "Top-K, and do not claim an address key is acquired evidence."
                    if attempt == 0
                    else "Repair the entry localization JSON to exactly match the "
                    "required schema. Copy only valid candidate aliases."
                ),
                payload=request_payload,
            )
            if set(candidate) == {"status", "preferred", "rationale"}:
                result = candidate
                break
            if attempt == 1:
                raise ValueError("entry localization fields do not match schema")
            repair_count += 1
            request_payload = {
                **base_payload,
                "repair_feedback": {
                    "validation_error": (
                        "return exactly status, preferred, and rationale"
                    ),
                    "invalid_response": candidate,
                },
            }
        assert result is not None
        status = str(result.get("status"))
        preferred = result.get("preferred")
        if status not in {"located", "inconclusive"}:
            raise ValueError("entry localization status is invalid")
        if not isinstance(preferred, list) or not all(
            isinstance(value, str) for value in preferred
        ):
            raise ValueError("entry localization preferred must be an alias list")
        preferred = list(dict.fromkeys(preferred))
        if len(preferred) > self.maximum_anchors:
            raise ValueError("entry localization exceeds its categorical bound")
        if status == "located" and not preferred:
            raise ValueError("located entry frontier cannot be empty")
        if any(value not in aliases for value in preferred):
            raise ValueError("entry localization returned an unknown address")
        selected = tuple(aliases[value] for value in preferred)
        self.audits.append(
            {
                "candidate_address_count": len(aliases),
                "selected_anchor_count": len(selected),
                "selected_node_ids": list(selected),
                "status": status,
                "all_addresses_inspected": True,
                "top_k_applied": False,
                "numeric_score_used": False,
                "unread_evidence_value_exposed": False,
                "schema_repair_count": repair_count,
            }
        )
        return selected
