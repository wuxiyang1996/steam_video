"""Categorical global entry localization separated from local graph reasoning."""

from __future__ import annotations

from typing import Any

from .action_compiler import visible_graph_nodes
from .contracts import RetainedEvidenceGraph
from .gpt_oss import CategoricalJSONClient
from .model_input import node_key_for_localization


class GPTOSSEntryLocalizer:
    """Select a bounded set of grounded entry addresses without numeric scoring."""

    def __init__(self, client: CategoricalJSONClient, *, maximum_anchors: int = 8) -> None:
        if maximum_anchors < 1:
            raise ValueError("maximum_anchors must be positive")
        self.client = client
        self.maximum_anchors = maximum_anchors
        self.audits: list[dict[str, Any]] = []

    def localize(
        self,
        *,
        question: str,
        missing_roles: tuple[str, ...],
        graph: RetainedEvidenceGraph,
    ) -> tuple[str, ...]:
        nodes = visible_graph_nodes(graph)
        aliases = {
            f"anchor_{_letters(index)}": node.node_id
            for index, node in enumerate(nodes)
        }
        node_by_id = {node.node_id: node for node in nodes}
        payload = {
            "question": question,
            "missing_roles": list(missing_roles),
            "candidate_addresses": {
                alias: {
                    "semantic_key": node_key_for_localization(node_by_id[node_id]).semantic_key,
                    "structural_tags": list(
                        node_key_for_localization(node_by_id[node_id]).structural_tags
                    ),
                }
                for alias, node_id in aliases.items()
            },
            "required_output": {
                "only_keys": ["status", "preferred", "rationale"],
                "status": ["located", "inconclusive"],
                "preferred": "candidate address aliases only",
            },
            "required_contract": {
                "localization_is_separate_from_multihop_reasoning": True,
                "semantic_addresses_only_no_unread_evidence_values": True,
                "select_only_directly_relevant_entry_addresses": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
                "no_fixed_ranking_or_top_k": True,
            },
        }
        result: dict[str, Any] | None = None
        alias_prefix_normalization_count = 0
        for attempt in range(2):
            candidate = self.client.complete_json(
                task=(
                    "Locate a small categorical set of grounded evidence entry addresses. "
                    "This establishes graph cursors only; do not plan later hops."
                    if attempt == 0
                    else "Repair the localization response using only the required fields."
                ),
                payload=payload,
            )
            try:
                if set(candidate) != {"status", "preferred", "rationale"}:
                    raise ValueError("entry localization fields do not match schema")
                status = str(candidate["status"])
                if status not in {"located", "inconclusive"}:
                    raise ValueError("entry localization status is invalid")
                preferred = candidate["preferred"]
                if not isinstance(preferred, list) or any(
                    not isinstance(value, str) for value in preferred
                ):
                    raise ValueError("entry localization returned an unknown address")
                normalized = [
                    value
                    if value in aliases
                    else (
                        f"anchor_{value}"
                        if f"anchor_{value}" in aliases
                        else value
                    )
                    for value in preferred
                ]
                if any(value not in aliases for value in normalized):
                    raise ValueError("entry localization returned an unknown address")
                alias_prefix_normalization_count = sum(
                    left != right for left, right in zip(preferred, normalized)
                )
                candidate = {**candidate, "preferred": normalized}
                preferred = normalized
                if len(preferred) != len(set(preferred)):
                    raise ValueError("entry localization returned duplicate addresses")
                if len(preferred) > self.maximum_anchors:
                    raise ValueError("entry localization frontier exceeds its contract")
                if status == "located" and not preferred:
                    raise ValueError("located entry frontier must not be empty")
                if status == "inconclusive" and preferred:
                    raise ValueError("inconclusive entry frontier must be empty")
                result = candidate
                break
            except ValueError:
                if attempt == 1:
                    raise
        assert result is not None
        selected = tuple(aliases[value] for value in result["preferred"])
        self.audits.append(
            {
                "candidate_address_count": len(nodes),
                "selected_anchor_count": len(selected),
                "selected_node_ids": list(selected),
                "status": result["status"],
                "top_k_applied": False,
                "numeric_score_used": False,
                "alias_prefix_normalization_count": (
                    alias_prefix_normalization_count
                ),
            }
        )
        return selected


def _letters(index: int) -> str:
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))
