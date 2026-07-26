"""Categorical global entry localization separated from local graph reasoning."""

from __future__ import annotations

import re
from typing import Any

from .action_compiler import visible_graph_nodes
from .contracts import RetainedEvidenceGraph
from .gpt_oss import CategoricalJSONClient
from .model_input import node_key_for_localization


class GPTOSSEntryLocalizer:
    """Select a bounded set of grounded entry addresses without numeric scoring."""

    def __init__(
        self, client: CategoricalJSONClient, *, maximum_anchors: int = 8
    ) -> None:
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
                    "semantic_key": node_key_for_localization(
                        node_by_id[node_id]
                    ).semantic_key,
                    "structural_tags": list(
                        node_key_for_localization(node_by_id[node_id]).structural_tags
                    ),
                }
                for alias, node_id in aliases.items()
            },
            "required_output": {
                "only_keys": ["status", "preferred", "rationale"],
                "status": ["located", "inconclusive"],
                "preferred": (
                    "either a flat alias list or a missing-role-to-alias mapping; "
                    "use a minimal complete frontier of nonredundant entry cursors "
                    "and multiple aliases only for semantically distinct event sequences"
                ),
            },
            "required_contract": {
                "localization_is_separate_from_multihop_reasoning": True,
                "semantic_addresses_only_no_unread_evidence_values": True,
                "select_only_directly_relevant_entry_addresses": True,
                "preserve_multiple_plausible_entries_when_inconclusive": True,
                "one_representative_cursor_per_missing_role_or_event_sequence": True,
                "do_not_enumerate_temporal_repetitions_of_the_same_event": True,
                "later_related_events_are_reached_by_graph_hops": True,
                "preferred_is_not_a_list_of_all_relevant_evidence": True,
                "no_numeric_reward_score_probability_confidence_or_utility": True,
                "no_fixed_ranking_or_top_k": True,
            },
        }
        result: dict[str, Any] | None = None
        alias_prefix_normalization_count = 0
        role_key_normalization_count = 0
        role_conditioned_output = False
        for attempt in range(2):
            candidate = self.client.complete_json(
                task=(
                    "Locate a categorical nonredundant frontier of grounded entry "
                    "cursors. Choose representative starts for distinct missing roles "
                    "or event sequences, not every repeated relevant event; temporal "
                    "and correlated repetitions are reached by later graph hops. This "
                    "establishes cursors only and does not plan later hops."
                    if attempt == 0
                    else "Repair the localization response using only the required "
                    "fields and nonredundant entry-cursor contract."
                ),
                payload=payload,
            )
            try:
                if set(candidate) != {"status", "preferred", "rationale"}:
                    raise ValueError("entry localization fields do not match schema")
                status = str(candidate["status"])
                if status not in {"located", "inconclusive"}:
                    raise ValueError("entry localization status is invalid")
                raw_preferred = candidate["preferred"]
                role_conditioned_output = isinstance(raw_preferred, dict)
                if role_conditioned_output:
                    if any(
                        not isinstance(role, str)
                        or not isinstance(values, (str, list))
                        or (
                            isinstance(values, list)
                            and any(not isinstance(value, str) for value in values)
                        )
                        for role, values in raw_preferred.items()
                    ):
                        raise ValueError(
                            "entry localization returned an invalid role frontier"
                        )
                    role_aliases: dict[str, str] = {}
                    for role in missing_roles:
                        normalized_role = _normalized_role(role)
                        if normalized_role in role_aliases:
                            raise ValueError(
                                "missing-role normalization is ambiguous"
                            )
                        role_aliases[normalized_role] = role
                    normalized_frontier: dict[str, str | list[str]] = {}
                    role_key_normalization_count = 0
                    for raw_role, values in raw_preferred.items():
                        resolved_role = (
                            raw_role
                            if raw_role in missing_roles
                            else role_aliases.get(_normalized_role(raw_role))
                        )
                        if resolved_role is None:
                            raise ValueError(
                                "entry localization returned an unknown missing role"
                            )
                        if resolved_role in normalized_frontier:
                            raise ValueError(
                                "entry localization returned duplicate missing roles"
                            )
                        role_key_normalization_count += raw_role != resolved_role
                        normalized_frontier[resolved_role] = values
                    expanded = []
                    for role in missing_roles:
                        values = normalized_frontier.get(role, [])
                        expanded.extend([values] if isinstance(values, str) else values)
                    preferred = list(dict.fromkeys(expanded))
                elif isinstance(raw_preferred, list) and all(
                    isinstance(value, str) for value in raw_preferred
                ):
                    preferred = raw_preferred
                    if len(preferred) != len(set(preferred)):
                        raise ValueError(
                            "entry localization returned duplicate addresses"
                        )
                else:
                    raise ValueError("entry localization returned an unknown address")
                normalized = [
                    value
                    if value in aliases
                    else (f"anchor_{value}" if f"anchor_{value}" in aliases else value)
                    for value in preferred
                ]
                if any(value not in aliases for value in normalized):
                    raise ValueError("entry localization returned an unknown address")
                alias_prefix_normalization_count = sum(
                    left != right for left, right in zip(preferred, normalized)
                )
                candidate = {**candidate, "preferred": normalized}
                preferred = normalized
                if len(preferred) > self.maximum_anchors:
                    raise ValueError("entry localization frontier exceeds its contract")
                if status == "located" and not preferred:
                    raise ValueError("located entry frontier must not be empty")
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
                "alias_prefix_normalization_count": (alias_prefix_normalization_count),
                "role_conditioned_output": role_conditioned_output,
                "role_key_normalization_count": role_key_normalization_count,
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


def _normalized_role(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.casefold()).strip("_")
