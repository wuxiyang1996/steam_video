"""Explicit, auditable visibility regimes for zero-shot IWM diagnostics."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .contracts import LegalGraphAction, RetainedEvidenceGraph
from .model_input import build_iwm_graph_input
from .contracts import CursorBeliefState


class TransitionInputRegime(str, Enum):
    """Information available before executing a legal evidence action."""

    SEMANTIC_ADDRESS = "semantic_address"
    RICH_QUESTION_INDEPENDENT = "rich_question_independent"


_RICH_METADATA_FIELDS = (
    "predicate",
    "action_kind",
    "participants",
    "states",
    "state_change",
    "actor",
    "target",
    "objects",
    "location",
    "modality",
    "source_type",
)


def transition_action_payload(
    action: LegalGraphAction,
    belief: CursorBeliefState,
    graph: RetainedEvidenceGraph,
    regime: TransitionInputRegime,
) -> dict[str, Any]:
    graph_input = build_iwm_graph_input(belief, graph, (action,))
    target = next(
        (row for row in graph_input.nodes if row.key.node_id == action.target_id),
        None,
    )
    payload: dict[str, Any] = {
        "kind": action.kind.value,
        "relation": action.relation,
        "target_semantic_key": target.key.semantic_key if target else None,
        "target_structural_tags": list(target.key.structural_tags) if target else [],
        "target_time_span": (
            {"start_s": target.key.start_s, "end_s": target.key.end_s}
            if target
            else None
        ),
        "input_regime": regime.value,
    }
    if regime is TransitionInputRegime.RICH_QUESTION_INDEPENDENT and target:
        node = graph.node_by_id[target.key.node_id]
        payload["question_independent_l1_descriptor"] = {
            "text": str(node.text or ""),
            "metadata": {
                field: node.metadata[field]
                for field in _RICH_METADATA_FIELDS
                if node.metadata.get(field) not in (None, "", [], {})
            },
            "provenance_type": str(node.provenance.get("producer") or ""),
        }
    return payload


def visibility_contract(regime: TransitionInputRegime) -> dict[str, Any]:
    return {
        "regime": regime.value,
        "question_independent_graph_required": True,
        "hidden_clue_intervals_visible": False,
        "hidden_answer_visible": False,
        "post_read_vlm_observation_visible": False,
        "full_l1_descriptor_visible_before_read": (
            regime is TransitionInputRegime.RICH_QUESTION_INDEPENDENT
        ),
        "oracle_observation_is_posthoc_only": True,
        "metadata_allowlist": list(_RICH_METADATA_FIELDS),
    }


def audit_graph_visibility(
    graph: RetainedEvidenceGraph,
    regime: TransitionInputRegime,
) -> dict[str, Any]:
    forbidden = {
        "answer",
        "answer_text",
        "correct_answer",
        "clue",
        "clues",
        "clue_intervals",
        "hidden_supervision",
        "ground_truth",
    }
    exposed_fields = set(_RICH_METADATA_FIELDS) if (
        regime is TransitionInputRegime.RICH_QUESTION_INDEPENDENT
    ) else set()
    exposed_forbidden = sorted(exposed_fields & forbidden)
    graph_contract = graph.metadata.get("question_independent_contract")
    return {
        **visibility_contract(regime),
        "graph_question_independent_contract": graph_contract,
        "exposed_forbidden_fields": exposed_forbidden,
        "leakage_detected": bool(exposed_forbidden) or graph_contract is False,
        "node_count": len(graph.nodes),
    }
