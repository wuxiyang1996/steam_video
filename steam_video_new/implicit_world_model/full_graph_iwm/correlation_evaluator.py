"""GPT-OSS categorical relation proposal over every retained-node pair."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from memory_graph.correlation_overlay import (
    CategoricalCorrelationJudgment,
    CorrelationPair,
    CorrelationStatus,
    CorrelationType,
)

from .gpt_oss import CategoricalJSONClient, _reject_numeric_output


class GPTOSSCategoricalCorrelationEvaluator:
    """Propose typed edges without producing confidence or admitting them.

    Every input pair is evaluated.  Batching only controls request size; it is
    not candidate retrieval and cannot drop a pair.  Model-proposed relations
    remain candidate/rejected/inconclusive until an external evidence gate
    supplies a verified seed edge.
    """

    def __init__(self, client: CategoricalJSONClient, *, batch_size: int = 32) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.client = client
        self.batch_size = batch_size
        self.evaluator_name = (
            f"{getattr(client, 'model', 'gpt-oss-120b')}:categorical-l1.5"
        )

    def evaluate(
        self,
        pairs: Sequence[CorrelationPair],
    ) -> Sequence[CategoricalCorrelationJudgment]:
        judgments: list[CategoricalCorrelationJudgment] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            ids = {_pair_id(pair): pair for pair in batch}
            result = self.client.complete_json(
                task=(
                    "For every question-independent retained-node pair, propose zero or "
                    "more categorical navigation correlations. Do not infer the answer."
                ),
                payload={
                    "pairs": [
                        {
                            "pair_id": pair_id,
                            "src": _node_payload(pair.src),
                            "dst": _node_payload(pair.dst),
                            "native_hints": list(pair.native_hints),
                        }
                        for pair_id, pair in ids.items()
                    ],
                    "allowed_relations": [value.value for value in CorrelationType],
                    "allowed_status": [
                        CorrelationStatus.CANDIDATE.value,
                        CorrelationStatus.REJECTED.value,
                        CorrelationStatus.INCONCLUSIVE.value,
                    ],
                    "required_contract": {
                        "one_row_per_pair": True,
                        "relations_may_be_empty": True,
                        "no_numeric_confidence_probability_score_reward_or_utility": True,
                    },
                },
            )
            _reject_numeric_output(result)
            rows = result.get("pair_judgments")
            if not isinstance(rows, list):
                raise ValueError("correlation evaluator requires pair_judgments")
            by_pair = {
                str(row.get("pair_id")): row for row in rows if isinstance(row, dict)
            }
            if set(by_pair) != set(ids):
                raise ValueError(
                    "correlation evaluator did not cover every retained pair"
                )
            for pair_id, pair in ids.items():
                relations = by_pair[pair_id].get("relations")
                if not isinstance(relations, list):
                    raise ValueError("pair relations must be a list")
                for relation in relations:
                    if not isinstance(relation, dict):
                        raise ValueError("relation judgment must be an object")
                    status = CorrelationStatus(
                        str(relation.get("status") or "inconclusive")
                    )
                    if status is CorrelationStatus.VERIFIED:
                        raise ValueError(
                            "model proposals cannot self-admit a verified edge"
                        )
                    judgments.append(
                        CategoricalCorrelationJudgment(
                            src=pair.src.node_id,
                            dst=pair.dst.node_id,
                            relation=CorrelationType(
                                str(relation.get("relation") or "")
                            ),
                            status=status,
                            evidence_refs=(pair.src.node_id, pair.dst.node_id),
                            candidate_sources=(self.evaluator_name,),
                            alignment=_categorical_object(relation.get("alignment")),
                            verifier_result={},
                            rationale=str(
                                relation.get("rationale") or "categorical proposal"
                            ),
                        )
                    )
        return tuple(judgments)


def _pair_id(pair: CorrelationPair) -> str:
    digest = hashlib.sha256(
        f"{pair.src.node_id}\x1f{pair.dst.node_id}".encode("utf-8")
    ).hexdigest()[:20]
    return f"pair:{digest}"


def _node_payload(node: object) -> dict[str, Any]:
    return {
        "node_id": getattr(node, "node_id"),
        "node_type": getattr(node, "node_type"),
        "text": getattr(node, "text"),
        "time_span": {
            "start_s": getattr(node, "time_span").start_s,
            "end_s": getattr(node, "time_span").end_s,
        },
        "participants": getattr(node, "metadata").get("participants") or [],
        "states": getattr(node, "metadata").get("states") or [],
        "provenance_refs": list(getattr(node, "source_segments")),
    }


def _categorical_object(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("alignment must be a categorical object")
    _reject_numeric_output(value)
    return dict(value)
