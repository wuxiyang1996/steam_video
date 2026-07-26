"""GPT-OSS-120B teacher labeling and held-out graph auditing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .causal_witness import witness_from_teacher_row
from .graph_builder import select_top_k_pairs
from .mechanism_candidates import MechanismCandidate
from .types import MemoryGraph, MemoryNode, RelationBelief, RelationStatus, RelationType


DEFAULT_MODEL = "openai/gpt-oss-120b"
PROBABILISTIC_TYPES = (
    RelationType.SAME_ENTITY.value,
    RelationType.STATE_TRANSITION.value,
    RelationType.TRANSITION_SUPPORT.value,
    RelationType.RESPONSE_CANDIDATE.value,
    RelationType.EXPLAINS.value,
    RelationType.ENABLES.value,
    RelationType.CONTRADICTS.value,
)


class GPTOSSGraphValidator:
    """Use GPT-OSS as an uncalibrated relation teacher and graph auditor."""

    def __init__(
        self,
        *,
        keys_py_path: str | Path,
        video_skills_root: str | Path,
        model: str = DEFAULT_MODEL,
        timeout_s: int = 240,
        audit_max_tokens: int = 8000,
    ) -> None:
        client_class, key_loader = _load_openrouter_client(Path(video_skills_root))
        api_key = key_loader(keys_py_path=str(keys_py_path))
        self.client = client_class(
            model=model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=3000,
            reasoning={"effort": "medium"},
            timeout_s=timeout_s,
        )
        self.model = model
        self.audit_max_tokens = audit_max_tokens
        self.last_label_errors: list[dict[str, str]] = []

    def label_relations(
        self,
        nodes: list[MemoryNode],
        embeddings: Sequence[Sequence[float]],
        *,
        top_k: int = 3,
        mechanism_candidates: Sequence[MechanismCandidate] | None = None,
    ) -> list[RelationBelief]:
        """Label the union of mechanism-first and embedding-recall candidates."""
        self.last_label_errors = []
        embedding_by_id = {node.node_id: embeddings[index] for index, node in enumerate(nodes)}
        candidates = select_top_k_pairs(nodes, embedding_by_id=embedding_by_id, top_k=top_k)
        node_by_id_for_candidates = {node.node_id: node for node in nodes}
        candidate_keys = {(src.node_id, dst.node_id) for src, dst, _ in candidates}
        mechanism_by_pair = {
            (candidate.src, candidate.dst): candidate
            for candidate in mechanism_candidates or []
        }
        for candidate in mechanism_candidates or []:
            pair = (candidate.src, candidate.dst)
            if pair in candidate_keys:
                continue
            src = node_by_id_for_candidates.get(candidate.src)
            dst = node_by_id_for_candidates.get(candidate.dst)
            if src is None or dst is None:
                continue
            candidates.append((src, dst, 0.0))
            candidate_keys.add(pair)
        node_payload = [
            {
                "node_id": node.node_id,
                "time_span": {
                    "start_s": node.time_span.start_s,
                    "end_s": node.time_span.end_s,
                },
                "text": node.text,
                "predicate": node.metadata.get("predicate"),
                "participants": node.metadata.get("participants") or [],
                "states": node.metadata.get("states") or [],
                "evidence_refs": node.metadata.get("evidence_refs")
                or node.source_segments,
            }
            for node in nodes
        ]
        pair_payload = [
            {
                "src": src.node_id,
                "dst": dst.node_id,
                "embedding_cosine": round(similarity, 6),
                "temporal_gap_s": max(
                    0.0,
                    round(dst.time_span.start_s - src.time_span.end_s, 6),
                ),
                "mechanism_hints": list(
                    mechanism_by_pair.get(
                        (src.node_id, dst.node_id),
                        MechanismCandidate(src.node_id, dst.node_id, ()),
                    ).mechanism_hints
                ),
                "requires_temporal_refine": bool(
                    mechanism_by_pair.get(
                        (src.node_id, dst.node_id),
                        MechanismCandidate(src.node_id, dst.node_id, ()),
                    ).requires_temporal_refine
                ),
            }
            for src, dst, similarity in candidates
        ]
        node_json = json.dumps(node_payload, indent=2)
        pair_json = json.dumps(pair_payload, indent=2)
        prompt = f"""
You are labeling a question-independent event graph for a video reasoning system.

Observed event nodes:
{node_json}

Sparse candidate pairs, directed from earlier to later event:
{pair_json}

For every candidate pair, first align grounded mentions and visible states, then
estimate these independent probabilities in [0,1]:
- same_entity: the same person/object participates across both events;
- state_transition: the later event changes or reveals a specific entity state established in the earlier event;
- transition_support: the earlier event supplies grounded entity/state context useful for retrieving or anticipating the later transition, without claiming it caused the transition;
- response_candidate: the later visible action may be a response to the earlier visible signal/action; this is a navigation hypothesis, not an intent or causal claim;
- explains: the earlier event answers why the later event occurs, beyond merely introducing a participant or preceding it;
- enables: the earlier event creates a necessary or materially facilitating condition; generic presence, awakening, or narrative setup is insufficient;
- contradicts: the event descriptions support incompatible claims or states.

Rules:
1. Temporal precedence alone is not causal evidence.
2. "explains" and "enables" are candidate causal support, not identified causal effects.
3. Use only the event descriptions above. Do not invent unseen events. This is
   a text-only teacher pass: never claim that a mechanism was visually verified.
4. Return a short warrant grounded in phrases from both event descriptions.
5. Return exactly one relation object per candidate pair.
6. Be conservative: generic temporal adjacency is below 0.5 for every semantic
   relation. Entity continuity may support transition_support but not causality.
7. mention_alignment must name local mention_id values from both supplied events.
8. evidence_quotes must be exact substrings from the corresponding event text.
9. state_delta is null unless one aligned entity has the same attribute with
   explicit before and after values.
10. mechanism must be one of state_bridge, observable_precondition,
    rule_response, contact_transfer, or none. Use none for narrative progression.
11. minimal_support_set contains supplied event node IDs and must include the
    source and destination. It may include one supplied mechanism_event_id.
12. If structured support is absent, explains/enables/state_transition must be <0.5.
13. transition_support requires a grounded shared entity, state, or trajectory role.
14. response_candidate requires a visible source signal/action and a distinct,
    temporally local destination response. Do not infer hidden intent.

Return JSON:
{{
  "relations": [
    {{
      "src": "node id",
      "dst": "node id",
      "probabilities": {{
        "same_entity": 0.0,
        "state_transition": 0.0,
        "transition_support": 0.0,
        "response_candidate": 0.0,
        "explains": 0.0,
        "enables": 0.0,
        "contradicts": 0.0
      }},
      "direction_confidence": 0.0,
      "warrant": "short grounded explanation",
      "mention_alignment": [
        {{
          "src_mention": "source local mention id",
          "dst_mention": "destination local mention id"
        }}
      ],
      "state_delta": {{
        "src_mention": "source local mention id",
        "dst_mention": "destination local mention id",
        "attribute": "same state attribute",
        "before": "visible source value",
        "after": "visible destination value"
      }},
      "evidence_quotes": {{
        "src": "exact quote from source event",
        "dst": "exact quote from destination event"
      }},
      "minimal_support_set": ["source node id", "destination node id"],
      "mechanism_event_id": null,
      "mechanism": "state_bridge|observable_precondition|rule_response|contact_transfer|none",
      "mechanism_detail": "concrete mechanism grounded in both quotes, or null",
      "precondition": "observable source condition required/facilitating destination, or null",
      "state_bridge": "explicit source state connected to destination result, or null",
      "dependency_basis": "entity_trajectory|state_continuity|observable_response|none",
      "alternative_explanations": [],
      "entity_continuity": 0.0,
      "state_delta_confidence": 0.0,
      "mechanism_visibility": 0.0,
      "effect_grounding": 0.0,
      "alternative_cause_penalty": 0.0
    }}
  ]
}}
""".strip()
        node_by_id = {str(row["node_id"]): row for row in node_payload}

        def request_batch(
            batch: list[tuple[MemoryNode, MemoryNode, float]],
        ) -> list[RelationBelief]:
            batch_pairs = [
                pair_payload[candidates.index(candidate)] for candidate in batch
            ]
            batch_node_ids = {
                str(value)
                for pair in batch_pairs
                for value in (pair["src"], pair["dst"])
            }
            batch_nodes = [node_by_id[node_id] for node_id in batch_node_ids]
            batch_prompt = prompt.replace(
                node_json,
                json.dumps(batch_nodes, indent=2),
                1,
            ).replace(
                pair_json,
                json.dumps(batch_pairs, indent=2),
                1,
            )
            last_error: Exception | None = None
            for _ in range(2):
                try:
                    response = self.client.chat_json(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Return conservative, evidence-grounded JSON "
                                    "relation judgments."
                                ),
                            },
                            {"role": "user", "content": batch_prompt},
                        ]
                    )
                    return _parse_teacher_relations(
                        response,
                        nodes=nodes,
                        expected_pairs={
                            (src.node_id, dst.node_id) for src, dst, _ in batch
                        },
                        model=self.model,
                    )
                except Exception as exc:
                    last_error = exc
            if len(batch) > 1:
                midpoint = len(batch) // 2
                return request_batch(batch[:midpoint]) + request_batch(batch[midpoint:])
            assert last_error is not None
            self.last_label_errors.append(
                {
                    "src": batch[0][0].node_id,
                    "dst": batch[0][1].node_id,
                    "error": f"{type(last_error).__name__}: {last_error}",
                }
            )
            return []

        relations: list[RelationBelief] = []
        batch_size = 4
        total_batches = (len(candidates) + batch_size - 1) // batch_size
        print(
            f"[relation_teacher] labeling {len(candidates)} candidate pairs "
            f"in {total_batches} batches (batch_size={batch_size}, "
            f"nodes={len(nodes)}, model={self.model})",
            flush=True,
        )
        for batch_index, start in enumerate(
            range(0, len(candidates), batch_size), start=1
        ):
            print(
                f"[relation_teacher] batch {batch_index}/{total_batches} "
                f"pairs={start}:{min(start + batch_size, len(candidates))}",
                flush=True,
            )
            relations.extend(request_batch(candidates[start : start + batch_size]))
        if self.last_label_errors:
            print(
                f"[relation_teacher] completed with {len(self.last_label_errors)} "
                f"pair-level errors",
                flush=True,
            )
        else:
            print(
                f"[relation_teacher] completed ok relations={len(relations)}",
                flush=True,
            )
        return relations

    def audit_graph(
        self,
        graph: MemoryGraph,
        *,
        held_out_annotation: dict[str, Any],
        held_out_questions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Audit graph output against annotations not shown during edge labeling."""
        predicted_relations: list[dict[str, Any]] = []
        for relation in graph.relations:
            if relation.status is RelationStatus.DETERMINISTIC:
                continue
            for name, probability in relation.relation_probabilities.items():
                if probability < 0.5:
                    continue
                predicted_relations.append(
                    {
                        "src": relation.src,
                        "dst": relation.dst,
                        "relation": name,
                        "probability": probability,
                        "warrant": relation.warrant,
                    }
                )
        graph_payload = {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "time_span": {
                        "start_s": node.time_span.start_s,
                        "end_s": node.time_span.end_s,
                    },
                    "text": node.text,
                }
                for node in graph.nodes
            ],
            "deterministic_relations": [
                relation.to_dict()
                for relation in graph.relations
                if relation.status is RelationStatus.DETERMINISTIC
            ],
            "predicted_relations_at_0.5": predicted_relations,
        }
        gold_payload = {
            "key_relationships": held_out_annotation.get("Key Relationships")
            or held_out_annotation.get("KeyRelationships")
            or [],
            "main_idea": held_out_annotation.get("MainIdea"),
            "supernatural_elements": held_out_annotation.get("SupernaturalElements"),
            "qa_explanations": [
                {
                    "question_type": row.get("Question Type"),
                    "question": row.get("Question"),
                    "explanation": row.get("Explanation"),
                }
                for row in held_out_questions
            ],
        }
        prompt = f"""
Audit the following temporal, predictive-dependency, and verified
candidate-causal event graph.

Graph:
{json.dumps(graph_payload, indent=2)}

Held-out Video-Holmes supervision:
{json.dumps(gold_payload, indent=2)}

Judge whether the graph makes sense. transition_support and response_candidate
are navigation dependencies, not causal claims. Candidate causal edges are
allowed to be plausible explanatory support; they must not be treated as
identified causal effects.
Audit every item in predicted_relations_at_0.5 separately. A shared participant
supports same_entity and may support transition_support, but does not by itself
support state_transition, response_candidate, explains, or enables.
"Plausible" means reasonable but not established by the supplied held-out evidence.

Return JSON:
{{
  "temporal_consistency": {{
    "passed": true,
    "issues": []
  }},
  "candidate_edge_audit": [
    {{
      "src": "node id",
      "dst": "node id",
      "relation": "relation type",
      "judgment": "supported|plausible|unsupported|contradicted",
      "reason": "brief reason"
    }}
  ],
  "missing_relations": [
    {{
      "src": "node id",
      "dst": "node id",
      "relation": "relation type",
      "reason": "brief reason"
    }}
  ],
  "summary": {{
    "supported_or_plausible_edges": 0,
    "audited_candidate_edges": 0,
    "precision_estimate": 0.0,
    "verdict": "pass|revise|fail",
    "main_issue": "brief text"
  }}
}}
""".strip()
        last_error: Exception | None = None
        result: dict[str, Any] | None = None
        original_max_tokens = getattr(self.client, "max_tokens", None)
        if hasattr(self.client, "max_tokens"):
            self.client.max_tokens = max(
                int(original_max_tokens or 0),
                self.audit_max_tokens,
            )
        try:
            for attempt in range(3):
                retry_instruction = ""
                if attempt:
                    retry_instruction = (
                        "\n\nThe previous response was not valid complete JSON. Return a "
                        "smaller complete object now: keep every requested audit row, "
                        "but limit each reason and issue to at most 12 words."
                    )
                try:
                    result = self.client.chat_json(
                        [
                            {
                                "role": "system",
                                "content": (
                                    "Act as a strict graph-structure auditor and return "
                                    "one complete JSON object only."
                                ),
                            },
                            {
                                "role": "user",
                                "content": prompt + retry_instruction,
                            },
                        ]
                    )
                    break
                except Exception as exc:
                    last_error = exc
        finally:
            if hasattr(self.client, "max_tokens"):
                self.client.max_tokens = original_max_tokens
        if result is None:
            assert last_error is not None
            raise ValueError(f"graph audit failed after 3 attempts: {last_error}") from last_error
        result["audit_model"] = self.model
        result["request_metadata"] = dict(self.client.last_response_metadata)
        result["computed_summary"] = _computed_audit_summary(
            result,
            expected_predictions=len(predicted_relations),
        )
        return result


def _load_openrouter_client(video_skills_root: Path) -> tuple[Any, Any]:
    root = str(video_skills_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from dataset_clip_wrapper.perception.openrouter_client import (
            OpenRouterClient,
            load_openrouter_api_key,
        )
    except ImportError as exc:
        raise RuntimeError(f"could not import Video_Skills OpenRouter client from {root}") from exc
    return OpenRouterClient, load_openrouter_api_key


def _parse_teacher_relations(
    payload: dict[str, Any],
    *,
    nodes: list[MemoryNode],
    expected_pairs: set[tuple[str, str]],
    model: str,
) -> list[RelationBelief]:
    node_by_id = {node.node_id: node for node in nodes}
    rows = payload.get("relations")
    if not isinstance(rows, list):
        raise ValueError("teacher response requires a relations list")

    parsed: list[RelationBelief] = []
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        src = str(row.get("src") or "")
        dst = str(row.get("dst") or "")
        pair = (src, dst)
        if pair not in expected_pairs or pair in seen:
            continue
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict):
            continue
        normalized = {
            name: max(0.0, min(1.0, float(probabilities.get(name, 0.0))))
            for name in PROBABILISTIC_TYPES
        }
        src_node = node_by_id[src]
        dst_node = node_by_id[dst]
        evidence_refs = list(
            dict.fromkeys(src_node.source_segments + dst_node.source_segments)
        )
        witness = witness_from_teacher_row(
            row,
            src=src,
            dst=dst,
            evidence_refs=evidence_refs,
        )
        provenance = {
            "producer": "gpt_oss_relation_teacher",
            "model": model,
            "input_scope": "atomic_events_only",
            "visual_verification_status": "not_requested",
            "mention_alignment": row.get("mention_alignment") or [],
            "participant_alignment": [
                {
                    "src_mention_id": alignment.get("src_mention"),
                    "dst_mention_id": alignment.get("dst_mention"),
                }
                for alignment in (row.get("mention_alignment") or [])
                if isinstance(alignment, dict)
            ],
            "state_delta": row.get("state_delta"),
            "evidence_quotes": row.get("evidence_quotes") or {},
            "minimal_support_set": row.get("minimal_support_set") or [],
            "mechanism_event_id": row.get("mechanism_event_id"),
            "mechanism": str(row.get("mechanism") or "none"),
            "mechanism_detail": row.get("mechanism_detail"),
            "precondition": row.get("precondition"),
            "state_bridge": row.get("state_bridge") or row.get("state_delta"),
            "dependency_basis": str(row.get("dependency_basis") or "none"),
            "alternative_explanations": row.get("alternative_explanations") or [],
        }
        if witness is not None:
            provenance["causal_witness"] = witness.to_dict()
        parsed.append(
            RelationBelief(
                edge_id=f"relation:{src}->{dst}:gpt-oss-teacher",
                src=src,
                dst=dst,
                relation_probabilities=normalized,
                status=RelationStatus.UNCALIBRATED_PRIOR,
                direction_confidence=max(0.0, min(1.0, float(row.get("direction_confidence", 0.5)))),
                warrant=str(row.get("warrant") or "").strip() or None,
                evidence_refs=evidence_refs,
                provenance=provenance,
            )
        )
        seen.add(pair)

    missing = expected_pairs - seen
    if missing:
        raise ValueError(f"teacher omitted candidate pairs: {sorted(missing)}")
    return parsed


def _computed_audit_summary(
    payload: dict[str, Any],
    *,
    expected_predictions: int,
) -> dict[str, Any]:
    rows = payload.get("candidate_edge_audit")
    if not isinstance(rows, list):
        rows = []
    judgments = [
        str(row.get("judgment") or "")
        for row in rows
        if isinstance(row, dict)
    ]
    supported = sum(judgment == "supported" for judgment in judgments)
    plausible = sum(judgment == "plausible" for judgment in judgments)
    unsupported = sum(judgment in {"unsupported", "contradicted"} for judgment in judgments)
    audited = len(judgments)
    return {
        "expected_predictions": expected_predictions,
        "audited_predictions": audited,
        "audit_complete": audited == expected_predictions,
        "supported": supported,
        "plausible": plausible,
        "unsupported_or_contradicted": unsupported,
        "strict_precision": supported / audited if audited else 0.0,
        "supported_or_plausible_rate": (supported + plausible) / audited if audited else 0.0,
    }
