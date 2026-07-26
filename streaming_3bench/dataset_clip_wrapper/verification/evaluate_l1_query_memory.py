"""Evaluate whether saved L1 clue graphs can support query-time L2 memory.

This is an offline diagnostic. It may compare against hidden gold answers for
evaluation, but it never feeds gold labels back into retrieval or scoring.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "what",
    "when",
    "where",
    "who",
    "why",
    "how",
    "which",
    "does",
    "do",
    "did",
    "in",
    "on",
    "at",
    "to",
    "of",
    "and",
    "or",
    "for",
    "with",
    "from",
    "that",
    "this",
    "it",
    "its",
    "their",
    "they",
    "he",
    "she",
    "his",
    "her",
    "video",
}
_SEMANTIC_NODE_TYPES = {
    "observation",
    "entity_mention",
    "entity",
    "event",
    "state",
    "dialogue_span",
    "clue",
    "visual_social_cue",
}
_REFERENCE_NODE_TYPES = {"clip"}
_DIAGNOSTIC_NODE_TYPES = {"question_requirement", "required_modality", "answerability_gap"}
_TOKEN_SYNONYMS = {
    "back": {"return", "returns", "returned", "previously", "earlier"},
    "echoes": {"repeats", "returns", "reappears", "same"},
    "earlier": {"previously", "before", "original"},
    "location": {"place", "position"},
    "original": {"previously", "earlier", "place", "position"},
    "place": {"location", "position"},
    "position": {"place", "location", "original"},
    "previously": {"earlier", "before", "original"},
    "reappears": {"returns", "again", "same"},
    "repeated": {"same", "again", "reappears"},
    "returns": {"back", "returned", "reappears", "original"},
    "walked": {"moved", "went", "returns"},
}


@dataclass(frozen=True)
class MemoryItem:
    source: str
    node_id: str
    node_type: str
    text: str
    time_span: dict[str, Any] | None
    hidden_supervision: bool


def _tokens(text: str) -> set[str]:
    tokens = {tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) > 1 and tok.lower() not in _STOPWORDS}
    expanded = set(tokens)
    for token in tokens:
        expanded.update(_TOKEN_SYNONYMS.get(token, set()))
    return expanded


def _parse_time_anchors_s(text: str) -> list[float]:
    anchors: list[float] = []
    for minutes, seconds in re.findall(r"\b(\d{1,2}):(\d{2})\b", text):
        anchors.append(float(int(minutes) * 60 + int(seconds)))
    for value in re.findall(r"\b(?:at|around|near|after|before)\s+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|second|seconds)\b", text, re.I):
        anchors.append(float(value))
    return list(dict.fromkeys(anchors))


def _near_time_anchor(time_span: dict[str, Any] | None, anchors_s: list[float], *, tolerance_s: float = 8.0) -> bool:
    if not time_span or not anchors_s:
        return False
    start = float(time_span.get("start_s", 0.0))
    end = float(time_span.get("end_s", start))
    return any(start - tolerance_s <= anchor <= end + tolerance_s for anchor in anchors_s)


def _text_of_node(node: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("text", "description", "label", "content", "scene_description"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    for key in ("observable_facts", "events", "entities", "dialogue", "ocr_text"):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())
                elif isinstance(item, dict):
                    parts.append(_text_of_node(item))
    return " ".join(part for part in parts if part)


def _is_hidden(node: dict[str, Any]) -> bool:
    visibility = node.get("visibility") or {}
    return bool(visibility.get("hidden_supervision") or node.get("hidden_supervision"))


def _iter_memory_items(example: dict[str, Any]) -> Iterable[MemoryItem]:
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    for node in graph.get("nodes") or []:
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="clue_memory_graph",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    coarse_fine = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    for node in ((coarse_fine.get("coarse_graph") or {}).get("nodes") or []):
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="coarse_fine_graph.coarse",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    for node in ((coarse_fine.get("fine_graph") or {}).get("nodes") or []):
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="coarse_fine_graph.fine",
                node_id=str(node.get("node_id") or ""),
                node_type=str(node.get("node_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )

    evidence_index = example.get("evidence_index") or {}
    for node in evidence_index.get("nodes") or []:
        text = _text_of_node(node)
        if text:
            yield MemoryItem(
                source="evidence_index",
                node_id=str(node.get("node_id") or node.get("evidence_id") or ""),
                node_type=str(node.get("node_type") or node.get("source_type") or ""),
                text=text,
                time_span=node.get("time_span"),
                hidden_supervision=_is_hidden(node),
            )


def _score(query: str, item: MemoryItem) -> float:
    query_tokens = _tokens(query)
    item_tokens = _tokens(item.text)
    if not query_tokens or not item_tokens:
        return 0.0
    overlap = query_tokens & item_tokens
    anchors = _parse_time_anchors_s(query)
    anchor_bonus = 1.0 if _near_time_anchor(item.time_span, anchors) else 0.0
    if not overlap and not anchor_bonus:
        return 0.0
    return (len(overlap) / max(1.0, len(query_tokens) ** 0.5)) + anchor_bonus


def _dedupe_memory_items(items: list[MemoryItem]) -> list[MemoryItem]:
    unique: list[MemoryItem] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        key = (item.node_id, " ".join(item.text.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _top_items(query: str, items: list[MemoryItem], *, topk: int) -> list[tuple[float, MemoryItem]]:
    scored = [(_score(query, item), item) for item in items if not item.hidden_supervision]
    scored = [entry for entry in scored if entry[0] > 0]
    scored.sort(key=lambda entry: entry[0], reverse=True)
    return scored[:topk]


def _option_scores(question_text: str, options: list[dict[str, Any]], items: list[MemoryItem], *, topk: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    answer_evidence_items = [
        item
        for item in items
        if item.node_type not in _DIAGNOSTIC_NODE_TYPES and not item.node_id.startswith("diagnostic.answerability_gap:")
    ]
    for option in options:
        label = str(option.get("label") or "")
        text = str(option.get("text") or "")
        query = f"{question_text} {text}".strip()
        hits = _top_items(query, answer_evidence_items, topk=topk)
        rows.append(
            {
                "label": label,
                "text": text,
                "score": round(sum(score for score, _ in hits), 4),
                "top_refs": [item.node_id for _, item in hits[:3]],
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def _compact_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _gold_label(example: dict[str, Any]) -> str | None:
    answer = (example.get("question") or {}).get("answer") or {}
    label = answer.get("label")
    if label:
        return str(label)
    text = answer.get("text")
    if not text:
        return None
    for option in (example.get("question") or {}).get("options") or []:
        if str(option.get("text") or "").strip().lower() == str(text).strip().lower():
            return str(option.get("label") or "")
    return None


def _l1_graph_quality(example: dict[str, Any]) -> dict[str, Any]:
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    nodes = [node for node in graph.get("nodes") or [] if isinstance(node, dict)]
    edges = [edge for edge in graph.get("edges") or [] if isinstance(edge, dict)]
    node_by_id = {str(node.get("node_id") or ""): node for node in nodes if node.get("node_id")}
    hidden_nodes = [node for node in nodes if _is_hidden(node)]
    semantic_nodes = [
        node
        for node in nodes
        if str(node.get("node_type") or "") in _SEMANTIC_NODE_TYPES and _text_of_node(node) and not _is_hidden(node)
    ]
    clip_ref_nodes = [node for node in nodes if str(node.get("node_type") or "") in _REFERENCE_NODE_TYPES]
    schema_anchor_nodes = [
        node
        for node in semantic_nodes
        if str(node.get("producer") or "") == "neighbor_vlm_l1_schema_anchor"
        or str(node.get("source_type") or "") == "qwen_clip_schema_anchor"
    ]
    invalid_edges = [
        edge
        for edge in edges
        if str(edge.get("src") or "") not in node_by_id or str(edge.get("dst") or "") not in node_by_id
    ]
    semantic_node_ids = {str(node.get("node_id") or "") for node in semantic_nodes}
    semantic_edges = [
        edge
        for edge in edges
        if str(edge.get("src") or "") in semantic_node_ids or str(edge.get("dst") or "") in semantic_node_ids
    ]
    successful_schemas = [
        schema
        for schema in ((example.get("metadata") or {}).get("clip_schemas") or [])
        if isinstance(schema, dict) and schema.get("clip_id") and not schema.get("model_error")
    ]
    semantic_clip_ids = {str(node.get("clip_id") or "") for node in semantic_nodes if node.get("clip_id")}
    successful_clip_ids = {str(schema.get("clip_id") or "") for schema in successful_schemas}
    covered_successful = semantic_clip_ids & successful_clip_ids
    coverage_ratio = round(len(covered_successful) / len(successful_clip_ids), 4) if successful_clip_ids else None
    graph_compose = (example.get("metadata") or {}).get("graph_compose") or {}
    trace = graph_compose.get("execution_trace") or []
    failed_steps = [step for step in trace if isinstance(step, dict) and step.get("ok") is False]

    if semantic_nodes and not invalid_edges and (coverage_ratio is None or coverage_ratio >= 0.5) and len(semantic_edges) >= max(1, len(semantic_nodes) // 4):
        grade = "high"
    elif semantic_nodes and not invalid_edges:
        grade = "medium"
    else:
        grade = "low"

    return {
        "grade": grade,
        "semantic_nodes": len(semantic_nodes),
        "semantic_edges": len(semantic_edges),
        "clip_ref_nodes": len(clip_ref_nodes),
        "schema_anchor_nodes": len(schema_anchor_nodes),
        "hidden_nodes": len(hidden_nodes),
        "invalid_edges": len(invalid_edges),
        "successful_clip_schemas": len(successful_schemas),
        "semantic_clip_coverage": coverage_ratio,
        "failed_compose_steps": len(failed_steps),
        "used_deterministic_fallback": bool(graph_compose.get("used_deterministic_fallback")),
    }


def _qa_answerability(example: dict[str, Any], top: list[tuple[float, MemoryItem]], option_rows: list[dict[str, Any]]) -> dict[str, Any]:
    top_hit_count = len(top)
    positive_option_count = sum(1 for row in option_rows if float(row.get("score") or 0.0) > 0)
    best_score = float(option_rows[0].get("score") or 0.0) if option_rows else 0.0
    second_score = float(option_rows[1].get("score") or 0.0) if len(option_rows) > 1 else 0.0
    margin = round(best_score - second_score, 4)
    if len(option_rows) > 1:
        best_refs = set(option_rows[0].get("top_refs") or [])
        second_refs = set(option_rows[1].get("top_refs") or [])
        ref_union = best_refs | second_refs
        shared_ref_ratio = round(len(best_refs & second_refs) / len(ref_union), 4) if ref_union else 0.0
    else:
        shared_ref_ratio = 0.0
    retrieval = (((example.get("metadata") or {}).get("coarse_fine_graph") or {}).get("retrieval") or {})
    fallback_reason = retrieval.get("fallback_reason")
    answerability_diagnostic = (example.get("metadata") or {}).get("answerability_diagnostic") or {}
    missing_requirements = answerability_diagnostic.get("missing_requirements") or []
    if missing_requirements:
        grade = "weak" if top_hit_count or best_score > 0 else "insufficient"
    elif fallback_reason == "uniform_probe_no_lexical_match":
        grade = "weak" if top_hit_count or best_score > 0 else "insufficient"
    elif shared_ref_ratio >= 0.75 and margin < 0.75:
        grade = "weak" if top_hit_count or best_score > 0 else "insufficient"
    elif top_hit_count >= 2 and best_score > 0 and margin >= 0.15:
        grade = "answerable"
    elif top_hit_count or best_score > 0:
        grade = "weak"
    else:
        grade = "insufficient"
    return {
        "grade": grade,
        "top_question_hit_count": top_hit_count,
        "positive_option_count": positive_option_count,
        "best_option_score": round(best_score, 4),
        "second_option_score": round(second_score, 4),
        "option_margin": margin,
        "top2_shared_ref_ratio": shared_ref_ratio,
        "retrieval_fallback_reason": fallback_reason,
        "missing_requirements": missing_requirements,
        "gap_category": answerability_diagnostic.get("gap_category"),
        "l2_repair_policy": answerability_diagnostic.get("l2_repair_policy"),
        "allowed_repair_l2": answerability_diagnostic.get("allowed_repair_l2") or [],
        "out_of_scope_modalities": answerability_diagnostic.get("out_of_scope_modalities") or [],
        "audio_repair_allowed": bool(answerability_diagnostic.get("audio_repair_allowed")),
        "answerability_diagnostic": answerability_diagnostic,
    }


def evaluate_example(example: dict[str, Any], *, topk: int) -> dict[str, Any]:
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    options = question.get("options") or []
    items = _dedupe_memory_items(list(_iter_memory_items(example)))
    graph = ((example.get("metadata") or {}).get("clue_memory_graph") or {})
    coarse_fine = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    top = _top_items(question_text, items, topk=topk)
    option_rows = _option_scores(question_text, options, items, topk=topk) if options else []
    qa_answerability = _qa_answerability(example, top, option_rows)
    raw_predicted = option_rows[0]["label"] if option_rows else None
    prediction_suppressed_reason = None
    if qa_answerability.get("missing_requirements"):
        prediction_suppressed_reason = "answerability_gap_missing_requirements"
    elif qa_answerability.get("grade") in {"insufficient", "weak"}:
        prediction_suppressed_reason = f"qa_answerability_{qa_answerability.get('grade')}"
    predicted = None if prediction_suppressed_reason else raw_predicted
    gold = _gold_label(example)
    l2 = ((example.get("metadata") or {}).get("reasoning_rollout") or {})
    l2_final = l2.get("final_answer") or {}
    gold_text = ((question.get("answer") or {}).get("text") or "").strip()
    l2_text = str(l2_final.get("text") or "").strip()
    visible_option_texts = {str(option.get("text") or "").strip() for option in options}

    return {
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "question": question_text,
        "graph_nodes": len(graph.get("nodes") or []),
        "graph_edges": len(graph.get("edges") or []),
        "memory_items": len(items),
        "hidden_memory_items": sum(1 for item in items if item.hidden_supervision),
        "coarse_fine_counts": coarse_fine.get("counts") or {},
        "coarse_coverage": ((coarse_fine.get("coarse_graph") or {}).get("coverage")),
        "fine_coverage": ((coarse_fine.get("fine_graph") or {}).get("coverage")),
        "selected_coarse_indices": coarse_fine.get("selected_coarse_indices") or [],
        "top_question_hits": [
            {
                "score": round(score, 4),
                "source": item.source,
                "node_id": item.node_id,
                "node_type": item.node_type,
                "time_span": item.time_span,
                "text": _compact_text(item.text),
            }
            for score, item in top
        ],
        "option_scores": option_rows,
        "predicted_label_by_l1_memory": predicted,
        "raw_predicted_label_by_l1_memory": raw_predicted,
        "prediction_suppressed_reason": prediction_suppressed_reason,
        "gold_label_eval_only": gold,
        "correct_eval_only": bool(predicted and gold and predicted == gold),
        "l2_rollout_source": l2.get("rollout_source"),
        "l2_final_answer": l2_final,
        "l2_uses_gold_text_warning": bool(gold_text and l2_text == gold_text and l2_text not in visible_option_texts),
        "l1_graph_quality": _l1_graph_quality(example),
        "qa_answerability": qa_answerability,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for path in args.paths:
        for example in _read_jsonl(path):
            report = evaluate_example(example, topk=args.topk)
            report["source_path"] = str(path)
            reports.append(report)

    correct = sum(1 for row in reports if row["correct_eval_only"])
    summary = {
        "examples": len(reports),
        "correct_eval_only": correct,
        "accuracy_eval_only": round(correct / len(reports), 4) if reports else 0.0,
        "datasets": sorted({str(row["dataset"]) for row in reports}),
    }
    payload = {"summary": summary, "reports": reports}

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
