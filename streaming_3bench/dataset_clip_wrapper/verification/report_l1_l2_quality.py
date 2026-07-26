#!/usr/bin/env python3
"""Summarize L1/L2 graph quality for mixed short, streaming, and long batches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .evaluate_l1_query_memory import evaluate_example
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_l1_query_memory import evaluate_example


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _support_refs(rollout: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    pack = rollout.get("verified_evidence_pack") or {}
    refs.extend(str(ref) for ref in pack.get("support_refs") or [] if ref)
    for chain in rollout.get("answer_support_chain") or []:
        if isinstance(chain, dict):
            refs.extend(str(ref) for ref in chain.get("evidence_refs") or [] if ref)
    for claim in rollout.get("claims") or []:
        if isinstance(claim, dict):
            refs.extend(str(ref) for ref in claim.get("supported_by_refs") or [] if ref)
    return list(dict.fromkeys(refs))


def _clip_schema_stats(example: dict[str, Any]) -> dict[str, int]:
    metadata = example.get("metadata") or {}
    fine = metadata.get("clip_schemas") or []
    coarse = metadata.get("coarse_clip_schemas") or []
    return {
        "fine_total": len(fine),
        "fine_errors": sum(1 for row in fine if isinstance(row, dict) and row.get("model_error")),
        "coarse_total": len(coarse),
        "coarse_errors": sum(1 for row in coarse if isinstance(row, dict) and row.get("model_error")),
        "fine_qwen": sum(1 for row in fine if isinstance(row, dict) and row.get("producer") == "qwen_clip_schema"),
        "fine_fallback": sum(1 for row in fine if isinstance(row, dict) and row.get("producer") == "video_tool_perception_backend"),
        "coarse_qwen": sum(1 for row in coarse if isinstance(row, dict) and row.get("producer") == "qwen_clip_schema"),
        "coarse_fallback": sum(1 for row in coarse if isinstance(row, dict) and row.get("producer") == "video_tool_perception_backend"),
    }


def _llm_usage_summary(rows: list[dict[str, Any]]) -> dict[str, int]:
    usages = [row.get("llm_usage") or {} for row in rows if isinstance(row, dict)]
    return {
        "calls": len(usages),
        "prompt_chars": sum(int(usage.get("prompt_chars") or 0) for usage in usages),
        "prompt_approx_tokens": sum(int(usage.get("prompt_approx_tokens") or 0) for usage in usages),
        "output_chars": sum(int(usage.get("output_chars") or 0) for usage in usages),
        "malformed_json_count": sum(int(usage.get("malformed_json") or 0) for usage in usages),
        "timeout_count": sum(int(usage.get("timeout_count") or 0) for usage in usages),
        "compact_retry_count": sum(int(usage.get("compact_retry_count") or 0) for usage in usages),
        "cache_hits": sum(1 for usage in usages if usage.get("cache_hit")),
        "cache_misses": sum(1 for usage in usages if usage and not usage.get("cache_hit")),
    }


def _budget_report(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.get("metadata") or {}
    fine = [row for row in metadata.get("clip_schemas") or [] if isinstance(row, dict)]
    coarse = [row for row in metadata.get("coarse_clip_schemas") or [] if isinstance(row, dict)]
    graph_plan = (metadata.get("graph_compose") or {}).get("skill_plan") or {}
    graph_budget = graph_plan.get("llm_budget_summary") if isinstance(graph_plan, dict) else {}
    return {
        "clip_schema": _llm_usage_summary(fine + coarse),
        "graph_compose": graph_budget or {},
    }


def _repair_needed(
    *,
    l1_quality: dict[str, Any],
    qa_answerability: dict[str, Any],
    l2_status: str,
    clip_stats: dict[str, int],
) -> bool:
    if clip_stats["fine_errors"] or clip_stats["coarse_errors"]:
        return True
    if l1_quality.get("grade") == "low":
        return True
    if qa_answerability.get("grade") in {"weak", "insufficient"}:
        return True
    if qa_answerability.get("missing_requirements"):
        return True
    return l2_status != "accepted_strong"


def summarize_example(example: dict[str, Any], *, source_path: str, topk: int) -> dict[str, Any]:
    metadata = example.get("metadata") or {}
    rollout = metadata.get("reasoning_rollout") or {}
    l1_report = evaluate_example(example, topk=topk)
    l1_quality = l1_report.get("l1_graph_quality") or {}
    qa_answerability = l1_report.get("qa_answerability") or {}
    clip_stats = _clip_schema_stats(example)
    support_refs = _support_refs(rollout)
    status = str(rollout.get("acceptance_status") or "missing")
    pack = rollout.get("verified_evidence_pack") or {}
    detail = (rollout.get("metadata") or {}).get("acceptance_status_detail") or {}
    commonsense_pack = (rollout.get("metadata") or {}).get("commonsense_repair_pack") or {}
    l2_trajectory = (rollout.get("metadata") or {}).get("l2_trajectory") or {}
    failure_reasons = rollout.get("failure_reasons") or []
    verifier_reason = (
        pack.get("verifier_reason")
        or detail.get("reason")
        or (failure_reasons[0] if failure_reasons else status)
    )
    repair_needed = _repair_needed(
        l1_quality=l1_quality,
        qa_answerability=qa_answerability,
        l2_status=status,
        clip_stats=clip_stats,
    )
    final_answer = rollout.get("final_answer") or {}
    gold = (example.get("question") or {}).get("answer") or {}
    final_label = final_answer.get("label") if isinstance(final_answer, dict) else None
    gold_label = gold.get("label") if isinstance(gold, dict) else None

    return {
        "dataset": example.get("dataset"),
        "example_id": example.get("example_id"),
        "source_path": source_path,
        "video_regime": metadata.get("video_regime"),
        "task_family": example.get("task_family"),
        "L1_quality": {
            "grade": l1_quality.get("grade"),
            "graph_nodes": l1_report.get("graph_nodes"),
            "graph_edges": l1_report.get("graph_edges"),
            "semantic_nodes": l1_quality.get("semantic_nodes"),
            "semantic_edges": l1_quality.get("semantic_edges"),
            "semantic_clip_coverage": l1_quality.get("semantic_clip_coverage"),
            "hidden_memory_items": l1_report.get("hidden_memory_items"),
            "clip_schema_stats": clip_stats,
            "coarse_fine_counts": l1_report.get("coarse_fine_counts") or {},
            "selected_coarse_indices": l1_report.get("selected_coarse_indices") or [],
        },
        "L2_status": {
            "acceptance_status": status,
            "final_answer": final_answer,
            "gold_eval_only": gold,
            "correct_eval_only": bool(final_label and gold_label and str(final_label) == str(gold_label)),
            "support_ref_count": len(support_refs),
            "trace_ok": (rollout.get("metadata") or {}).get("llm_trace_ok"),
            "trace_fail": (rollout.get("metadata") or {}).get("llm_trace_fail"),
        },
        "verifier_reason": verifier_reason,
        "llm_budget_report": _budget_report(example),
        "l2_trajectory": l2_trajectory,
        "l2_trajectory_complete": bool(l2_trajectory.get("rounds")),
        "repair_needed": repair_needed,
        "repair_hints": {
            "qa_answerability_grade": qa_answerability.get("grade"),
            "missing_requirements": qa_answerability.get("missing_requirements") or [],
            "retrieval_fallback_reason": qa_answerability.get("retrieval_fallback_reason"),
            "option_margin": qa_answerability.get("option_margin"),
            "top2_shared_ref_ratio": qa_answerability.get("top2_shared_ref_ratio"),
            "failure_reasons": failure_reasons,
            "commonsense_repair": {
                "present": bool(commonsense_pack),
                "trust_level": commonsense_pack.get("trust_level"),
                "missing_requirements": commonsense_pack.get("missing_requirements") or [],
                "visual_context_ref_count": len(commonsense_pack.get("visual_context_refs") or []),
                "top_hypotheses": (commonsense_pack.get("commonsense_hypotheses") or [])[:3],
                "recommended_next_action": commonsense_pack.get("recommended_next_action"),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Report L1/L2 quality for graph-building batches.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.paths:
        for example in _read_jsonl(path):
            rows.append(summarize_example(example, source_path=str(path), topk=args.topk))

    by_status: dict[str, int] = {}
    by_l1: dict[str, int] = {}
    for row in rows:
        status = str((row.get("L2_status") or {}).get("acceptance_status"))
        grade = str((row.get("L1_quality") or {}).get("grade"))
        by_status[status] = by_status.get(status, 0) + 1
        by_l1[grade] = by_l1.get(grade, 0) + 1
    summary = {
        "examples": len(rows),
        "datasets": sorted({str(row.get("dataset")) for row in rows}),
        "l1_quality_counts": by_l1,
        "l2_status_counts": by_status,
        "repair_needed": sum(1 for row in rows if row.get("repair_needed")),
        "accepted_strong": sum(1 for row in rows if (row.get("L2_status") or {}).get("acceptance_status") == "accepted_strong"),
    }
    payload = {"summary": summary, "reports": rows}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
