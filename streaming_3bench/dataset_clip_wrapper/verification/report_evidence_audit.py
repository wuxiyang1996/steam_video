#!/usr/bin/env python3
"""LLM evidence audit for non-accepted L1/L2 repair cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..cluster_paths import DEFAULT_KEYS_PY
from ..perception.openrouter_client import OpenRouterClient, load_openrouter_api_key


ACCEPTED_STATUSES = {"accepted_strong", "accepted_bridge"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _text_from_node(node: dict[str, Any]) -> str:
    fields = (
        "text",
        "description",
        "summary",
        "claim_text",
        "surface_form",
        "scene_description",
        "reason_short",
    )
    parts = [str(node.get(field)) for field in fields if node.get(field)]
    payload = node.get("payload")
    if isinstance(payload, dict):
        parts.extend(str(payload.get(field)) for field in fields if payload.get(field))
    return " ".join(parts)[:600]


def _node_time(node: dict[str, Any]) -> dict[str, Any]:
    for key in ("time_span", "source_span"):
        value = node.get(key)
        if isinstance(value, dict):
            return value
    payload = node.get("payload")
    if isinstance(payload, dict):
        for key in ("time_span", "source_span"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
    return {}


def _source_example(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source_path")
    example_id = str(row.get("example_id") or "")
    if not source or not example_id:
        return {}
    for example in _read_jsonl(Path(str(source))):
        if str(example.get("example_id") or "") == example_id:
            return example
    return {}


def _repair_by_id(repair_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("example_id")): row
        for row in repair_report.get("reports") or []
        if row.get("example_id")
    }


def _node_lookup(example: dict[str, Any], repair: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    graph = (example.get("metadata") or {}).get("clue_memory_graph") or {}
    for node in graph.get("nodes") or []:
        if isinstance(node, dict) and node.get("node_id"):
            lookup[str(node["node_id"])] = node
    subgraph = repair.get("repair_subgraph") or {}
    for node in subgraph.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        if node.get("node_id"):
            lookup[str(node["node_id"])] = node
        for pack in node.get("option_evidence_packs") or []:
            if not isinstance(pack, dict):
                continue
            for ref in (pack.get("positive_refs") or []) + (pack.get("negative_refs") or []):
                lookup.setdefault(str(ref), {"node_id": str(ref), "node_type": "repair_ref"})
    patch_path = ((repair.get("artifact_paths") or {}).get("l1_patch"))
    if patch_path:
        path = Path(str(patch_path))
        if path.exists():
            patch = _read_json(path)
            for node in patch.get("nodes") or []:
                if isinstance(node, dict) and node.get("node_id"):
                    lookup[str(node["node_id"])] = node
    return lookup


def _resolve_refs(refs: list[str], lookup: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for ref in refs:
        node = lookup.get(str(ref), {})
        out.append(
            {
                "ref": str(ref),
                "node_type": node.get("node_type") or node.get("type"),
                "time_span": _node_time(node),
                "text": _text_from_node(node),
            }
        )
    return out


def _repair_clip_summaries(repair: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    path = ((repair.get("artifact_paths") or {}).get("clip_schemas"))
    if not path:
        return []
    rows = _read_jsonl(Path(str(path)))[:limit]
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        facts = []
        for item in row.get("observable_facts") or []:
            if isinstance(item, dict):
                facts.append(str(item.get("text") or item.get("description") or "")[:240])
            elif item:
                facts.append(str(item)[:240])
        events = []
        for item in row.get("events") or []:
            if isinstance(item, dict):
                events.append(str(item.get("description") or item.get("text") or "")[:240])
            elif item:
                events.append(str(item)[:240])
        out.append(
            {
                "clip_id": row.get("clip_id"),
                "time_span": row.get("time_span") or {},
                "scene_description": str(row.get("scene_description") or "")[:500],
                "observable_facts": [fact for fact in facts if fact][:6],
                "events": [event for event in events if event][:4],
                "model_error": row.get("model_error"),
            }
        )
    return out


def _audit_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "video_failure_evidence_audit",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "audit_status": {"type": "string"},
                    "primary_failure_class": {
                        "type": "string",
                        "enum": [
                            "benchmark_not_visually_answerable",
                            "repair_retrieval_missed_clip",
                            "evidence_exists_verifier_too_strict",
                            "l1_graph_lacks_discriminative_node",
                            "prompt_or_output_budget_issue",
                            "insufficient_evidence_after_repair",
                            "unknown",
                        ],
                    },
                    "visual_answerability": {
                        "type": "string",
                        "enum": ["visually_answerable", "partially_visual_with_bridge", "not_visually_answerable", "unclear"],
                    },
                    "evidence_assessment": {"type": "string"},
                    "selected_refs_assessment": {"type": "string"},
                    "missing_clue": {"type": "string"},
                    "should_rerun_retrieval": {"type": "boolean"},
                    "should_rerun_vlm_perception": {"type": "boolean"},
                    "should_adjust_verifier": {"type": "boolean"},
                    "should_mark_dataset_fit_risk": {"type": "boolean"},
                    "recommended_next_action": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "audit_status",
                    "primary_failure_class",
                    "visual_answerability",
                    "evidence_assessment",
                    "selected_refs_assessment",
                    "missing_clue",
                    "should_rerun_retrieval",
                    "should_rerun_vlm_perception",
                    "should_adjust_verifier",
                    "should_mark_dataset_fit_risk",
                    "recommended_next_action",
                    "confidence",
                ],
            },
        },
    }


def _compact_usage(usages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "calls": len(usages),
        "prompt_chars": sum(int(row.get("prompt_chars") or 0) for row in usages),
        "prompt_approx_tokens": sum(int(row.get("prompt_approx_tokens") or 0) for row in usages),
        "output_chars": sum(int(row.get("output_chars") or 0) for row in usages),
        "malformed_json_count": sum(int(row.get("malformed_json") or row.get("malformed_json_count") or 0) for row in usages),
        "timeout_count": sum(int(row.get("timeout_count") or 0) for row in usages),
        "compact_retry_count": sum(int(row.get("compact_retry_count") or 0) for row in usages),
    }


def _build_audit_input(row: dict[str, Any], repair: dict[str, Any], *, max_refs: int, max_clips: int) -> dict[str, Any]:
    example = _source_example(row)
    question = example.get("question") or {}
    lookup = _node_lookup(example, repair)
    option_packs = []
    refs_to_resolve: list[str] = []
    for pack in repair.get("option_evidence_packs") or []:
        if not isinstance(pack, dict):
            continue
        positive = [str(ref) for ref in pack.get("positive_refs") or []][:max_refs]
        negative = [str(ref) for ref in pack.get("negative_refs") or []][:max_refs]
        refs_to_resolve.extend(positive + negative)
        option_packs.append(
            {
                "option_label": pack.get("option_label"),
                "verifier_decision": pack.get("verifier_decision"),
                "confidence": pack.get("confidence"),
                "positive_refs": positive,
                "negative_refs": negative,
                "missing_requirements": pack.get("missing_requirements") or [],
                "selector_reason_short": pack.get("selector_reason_short") or "",
                "reason_short": pack.get("reason_short") or "",
            }
        )
    refs = list(dict.fromkeys(refs_to_resolve))[: max_refs * 12]
    return {
        "dataset": row.get("dataset"),
        "example_id": row.get("example_id"),
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
            "answer_format": question.get("answer_format"),
        },
        "final_status": row.get("final_acceptance_status"),
        "l1_quality": row.get("L1_quality") or {},
        "repair_summary": {
            "repair_status": repair.get("repair_status"),
            "failure_type": repair.get("failure_type"),
            "selected_coarse_indices": repair.get("selected_coarse_indices") or [],
            "selector_abstained": repair.get("selector_abstained"),
            "selection_mode": repair.get("selection_mode"),
            "verifier_backend": repair.get("verifier_backend"),
            "best_option": repair.get("best_option") or {},
            "patch_counts": repair.get("patch_counts") or {},
            "recommended_next_action": repair.get("recommended_next_action"),
        },
        "option_evidence_packs": option_packs,
        "resolved_refs": _resolve_refs(refs, lookup),
        "repair_clip_summaries": _repair_clip_summaries(repair, limit=max_clips),
    }


def _audit_with_llm(audit_input: dict[str, Any], *, api_key: str, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    prompt = {
        "task": "Audit why this video-only L1/L2 repair case still needs more evidence.",
        "rules": [
            "Use only the provided L1/L2/repair evidence. Do not use hidden labels or gold answers.",
            "Do not guess the final answer.",
            "Classify whether the failure is dataset visual-fit, retrieval missed clip, verifier too strict, L1 missing node, prompt budget issue, or genuinely insufficient evidence.",
            "Prefer concrete missing visual clues over generic statements.",
            "If evidence refs are empty or selector says no visible evidence, say so.",
            "Do not propose audio/ASR/subtitle fixes.",
        ],
        "case": audit_input,
    }
    client = OpenRouterClient(
        model=args.audit_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.audit_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.audit_timeout_s,
    )
    messages = [
        {
            "role": "system",
            "content": "You output JSON only. You are auditing video-only evidence graph failures.",
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    try:
        payload = client.chat_json(messages, response_format=_audit_schema())
        usage = dict(client.last_response_metadata or {})
        return payload, usage
    except Exception as exc:
        compact = dict(prompt)
        compact["case"] = {
            key: audit_input.get(key)
            for key in (
                "dataset",
                "example_id",
                "question",
                "final_status",
                "repair_summary",
                "option_evidence_packs",
            )
        }
        compact_client = OpenRouterClient(
            model=args.audit_model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=min(args.audit_max_tokens, 900),
            reasoning={"effort": "medium", "exclude": True},
            timeout_s=args.audit_timeout_s,
        )
        payload = compact_client.chat_json(
            [
                {"role": "system", "content": "You output compact valid JSON only."},
                {"role": "user", "content": json.dumps(compact, ensure_ascii=False)},
            ],
            response_format=_audit_schema(),
        )
        usage = dict(compact_client.last_response_metadata or {})
        usage["compact_retry_count"] = int(usage.get("compact_retry_count") or 0) + 1
        usage["malformed_json_count"] = int(usage.get("malformed_json_count") or 0) + 1
        usage["retry_after_error"] = str(exc)[:240]
        return payload, usage


def build_audit(final_report: dict[str, Any], repair_report: dict[str, Any], *, api_key: str, args: argparse.Namespace) -> dict[str, Any]:
    repair_map = _repair_by_id(repair_report)
    reports = []
    usages = []
    for row in final_report.get("reports") or []:
        status = str(row.get("final_acceptance_status") or "")
        if status in ACCEPTED_STATUSES:
            continue
        example_id = str(row.get("example_id") or "")
        repair = repair_map.get(example_id, {})
        audit_input = _build_audit_input(row, repair, max_refs=args.max_refs_per_option, max_clips=args.max_repair_clips)
        audit, usage = _audit_with_llm(audit_input, api_key=api_key, args=args)
        usages.append(usage)
        reports.append(
            {
                "dataset": row.get("dataset"),
                "example_id": example_id,
                "final_acceptance_status": status,
                "repair_status": repair.get("repair_status"),
                "llm_audit": audit,
                "audit_input": audit_input if args.include_audit_input else {},
                "llm_usage": usage,
            }
        )
    class_counts: dict[str, int] = {}
    answerability_counts: dict[str, int] = {}
    for row in reports:
        audit = row.get("llm_audit") or {}
        primary = str(audit.get("primary_failure_class") or "missing")
        answerability = str(audit.get("visual_answerability") or "missing")
        class_counts[primary] = class_counts.get(primary, 0) + 1
        answerability_counts[answerability] = answerability_counts.get(answerability, 0) + 1
    summary = {
        "audited_failures": len(reports),
        "primary_failure_class_counts": dict(sorted(class_counts.items())),
        "visual_answerability_counts": dict(sorted(answerability_counts.items())),
        "rerun_retrieval_count": sum(1 for row in reports if (row.get("llm_audit") or {}).get("should_rerun_retrieval")),
        "rerun_vlm_perception_count": sum(1 for row in reports if (row.get("llm_audit") or {}).get("should_rerun_vlm_perception")),
        "adjust_verifier_count": sum(1 for row in reports if (row.get("llm_audit") or {}).get("should_adjust_verifier")),
        "dataset_fit_risk_count": sum(1 for row in reports if (row.get("llm_audit") or {}).get("should_mark_dataset_fit_risk")),
        "llm_budget_summary": _compact_usage(usages),
    }
    return {"summary": summary, "reports": reports}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPT-OSS evidence audit over non-accepted final repair cases.")
    parser.add_argument("--final-report", type=Path, required=True)
    parser.add_argument("--repair-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--keys-py", type=Path, default=Path(DEFAULT_KEYS_PY) if DEFAULT_KEYS_PY else None)
    parser.add_argument("--audit-model", default="openai/gpt-oss-120b")
    parser.add_argument("--audit-max-tokens", type=int, default=1200)
    parser.add_argument("--audit-timeout-s", type=int, default=120)
    parser.add_argument("--max-refs-per-option", type=int, default=6)
    parser.add_argument("--max-repair-clips", type=int, default=8)
    parser.add_argument("--include-audit-input", action="store_true")
    args = parser.parse_args()

    api_key = load_openrouter_api_key(keys_py_path=str(args.keys_py) if args.keys_py else None)
    payload = build_audit(_read_json(args.final_report), _read_json(args.repair_report), api_key=api_key, args=args)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
