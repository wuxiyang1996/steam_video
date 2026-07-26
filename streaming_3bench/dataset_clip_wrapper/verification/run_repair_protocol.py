#!/usr/bin/env python3
"""Run targeted L1/L2 repair for examples flagged by the quality report.

The repair protocol is deliberately narrow:
1. read the L1/L2 quality report,
2. expand the coarse windows around the reported failure,
3. ask the VLM for focused repair clip schemas,
4. write a non-destructive graph patch, and
5. verify answer claims with the existing atomic verifier.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

try:
    from atomic_skills.common import stable_id
    from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
    from atomic_skills.skill_executor import SkillExecutor
    from atomic_skills.skill_model_client import SkillModelClient
    from ..perception.clip_policy import segment_coarse_index, segment_perception_clips
    from ..l1_clue_graph.clip_retrieval import retrieve_coarse_clips
    from ..perception.clip_schema import QwenClipSchemaProducer
    from ..l2_reasoning_graph.l2_recursive_trace import repair_artifacts_to_trajectory
    from ..perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
    from ..schemas import ClipPolicyConfig, ClipSchemaConfig, ClipSpan, VideoRegime
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from atomic_skills.common import stable_id
    from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
    from atomic_skills.skill_executor import SkillExecutor
    from atomic_skills.skill_model_client import SkillModelClient
    from dataset_clip_wrapper.perception.clip_policy import segment_coarse_index, segment_perception_clips
    from dataset_clip_wrapper.l1_clue_graph.clip_retrieval import retrieve_coarse_clips
    from dataset_clip_wrapper.perception.clip_schema import QwenClipSchemaProducer
    from dataset_clip_wrapper.l2_reasoning_graph.l2_recursive_trace import repair_artifacts_to_trajectory
    from dataset_clip_wrapper.perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
    from dataset_clip_wrapper.schemas import ClipPolicyConfig, ClipSchemaConfig, ClipSpan, VideoRegime


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _llm_usage_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def _load_source_example(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(row["source_path"]))
    example_id = row.get("example_id")
    for example in _read_jsonl(path):
        if example.get("example_id") == example_id:
            return example
    raise ValueError(f"Could not find example_id={example_id} in {path}")


def _option_text(question: dict[str, Any]) -> str:
    options = question.get("options") or []
    return "\n".join(f"{opt.get('label')}. {opt.get('text')}" for opt in options)


def _gap_types(row: dict[str, Any]) -> list[str]:
    hints = row.get("repair_hints") or {}
    gaps: list[str] = []
    gaps.extend(str(item) for item in hints.get("missing_requirements") or [])
    commonsense = hints.get("commonsense_repair") or {}
    gaps.extend(str(item) for item in commonsense.get("missing_requirements") or [])
    if not gaps and row.get("verifier_reason"):
        gaps.append(str(row.get("verifier_reason")))
    return list(dict.fromkeys(gaps))


def _load_evidence_audit_map(path: Path | None) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    payload = _read_json(path)
    rows = payload.get("reports") if isinstance(payload, dict) else payload
    out: dict[str, dict[str, Any]] = {}
    for row in rows or []:
        if isinstance(row, dict) and row.get("example_id"):
            audit = row.get("llm_audit") if isinstance(row.get("llm_audit"), dict) else row
            merged = dict(audit)
            merged.setdefault("dataset", row.get("dataset"))
            merged.setdefault("example_id", row.get("example_id"))
            out[str(row["example_id"])] = merged
    return out


def _compact_audit_hint(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(audit, dict) or not audit:
        return {}
    keys = (
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
    )
    return {key: audit.get(key) for key in keys if audit.get(key) not in (None, "", [])}


def _attach_audit_hint(spec: dict[str, Any], audit: dict[str, Any] | None) -> dict[str, Any]:
    hint = _compact_audit_hint(audit)
    if not hint:
        return spec
    spec = dict(spec)
    spec["evidence_audit_hint"] = hint
    missing = str(hint.get("missing_clue") or "").strip()
    if missing:
        spec["must_find_visual_evidence"] = list(dict.fromkeys([*(spec.get("must_find_visual_evidence") or []), missing]))
        spec["visual_attributes_to_resolve"] = list(dict.fromkeys([*(spec.get("visual_attributes_to_resolve") or []), missing]))
        queries = list(spec.get("coarse_search_queries") or [])
        queries.append({"role": "audit_missing_clue_search", "query": missing})
        spec["coarse_search_queries"] = queries
    return spec


def _strategy_for_gaps(gaps: list[str]) -> str:
    if "discriminative_visual_evidence" in gaps:
        return "visual_disambiguation_retrieval"
    if {"social_intent_or_affect", "causal_explanation"} & set(gaps):
        return "social_causal_commonsense_bridge"
    return "evidence_sufficiency_retrieval"


def _expanded_indices(selected: list[int], coarse_count: int, *, radius: int) -> list[int]:
    out: list[int] = []
    for idx in selected:
        for j in range(idx - radius, idx + radius + 1):
            if 0 <= j < coarse_count:
                out.append(j)
    return list(dict.fromkeys(out))


def _repair_query(example: dict[str, Any], row: dict[str, Any], gaps: list[str], clue_spec: dict[str, Any] | None = None) -> str:
    question = example.get("question") or {}
    commonsense = ((row.get("repair_hints") or {}).get("commonsense_repair") or {})
    hypotheses = commonsense.get("top_hypotheses") or []
    hyp_text = "\n".join(
        f"- {h.get('label')}: {h.get('text')} (commonsense_score={h.get('commonsense_score')})"
        for h in hypotheses[:4]
    )
    clue_text = json.dumps(clue_spec or {}, ensure_ascii=False, indent=2)
    return (
        "Visible QA repair context. Do not use hidden answer labels.\n"
        "STRICT VIDEO-ONLY RULES:\n"
        "- Use visual frames only. Ignore audio, ASR, subtitle, narration, and the wording of this prompt as evidence.\n"
        "- Do not copy the question or options into observable_facts.\n"
        "- If the requested target/clue is not visible in the clip, explicitly say visual evidence insufficient.\n"
        "- Negative evidence such as no visible target is useful and should be recorded as visual uncertainty.\n"
        f"Question: {question.get('question_text')}\n"
        f"Options:\n{_option_text(question)}\n"
        f"Current missing requirements: {', '.join(gaps) or 'evidence_sufficiency'}\n"
        f"Candidate commonsense hypotheses to check against visual evidence:\n{hyp_text}\n"
        f"Clue need spec:\n{clue_text}\n"
        "Inspect the clip only for the visual targets, attributes, actions, and exclusion criteria in the clue need spec. "
        "Return grounded visual observations or a clear visual-insufficient note."
    )


def _prior_negative_notes(schemas: list[dict[str, Any]], *, limit: int = 8) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    for schema in schemas:
        text = " ".join(_text_items(schema))
        if not text or not _has_negative_target_evidence(text):
            continue
        notes.append(
            {
                "clip_id": schema.get("clip_id"),
                "time_span": schema.get("time_span"),
                "negative_visual_note": text[:500],
            }
        )
        if len(notes) >= limit:
            break
    return notes


def _fallback_clue_need_spec(example: dict[str, Any], row: dict[str, Any], gaps: list[str]) -> dict[str, Any]:
    question = example.get("question") or {}
    options = question.get("options") or []
    return {
        "planner_backend": "fallback_no_api",
        "visual_target": question.get("question_text") or "",
        "must_find_visual_evidence": [question.get("question_text") or ""],
        "visual_attributes_to_resolve": [str(opt.get("text") or "") for opt in options],
        "forbidden_modalities": ["audio", "asr", "subtitle", "dialogue"],
        "positive_evidence_criteria": [
            "The target object/event/action is visible in the frames.",
            "The answer attribute is directly visible or explicitly absent.",
        ],
        "objective_background_facts": [],
        "bridge_evidence_criteria": [
            "Visible anchors identify the situation, place, objects, or actions needed for an objective background bridge.",
            "Background facts may explain the answer but must not be counted as visual evidence.",
        ],
        "answer_mode_hint": "direct_visual",
        "insufficient_evidence_rule": "If the target or attribute is not visible, mark visual evidence insufficient.",
        "coarse_search_queries": _query_variants(example, row, gaps, clue_spec=None),
    }


def _clue_need_spec_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "clue_need_spec",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "visual_target": {"type": "string"},
                    "must_find_visual_evidence": {"type": "array", "items": {"type": "string"}},
                    "visual_attributes_to_resolve": {"type": "array", "items": {"type": "string"}},
                    "temporal_or_action_cues": {"type": "array", "items": {"type": "string"}},
                    "negative_evidence_to_exclude": {"type": "array", "items": {"type": "string"}},
                    "forbidden_modalities": {"type": "array", "items": {"type": "string"}},
                    "positive_evidence_criteria": {"type": "array", "items": {"type": "string"}},
                    "objective_background_facts": {"type": "array", "items": {"type": "string"}},
                    "bridge_evidence_criteria": {"type": "array", "items": {"type": "string"}},
                    "answer_mode_hint": {"type": "string"},
                    "insufficient_evidence_rule": {"type": "string"},
                    "coarse_search_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "role": {"type": "string"},
                                "query": {"type": "string"},
                            },
                            "required": ["role", "query"],
                        },
                    },
                    "clip_inspection_instruction": {"type": "string"},
                },
                "required": [
                    "visual_target",
                    "must_find_visual_evidence",
                    "visual_attributes_to_resolve",
                    "forbidden_modalities",
                    "positive_evidence_criteria",
                    "objective_background_facts",
                    "bridge_evidence_criteria",
                    "answer_mode_hint",
                    "insufficient_evidence_rule",
                    "coarse_search_queries",
                    "clip_inspection_instruction",
                ],
            },
        },
    }


def _coarse_selector_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "coarse_window_selection",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "selected_coarse_indices": {"type": "array", "items": {"type": "integer"}},
                    "selection_mode": {"type": "string"},
                    "background_bridge_possible": {"type": "boolean"},
                    "selection_rounds": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "role": {"type": "string"},
                                "query_or_need": {"type": "string"},
                                "selected_after_exclusion": {"type": "array", "items": {"type": "integer"}},
                                "reason": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                        },
                    },
                    "missing_clue_diagnosis": {"type": "string"},
                },
                "required": ["selected_coarse_indices", "selection_rounds", "missing_clue_diagnosis"],
            },
        },
    }


def _coarse_selector_compact_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "coarse_window_selection_compact",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "selected_coarse_indices": {"type": "array", "items": {"type": "integer"}},
                    "selection_mode": {"type": "string"},
                    "background_bridge_possible": {"type": "boolean"},
                    "missing_clue_diagnosis": {"type": "string"},
                },
                "required": ["selected_coarse_indices", "selection_mode", "missing_clue_diagnosis"],
            },
        },
    }


def _valid_clue_need_spec(spec: dict[str, Any]) -> bool:
    return bool(
        isinstance(spec, dict)
        and spec.get("visual_target")
        and isinstance(spec.get("coarse_search_queries"), list)
        and any(isinstance(item, dict) and item.get("query") for item in spec.get("coarse_search_queries") or [])
        and spec.get("clip_inspection_instruction")
    )


def _compact_clue_need_retry(
    *,
    prompt: dict[str, Any],
    api_key: str,
    args: argparse.Namespace,
    first_error: Exception | str,
) -> dict[str, Any]:
    compact_prompt = {
        "task": "Return compact JSON for video-only clue seeking repair.",
        "rules": [
            "JSON only; no markdown or prose outside JSON.",
            "Use visual frames only; forbid audio, ASR, subtitles, narration, and dialogue.",
            "Keep every string under 120 characters.",
            "Return at most 4 coarse_search_queries.",
            "Use stable objective background facts only when visual anchors can bridge to them.",
        ],
        "question": prompt.get("question") or {},
        "gap_types": prompt.get("gap_types") or [],
        "prior_negative_visual_notes": (prompt.get("prior_negative_visual_notes") or [])[:4],
        "required_fields": [
            "visual_target",
            "must_find_visual_evidence",
            "visual_attributes_to_resolve",
            "forbidden_modalities",
            "positive_evidence_criteria",
            "objective_background_facts",
            "bridge_evidence_criteria",
            "answer_mode_hint",
            "insufficient_evidence_rule",
            "coarse_search_queries",
            "clip_inspection_instruction",
        ],
        "first_error": str(first_error)[:220],
    }
    compact_client = OpenRouterClient(
        model=args.clue_planner_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=min(args.clue_planner_max_tokens, 900),
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.clue_planner_timeout_s,
    )
    spec = compact_client.chat_json(
        [
            {
                "role": "system",
                "content": "You output compact valid JSON only. No markdown, no prose, no newlines inside strings.",
            },
            {"role": "user", "content": json.dumps(compact_prompt, ensure_ascii=False)},
        ],
        response_format=_clue_need_spec_response_schema(),
    )
    usage = dict(compact_client.last_response_metadata or {})
    usage["compact_retry_count"] = int(usage.get("compact_retry_count") or 0) + 1
    spec["llm_usage"] = usage
    spec["planner_retry"] = "compact_json"
    return spec


def _build_clue_need_spec(
    example: dict[str, Any],
    row: dict[str, Any],
    gaps: list[str],
    *,
    prior_schemas: list[dict[str, Any]],
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fallback = _fallback_clue_need_spec(example, row, gaps)
    if not api_key or args.disable_llm_clue_planner or args.dry_run:
        return fallback

    question = example.get("question") or {}
    prompt = {
        "task": "Create a video-only clue seeking specification for long-video repair.",
        "rules": [
            "Use only the visible question/options and prior negative visual notes.",
            "Do not use hidden answer labels or dataset annotations.",
            "The downstream VLM may only use frames; audio, ASR, subtitles, narration, and dialogue are forbidden evidence.",
            "Describe what visual clue must be found and what would count as insufficient evidence.",
            "If direct visual proof is unlikely but objective background knowledge can bridge from visible anchors, mark answer_mode_hint=visual_context_plus_background.",
            "List only stable objective background facts, not subjective guesses, stereotypes, private intentions, or dataset-specific answer leakage.",
            "Keep objective background facts separate from visual evidence; they may guide L2 bridge verification but cannot become L1 evidence.",
            "Generate concrete search queries for coarse visual summaries.",
        ],
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
            "answer_format": question.get("answer_format"),
        },
        "gap_types": gaps,
        "repair_hints": row.get("repair_hints") or {},
        "prior_negative_visual_notes": _prior_negative_notes(prior_schemas),
        "required_json_shape": _clue_need_spec_response_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.clue_planner_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clue_planner_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.clue_planner_timeout_s,
    )
    try:
        spec = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You output JSON only. Plan grounded visual clue searches for video-only QA repair. "
                        "Do not include analysis, markdown, or prose outside JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format=_clue_need_spec_response_schema(),
        )
    except Exception as exc:
        try:
            spec = _compact_clue_need_retry(prompt=prompt, api_key=api_key, args=args, first_error=exc)
        except Exception as retry_exc:
            if args.allow_lexical_fallback:
                fallback["planner_error"] = f"{exc}; compact retry failed: {retry_exc}"
                return fallback
            raise RuntimeError(
                "LLM clue planner failed and heuristic fallback is disabled: "
                f"{exc}; compact retry failed: {retry_exc}"
            ) from retry_exc
    if not _valid_clue_need_spec(spec):
        try:
            spec = _compact_clue_need_retry(
                prompt=prompt,
                api_key=api_key,
                args=args,
                first_error="invalid clue_need_spec",
            )
        except Exception as retry_exc:
            if args.allow_lexical_fallback:
                fallback["planner_error"] = f"invalid clue_need_spec; compact retry failed: {retry_exc}"
                fallback["invalid_planner_payload"] = spec
                return fallback
            raise RuntimeError(
                "LLM clue planner returned invalid clue_need_spec and compact retry failed: "
                f"{json.dumps(spec, ensure_ascii=False)[:800]}; retry_error={retry_exc}"
            ) from retry_exc
    if not _valid_clue_need_spec(spec):
        if args.allow_lexical_fallback:
            fallback["planner_error"] = "invalid clue_need_spec_after_compact_retry"
            fallback["invalid_planner_payload"] = spec
            return fallback
        raise RuntimeError(f"LLM clue planner returned invalid clue_need_spec: {json.dumps(spec, ensure_ascii=False)[:800]}")
    spec["planner_backend"] = args.clue_planner_model
    spec.setdefault("llm_usage", client.last_response_metadata)
    spec.setdefault("forbidden_modalities", ["audio", "asr", "subtitle", "dialogue"])
    return spec


def _coarse_schema_segments(example: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for idx, schema in enumerate(((example.get("metadata") or {}).get("coarse_clip_schemas") or [])):
        if not isinstance(schema, dict):
            continue
        text_items = _text_items(schema)
        if not text_items:
            continue
        time_span = schema.get("time_span") or {}
        segments.append(
            {
                "segment_id": f"repair.coarse_schema:{idx:04d}",
                "source_type": "coarse_visual_summary",
                "time_span": time_span,
                "text": " ".join(text_items),
            }
        )
    return segments


def _query_variants(
    example: dict[str, Any],
    row: dict[str, Any],
    gaps: list[str],
    clue_spec: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    if clue_spec:
        planned = []
        for item in clue_spec.get("coarse_search_queries") or []:
            if isinstance(item, dict) and item.get("query"):
                planned.append({"role": str(item.get("role") or "planned_clue_retrieval"), "query": str(item["query"])})
        if planned:
            return planned
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    option_text = " ".join(str(opt.get("text") or "") for opt in question.get("options") or [])
    commonsense = ((row.get("repair_hints") or {}).get("commonsense_repair") or {})
    top_hypotheses = " ".join(str(h.get("text") or "") for h in (commonsense.get("top_hypotheses") or [])[:4])
    variants = [
        {"role": "target_retrieval", "query": question_text},
        {"role": "attribute_retrieval", "query": f"{question_text} {option_text}"},
        {"role": "temporal_context_retrieval", "query": f"before after context event setup payoff {question_text}"},
    ]
    if {"social_intent_or_affect", "causal_explanation"} & set(gaps):
        variants.append(
            {
                "role": "social_causal_bridge_retrieval",
                "query": f"{question_text} intention motive reason social affect causal context {top_hypotheses}",
            }
        )
    if "discriminative_visual_evidence" in gaps:
        variants.append(
            {
                "role": "visual_disambiguation_retrieval",
                "query": f"{question_text} discriminative visual attribute color object moving direction animation {option_text}",
            }
        )
    return [variant for variant in variants if variant["query"].strip()]


def _negative_parent_indices_from_schemas(example: dict[str, Any], schemas: list[dict[str, Any]]) -> list[int]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    coarse = segment_coarse_index(duration_s, policy, regime=VideoRegime.LONG)
    negative_indices: list[int] = []
    for schema in schemas:
        text = " ".join(_text_items(schema)).lower()
        if not text or not _has_negative_target_evidence(text):
            continue
        span = schema.get("time_span") or {}
        try:
            midpoint = (float(span.get("start_s")) + float(span.get("end_s"))) / 2.0
        except (TypeError, ValueError):
            continue
        for idx, coarse_span in enumerate(coarse):
            if coarse_span.start_s <= midpoint <= coarse_span.end_s:
                negative_indices.append(idx)
                break
    return list(dict.fromkeys(negative_indices))


def _coarse_summaries_for_prompt(example: dict[str, Any], *, max_chars: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, schema in enumerate(((example.get("metadata") or {}).get("coarse_clip_schemas") or [])):
        if not isinstance(schema, dict):
            continue
        text = " ".join(_text_items(schema))
        rows.append(
            {
                "coarse_index": idx,
                "time_span": schema.get("time_span") or {},
                "visual_summary": text[:max_chars],
            }
        )
    return rows


def _llm_select_coarse_indices(
    example: dict[str, Any],
    *,
    clue_spec: dict[str, Any],
    negative_indices: set[int],
    api_key: str | None,
    args: argparse.Namespace,
) -> tuple[list[int], list[dict[str, Any]]]:
    if not api_key or args.disable_llm_reroute_selector or args.dry_run:
        return [], []
    question = example.get("question") or {}
    coarse_rows = _coarse_summaries_for_prompt(example, max_chars=args.coarse_summary_prompt_chars)
    if not coarse_rows:
        return [], []
    prompt = {
        "task": "Select coarse video windows to inspect for the needed visual clue.",
        "rules": [
            "Use only the coarse visual summaries as search evidence.",
            "Do not infer from audio, subtitles, hidden labels, or the gold answer.",
            "Do not select excluded coarse indices unless every other window is worse.",
            "Prefer windows likely to contain positive visible evidence, not windows that merely mention the target is absent.",
            "If direct proof is absent but a summary contains visible anchors for an objective background bridge, select it with selection_mode=bridge_context.",
            "Coarse summaries are lossy. If the target is a short event or small object that may be omitted, select exploratory_probe windows likely to contain the surrounding scene/action instead of abstaining.",
            "Use selection_mode=exploratory_probe when the question is visually answerable but the coarse summaries do not explicitly name the needed clue.",
            "Only abstain when the question appears outside video-only scope or the summaries give no plausible direct, bridge, or exploratory visual window.",
        ],
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
        },
        "clue_need_spec": clue_spec,
        "excluded_negative_coarse_indices": sorted(negative_indices),
        "max_indices": args.reroute_topk,
        "coarse_summaries": coarse_rows,
        "required_json_shape": _coarse_selector_response_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.clue_planner_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clue_selector_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.clue_planner_timeout_s,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You output JSON only. Select long-video coarse windows for visual clue discovery. "
                "Do not include analysis, markdown, or prose outside JSON."
            ),
        },
        {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
    ]
    try:
        payload = client.chat_json(
            messages,
            response_format=_coarse_selector_response_schema(),
        )
        payload["llm_usage"] = client.last_response_metadata
    except Exception as exc:
        compact_prompt = dict(prompt)
        compact_prompt["rules"] = list(prompt["rules"]) + [
            "Retry in compact JSON. Do not include selection_rounds.",
            "missing_clue_diagnosis must be one short sentence with no newline.",
        ]
        compact_prompt["required_json_shape"] = _coarse_selector_compact_response_schema()["json_schema"]["schema"]["properties"]
        compact_client = OpenRouterClient(
            model=args.clue_planner_model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=min(args.clue_selector_max_tokens, 900),
            reasoning={"effort": "medium", "exclude": True},
            timeout_s=args.clue_planner_timeout_s,
        )
        try:
            payload = compact_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You output compact valid JSON only. No markdown, no prose, no newlines inside strings."
                        ),
                    },
                    {"role": "user", "content": json.dumps(compact_prompt, ensure_ascii=False)},
                ],
                response_format=_coarse_selector_compact_response_schema(),
            )
            usage = dict(compact_client.last_response_metadata or {})
            usage["compact_retry_count"] = int(usage.get("compact_retry_count") or 0) + 1
            payload["llm_usage"] = usage
            payload["selection_rounds"] = [
                {
                    "role": "model_clue_selector_compact_retry",
                    "query_or_need": clue_spec.get("visual_target"),
                    "selected_after_exclusion": payload.get("selected_coarse_indices") or [],
                    "reason": payload.get("missing_clue_diagnosis") or f"compact retry after selector JSON error: {exc}",
                    "confidence": 0.0,
                    "selection_mode": payload.get("selection_mode") or "direct_visual",
                }
            ]
        except Exception as retry_exc:
            reason = f"{exc}; compact retry failed: {retry_exc}"
            if args.allow_lexical_fallback:
                return [], [
                    {
                        "role": "model_clue_selector_error",
                        "reason": reason,
                        "selected_after_exclusion": [],
                    }
                ]
            return [], [
                {
                    "role": "model_clue_selector_error",
                    "query_or_need": clue_spec.get("visual_target"),
                    "selected_after_exclusion": [],
                    "reason": reason,
                    "confidence": 0.0,
                    "selection_mode": "abstain",
                    "selector_backend": args.clue_planner_model,
                    "selector_status": "error",
                    "selector_abstained": True,
                    "llm_usage": {
                        "malformed_json_count": 1,
                        "compact_retry_count": 1,
                    },
                }
            ]
    selected: list[int] = []
    max_index = len(coarse_rows) - 1
    for item in payload.get("selected_coarse_indices") or []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx <= max_index and idx not in negative_indices and idx not in selected:
            selected.append(idx)
        if len(selected) >= args.reroute_topk:
            break
    rounds = payload.get("selection_rounds") if isinstance(payload.get("selection_rounds"), list) else []
    if rounds:
        for row in rounds:
            if isinstance(row, dict):
                row["selector_backend"] = args.clue_planner_model
                row.setdefault("selection_mode", payload.get("selection_mode") or "direct_visual")
                row.setdefault("llm_usage", payload.get("llm_usage") or {})
    else:
        rounds = [
            {
                "role": "model_clue_selector",
                "query_or_need": clue_spec.get("visual_target"),
                "selected_after_exclusion": selected,
                "reason": payload.get("missing_clue_diagnosis") or "",
                "confidence": 0.0,
                "selection_mode": payload.get("selection_mode") or ("direct_visual" if selected else "abstain"),
                "background_bridge_possible": bool(payload.get("background_bridge_possible")),
                "selector_backend": args.clue_planner_model,
                "selector_abstained": not bool(selected),
                "llm_usage": payload.get("llm_usage") or {},
            }
        ]
    if not selected and not args.disable_exploratory_selector_retry:
        forced_prompt = {
            "task": "Select exploratory coarse windows from full-video coverage after an abstaining selector.",
            "rules": [
                "The previous selector found no explicit target mention, but coarse summaries are lossy.",
                "Use only the coarse visual summaries, question, and clue_need_spec.",
                "Do not use hidden labels, audio, subtitles, or the gold answer.",
                "You must select 1 to max_indices coarse windows unless every summary is empty.",
                "Choose windows where a small, short, or omitted visual clue could plausibly occur based on surrounding scene, temporal setup, objects, or action.",
                "Use selection_mode=exploratory_probe.",
                "missing_clue_diagnosis must be one short sentence with no newline.",
            ],
            "question": prompt["question"],
            "clue_need_spec": clue_spec,
            "excluded_negative_coarse_indices": sorted(negative_indices),
            "max_indices": args.reroute_topk,
            "coarse_summaries": coarse_rows,
            "required_json_shape": _coarse_selector_compact_response_schema()["json_schema"]["schema"]["properties"],
        }
        forced_client = OpenRouterClient(
            model=args.clue_planner_model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=args.clue_selector_max_tokens,
            reasoning={"effort": "medium", "exclude": True},
            timeout_s=args.clue_planner_timeout_s,
        )
        try:
            forced = forced_client.chat_json(
                [
                    {
                        "role": "system",
                        "content": "You output compact valid JSON only. No markdown, no prose, no newlines inside strings.",
                    },
                    {"role": "user", "content": json.dumps(forced_prompt, ensure_ascii=False)},
                ],
                response_format=_coarse_selector_compact_response_schema(),
            )
            forced_usage = dict(forced_client.last_response_metadata or {})
            forced_usage["compact_retry_count"] = int(forced_usage.get("compact_retry_count") or 0) + 1
            for item in forced.get("selected_coarse_indices") or []:
                try:
                    idx = int(item)
                except (TypeError, ValueError):
                    continue
                if 0 <= idx <= max_index and idx not in negative_indices and idx not in selected:
                    selected.append(idx)
                if len(selected) >= args.reroute_topk:
                    break
            rounds.append(
                {
                    "role": "model_clue_selector_forced_exploratory_retry",
                    "query_or_need": clue_spec.get("visual_target"),
                    "selected_after_exclusion": selected,
                    "reason": forced.get("missing_clue_diagnosis") or "",
                    "confidence": 0.0,
                    "selection_mode": forced.get("selection_mode") or "exploratory_probe",
                    "background_bridge_possible": bool(forced.get("background_bridge_possible")),
                    "selector_backend": args.clue_planner_model,
                    "selector_abstained": not bool(selected),
                    "llm_usage": forced_usage,
                }
            )
        except Exception as exc:
            rounds.append(
                {
                    "role": "model_clue_selector_forced_exploratory_error",
                    "query_or_need": clue_spec.get("visual_target"),
                    "selected_after_exclusion": [],
                    "reason": str(exc),
                    "confidence": 0.0,
                    "selection_mode": "abstain",
                    "selector_backend": args.clue_planner_model,
                    "selector_abstained": True,
                }
            )
    return selected, rounds


def _select_rerouted_repair_spans(
    example: dict[str, Any],
    row: dict[str, Any],
    *,
    gaps: list[str],
    clue_spec: dict[str, Any],
    prior_schemas: list[dict[str, Any]],
    max_repair_clips: int,
    reroute_topk: int,
    reroute_topk_per_query: int,
    api_key: str | None,
    args: argparse.Namespace,
) -> tuple[list[ClipSpan], dict[str, Any]]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    coarse = segment_coarse_index(duration_s, policy, regime=VideoRegime.LONG)
    segments = _coarse_schema_segments(example)
    negative_indices = set(_negative_parent_indices_from_schemas(example, prior_schemas))
    rounds: list[dict[str, Any]] = []
    combined_scores: dict[int, float] = {}

    selected, llm_rounds = _llm_select_coarse_indices(
        example,
        clue_spec=clue_spec,
        negative_indices=negative_indices,
        api_key=api_key,
        args=args,
    )
    rounds.extend(llm_rounds)

    selector_abstained = bool(
        api_key
        and not selected
        and not args.allow_lexical_fallback
        and not args.disable_llm_reroute_selector
        and not args.dry_run
    )

    for variant in ([] if selected or selector_abstained else _query_variants(example, row, gaps, clue_spec=clue_spec)):
        retrieval = retrieve_coarse_clips(
            coarse_spans=coarse,
            query_text=variant["query"],
            segments=segments,
            topk=max(reroute_topk_per_query + len(negative_indices), reroute_topk_per_query),
            threshold=0.0,
            mode="lexical",
        )
        kept_scores: list[dict[str, Any]] = []
        for score_row in retrieval.get("scores") or []:
            idx = int(score_row["coarse_index"])
            if idx in negative_indices:
                continue
            score = float(score_row.get("score") or 0.0)
            kept_scores.append(score_row)
            combined_scores[idx] = combined_scores.get(idx, 0.0) + score + 0.05
            if len(kept_scores) >= reroute_topk_per_query:
                break
        rounds.append(
            {
                "role": variant["role"],
                "query": variant["query"],
                "fallback_reason": retrieval.get("fallback_reason"),
                "selected_before_exclusion": retrieval.get("selected_coarse_indices") or [],
                "selected_after_exclusion": [int(row["coarse_index"]) for row in kept_scores],
                "scores": kept_scores,
            }
        )

    if not selected:
        ranked = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)
        selected = [idx for idx, _ in ranked[:reroute_topk]]
    if not selected:
        if selector_abstained:
            return [], {
                "mode": "reroute",
                "duration_s": duration_s,
                "negative_coarse_indices": sorted(negative_indices),
                "selected_coarse_indices": [],
                "candidate_span_count": 0,
                "candidate_fine_span_count": 0,
                "fresh_span_count": 0,
                "chosen_span_count": 0,
                "retrieval_rounds": rounds,
                "clue_planner_backend": clue_spec.get("planner_backend"),
                "reroute_selector_backend": args.clue_planner_model,
                "selector_abstained": True,
                "selection_mode": "abstain",
            }
        selected = [idx for idx in range(min(reroute_topk, len(coarse))) if idx not in negative_indices]

    spans = segment_perception_clips(
        duration_s,
        policy,
        regime=VideoRegime.LONG,
        selected_coarse_indices=selected,
    )
    fine_spans = [span for span in spans if span.granularity == "fine"]
    existing_ids = {
        str(schema.get("clip_id"))
        for schema in ((example.get("metadata") or {}).get("clip_schemas") or [])
        if isinstance(schema, dict) and schema.get("clip_id")
    }
    video_id = (example.get("video") or {}).get("video_id")
    fresh = [
        span
        for span in fine_spans
        if f"clip:{video_id}:fine:{span.clip_index:04d}" not in existing_ids
    ]
    chosen = (fresh or fine_spans or spans)[:max_repair_clips]
    successful_rounds = [
        round_row
        for round_row in rounds
        if isinstance(round_row, dict)
        and round_row.get("selected_after_exclusion")
        and not round_row.get("selector_abstained")
    ]
    selection_mode = (
        successful_rounds[-1].get("selection_mode")
        if successful_rounds
        else (rounds[0].get("selection_mode") if rounds and isinstance(rounds[0], dict) else None)
    )
    return chosen, {
        "mode": "reroute",
        "duration_s": duration_s,
        "negative_coarse_indices": sorted(negative_indices),
        "selected_coarse_indices": selected,
        "candidate_span_count": len(spans),
        "candidate_fine_span_count": len(fine_spans),
        "fresh_span_count": len(fresh),
        "chosen_span_count": len(chosen),
        "retrieval_rounds": rounds,
        "clue_planner_backend": clue_spec.get("planner_backend"),
        "reroute_selector_backend": args.clue_planner_model if rounds and rounds[0].get("selector_backend") else "lexical_fallback",
        "selector_abstained": False,
        "selection_mode": selection_mode or "direct_visual",
    }


def _select_repair_spans(
    example: dict[str, Any],
    row: dict[str, Any],
    *,
    radius: int,
    max_repair_clips: int,
) -> tuple[list[ClipSpan], dict[str, Any]]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    policy = ClipPolicyConfig.for_regime(VideoRegime.LONG, duration_s=duration_s)
    cf_graph = ((example.get("metadata") or {}).get("coarse_fine_graph") or {})
    coarse_count = int(((cf_graph.get("counts") or {}).get("coarse_nodes")) or 0)
    selected = [int(i) for i in ((row.get("L1_quality") or {}).get("selected_coarse_indices") or [])]
    if not selected:
        selected = list(range(min(coarse_count, 3)))
    expanded = _expanded_indices(selected, coarse_count or max(selected, default=0) + 1, radius=radius)
    spans = segment_perception_clips(
        duration_s,
        policy,
        regime=VideoRegime.LONG,
        selected_coarse_indices=expanded,
    )
    fine_spans = [span for span in spans if span.granularity == "fine"]

    existing_ids = {
        str(schema.get("clip_id"))
        for schema in ((example.get("metadata") or {}).get("clip_schemas") or [])
        if isinstance(schema, dict) and schema.get("clip_id")
    }
    fresh = [
        span
        for span in fine_spans
        if f"clip:{(example.get('video') or {}).get('video_id')}:fine:{span.clip_index:04d}" not in existing_ids
    ]
    chosen = (fresh or fine_spans or spans)[:max_repair_clips]
    return chosen, {
        "mode": "local",
        "duration_s": duration_s,
        "selected_coarse_indices": selected,
        "expanded_coarse_indices": expanded,
        "candidate_span_count": len(spans),
        "candidate_fine_span_count": len(fine_spans),
        "fresh_span_count": len(fresh),
        "chosen_span_count": len(chosen),
    }


def _select_existing_l1_repair_spans(example: dict[str, Any]) -> tuple[list[ClipSpan], dict[str, Any]]:
    duration_s = float((example.get("video") or {}).get("duration_s") or 0.0)
    return [], {
        "mode": "existing_l1_option_verification",
        "duration_s": duration_s,
        "selected_coarse_indices": [],
        "candidate_span_count": 0,
        "candidate_fine_span_count": 0,
        "fresh_span_count": 0,
        "chosen_span_count": 0,
        "selection_mode": "existing_l1_option_verification",
        "retrieval_rounds": [
            {
                "role": "existing_l1_evidence_pack",
                "selection_mode": "no_new_perception",
                "selected_after_exclusion": [],
                "reason": "Short/streaming repair verifies option-specific evidence packs against the existing L1 graph before requesting more perception.",
            }
        ],
    }


def _build_producer(args: argparse.Namespace, api_key: str) -> QwenClipSchemaProducer:
    reasoning = None if args.clip_schema_reasoning_effort in {"", "none", "off"} else {"effort": args.clip_schema_reasoning_effort}
    if args.clip_schema_reasoning_effort == "none":
        reasoning = {"effort": "none", "exclude": True}
    client = OpenRouterClient(
        model=args.clip_schema_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.clip_schema_max_tokens,
        reasoning=reasoning,
        timeout_s=args.clip_schema_timeout_s,
    )
    cfg = ClipSchemaConfig(
        model=args.clip_schema_model,
        keys_py_path=str(args.keys_py) if args.keys_py else None,
        request_frames=args.request_frames,
        max_tokens=args.clip_schema_max_tokens,
        timeout_s=args.clip_schema_timeout_s,
    )
    return QwenClipSchemaProducer(cfg, client)


def _schema_for_span(
    producer: QwenClipSchemaProducer,
    *,
    example: dict[str, Any],
    span: ClipSpan,
    repair_query: str,
) -> dict[str, Any]:
    video = example.get("video") or {}
    video_id = str(video.get("video_id") or "video")
    clip_id = f"clip:{video_id}:fine:{span.clip_index:04d}"
    return producer.build_clip_schema(
        clip_id=clip_id,
        clip=span,
        video_path=Path(str(video.get("primary_path"))) if video.get("primary_path") else None,
        subtitle_context=None,
        question_context=repair_query,
    )


def _clip_id_for_span(example: dict[str, Any], span: ClipSpan) -> str:
    video = example.get("video") or {}
    video_id = str(video.get("video_id") or "video")
    return f"clip:{video_id}:fine:{span.clip_index:04d}"


def _schema_worker_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "clip_schema_model": args.clip_schema_model,
        "keys_py_path": str(args.keys_py) if args.keys_py else None,
        "request_frames": args.request_frames,
        "clip_schema_max_tokens": args.clip_schema_max_tokens,
        "clip_schema_reasoning_effort": args.clip_schema_reasoning_effort,
        "clip_schema_timeout_s": args.clip_schema_timeout_s,
    }


def _repair_clip_schema_worker(job: dict[str, Any]) -> dict[str, Any]:
    args = job["args"]
    span_payload = job["span"]
    span = ClipSpan(
        start_s=float(span_payload["start_s"]),
        end_s=float(span_payload["end_s"]),
        granularity=span_payload.get("granularity", "fine"),
        parent_index=span_payload.get("parent_index"),
        clip_index=int(span_payload.get("clip_index") or 0),
    )
    reasoning = None if args["clip_schema_reasoning_effort"] in {"", "none", "off"} else {"effort": args["clip_schema_reasoning_effort"]}
    if args["clip_schema_reasoning_effort"] == "none":
        reasoning = {"effort": "none", "exclude": True}
    client = OpenRouterClient(
        model=args["clip_schema_model"],
        api_key=job["api_key"],
        temperature=0.0,
        max_tokens=int(args["clip_schema_max_tokens"]),
        reasoning=reasoning,
        timeout_s=int(args["clip_schema_timeout_s"]),
    )
    cfg = ClipSchemaConfig(
        model=args["clip_schema_model"],
        keys_py_path=args.get("keys_py_path"),
        request_frames=int(args["request_frames"]),
        max_tokens=int(args["clip_schema_max_tokens"]),
        timeout_s=int(args["clip_schema_timeout_s"]),
    )
    producer = QwenClipSchemaProducer(cfg, client)
    try:
        return _schema_for_span(producer, example=job["example"], span=span, repair_query=job["repair_query"])
    except Exception as exc:
        video = job["example"].get("video") or {}
        video_id = str(video.get("video_id") or "video")
        return {
            "clip_id": f"clip:{video_id}:fine:{span.clip_index:04d}",
            "time_span": span.to_dict(),
            "granularity": span.granularity,
            "producer": "qwen_repair_clip_schema",
            "model": args["clip_schema_model"],
            "model_error": str(exc),
            "schema_attempt": "repair_parallel_worker_error",
        }


def _produce_repair_schemas(
    *,
    example: dict[str, Any],
    spans: list[ClipSpan],
    repair_query: str,
    api_key: str,
    args: argparse.Namespace,
    schemas_path: Path,
) -> list[dict[str, Any]]:
    if not spans:
        return []
    prior = _read_jsonl(schemas_path)
    by_clip_id = {
        str(row.get("clip_id")): row
        for row in prior
        if isinstance(row, dict) and row.get("clip_id") and not row.get("model_error")
    }
    pending_spans = [span for span in spans if _clip_id_for_span(example, span) not in by_clip_id]
    if not pending_spans:
        return [by_clip_id[_clip_id_for_span(example, span)] for span in spans if _clip_id_for_span(example, span) in by_clip_id]
    if args.repair_clip_schema_workers <= 1 or len(spans) <= 1:
        producer = _build_producer(args, api_key)
        for span in pending_spans:
            schema = _schema_for_span(producer, example=example, span=span, repair_query=repair_query)
            by_clip_id[str(schema.get("clip_id"))] = schema
            _write_jsonl(schemas_path, [by_clip_id[_clip_id_for_span(example, item)] for item in spans if _clip_id_for_span(example, item) in by_clip_id])
        return [by_clip_id[_clip_id_for_span(example, span)] for span in spans if _clip_id_for_span(example, span) in by_clip_id]

    worker_args = _schema_worker_args(args)
    jobs = [
        {
            "example": example,
            "span": {
                "start_s": span.start_s,
                "end_s": span.end_s,
                "granularity": span.granularity,
                "parent_index": span.parent_index,
                "clip_index": span.clip_index,
            },
            "repair_query": repair_query,
            "api_key": api_key,
            "args": worker_args,
        }
        for span in pending_spans
    ]
    indexed: list[tuple[int, dict[str, Any]]] = []
    with ProcessPoolExecutor(max_workers=min(args.repair_clip_schema_workers, len(jobs))) as executor:
        future_to_index = {executor.submit(_repair_clip_schema_worker, job): idx for idx, job in enumerate(jobs)}
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                schema = future.result()
            except Exception as exc:
                span = pending_spans[idx]
                video = example.get("video") or {}
                video_id = str(video.get("video_id") or "video")
                schema = {
                    "clip_id": f"clip:{video_id}:fine:{span.clip_index:04d}",
                    "time_span": span.to_dict(),
                    "granularity": span.granularity,
                    "producer": "qwen_repair_clip_schema",
                    "model": args.clip_schema_model,
                    "model_error": str(exc),
                    "schema_attempt": "repair_parallel_future_error",
                }
            indexed.append((idx, schema))
            if not schema.get("model_error"):
                by_clip_id[str(schema.get("clip_id"))] = schema
            else:
                by_clip_id.setdefault(str(schema.get("clip_id")), schema)
            _write_jsonl(schemas_path, [by_clip_id[_clip_id_for_span(example, item)] for item in spans if _clip_id_for_span(example, item) in by_clip_id])
    return [by_clip_id[_clip_id_for_span(example, span)] for span in spans if _clip_id_for_span(example, span) in by_clip_id]


def _audio_or_subtitle_like(text: str) -> bool:
    lowered = text.lower()
    return any(
        token in lowered
        for token in ("'modality': 'audio'", '"modality": "audio"', "'modality': 'subtitle'", '"modality": "subtitle"')
    )


def _text_items(schema: dict[str, Any]) -> list[str]:
    items: list[str] = []
    for key in ("scene_description", "uncertainty"):
        text = str(schema.get(key) or "").strip()
        if text and not _audio_or_subtitle_like(text):
            items.append(text)
    for key in ("observable_facts", "events", "visual_social_cues", "cross_clip_cues", "searchable_phrases"):
        value = schema.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, str):
                    if not _audio_or_subtitle_like(item):
                        items.append(item)
                elif isinstance(item, dict):
                    modality = str(item.get("modality") or "visual").lower()
                    if modality in {"audio", "subtitle", "asr"}:
                        continue
                    text = item.get("text") or item.get("event_description") or item.get("description") or item.get("cue")
                    if text and not _audio_or_subtitle_like(str(text)):
                        items.append(str(text))
    return [item.strip() for item in items if item and item.strip()]


def _has_negative_target_evidence(text: str) -> bool:
    lowered = text.lower()
    negative_terms = (
        "no vehicle",
        "no animated vehicle",
        "no animation",
        "not visible",
        "cannot determine",
        "does not provide enough information",
        "no visible",
    )
    return any(term in lowered for term in negative_terms)


def _negative_target_count(patch: dict[str, Any]) -> int:
    count = 0
    for node in patch.get("nodes") or []:
        if node.get("node_type") in {"l2_repair_reminder", "answerability_gap"}:
            continue
        text = str(node.get("text") or "").lower()
        if _has_negative_target_evidence(text):
            count += 1
    return count


def _build_l1_patch(example: dict[str, Any], row: dict[str, Any], schemas: list[dict[str, Any]], gaps: list[str]) -> dict[str, Any]:
    video_id = str((example.get("video") or {}).get("video_id") or "")
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    gap_id = stable_id("repair.gap", example.get("example_id"), gaps)
    nodes.append(
        {
            "node_id": gap_id,
            "node_type": "l2_repair_reminder",
            "text": f"Repair needed: {', '.join(gaps) or row.get('verifier_reason')}",
            "source_type": "quality_report",
            "producer": "run_repair_protocol",
            "visibility": {"mode": "video_only", "hidden_supervision": False},
        }
    )
    for schema in schemas:
        if schema.get("model_error"):
            continue
        clip_id = str(schema.get("clip_id") or "")
        time_span = schema.get("time_span") or {}
        for text in _text_items(schema)[:10]:
            node_type = "visual_social_cue" if any(gap in {"social_intent_or_affect", "causal_explanation"} for gap in gaps) else "observation"
            node_id = stable_id("repair.obs", example.get("example_id"), clip_id, text)
            nodes.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "text": text,
                    "modality": "visual",
                    "confidence": 0.72,
                    "clip_id": clip_id,
                    "time_span": time_span,
                    "source_type": "repair_clip_schema",
                    "producer": "qwen_repair_clip_schema",
                    "video_id": video_id,
                    "visibility": {"mode": "video_only", "hidden_supervision": False},
                    "provenance": {"created_by": "dataset_clip_wrapper.run_repair_protocol"},
                }
            )
            edges.append(
                {
                    "edge_id": stable_id("repair.edge", gap_id, node_id),
                    "src": node_id,
                    "dst": gap_id,
                    "edge_type": "repair_candidate_for",
                    "text": "Repair candidate evidence for the reported L2/L1 gap.",
                    "confidence": 0.6,
                    "evidence_refs": [node_id, gap_id],
                    "producer": "run_repair_protocol",
                    "visibility": {"mode": "video_only", "hidden_supervision": False},
                }
            )
    return {
        "schema_version": "video-skills-relaunch/repair-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "strategy": _strategy_for_gaps(gaps),
        "gap_types": gaps,
        "nodes": nodes,
        "edges": edges,
        "counts": {"nodes": len(nodes), "edges": len(edges), "repair_observation_nodes": max(0, len(nodes) - 1)},
    }


def _semantic_patch_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "semantic_repair_patch_nodes",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "patch_status": {"type": "string"},
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "node_type": {"type": "string"},
                                "text": {"type": "string"},
                                "support_refs": {"type": "array", "items": {"type": "string"}},
                                "reason_short": {"type": "string"},
                            },
                            "required": ["node_type", "text", "support_refs", "reason_short"],
                        },
                    },
                    "reason_short": {"type": "string"},
                },
                "required": ["patch_status", "nodes", "reason_short"],
            },
        },
    }


def _semantic_patch_candidates(patch: dict[str, Any], *, limit: int = 48, text_chars: int = 260) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for node in patch.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if not node_id or node.get("node_type") in {"l2_repair_reminder", "answerability_gap"}:
            continue
        text = str(node.get("text") or node.get("summary") or node.get("label") or "").strip()
        if not text:
            continue
        out.append(
            {
                "ref": str(node_id),
                "node_type": node.get("node_type"),
                "clip_id": node.get("clip_id"),
                "time_span": node.get("time_span") or {},
                "text": text[:text_chars],
            }
        )
        if len(out) >= limit:
            break
    return out


def _semantic_patch_allowed(clue_spec: dict[str, Any] | None) -> bool:
    hint = (clue_spec or {}).get("evidence_audit_hint")
    if not isinstance(hint, dict):
        return False
    if hint.get("primary_failure_class") in {
        "l1_graph_lacks_discriminative_node",
        "evidence_exists_verifier_too_strict",
        "insufficient_evidence_after_repair",
    }:
        return True
    return bool(hint.get("should_adjust_verifier") or hint.get("should_rerun_retrieval"))


def _augment_patch_with_llm_semantic_nodes(
    example: dict[str, Any],
    patch: dict[str, Any],
    *,
    clue_spec: dict[str, Any] | None,
    support_graph: dict[str, Any] | None = None,
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if not api_key or args.dry_run or args.skip_gptoss_verifier or args.disable_llm_semantic_patch:
        return patch
    if not _semantic_patch_allowed(clue_spec):
        return patch
    patch_candidates = _semantic_patch_candidates(patch, limit=args.semantic_patch_candidate_nodes)
    graph_candidates = (
        _compact_evidence_candidates(
            support_graph,
            limit=args.semantic_patch_candidate_nodes,
            text_chars=args.evidence_candidate_text_chars,
        )
        if isinstance(support_graph, dict)
        else []
    )
    seen_refs: set[str] = set()
    candidates: list[dict[str, Any]] = []
    for row in patch_candidates + graph_candidates:
        ref = str(row.get("ref") or "")
        if not ref or ref in seen_refs:
            continue
        candidates.append(row)
        seen_refs.add(ref)
        if len(candidates) >= args.semantic_patch_candidate_nodes:
            break
    if not candidates:
        return patch
    question = example.get("question") or {}
    allowed_refs = {row["ref"] for row in candidates}
    prompt = {
        "task": "Compose compact evidence-grounded semantic repair nodes for L1/L2 video QA.",
        "rules": [
            "Use only the provided visual_evidence_candidates as support.",
            "Do not use hidden answer labels, dataset annotations, audio, ASR, subtitles, narration, or dialogue.",
            "Create a semantic node only when the support_refs visibly justify the text.",
            "Prefer discriminative clue/context nodes that help option verification: scene category, visible attribute, action state, temporal relation, or missing visual clue.",
            "For scene/environment/category answers, a node may summarize multiple visible objects/actions that jointly establish the category.",
            "Keep each text short and factual. Do not answer the question unless the visual refs support it.",
            "If the refs do not support a useful node, return patch_status=no_grounded_semantic_patch and nodes=[].",
        ],
        "question": {
            "question_text": question.get("question_text"),
            "options": question.get("options") or [],
            "answer_format": question.get("answer_format"),
        },
        "clue_need_spec": clue_spec or {},
        "visual_evidence_candidates": candidates,
        "max_nodes": args.semantic_patch_max_nodes,
        "required_json_shape": _semantic_patch_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.verifier_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.semantic_patch_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.verifier_timeout_s,
    )
    try:
        payload = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You output JSON only. You create evidence-grounded semantic L1 repair nodes. "
                        "No markdown or prose outside JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format=_semantic_patch_schema(),
        )
    except Exception as exc:
        patch = dict(patch)
        patch.setdefault("metadata", {})
        patch["metadata"]["semantic_patch_error"] = str(exc)[:240]
        return patch

    new_nodes: list[dict[str, Any]] = []
    new_edges: list[dict[str, Any]] = []
    for row in (payload.get("nodes") or [])[: args.semantic_patch_max_nodes]:
        if not isinstance(row, dict):
            continue
        refs = [str(ref) for ref in row.get("support_refs") or [] if str(ref) in allowed_refs]
        text = str(row.get("text") or "").strip()
        if len(refs) < args.semantic_patch_min_refs or not text:
            continue
        node_id = stable_id("repair.semantic", example.get("example_id"), text, refs)
        node = {
            "node_id": node_id,
            "node_type": str(row.get("node_type") or "repair_semantic_observation"),
            "text": text[:500],
            "modality": "visual",
            "confidence": 0.78,
            "source_type": "repair_semantic_patch",
            "producer": args.verifier_model,
            "support_refs": refs,
            "reason_short": str(row.get("reason_short") or "")[:240],
            "visibility": {"mode": "video_only", "hidden_supervision": False},
            "provenance": {"created_by": "dataset_clip_wrapper.run_repair_protocol.semantic_patch"},
        }
        new_nodes.append(node)
        for ref in refs:
            new_edges.append(
                {
                    "edge_id": stable_id("repair.semantic.edge", ref, node_id),
                    "src": ref,
                    "dst": node_id,
                    "edge_type": "supports_semantic_patch",
                    "text": "Visual evidence supports a semantic repair clue node.",
                    "confidence": 0.72,
                    "evidence_refs": [ref, node_id],
                    "producer": args.verifier_model,
                    "visibility": {"mode": "video_only", "hidden_supervision": False},
                }
            )
    patch = dict(patch)
    patch["nodes"] = list(patch.get("nodes") or []) + new_nodes
    patch["edges"] = list(patch.get("edges") or []) + new_edges
    patch.setdefault("metadata", {})
    patch["metadata"]["semantic_patch"] = {
        "patch_status": payload.get("patch_status"),
        "node_count": len(new_nodes),
        "edge_count": len(new_edges),
        "reason_short": str(payload.get("reason_short") or "")[:240],
        "llm_usage": client.last_response_metadata,
    }
    patch["counts"] = {
        "nodes": len(patch.get("nodes") or []),
        "edges": len(patch.get("edges") or []),
        "repair_observation_nodes": sum(
            1
            for node in patch.get("nodes") or []
            if isinstance(node, dict) and node.get("node_type") not in {"l2_repair_reminder", "answerability_gap"}
        ),
        "semantic_repair_nodes": len(new_nodes),
    }
    return patch


def _merge_patch_graph(example: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    base = ((example.get("metadata") or {}).get("clue_memory_graph") or {"nodes": [], "edges": []})
    graph = dict(base)
    graph["nodes"] = list(base.get("nodes") or []) + list(patch.get("nodes") or [])
    graph["edges"] = list(base.get("edges") or []) + list(patch.get("edges") or [])
    graph.setdefault("metadata", {})
    graph["metadata"] = dict(graph.get("metadata") or {})
    graph["metadata"]["repair_patch_schema_version"] = patch.get("schema_version")
    graph["metadata"]["repair_gap_types"] = patch.get("gap_types") or []
    return graph


def _candidate_refs_for_option(graph: dict[str, Any], option: dict[str, Any], question_text: str, *, limit: int) -> list[str]:
    query_words = set(re.findall(r"[a-zA-Z0-9]+", f"{question_text} {option.get('text', '')}".lower()))
    option_words = set(re.findall(r"[a-zA-Z0-9]+", str(option.get("text") or "").lower()))
    target_words = set(re.findall(r"[a-zA-Z0-9]+", question_text.lower()))
    scored: list[tuple[float, str]] = []
    seen: set[str] = set()
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id or node.get("node_type") in {"l2_repair_reminder", "answerability_gap", "question_requirement"}:
            continue
        node_id = str(node_id)
        if node_id in seen:
            continue
        if not _is_verifiable_l1_node(node):
            continue
        text = str(node.get("text") or node.get("summary") or node.get("label") or "")
        words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        overlap = len(query_words & words)
        option_overlap = len(option_words & words)
        target_overlap = len(target_words & words)
        bonus = 2 if node.get("source_type") == "repair_clip_schema" else 0
        negative_penalty = 8 if _has_negative_target_evidence(text) else 0
        positive_option_bonus = 10 if option_overlap and not negative_penalty else 0
        target_bonus = min(target_overlap, 4)
        score = overlap + bonus + positive_option_bonus + option_overlap * 4 + target_bonus - negative_penalty
        if score > 0:
            scored.append((score, node_id))
            seen.add(node_id)
    scored.sort(reverse=True)
    return [ref for _, ref in scored[:limit]]


def _negative_refs_for_option(graph: dict[str, Any], option: dict[str, Any], question_text: str, *, limit: int) -> list[str]:
    query_words = set(re.findall(r"[a-zA-Z0-9]+", f"{question_text} {option.get('text', '')}".lower()))
    scored: list[tuple[float, str]] = []
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id or not _is_verifiable_l1_node(node):
            continue
        text = str(node.get("text") or node.get("summary") or node.get("label") or "")
        if not text or not _has_negative_target_evidence(text):
            continue
        words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        overlap = len(query_words & words)
        scored.append((overlap + 1.0, str(node_id)))
    scored.sort(reverse=True)
    return [ref for _, ref in scored[:limit]]


def _option_evidence_selector_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "option_evidence_pack_selection",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "option_packs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "option_label": {"type": "string"},
                                "positive_refs": {"type": "array", "items": {"type": "string"}},
                                "negative_refs": {"type": "array", "items": {"type": "string"}},
                                "missing_requirements": {"type": "array", "items": {"type": "string"}},
                                "reason_short": {"type": "string"},
                            },
                            "required": ["option_label", "positive_refs", "negative_refs", "missing_requirements", "reason_short"],
                        },
                    },
                    "selector_status": {"type": "string"},
                    "reason_short": {"type": "string"},
                },
                "required": ["option_packs", "selector_status", "reason_short"],
            },
        },
    }


def _compact_evidence_candidates(graph: dict[str, Any], *, limit: int, text_chars: int) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for node in graph.get("nodes") or []:
        if not isinstance(node, dict) or not _is_verifiable_l1_node(node):
            continue
        if not node.get("node_id"):
            continue
        text = " ".join(str(node.get(key) or "").strip() for key in ("text", "summary", "label") if node.get(key)).strip()
        if not text:
            continue
        candidates.append(
            {
                "ref": str(node.get("node_id")),
                "node_type": node.get("node_type"),
                "source_type": node.get("source_type"),
                "producer": node.get("producer"),
                "clip_id": node.get("clip_id"),
                "time_span": node.get("time_span"),
                "text": text[:text_chars],
            }
        )
    if len(candidates) <= limit:
        return candidates
    repair = [
        row
        for row in candidates
        if row.get("source_type") in {"repair_clip_schema", "repair_semantic_patch"}
        or row.get("producer") == "qwen_repair_clip_schema"
    ]
    base = [row for row in candidates if row not in repair]
    remaining = max(0, limit - len(repair))
    if len(base) <= remaining:
        return (repair + base)[:limit]
    if remaining <= 0:
        return repair[:limit]
    # Budget-only temporal coverage: keep the candidate table compact without semantic scoring.
    step = max(1, len(base) / remaining)
    sampled: list[dict[str, Any]] = []
    for idx in range(remaining):
        sampled.append(base[min(len(base) - 1, int(idx * step))])
    return repair[:limit] + sampled[:remaining]


def _select_option_evidence_packs_with_llm(
    example: dict[str, Any],
    graph: dict[str, Any],
    *,
    clue_spec: dict[str, Any] | None,
    gaps: list[str] | None,
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not api_key or args.skip_gptoss_verifier:
        return None
    question = example.get("question") or {}
    options = [opt for opt in question.get("options") or [] if isinstance(opt, dict)]
    if not options:
        return None
    total_verifiable = sum(1 for node in graph.get("nodes") or [] if isinstance(node, dict) and _is_verifiable_l1_node(node))
    base_candidates = _compact_evidence_candidates(
        graph,
        limit=args.max_evidence_candidate_nodes,
        text_chars=args.evidence_candidate_text_chars,
    )
    if not base_candidates:
        return {
            "selector_status": "no_candidates",
            "option_packs": [],
            "reason_short": "No verifiable L1 evidence nodes are available.",
            "llm_usage": {},
        }
    attempts = [
        {
            "name": "full",
            "candidates": base_candidates,
            "max_tokens": args.evidence_selector_max_tokens,
            "system_suffix": "Select refs; do not answer from memory or hidden labels.",
        },
        {
            "name": "compact_json_retry",
            "candidates": _compact_evidence_candidates(
                graph,
                limit=min(args.max_evidence_candidate_nodes, 60),
                text_chars=min(args.evidence_candidate_text_chars, 140),
            ),
            "max_tokens": max(args.evidence_selector_max_tokens, 1200),
            "system_suffix": (
                "Return exactly one JSON object matching the schema. "
                "Use short arrays of refs and short strings only; no markdown or prose."
            ),
        },
    ]
    payload: dict[str, Any] | None = None
    candidates: list[dict[str, Any]] = base_candidates
    client: OpenRouterClient | None = None
    errors: list[str] = []
    for attempt_index, attempt in enumerate(attempts):
        candidates = attempt["candidates"]
        allowed_refs = {row["ref"] for row in candidates}
        prompt = {
            "task": "Select option-specific evidence packs for video QA repair.",
            "selector_attempt": attempt["name"],
            "rules": [
                "Use only refs from evidence_candidates.",
            "Do not use hidden answer labels, dataset annotations, audio, ASR, subtitles, or dialogue as evidence.",
            "For each option, choose positive_refs that directly support the option from visible evidence.",
            "Choose negative_refs only when a ref visibly contradicts or weakens that option.",
            "If no direct visual support exists, leave positive_refs empty and explain missing_requirements.",
            "For scene, environment, or category options, visible objects/actions may jointly support the option; select the grouped refs and explain the category cue.",
            "Prefer compact semantic repair nodes when they are supported by visual refs, but keep their support_refs traceable through the graph.",
            "Do not select the same generic context refs for every option unless they truly support every option.",
            "Keep reason_short to one short sentence per option.",
        ],
            "question": {
                "question_text": question.get("question_text"),
                "options": options,
                "answer_format": question.get("answer_format"),
            },
            "gap_types": gaps or [],
            "clue_need_spec": clue_spec or {},
            "max_positive_refs_per_option": args.max_verify_refs,
            "max_negative_refs_per_option": max(2, args.max_verify_refs // 2),
            "evidence_candidates": candidates,
            "required_json_shape": _option_evidence_selector_schema()["json_schema"]["schema"]["properties"],
        }
        client = OpenRouterClient(
            model=args.verifier_model,
            api_key=api_key,
            temperature=0.0,
            max_tokens=attempt["max_tokens"],
            reasoning={"effort": "medium", "exclude": True},
            timeout_s=args.verifier_timeout_s,
        )
        try:
            payload = client.chat_json(
                [
                    {
                        "role": "system",
                        "content": (
                            "You output JSON only. You are a strict evidence-pack selector for video QA. "
                            f"{attempt['system_suffix']}"
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
                ],
                response_format=_option_evidence_selector_schema(),
            )
            payload["selector_attempt"] = attempt["name"]
            usage = dict(client.last_response_metadata or {})
            usage["compact_retry_count"] = int(usage.get("compact_retry_count") or 0) + attempt_index
            if attempt_index:
                usage["malformed_json_count"] = int(usage.get("malformed_json_count") or 0) + len(errors)
            payload["llm_usage"] = usage
            break
        except Exception as exc:
            errors.append(f"{attempt['name']}: {exc}")
            payload = None
            continue
    if payload is None:
        if args.allow_lexical_fallback:
            return {
                "selector_status": "llm_selector_failed_heuristic_fallback_allowed",
                "option_packs": [],
                "reason_short": "; ".join(errors)[-240:],
                "llm_usage": (client.last_response_metadata if client is not None else {}),
            }
        return {
            "selector_status": "error",
            "selector_backend": args.verifier_model,
            "option_packs": [],
            "reason_short": "; ".join(errors)[-240:],
            "llm_usage": {
                "malformed_json_count": len(errors),
                "compact_retry_count": max(0, len(errors) - 1),
            },
        }
    allowed_refs = {row["ref"] for row in candidates}
    option_labels = {str(opt.get("label")) for opt in options if opt.get("label") is not None}
    packs: list[dict[str, Any]] = []
    for pack in payload.get("option_packs") or []:
        if not isinstance(pack, dict) or str(pack.get("option_label")) not in option_labels:
            continue
        positive_refs = [str(ref) for ref in pack.get("positive_refs") or [] if str(ref) in allowed_refs]
        negative_refs = [str(ref) for ref in pack.get("negative_refs") or [] if str(ref) in allowed_refs]
        packs.append(
            {
                "option_label": str(pack.get("option_label")),
                "positive_refs": positive_refs[: args.max_verify_refs],
                "negative_refs": negative_refs[: max(2, args.max_verify_refs // 2)],
                "missing_requirements": [str(item) for item in pack.get("missing_requirements") or []],
                "reason_short": str(pack.get("reason_short") or "")[:240],
            }
        )
    payload["option_packs"] = packs
    payload["candidate_count"] = len(candidates)
    payload["candidate_truncated"] = len(candidates) < total_verifiable
    payload["selector_backend"] = args.verifier_model
    payload.setdefault("llm_usage", client.last_response_metadata if client is not None else {})
    return payload


def _is_verifiable_l1_node(node: dict[str, Any]) -> bool:
    if node.get("node_type") in {"l2_repair_reminder", "answerability_gap", "question_requirement", "clip"}:
        return False
    text = str(node.get("text") or node.get("summary") or node.get("label") or "")
    if not text.strip():
        return False
    source_type = str(node.get("source_type") or "")
    producer = str(node.get("producer") or "")
    if source_type in {
        "repair_clip_schema",
        "repair_semantic_patch",
        "observation",
        "qwen_clip_schema_anchor",
        "entity_mention",
        "state",
        "event",
        "place",
        "searchable_phrase",
    }:
        return True
    return producer in {
        "qwen_repair_clip_schema",
        "openai/gpt-oss-120b",
        "neighbor_vlm_l1_graph_composer",
        "neighbor_vlm_l1_schema_anchor",
    }


def _bridge_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "objective_background_bridge_verification",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "bridge_status": {"type": "string"},
                    "best_option": {
                        "type": "object",
                        "additionalProperties": True,
                        "properties": {
                            "label": {"type": "string"},
                            "text": {"type": "string"},
                            "confidence": {"type": "number"},
                        },
                        "required": ["label", "text", "confidence"],
                    },
                    "visual_anchor_refs": {"type": "array", "items": {"type": "string"}},
                    "visual_anchor_summary": {"type": "array", "items": {"type": "string"}},
                    "objective_background_facts": {"type": "array", "items": {"type": "string"}},
                    "bridge_claim": {"type": "string"},
                    "not_direct_visual_evidence": {"type": "boolean"},
                    "reason_short": {"type": "string"},
                },
                "required": [
                    "bridge_status",
                    "best_option",
                    "visual_anchor_refs",
                    "visual_anchor_summary",
                    "objective_background_facts",
                    "bridge_claim",
                    "not_direct_visual_evidence",
                    "reason_short",
                ],
            },
        },
    }


def _node_text_by_id(graph: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id:
            continue
        text = str(node.get("text") or node.get("summary") or node.get("label") or "")
        if text:
            out[str(node_id)] = text
    return out


def _bridge_context_refs(
    graph: dict[str, Any],
    question_text: str,
    options: list[dict[str, Any]],
    clue_spec: dict[str, Any],
    *,
    limit: int,
) -> list[str]:
    query_parts = [question_text, str(clue_spec.get("visual_target") or "")]
    query_parts.extend(str(item) for item in clue_spec.get("must_find_visual_evidence") or [])
    query_parts.extend(str(item) for item in clue_spec.get("bridge_evidence_criteria") or [])
    query_parts.extend(str(opt.get("text") or "") for opt in options)
    query_words = set(re.findall(r"[a-zA-Z0-9]+", " ".join(query_parts).lower()))
    scored: list[tuple[float, str]] = []
    for node in graph.get("nodes") or []:
        node_id = node.get("node_id")
        if not node_id or not _is_verifiable_l1_node(node):
            continue
        text = str(node.get("text") or node.get("summary") or node.get("label") or "")
        words = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
        overlap = len(query_words & words)
        bonus = 2 if node.get("source_type") == "repair_clip_schema" else 0
        if overlap or bonus:
            scored.append((overlap + bonus, str(node_id)))
    scored.sort(reverse=True)
    return [ref for _, ref in scored[:limit]]


def _can_attempt_bridge(gaps: list[str], clue_spec: dict[str, Any]) -> bool:
    mode = str(clue_spec.get("answer_mode_hint") or "")
    if mode == "visual_context_plus_background":
        return True
    if clue_spec.get("objective_background_facts"):
        return True
    return bool({"social_intent_or_affect", "causal_explanation", "commonsense_bridge"} & set(gaps))


def _option_by_label(options: list[dict[str, Any]], label: Any) -> dict[str, Any]:
    for opt in options:
        if str(opt.get("label")) == str(label):
            return opt
    return {}


def _attempt_objective_bridge(
    example: dict[str, Any],
    graph: dict[str, Any],
    *,
    clue_spec: dict[str, Any],
    gaps: list[str],
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if not api_key or args.disable_background_bridge or not _can_attempt_bridge(gaps, clue_spec):
        return None
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    options = [opt for opt in question.get("options") or [] if isinstance(opt, dict)]
    option_labels = {str(opt.get("label")) for opt in options if opt.get("label") is not None}
    refs = _bridge_context_refs(graph, question_text, options, clue_spec, limit=args.max_bridge_refs)
    if not refs:
        return None
    text_by_id = _node_text_by_id(graph)
    evidence_pack = [{"ref": ref, "text": text_by_id.get(ref, "")[:500]} for ref in refs if text_by_id.get(ref)]
    if not evidence_pack:
        return None
    prompt = {
        "task": "Verify whether a video-only answer can be accepted via objective background bridge.",
        "rules": [
            "Visual anchors must come only from the provided evidence_pack refs.",
            "Objective background facts may be used only as bridge knowledge, never as direct visual evidence.",
            "Accept accepted_bridge only when the visual anchors identify the relevant situation/context and the background facts are stable, objective, and sufficient to choose one option.",
            "Reject bridge claims based on subjective intention, emotion, social stereotypes, audio/dialogue/subtitles, hidden labels, or private dataset knowledge.",
            "If multiple options remain plausible, return bridge_insufficient.",
            "Keep visual_anchor_summary to at most 4 short strings and reason_short to one sentence.",
        ],
        "question": {
            "question_text": question_text,
            "options": options,
        },
        "gap_types": gaps,
        "clue_need_spec": clue_spec,
        "allowed_visual_anchor_refs": refs,
        "evidence_pack": evidence_pack,
        "required_json_shape": _bridge_response_schema()["json_schema"]["schema"]["properties"],
    }
    client = OpenRouterClient(
        model=args.bridge_model,
        api_key=api_key,
        temperature=0.0,
        max_tokens=args.bridge_max_tokens,
        reasoning={"effort": "medium", "exclude": True},
        timeout_s=args.verifier_timeout_s,
    )
    try:
        payload = client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You output JSON only. You are a strict verifier for objective background bridges. "
                        "Do not include analysis, markdown, or prose outside JSON."
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            response_format=_bridge_response_schema(),
        )
        payload["llm_usage"] = client.last_response_metadata
    except Exception as exc:
        return {
            "bridge_status": "bridge_insufficient",
            "best_option": {"label": None, "text": "", "confidence": 0.0},
            "visual_anchor_refs": refs[: args.max_bridge_refs],
            "visual_anchor_summary": [row["text"] for row in evidence_pack[: args.max_bridge_refs]],
            "objective_background_facts": clue_spec.get("objective_background_facts") or [],
            "bridge_claim": "",
            "not_direct_visual_evidence": True,
            "reason_short": f"bridge verifier failed: {exc}",
            "bridge_backend": args.bridge_model,
            "llm_usage": client.last_response_metadata,
        }
    visual_refs = [str(ref) for ref in payload.get("visual_anchor_refs") or [] if str(ref) in refs]
    best = payload.get("best_option") if isinstance(payload.get("best_option"), dict) else {}
    confidence = float(best.get("confidence") or 0.0)
    if (
        payload.get("bridge_status") == "accepted_bridge"
        and str(best.get("label")) in option_labels
        and len(visual_refs) >= args.min_bridge_refs
        and confidence >= args.min_bridge_confidence
        and payload.get("objective_background_facts")
    ):
        payload["visual_anchor_refs"] = visual_refs[: args.max_bridge_refs]
        payload["bridge_backend"] = args.bridge_model
        return payload
    payload["bridge_status"] = "bridge_insufficient"
    payload["visual_anchor_refs"] = visual_refs[: args.max_bridge_refs]
    payload["bridge_backend"] = args.bridge_model
    return payload


def _verify_options(
    example: dict[str, Any],
    graph: dict[str, Any],
    *,
    clue_spec: dict[str, Any] | None = None,
    gaps: list[str] | None = None,
    api_key: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    question = example.get("question") or {}
    question_text = str(question.get("question_text") or "")
    options = question.get("options") or []
    executor: SkillExecutor
    llm_client: SkillModelClient | None = None
    if api_key and not args.skip_gptoss_verifier:
        llm_client = SkillModelClient(
            model=args.verifier_model,
            api_key=api_key,
            max_tokens=args.verifier_max_tokens,
            temperature=0.0,
            timeout_s=args.verifier_timeout_s,
        )
        executor = SkillExecutor(
            llm_client=llm_client,
            config=SkillBackendConfig(default_mode=SkillBackendMode.RULE, llm_skills={"verify_claim_support"}),
        )
        backend = "gptoss_verifier"
    else:
        executor = SkillExecutor(config=SkillBackendConfig(default_mode=SkillBackendMode.RULE))
        backend = "rule_verifier"

    selector_payload = _select_option_evidence_packs_with_llm(
        example,
        graph,
        clue_spec=clue_spec,
        gaps=gaps,
        api_key=api_key,
        args=args,
    )
    pack_by_label = {
        str(pack.get("option_label")): pack
        for pack in ((selector_payload or {}).get("option_packs") or [])
        if isinstance(pack, dict)
    }
    selector_uses_llm = bool(selector_payload and selector_payload.get("selector_backend"))
    selector_failed_strict = bool(
        selector_payload
        and selector_payload.get("selector_status") in {"error", "no_candidates"}
        and not args.allow_lexical_fallback
    )
    audit_hint = (clue_spec or {}).get("evidence_audit_hint") if isinstance(clue_spec, dict) else {}
    allow_llm_alignment_override = bool(
        isinstance(audit_hint, dict)
        and audit_hint.get("should_adjust_verifier")
        and str(audit_hint.get("primary_failure_class") or "")
        in {"evidence_exists_verifier_too_strict", "insufficient_evidence_after_repair"}
    )

    verifications: list[dict[str, Any]] = []
    for opt in options:
        selected_pack = pack_by_label.get(str(opt.get("label"))) if selector_payload else None
        if selected_pack is not None:
            refs = [str(ref) for ref in selected_pack.get("positive_refs") or []]
            negative_refs = [str(ref) for ref in selected_pack.get("negative_refs") or []]
            missing_requirements = [str(item) for item in selected_pack.get("missing_requirements") or []]
            selector_reason = str(selected_pack.get("reason_short") or "")
            evidence_selector = "gptoss_option_evidence_selector"
        else:
            refs = [] if selector_failed_strict else _candidate_refs_for_option(graph, opt, question_text, limit=args.max_verify_refs)
            negative_refs = (
                []
                if selector_failed_strict
                else _negative_refs_for_option(graph, opt, question_text, limit=max(2, args.max_verify_refs // 2))
            )
            missing_requirements = []
            selector_reason = str((selector_payload or {}).get("reason_short") or "")
            evidence_selector = (
                "gptoss_option_evidence_selector_error"
                if selector_failed_strict
                else "heuristic_diagnostic_selector"
            )
        claim = {
            "claim_text": f"{opt.get('label')}: {opt.get('text')}",
            "option_label": opt.get("label"),
            "question_text": question_text,
        }
        if not refs:
            verifications.append(
                {
                    "option_label": opt.get("label"),
                    "option_text": opt.get("text"),
                    "positive_refs": [],
                    "negative_refs": negative_refs,
                    "evidence_refs": [],
                    "ok": False,
                    "failure_code": "no_positive_refs_selected",
                    "confidence": 0.0,
                    "support_score": 0.0,
                    "target_alignment_score": 0.0,
                    "verifier_decision": "insufficient",
                    "reason_short": "No positive visual evidence refs were selected for this option.",
                    "selector_reason_short": selector_reason[:240],
                    "missing_requirements": missing_requirements,
                    "evidence_selector": evidence_selector,
                    "llm_usage": {},
                    "outputs": {"messages": ["no_positive_refs_selected"]},
                }
            )
            continue
        result = executor.execute(
            "verify_claim_support",
            args={
                "claim": claim,
                "question_text": question_text,
                "evidence_chain": {"evidence_refs": refs},
                "support_policy": {
                    "min_evidence_refs": args.min_verify_refs,
                    "min_claim_score": 0.05,
                    "min_target_score": 0.05,
                    "allow_llm_target_alignment_override": allow_llm_alignment_override,
                    "llm_alignment_override_min_score": 0.75,
                },
            },
            graph=graph,
        )
        usage = dict(llm_client.last_response_metadata or {}) if llm_client is not None else {}
        messages = result.outputs.get("messages") if isinstance(result.outputs, dict) else []
        verifications.append(
            {
                "option_label": opt.get("label"),
                "option_text": opt.get("text"),
                "positive_refs": result.evidence_refs if result.ok else refs,
                "negative_refs": negative_refs,
                "evidence_refs": refs,
                "ok": bool(result.ok),
                "failure_code": result.failure_code,
                "confidence": result.confidence,
                "support_score": (result.outputs or {}).get("claim_support_score") if isinstance(result.outputs, dict) else None,
                "target_alignment_score": (result.outputs or {}).get("target_alignment_score") if isinstance(result.outputs, dict) else None,
                "verifier_decision": "supported" if result.ok else "insufficient",
                "reason_short": str(messages[0])[:240] if isinstance(messages, list) and messages else "",
                "selector_reason_short": selector_reason[:240],
                "missing_requirements": missing_requirements,
                "evidence_selector": evidence_selector,
                "llm_usage": usage,
                "outputs": result.outputs,
            }
        )

    passed = [row for row in verifications if row["ok"]]
    if passed:
        passed.sort(key=lambda row: float(row.get("confidence") or 0.0), reverse=True)
        best = passed[0]
        second_conf = float(passed[1].get("confidence") or 0.0) if len(passed) > 1 else 0.0
        margin = float(best.get("confidence") or 0.0) - second_conf
        strong_enough = (
            backend == "gptoss_verifier"
            and selector_uses_llm
            and
            len(best.get("positive_refs") or []) >= args.min_verify_refs
            and float(best.get("confidence") or 0.0) >= args.min_verify_confidence
            and margin >= args.min_option_margin
        )
        status = "resolved_strong" if strong_enough else "needs_more_evidence"
    else:
        status = "needs_more_evidence"
        best = max(verifications, key=lambda row: float(row.get("confidence") or 0.0), default={})
        second_conf = 0.0
        margin = 0.0
    l2 = {
        "schema_version": "video-skills-relaunch/repair-l2-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "backend": backend,
        "repair_status": status,
        "best_option": {
            "label": best.get("option_label"),
            "text": best.get("option_text"),
            "confidence": best.get("confidence"),
        },
        "option_verifier_policy": {
            "min_verify_refs": args.min_verify_refs,
            "max_verify_refs": args.max_verify_refs,
            "min_verify_confidence": args.min_verify_confidence,
            "min_option_margin": args.min_option_margin,
            "top_margin": round(margin, 4),
            "supported_option_count": len(passed),
            "requires_llm_evidence_selector": True,
        },
        "option_evidence_selector": selector_payload or {"selector_status": "not_run", "reason_short": "No API verifier available; heuristic diagnostic selector used."},
        "option_verifications": verifications,
    }
    if status != "resolved_strong":
        bridge = _attempt_objective_bridge(
            example,
            graph,
            clue_spec=clue_spec or {},
            gaps=gaps or [],
            api_key=api_key,
            args=args,
        )
        if bridge:
            l2["background_bridge_verification"] = bridge
            if bridge.get("bridge_status") == "accepted_bridge":
                best_bridge = bridge.get("best_option") if isinstance(bridge.get("best_option"), dict) else {}
                bridge_refs = [str(ref) for ref in bridge.get("visual_anchor_refs") or []]
                direct_bridge = bridge.get("not_direct_visual_evidence") is False
                if direct_bridge and len(bridge_refs) >= args.min_verify_refs:
                    bridge_opt = _option_by_label(options, best_bridge.get("label"))
                    bridge_claim = {
                        "claim_text": f"{bridge_opt.get('label')}: {bridge_opt.get('text')}",
                        "option_label": bridge_opt.get("label"),
                        "question_text": question_text,
                    }
                    bridge_check = executor.execute(
                        "verify_claim_support",
                        args={
                            "claim": bridge_claim,
                            "question_text": question_text,
                            "evidence_chain": {"evidence_refs": bridge_refs},
                            "support_policy": {
                                "min_evidence_refs": args.min_verify_refs,
                                "min_claim_score": 0.05,
                                "min_target_score": 0.05,
                                "allow_llm_target_alignment_override": allow_llm_alignment_override,
                                "llm_alignment_override_min_score": 0.75,
                            },
                        },
                        graph=graph,
                    )
                    l2["bridge_ref_verification"] = {
                        "evidence_refs": bridge_refs,
                        "ok": bool(bridge_check.ok),
                        "failure_code": bridge_check.failure_code,
                        "confidence": bridge_check.confidence,
                        "outputs": bridge_check.outputs,
                    }
                    if bridge_check.ok:
                        l2["repair_status"] = "resolved_strong"
                        l2["backend"] = f"{backend}+bridge_ref_verify"
                    else:
                        l2["repair_status"] = "accepted_bridge"
                        l2["backend"] = f"{backend}+objective_bridge"
                else:
                    l2["repair_status"] = "accepted_bridge"
                    l2["backend"] = f"{backend}+objective_bridge"
                l2["best_option"] = {
                    "label": best_bridge.get("label"),
                    "text": best_bridge.get("text"),
                    "confidence": best_bridge.get("confidence"),
                }
    budget_rows = []
    if isinstance(selector_payload, dict):
        budget_rows.append(selector_payload)
    budget_rows.extend(row for row in verifications if isinstance(row, dict))
    bridge_payload = l2.get("background_bridge_verification")
    if isinstance(bridge_payload, dict):
        budget_rows.append(bridge_payload)
    l2["llm_budget_summary"] = _llm_usage_summary(budget_rows)
    return l2


def _build_report(plan: dict[str, Any], patch: dict[str, Any], l2: dict[str, Any]) -> dict[str, Any]:
    strong = l2.get("repair_status") == "resolved_strong"
    bridge = l2.get("repair_status") == "accepted_bridge"
    negative_count = _negative_target_count(patch)
    span_selection = plan.get("span_selection") or {}
    reported_selection_mode = span_selection.get("selection_mode")
    reported_selector_status = span_selection.get("selector_status")
    if reported_selection_mode in {None, "", "none", "abstain"}:
        for round_row in reversed(span_selection.get("retrieval_rounds") or []):
            if (
                isinstance(round_row, dict)
                and round_row.get("selected_after_exclusion")
                and not round_row.get("selector_abstained")
                and round_row.get("selection_mode")
            ):
                reported_selection_mode = round_row.get("selection_mode")
                break
    if reported_selector_status in {None, "", "none"}:
        for round_row in reversed(span_selection.get("retrieval_rounds") or []):
            if isinstance(round_row, dict) and round_row.get("selector_status"):
                reported_selector_status = round_row.get("selector_status")
                break
    missing = set(plan.get("gap_types") or [])
    if strong:
        next_action = "commit repaired evidence pack"
        failure_type = "resolved"
    elif bridge:
        next_action = "commit bridge answer with explicit non-visual background facts kept outside L1 evidence"
        failure_type = "resolved_with_objective_background_bridge"
    elif span_selection.get("selector_abstained"):
        next_action = "mark visual-only evidence missing; do not use heuristic fallback"
        failure_type = "visual_only_benchmark_limitation"
    elif (
        (l2.get("option_verifier_policy") or {}).get("supported_option_count", 0) > 1
        and float((l2.get("option_verifier_policy") or {}).get("top_margin") or 0.0) < float((l2.get("option_verifier_policy") or {}).get("min_option_margin") or 0.0)
    ):
        next_action = "retrieve or compose more discriminative evidence before committing any option"
        failure_type = "l2_option_margin_insufficient"
    elif negative_count or span_selection.get("negative_coarse_indices"):
        next_action = "reroute coarse retrieval because repair clips contain negative target evidence"
        failure_type = "l1_target_coverage_failure"
    elif {"social_intent_or_affect", "causal_explanation"} & missing:
        next_action = "keep visual-only boundary and try bridge verification with more target-aligned context"
        failure_type = "l1_context_partial_l2_bridge_needed"
    else:
        next_action = "expand another fine-pass around visually relevant context"
        failure_type = "l1_attribute_or_evidence_resolution_failure"
    return {
        "example_id": plan.get("example_id"),
        "dataset": plan.get("dataset"),
        "strategy": plan.get("strategy"),
        "repair_mode": span_selection.get("mode"),
        "gap_types": plan.get("gap_types") or [],
        "failure_type": failure_type,
        "repair_status": l2.get("repair_status"),
        "best_option": l2.get("best_option"),
        "patch_counts": patch.get("counts") or {},
        "negative_target_evidence_nodes": negative_count,
        "negative_coarse_indices": span_selection.get("negative_coarse_indices") or [],
        "selected_coarse_indices": span_selection.get("selected_coarse_indices") or [],
        "retrieval_round_count": len(span_selection.get("retrieval_rounds") or []),
        "selector_abstained": bool(span_selection.get("selector_abstained")),
        "selector_status": reported_selector_status,
        "selection_mode": reported_selection_mode,
        "verifier_backend": l2.get("backend"),
        "option_verifier_policy": l2.get("option_verifier_policy") or {},
        "option_evidence_packs": [
            {
                "option_label": item.get("option_label"),
                "verifier_decision": item.get("verifier_decision"),
                "confidence": item.get("confidence"),
                "positive_refs": item.get("positive_refs") or [],
                "negative_refs": item.get("negative_refs") or [],
                "missing_requirements": item.get("missing_requirements") or [],
                "evidence_selector": item.get("evidence_selector"),
                "selector_reason_short": item.get("selector_reason_short") or "",
                "reason_short": item.get("reason_short") or "",
            }
            for item in l2.get("option_verifications") or []
            if isinstance(item, dict)
        ],
        "option_evidence_selector": l2.get("option_evidence_selector") or {},
        "not_direct_visual_evidence": bridge,
        "repair_needed_after_round": not (strong or bridge),
        "verifier_reason": (
            "strong repair evidence verified"
            if strong
            else (
                "accepted via visual anchors plus objective background bridge"
                if bridge
                else "repair evidence remains weak or insufficient"
            )
        ),
        "background_bridge_verification": l2.get("background_bridge_verification"),
        "llm_budget_summary": {
            "repair_plan": plan.get("llm_budget_summary") or {},
            "l2_verifier": l2.get("llm_budget_summary") or {},
        },
        "recommended_next_action": next_action,
        "artifact_paths": plan.get("artifact_paths") or {},
    }


def _spans_from_cached_plan(plan: dict[str, Any]) -> list[ClipSpan]:
    spans: list[ClipSpan] = []
    for row in plan.get("spans") or []:
        if not isinstance(row, dict):
            continue
        payload = row.get("time_span") if isinstance(row.get("time_span"), dict) else row
        try:
            spans.append(
                ClipSpan(
                    start_s=float(payload.get("start_s")),
                    end_s=float(payload.get("end_s")),
                    granularity=str(payload.get("granularity") or row.get("granularity") or "fine"),
                    parent_index=payload.get("parent_index", row.get("parent_index")),
                    clip_index=int(payload.get("clip_index", row.get("clip_index") or 0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return spans


def _process_row(
    row: dict[str, Any],
    args: argparse.Namespace,
    api_key: str | None,
    *,
    evidence_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    example = _load_source_example(row)
    gaps = _gap_types(row)
    strategy = _strategy_for_gaps(gaps)
    out_dir = args.stage_dir / _safe_name(f"{row.get('dataset')}_{row.get('example_id')}")
    plan_path = out_dir / "repair_01_plan.json"
    schemas_path = out_dir / "repair_02_clip_schemas.jsonl"
    patch_path = out_dir / "repair_03_l1_patch.json"
    l2_path = out_dir / "repair_04_l2_verifier.json"
    report_path = out_dir / "repair_05_report.json"

    cached_plan = _read_json(plan_path) if plan_path.exists() and not args.force_repair_stages else None
    prior_schemas = _read_jsonl(schemas_path) if schemas_path.exists() else []
    prior_negative_count = sum(1 for schema in prior_schemas if _has_negative_target_evidence(" ".join(_text_items(schema))))
    video_regime = str(row.get("video_regime") or (example.get("metadata") or {}).get("video_regime") or "")
    if cached_plan and isinstance(cached_plan.get("clue_need_spec"), dict):
        clue_spec = cached_plan["clue_need_spec"]
    elif video_regime and video_regime != "long":
        clue_spec = _fallback_clue_need_spec(example, row, gaps)
        clue_spec["planner_backend"] = "existing_l1_option_verification_no_clue_planner"
    else:
        clue_spec = _build_clue_need_spec(
            example,
            row,
            gaps,
            prior_schemas=prior_schemas,
            api_key=api_key,
            args=args,
        )
    clue_spec = _attach_audit_hint(clue_spec, evidence_audit)
    use_reroute = args.repair_mode == "reroute" or (
        args.repair_mode == "auto" and prior_negative_count >= args.negative_reroute_threshold
    )
    if cached_plan and isinstance(cached_plan.get("span_selection"), dict):
        span_meta = cached_plan["span_selection"]
        spans = _spans_from_cached_plan(cached_plan)
    elif video_regime and video_regime != "long":
        spans, span_meta = _select_existing_l1_repair_spans(example)
    elif use_reroute:
        spans, span_meta = _select_rerouted_repair_spans(
            example,
            row,
            gaps=gaps,
            clue_spec=clue_spec,
            prior_schemas=prior_schemas,
            max_repair_clips=args.max_repair_clips,
            reroute_topk=args.reroute_topk,
            reroute_topk_per_query=args.reroute_topk_per_query,
            api_key=api_key,
            args=args,
        )
    else:
        spans, span_meta = _select_repair_spans(
            example,
            row,
            radius=args.coarse_radius,
            max_repair_clips=args.max_repair_clips,
        )
        span_meta["prior_negative_schema_count"] = prior_negative_count
    query = _repair_query(example, row, gaps, clue_spec=clue_spec)
    plan = {
        "schema_version": "video-skills-relaunch/repair-plan-v0.1",
        "example_id": example.get("example_id"),
        "dataset": example.get("dataset"),
        "video_regime": video_regime,
        "source_path": row.get("source_path"),
        "strategy": strategy,
        "repair_mode": span_meta.get("mode"),
        "gap_types": gaps,
        "clue_need_spec": clue_spec,
        "evidence_audit_hint": _compact_audit_hint(evidence_audit),
        "repair_query": query,
        "span_selection": span_meta,
        "spans": [
            {"clip_index": span.clip_index, "parent_index": span.parent_index, "time_span": span.to_dict(), "granularity": span.granularity}
            for span in spans
        ],
        "acceptance_rule": (
            "Repair becomes resolved_strong only when verify_claim_support passes with non-diagnostic visual evidence refs. "
            "It may become accepted_bridge when visual anchors plus stable objective background facts support one option; "
            "background facts are L2 bridge context, not L1 evidence."
        ),
        "artifact_paths": {
            "plan": str(plan_path),
            "clip_schemas": str(schemas_path),
            "l1_patch": str(patch_path),
            "l2_verifier": str(l2_path),
            "report": str(report_path),
        },
    }
    plan["llm_budget_summary"] = _llm_usage_summary(
        [clue_spec] + [row for row in span_meta.get("retrieval_rounds") or [] if isinstance(row, dict)]
    )
    _write_json(plan_path, plan)

    if args.dry_run:
        patch = _build_l1_patch(example, row, [], gaps)
        l2 = {"repair_status": "dry_run", "backend": "none", "best_option": {}}
        report = _build_report(plan, patch, l2)
        trajectory = repair_artifacts_to_trajectory(plan=plan, patch=patch, l2=l2, report=report)
        l2["l2_trajectory"] = trajectory
        l2["repair_subgraph"] = trajectory.get("repair_subgraph") or {}
        report["l2_trajectory"] = trajectory
        report["repair_subgraph"] = trajectory.get("repair_subgraph") or {}
        _write_json(patch_path, patch)
        _write_json(l2_path, l2)
        _write_json(report_path, report)
        return report

    schemas: list[dict[str, Any]]
    if not spans:
        schemas = []
    elif args.skip_api and schemas_path.exists():
        schemas = _read_jsonl(schemas_path)
    elif args.skip_api:
        schemas = []
    else:
        if not api_key:
            raise RuntimeError("OpenRouter API key is required unless --dry-run or --skip-api is used")
        schemas = _produce_repair_schemas(
            example=example,
            spans=spans,
            repair_query=query,
            api_key=api_key,
            args=args,
            schemas_path=schemas_path,
        )
    if schemas and not schemas_path.exists():
        _write_jsonl(schemas_path, schemas)

    if patch_path.exists() and not args.force_repair_stages:
        patch = _read_json(patch_path)
    else:
        patch = _build_l1_patch(example, row, schemas, gaps)
        patch = _augment_patch_with_llm_semantic_nodes(
            example,
            patch,
            clue_spec=clue_spec,
            support_graph=(example.get("metadata") or {}).get("clue_memory_graph") or {},
            api_key=api_key,
            args=args,
        )
        _write_json(patch_path, patch)
    repaired_graph = _merge_patch_graph(example, patch)
    if l2_path.exists() and not args.force_repair_stages:
        l2 = _read_json(l2_path)
    else:
        l2 = _verify_options(example, repaired_graph, clue_spec=clue_spec, gaps=gaps, api_key=api_key, args=args)
        if not spans and span_meta.get("selector_abstained") and l2.get("repair_status") not in {"resolved_strong", "accepted_bridge"}:
            l2["backend"] = f"{l2.get('backend')}+selector_abstained"
            l2["missing_clue_diagnosis"] = (span_meta.get("retrieval_rounds") or [{}])[0].get("reason", "")
        _write_json(l2_path, l2)
    report = _build_report(plan, patch, l2)
    trajectory = repair_artifacts_to_trajectory(plan=plan, patch=patch, l2=l2, report=report)
    l2["l2_trajectory"] = trajectory
    l2["repair_subgraph"] = trajectory.get("repair_subgraph") or {}
    report["l2_trajectory"] = trajectory
    report["repair_subgraph"] = trajectory.get("repair_subgraph") or {}
    _write_json(l2_path, l2)
    _write_json(report_path, report)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run targeted graph repair for L1/L2 quality failures.")
    parser.add_argument("--quality-report", type=Path, required=True)
    parser.add_argument("--evidence-audit-report", type=Path)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=["video_holmes", "videomme", "ovo_bench", "cg_bench", "vrbench"])
    parser.add_argument("--example-ids", nargs="+", help="Optional exact example_ids to repair.")
    parser.add_argument(
        "--video-regimes",
        nargs="+",
        default=["short", "streaming", "long"],
        help="Repair regimes to include from the quality report.",
    )
    parser.add_argument("--keys-py", type=Path)
    parser.add_argument("--clip-schema-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--verifier-model", default="openai/gpt-oss-120b")
    parser.add_argument("--clue-planner-model", default="openai/gpt-oss-120b")
    parser.add_argument("--bridge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--clue-planner-max-tokens", type=int, default=1200)
    parser.add_argument("--clue-selector-max-tokens", type=int, default=1600)
    parser.add_argument("--evidence-selector-max-tokens", type=int, default=1600)
    parser.add_argument("--max-evidence-candidate-nodes", type=int, default=120)
    parser.add_argument("--evidence-candidate-text-chars", type=int, default=220)
    parser.add_argument("--bridge-max-tokens", type=int, default=1600)
    parser.add_argument("--clue-planner-timeout-s", type=int, default=180)
    parser.add_argument("--coarse-summary-prompt-chars", type=int, default=260)
    parser.add_argument("--disable-llm-clue-planner", action="store_true")
    parser.add_argument("--disable-llm-reroute-selector", action="store_true")
    parser.add_argument("--disable-exploratory-selector-retry", action="store_true")
    parser.add_argument(
        "--allow-lexical-fallback",
        action="store_true",
        help="Allow heuristic lexical fallback when the LLM clue planner/selector fails. Off by default for API runs.",
    )
    parser.add_argument("--request-frames", type=int, default=4)
    parser.add_argument("--repair-clip-schema-workers", type=int, default=1)
    parser.add_argument("--clip-schema-max-tokens", type=int, default=1200)
    parser.add_argument("--clip-schema-timeout-s", type=int, default=180)
    parser.add_argument("--clip-schema-reasoning-effort", default="none")
    parser.add_argument("--verifier-max-tokens", type=int, default=512)
    parser.add_argument("--verifier-timeout-s", type=int, default=120)
    parser.add_argument(
        "--repair-mode",
        default="local",
        choices=["local", "reroute", "auto"],
        help=(
            "local expands around prior coarse hits; reroute re-ranks the full coarse "
            "index with multiple query roles; auto reroutes when cached repair schemas "
            "contain enough negative target evidence."
        ),
    )
    parser.add_argument("--coarse-radius", type=int, default=1)
    parser.add_argument("--max-repair-clips", type=int, default=12)
    parser.add_argument("--reroute-topk", type=int, default=6)
    parser.add_argument("--reroute-topk-per-query", type=int, default=3)
    parser.add_argument("--negative-reroute-threshold", type=int, default=2)
    parser.add_argument("--max-verify-refs", type=int, default=8)
    parser.add_argument("--min-verify-refs", type=int, default=2)
    parser.add_argument("--min-verify-confidence", type=float, default=0.55)
    parser.add_argument("--min-option-margin", type=float, default=0.08)
    parser.add_argument("--disable-llm-semantic-patch", action="store_true")
    parser.add_argument("--semantic-patch-candidate-nodes", type=int, default=48)
    parser.add_argument("--semantic-patch-max-nodes", type=int, default=4)
    parser.add_argument("--semantic-patch-min-refs", type=int, default=1)
    parser.add_argument("--semantic-patch-max-tokens", type=int, default=1000)
    parser.add_argument("--max-bridge-refs", type=int, default=10)
    parser.add_argument("--min-bridge-refs", type=int, default=1)
    parser.add_argument("--min-bridge-confidence", type=float, default=0.55)
    parser.add_argument("--skip-gptoss-verifier", action="store_true")
    parser.add_argument("--disable-background-bridge", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force-repair-stages",
        action="store_true",
        help="Ignore cached repair plan/patch/L2 stages and recompute them.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = _read_json(args.quality_report)
    reports = payload.get("reports") if isinstance(payload, dict) else payload
    wanted = set(args.datasets)
    wanted_regimes = {str(item) for item in args.video_regimes}
    wanted_examples = {str(item) for item in args.example_ids or []}
    audit_map = _load_evidence_audit_map(args.evidence_audit_report)
    rows = [
        row
        for row in reports
        if row.get("dataset") in wanted
        and row.get("repair_needed")
        and (not wanted_regimes or str(row.get("video_regime")) in wanted_regimes)
        and (not wanted_examples or str(row.get("example_id")) in wanted_examples)
    ]

    api_key = None
    if not args.dry_run and not args.skip_api:
        api_key = load_openrouter_api_key(keys_py_path=str(args.keys_py) if args.keys_py else None)

    out_reports = [
        _process_row(row, args, api_key, evidence_audit=audit_map.get(str(row.get("example_id"))))
        for row in rows
    ]
    summary = {
        "examples": len(out_reports),
        "datasets": sorted({str(row.get("dataset")) for row in out_reports}),
        "repair_status_counts": {},
        "repair_needed_after_round": sum(1 for row in out_reports if row.get("repair_needed_after_round")),
    }
    for row in out_reports:
        status = str(row.get("repair_status"))
        summary["repair_status_counts"][status] = summary["repair_status_counts"].get(status, 0) + 1
    final = {"summary": summary, "reports": out_reports}
    _write_json(args.output, final)
    print(json.dumps(final, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
