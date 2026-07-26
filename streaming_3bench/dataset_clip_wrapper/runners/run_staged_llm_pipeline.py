#!/usr/bin/env python3
"""Staged/resumable runner for Qwen L1 and GPT-OSS L2 graph creation."""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..adapters import get_adapter
from ..cluster_paths import DEFAULT_DATASET_ROOT, DEFAULT_KEYS_PY, output_path
from ..perception.clip_schema import QwenClipSchemaProducer
from ..l1_clue_graph.clue_memory import extract_clue_memory_graph
from ..l1_clue_graph.graph_composer import GraphComposer
from .llm_pipeline import (
    _answerability_diagnostic_graph,
    _build_coarse_fine_reference_graph,
    _build_skill_executor,
    _coarse_schema_segments,
    _coarse_fine_context_for_evidence_index,
    _derived_clips_for_spans,
    _parse_time_anchors_s,
    _question_retrieval_query,
    _resolve_perception_spans,
    _subtitle_context_for_clip,
)
from ..perception.clip_policy import segment_coarse_index
from ..dataset_graph_presets import apply_profile_defaults, clip_policy_for, regime_for_dataset, retrieval_for
from ..perception.openrouter_client import OpenRouterClient, load_openrouter_api_key
from ..pipeline import build_canonical_example
from ..l2_reasoning_graph.reasoning_rollout import build_reasoning_rollout
from ..schemas import (
    BenchmarkProfile,
    BackboneConfig,
    ClipPolicyConfig,
    ClipRetrievalConfig,
    ClipSchemaConfig,
    GraphComposerConfig,
    RuntimeMode,
    SkillExecutionConfig,
    VideoRegime,
    WrapperConfig,
)
from ..perception.video_tool_backend import VideoToolConfig, VideoToolPerceptionBackend


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:180]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a staged/resumable Qwen + GPT-OSS graph pipeline.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["video_holmes", "cg_bench", "vrbench", "siv_bench", "ovo_bench", "videomme"],
    )
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--regime", default=None, choices=["short", "long", "streaming"])
    parser.add_argument(
        "--benchmark-profile",
        default="default",
        choices=["default", "short_multi_hop", "long_coarse_fine"],
        help="Benchmark profile override; long_coarse_fine uses full coarse coverage + retrieved fine graph for CG/VR.",
    )
    parser.add_argument("--mode", default="video_only", choices=["expert_demo", "video_only"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default=str(output_path("staged_llm_pipeline.jsonl")))
    parser.add_argument("--stage-dir", default=str(output_path("staged")))
    parser.add_argument("--keys-py", default=DEFAULT_KEYS_PY)
    parser.add_argument("--force", action="store_true", help="Ignore cached stage files and rebuild.")
    parser.add_argument(
        "--rebuild-from-stages",
        action="store_true",
        help="Ignore cached final_example.json but reuse cached clip schemas without retrying model_error rows.",
    )
    parser.add_argument(
        "--no-fill-missing-clip-schemas",
        action="store_true",
        help="When rebuilding from stages, do not call the clip-schema backend for missing clip ids.",
    )
    parser.add_argument(
        "--retry-failed-clip-schemas",
        action="store_true",
        help="Reuse good cached clip schemas but discard cached model_error rows and retry those clips.",
    )
    parser.add_argument(
        "--retry-non-backbone-clip-schemas",
        action="store_true",
        help="Discard cached clip schemas not produced by the selected clip-schema backend and retry them.",
    )

    parser.add_argument("--clip-schema-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--clip-schema-backend", default="qwen", choices=["qwen", "video_tools"])
    parser.add_argument("--clip-schema-max-clips", type=int, default=999)
    parser.add_argument("--clip-schema-frames", type=int, default=1)
    parser.add_argument("--clip-schema-max-tokens", type=int, default=700)
    parser.add_argument("--clip-schema-timeout-s", type=int, default=45)
    parser.add_argument("--clip-schema-workers", type=int, default=1, help="Parallel workers for missing clip-schema API calls.")
    parser.add_argument(
        "--no-coarse-summary-index",
        action="store_true",
        help="Disable full coarse Qwen summary indexing before long-video retrieval.",
    )
    parser.add_argument("--anchor-repass-frames", type=int, default=6)
    parser.add_argument("--anchor-repass-window-s", type=float, default=8.0)
    parser.add_argument("--no-anchor-repass", action="store_true")

    parser.add_argument("--graph-model", default="openai/gpt-oss-120b")
    parser.add_argument("--graph-max-tokens", type=int, default=3500)
    parser.add_argument("--graph-timeout-s", type=int, default=180)
    parser.add_argument("--graph-neighbor-workers", type=int, default=1)
    parser.add_argument(
        "--graph-composer-mode",
        default="neighbor_vlm_l1",
        choices=["neighbor_vlm_l1", "vlm_l1", "skill_plan", "deterministic"],
    )
    parser.add_argument("--graph-deterministic", action="store_true")
    parser.add_argument("--skip-l2-planner", action="store_true")
    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--llm-skill-scope", default="all", choices=["all", "verifier"])
    parser.add_argument("--disable-llm-skills", action="store_true")
    parser.add_argument("--disable-vlm-skills", action="store_true")

    parser.add_argument("--retrieval-topk", type=int, default=2)
    parser.add_argument("--retrieval-mode", default="lexical", choices=["lexical", "sequential"])
    parser.add_argument("--query-time-retrieval", action="store_true")
    parser.add_argument("--no-time-anchor-expansion", action="store_true")
    parser.add_argument("--index-fine-expansion", default=None, choices=["none", "all", "retrieval_gated"])
    return parser


def _config_from_args(args: argparse.Namespace) -> WrapperConfig:
    benchmark_profile = BenchmarkProfile(args.benchmark_profile)
    regime = VideoRegime(args.regime) if args.regime else None
    dataset_regime = regime or regime_for_dataset(args.dataset, benchmark_profile)

    clip_policy = clip_policy_for(args.dataset, dataset_regime)
    retrieval = retrieval_for(dataset_regime)
    apply_profile_defaults(
        dataset=args.dataset,
        regime=dataset_regime,
        profile=benchmark_profile,
        clip_policy=clip_policy,
        retrieval=retrieval,
    )
    if args.index_fine_expansion:
        clip_policy.index_fine_expansion = args.index_fine_expansion  # type: ignore[assignment]

    return WrapperConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        regime=dataset_regime,
        benchmark_profile=benchmark_profile,
        mode=RuntimeMode(args.mode),
        clip_policy=clip_policy,
        retrieval=ClipRetrievalConfig(
            enabled=retrieval.enabled,
            topk=max(args.retrieval_topk, retrieval.topk),
            threshold=retrieval.threshold,
            mode=args.retrieval_mode or retrieval.mode,  # type: ignore[arg-type]
            query_in_video_only=args.query_time_retrieval or retrieval.query_in_video_only,
            expand_time_anchors=retrieval.expand_time_anchors and not args.no_time_anchor_expansion,
        ),
        backbone=BackboneConfig(keys_py_path=args.keys_py),
        clip_schema=ClipSchemaConfig(
            backend=args.clip_schema_backend,
            model=args.clip_schema_model,
            keys_py_path=args.keys_py,
            max_clips=args.clip_schema_max_clips,
            request_frames=args.clip_schema_frames,
            max_tokens=args.clip_schema_max_tokens,
            timeout_s=args.clip_schema_timeout_s,
        ),
        graph_composer=GraphComposerConfig(
            model=args.graph_model,
            keys_py_path=args.keys_py,
            use_llm_planner=not args.graph_deterministic,
            composer_mode="deterministic" if args.graph_deterministic else args.graph_composer_mode,
            max_tokens=args.graph_max_tokens,
            timeout_s=args.graph_timeout_s,
            neighbor_workers=args.graph_neighbor_workers,
        ),
        skill_execution=SkillExecutionConfig(
            skill_model=args.skill_model,
            enable_llm_skills=not args.disable_llm_skills,
            enable_vlm_skills=not args.disable_vlm_skills,
            llm_skill_scope=args.llm_skill_scope,
        ),
        split=args.split,
        limit=args.limit,
        run_clip_schema=True,
        run_graph_compose=True,
        run_l2_llm_planner=not args.skip_l2_planner,
    )


def _clip_schema_producer(config: WrapperConfig) -> Any:
    if config.clip_schema.backend == "video_tools":
        return VideoToolPerceptionBackend(VideoToolConfig(request_frames=config.clip_schema.request_frames))
    keys_py = config.clip_schema.keys_py_path or config.backbone.keys_py_path
    api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.clip_schema.api_key_env)
    client = OpenRouterClient(
        model=config.clip_schema.model,
        api_key=api_key,
        api_base=config.clip_schema.api_base,
        temperature=config.clip_schema.temperature,
        max_tokens=config.clip_schema.max_tokens,
        reasoning={"effort": config.clip_schema.reasoning_effort, "exclude": True}
        if config.clip_schema.reasoning_effort
        else None,
        timeout_s=config.clip_schema.timeout_s,
    )
    return QwenClipSchemaProducer(config.clip_schema, client)


def _produce_or_resume_clip_schemas(
    *,
    item: Any,
    config: WrapperConfig,
    spans: list[Any],
    derived_clips: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
    stage_path: Path,
    force: bool,
    retry_failed: bool,
    retry_non_backbone: bool,
    fill_missing: bool,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if force and stage_path.exists():
        stage_path.unlink()

    cached = _read_jsonl(stage_path)
    if retry_failed and cached:
        cached = [row for row in cached if not row.get("model_error")]
        _write_jsonl(stage_path, cached)
    if retry_non_backbone and cached:
        expected_producer = "qwen_clip_schema" if config.clip_schema.backend == "qwen" else "video_tool_perception_backend"
        cached = [row for row in cached if row.get("producer") == expected_producer]
        _write_jsonl(stage_path, cached)
    by_clip_id = {row.get("clip_id"): row for row in cached if row.get("clip_id")}
    budget = config.clip_schema.max_clips
    targets: list[tuple[Any, dict[str, Any]]] = []

    for index, (span, derived) in enumerate(zip(spans, derived_clips)):
        if budget is not None and index >= budget:
            break
        clip_id = derived["clip_id"]
        if clip_id in by_clip_id:
            continue
        if not fill_missing:
            continue
        targets.append((span, derived))

    def _build_one(span: Any, derived: dict[str, Any]) -> dict[str, Any]:
        producer = _clip_schema_producer(config)
        return producer.build_clip_schema(
            clip_id=derived["clip_id"],
            clip=span,
            video_path=item.video_path,
            subtitle_context=_subtitle_context_for_clip(visible_segments, span.to_dict()),
            question_context=item.question.get("question_text") if config.mode == RuntimeMode.EXPERT_DEMO else None,
        )

    def _checkpoint() -> None:
        ordered_rows = [
            by_clip_id[derived["clip_id"]]
            for index, derived in enumerate(derived_clips)
            if (budget is None or index < budget) and derived["clip_id"] in by_clip_id
        ]
        _write_jsonl(stage_path, ordered_rows)

    if targets and max(1, workers) == 1:
        producer = _clip_schema_producer(config)
        for span, derived in targets:
            schema = producer.build_clip_schema(
                clip_id=derived["clip_id"],
                clip=span,
                video_path=item.video_path,
                subtitle_context=_subtitle_context_for_clip(visible_segments, span.to_dict()),
                question_context=item.question.get("question_text") if config.mode == RuntimeMode.EXPERT_DEMO else None,
            )
            by_clip_id[derived["clip_id"]] = schema
            _checkpoint()
    elif targets:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_build_one, span, derived): derived["clip_id"] for span, derived in targets}
            for future in as_completed(futures):
                clip_id = futures[future]
                try:
                    by_clip_id[clip_id] = future.result()
                except Exception as exc:
                    by_clip_id[clip_id] = {
                        "clip_id": clip_id,
                        "model_error": str(exc),
                        "schema_attempt": "parallel_worker_error",
                        "producer": config.clip_schema.backend,
                    }
                _checkpoint()

    return [by_clip_id[derived["clip_id"]] for derived in derived_clips if derived["clip_id"] in by_clip_id]


def _coarse_summary_index_enabled(args: argparse.Namespace, config: WrapperConfig) -> bool:
    return (
        not args.no_coarse_summary_index
        and config.clip_policy.strategy == "hierarchical"
        and config.clip_policy.index_fine_expansion == "retrieval_gated"
        and config.retrieval.enabled
        and config.retrieval.query_in_video_only
    )


def _produce_or_resume_coarse_summaries(
    *,
    item: Any,
    config: WrapperConfig,
    coarse_spans: list[Any],
    visible_segments: list[dict[str, Any]],
    primary_path: str,
    stage_path: Path,
    force: bool,
    retry_failed: bool,
    retry_non_backbone: bool,
    fill_missing: bool,
    workers: int = 1,
) -> list[dict[str, Any]]:
    derived = _derived_clips_for_spans(video_id=item.video_id, primary_path=primary_path, spans=coarse_spans)
    return _produce_or_resume_clip_schemas(
        item=item,
        config=config,
        spans=coarse_spans,
        derived_clips=derived,
        visible_segments=visible_segments,
        stage_path=stage_path,
        force=force,
        retry_failed=retry_failed,
        retry_non_backbone=retry_non_backbone,
        fill_missing=fill_missing,
        workers=workers,
    )


def _anchor_repass_spans(
    *,
    spans: list[Any],
    derived_clips: list[dict[str, Any]],
    question_text: str,
    window_s: float,
) -> tuple[list[Any], list[dict[str, Any]], list[float]]:
    anchors = _parse_time_anchors_s(question_text)
    if not anchors:
        return [], [], []
    selected_spans: list[Any] = []
    selected_derived: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span, derived in zip(spans, derived_clips):
        start_s = float(span.start_s)
        end_s = float(span.end_s)
        if any(start_s - window_s <= anchor <= end_s + window_s for anchor in anchors):
            clip_id = derived.get("clip_id")
            if clip_id in seen:
                continue
            seen.add(clip_id)
            selected_spans.append(span)
            selected_derived.append(derived)
    return selected_spans, selected_derived, anchors


def _produce_or_resume_anchor_repass(
    *,
    item: Any,
    config: WrapperConfig,
    spans: list[Any],
    derived_clips: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
    stage_path: Path,
    force: bool,
    retry_failed: bool,
    fill_missing: bool,
    request_frames: int,
) -> list[dict[str, Any]]:
    if not spans:
        return []
    if force and stage_path.exists():
        stage_path.unlink()

    cached = _read_jsonl(stage_path)
    if retry_failed and cached:
        cached = [row for row in cached if not row.get("model_error")]
        _write_jsonl(stage_path, cached)
    by_clip_id = {row.get("clip_id"): row for row in cached if row.get("clip_id")}

    repass_config = replace(
        config.clip_schema,
        request_frames=max(config.clip_schema.request_frames, request_frames),
        max_tokens=max(config.clip_schema.max_tokens or 0, 1000),
    )
    repass_wrapper = replace(config, clip_schema=repass_config)
    producer = _clip_schema_producer(repass_wrapper)

    question_text = item.question.get("question_text") or ""
    for span, derived in zip(spans, derived_clips):
        clip_id = derived["clip_id"]
        if clip_id in by_clip_id:
            continue
        if not fill_missing:
            continue
        schema = producer.build_clip_schema(
            clip_id=clip_id,
            clip=span,
            video_path=item.video_path,
            subtitle_context=_subtitle_context_for_clip(visible_segments, span.to_dict()),
            question_context=question_text,
        )
        schema["schema_attempt_context"] = "query_time_anchor_repass"
        schema["request_frames"] = repass_config.request_frames
        _append_jsonl(stage_path, schema)
        by_clip_id[clip_id] = schema

    return [by_clip_id[derived["clip_id"]] for derived in derived_clips if derived["clip_id"] in by_clip_id]


def _compose_l1_and_l2(
    *,
    example: dict[str, Any],
    item: Any,
    config: WrapperConfig,
    clip_policy: ClipPolicyConfig,
    clip_schemas: list[dict[str, Any]],
    visible_segments: list[dict[str, Any]],
    duration_s: float,
    neighbor_cache_path: Path | None = None,
) -> dict[str, Any]:
    api_key: str | None = None
    if config.graph_composer.use_llm_planner or config.run_l2_llm_planner:
        keys_py = config.graph_composer.keys_py_path or config.backbone.keys_py_path
        api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.graph_composer.api_key_env)
        client = OpenRouterClient(
            model=config.graph_composer.model,
            api_key=api_key,
            api_base=config.graph_composer.api_base,
            temperature=config.graph_composer.temperature,
            max_tokens=config.graph_composer.max_tokens,
            reasoning={"effort": config.graph_composer.reasoning_effort, "exclude": True},
            timeout_s=config.graph_composer.timeout_s,
        )
    else:
        client = OpenRouterClient(model="offline", api_key="offline")

    graph_config = (
        replace(config.graph_composer, neighbor_cache_path=str(neighbor_cache_path))
        if neighbor_cache_path and config.graph_composer.composer_mode == "neighbor_vlm_l1"
        else config.graph_composer
    )
    composer = GraphComposer(graph_config, client)
    composed = composer.compose_from_clip_schemas(
        example_id=example["example_id"],
        video_id=item.video_id,
        clip_policy=clip_policy.to_dict(),
        clip_schemas=clip_schemas,
        segments=visible_segments,
        mode=config.mode,
        duration_s=duration_s,
        observation_end_s=clip_policy.observation_end_s,
    )
    graph = composed["graph"]
    context_nodes, context_edges = _coarse_fine_context_for_evidence_index(
        example["metadata"].get("coarse_fine_graph") or {}
    )
    diagnostic = _answerability_diagnostic_graph(
        example=example,
        graph_nodes=graph.get("nodes", []) + context_nodes,
        clip_schemas=clip_schemas,
        visible_segments=visible_segments,
    )
    diagnostic_nodes = diagnostic.get("nodes") or []
    diagnostic_edges = diagnostic.get("edges") or []
    node_by_id = {
        node.get("node_id"): node
        for node in graph.get("nodes", []) + context_nodes + diagnostic_nodes
        if node.get("node_id")
    }
    graph_edges = graph.get("edges", []) + context_edges + diagnostic_edges
    valid_ids = set(node_by_id)
    edge_by_id = {
        edge.get("edge_id"): edge
        for edge in graph_edges
        if edge.get("edge_id") and edge.get("src") in valid_ids and edge.get("dst") in valid_ids
    }
    example["evidence_index"]["nodes"] = list(node_by_id.values())
    example["evidence_index"]["edges"] = list(edge_by_id.values())
    example["evidence_index"]["graph_composer"] = config.graph_composer.to_dict()
    example["evidence_index"]["retrieval"] = config.retrieval.to_dict()
    example["metadata"]["graph_compose"] = {
        "composer_model": composed.get("composer_model"),
        "composer_mode": composed.get("composer_mode"),
        "used_deterministic_fallback": composed.get("used_deterministic_fallback"),
        "execution_trace": composed.get("execution_trace"),
        "skill_plan": composed.get("skill_plan"),
    }
    example["metadata"]["answerability_diagnostic"] = diagnostic.get("summary") or {}

    for node in graph.get("nodes", []):
        if node.get("node_type") == "observation" and node.get("text"):
            example["evidence_candidates"].append(
                {
                    "evidence_id": f"ev:{node['node_id']}",
                    "source_type": "caption_span",
                    "time_span": node.get("time_span"),
                    "text": node.get("text"),
                    "trust_level": "model_labeled",
                    "provenance": {"created_by": "dataset_clip_wrapper.staged_graph_composer"},
                    "discovery_status": "discovered_runtime",
                }
            )

    clue_graph = extract_clue_memory_graph(example, mode=config.mode)
    example["metadata"]["clue_memory_graph"] = clue_graph

    if config.run_l2_llm_planner:
        from ..l2_reasoning_graph.reasoning_planner import build_llm_reasoning_rollout

        if api_key is None:
            keys_py = config.graph_composer.keys_py_path or config.backbone.keys_py_path
            api_key = load_openrouter_api_key(keys_py_path=keys_py, env_var=config.graph_composer.api_key_env)
        l2_client = OpenRouterClient(
            model=config.graph_composer.model,
            api_key=api_key,
            max_tokens=1800,
            reasoning={"effort": "minimal", "exclude": True},
            timeout_s=config.graph_composer.timeout_s,
        )
        skill_exec = _build_skill_executor(api_key, config)
        rollout = build_llm_reasoning_rollout(example, clue_graph, client=l2_client, skill_executor=skill_exec)
    else:
        rollout = build_reasoning_rollout(example, clue_graph, rollout_source="staged_llm_pipeline")
    example["metadata"]["reasoning_rollout"] = rollout
    example["metadata"]["reasoning_rollout_shell"] = rollout
    return example


def _run_item(
    item: Any,
    *,
    config: WrapperConfig,
    root_stage_dir: Path,
    force: bool,
    rebuild_from_stages: bool,
    retry_failed_clip_schemas: bool,
    retry_non_backbone_clip_schemas: bool,
    fill_missing_clip_schemas: bool,
    build_coarse_summary_index: bool,
    clip_schema_workers: int,
    anchor_repass_frames: int,
    anchor_repass_window_s: float,
    anchor_repass_enabled: bool,
) -> dict[str, Any]:
    example_stage_dir = root_stage_dir / _safe_name(item.example_id)
    example_stage_dir.mkdir(parents=True, exist_ok=True)

    final_path = example_stage_dir / "final_example.json"
    if final_path.exists() and not force and not rebuild_from_stages and not retry_failed_clip_schemas:
        return _read_json(final_path)

    example = build_canonical_example(item, config=config, backbone=None)
    duration_s = float(example["video"].get("duration_s") or 0.0)
    clip_policy = config.resolved_clip_policy(duration_s)
    visible_segments = example["video"]["segments"]
    primary_path = str(item.video_path) if item.video_path else ""
    coarse_schemas: list[dict[str, Any]] = []
    retrieval_segments = visible_segments
    if build_coarse_summary_index:
        coarse_spans = segment_coarse_index(duration_s, clip_policy, regime=config.regime)
        coarse_schemas = _produce_or_resume_coarse_summaries(
            item=item,
            config=config,
            coarse_spans=coarse_spans,
            visible_segments=visible_segments,
            primary_path=primary_path,
            stage_path=example_stage_dir / "00b_coarse_clip_schemas.jsonl",
            force=force,
            retry_failed=retry_failed_clip_schemas,
            retry_non_backbone=retry_non_backbone_clip_schemas,
            fill_missing=fill_missing_clip_schemas,
            workers=clip_schema_workers,
        )
        retrieval_segments = visible_segments + _coarse_schema_segments(coarse_schemas)
        example["metadata"]["coarse_clip_schemas"] = coarse_schemas
        example["metadata"]["coarse_summary_index"] = {
            "enabled": True,
            "clip_schema_count": len(coarse_schemas),
            "stage_path": str(example_stage_dir / "00b_coarse_clip_schemas.jsonl"),
        }

    perception_spans, perception_meta = _resolve_perception_spans(
        duration_s=duration_s,
        clip_policy=clip_policy,
        regime=config.regime,
        retrieval_config=config.retrieval,
        question_text=_question_retrieval_query(item.question),
        visible_segments=retrieval_segments,
        mode=config.mode,
    )
    derived = _derived_clips_for_spans(video_id=item.video_id, primary_path=primary_path, spans=perception_spans)
    example["metadata"]["perception"] = perception_meta
    _write_json(example_stage_dir / "00_shell.json", example)
    _write_json(
        example_stage_dir / "01_perception_spans.json",
        {"perception": perception_meta, "derived_clips": derived},
    )

    clip_schemas = _produce_or_resume_clip_schemas(
        item=item,
        config=config,
        spans=perception_spans,
        derived_clips=derived,
        visible_segments=visible_segments,
        stage_path=example_stage_dir / "02_clip_schemas.jsonl",
        force=force,
        retry_failed=retry_failed_clip_schemas,
        retry_non_backbone=retry_non_backbone_clip_schemas,
        fill_missing=fill_missing_clip_schemas,
        workers=clip_schema_workers,
    )
    anchor_schemas: list[dict[str, Any]] = []
    if anchor_repass_enabled and config.mode == RuntimeMode.VIDEO_ONLY and anchor_repass_frames > config.clip_schema.request_frames:
        anchor_spans, anchor_derived, anchors_s = _anchor_repass_spans(
            spans=perception_spans,
            derived_clips=derived,
            question_text=_question_retrieval_query(item.question),
            window_s=anchor_repass_window_s,
        )
        anchor_schemas = _produce_or_resume_anchor_repass(
            item=item,
            config=config,
            spans=anchor_spans,
            derived_clips=anchor_derived,
            visible_segments=visible_segments,
            stage_path=example_stage_dir / "02b_anchor_clip_schemas.jsonl",
            force=force,
            retry_failed=retry_failed_clip_schemas,
            fill_missing=fill_missing_clip_schemas,
            request_frames=anchor_repass_frames,
        )
        if anchor_schemas:
            by_clip_id = {schema.get("clip_id"): schema for schema in clip_schemas if schema.get("clip_id")}
            for schema in anchor_schemas:
                if schema.get("clip_id"):
                    by_clip_id[schema["clip_id"]] = schema
            clip_schemas = [by_clip_id[clip["clip_id"]] for clip in derived if clip["clip_id"] in by_clip_id]
        example["metadata"]["anchor_repass"] = {
            "enabled": bool(anchor_spans),
            "anchors_s": anchors_s,
            "window_s": anchor_repass_window_s,
            "request_frames": anchor_repass_frames,
            "clip_schema_count": len(anchor_schemas),
            "stage_path": str(example_stage_dir / "02b_anchor_clip_schemas.jsonl"),
        }
    example["metadata"]["clip_schemas"] = clip_schemas
    example["metadata"]["clip_schema_model"] = (
        config.clip_schema.model if config.clip_schema.backend == "qwen" else "local-video-tools"
    )
    example["metadata"]["clip_schema_backend"] = config.clip_schema.backend
    example["metadata"]["coarse_fine_graph"] = _build_coarse_fine_reference_graph(
        video_id=item.video_id,
        primary_path=primary_path,
        duration_s=duration_s,
        clip_policy=clip_policy,
        regime=config.regime,
        perception_spans=perception_spans,
        perception_meta=perception_meta,
        clip_schemas=clip_schemas,
        coarse_schemas=coarse_schemas,
    )
    _write_json(example_stage_dir / "03_l1_inputs.json", example)

    example = _compose_l1_and_l2(
        example=example,
        item=item,
        config=config,
        clip_policy=clip_policy,
        clip_schemas=clip_schemas,
        visible_segments=visible_segments,
        duration_s=duration_s,
        neighbor_cache_path=example_stage_dir / "03_neighbor_vlm_l1_clip_results.jsonl",
    )
    _write_json(example_stage_dir / "04_l1_example.json", {**example, "metadata": {**example["metadata"], "reasoning_rollout": None}})
    _write_json(example_stage_dir / "05_l2_rollout.json", example["metadata"].get("reasoning_rollout"))
    example["metadata"]["llm_pipeline"] = {
        "clip_schema": config.clip_schema.to_dict(),
        "graph_composer": config.graph_composer.to_dict(),
        "retrieval": config.retrieval.to_dict(),
        "staged_dir": str(example_stage_dir),
    }
    _write_json(final_path, example)
    return example


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = _config_from_args(args)
    adapter = get_adapter(config.dataset, Path(config.dataset_root), split=config.split)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.force and output_path.exists():
        output_path.unlink()

    written = 0
    for item in adapter.iter_items(limit=config.limit):
        example = _run_item(
            item,
            config=config,
            root_stage_dir=Path(args.stage_dir),
            force=args.force,
            rebuild_from_stages=args.rebuild_from_stages,
            retry_failed_clip_schemas=args.retry_failed_clip_schemas,
            retry_non_backbone_clip_schemas=args.retry_non_backbone_clip_schemas,
            fill_missing_clip_schemas=not args.no_fill_missing_clip_schemas,
            build_coarse_summary_index=_coarse_summary_index_enabled(args, config),
            clip_schema_workers=args.clip_schema_workers,
            anchor_repass_frames=args.anchor_repass_frames,
            anchor_repass_window_s=args.anchor_repass_window_s,
            anchor_repass_enabled=not args.no_anchor_repass,
        )
        _append_jsonl(output_path, example)
        written += 1
        print(
            json.dumps(
                {
                    "example_id": example["example_id"],
                    "dataset": example["dataset"],
                    "stage_dir": example["metadata"]["llm_pipeline"]["staged_dir"],
                    "clip_schema_count": len(example["metadata"].get("clip_schemas") or []),
                    "output": str(output_path),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    print(json.dumps({"written": written, "output": str(output_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
