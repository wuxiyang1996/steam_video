#!/usr/bin/env python3
"""CLI for Qwen clip-schema + gpt-oss graph-composition pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..cluster_paths import DEFAULT_DATASET_ROOT, DEFAULT_KEYS_PY, output_path
from ..dataset_graph_presets import apply_profile_defaults, clip_policy_for, regime_for_dataset, retrieval_for
from .llm_pipeline import iter_llm_enriched_examples
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Qwen clip-schema + gpt-oss graph-composition over dataset clips."
    )
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
    parser.add_argument("--mode", default="expert_demo", choices=["expert_demo", "video_only"])
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--output", default=str(output_path("llm_pipeline.jsonl")))
    parser.add_argument("--keys-py", default=DEFAULT_KEYS_PY)

    parser.add_argument("--clip-schema-model", default="qwen/qwen3.5-9b")
    parser.add_argument("--clip-schema-api-base", default=None)
    parser.add_argument("--clip-schema-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--clip-schema-backend", default="qwen", choices=["qwen", "video_tools"])
    parser.add_argument("--clip-schema-max-clips", type=int, default=3)
    parser.add_argument("--clip-schema-frames", type=int, default=4)
    parser.add_argument("--clip-schema-max-tokens", type=int, default=1200)
    parser.add_argument("--clip-schema-reasoning-effort", default="none")
    parser.add_argument("--clip-schema-timeout-s", type=int, default=180)
    parser.add_argument("--skip-clip-schema", action="store_true")

    parser.add_argument("--graph-model", default="openai/gpt-oss-120b")
    parser.add_argument("--graph-api-base", default=None)
    parser.add_argument("--graph-api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--graph-max-tokens", type=int, default=1800)
    parser.add_argument("--graph-reasoning-effort", default="minimal")
    parser.add_argument("--graph-timeout-s", type=int, default=180)
    parser.add_argument("--graph-neighbor-workers", type=int, default=1)
    parser.add_argument(
        "--graph-composer-mode",
        default="neighbor_vlm_l1",
        choices=["neighbor_vlm_l1", "vlm_l1", "skill_plan", "deterministic"],
    )
    parser.add_argument("--graph-deterministic", action="store_true", help="Skip gpt-oss planner; apply atomic skills directly")
    parser.add_argument("--skip-graph-compose", action="store_true")

    parser.add_argument("--skill-model", default="qwen/qwen3.5-9b", help="Model for skill-level reasoning/perception execution (student model)")
    parser.add_argument("--llm-skill-scope", default="all", choices=["all", "verifier"], help="Limit LLM-backed skill execution to verifier skills")
    parser.add_argument("--run-l2-planner", action="store_true", help="Enable L2 LLM reasoning planner with skill execution")
    parser.add_argument("--disable-llm-skills", action="store_true", help="Disable LLM-backed skill execution (use rule only)")
    parser.add_argument("--disable-vlm-skills", action="store_true", help="Disable VLM-backed perception skills")

    parser.add_argument("--observation-end-s", type=float, default=None)
    parser.add_argument("--retrieval-topk", type=int, default=2)
    parser.add_argument("--retrieval-mode", default="lexical", choices=["lexical", "sequential"])
    parser.add_argument(
        "--query-time-retrieval",
        action="store_true",
        help="In video_only, use the visible question to retrieve coarse neighborhoods for fine perception.",
    )
    parser.add_argument(
        "--no-time-anchor-expansion",
        action="store_true",
        help="Disable automatic fine expansion around timestamps mentioned in the visible question.",
    )
    parser.add_argument("--no-retrieval", action="store_true")
    parser.add_argument(
        "--index-fine-expansion",
        default=None,
        choices=["none", "all", "retrieval_gated"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    if args.observation_end_s is not None:
        clip_policy.observation_end_s = args.observation_end_s
    if args.index_fine_expansion:
        clip_policy.index_fine_expansion = args.index_fine_expansion  # type: ignore[assignment]

    config = WrapperConfig(
        dataset_root=args.dataset_root,
        dataset=args.dataset,
        regime=dataset_regime,
        benchmark_profile=benchmark_profile,
        mode=RuntimeMode(args.mode),
        clip_policy=clip_policy,
        retrieval=ClipRetrievalConfig(
            enabled=retrieval.enabled and not args.no_retrieval,
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
            api_base=args.clip_schema_api_base or ClipSchemaConfig.api_base,
            api_key_env=args.clip_schema_api_key_env,
            keys_py_path=args.keys_py,
            max_clips=args.clip_schema_max_clips,
            request_frames=args.clip_schema_frames,
            max_tokens=args.clip_schema_max_tokens,
            reasoning_effort=args.clip_schema_reasoning_effort,
            timeout_s=args.clip_schema_timeout_s,
        ),
        graph_composer=GraphComposerConfig(
            model=args.graph_model,
            api_base=args.graph_api_base or GraphComposerConfig.api_base,
            api_key_env=args.graph_api_key_env,
            keys_py_path=args.keys_py,
            use_llm_planner=not args.graph_deterministic,
            composer_mode="deterministic" if args.graph_deterministic else args.graph_composer_mode,
            max_tokens=args.graph_max_tokens,
            reasoning_effort=args.graph_reasoning_effort,
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
        run_clip_schema=not args.skip_clip_schema,
        run_graph_compose=not args.skip_graph_compose,
        run_l2_llm_planner=args.run_l2_planner,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for example in iter_llm_enriched_examples(config):
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1

    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "regime": dataset_regime.value,
                "benchmark_profile": benchmark_profile.value,
                "clip_schema_model": config.clip_schema.model
                if config.clip_schema.backend == "qwen"
                else "local-video-tools",
                "clip_schema_backend": config.clip_schema.backend,
                "graph_model": config.graph_composer.model,
                "run_clip_schema": config.run_clip_schema,
                "run_graph_compose": config.run_graph_compose,
                "written": count,
                "output": str(output_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
