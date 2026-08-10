#!/usr/bin/env python3
"""Matched three-arm QA evaluation for the frozen Q-Former retriever."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .iterative_rag_memory_query import (
    _normalize_open_answer,
    _question_answer_label,
    _question_answer_text,
    _question_options,
    iter_examples,
    metric_summary,
    parse_answer_label,
    parse_evidence_summary,
)
from .per_video_embedding_rag import LocalVideoQwen, media_records_from_retrieved
from .qformer_retrieval import ThreeBenchRetriever


FULL_DATASET_COUNTS = {"ovo_bench": 3035, "videomme": 2700, "streaming_bench": 4500}


def _parse_answer_text(response: str) -> str | None:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return response.strip() or None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return response.strip() or None
    value = payload.get("answer_text") or payload.get("answer")
    return str(value).strip() if value is not None else None


def build_answer_prompt(
    example: dict[str, Any], media_records: list[dict[str, Any]], cutoff: float | None
) -> str:
    question = example.get("question") or {}
    options = _question_options(example)
    lines = [
        "Use only the provided visible video clips to answer the streaming-video question.",
        "Never use information after the visible cutoff.",
    ]
    if cutoff is not None:
        lines.append(f"Visible cutoff: {cutoff:.2f} seconds.")
    for record in media_records:
        lines.append(
            f"Clip {record['memory_id']}: {record['video_start']:.2f}-"
            f"{record['video_end']:.2f}s."
        )
    lines.append(f"Question: {question.get('question_text') or ''}")
    if options:
        lines.append("Options:")
        lines.extend(f"{item['label']}. {item['text']}" for item in options)
        lines.append(
            'Output JSON only: {"answer_label":"one option label",'
            '"evidence_summary":"one short grounded sentence"}.'
        )
    else:
        lines.append(
            'Output JSON only: {"answer_text":"concise answer",'
            '"evidence_summary":"one short grounded sentence"}.'
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--steam-root", type=Path, required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--projected-manifest", type=Path, required=True)
    parser.add_argument("--clip-sidecar", type=Path, required=True)
    parser.add_argument("--qformer-checkpoint", type=Path, required=True)
    parser.add_argument("--embedding-model", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--arm", choices=("uniform", "visual", "qformer"), required=True)
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--limit-per-dataset", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--videomme-observation-end-s", type=float, default=None)
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-fps", type=float, default=1.0)
    parser.add_argument("--video-max-frames-per-clip", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Return non-zero when any example fails; intended for smoke gates.",
    )
    args = parser.parse_args(argv)
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    from .schemas import visible_until_from_canonical

    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples = iter_examples(args)
    if args.limit_per_dataset is None and len(args.datasets) == 1:
        total = FULL_DATASET_COUNTS[args.datasets[0]]
        expected = len(range(args.shard_index, total, args.num_shards))
        if len(examples) != expected:
            raise ValueError(
                f"canonical shard cardinality mismatch: got {len(examples)}, expected {expected}"
            )
    retriever = ThreeBenchRetriever(
        steam_root=args.steam_root,
        feature_manifest=args.feature_manifest,
        projected_manifest=args.projected_manifest,
        clip_sidecar=args.clip_sidecar,
        checkpoint=args.qformer_checkpoint,
        embedding_model=args.embedding_model,
        device=args.device,
    )
    answerer = LocalVideoQwen(
        args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        enable_thinking=False,
    )
    config = {
        "schema_version": "streaming-3bench-qformer-rag-eval/v0.1",
        "arm": args.arm,
        "datasets": args.datasets,
        "top_k": args.top_k,
        "query_contract": "question-text-only",
        "candidate_contract": "same-question-independent-captioned-clips",
        "future_cutoff": "start-before-cutoff-and-end-clipped-to-cutoff",
        "feature_manifest": str(args.feature_manifest.resolve()),
        "projected_manifest": str(args.projected_manifest.resolve()),
        "qformer_checkpoint": str(args.qformer_checkpoint.resolve()),
        "reasoner": args.model,
        "deterministic_decoding": True,
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "example_count": len(examples),
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    records: list[dict[str, Any]] = []
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as output:
        for dataset, example in examples:
            started = time.perf_counter()
            question = example.get("question") or {}
            gold_label = _question_answer_label(example)
            gold_text = _question_answer_text(example)
            task_type = str(question.get("question_type") or "")
            metadata = example.get("metadata") or {}
            protocol_compatible = metadata.get("official_protocol_compatible")
            if protocol_compatible is None:
                protocol_compatible = not (
                    metadata.get("future_trigger_instruction")
                    or task_type.strip().lower() == "sequential question answering"
                )
            try:
                cutoff = visible_until_from_canonical(
                    example, default_videomme_cutoff_s=args.videomme_observation_end_s
                )
                video = example.get("video") or {}
                retrieved = retriever.retrieve(
                    video_path=str(video.get("primary_path") or ""),
                    question_text=str(question.get("question_text") or ""),
                    visible_until_s=cutoff,
                    arm=args.arm,
                    top_k=args.top_k,
                )
                media = media_records_from_retrieved(
                    retrieved,
                    video_fps=args.video_fps,
                    video_max_frames_per_clip=args.video_max_frames_per_clip,
                )
                response = answerer.generate(
                    media_records=media,
                    prompt_text=build_answer_prompt(example, media, cutoff),
                )
                options = _question_options(example)
                prediction_label = parse_answer_label(response, options) if options else None
                prediction_text = _parse_answer_text(response) if not options else None
                if gold_label:
                    correct = prediction_label == gold_label if prediction_label else False
                elif gold_text:
                    correct = _normalize_open_answer(prediction_text) == _normalize_open_answer(gold_text)
                else:
                    correct = False
                record = {
                    "ok": True,
                    "dataset": dataset,
                    "arm": args.arm,
                    "example_id": example.get("example_id"),
                    "question_id": question.get("question_id"),
                    "task_type": task_type,
                    "official_protocol_compatible": bool(protocol_compatible),
                    "video_id": video.get("video_id"),
                    "visible_until_s": cutoff,
                    "retrieved_memory": retrieved,
                    "media_records": media,
                    "raw_response": response,
                    "evidence_summary": parse_evidence_summary(response),
                    "prediction_label": prediction_label,
                    "prediction_text": prediction_text,
                    "gold_label": gold_label,
                    "gold_text": gold_text,
                    "correct": bool(correct),
                    "timing_s": {"total": time.perf_counter() - started},
                }
            except Exception as exc:
                record = {
                    "ok": False,
                    "dataset": dataset,
                    "arm": args.arm,
                    "example_id": example.get("example_id"),
                    "question_id": question.get("question_id"),
                    "task_type": task_type,
                    "official_protocol_compatible": bool(protocol_compatible),
                    "error": f"{type(exc).__name__}: {exc}",
                    "prediction_label": None,
                    "prediction_text": None,
                    "gold_label": gold_label,
                    "gold_text": gold_text,
                    "correct": False,
                    "timing_s": {"total": time.perf_counter() - started},
                }
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
    metrics = metric_summary(records)
    (args.output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"records": len(records), "metrics": metrics}, indent=2))
    if args.fail_on_error and any(not row.get("ok", False) for row in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
