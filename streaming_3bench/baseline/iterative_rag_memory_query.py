#!/usr/bin/env python3
"""Iterative RAG baseline over the baseline FAISS clip memory.

This runner evaluates a text-memory baseline:

canonical video QA example -> iterative FAISS memory retrieval -> answer from
retrieved clip-memory text.  It does not decode video frames during answering.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_DATASETS = ("ovo_bench", "videomme")
SUPPORTED_DATASETS = ("ovo_bench", "videomme", "streaming_bench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/iterative_rag_memory_query"
)


def ensure_repo_on_path(repo_root: str) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _json_dump_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _question_with_options(example: dict[str, Any]) -> str:
    question = example.get("question") or {}
    text = str(question.get("question_text") or "").strip()
    options = _question_options(example)
    if not options:
        return text
    option_lines = [f"{option['label']}. {option['text']}" for option in options]
    return f"{text}\nOptions:\n" + "\n".join(option_lines)


def _question_answer_label(example: dict[str, Any]) -> str | None:
    answer = (example.get("question") or {}).get("answer") or {}
    label = answer.get("label")
    return str(label).strip().upper() if label is not None else None


def _question_options(example: dict[str, Any]) -> list[dict[str, str]]:
    options = []
    for option in (example.get("question") or {}).get("options") or []:
        label = str(option.get("label") or "").strip().upper()
        text = str(option.get("text") or "").strip()
        if label:
            options.append({"label": label, "text": text})
    return options


def parse_answer_label(response: str, options: list[dict[str, str]]) -> str | None:
    if not response:
        return None
    valid = {option["label"] for option in options}
    stripped = response.strip().upper()
    if stripped in valid:
        return stripped
    patterns = [
        r'"answer_label"\s*:\s*"([A-Z])"',
        r'"answer"\s*:\s*"([A-Z])"',
        r"\banswer(?:_label)?\s*[:=]\s*([A-Z])\b",
        r"\boption\s+([A-Z])\b",
        r"^\s*([A-Z])[\).:\s]",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if label in valid:
                return label
    for option in options:
        text = option["text"].strip().lower()
        if text and text in response.lower():
            return option["label"]
    return None


def parse_evidence_summary(response: str) -> str | None:
    if not response:
        return None
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, flags=re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    summary = payload.get("evidence_summary") or payload.get("rationale") or payload.get("reason")
    if summary is None:
        return None
    summary = str(summary).strip()
    return summary or None


def _clip_is_visible_for_example(result: Any, example: dict[str, Any], visible_until_s: float | None) -> bool:
    clip = result.clip
    video = example.get("video") or {}
    video_id = str(video.get("video_id") or "")
    video_path = str(video.get("primary_path") or "")
    example_id = str(example.get("example_id") or "")
    # Prefer path match so StreamingBench video_id collisions stay correct.
    if clip.video_path and video_path:
        if clip.video_path != video_path:
            return False
    elif clip.video_id and video_id and clip.video_id != video_id:
        return False
    # Example binding only for legacy per-QA indexes that set example_id.
    if clip.example_id and example_id and clip.example_id != example_id:
        return False
    if visible_until_s is not None and clip.start_s > visible_until_s:
        return False
    return True


def _retrieve_visible(
    store: Any,
    query_embedding: Any,
    *,
    example: dict[str, Any],
    visible_until_s: float | None,
    top_k: int,
    pool_k: int,
) -> list[Any]:
    from .schemas import RetrievedClip

    # Prefer video-local search for per-video FAISS refs.
    if hasattr(store, "search_in_video"):
        return store.search_in_video(
            query_embedding,
            example=example,
            topk=top_k,
            visible_until_s=visible_until_s,
        )

    raw = store.search(query_embedding, topk=max(top_k, pool_k))
    filtered = [
        result
        for result in raw
        if _clip_is_visible_for_example(result, example, visible_until_s)
    ]
    reranked: list[RetrievedClip] = []
    for rank, result in enumerate(filtered[:top_k], start=1):
        reranked.append(
            RetrievedClip(rank=rank, score=result.score, row_id=result.row_id, clip=result.clip)
        )
    return reranked


def _dedupe_append(existing: list[Any], new_results: list[Any]) -> list[Any]:
    seen = {result.clip.clip_id for result in existing}
    merged = list(existing)
    for result in new_results:
        if result.clip.clip_id in seen:
            continue
        seen.add(result.clip.clip_id)
        merged.append(result)
    return merged


def _result_to_dict(result: Any, *, include_embeddings: bool = False) -> dict[str, Any]:
    payload = result.to_dict()
    if not include_embeddings:
        payload.get("clip", {}).pop("embedding", None)
    return payload


def _tokenize(text: str) -> set[str]:
    stopwords = {
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
    }
    return {
        token.lower()
        for token in re.findall(r"[a-z0-9]+", text, flags=re.IGNORECASE)
        if len(token) > 1 and token.lower() not in stopwords
    }


def _next_query(base_query: str, options: list[dict[str, str]], retrieved: list[Any]) -> str:
    evidence_text = "\n".join(result.clip.text for result in retrieved[-5:])
    evidence_tokens = _tokenize(evidence_text)
    option_hints = []
    for option in options:
        option_tokens = _tokenize(option.get("text", ""))
        missing = sorted(option_tokens - evidence_tokens)
        if missing:
            option_hints.append(f"{option['label']}: {' '.join(missing[:8])}")
    if not option_hints:
        return base_query
    return base_query + "\nFocus on missing option evidence:\n" + "\n".join(option_hints)


def _heuristic_answer(example: dict[str, Any], retrieved: list[Any]) -> dict[str, Any]:
    question = example.get("question") or {}
    options = _question_options(example)
    evidence_text = "\n".join(result.clip.text for result in retrieved)
    evidence_tokens = _tokenize(evidence_text)
    if not options:
        return {
            "answer_label": None,
            "answer_text": "",
            "evidence_summary": "No multiple-choice options were available.",
        }
    scored: list[tuple[int, dict[str, str]]] = []
    for option in options:
        scored.append((len(_tokenize(option.get("text", "")) & evidence_tokens), option))
    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best = scored[0]
    return {
        "answer_label": best.get("label"),
        "answer_text": best.get("text"),
        "evidence_summary": (
            "Selected by option-token overlap with retrieved memory."
            if best_score > 0
            else "No option had lexical support; selected the first ranked fallback."
        ),
    }


def _build_answer_prompt(example: dict[str, Any], retrieved: list[Any], visible_until_s: float | None) -> str:
    question = example.get("question") or {}
    options = _question_options(example)
    evidence = []
    for result in retrieved:
        clip = result.clip
        evidence.append(
            {
                "rank": result.rank,
                "clip_id": clip.clip_id,
                "time_span_s": [clip.start_s, clip.end_s],
                "score": result.score,
                "text": clip.text,
            }
        )
    payload = {
        "task": "answer_streaming_video_question_from_retrieved_memory",
        "rules": [
            "Use only the retrieved memory text.",
            "Do not use video content beyond the visible cutoff.",
            "Output valid JSON only.",
            "Keep evidence_summary short; do not provide step-by-step reasoning.",
        ],
        "visible_until_s": visible_until_s,
        "question": {
            "question_text": question.get("question_text"),
            "options": options,
        },
        "retrieved_memory": evidence,
        "output_schema": {
            "answer_label": "A|B|C|D",
            "evidence_summary": "one short grounded sentence",
        },
    }
    return json.dumps(payload, ensure_ascii=False)


class LocalTextQwen:
    def __init__(self, model_path: str, *, max_new_tokens: int, device: str | None = None) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from .embeddings import resolve_torch_device

        self.max_new_tokens = max_new_tokens
        self.device = resolve_torch_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, prompt: str) -> str:
        import torch

        messages = [
            {"role": "system", "content": "You are an evidence-bounded video QA assistant."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[-1] :]
        return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def _qwen_answer(
    qwen: LocalTextQwen,
    example: dict[str, Any],
    retrieved: list[Any],
    visible_until_s: float | None,
) -> dict[str, Any]:
    prompt = _build_answer_prompt(example, retrieved, visible_until_s)
    response = qwen.generate(prompt)
    label = parse_answer_label(response, _question_options(example))
    summary = parse_evidence_summary(response)
    return {
        "answer_label": label,
        "answer_text": None,
        "evidence_summary": summary,
        "raw_response": response,
    }


def _build_embedder(args: argparse.Namespace, store: Any) -> Any:
    from .embeddings import CLIPVideoTextEmbedder, HashingTextEmbedder, Qwen3VLVLLMEmbedder

    backend = args.embedding_backend or store.manifest.get("embedding_backend") or "hashing_text"
    embed_device = args.embed_device
    if backend == "clip":
        return CLIPVideoTextEmbedder(
            model_name=args.clip_model or store.manifest["embedding_model"],
            device=embed_device,
        )
    if backend == "qwen3_vl":
        return Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model or store.manifest["embedding_model"],
            dtype=args.qwen3_vl_dtype,
            device=embed_device,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
    if backend == "qwen3_text_caption":
        return Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model or store.manifest["embedding_model"],
            dtype=args.qwen3_vl_dtype,
            device=embed_device,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
    if backend != "hashing_text":
        raise ValueError(f"unsupported embedding backend for query: {backend}")
    return HashingTextEmbedder(dim=int(store.manifest["dim"]))


def build_wrapper_config(dataset: str, args: argparse.Namespace) -> Any:
    from dataset_clip_wrapper.dataset_graph_presets import apply_profile_defaults, clip_policy_for, retrieval_for
    from dataset_clip_wrapper.schemas import BenchmarkProfile, BackboneConfig, RuntimeMode, VideoRegime, WrapperConfig

    regime = VideoRegime.STREAMING
    profile = BenchmarkProfile.DEFAULT
    clip_policy = clip_policy_for(dataset, regime)
    retrieval = retrieval_for(regime)
    if dataset == "videomme":
        clip_policy.observation_end_s = args.videomme_observation_end_s
    if args.window_s is not None:
        clip_policy.window_s = args.window_s
    if args.overlap_s is not None:
        clip_policy.overlap_s = args.overlap_s
    apply_profile_defaults(
        dataset=dataset,
        regime=regime,
        profile=profile,
        clip_policy=clip_policy,
        retrieval=retrieval,
    )
    return WrapperConfig(
        dataset_root=args.dataset_root,
        dataset=dataset,
        regime=regime,
        benchmark_profile=profile,
        mode=RuntimeMode.VIDEO_ONLY,
        clip_policy=clip_policy,
        retrieval=retrieval,
        backbone=BackboneConfig(name="annotation_only"),
        split=args.split,
        limit=args.limit_per_dataset,
        run_backbone=False,
    )


def iter_examples(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    from dataset_clip_wrapper.pipeline import iter_canonical_examples

    examples: list[tuple[str, dict[str, Any]]] = []
    for dataset in args.datasets:
        config = build_wrapper_config(dataset, args)
        for example in iter_canonical_examples(config):
            examples.append((dataset, example))
    if args.num_shards > 1:
        examples = [item for row, item in enumerate(examples) if row % args.num_shards == args.shard_index]
    return examples


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_dataset": {}}
    for key, rows in [("overall", records)] + [
        (dataset, [row for row in records if row.get("dataset") == dataset])
        for dataset in sorted({row.get("dataset") for row in records if row.get("dataset")})
    ]:
        total = len(rows)
        ok_rows = [row for row in rows if row.get("ok")]
        parsed = [row for row in ok_rows if row.get("prediction_label")]
        correct = [row for row in ok_rows if row.get("correct") is True]
        latencies = [float(row["timing_s"]["total"]) for row in ok_rows if row.get("timing_s", {}).get("total") is not None]
        payload = {
            "total": total,
            "ok": len(ok_rows),
            "failed": total - len(ok_rows),
            "parsed": len(parsed),
            "parse_rate": (len(parsed) / len(ok_rows)) if ok_rows else 0.0,
            "correct": len(correct),
            "accuracy": (len(correct) / len(ok_rows)) if ok_rows else 0.0,
            "accuracy_on_parsed": (len(correct) / len(parsed)) if parsed else 0.0,
            "avg_total_s": statistics.fmean(latencies) if latencies else None,
        }
        if key == "overall":
            summary["overall"] = payload
        else:
            summary["by_dataset"][key] = payload
    return summary


def write_run_config(args: argparse.Namespace, output_dir: Path, store: Any, embedder: Any) -> None:
    payload = {
        "runner": "iterative_rag_memory_query.py",
        "datasets": list(args.datasets),
        "dataset_root": args.dataset_root,
        "index_dir": str(args.index_dir),
        "index_manifest": store.manifest,
        "query_embedding_model": getattr(embedder, "name", None),
        "embed_device": getattr(embedder, "device", None),
        "answer_device": args.answer_device,
        "answer_backend": args.answer_backend,
        "model": args.model if args.answer_backend == "local_qwen" else None,
        "iterations": args.iterations,
        "per_iteration_top_k": args.per_iteration_top_k,
        "final_top_k": args.final_top_k,
        "pool_k": args.pool_k,
        "include_embeddings": args.include_embeddings,
        "streaming_policy": {
            "index_granularity": (store.manifest or {}).get("index_granularity"),
            "same_example_only": "only when indexed clip.example_id is set (legacy)",
            "same_video_only": True,
            "video_local_search": True,
            "no_future_leak": "clip.start_s <= visible_until_s",
        },
        "env": {
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iterative RAG over baseline FAISS clip memory.")
    parser.add_argument("--repo-root", default="/home/xwu/atomic_skills_for_video")
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--per-iteration-top-k", type=int, default=4)
    parser.add_argument("--final-top-k", type=int, default=8)
    parser.add_argument("--pool-k", type=int, default=128)
    parser.add_argument("--embedding-backend", default=None, choices=["hashing_text", "clip", "qwen3_vl", "qwen3_text_caption"])
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--qwen3-vl-model", default=None)
    parser.add_argument("--qwen3-vl-dtype", default="bfloat16")
    parser.add_argument("--qwen3-vl-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--qwen3-instruction",
        default="Represent the input for retrieving relevant video clips for a question.",
    )
    parser.add_argument(
        "--embed-device",
        default=None,
        help="Torch device for the retrieval embedder (e.g. cuda:0).",
    )
    parser.add_argument(
        "--answer-device",
        default=None,
        help="Torch device for local Qwen3.5 answering (e.g. cuda:1).",
    )
    parser.add_argument("--answer-backend", default="heuristic", choices=["heuristic", "local_qwen"])
    parser.add_argument("--model", default="/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--include-embeddings", action="store_true")
    args = parser.parse_args()

    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.per_iteration_top_k <= 0 or args.final_top_k <= 0:
        parser.error("--per-iteration-top-k and --final-top-k must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    ensure_repo_on_path(args.repo_root)
    from .faiss_store import FaissClipStore
    from .schemas import visible_until_from_canonical

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    metrics_path = args.output_dir / "metrics_summary.json"

    store = FaissClipStore.load(args.index_dir)
    embedder = _build_embedder(args, store)
    qwen = (
        LocalTextQwen(args.model, max_new_tokens=args.max_new_tokens, device=args.answer_device)
        if args.answer_backend == "local_qwen"
        else None
    )
    write_run_config(args, args.output_dir, store, embedder)

    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"index_clips={len(store.clips)} datasets={','.join(args.datasets)}", flush=True)
    examples = iter_examples(args)
    print(f"canonical_examples={len(examples)} shard={args.shard_index}/{args.num_shards}", flush=True)

    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as handle:
        for dataset, example in examples:
            started = time.perf_counter()
            options = _question_options(example)
            gold = _question_answer_label(example)
            visible_until_s = visible_until_from_canonical(
                example,
                default_videomme_cutoff_s=args.videomme_observation_end_s,
            )

            base_query = _question_with_options(example)
            query = base_query
            retrieved: list[RetrievedClip] = []
            trace: list[dict[str, Any]] = []
            try:
                for iteration in range(1, args.iterations + 1):
                    query_embedding = embedder.encode([query])
                    step_results = _retrieve_visible(
                        store,
                        query_embedding,
                        example=example,
                        visible_until_s=visible_until_s,
                        top_k=args.per_iteration_top_k,
                        pool_k=args.pool_k,
                    )
                    retrieved = _dedupe_append(retrieved, step_results)
                    trace.append(
                        {
                            "iteration": iteration,
                            "query": query,
                            "retrieved": [
                                _result_to_dict(result, include_embeddings=args.include_embeddings)
                                for result in step_results
                            ],
                            "unique_memory_count": len(retrieved),
                        }
                    )
                    if len(retrieved) >= args.final_top_k:
                        break
                    query = _next_query(base_query, options, retrieved)

                final_retrieved = retrieved[: args.final_top_k]
                if qwen is not None:
                    prediction = _qwen_answer(qwen, example, final_retrieved, visible_until_s)
                else:
                    prediction = _heuristic_answer(example, final_retrieved)
                prediction_label = prediction.get("answer_label")
                if prediction_label is not None:
                    prediction_label = str(prediction_label).strip().upper()
                correct = (prediction_label == gold) if gold and prediction_label else None
                record = {
                    "ok": True,
                    "dataset": dataset,
                    "example_id": example.get("example_id"),
                    "question_id": (example.get("question") or {}).get("question_id"),
                    "video_id": (example.get("video") or {}).get("video_id"),
                    "visible_until_s": visible_until_s,
                    "question": example.get("question"),
                    "retrieval_trace": trace,
                    "retrieved_memory": [
                        _result_to_dict(result, include_embeddings=args.include_embeddings)
                        for result in final_retrieved
                    ],
                    "prediction": prediction,
                    "prediction_label": prediction_label,
                    "gold_label": gold,
                    "correct": correct,
                    "timing_s": {"total": time.perf_counter() - started},
                }
            except Exception as exc:
                record = {
                    "ok": False,
                    "dataset": dataset,
                    "example_id": example.get("example_id"),
                    "question_id": (example.get("question") or {}).get("question_id"),
                    "video_id": (example.get("video") or {}).get("video_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "prediction_label": None,
                    "gold_label": gold,
                    "correct": None,
                    "timing_s": {"total": time.perf_counter() - started},
                }
            records.append(record)
            _json_dump_line(handle, record)

    metrics = metric_summary(records)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output_dir": str(args.output_dir), "metrics": metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
