#!/usr/bin/env python3
"""Per-video embedding retrieval + video-clip answering for streaming QA.

Matches the same streaming setting used by M3-style and direct-video baselines:
  - retrieve only within the current example/video
  - keep only clips with start_s <= visible_until_s (no future leak)
  - no global FAISS index across the corpus

Difference vs M3: clip selection uses CLIP / Qwen3-VL-Embedding similarity
instead of temporal heuristics (uniform/latest). Answering still feeds the
retrieved video clips into local Qwen3.5-9B (VL), same as M3 video_clip mode.
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

import numpy as np

DEFAULT_DATASETS = ("ovo_bench", "videomme", "streaming_bench")
SUPPORTED_DATASETS = ("ovo_bench", "videomme", "streaming_bench")
DEFAULT_OUTPUT_ROOT = Path(
    "/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/per_video_embedding_rag"
)


def ensure_repo_on_path(repo_root: str) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def _json_dump_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


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


def _question_with_options(example: dict[str, Any]) -> str:
    question = example.get("question") or {}
    lines = [str(question.get("question_text") or "").strip()]
    options = _question_options(example)
    if options:
        lines.append("Options:")
        for option in options:
            lines.append(f"{option['label']}. {option['text']}")
    return "\n".join(line for line in lines if line)


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
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    payload = None
    if match:
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict):
        summary = payload.get("evidence_summary") or payload.get("rationale") or payload.get("reason")
        if summary is None:
            return None
        summary = str(summary).strip()
        return summary or None
    return None


class LocalVideoQwen:
    """Qwen3.5 VL answerer: consumes retrieved video clips (M3 video_clip path)."""

    def __init__(
        self,
        model_path: str,
        *,
        max_new_tokens: int,
        device: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        from .embeddings import resolve_torch_device

        self.max_new_tokens = max_new_tokens
        self.enable_thinking = enable_thinking
        self.device = resolve_torch_device(device)
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def generate(self, *, media_records: list[dict[str, Any]], prompt_text: str) -> str:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers.video_utils import VideoMetadata

        content: list[dict[str, Any]] = []
        for record in media_records:
            content.append(
                {
                    "type": "video",
                    "video": record["path"],
                    "video_start": record["video_start"],
                    "video_end": record["video_end"],
                    "fps": record["fps"],
                    "max_frames": record["max_frames"],
                }
            )
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            return_video_kwargs=True,
            return_video_metadata=True,
        )
        videos = []
        video_metadata = []
        for video_input in video_inputs or []:
            if isinstance(video_input, tuple) and len(video_input) == 2:
                video_tensor, metadata = video_input
                frames_indices = metadata.get("frames_indices")
                if hasattr(frames_indices, "tolist"):
                    frames_indices = frames_indices.tolist()
                video_metadata.append(
                    VideoMetadata(
                        total_num_frames=int(metadata.get("total_num_frames") or video_tensor.shape[0]),
                        fps=metadata.get("fps"),
                        frames_indices=[int(index) for index in frames_indices] if frames_indices is not None else None,
                        video_backend=metadata.get("video_backend"),
                    )
                )
                videos.append(video_tensor)
            else:
                videos.append(video_input)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata or None,
            **video_kwargs,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()
        }
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        generated = output_ids[:, inputs["input_ids"].shape[-1] :]
        return self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def media_records_from_retrieved(
    retrieved: list[dict[str, Any]],
    *,
    video_fps: float,
    video_max_frames_per_clip: int,
) -> list[dict[str, Any]]:
    records = []
    for row, item in enumerate(retrieved):
        clip = item["clip"]
        start_s = float(clip.get("start_s", 0.0))
        end_s = float(clip.get("end_s", start_s))
        path = str(clip.get("video_path") or "")
        if not path:
            continue
        records.append(
            {
                "media_type": "video_clip",
                "memory_id": f"CLIP_{row}",
                "rank": item.get("rank"),
                "score": item.get("score"),
                "clip_id": clip.get("clip_id"),
                "path": path,
                "video_start": start_s,
                "video_end": end_s,
                "source_span": {"start_s": start_s, "end_s": end_s},
                "granularity": clip.get("granularity"),
                "fps": video_fps,
                "max_frames": video_max_frames_per_clip,
            }
        )
    return records


def build_answer_prompt(
    example: dict[str, Any],
    media_records: list[dict[str, Any]],
    visible_until_s: float | None,
    *,
    answer_mode: str,
) -> str:
    question = example.get("question") or {}
    options = _question_options(example)
    lines = [
        "You are a streaming video QA assistant.",
        "Use only the provided retrieved video clips from the visible part of the video.",
        "Do not use information after the streaming cutoff.",
    ]
    if visible_until_s is not None:
        lines.append(f"Visible video cutoff: {visible_until_s:.2f} seconds.")
    lines.append("Retrieved clips:")
    for record in media_records:
        lines.append(
            f"- {record['memory_id']}: clip_id={record.get('clip_id')} "
            f"time={record['video_start']:.2f}-{record['video_end']:.2f}s "
            f"score={record.get('score')}"
        )
    lines.append(f"Question: {question.get('question_text') or ''}")
    if options:
        lines.append("Options:")
        for option in options:
            lines.append(f"{option['label']}. {option['text']}")
    if answer_mode == "json_rationale":
        lines.extend(
            [
                "Output valid JSON only.",
                'Required schema: {"answer_label": "A|B|C|D", "evidence_summary": "one short grounded sentence"}',
                "Keep evidence_summary concise; do not write a step-by-step chain of thought.",
            ]
        )
    else:
        lines.append("Output exactly one option label, such as A, B, C, or D. Do not explain.")
    lines.append("Final answer:")
    return "\n".join(lines)


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


def build_embedder(args: argparse.Namespace) -> Any:
    from .embeddings import CLIPVideoTextEmbedder, Qwen3VLVLLMEmbedder

    if args.embedding_backend == "clip":
        return CLIPVideoTextEmbedder(
            model_name=args.clip_model,
            device=args.embed_device,
            frames_per_clip=args.frames_per_clip,
            image_batch_size=args.image_batch_size,
            decode_workers=args.decode_workers,
            decode_strategy=args.decode_strategy,
            encode_chunk_size=args.encode_chunk_size,
        )
    if args.embedding_backend == "qwen3_vl":
        return Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model,
            dtype=args.qwen3_vl_dtype,
            device=args.embed_device,
            frames_per_clip=args.frames_per_clip,
            image_batch_size=args.image_batch_size,
            decode_workers=args.decode_workers,
            decode_strategy=args.decode_strategy,
            encode_chunk_size=args.encode_chunk_size,
            clip_encode_mode=args.clip_encode_mode,
            instruction=args.qwen3_instruction,
        )
    raise ValueError(f"unsupported embedding backend: {args.embedding_backend}")


def subsample_clips_for_embedding(
    clips: list[Any],
    *,
    max_candidates: int,
    strategy: str,
    visible_until_s: float | None,
) -> list[Any]:
    """Temporally subsample visible clips before embedding (M3-style prefilter)."""

    if max_candidates <= 0 or len(clips) <= max_candidates:
        return list(clips)
    ordered = sorted(clips, key=lambda clip: float(getattr(clip, "start_s", 0.0)))
    if strategy == "latest":
        return ordered[-max_candidates:]
    if strategy == "uniform":
        if max_candidates == 1:
            return [ordered[len(ordered) // 2]]
        last = len(ordered) - 1
        indices = [round(i * last / (max_candidates - 1)) for i in range(max_candidates)]
        # Stable unique while preserving temporal order.
        seen = set()
        picked = []
        for index in indices:
            if index in seen:
                continue
            seen.add(index)
            picked.append(ordered[index])
        return picked
    if strategy == "centered_at_cutoff":
        if visible_until_s is None:
            return ordered[-max_candidates:]
        return sorted(
            ordered,
            key=lambda clip: abs(
                (float(getattr(clip, "start_s", 0.0)) + float(getattr(clip, "end_s", 0.0))) / 2.0
                - visible_until_s
            ),
        )[:max_candidates]
    raise ValueError(f"unknown candidate strategy: {strategy}")


def retrieve_within_example(
    embedder: Any,
    example: dict[str, Any],
    *,
    top_k: int,
    max_embed_candidates: int,
    candidate_strategy: str,
    videomme_observation_end_s: float | None,
    cache: Any,
    embedding_backend: str,
) -> tuple[list[dict[str, Any]], float | None, dict[str, Any]]:
    from .embeddings import embed_clips_with_cache
    from .schemas import clip_records_from_canonical, visible_until_from_canonical

    visible_until_s = visible_until_from_canonical(
        example,
        default_videomme_cutoff_s=videomme_observation_end_s,
    )
    clips = clip_records_from_canonical(
        example,
        start_row_id=0,
        default_videomme_cutoff_s=videomme_observation_end_s,
        text_mode="caption_metadata",
    )
    # Defense in depth: same-example / same-video / no-future (clip builder already applies cutoff).
    example_id = str(example.get("example_id") or "")
    video_id = str((example.get("video") or {}).get("video_id") or "")
    filtered = []
    for clip in clips:
        if clip.example_id and example_id and clip.example_id != example_id:
            continue
        if clip.video_id and video_id and clip.video_id != video_id:
            continue
        if visible_until_s is not None and clip.start_s > visible_until_s:
            continue
        filtered.append(clip)

    candidates = subsample_clips_for_embedding(
        filtered,
        max_candidates=max_embed_candidates,
        strategy=candidate_strategy,
        visible_until_s=visible_until_s,
    )

    stats = {
        "candidate_clips": len(clips),
        "filtered_clips": len(filtered),
        "subsampled_clips": len(candidates),
        "max_embed_candidates": max_embed_candidates,
        "candidate_strategy": candidate_strategy,
        "embedded_clips": 0,
        "skipped_decode_failures": 0,
        "cache_hits": 0,
        "cache_misses": 0,
    }
    if not candidates:
        return [], visible_until_s, stats

    vectors, kept, cache_stats = embed_clips_with_cache(
        embedder,
        candidates,
        cache=cache,
        embedding_backend=embedding_backend,
        skip_failed_clips=True,
    )
    stats["embedded_clips"] = len(kept)
    stats["skipped_decode_failures"] = len(candidates) - len(kept)
    stats.update(cache_stats)
    if vectors.size == 0 or not kept:
        return [], visible_until_s, stats

    kept_clips = [candidates[index] for index in kept]
    query = _question_with_options(example)
    query_vec = embedder.encode([query])[0]
    scores = vectors @ query_vec
    order = np.argsort(-scores)[: max(1, top_k)]
    retrieved = []
    for rank, idx in enumerate(order, start=1):
        clip = kept_clips[int(idx)]
        retrieved.append(
            {
                "rank": rank,
                "score": float(scores[int(idx)]),
                "clip": clip.to_dict(),
            }
        )
    return retrieved, visible_until_s, stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-video embedding RAG (no global FAISS).")
    parser.add_argument("--repo-root", default="/home/xwu/atomic_skills_for_video")
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3, help="Embedding-ranked clips fed to the VL answerer.")
    parser.add_argument(
        "--max-embed-candidates",
        type=int,
        default=16,
        help="Temporally subsample visible clips to at most this many before embedding.",
    )
    parser.add_argument(
        "--candidate-strategy",
        default="uniform",
        choices=["uniform", "latest", "centered_at_cutoff"],
        help="How to subsample visible clips before embedding (M3-style).",
    )
    parser.add_argument("--embedding-backend", default="qwen3_vl", choices=["clip", "qwen3_vl"])
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--qwen3-vl-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--qwen3-vl-dtype", default="bfloat16")
    parser.add_argument(
        "--qwen3-instruction",
        default="Represent the input for retrieving relevant video clips for a question.",
    )
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument(
        "--clip-encode-mode",
        default="image_mean",
        choices=["image_mean", "video"],
        help=(
            "How to embed each temporal clip with Qwen3-VL: "
            "'image_mean' encodes frames as images then mean-pools; "
            "'video' encodes each clip as one native video input (frame list)."
        ),
    )
    parser.add_argument("--image-batch-size", type=int, default=32)
    parser.add_argument("--decode-workers", type=int, default=2)
    parser.add_argument("--decode-strategy", default="scan", choices=["seek", "scan"])
    parser.add_argument("--encode-chunk-size", type=int, default=64)
    parser.add_argument("--embed-device", default=None)
    parser.add_argument("--answer-device", default=None)
    parser.add_argument("--model", default="/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B")
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames-per-clip", type=int, default=8)
    parser.add_argument("--answer-mode", default="label_only", choices=["label_only", "json_rationale"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()

    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.max_embed_candidates <= 0:
        parser.error("--max-embed-candidates must be positive")
    if args.clip_encode_mode == "video" and args.embedding_backend != "qwen3_vl":
        parser.error("--clip-encode-mode video requires --embedding-backend qwen3_vl")
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    ensure_repo_on_path(args.repo_root)
    from .embeddings import ClipEmbeddingCache

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.output_dir / "records.jsonl"
    metrics_path = args.output_dir / "metrics_summary.json"

    embedder = build_embedder(args)
    qwen = LocalVideoQwen(
        args.model,
        max_new_tokens=args.max_new_tokens,
        device=args.answer_device,
        enable_thinking=args.enable_thinking,
    )
    cache = ClipEmbeddingCache()

    run_config = {
        "runner": "per_video_embedding_rag.py",
        "setting": {
            "scope": "per_example_same_video_only",
            "no_global_faiss": True,
            "no_future_leak": "clip.start_s <= visible_until_s",
            "retrieval": "embedding_similarity_replaces_m3_heuristic",
            "answer_input": "retrieved_video_clips",
            "batch_encode": True,
            "cross_example_clip_cache": True,
            "candidate_subsample_before_embed": True,
            "aligned_with": ["m3_agent/streaming_eval.py", "baseline/qwen35_streaming_eval.py"],
        },
        "datasets": list(args.datasets),
        "embedding_backend": args.embedding_backend,
        "embedding_model": getattr(embedder, "name", None),
        "embed_device": getattr(embedder, "device", None),
        "answer_device": args.answer_device,
        "model": args.model,
        "top_k": args.top_k,
        "max_embed_candidates": args.max_embed_candidates,
        "candidate_strategy": args.candidate_strategy,
        "frames_per_clip": args.frames_per_clip,
        "clip_encode_mode": args.clip_encode_mode,
        "image_batch_size": args.image_batch_size,
        "encode_chunk_size": args.encode_chunk_size,
        "decode_workers": args.decode_workers,
        "video_fps": args.video_fps,
        "video_max_frames_per_clip": args.video_max_frames_per_clip,
        "answer_mode": args.answer_mode,
        "enable_thinking": args.enable_thinking,
        "env": {
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(
        f"per_video_embed_retrieve backend={args.embedding_backend} top_k={args.top_k} "
        f"max_embed_candidates={args.max_embed_candidates} candidate_strategy={args.candidate_strategy} "
        f"frames_per_clip={args.frames_per_clip} clip_encode_mode={args.clip_encode_mode} "
        f"answer=video_clip fps={args.video_fps} max_frames={args.video_max_frames_per_clip} "
        f"answer_mode={args.answer_mode} thinking={args.enable_thinking} "
        f"embed_device={getattr(embedder, 'device', None)} answer_device={args.answer_device}",
        flush=True,
    )
    examples = iter_examples(args)
    print(f"canonical_examples={len(examples)} shard={args.shard_index}/{args.num_shards}", flush=True)

    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as handle:
        for index, (dataset, example) in enumerate(examples, start=1):
            started = time.perf_counter()
            gold = _question_answer_label(example)
            options = _question_options(example)
            try:
                retrieved, visible_until_s, retrieval_stats = retrieve_within_example(
                    embedder,
                    example,
                    top_k=args.top_k,
                    max_embed_candidates=args.max_embed_candidates,
                    candidate_strategy=args.candidate_strategy,
                    videomme_observation_end_s=args.videomme_observation_end_s,
                    cache=cache,
                    embedding_backend=args.embedding_backend,
                )
                if not retrieved:
                    raise RuntimeError(
                        f"no retrievable visible clips after decode "
                        f"(candidates={retrieval_stats['candidate_clips']}, "
                        f"embedded={retrieval_stats['embedded_clips']})"
                    )
                media_records = media_records_from_retrieved(
                    retrieved,
                    video_fps=args.video_fps,
                    video_max_frames_per_clip=args.video_max_frames_per_clip,
                )
                if not media_records:
                    raise RuntimeError("retrieved clips missing video paths")
                prompt = build_answer_prompt(
                    example,
                    media_records,
                    visible_until_s,
                    answer_mode=args.answer_mode,
                )
                gen_started = time.perf_counter()
                response = qwen.generate(media_records=media_records, prompt_text=prompt)
                generate_s = time.perf_counter() - gen_started
                prediction_label = parse_answer_label(response, options)
                evidence_summary = parse_evidence_summary(response) if args.answer_mode == "json_rationale" else None
                correct = (prediction_label == gold) if gold and prediction_label else None
                record = {
                    "ok": True,
                    "dataset": dataset,
                    "example_id": example.get("example_id"),
                    "question_id": (example.get("question") or {}).get("question_id"),
                    "video_id": (example.get("video") or {}).get("video_id"),
                    "input_mode": "video_clip_embedding_retrieve",
                    "visible_until_s": visible_until_s,
                    "retrieval_stats": retrieval_stats,
                    "retrieved_memory": retrieved,
                    "media_records": media_records,
                    "gold_label": gold,
                    "prediction_label": prediction_label,
                    "evidence_summary": evidence_summary,
                    "response": response,
                    "correct": correct,
                    "timing_s": {
                        "total": time.perf_counter() - started,
                        "generate": generate_s,
                    },
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
            if index == 1 or index % max(1, args.progress_every) == 0 or index == len(examples):
                ok = sum(1 for row in records if row.get("ok"))
                correct_n = sum(1 for row in records if row.get("correct") is True)
                parsed_n = sum(1 for row in records if row.get("prediction_label"))
                cache_stats = cache.stats()
                print(
                    f"progress={index}/{len(examples)} ok={ok} parsed={parsed_n} correct={correct_n} "
                    f"cache_hits={cache_stats['cache_hits']} cache_misses={cache_stats['cache_misses']} "
                    f"last_example={record.get('example_id')} last_ok={record.get('ok')} "
                    f"last_pred={record.get('prediction_label')} "
                    f"last_total_s={(record.get('timing_s') or {}).get('total')}",
                    flush=True,
                )

    metrics = metric_summary(records)
    metrics["cache"] = cache.stats()
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"records": len(records), "output_dir": str(args.output_dir), "metrics": metrics}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
