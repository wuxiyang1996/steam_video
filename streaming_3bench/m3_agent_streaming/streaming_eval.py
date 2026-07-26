#!/usr/bin/env python3
"""M3-style streaming evaluation on OVO-Bench, VideoMME, and StreamingBench.

This runner reuses the canonical video clip schema produced by
``atomic_skills_for_video`` and evaluates local Qwen3.5-9B under a causal
streaming visibility policy. It keeps M3-Agent's memory/retrieval framing while
avoiding hosted embedding or judge APIs: visible clips become the memory bank,
the runner retrieves a small set of causal clips, and Qwen answers from those
retrieved memories/video clips.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import os
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATASETS = ("ovo_bench", "videomme")
SUPPORTED_DATASETS = ("ovo_bench", "videomme", "streaming_bench")


@dataclass
class PreparedExample:
    dataset: str
    example: dict[str, Any]
    media_records: list[dict[str, Any]]
    images: list[Any]
    visible_until_s: float | None
    retrieved_memory: list[dict[str, Any]]
    prepare_error: str | None = None


def ensure_repo_on_path(repo_root: str) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def dump_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "example"


def question_options(example: dict[str, Any]) -> list[dict[str, str]]:
    options = []
    for option in (example.get("question") or {}).get("options") or []:
        label = str(option.get("label") or "").strip().upper()
        text = str(option.get("text") or "").strip()
        if label:
            options.append({"label": label, "text": text})
    return options


def gold_label(example: dict[str, Any]) -> str | None:
    answer = (example.get("question") or {}).get("answer") or {}
    label = answer.get("label")
    return str(label).strip().upper() if label is not None else None


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
    return str(summary).strip() if summary else None


def streaming_visible_until(example: dict[str, Any], *, videomme_observation_end_s: float | None) -> float | None:
    question = example.get("question") or {}
    video = example.get("video") or {}
    duration = video.get("duration_s")
    duration_s = float(duration) if isinstance(duration, (int, float)) else None
    anchor = question.get("time_anchor_s")
    if isinstance(anchor, (int, float)):
        visible = float(anchor)
    elif example.get("dataset") == "videomme":
        visible = videomme_observation_end_s if videomme_observation_end_s is not None else duration_s
    else:
        visible = duration_s
    if visible is not None and duration_s is not None:
        visible = max(0.0, min(float(visible), duration_s))
    return visible


def visible_clips(example: dict[str, Any], visible_until_s: float | None) -> list[dict[str, Any]]:
    clips = ((example.get("video") or {}).get("derived_clips") or [])
    usable = []
    for clip in clips:
        path = clip.get("path")
        span = clip.get("source_span") or {}
        start_s = float(span.get("start_s", 0.0))
        end_s = float(span.get("end_s", start_s))
        if not path or end_s <= start_s:
            continue
        if visible_until_s is not None and start_s > visible_until_s:
            continue
        clipped = copy.deepcopy(clip)
        clipped.setdefault("source_span", {})
        if visible_until_s is not None:
            clipped["source_span"]["end_s"] = min(end_s, visible_until_s)
        usable.append(clipped)
    fine = [clip for clip in usable if clip.get("granularity") == "fine"]
    return fine or usable


def retrieve_visible_clips(
    clips: list[dict[str, Any]],
    *,
    visible_until_s: float | None,
    topk: int,
    strategy: str,
) -> list[dict[str, Any]]:
    if topk <= 0 or not clips:
        return []
    ordered = sorted(clips, key=lambda clip: float((clip.get("source_span") or {}).get("start_s", 0.0)))
    if strategy == "uniform":
        if len(ordered) <= topk:
            return ordered
        if topk == 1:
            return [ordered[len(ordered) // 2]]
        last = len(ordered) - 1
        return [ordered[round(i * last / (topk - 1))] for i in range(topk)]
    if strategy == "latest":
        return ordered[-topk:]
    if strategy == "centered_at_cutoff":
        if visible_until_s is None:
            return ordered[-topk:]
        return sorted(
            ordered,
            key=lambda clip: abs(
                (
                    float((clip.get("source_span") or {}).get("start_s", 0.0))
                    + float((clip.get("source_span") or {}).get("end_s", 0.0))
                )
                / 2.0
                - visible_until_s
            ),
        )[:topk]
    raise ValueError(f"unknown retrieval strategy: {strategy}")


def clip_memory_record(clip: dict[str, Any], row: int) -> dict[str, Any]:
    span = clip.get("source_span") or {}
    start_s = float(span.get("start_s", 0.0))
    end_s = float(span.get("end_s", start_s))
    return {
        "memory_id": f"CLIP_{row}",
        "clip_id": clip.get("clip_id"),
        "path": clip.get("path"),
        "source_span": {"start_s": start_s, "end_s": end_s},
        "granularity": clip.get("granularity"),
        "text": f"CLIP_{row}: {clip.get('clip_id')} from {start_s:.2f}s to {end_s:.2f}s.",
    }


def media_record_from_clip(clip: dict[str, Any], row: int, *, video_fps: float, max_frames: int) -> dict[str, Any]:
    span = clip.get("source_span") or {}
    start_s = float(span.get("start_s", 0.0))
    end_s = float(span.get("end_s", start_s))
    return {
        "media_type": "video_clip",
        "memory_id": f"CLIP_{row}",
        "clip_id": clip.get("clip_id"),
        "path": clip.get("path"),
        "video_start": start_s,
        "video_end": end_s,
        "source_span": {"start_s": start_s, "end_s": end_s},
        "granularity": clip.get("granularity"),
        "fps": video_fps,
        "max_frames": max_frames,
    }


def read_frame(video_path: str, timestamp_s: float) -> Any:
    import cv2  # type: ignore
    from PIL import Image

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame at {timestamp_s:.2f}s from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def prepare_example(
    dataset: str,
    example: dict[str, Any],
    *,
    input_mode: str,
    retrieval_topk: int,
    retrieval_strategy: str,
    video_fps: float,
    video_max_frames_per_clip: int,
    videomme_observation_end_s: float | None,
) -> PreparedExample:
    try:
        visible_until_s = streaming_visible_until(
            example,
            videomme_observation_end_s=videomme_observation_end_s,
        )
        clips = visible_clips(example, visible_until_s)
        retrieved = retrieve_visible_clips(
            clips,
            visible_until_s=visible_until_s,
            topk=retrieval_topk,
            strategy=retrieval_strategy,
        )
        if not retrieved:
            raise RuntimeError("no visible clips available after streaming cutoff")

        memory = [clip_memory_record(clip, row) for row, clip in enumerate(retrieved)]
        images: list[Any] = []
        media_records: list[dict[str, Any]] = []
        if input_mode == "video_clip":
            media_records = [
                media_record_from_clip(
                    clip,
                    row,
                    video_fps=video_fps,
                    max_frames=video_max_frames_per_clip,
                )
                for row, clip in enumerate(retrieved)
            ]
        elif input_mode == "frames":
            for row, clip in enumerate(retrieved):
                span = clip.get("source_span") or {}
                start_s = float(span.get("start_s", 0.0))
                end_s = float(span.get("end_s", start_s))
                timestamp_s = max(0.0, (start_s + end_s) / 2.0)
                images.append(read_frame(str(clip["path"]), timestamp_s))
                media_records.append(
                    {
                        "media_type": "frame",
                        "memory_id": f"CLIP_{row}",
                        "clip_id": clip.get("clip_id"),
                        "path": clip.get("path"),
                        "timestamp_s": timestamp_s,
                        "source_span": {"start_s": start_s, "end_s": end_s},
                        "granularity": clip.get("granularity"),
                    }
                )
        else:
            raise ValueError(f"unsupported input mode: {input_mode}")
        return PreparedExample(dataset, example, media_records, images, visible_until_s, memory)
    except Exception as exc:
        return PreparedExample(dataset, example, [], [], None, [], f"{type(exc).__name__}: {exc}")


def build_prompt(
    example: dict[str, Any],
    *,
    retrieved_memory: list[dict[str, Any]],
    visible_until_s: float | None,
    answer_mode: str,
) -> str:
    question = example.get("question") or {}
    options = question_options(example)
    lines = [
        "You are an M3-style streaming video memory agent.",
        "You may use only retrieved memories and media from the visible part of the video.",
        "Do not use information after the streaming cutoff.",
    ]
    if visible_until_s is not None:
        lines.append(f"Visible video cutoff: {visible_until_s:.2f} seconds.")
    lines.append("Retrieved memory bank:")
    for memory in retrieved_memory:
        lines.append(
            f"- {memory['memory_id']}: clip_id={memory['clip_id']} "
            f"time={memory['source_span']['start_s']:.2f}-{memory['source_span']['end_s']:.2f}s"
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
                "Keep evidence_summary concise and cite the relevant CLIP id if possible.",
            ]
        )
    else:
        lines.append("Output exactly one option label, such as A, B, C, or D. Do not explain.")
    lines.append("Final answer:")
    return "\n".join(lines)


def build_messages(
    *,
    images: list[Any],
    media_records: list[dict[str, Any]],
    text: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if input_mode == "video_clip":
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
    else:
        content.extend({"type": "image", "image": image} for image in images)
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def generate_one(
    model: Any,
    processor: Any,
    *,
    images: list[Any],
    media_records: list[dict[str, Any]],
    input_mode: str,
    prompt_text: str,
    max_new_tokens: int,
    enable_thinking: bool,
) -> str:
    import torch

    messages = build_messages(images=images, media_records=media_records, text=prompt_text, input_mode=input_mode)
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    if input_mode == "video_clip":
        from qwen_vl_utils import process_vision_info
        from transformers.video_utils import VideoMetadata

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
        inputs = processor(
            text=[prompt],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadata or None,
            **video_kwargs,
            return_tensors="pt",
        )
    else:
        inputs = processor(text=[prompt], images=images, return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[-1]
    generated = output_ids[:, input_len:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


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
    if args.canonical_jsonl:
        examples = [(str(example.get("dataset") or ""), example) for example in iter_jsonl(args.canonical_jsonl)]
    else:
        from dataset_clip_wrapper.pipeline import iter_canonical_examples

        examples = []
        for dataset in args.datasets:
            config = build_wrapper_config(dataset, args)
            for example in iter_canonical_examples(config):
                examples.append((dataset, example))
    if args.require_multiple_choice:
        examples = [
            (dataset, example)
            for dataset, example in examples
            if gold_label(example) and len(question_options(example)) >= 2
        ]
    if args.num_shards > 1:
        examples = [item for row, item in enumerate(examples) if row % args.num_shards == args.shard_index]
    return examples


def metric_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"overall": {}, "by_dataset": {}}
    datasets = sorted({row.get("dataset") for row in records if row.get("dataset")})
    for key, rows in [("overall", records)] + [(dataset, [row for row in records if row.get("dataset") == dataset]) for dataset in datasets]:
        ok_rows = [row for row in rows if row.get("ok")]
        parsed = [row for row in ok_rows if row.get("prediction_label")]
        correct = [row for row in ok_rows if row.get("correct") is True]
        latencies = [float(row["timing_s"]["generate"]) for row in ok_rows if row.get("timing_s", {}).get("generate") is not None]
        payload = {
            "total": len(rows),
            "ok": len(ok_rows),
            "failed": len(rows) - len(ok_rows),
            "parsed": len(parsed),
            "parse_rate": (len(parsed) / len(ok_rows)) if ok_rows else 0.0,
            "correct": len(correct),
            "accuracy": (len(correct) / len(ok_rows)) if ok_rows else 0.0,
            "accuracy_on_parsed": (len(correct) / len(parsed)) if parsed else 0.0,
            "avg_generate_s": statistics.fmean(latencies) if latencies else None,
        }
        if key == "overall":
            summary["overall"] = payload
        else:
            summary["by_dataset"][key] = payload
    return summary


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload = {
        "runner": "m3_agent.streaming_eval",
        "datasets": list(args.datasets),
        "model": args.model,
        "atomic_repo_root": args.atomic_repo_root,
        "dataset_root": args.dataset_root,
        "canonical_jsonl": str(args.canonical_jsonl) if args.canonical_jsonl else None,
        "split": args.split,
        "limit_per_dataset": args.limit_per_dataset,
        "shard": {"shard_index": args.shard_index, "num_shards": args.num_shards},
        "streaming_definition": {
            "ovo_bench": "Use question.time_anchor_s when present, clipped to video duration.",
            "videomme": "Use --videomme-observation-end-s as visibility cutoff.",
            "streaming_bench": "Use question.time_anchor_s from official StreamingBench records; by default only multiple-choice rows are scored.",
            "input_mode": args.input_mode,
            "retrieval_topk": args.retrieval_topk,
            "retrieval_strategy": args.retrieval_strategy,
            "answer_mode": args.answer_mode,
            "require_multiple_choice": args.require_multiple_choice,
        },
        "env": {
            "hostname": os.uname().nodename,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
    }
    (output_dir / "run_config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B")
    parser.add_argument("--atomic-repo-root", default="/home/xwu/atomic_skills_for_video")
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--canonical-jsonl", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--input-mode", default="video_clip", choices=["frames", "video_clip"])
    parser.add_argument("--retrieval-topk", type=int, default=3)
    parser.add_argument(
        "--retrieval-strategy",
        default="latest",
        choices=["latest", "uniform", "centered_at_cutoff"],
    )
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames-per-clip", type=int, default=8)
    parser.add_argument("--frame-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--answer-mode", default="json_rationale", choices=["label_only", "json_rationale"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--require-multiple-choice", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    args = parser.parse_args()

    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    ensure_repo_on_path(args.atomic_repo_root)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from baseline.model_runtime import attention_kwargs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.output_dir / "canonical_schemas.jsonl"
    records_path = args.output_dir / "records.jsonl"
    metrics_path = args.output_dir / "metrics_summary.json"
    write_run_config(args, args.output_dir)

    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)
    print(f"m3_streaming_retrieval={args.retrieval_strategy} topk={args.retrieval_topk}", flush=True)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        **attention_kwargs(),
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"model_loaded_s={load_s:.2f} device={model.device}", flush=True)

    examples = iter_examples(args)
    print(f"canonical_examples={len(examples)} shard={args.shard_index}/{args.num_shards}", flush=True)
    with schema_path.open("w", encoding="utf-8") as schema_handle:
        for _dataset, example in examples:
            dump_jsonl(schema_handle, example)
            per_example = args.output_dir / "schemas" / str(example.get("dataset") or _dataset)
            per_example.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(str(example.get("example_id") or "example")) + ".json"
            (per_example / filename).write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    prepare_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.frame_workers)) as pool:
        futures = [
            pool.submit(
                prepare_example,
                dataset,
                example,
                input_mode=args.input_mode,
                retrieval_topk=args.retrieval_topk,
                retrieval_strategy=args.retrieval_strategy,
                video_fps=args.video_fps,
                video_max_frames_per_clip=args.video_max_frames_per_clip,
                videomme_observation_end_s=args.videomme_observation_end_s,
            )
            for dataset, example in examples
        ]
        prepared = [future.result() for future in concurrent.futures.as_completed(futures)]
    prepare_total_s = time.perf_counter() - prepare_started
    prepared.sort(key=lambda item: (item.dataset, str(item.example.get("example_id") or "")))
    print(f"prepared={len(prepared)} prepare_total_s={prepare_total_s:.2f}", flush=True)

    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as records_handle:
        for item in prepared:
            example = item.example
            example_id = example.get("example_id")
            gold = gold_label(example)
            options = question_options(example)
            base_record = {
                "dataset": item.dataset,
                "example_id": example_id,
                "question_id": (example.get("question") or {}).get("question_id"),
                "video_id": (example.get("video") or {}).get("video_id"),
                "task_family": example.get("task_family"),
                "model": args.model,
                "input_mode": args.input_mode,
                "retrieval_strategy": args.retrieval_strategy,
                "retrieval_topk": args.retrieval_topk,
                "visible_until_s": item.visible_until_s,
                "retrieved_memory": item.retrieved_memory,
                "media_records": item.media_records,
                "gold_label": gold,
                "gold_text": ((example.get("question") or {}).get("answer") or {}).get("text"),
            }
            if item.prepare_error:
                record = {**base_record, "ok": False, "error": item.prepare_error, "timing_s": {"load": load_s}}
                records.append(record)
                dump_jsonl(records_handle, record)
                print(json.dumps({"example_id": example_id, "ok": False, "error": item.prepare_error}, ensure_ascii=False), flush=True)
                continue

            prompt = build_prompt(
                example,
                retrieved_memory=item.retrieved_memory,
                visible_until_s=item.visible_until_s,
                answer_mode=args.answer_mode,
            )
            start = time.perf_counter()
            try:
                response = generate_one(
                    model,
                    processor,
                    images=item.images,
                    media_records=item.media_records,
                    input_mode=args.input_mode,
                    prompt_text=prompt,
                    max_new_tokens=args.max_new_tokens,
                    enable_thinking=args.enable_thinking,
                )
                generate_s = time.perf_counter() - start
                pred = parse_answer_label(response, options)
                record = {
                    **base_record,
                    "ok": True,
                    "prompt": prompt,
                    "response": response,
                    "prediction_label": pred,
                    "evidence_summary": parse_evidence_summary(response) if args.answer_mode == "json_rationale" else None,
                    "correct": bool(pred and gold and pred == gold),
                    "timing_s": {"load": load_s, "prepare_total": prepare_total_s, "generate": generate_s},
                }
            except Exception as exc:
                generate_s = time.perf_counter() - start
                record = {
                    **base_record,
                    "ok": False,
                    "prompt": prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing_s": {"load": load_s, "prepare_total": prepare_total_s, "generate": generate_s},
                }
            records.append(record)
            dump_jsonl(records_handle, record)
            print(
                json.dumps(
                    {
                        "dataset": item.dataset,
                        "example_id": example_id,
                        "ok": record.get("ok"),
                        "gold": gold,
                        "pred": record.get("prediction_label"),
                        "correct": record.get("correct"),
                        "generate_s": record.get("timing_s", {}).get("generate"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = metric_summary(records)
    summary["run"] = {
        "schema_path": str(schema_path),
        "records_path": str(records_path),
        "metrics_path": str(metrics_path),
        "total_wall_s": time.perf_counter() - t0,
        "model_load_s": load_s,
        "frame_prepare_total_s": prepare_total_s,
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
