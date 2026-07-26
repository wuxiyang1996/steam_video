#!/usr/bin/env python3
"""Evaluate local Qwen3.5-9B on streaming-style OVO-Bench and VideoMME.

The model runs sequentially on one GPU. Video frame extraction is parallelized so
the GPU is not blocked on OpenCV decode more than necessary.
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

from PIL import Image


DEFAULT_DATASETS = ("ovo_bench", "videomme")
SUPPORTED_DATASETS = ("ovo_bench", "videomme", "streaming_bench")


def ensure_repo_on_path(repo_root: str) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


@dataclass
class PreparedExample:
    dataset: str
    example: dict[str, Any]
    images: list[Image.Image]
    media_records: list[dict[str, Any]]
    visible_until_s: float | None
    prepare_error: str | None = None


def _json_dump_line(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "example"


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


def _streaming_visible_until(example: dict[str, Any], *, videomme_observation_end_s: float | None) -> float | None:
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


def _visible_clips(example: dict[str, Any], visible_until_s: float | None) -> list[dict[str, Any]]:
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


def _pick_sampling_points(clips: list[dict[str, Any]], frames_per_example: int) -> list[tuple[dict[str, Any], float]]:
    if not clips or frames_per_example <= 0:
        return []
    if len(clips) <= frames_per_example:
        selected = clips
    else:
        if frames_per_example == 1:
            selected = [clips[len(clips) // 2]]
        else:
            selected = []
            last = len(clips) - 1
            for i in range(frames_per_example):
                selected.append(clips[round(i * last / (frames_per_example - 1))])
    points = []
    for clip in selected:
        span = clip.get("source_span") or {}
        start_s = float(span.get("start_s", 0.0))
        end_s = float(span.get("end_s", start_s))
        points.append((clip, max(0.0, (start_s + end_s) / 2.0)))
    return points


def _pick_clips(clips: list[dict[str, Any]], clips_per_example: int) -> list[dict[str, Any]]:
    if not clips or clips_per_example <= 0:
        return []
    if len(clips) <= clips_per_example:
        return clips
    if clips_per_example == 1:
        return [clips[len(clips) // 2]]
    last = len(clips) - 1
    return [clips[round(i * last / (clips_per_example - 1))] for i in range(clips_per_example)]


def _read_frame(video_path: str, timestamp_s: float) -> Image.Image:
    import cv2  # type: ignore

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
    frames_per_example: int,
    clips_per_example: int,
    video_fps: float,
    video_max_frames_per_clip: int,
    videomme_observation_end_s: float | None,
) -> PreparedExample:
    try:
        visible_until_s = _streaming_visible_until(example, videomme_observation_end_s=videomme_observation_end_s)
        clips = _visible_clips(example, visible_until_s)
        if input_mode == "video_clip":
            selected = _pick_clips(clips, clips_per_example)
            if not selected:
                raise RuntimeError("no visible clips available for video_clip input")
            media_records = []
            for clip in selected:
                span = clip.get("source_span") or {}
                start_s = float(span.get("start_s", 0.0))
                end_s = float(span.get("end_s", start_s))
                if end_s <= start_s:
                    continue
                media_records.append(
                    {
                        "media_type": "video_clip",
                        "clip_id": clip.get("clip_id"),
                        "path": clip.get("path"),
                        "video_start": start_s,
                        "video_end": end_s,
                        "source_span": {"start_s": start_s, "end_s": end_s},
                        "granularity": clip.get("granularity"),
                        "fps": video_fps,
                        "max_frames": video_max_frames_per_clip,
                    }
                )
            if not media_records:
                raise RuntimeError("no valid visible video clips after clipping")
            return PreparedExample(dataset, example, [], media_records, visible_until_s)

        points = _pick_sampling_points(clips, frames_per_example)
        if not points:
            raise RuntimeError("no visible clips available for frame sampling")
        images: list[Image.Image] = []
        media_records: list[dict[str, Any]] = []
        for clip, timestamp_s in points:
            image = _read_frame(str(clip["path"]), timestamp_s)
            images.append(image)
            media_records.append(
                {
                    "media_type": "frame",
                    "clip_id": clip.get("clip_id"),
                    "path": clip.get("path"),
                    "timestamp_s": timestamp_s,
                    "source_span": clip.get("source_span"),
                    "granularity": clip.get("granularity"),
                }
            )
        return PreparedExample(dataset, example, images, media_records, visible_until_s)
    except Exception as exc:
        return PreparedExample(dataset, example, [], [], None, f"{type(exc).__name__}: {exc}")


def build_prompt(
    example: dict[str, Any],
    media_count: int,
    visible_until_s: float | None,
    input_mode: str,
    answer_mode: str,
) -> str:
    question = example.get("question") or {}
    options = _question_options(example)
    media_name = "sampled frames" if input_mode == "frames" else "video clips"
    lines = [
        "Answer this streaming video multiple-choice question.",
        f"Use only the provided {media_name}, which are from the visible part of the video.",
    ]
    if answer_mode == "json_rationale":
        lines.extend(
            [
                "Output valid JSON only.",
                'Required schema: {"answer_label": "A|B|C|D", "evidence_summary": "one short grounded sentence"}.',
                "Keep evidence_summary concise and grounded in the visible media; do not write a step-by-step chain of thought.",
            ]
        )
    else:
        lines.append("Output exactly one option label, such as A, B, C, or D. Do not explain.")
    if visible_until_s is not None:
        lines.append(f"Visible video cutoff: {visible_until_s:.2f} seconds.")
    lines.append(f"Number of provided {media_name}: {media_count}.")
    lines.append(f"Question: {question.get('question_text') or ''}")
    if options:
        lines.append("Options:")
        for option in options:
            lines.append(f"{option['label']}. {option['text']}")
    lines.append("Final answer:")
    return "\n".join(lines)


def build_messages(
    *,
    images: list[Image.Image],
    media_records: list[dict[str, Any]],
    text: str,
    input_mode: str,
) -> list[dict[str, Any]]:
    if input_mode == "video_clip":
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
    else:
        content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": text})
    return [{"role": "user", "content": content}]


def generate_one(
    model: Any,
    processor: Any,
    *,
    images: list[Image.Image],
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
        (dataset, [row for row in records if row.get("dataset") == dataset]) for dataset in sorted({row.get("dataset") for row in records if row.get("dataset")})
    ]:
        total = len(rows)
        ok_rows = [row for row in rows if row.get("ok")]
        parsed = [row for row in ok_rows if row.get("prediction_label")]
        correct = [row for row in ok_rows if row.get("correct") is True]
        latencies = [float(row["timing_s"]["generate"]) for row in ok_rows if row.get("timing_s", {}).get("generate") is not None]
        payload = {
            "total": total,
            "ok": len(ok_rows),
            "failed": total - len(ok_rows),
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
        "runner": "qwen35_streaming_eval.py",
        "datasets": list(args.datasets),
        "model": args.model,
        "dataset_root": args.dataset_root,
        "split": args.split,
        "limit_per_dataset": args.limit_per_dataset,
        "shard": {
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
        },
        "streaming_definition": {
            "ovo_bench": "Use question.time_anchor_s when present, clipped to probed video duration.",
            "videomme": "Force wrapper regime=streaming and use --videomme-observation-end-s as visibility cutoff.",
            "input_mode": args.input_mode,
            "frames_per_example": args.frames_per_example,
            "clips_per_example": args.clips_per_example,
            "video_fps": args.video_fps,
            "video_max_frames_per_clip": args.video_max_frames_per_clip,
            "answer_mode": args.answer_mode,
            "enable_thinking": args.enable_thinking,
        },
        "records_schema": {
            "dataset": "dataset id",
            "example_id": "canonical wrapper example id",
            "visible_until_s": "streaming observation cutoff",
            "media_records": "sampled frame or direct video-clip provenance",
            "gold_label": "gold multiple-choice label from wrapper",
            "correct": "gold_label == prediction_label",
            "response": "raw model response",
            "prediction_label": "parsed single-letter answer label",
            "evidence_summary": "optional short grounded rationale/evidence sentence",
            "timing_s": "prepare/generate timings",
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
    parser.add_argument("--repo-root", default="/home/xwu/atomic_skills_for_video")
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--input-mode", default="frames", choices=["frames", "video_clip"])
    parser.add_argument("--frames-per-example", type=int, default=1)
    parser.add_argument("--clips-per-example", type=int, default=1)
    parser.add_argument("--video-fps", type=float, default=2.0)
    parser.add_argument("--video-max-frames-per-clip", type=int, default=8)
    parser.add_argument("--frame-workers", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--answer-mode", default="label_only", choices=["label_only", "json_rationale"])
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--window-s", type=float, default=None)
    parser.add_argument("--overlap-s", type=float, default=None)
    args = parser.parse_args()
    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    ensure_repo_on_path(args.repo_root)

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    args.output_dir.mkdir(parents=True, exist_ok=True)
    schema_path = args.output_dir / "canonical_schemas.jsonl"
    records_path = args.output_dir / "records.jsonl"
    metrics_path = args.output_dir / "metrics_summary.json"
    write_run_config(args, args.output_dir)

    print(f"started_at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}", flush=True)

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    load_s = time.perf_counter() - t0
    print(f"model_loaded_s={load_s:.2f} device={model.device}", flush=True)
    print(f"datasets={','.join(args.datasets)} shard={args.shard_index}/{args.num_shards}", flush=True)

    examples = iter_examples(args)
    print(f"canonical_examples={len(examples)}", flush=True)
    with schema_path.open("w", encoding="utf-8") as schema_handle:
        for _dataset, example in examples:
            _json_dump_line(schema_handle, example)
            per_example = args.output_dir / "schemas" / _dataset
            per_example.mkdir(parents=True, exist_ok=True)
            filename = _safe_filename(str(example.get("example_id") or "example")) + ".json"
            (per_example / filename).write_text(json.dumps(example, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    records: list[dict[str, Any]] = []
    prepare_started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.frame_workers)) as pool:
        futures = [
            pool.submit(
                prepare_example,
                dataset,
                example,
                input_mode=args.input_mode,
                frames_per_example=args.frames_per_example,
                clips_per_example=args.clips_per_example,
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

    with records_path.open("w", encoding="utf-8") as records_handle:
        for item in prepared:
            example = item.example
            example_id = example.get("example_id")
            gold = _question_answer_label(example)
            options = _question_options(example)
            base_record = {
                "dataset": item.dataset,
                "example_id": example_id,
                "question_id": (example.get("question") or {}).get("question_id"),
                "video_id": (example.get("video") or {}).get("video_id"),
                "task_family": example.get("task_family"),
                "model": args.model,
                "input_mode": args.input_mode,
                "visible_until_s": item.visible_until_s,
                "media_records": item.media_records,
                "gold_label": gold,
                "gold_text": ((example.get("question") or {}).get("answer") or {}).get("text"),
            }
            if item.prepare_error:
                record = {**base_record, "ok": False, "error": item.prepare_error, "timing_s": {"load": load_s}}
                records.append(record)
                _json_dump_line(records_handle, record)
                print(json.dumps({"example_id": example_id, "ok": False, "error": item.prepare_error}, ensure_ascii=False), flush=True)
                continue

            prompt = build_prompt(example, len(item.media_records), item.visible_until_s, args.input_mode, args.answer_mode)
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
                    "timing_s": {"load": load_s, "generate": generate_s},
                }
            except Exception as exc:
                generate_s = time.perf_counter() - start
                record = {
                    **base_record,
                    "ok": False,
                    "prompt": prompt,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing_s": {"load": load_s, "generate": generate_s},
                }
            records.append(record)
            _json_dump_line(records_handle, record)
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
