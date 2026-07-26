#!/usr/bin/env python3
"""Evaluate Dispider as an end-to-end streaming VideoLLM baseline.

This runner keeps the same records/metrics shape as qwen35_streaming_eval.py,
but delegates generation to the official Dispider inference wrapper. Because
Dispider accepts a video file path rather than (path, start, end) media records,
the runner materializes a visible video prefix per example under output_dir.
That preserves the streaming no-future-leak boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import shutil
from pathlib import Path
from typing import Any

DEFAULT_DATASETS = ("ovo_bench", "videomme")
SUPPORTED_DATASETS = ("ovo_bench", "videomme", "streaming_bench")


def ensure_repo_on_path(repo_root: str) -> None:
    repo = str(Path(repo_root).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)


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


def build_wrapper_config(dataset: str, args: argparse.Namespace) -> Any:
    from dataset_clip_wrapper.dataset_graph_presets import apply_profile_defaults, clip_policy_for, retrieval_for
    from dataset_clip_wrapper.schemas import BenchmarkProfile, BackboneConfig, RuntimeMode, VideoRegime, WrapperConfig

    regime = VideoRegime.STREAMING
    profile = BenchmarkProfile.DEFAULT
    clip_policy = clip_policy_for(dataset, regime)
    retrieval = retrieval_for(regime)
    if dataset == "videomme":
        clip_policy.observation_end_s = args.videomme_observation_end_s
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
    datasets = sorted({row.get("dataset") for row in records if row.get("dataset")})
    for key, rows in [("overall", records)] + [(dataset, [row for row in records if row.get("dataset") == dataset]) for dataset in datasets]:
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


def build_dispider_prompt(example: dict[str, Any], visible_until_s: float | None) -> str:
    question = example.get("question") or {}
    options = _question_options(example)
    lines = [
        "Answer this streaming video multiple-choice question using only the visible video.",
        "Output exactly one option label, such as A, B, C, or D. Do not explain.",
    ]
    if visible_until_s is not None:
        lines.append(f"The video has been clipped to the visible prefix ending at {visible_until_s:.2f} seconds.")
    lines.append(f"Question: {question.get('question_text') or ''}")
    if options:
        lines.append("Options:")
        for option in options:
            lines.append(f"{option['label']}. {option['text']}")
    lines.append("Final answer:")
    return "\n".join(lines)


def _source_video_path(example: dict[str, Any]) -> str | None:
    video = example.get("video") or {}
    for key in ("path", "video_path", "source_path"):
        value = video.get(key)
        if value:
            return str(value)
    for clip in video.get("derived_clips") or []:
        value = clip.get("path")
        if value:
            return str(value)
    return None


def _video_duration_s(example: dict[str, Any]) -> float | None:
    value = (example.get("video") or {}).get("duration_s")
    return float(value) if isinstance(value, (int, float)) else None


def materialize_visible_prefix(
    *,
    source_path: str,
    output_path: Path,
    visible_until_s: float | None,
    duration_s: float | None,
    ffmpeg_bin: str,
    overwrite: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path

    limit_s = visible_until_s if visible_until_s is not None else duration_s
    if limit_s is None or (duration_s is not None and abs(limit_s - duration_s) < 1e-3):
        # Symlink full videos when no prefix clipping is needed.
        if output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        output_path.symlink_to(Path(source_path).resolve())
        return output_path

    ffmpeg_executable = ffmpeg_bin
    if shutil.which(ffmpeg_executable) is None:
        try:
            import imageio_ffmpeg

            ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            pass

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp.mp4")
    if tmp_path.exists():
        tmp_path.unlink()
    cmd = [
        ffmpeg_executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        source_path,
        "-t",
        f"{max(0.1, float(limit_s)):.3f}",
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        str(tmp_path),
    ]
    subprocess.run(cmd, check=True)
    tmp_path.replace(output_path)
    return output_path


def load_dispider(dispider_repo: str, model_path: str) -> tuple[Any, float]:
    repo = str(Path(dispider_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    start = time.perf_counter()
    from inference import videoStream  # type: ignore

    return videoStream(model_path), time.perf_counter() - start


def write_run_config(args: argparse.Namespace, output_dir: Path) -> None:
    payload = {
        "runner": "dispider_streaming_eval.py",
        "baseline": "Dispider end-to-end streaming VideoLLM",
        "datasets": list(args.datasets),
        "model": args.model,
        "dispider_repo": args.dispider_repo,
        "dataset_root": args.dataset_root,
        "split": args.split,
        "limit_per_dataset": args.limit_per_dataset,
        "shard": {"shard_index": args.shard_index, "num_shards": args.num_shards},
        "streaming_definition": {
            "no_future_leakage": "Each example is evaluated on a materialized visible prefix video.",
            "ovo_bench": "Use question.time_anchor_s when present, clipped to probed duration.",
            "videomme": "Use --videomme-observation-end-s as visibility cutoff.",
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
    parser.add_argument("--model", required=True, help="Path to Dispider checkpoint.")
    parser.add_argument("--dispider-repo", required=True, help="Path to cloned Mark12Ding/Dispider repository.")
    parser.add_argument("--repo-root", default="/home/xwu/atomic_skills_for_video")
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), choices=list(SUPPORTED_DATASETS))
    parser.add_argument("--limit-per-dataset", type=int, default=5)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--overwrite-media-cache", action="store_true")
    args = parser.parse_args()

    if args.limit_per_dataset is not None and args.limit_per_dataset < 0:
        args.limit_per_dataset = None
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    ensure_repo_on_path(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    media_cache = args.output_dir / "visible_prefix_videos"
    records_path = args.output_dir / "records.jsonl"
    metrics_path = args.output_dir / "metrics_summary.json"
    schema_path = args.output_dir / "canonical_schemas.jsonl"
    write_run_config(args, args.output_dir)

    t0 = time.perf_counter()
    model, load_s = load_dispider(args.dispider_repo, args.model)
    examples = iter_examples(args)

    with schema_path.open("w", encoding="utf-8") as schema_handle:
        for _dataset, example in examples:
            _json_dump_line(schema_handle, example)

    records: list[dict[str, Any]] = []
    with records_path.open("w", encoding="utf-8") as records_handle:
        for dataset, example in examples:
            example_id = str(example.get("example_id") or "example")
            gold = _question_answer_label(example)
            options = _question_options(example)
            visible_until_s = _streaming_visible_until(
                example,
                videomme_observation_end_s=args.videomme_observation_end_s,
            )
            source_path = _source_video_path(example)
            base_record = {
                "dataset": dataset,
                "example_id": example.get("example_id"),
                "question_id": (example.get("question") or {}).get("question_id"),
                "video_id": (example.get("video") or {}).get("video_id"),
                "task_family": example.get("task_family"),
                "model": args.model,
                "baseline": "dispider",
                "input_mode": "visible_prefix_video",
                "visible_until_s": visible_until_s,
                "gold_label": gold,
                "gold_text": ((example.get("question") or {}).get("answer") or {}).get("text"),
            }
            if not source_path:
                record = {**base_record, "ok": False, "error": "source video path not found", "timing_s": {"load": load_s}}
                records.append(record)
                _json_dump_line(records_handle, record)
                continue

            prompt = build_dispider_prompt(example, visible_until_s)
            video_path = media_cache / dataset / f"{_safe_filename(example_id)}.mp4"
            start = time.perf_counter()
            try:
                visible_video = materialize_visible_prefix(
                    source_path=source_path,
                    output_path=video_path,
                    visible_until_s=visible_until_s,
                    duration_s=_video_duration_s(example),
                    ffmpeg_bin=args.ffmpeg_bin,
                    overwrite=args.overwrite_media_cache,
                )
                prepare_s = time.perf_counter() - start
                gen_start = time.perf_counter()
                response = model.Run(str(visible_video), prompt)
                generate_s = time.perf_counter() - gen_start
                pred = parse_answer_label(response, options)
                record = {
                    **base_record,
                    "ok": True,
                    "prompt": prompt,
                    "visible_video": str(visible_video),
                    "source_video": source_path,
                    "response": response,
                    "prediction_label": pred,
                    "evidence_summary": None,
                    "correct": bool(pred and gold and pred == gold),
                    "timing_s": {"load": load_s, "prepare": prepare_s, "generate": generate_s},
                }
            except Exception as exc:
                record = {
                    **base_record,
                    "ok": False,
                    "prompt": prompt,
                    "source_video": source_path,
                    "error": f"{type(exc).__name__}: {exc}",
                    "timing_s": {"load": load_s, "prepare_or_generate": time.perf_counter() - start},
                }
            records.append(record)
            _json_dump_line(records_handle, record)
            print(
                json.dumps(
                    {
                        "dataset": dataset,
                        "example_id": example_id,
                        "ok": record.get("ok"),
                        "gold": gold,
                        "pred": record.get("prediction_label"),
                        "correct": record.get("correct"),
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
    }
    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
