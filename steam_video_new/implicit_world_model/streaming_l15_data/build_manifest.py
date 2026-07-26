"""Build split-safe StreamingBench + OVO-Bench manifests for L1/L1.5 gathering."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "steam-streaming-l15-sft-selection/v0.1"
HIDDEN_SCHEMA_VERSION = "steam-streaming-l15-sft-hidden-key/v0.1"
ROLES = ("sft_train", "posttrain_pool", "validation", "heldout_test")


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _role_for_video(video_key: str, *, seed: str) -> str:
    bucket = int(_digest(f"{seed}:{video_key}")[:8], 16) % 100
    if bucket < 70:
        return "sft_train"
    if bucket < 85:
        return "posttrain_pool"
    if bucket < 95:
        return "validation"
    return "heldout_test"


def _parse_timestamp_s(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        nums = [float(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        return nums[0] * 3600 + nums[1] * 60 + nums[2]
    if len(nums) == 2:
        return nums[0] * 60 + nums[1]
    return None


def _normalize_options(value: Any) -> list[dict[str, str]]:
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return []
    if isinstance(value, dict):
        return [{"label": str(label), "text": str(text)} for label, text in value.items()]
    if not isinstance(value, list):
        return []
    out = []
    for index, option in enumerate(value):
        text = str(option)
        label = chr(ord("A") + index)
        if len(text) >= 3 and text[0].isalpha() and text[1:3] in {". ", ") "}:
            label = text[0].upper()
            text = text[3:].strip()
        out.append({"label": label, "text": text})
    return out


def _duration_s(path: Path) -> float | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    try:
        value = float(result.stdout.strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _opencv_duration_s(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
        if fps <= 0 or frames <= 0:
            return None
        return frames / fps
    finally:
        capture.release()


def _probe_duration_s(path: Path) -> float | None:
    return _duration_s(path) or _opencv_duration_s(path)


def _range_folder(prefix: str, sample_index: int, step: int) -> str:
    start = ((sample_index - 1) // step) * step + 1
    end = start + step - 1
    return f"{prefix}_{start}-{end}"


def _streaming_video_ref(dataset_root: Path, row: dict[str, str], csv_name: str) -> str | None:
    qid = row.get("question_id") or ""
    sample_match = re.search(r"sample_(\d+)", qid)
    if sample_match:
        sample_index = int(sample_match.group(1))
    else:
        tail = re.search(r"_(\d+)$", qid)
        sample_index = int(tail.group(1)) if tail else 0
    if sample_index <= 0:
        return None
    task_type = row.get("task_type") or ""
    if csv_name.startswith("Real_Time_Visual_Understanding"):
        folder = _range_folder("Real-Time Visual Understanding", sample_index, 50)
    elif csv_name.startswith("Sequential_Question_Answering"):
        folder = _range_folder("Sequential Question Answering", sample_index, 25)
    elif csv_name.startswith("Proactive_Output"):
        folder = _range_folder("Proactive Output", sample_index, 25)
    elif task_type == "Misleading Context Recognition":
        folder = "Misleading Context Understanding"
    elif task_type == "Scene Understanding":
        folder = _range_folder("Scene Understanding", sample_index, 25)
    else:
        folder = task_type
    ref = Path("StreamingBench") / "extracted" / folder / f"sample_{sample_index}" / "video.mp4"
    return str(ref) if (dataset_root / ref).is_file() else None


def _load_streamingbench(dataset_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    ann_root = dataset_root / "StreamingBench" / "StreamingBench"
    videos: dict[str, dict[str, Any]] = {}
    hidden: list[dict[str, Any]] = []
    for csv_path in sorted(ann_root.glob("*.csv")):
        if csv_path.name.endswith(".official.csv"):
            continue
        with csv_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                ref = _streaming_video_ref(dataset_root, row, csv_path.name)
                if ref is None:
                    continue
                video_id = f"streamingbench:{Path(ref).parent.name}:{Path(ref).parent.parent.name}"
                query_time_s = _parse_timestamp_s(row.get("time_stamp"))
                gt_time_s = _parse_timestamp_s(row.get("ground_truth_time_stamp"))
                key = f"streaming_bench:{video_id}"
                videos.setdefault(
                    key,
                    {
                        "dataset": "streaming_bench",
                        "video_id": video_id,
                        "video_ref": ref,
                        "source_video_group": video_id,
                        "benchmark_tasks": set(),
                        "query_times": [],
                    },
                )
                videos[key]["benchmark_tasks"].add(row.get("task_type") or csv_path.stem)
                if query_time_s is not None:
                    videos[key]["query_times"].append(query_time_s)
                if gt_time_s is not None:
                    videos[key]["query_times"].append(gt_time_s)
                hidden.append(
                    {
                        "dataset": "streaming_bench",
                        "video_key": key,
                        "question_id": row.get("question_id"),
                        "task_type": row.get("task_type"),
                        "question": row.get("question"),
                        "options": _normalize_options(row.get("options")),
                        "answer": row.get("answer") or row.get("ground_truth_output"),
                        "query_time_s": query_time_s,
                        "ground_truth_time_s": gt_time_s,
                        "temporal_clue_type": row.get("temporal_clue_type"),
                        "frames_required": row.get("frames_required"),
                        "source_annotation": csv_path.name,
                    }
                )
    return videos, hidden


def _load_ovo(dataset_root: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    qa_path = dataset_root / "OVO-Bench" / "data" / "ovo_bench_new.json"
    records = json.loads(qa_path.read_text(encoding="utf-8"))
    videos: dict[str, dict[str, Any]] = {}
    hidden: list[dict[str, Any]] = []
    for row in records:
        qid = str(row.get("id"))
        ref = Path("OVO-Bench") / "data" / "chunked_videos" / f"{qid}.mp4"
        if not (dataset_root / ref).is_file():
            continue
        video_id = f"ovo:{qid}"
        key = f"ovo_bench:{video_id}"
        realtime_s = float(row["realtime"]) if row.get("realtime") is not None else None
        videos[key] = {
            "dataset": "ovo_bench",
            "video_id": video_id,
            "video_ref": str(ref),
            "source_video_group": str(row.get("video") or video_id),
            "benchmark_tasks": {str(row.get("task") or "ovo")},
            "query_times": [realtime_s] if realtime_s is not None else [],
        }
        hidden.append(
            {
                "dataset": "ovo_bench",
                "video_key": key,
                "question_id": qid,
                "task_type": row.get("task"),
                "question": row.get("question"),
                "options": _normalize_options(row.get("options")),
                "answer": row.get("answer"),
                "gt": row.get("gt"),
                "query_time_s": realtime_s,
                "source_video": row.get("video"),
                "source_annotation": "ovo_bench_new.json",
            }
        )
    return videos, hidden


def _balanced_take(videos: list[dict[str, Any]], limit: int | None, seed: str) -> list[dict[str, Any]]:
    if limit is None or len(videos) <= limit:
        return videos
    by_dataset_role: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for video in videos:
        by_dataset_role[(video["dataset"], video["split"])].append(video)
    selected: list[dict[str, Any]] = []
    strata = sorted(by_dataset_role)
    per_round = {key: sorted(rows, key=lambda row: _digest(f"{seed}:take:{row['video_key']}")) for key, rows in by_dataset_role.items()}
    while len(selected) < limit and any(per_round.values()):
        for key in strata:
            rows = per_round[key]
            if rows and len(selected) < limit:
                selected.append(rows.pop(0))
    return sorted(selected, key=lambda row: (row["dataset"], row["split"], row["video_id"]))


def build_manifest(
    *,
    dataset_root: Path,
    output_dir: Path,
    seed: str,
    max_videos: int | None,
    horizon_policy: str,
) -> dict[str, Any]:
    streaming_videos, streaming_hidden = _load_streamingbench(dataset_root)
    ovo_videos, ovo_hidden = _load_ovo(dataset_root)
    raw_videos = {**streaming_videos, **ovo_videos}
    hidden_rows = streaming_hidden + ovo_hidden
    candidates = []
    for key, row in raw_videos.items():
        role = _role_for_video(key, seed=seed)
        candidates.append(
            {
                "video_key": key,
                "dataset": row["dataset"],
                "video_id": row["video_id"],
                "video_ref": row["video_ref"],
                "source_video_group": row["source_video_group"],
                "split": role,
                "horizon_policy": horizon_policy,
                "qa_count": sum(1 for item in hidden_rows if item["video_key"] == key),
                "benchmark_tasks": sorted(row["benchmark_tasks"]),
                "_query_horizon_s": max(row["query_times"] or [0.0]),
            }
        )
    selected_candidates = _balanced_take(candidates, max_videos, seed)
    videos = []
    for row in selected_candidates:
        duration = _probe_duration_s(dataset_root / row["video_ref"])
        if duration is None:
            continue
        query_horizon = float(row.pop("_query_horizon_s", 0.0))
        if horizon_policy == "query_prefix":
            horizon = min(duration, max(1.0, query_horizon))
        else:
            horizon = duration
        videos.append(
            {
                **row,
                "duration_s": round(duration, 3),
                "observation_horizon_s": round(horizon, 3),
            }
        )
    selected_keys = {row["video_key"] for row in videos}
    hidden_rows = [row for row in hidden_rows if row["video_key"] in selected_keys]
    public_videos = [{key: value for key, value in row.items() if key != "video_key"} for row in videos]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": "streamingbench_ovo_l1_l15_sft_v1",
        "seed": seed,
        "dataset_root": str(dataset_root),
        "selection_uses_question_or_gt": False,
        "hidden_key_required_for_sft_targets": True,
        "roles": list(ROLES),
        "horizon_policy": horizon_policy,
        "videos": public_videos,
        "counts": {
            "videos": len(public_videos),
            "hidden_qa_rows": len(hidden_rows),
            "by_dataset": {
                dataset: sum(1 for row in public_videos if row["dataset"] == dataset)
                for dataset in ("streaming_bench", "ovo_bench")
            },
            "by_role": {
                role: sum(1 for row in public_videos if row["split"] == role)
                for role in ROLES
            },
        },
        "forbidden_worker_inputs": ["question", "choices", "answer", "answer_key", "answer_text", "clue_intervals"],
        "training_performed": False,
    }
    hidden = {
        "schema_version": HIDDEN_SCHEMA_VERSION,
        "warning": "Contains questions, options, answers, and query/proactive timestamps. Never feed to L1/L1.5 graph extraction.",
        "dataset_id": summary["dataset_id"],
        "seed": seed,
        "qas": hidden_rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.public.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "targets.hidden_key.json").write_text(json.dumps(hidden, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("/fs/gamma-projects/vlm-robot/datasets"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", default="steam-streaming-l15-sft-v1")
    parser.add_argument("--max-videos", type=int, default=96)
    parser.add_argument("--horizon-policy", choices=["full_video", "query_prefix"], default="full_video")
    args = parser.parse_args()
    summary = build_manifest(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        seed=args.seed,
        max_videos=args.max_videos,
        horizon_policy=args.horizon_policy,
    )
    print(json.dumps({key: summary[key] for key in ("schema_version", "dataset_id", "counts", "horizon_policy")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
