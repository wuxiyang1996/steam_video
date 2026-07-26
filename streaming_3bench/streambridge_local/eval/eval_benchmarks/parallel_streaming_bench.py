#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Evaluate Qwen3.5-VL on StreamingBench with StreamBridge's causal loop."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))
from eval.streaming_models.online_qwen35_vl import Qwen35_VL


def result_path(filename: str) -> Path:
    result_dir = Path(os.environ.get("RESULT_DIR", "eval/results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir / filename


def parse_label(response: str, labels: set[str]) -> str | None:
    if not response:
        return None
    text = response.strip()
    if text.upper() in labels:
        return text.upper()
    for pattern in [
        r'"answer_label"\s*:\s*"([A-Z])"',
        r'"answer"\s*:\s*"([A-Z])"',
        r"\banswer(?:_label)?\s*[:=]\s*([A-Z])\b",
        r"\boption\s+([A-Z])\b",
        r"^\s*([A-Z])[\).:\s]",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match and match.group(1).upper() in labels:
            return match.group(1).upper()
    for label in sorted(labels):
        if f"{label}." in text or f"{label})" in text:
            return label
    return None


def build_streaming_bench_data() -> list[dict]:
    atomic_repo = Path(os.environ.get("ATOMIC_REPO", "/home/xwu/atomic_skills_for_video"))
    if str(atomic_repo) not in sys.path:
        sys.path.insert(0, str(atomic_repo))

    from dataset_clip_wrapper.adapters import get_adapter

    dataset_root = Path(os.environ.get("DATASET_ROOT", "/mnt/is_data/xwu/video_skills/data/datasets"))
    adapter = get_adapter("streaming_bench", dataset_root, split=os.environ.get("SPLIT", "train"))
    records: list[dict] = []
    for item in adapter.iter_items():
        question = item.question or {}
        options = question.get("options") or []
        answer = question.get("answer") or {}
        if not item.video_path or len(options) < 2 or not answer.get("label"):
            continue
        timestamp_s = question.get("time_anchor_s")
        if not isinstance(timestamp_s, (int, float)):
            continue
        labels = [str(option["label"]).strip().upper() for option in options]
        option_lines = [f"{label}. {option['text']}" for label, option in zip(labels, options)]
        prompt = (
            f"Question: {question.get('question_text') or ''}\n"
            "Options:\n"
            + "\n".join(option_lines)
            + "\nPlease respond with only the letter of the correct answer."
        )
        records.append(
            {
                "example_id": item.example_id,
                "question_id": question.get("question_id"),
                "video": str(item.video_path),
                "video_id": item.video_id,
                "time_s": float(timestamp_s),
                "type": question.get("question_type"),
                "question": prompt,
                "raw_question": question.get("question_text"),
                "options": options,
                "answer": str(answer.get("label")).strip().upper(),
                "answer_text": answer.get("text"),
                "annotation_path": item.metadata.get("annotation_path"),
            }
        )
    return records


def save_json(records: list[dict], path: Path) -> None:
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    if os.environ.get("MODEL") != "qwen35vl":
        raise ValueError("parallel_streaming_bench.py currently supports MODEL=qwen35vl")

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    ckpt = os.environ["CKPT"]
    sampled_fps = 1.0

    data = build_streaming_bench_data()
    if "STREAMBRIDGE_LIMIT_RECORDS" in os.environ:
        data = data[: int(os.environ["STREAMBRIDGE_LIMIT_RECORDS"])]
    if "STREAMBRIDGE_NUM_SHARDS" in os.environ:
        num_shards = int(os.environ["STREAMBRIDGE_NUM_SHARDS"])
        shard_index = int(os.environ.get("STREAMBRIDGE_SHARD_INDEX", 0))
        if not 0 <= shard_index < num_shards:
            raise ValueError("STREAMBRIDGE_SHARD_INDEX must be in [0, STREAMBRIDGE_NUM_SHARDS)")
        data = [item for index, item in enumerate(data) if index % num_shards == shard_index]

    model = Qwen35_VL(
        ckpt=ckpt,
        video_path="",
        dtype=torch.bfloat16,
        device_map={"": device},
        stream_fps=sampled_fps,
        clip_window_s=float(os.environ.get("CLIP_WINDOW_S", 4.0)),
        video_fps=float(os.environ.get("QWEN35_VIDEO_FPS", 2.0)),
        video_max_frames_per_clip=int(os.environ.get("QWEN35_VIDEO_MAX_FRAMES_PER_CLIP", 8)),
    )
    model.eval()

    records: list[dict] = []
    generate_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "min_length": 1,
        "num_return_sequences": 1,
        "max_new_tokens": 128,
        "temperature": None,
        "top_p": None,
        "top_k": None,
    }

    for item in tqdm(data, desc="StreamingBench"):
        model.video_path = item["video"]
        model.reset()
        try:
            model.receive_one_frame(timestamp_s=item["time_s"])
            model.receive_user_input(item["question"])
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=model.dtype):
                output_text = model.response(**generate_kwargs)
            pred_text = output_text[0]
            error_text = None
        except Exception as exc:
            pred_text = ""
            error_text = f"{type(exc).__name__}: {exc}"
        labels = {str(option["label"]).strip().upper() for option in item["options"]}
        prediction_label = parse_label(pred_text, labels)
        records.append(
            {
                **item,
                "pred": pred_text,
                "prediction_label": prediction_label,
                "correct": bool(prediction_label and prediction_label == item["answer"]),
                "model_name": os.environ["MODEL"],
                "ckpt": str(ckpt),
                "ok": error_text is None,
                "error": error_text,
                "streaming_visible_until_s": item["time_s"],
                "streambridge_mode": "qwen35vl_causal_clip",
                "clip_window_s": float(os.environ.get("CLIP_WINDOW_S", 4.0)),
                "qwen35_video_fps": float(os.environ.get("QWEN35_VIDEO_FPS", 2.0)),
                "qwen35_video_max_frames_per_clip": int(os.environ.get("QWEN35_VIDEO_MAX_FRAMES_PER_CLIP", 8)),
            }
        )
        save_json(records, result_path("streaming_bench_eval_gpu0.json"))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
