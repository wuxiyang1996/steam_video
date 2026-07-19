"""Extract fine-grained frame-grounded L1 from Video-Holmes raw videos."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .video_l1 import QwenVideoL1Extractor, VideoL1Config
from .video_l1_evaluation import (
    build_video_l1_annotation_packet,
    summarize_video_l1,
)


def build_parser() -> argparse.ArgumentParser:
    workspace = Path("/fs/gamma-projects/vlm-robot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(workspace / "datasets"))
    parser.add_argument("--video-skills-root", default=str(workspace / "Video_Skills"))
    parser.add_argument("--video-id", action="append", dest="video_ids", required=True)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "outputs" / "video_holmes_visual_l1"),
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--coarse-window-s", type=float, default=8.0)
    parser.add_argument("--coarse-stride-s", type=float, default=6.0)
    parser.add_argument("--coarse-frames", type=int, default=8)
    parser.add_argument("--fine-frames", type=int, default=12)
    parser.add_argument("--minimum-confidence", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = str(Path(args.video_skills_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from atomic_skills.skill_model_client import SkillModelClient

    client = SkillModelClient.from_local(
        model=args.model,
        base_url=args.api_base,
        max_tokens=1400,
        timeout_s=180,
    )
    extractor = QwenVideoL1Extractor(
        client=client,
        config=VideoL1Config(
            coarse_window_s=args.coarse_window_s,
            coarse_stride_s=args.coarse_stride_s,
            frames_per_coarse_window=args.coarse_frames,
            frames_per_fine_window=args.fine_frames,
            minimum_confidence=args.minimum_confidence,
        ),
    )
    video_dir = (
        Path(args.dataset_root)
        / "Video-Holmes"
        / "Benchmark"
        / "videos_cropped"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for video_id in args.video_ids:
        video_path = video_dir / f"{video_id}.mp4"
        sample_dir = output_dir / video_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = extractor.extract(video_path=video_path, video_id=video_id)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"video_id": video_id, "error": error})
            print(f"[{video_id}] visual L1 extraction failed: {error}", flush=True)
            continue
        payload = result.to_dict()
        (sample_dir / "video_l1.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        annotation_packet = build_video_l1_annotation_packet(list(result.nodes))
        (sample_dir / "video_l1_annotation_packet.json").write_text(
            json.dumps(annotation_packet, indent=2) + "\n",
            encoding="utf-8",
        )
        structural = summarize_video_l1(list(result.nodes))
        summaries.append(
            {
                "video_id": video_id,
                "video_l1_path": str(sample_dir / "video_l1.json"),
                "annotation_packet_path": str(
                    sample_dir / "video_l1_annotation_packet.json"
                ),
                "extraction": {
                    key: value for key, value in payload.items() if key != "nodes"
                },
                "structural_summary": structural,
            }
        )
        print(
            f"[{video_id}] localized {len(result.nodes)} visual L1 events",
            flush=True,
        )
    summary = {
        "protocol_version": "video-only-l1/v0.1",
        "model": extractor.model,
        "samples": summaries,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
