"""Render timestamped contact sheets for blinded CG-Bench interval review."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from .builder import _write_json
from .evaluation import build_blinded_review_packet


def render_audit_assets(
    packet: dict[str, Any], *, video_root: Path, output_dir: Path,
    frames_per_interval: int = 6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv-python and numpy are required") from exc
    if frames_per_interval != 6:
        raise ValueError("the v0.1 contact-sheet layout requires six frames")
    result = deepcopy(packet)
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered = failed = 0
    for item in result.get("items") or []:
        path = video_root / Path(item["video_ref"]).name
        interval = item["interval"]
        start, end = float(interval["start_s"]), float(interval["end_s"])
        times = [start + (end - start) * index / 5 for index in range(6)]
        capture = cv2.VideoCapture(str(path))
        frames = []
        try:
            for index, time_s in enumerate(times):
                capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame = cv2.resize(frame, (384, 216), interpolation=cv2.INTER_AREA)
                cv2.putText(frame, f"F{index} {time_s:.2f}s", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
                frames.append(frame)
        finally:
            capture.release()
        if len(frames) != 6:
            item["contact_sheet_status"] = "failed"
            failed += 1
            continue
        sheet = np.vstack([np.hstack(frames[:3]), np.hstack(frames[3:])])
        filename = f"{item['review_id'].replace(':', '_')}.jpg"
        destination = output_dir / filename
        if not cv2.imwrite(str(destination), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90]):
            item["contact_sheet_status"] = "failed"
            failed += 1
            continue
        item["contact_sheet_status"] = "available"
        item["contact_sheet_ref"] = f"{output_dir.name}/{filename}"
        item["contact_sheet_frame_times_s"] = [round(value, 3) for value in times]
        rendered += 1
    report = {
        "schema_version": "steam-cgbench-audit-asset-report/v0.1",
        "item_count": len(result.get("items") or []), "rendered_count": rendered,
        "failed_count": failed, "frames_per_interval": frames_per_interval,
        "labels_exposed": False,
    }
    return result, report


def prepare_grounded_asset_packet(
    grounded_dataset: dict[str, Any], terminal_key: dict[str, Any],
    asset_packet: dict[str, Any], *, sample_videos: int = 32,
) -> dict[str, Any]:
    """Refresh observations while preserving label-free contact-sheet references."""
    packet, _ = build_blinded_review_packet(
        grounded_dataset, terminal_key, sample_videos=sample_videos
    )
    assets = {str(item["review_id"]): item for item in asset_packet.get("items") or []}
    for item in packet.get("items") or []:
        source = assets.get(str(item["review_id"]))
        if source is None or source.get("contact_sheet_status") != "available":
            raise ValueError(f"contact sheet missing for {item.get('review_id')}")
        for key in ("contact_sheet_status", "contact_sheet_ref", "contact_sheet_frame_times_s"):
            item[key] = source[key]
        if (item.get("observation") or {}).get("descriptor_status") != "grounded_qwen_vl_read":
            raise ValueError(f"Qwen observation is not grounded for {item.get('review_id')}")
    return packet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-packet", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--grounded-dataset", type=Path)
    parser.add_argument("--terminal-key", type=Path)
    args = parser.parse_args(argv)
    packet = json.loads(args.packet.read_text(encoding="utf-8"))
    if args.grounded_dataset is not None:
        if args.terminal_key is None:
            raise ValueError("--terminal-key is required with --grounded-dataset")
        grounded = json.loads(args.grounded_dataset.read_text(encoding="utf-8"))
        terminal = json.loads(args.terminal_key.read_text(encoding="utf-8"))
        result = prepare_grounded_asset_packet(grounded, terminal, packet)
        report = {
            "schema_version": "steam-cgbench-grounded-audit-packet-report/v0.1",
            "item_count": len(result.get("items") or []), "grounded_count": len(result.get("items") or []),
            "contact_sheet_count": len(result.get("items") or []), "labels_exposed": False,
        }
        _write_json(args.output_packet, result); _write_json(args.report, report)
        print(json.dumps(report, indent=2))
        return 0
    result, report = render_audit_assets(packet, video_root=args.video_root,
                                         output_dir=args.output_dir)
    _write_json(args.output_packet, result); _write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
