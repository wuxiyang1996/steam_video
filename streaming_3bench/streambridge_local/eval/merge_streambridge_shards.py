#!/usr/bin/env python3
"""Merge StreamBridge OVO-Bench, VideoMME, and StreamingBench shard outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"expected list JSON: {path}")
    return payload


def write_json(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def shard_result_path(shard_dir: Path, filename: str) -> Path:
    direct = shard_dir / filename
    if direct.exists():
        return direct
    nested = shard_dir / "results" / filename
    if nested.exists():
        return nested
    return direct


def streaming_bench_summary(records: list[dict]) -> dict:
    summary: dict = {"overall": {}, "by_task": {}}
    tasks = sorted({str(row.get("type") or "unknown") for row in records})
    for key, rows in [("overall", records)] + [
        (task, [row for row in records if str(row.get("type") or "unknown") == task]) for task in tasks
    ]:
        ok_rows = [row for row in rows if row.get("ok")]
        parsed = [row for row in ok_rows if row.get("prediction_label")]
        correct = [row for row in ok_rows if row.get("correct") is True]
        payload = {
            "total": len(rows),
            "ok": len(ok_rows),
            "failed": len(rows) - len(ok_rows),
            "parsed": len(parsed),
            "parse_rate": (len(parsed) / len(ok_rows)) if ok_rows else 0.0,
            "correct": len(correct),
            "accuracy": (len(correct) / len(ok_rows)) if ok_rows else 0.0,
            "accuracy_on_parsed": (len(correct) / len(parsed)) if parsed else 0.0,
        }
        if key == "overall":
            summary["overall"] = payload
        else:
            summary["by_task"][key] = payload
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shard_dirs = sorted(path for path in args.shards_root.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard_* directories under {args.shards_root}")

    summary = {"shards_root": str(args.shards_root), "output_dir": str(args.output_dir), "shard_count": len(shard_dirs)}
    for filename in ("ovo_bench_eval.json", "videomme_eval.json", "streaming_bench_eval.json"):
        merged: list[dict] = []
        per_shard = {}
        for shard_dir in shard_dirs:
            records = load_json(shard_result_path(shard_dir, filename))
            per_shard[shard_dir.name] = len(records)
            merged.extend(records)
        write_json(args.output_dir / filename, merged)
        summary[filename] = {"total": len(merged), "per_shard": per_shard}
        if filename == "streaming_bench_eval.json":
            summary["streaming_bench_metrics"] = streaming_bench_summary(merged)

    if summary["ovo_bench_eval.json"]["total"] or summary["videomme_eval.json"]["total"]:
        env = os.environ.copy()
        env["RESULT_DIR"] = str(args.output_dir)
        report = subprocess.run(
            [sys.executable, "eval/metric_report.py"],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        (args.output_dir / "metric_report.txt").write_text(report.stdout, encoding="utf-8")
        summary["metric_report_returncode"] = report.returncode
    else:
        report = None
        summary["metric_report_returncode"] = None
    (args.output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if report is not None and report.returncode != 0:
        print(report.stdout, file=sys.stderr)
    return report.returncode if report is not None else 0


if __name__ == "__main__":
    raise SystemExit(main())
