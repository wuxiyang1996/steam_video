#!/usr/bin/env python3
"""Merge sharded M3 streaming-eval outputs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    shard_record_paths = sorted(args.shards_root.glob("shard_*/records.jsonl"))
    if not shard_record_paths:
        raise SystemExit(f"no shard records found under {args.shards_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with (args.output_dir / "records.jsonl").open("w", encoding="utf-8") as out:
        for path in shard_record_paths:
            for record in iter_jsonl(path):
                record = {**record, "source_shard_dir": str(path.parent)}
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = metric_summary(records)
    summary["merge"] = {
        "shards_root": str(args.shards_root),
        "shard_count": len(shard_record_paths),
        "records_path": str(args.output_dir / "records.jsonl"),
        "metrics_path": str(args.output_dir / "metrics_summary.json"),
    }
    (args.output_dir / "metrics_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
