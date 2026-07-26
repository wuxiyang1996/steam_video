"""Emit a train-majority categorical baseline for held-out comparison."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from .calibration_gate import FIELDS
from .schemas import TRANSITION_TASK, validate_sft_record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args(argv)
    rows: list[dict[str, Any]] = []
    with args.records.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            validate_sft_record(row)
            if row["task"] == TRANSITION_TASK:
                rows.append(row)
    train = [row for row in rows if row["split"] == "train"]
    held_out = [row for row in rows if row["split"] == args.split]
    if not train or not held_out:
        raise ValueError("baseline requires train and held-out transition records")
    majority: dict[str, str] = {}
    for field in FIELDS:
        counts = Counter(
            str(row["target"]["categorical_audit"][field]["value"])
            for row in train
        )
        majority[field] = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[
            0
        ][0]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in held_out:
            stream.write(
                json.dumps(
                    {
                        "record_id": row["record_id"],
                        "prediction": majority,
                        "baseline": "train_majority",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "schema_version": "steam-iwm-categorical-baseline/v0.1",
                "held_out_record_count": len(held_out),
                "majority_labels": majority,
                "numeric_reward_present": False,
                "training_performed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
