"""Emit the compact semantic-address copy baseline for v4 observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .schemas import OBSERVATION_TASK, validate_sft_record


def build_predictions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    predictions = []
    for row in records:
        validate_sft_record(row)
        if (
            row["task"] != OBSERVATION_TASK
            or row["split"] != "validation"
            or row.get("representation_eligible") is not True
        ):
            continue
        semantic_key = str(
            row["input"]["target_node_key"].get("semantic_key") or ""
        )
        predictions.append(
            {
                "record_id": row["record_id"],
                "prediction": {
                    "observation_descriptor": {
                        "event": semantic_key,
                        "entities": [],
                        "states": [],
                        "state_change": None,
                    }
                },
                "generation_valid": True,
                "generation_error": None,
                "raw_generation": None,
                "model_output_contains_numeric_scalar": False,
                "baseline": "copy_target_semantic_address",
            }
        )
    if not predictions:
        raise ValueError("no representation-complete validation observations")
    return predictions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    predictions = build_predictions(_read_jsonl(args.records))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "schema_version": "steam-iwm-observation-baseline/v0.1",
                "baseline": "copy_target_semantic_address",
                "prediction_count": len(predictions),
                "training_performed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
