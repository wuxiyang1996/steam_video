"""Export existing artifacts into audited transition and Planner SFT JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .adapters import (
    adapt_grounded_transition_corpus,
    adapt_local_choice_packet,
    build_readiness_report,
)
from .serialization import render_sft_example


def export_existing_data(
    *,
    transition_corpus: Path,
    local_choice_public: Path,
    local_choice_hidden: Path,
    output_dir: Path,
) -> dict[str, Any]:
    transition_records = adapt_grounded_transition_corpus(_read_json(transition_corpus))
    planner_records = adapt_local_choice_packet(
        _read_json(local_choice_public),
        _read_json(local_choice_hidden),
    )
    records = transition_records + planner_records
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_jsonl(
        output_dir / "chat_sft.jsonl",
        [render_sft_example(row) for row in records],
    )
    report = build_readiness_report(records)
    report["source_artifacts"] = {
        "transition_corpus": str(transition_corpus.resolve()),
        "local_choice_public": str(local_choice_public.resolve()),
        "local_choice_hidden": str(local_choice_hidden.resolve()),
    }
    report["limitations"] = [
        "transition records are positive-only clue advances",
        "transition records are not hypothesis-conditioned",
        "transition records lack runtime L1/L1.5 target node views",
        "current Planner preferences are held-out diagnostics only",
    ]
    (output_dir / "readiness.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transition-corpus", type=Path, required=True)
    parser.add_argument("--local-choice-public", type=Path, required=True)
    parser.add_argument("--local-choice-hidden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = export_existing_data(
        transition_corpus=args.transition_corpus,
        local_choice_public=args.local_choice_public,
        local_choice_hidden=args.local_choice_hidden,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
