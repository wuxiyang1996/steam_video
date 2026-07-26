#!/usr/bin/env python3
"""Attach latest L2 trajectory metadata to existing graph JSONL artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from ..l1_clue_graph.clue_memory import extract_clue_memory_graph
    from ..l2_reasoning_graph.l2_recursive_trace import attach_initial_l2_trajectory
except ImportError:  # pragma: no cover - direct script execution
    from clue_memory import extract_clue_memory_graph
    from l2_recursive_trace import attach_initial_l2_trajectory


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def retrofit_example(example: dict[str, Any]) -> dict[str, Any]:
    metadata = example.setdefault("metadata", {})
    rollout = metadata.get("reasoning_rollout") or {}
    if not rollout:
        return example
    clue_graph = metadata.get("clue_memory_graph")
    if not isinstance(clue_graph, dict) or not clue_graph.get("nodes"):
        clue_graph = extract_clue_memory_graph(example, mode=(example.get("available_inputs") or {}).get("mode"))
        metadata["clue_memory_graph"] = clue_graph
    attach_initial_l2_trajectory(rollout, clue_graph)
    metadata["reasoning_rollout"] = rollout
    metadata["reasoning_rollout_shell"] = rollout
    return example


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrofit latest L2 trajectory metadata onto existing JSONL graph outputs.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in args.paths:
        for example in _read_jsonl(path):
            rows.append(retrofit_example(example))
    _write_jsonl(args.output, rows)
    summary = {
        "written": len(rows),
        "output": str(args.output),
        "with_l2_trajectory": sum(
            1
            for row in rows
            if (((row.get("metadata") or {}).get("reasoning_rollout") or {}).get("metadata") or {}).get("l2_trajectory")
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
