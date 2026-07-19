"""Command-line reliability gate for one canonical Video_Skills L1 graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .adapter import canonical_to_memory_nodes, load_json
from .contracts import L1HumanAudit
from .reliability import audit_l1_nodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", required=True, type=Path)
    parser.add_argument("--human-audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    canonical = load_json(args.canonical)
    graph, nodes = canonical_to_memory_nodes(canonical)
    human_audit = (
        L1HumanAudit.from_dict(load_json(args.human_audit))
        if args.human_audit
        else None
    )
    observation_end = graph.get("observation_end_s")
    report = audit_l1_nodes(
        nodes,
        human_audit=human_audit,
        observation_end_s=float(observation_end) if observation_end is not None else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {"pass": 0, "fail": 1, "incomplete": 2}[report.status]


if __name__ == "__main__":
    raise SystemExit(main())
