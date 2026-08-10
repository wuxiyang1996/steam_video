"""Merge per-benchmark Q-Former feature stores and their audited clip sidecars."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .feature_store import FourSlotFeatureStore, merge_feature_stores


def merge_threebench(inputs: list[Path], destination: Path) -> dict[str, object]:
    manifests = [path.expanduser().resolve() / "manifest.json" for path in inputs]
    manifest = merge_feature_stores(manifests, destination)
    destination = destination.expanduser().resolve()
    sidecar = destination / "clips.jsonl"
    node_ids: set[str] = set()
    with sidecar.open("w", encoding="utf-8") as output:
        for directory in inputs:
            with (directory.expanduser().resolve() / "clips.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    node_id = str(row["node_id"])
                    if node_id in node_ids:
                        raise ValueError(f"duplicate clip sidecar node: {node_id}")
                    node_ids.add(node_id)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
    store = FourSlotFeatureStore(manifest)
    manifest_ids = {row.node_id for row in store.manifest.rows}
    if node_ids != manifest_ids:
        raise ValueError("merged feature rows and clip sidecar rows disagree")
    return {
        "schema_version": "steam-qformer-threebench-merge/v0.1",
        "manifest": str(manifest),
        "clip_sidecar": str(sidecar),
        "node_count": len(store),
        "coverage": store.coverage(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = merge_threebench(args.inputs, args.destination)
    args.report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
