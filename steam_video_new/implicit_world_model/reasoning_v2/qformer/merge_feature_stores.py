"""Merge disjoint, contract-compatible four-slot Q-Former feature stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .feature_store import FourSlotFeatureStore, merge_feature_stores


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-overlapping-videos", action="store_true")
    args = parser.parse_args(argv)
    manifest = merge_feature_stores(
        args.manifest,
        args.output_dir,
        require_disjoint_videos=not args.allow_overlapping_videos,
    )
    store = FourSlotFeatureStore(manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "node_count": len(store),
                "video_count": len({row.video_id for row in store.manifest.rows}),
                "coverage": store.coverage(),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
