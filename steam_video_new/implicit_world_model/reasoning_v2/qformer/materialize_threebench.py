"""Materialize leakage-safe Q-Former slots for the streaming 3-benchmark clips."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .contracts import FeatureRow
from .feature_store import FourSlotFeatureStore, write_feature_store
from .materialize_cohort import (
    CAPTION_PROMPT,
    MODEL_NAME,
    VISUAL_PROMPT,
    _encode,
    _encode_visual_nodes,
)


SOURCE_CONTRACT = "threebench-question-independent-caption-visual-time/v1"


def load_clip_rows(paths: Iterable[Path], max_nodes: int | None = None) -> list[dict[str, Any]]:
    """Load a stable, collision-free clip cohort without question or answer fields."""

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(Path(value).expanduser().resolve() for value in paths):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                source = json.loads(line)
                dataset = str(source.get("dataset") or "").strip()
                video_id = str(source.get("video_id") or "").strip()
                clip_id = str(source.get("clip_id") or "").strip()
                video_path = str(source.get("video_path") or "").strip()
                if not dataset or not video_id or not clip_id or not video_path:
                    raise ValueError(f"incomplete clip row in {path}")
                # StreamingBench reuses short video IDs across source paths.
                # The canonical benchmark wrapper also treats the path as the
                # primary video identity, so include a stable path digest here.
                path_digest = hashlib.sha256(video_path.encode("utf-8")).hexdigest()[:16]
                canonical_video_id = f"{dataset}|{path_digest}"
                node_id = f"{canonical_video_id}|{clip_id}"
                if node_id in seen:
                    continue
                seen.add(node_id)
                rows.append(
                    {
                        "node_id": node_id,
                        "video_id": canonical_video_id,
                        "dataset": dataset,
                        "source_video_id": video_id,
                        "clip_id": clip_id,
                        "text": str(source.get("text") or "").strip(),
                        "time_span": {
                            "start_s": float(source.get("start_s") or 0.0),
                            "end_s": float(source.get("end_s") or 0.0),
                        },
                        "_raw_video_path": video_path,
                    }
                )
                if max_nodes is not None and len(rows) >= max_nodes:
                    return rows
    if not rows:
        raise ValueError("clip inputs contain no rows")
    durations: dict[str, float] = {}
    for row in rows:
        durations[row["video_id"]] = max(
            durations.get(row["video_id"], 0.0), float(row["time_span"]["end_s"])
        )
    for row in rows:
        row["_video_duration_s"] = max(durations[row["video_id"]], 1.0)
    return rows


def materialize_threebench(
    clip_paths: Sequence[Path],
    destination: Path,
    *,
    device: str,
    batch_size: int = 8,
    frames_per_node: int = 3,
    max_nodes: int | None = None,
) -> dict[str, Any]:
    nodes = load_clip_rows(clip_paths, max_nodes=max_nodes)
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("three-benchmark materialization requires sentence-transformers") from exc

    model = SentenceTransformer(MODEL_NAME, device=device)
    captions = [row["text"] for row in nodes]
    caption_validity = np.asarray([bool(value) for value in captions], dtype=np.bool_)
    caption = _encode(
        model,
        [value or "Unavailable bounded caption." for value in captions],
        CAPTION_PROMPT,
        batch_size,
    )
    caption[~caption_validity] = 0.0
    visual, visual_validity, visual_failures = _encode_visual_nodes(
        model, nodes, batch_size=batch_size, frames_per_node=frames_per_node
    )
    dimension = int(caption.shape[1])
    entity_state = np.zeros((len(nodes), dimension), dtype=np.float32)
    time = np.asarray([_time_values(row) for row in nodes], dtype=np.float32)
    validity = np.stack(
        (
            caption_validity,
            np.zeros(len(nodes), dtype=np.bool_),
            visual_validity,
            np.ones(len(nodes), dtype=np.bool_),
        ),
        axis=1,
    )
    feature_rows = tuple(
        FeatureRow(index, row["node_id"], row["video_id"], _lineage_hash(row))
        for index, row in enumerate(nodes)
    )
    manifest = write_feature_store(
        destination,
        arrays={
            "caption": caption.astype(np.float32),
            "entity_state": entity_state,
            "visual": visual.astype(np.float32),
            "time": time,
        },
        validity=validity,
        rows=feature_rows,
        encoders={
            "caption": f"{MODEL_NAME}:bounded-caption/v1",
            "entity_state": "unavailable-validity-masked/v1",
            "visual": f"{MODEL_NAME}:visual-{frames_per_node}-frame-mean/v1",
            "time": "deterministic-normalized-time/v1",
        },
        source_contract=SOURCE_CONTRACT,
        boundary_audit_version="threebench-question-answer-exclusion/v1",
        normalized={"caption": True, "entity_state": False, "visual": True, "time": False},
    )
    sidecar = destination / "clips.jsonl"
    with sidecar.open("w", encoding="utf-8") as handle:
        for row in nodes:
            public = {key: value for key, value in row.items() if not key.startswith("_")}
            public["video_path"] = row["_raw_video_path"]
            public["duration_s"] = row["_video_duration_s"]
            handle.write(json.dumps(public, ensure_ascii=False) + "\n")
    store = FourSlotFeatureStore(manifest)
    return {
        "schema_version": "steam-qformer-threebench-materialization/v0.1",
        "manifest": str(manifest),
        "clip_sidecar": str(sidecar),
        "node_count": len(store),
        "video_count": len({row.video_id for row in feature_rows}),
        "coverage": store.coverage(),
        "frames_per_node": frames_per_node,
        "visual_failure_count": len(visual_failures),
        "visual_failures": visual_failures,
        "source_contract": SOURCE_CONTRACT,
    }


def _time_values(row: dict[str, Any]) -> list[float]:
    start = float(row["time_span"]["start_s"])
    end = float(row["time_span"]["end_s"])
    duration = float(row["_video_duration_s"])
    if start < 0 or end < start or duration <= 0:
        raise ValueError(f"invalid clip span: {row['node_id']}")
    center = (start + end) / 2.0
    return [start / duration, end / duration, (end - start) / duration, center / duration]


def _lineage_hash(row: dict[str, Any]) -> str:
    payload = {
        "node_id": row["node_id"],
        "video_id": row["video_id"],
        "time_span": row["time_span"],
        "text": row["text"],
        "video_path": row["_raw_video_path"],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-node", type=int, default=3)
    parser.add_argument("--max-nodes", type=int)
    args = parser.parse_args(argv)
    report = materialize_threebench(
        args.clips,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        frames_per_node=args.frames_per_node,
        max_nodes=args.max_nodes,
    )
    args.report.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.expanduser().resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
