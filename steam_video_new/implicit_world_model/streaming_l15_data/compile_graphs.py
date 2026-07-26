"""Compile StreamingBench/OVO L1 overlays into L1/L1.5 graph artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from memory_graph.embedding import Qwen3VLEmbeddingProvider, embed_memory_nodes
from steam_video_new.implicit_world_model.full_graph_iwm.graph_adapter import compile_l1_l15_navigation_graph
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import overlay_from_dict


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compile_graphs(
    *,
    selection_path: Path,
    graph_root: Path,
    report_path: Path,
    memory_capacity: int,
    device: str,
    video_limit: int | None,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    videos = list(selection.get("videos") or [])
    if video_limit is not None:
        videos = videos[:video_limit]
    provider = Qwen3VLEmbeddingProvider(device=device)
    completed = []
    errors = []
    for entry in videos:
        video_id = str(entry["video_id"])
        sample_dir = graph_root / video_id
        overlay_path = sample_dir / "causal_temporal_overlay.json"
        try:
            payload = json.loads(overlay_path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata") or {}
            if metadata.get("question_independent_contract") is not True:
                raise ValueError("overlay is not question-independent")
            overlay = overlay_from_dict(payload)
            nodes = [*overlay.l1_observations, *overlay.atomic_events]
            embed_memory_nodes(nodes, provider, output_path=sample_dir / "node_embeddings.npy")
            compiled = compile_l1_l15_navigation_graph(overlay, capacity=memory_capacity)
            _write_json(sample_dir / "l1_l15_navigation_graph.json", compiled.graph.to_dict())
            _write_json(sample_dir / "l1_l15_correlation_pair_audit.json", compiled.correlation_pair_audit)
            _write_json(sample_dir / "l1_l15_multichannel_pair_audit.json", compiled.multichannel_pair_audit)
            completed.append(
                {
                    "video_id": video_id,
                    "dataset": entry.get("dataset"),
                    "split": entry.get("split"),
                    "l1_observation_count": len(overlay.l1_observations),
                    "atomic_event_count": len(overlay.atomic_events),
                    "retained_node_count": len(compiled.graph.nodes),
                    "temporal_edge_count": len(compiled.graph.temporal_edges),
                }
            )
        except Exception as exc:
            errors.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
        _write_json(report_path, {"status": "running", "completed": completed, "errors": errors})
    report = {
        "schema_version": "steam-streaming-l15-compile-report/v0.1",
        "selection": str(selection_path),
        "graph_root": str(graph_root),
        "memory_capacity": memory_capacity,
        "completed_count": len(completed),
        "error_count": len(errors),
        "completed": completed,
        "errors": errors,
        "training_performed": False,
    }
    _write_json(report_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--memory-capacity", type=int, default=192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--video-limit", type=int)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Write a partial report and exit 0 when some selected graphs are missing.",
    )
    args = parser.parse_args()
    report = compile_graphs(
        selection_path=args.selection,
        graph_root=args.graph_root,
        report_path=args.report,
        memory_capacity=args.memory_capacity,
        device=args.device,
        video_limit=args.video_limit,
    )
    print(json.dumps({k: report[k] for k in ("schema_version", "completed_count", "error_count")}, indent=2))
    return 0 if args.allow_missing or not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
