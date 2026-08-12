"""Generate sparse question-independent caption candidates for frozen L1.5 graphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from steam_video_new.implicit_world_model.full_graph_iwm.caption_candidates import (
    propose_caption_candidate_overlay,
)
from steam_video_new.implicit_world_model.full_graph_iwm.graph_adapter import (
    build_l1_l15_navigation_graph,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    OpenAICompatibleCategoricalClient,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)


def generate_caption_candidate_artifacts(
    selection: dict[str, Any],
    *,
    graph_root: Path,
    client: OpenAICompatibleCategoricalClient,
    capacity: int,
    video_limit: int | None = None,
) -> dict[str, Any]:
    rows = []
    videos = list(selection.get("videos") or [])
    if video_limit is not None:
        videos = videos[:video_limit]
    for entry in videos:
        video_id = str(entry["video_id"])
        sample_dir = graph_root / video_id
        overlay_path = sample_dir / "causal_temporal_overlay.json"
        output_path = sample_dir / "l1_l15_caption_candidates.json"
        row: dict[str, Any] = {
            "video_id": video_id,
            "overlay_path": str(overlay_path),
            "output_path": str(output_path),
        }
        try:
            loaded = load_overlay_artifact(overlay_path, validate_schema=True)
            graph = build_l1_l15_navigation_graph(loaded.overlay, capacity=capacity)
            artifact = propose_caption_candidate_overlay(graph, client)
            _write_json(output_path, artifact)
            row.update(
                {
                    "status": "written",
                    "candidate_edge_count": artifact["candidate_edge_count"],
                    "candidate_edge_density": artifact["candidate_edge_density"],
                    "base_graph_fingerprint": artifact["base_graph_fingerprint"],
                }
            )
        except Exception as exc:
            row.update(
                {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            )
        rows.append(row)
    return {
        "schema_version": "steam-cgbench-caption-candidate-build/v0.1",
        "model": client.model,
        "video_count": len(rows),
        "successful_video_count": sum(row.get("status") == "written" for row in rows),
        "question_independent": True,
        "contains_question_answer_or_clue": False,
        "numeric_model_output": False,
        "training_performed": False,
        "videos": rows,
        "model_response_audits": client.response_audits,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--keys-py", type=Path, required=True)
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--timeout-s", type=int, default=240)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="low")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    client = OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
        args.keys_py,
        model=args.model,
        timeout_s=args.timeout_s,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
    )
    report = generate_caption_candidate_artifacts(
        selection,
        graph_root=args.graph_root,
        client=client,
        capacity=args.capacity,
        video_limit=args.video_limit,
    )
    _write_json(args.report, report)
    print(json.dumps({key: report[key] for key in ("model", "video_count", "successful_video_count")}, indent=2))
    return 0 if report["successful_video_count"] == report["video_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
