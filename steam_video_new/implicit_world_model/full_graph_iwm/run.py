"""CLI smoke for retained-graph compilation and one full-IWM planning step."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

from steam_video_new.implicit_world_model.l15_graph_navigator.gpt_oss import (
    DEFAULT_GPT_OSS_MODEL,
    OpenAICompatibleCategoricalClient,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)

from .action_compiler import GraphActionCompiler
from .contracts import CursorBeliefState
from .gpt_oss import GPTOSSFullGraphPreferenceModel, GPTOSSFullGraphWorldModel
from .graph_adapter import build_l1_l15_navigation_graph
from .model_input import build_iwm_graph_input, graph_input_to_categorical_payload
from .planner import FullGraphIWMPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile or plan over the full retained L1/L1.5 graph",
    )
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--capacity", type=int, default=64)
    parser.add_argument("--graph-read-budget", type=int, default=8)
    parser.add_argument("--horizon", type=int, choices=(1, 2), default=1)
    parser.add_argument("--max-trajectory-pairs", type=int, default=4096)
    parser.add_argument(
        "--mode",
        choices=("compile-only", "gpt-oss-120b"),
        default="compile-only",
    )
    parser.add_argument("--keys-py", type=Path)
    parser.add_argument("--model", default=DEFAULT_GPT_OSS_MODEL)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument(
        "--reasoning-effort",
        choices=("low", "medium", "high"),
        default="low",
    )
    parser.add_argument("--skip-schema-validation", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.capacity < 1 or args.graph_read_budget < 0 or args.max_trajectory_pairs < 1:
        raise ValueError(
            "capacity and max-trajectory-pairs must be positive; "
            "graph-read budget must be non-negative"
        )
    if args.mode == "gpt-oss-120b" and args.keys_py is None:
        raise ValueError("--mode gpt-oss-120b requires --keys-py")

    loaded = load_overlay_artifact(
        args.overlay,
        validate_schema=not args.skip_schema_validation,
    )
    client = (
        OpenAICompatibleCategoricalClient.from_openrouter_keys_file(
            args.keys_py,
            model=args.model,
            timeout_s=args.timeout_s,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
        )
        if args.keys_py is not None
        else None
    )
    graph = build_l1_l15_navigation_graph(
        loaded.overlay,
        capacity=args.capacity,
    )
    belief = CursorBeliefState(
        belief_id="belief:initial",
        question=args.question,
        remaining_reads=args.graph_read_budget,
    )
    actions = GraphActionCompiler().compile(belief, graph)
    graph_input = build_iwm_graph_input(belief, graph, actions)
    result: dict[str, Any] = {
        "schema_version": "steam-full-graph-iwm-smoke/v0.1",
        "mode": args.mode,
        "source_overlay": str(loaded.source_path),
        "retained_graph": {
            "graph_id": graph.graph_id,
            "capacity": graph.capacity,
            "node_count": len(graph.nodes),
            "temporal_edge_count": len(graph.temporal_edges),
            "correlation_edge_count": len(graph.correlation_edges),
            "verified_relation_count": len(graph.verified_relations),
            "metadata": graph.metadata,
        },
        "model_input": graph_input_to_categorical_payload(graph_input),
    }
    if args.mode == "gpt-oss-120b":
        assert client is not None
        decision = FullGraphIWMPlanner(
            GPTOSSFullGraphWorldModel(client),
            GPTOSSFullGraphPreferenceModel(client),
            horizon=args.horizon,
            max_trajectory_pairs=args.max_trajectory_pairs,
        ).plan(belief, graph)
        result["plan"] = _jsonable(decision)
        result["model_response_audits"] = list(client.response_audits)
    else:
        result["plan"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


def _jsonable(value: object) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


if __name__ == "__main__":
    raise SystemExit(main())
