"""Prepare leakage-safe CG-Bench supervision and L1 node-alignment audits.

The artifacts produced here are training/evaluation data for the future IWM.
They are explicitly forbidden as inputs to L1 or L1.5 graph construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SUPERVISION_SCHEMA = "steam-cgbench-iwm-supervision/v0.1"
ALIGNMENT_SCHEMA = "steam-cgbench-iwm-node-alignment/v0.1"
READINESS_SCHEMA = "steam-cgbench-iwm-supervision-readiness/v0.1"
GENERATION_QUEUE_SCHEMA = "steam-cgbench-full-video-l1-generation-queue/v0.1"


def build_iwm_supervision_artifacts(
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    *,
    graph_paths: Iterable[Path] = (),
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build supervision, hidden node alignment, readiness, and graph queue.

    GT intervals are joined to already-frozen graphs only inside the hidden
    alignment artifact.  They are never copied into the graph-generation queue.
    """

    graphs = _load_graphs(graph_paths)
    hidden_by_case = {
        str(row["case_id"]): row for row in hidden_key.get("cases") or []
    }
    supervision_cases: list[dict[str, Any]] = []
    alignment_cases: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, Any]] = {}
    video_rows: dict[str, dict[str, Any]] = {}
    belief_ready = 0
    observation_ready = 0
    aligned_bridge_groups = 0
    aligned_bridge_pairs = 0
    pending_bridge_groups = 0

    for case in dataset.get("cases") or []:
        case_id = str(case["case_id"])
        video_id = str(case["video_id"])
        split = str(case["split"])
        key = hidden_by_case.get(case_id)
        if key is None:
            raise ValueError(f"hidden key is missing case {case_id}")
        graph = graphs.get(video_id)
        nodes = list(graph.get("nodes") or []) if graph is not None else []
        clue_intervals = list(key.get("clue_intervals") or [])
        clue_groups = [
            {
                "clue_index": index,
                "node_ids": sorted(
                    str(node["node_id"])
                    for node in nodes
                    if _overlaps(_span(node["time_span"]), _span(interval))
                ),
            }
            for index, interval in enumerate(clue_intervals)
        ]
        bridges: list[dict[str, Any]] = []
        for left, right in zip(clue_groups, clue_groups[1:]):
            left_ids = tuple(left["node_ids"])
            right_ids = tuple(right["node_ids"])
            distinct_pairs = sorted(
                [src, dst]
                for src in left_ids
                for dst in right_ids
                if src != dst
            )
            same_node_ids = sorted(set(left_ids) & set(right_ids))
            if left_ids and right_ids:
                status = "aligned_multi_positive"
                aligned_bridge_groups += 1
                aligned_bridge_pairs += len(distinct_pairs)
            else:
                status = "pending_full_video_l1_alignment"
                pending_bridge_groups += 1
            bridges.append(
                {
                    "from_clue_index": left["clue_index"],
                    "to_clue_index": right["clue_index"],
                    "status": status,
                    "positive_node_pairs": distinct_pairs,
                    "same_node_ids": same_node_ids,
                    "unmatched_same_video_pairs_are_negative": False,
                }
            )

        transitions = []
        for transition in case.get("executed_transitions") or []:
            target = transition.get("target") or {}
            belief_delta = target.get("belief_delta")
            real_observation = transition.get("real_observation") or {}
            descriptor_status = str(real_observation.get("descriptor_status") or "")
            descriptor = real_observation.get("descriptor")
            belief_delta_ready = isinstance(belief_delta, dict) and bool(belief_delta)
            observation_descriptor_ready = (
                descriptor is not None and not descriptor_status.startswith("pending")
            )
            belief_ready += int(belief_delta_ready)
            observation_ready += int(observation_descriptor_ready)
            transitions.append(
                {
                    "transition_id": transition["transition_id"],
                    "checkpoint": transition.get("checkpoint"),
                    "action": transition.get("action"),
                    "target": target,
                    "supervision_readiness": {
                        "belief_delta": (
                            "ready_gt_clue_coverage" if belief_delta_ready else "missing"
                        ),
                        "observation_descriptor": (
                            "ready_grounded_observation"
                            if observation_descriptor_ready
                            else "pending_grounded_observation"
                        ),
                    },
                }
            )
        supervision_cases.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "split": split,
                "question": (case.get("planner_input") or {}).get("question"),
                "choices": (case.get("planner_input") or {}).get("choices"),
                "transitions": transitions,
                "trajectory_preference": case.get("trajectory_preference"),
                "node_alignment_ref": f"iwmalign:{_digest(case_id)}",
            }
        )
        alignment_cases.append(
            {
                "alignment_id": f"iwmalign:{_digest(case_id)}",
                "case_id": case_id,
                "video_id": video_id,
                "split": split,
                "graph_id": graph.get("graph_id") if graph else None,
                "graph_fingerprint": _checksum(graph) if graph else None,
                "clue_intervals": clue_intervals,
                "clue_node_groups": clue_groups,
                "consecutive_bridge_groups": bridges,
                "gt_joined_after_graph_freeze": graph is not None,
            }
        )
        stats = split_stats.setdefault(
            split,
            {
                "case_count": 0,
                "video_ids": set(),
                "graph_aligned_case_count": 0,
                "aligned_bridge_group_count": 0,
                "pending_bridge_group_count": 0,
            },
        )
        stats["case_count"] += 1
        stats["video_ids"].add(video_id)
        stats["graph_aligned_case_count"] += int(graph is not None)
        stats["aligned_bridge_group_count"] += sum(
            row["status"] == "aligned_multi_positive" for row in bridges
        )
        stats["pending_bridge_group_count"] += sum(
            row["status"] == "pending_full_video_l1_alignment" for row in bridges
        )
        video_rows.setdefault(
            video_id,
            {
                "video_id": video_id,
                "video_ref": case.get("video_ref"),
                "split": split,
                "video_duration_s": case.get("video_duration_s"),
                "required_graph_scope": "full_video_question_independent",
                "current_graph_status": (
                    "artifact_available_scope_must_be_audited"
                    if graph is not None
                    else "missing"
                ),
            },
        )

    supervision = {
        "schema_version": SUPERVISION_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "purpose": "future action-conditioned IWM training; no training performed",
        "allowed_consumers": ["iwm_training", "offline_iwm_evaluation"],
        "forbidden_consumers": [
            "l1_writer",
            "l1_consolidation",
            "l1.5_graph_builder",
            "runtime_planner_input",
        ],
        "numeric_reward_present": False,
        "preference_supervision_is_categorical": True,
        "training_performed": False,
        "cases": supervision_cases,
    }
    alignment = {
        "schema_version": ALIGNMENT_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "supervision_sha256": _checksum(supervision),
        "warning": "Hidden GT join; never provide this artifact to graph construction or runtime planning.",
        "unmatched_same_video_pairs_are_negative": False,
        "cases": alignment_cases,
    }
    normalized_splits = {
        split: {
            **{key: value for key, value in stats.items() if key != "video_ids"},
            "video_count": len(stats["video_ids"]),
        }
        for split, stats in sorted(split_stats.items())
    }
    heldout = [normalized_splits[name] for name in ("validation", "test") if name in normalized_splits]
    readiness = {
        "schema_version": READINESS_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(supervision_cases),
        "video_count": len(video_rows),
        "graph_video_count": len(graphs),
        "belief_delta_ready_transition_count": belief_ready,
        "grounded_observation_descriptor_ready_transition_count": observation_ready,
        "aligned_consecutive_bridge_group_count": aligned_bridge_groups,
        "aligned_distinct_positive_node_pair_count": aligned_bridge_pairs,
        "pending_consecutive_bridge_group_count": pending_bridge_groups,
        "trusted_same_video_negative_pair_count": 0,
        "unmatched_same_video_pairs_are_negative": False,
        "splits": normalized_splits,
        "heldout_aligned_bridge_group_count": sum(
            row["aligned_bridge_group_count"] for row in heldout
        ),
        "phase_3_training_ready": False,
        "phase_3_blockers": [
            "full-video frozen L1 graphs are missing for most benchmark videos",
            "held-out aligned clue bridges are insufficient",
            "grounded observation descriptors remain pending",
            "trusted same-video hard negatives are unavailable",
        ],
        "training_performed": False,
    }
    generation_queue = {
        "schema_version": GENERATION_QUEUE_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "purpose": "question-independent full-video L1 generation before hidden GT alignment",
        "graph_scope": "entire video; never a GT-selected interval or prefix",
        "forbidden_builder_inputs": [
            "question",
            "choices",
            "answer",
            "clue_intervals",
            "node_alignment",
            "transition_targets",
        ],
        "videos": [video_rows[key] for key in sorted(video_rows)],
        "training_performed": False,
    }
    return supervision, alignment, readiness, generation_queue


def write_iwm_supervision_artifacts(
    output_dir: Path,
    artifacts: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "iwm_supervision.untrained.json",
        "iwm_node_alignment.hidden_key.json",
        "iwm_supervision_readiness.json",
        "full_video_l1_generation_queue.json",
    )
    for name, payload in zip(names, artifacts):
        (output_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


def _load_graphs(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    graphs: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        nodes = payload.get("nodes") or []
        if not nodes:
            continue
        video_ids = {str(node.get("video_id")) for node in nodes if node.get("video_id")}
        if len(video_ids) != 1:
            raise ValueError(f"graph {path} does not contain exactly one video")
        video_id = next(iter(video_ids))
        if video_id in graphs:
            raise ValueError(f"duplicate graph for video {video_id}")
        graphs[video_id] = payload
    return graphs


def _span(payload: dict[str, Any]) -> tuple[float, float]:
    return float(payload["start_s"]), float(payload["end_s"])


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return max(left[0], right[0]) <= min(left[1], right[1])


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--hidden-key", type=Path, required=True)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    graph_paths = (
        sorted(args.graph_root.glob("*/l1_l15_navigation_graph.json"))
        if args.graph_root is not None
        else []
    )
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_key.read_text(encoding="utf-8"))
    artifacts = build_iwm_supervision_artifacts(
        dataset,
        hidden,
        graph_paths=graph_paths,
    )
    write_iwm_supervision_artifacts(args.output_dir, artifacts)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
