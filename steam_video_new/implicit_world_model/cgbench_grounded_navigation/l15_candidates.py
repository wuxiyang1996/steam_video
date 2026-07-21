"""Build leakage-safe real candidate hops and hidden CG-Bench temporal alignment."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .builder import _write_json


PUBLIC_SCHEMA = "steam-cgbench-l15-candidates/v0.1"
HIDDEN_SCHEMA = "steam-cgbench-l15-candidate-alignment/v0.1"
PROTOCOL_SCHEMA = "steam-cgbench-closed-loop-protocol/v0.1"


def build_l15_candidate_artifact(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    *,
    graph_paths: Iterable[Path] = (),
    subtitle_root: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build public candidates without exposing which hops overlap GT clues.

    Persisted L1/L1.5 graph nodes are preferred.  Time-aligned subtitle cues
    are a grounded L1 fallback, not a claim that an L1.5 graph exists.  No
    fixed windows or GT clue intervals are used to create public candidates.
    """

    hidden_by_case = {row["case_id"]: row for row in hidden.get("cases") or []}
    wanted_videos = {str(case["video_id"]) for case in dataset.get("cases") or []}
    graph_nodes = _load_graph_nodes(graph_paths, wanted_videos)
    subtitle_nodes: dict[str, list[dict[str, Any]]] = {}
    if subtitle_root is not None:
        root = subtitle_root.expanduser().resolve()
        for video_id in sorted(wanted_videos - set(graph_nodes)):
            subtitle_path = root / f"{video_id}.srt"
            if subtitle_path.is_file():
                subtitle_nodes[video_id] = _subtitle_candidate_nodes(video_id, subtitle_path)

    public_cases: list[dict[str, Any]] = []
    candidate_sets: dict[str, dict[str, Any]] = {}
    hidden_cases: list[dict[str, Any]] = []
    source_case_counts: dict[str, int] = {}
    source_video_ids: dict[str, set[str]] = {}
    unavailable_videos: set[str] = set()
    candidate_case_association_count = 0
    gt_overlap_hops = 0
    for case in dataset.get("cases") or []:
        case_id = str(case["case_id"])
        video_id = str(case["video_id"])
        if graph_nodes.get(video_id):
            nodes = graph_nodes[video_id]
            source = "persisted_l1_l15_graph"
        elif subtitle_nodes.get(video_id):
            nodes = subtitle_nodes[video_id]
            source = "time_aligned_subtitle_l1_fallback"
        else:
            nodes = []
            source = "unavailable"
            unavailable_videos.add(video_id)
        source_case_counts[source] = source_case_counts.get(source, 0) + 1
        source_video_ids.setdefault(source, set()).add(video_id)
        candidate_set_id = "cgset:" + _digest(video_id)[:20]
        if candidate_set_id not in candidate_sets:
            candidate_sets[candidate_set_id] = {
                "candidate_set_id": candidate_set_id,
                "video_id": video_id,
                "candidate_source": source,
                "candidate_hops": [_public_hop(video_id, node) for node in nodes],
            }
        hops = candidate_sets[candidate_set_id]["candidate_hops"]
        candidate_case_association_count += len(hops)
        public_cases.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "video_ref": case["video_ref"],
                "split": case["split"],
                "question": case["planner_input"]["question"],
                "choices": case["planner_input"]["choices"],
                "candidate_source": source,
                "candidate_status": "available" if hops else "unavailable",
                "candidate_set_id": candidate_set_id,
                "candidate_hop_count": len(hops),
            }
        )
        key = hidden_by_case[case_id]
        clue_intervals = [
            (float(row["start_s"]), float(row["end_s"]))
            for row in key.get("clue_intervals") or []
        ]
        alignments = []
        for hop, node in zip(hops, nodes):
            span = _span_tuple(node["time_span"])
            matched = [index for index, clue in enumerate(clue_intervals) if _overlaps(span, clue)]
            if matched:
                gt_overlap_hops += 1
            alignments.append(
                {
                    "hop_id": hop["hop_id"],
                    "temporal_gt_relation": (
                        "overlaps_annotated_clue" if matched
                        else "no_overlap_with_annotated_clue_unlabeled"
                    ),
                    "clue_indices": matched,
                    "semantic_negative": False,
                }
            )
        hidden_cases.append(
            {
                "case_id": case_id,
                "answer_text": key["answer_text"],
                "answer_key": key["answer_key"],
                "clue_intervals": key["clue_intervals"],
                "hop_alignment": alignments,
            }
        )

    public = {
        "schema_version": PUBLIC_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "candidate_contract": (
            "public real-memory candidates only; GT overlap and terminal answer remain hidden"
        ),
        "non_gt_candidate_contract": (
            "unlabeled; absence of temporal overlap is never semantic-negative supervision"
        ),
        "candidate_source_precedence": [
            "persisted_l1_l15_graph", "time_aligned_subtitle_l1_fallback"
        ],
        "candidate_sets": [candidate_sets[key] for key in sorted(candidate_sets)],
        "cases": public_cases,
        "training_ready": False,
        "training_performed": False,
    }
    public_checksum = _checksum(public)
    hidden_alignment = {
        "schema_version": HIDDEN_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "public_sha256": public_checksum,
        "warning": "Never expose hop_alignment, clue intervals, or terminal targets to the planner.",
        "cases": hidden_cases,
    }
    report = {
        "schema_version": "steam-cgbench-l15-candidate-report/v0.1",
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(public_cases),
        "unique_candidate_hop_count": sum(
            len(row["candidate_hops"]) for row in candidate_sets.values()
        ),
        "candidate_case_association_count": candidate_case_association_count,
        "gt_overlap_hop_count_for_offline_evaluation_only": gt_overlap_hops,
        "source_case_counts": dict(sorted(source_case_counts.items())),
        "source_video_counts": {
            source: len(video_ids) for source, video_ids in sorted(source_video_ids.items())
        },
        "unavailable_video_count": len(unavailable_videos),
        "unavailable_video_ids": sorted(unavailable_videos),
        "persisted_l1_l15_graph_video_count": len(graph_nodes),
        "videos_requiring_persisted_l1_l15_graph_count": len(wanted_videos - set(graph_nodes)),
        "videos_requiring_persisted_l1_l15_graph_ids": sorted(wanted_videos - set(graph_nodes)),
        "oracle_candidate_generation": False,
        "semantic_negative_labels": False,
        "human_review_required": False,
        "training_performed": False,
    }
    errors = validate_l15_candidate_artifact(public, hidden_alignment)
    if errors:
        raise ValueError("invalid L1.5 candidate artifact: " + "; ".join(errors[:8]))
    return public, hidden_alignment, report


def build_graph_generation_manifest(
    public: dict[str, Any], report: dict[str, Any]
) -> dict[str, Any]:
    required = set(report.get("videos_requiring_persisted_l1_l15_graph_ids") or [])
    cases_by_video: dict[str, list[dict[str, str]]] = {}
    refs: dict[str, str] = {}
    fallback: dict[str, str] = {}
    for case in public.get("cases") or []:
        video_id = str(case["video_id"])
        if video_id not in required:
            continue
        refs[video_id] = str(case["video_ref"])
        fallback[video_id] = str(case["candidate_source"])
        cases_by_video.setdefault(video_id, []).append(
            {"case_id": str(case["case_id"]), "split": str(case["split"])}
        )
    return {
        "schema_version": "steam-cgbench-l15-graph-generation-manifest/v0.1",
        "dataset_id": public.get("dataset_id"),
        "purpose": "generate real L1/L1.5 memory once per video without GT clue access",
        "videos": [
            {
                "video_id": video_id,
                "video_ref": refs[video_id],
                "current_candidate_source": fallback[video_id],
                "cases": sorted(cases_by_video[video_id], key=lambda row: row["case_id"]),
            }
            for video_id in sorted(required)
        ],
        "forbidden_builder_inputs": [
            "question", "choices", "answer", "clue_intervals", "hop_alignment"
        ],
        "training_performed": False,
    }


def build_closed_loop_protocol(public: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset_id": public.get("dataset_id"),
        "purpose": "matched-candidate closed-loop reasoning-navigation evaluation",
        "shared_candidate_artifact_sha256": _checksum(public),
        "arms": [
            {
                "arm": "world_model_guided",
                "transition_source": "action_conditioned_iwm",
                "planning_horizon": "short_rollout_then_replan",
            },
            {
                "arm": "no_world_model",
                "transition_source": "none",
                "planning_horizon": "reactive_current_belief_only",
            },
            {
                "arm": "shuffled_world_model_prediction",
                "transition_source": "case_internal_deterministic_permutation",
                "planning_horizon": "short_rollout_then_replan",
            },
            {
                "arm": "immediate_effect_only",
                "transition_source": "action_conditioned_iwm",
                "planning_horizon": "one_hop_no_delayed_effect",
            },
            {
                "arm": "oracle_clue_ceiling",
                "transition_source": "hidden_gt_alignment_evaluator_only",
                "planning_horizon": "coverage_ceiling_not_a_method_result",
            },
        ],
        "comparison_contract": (
            "same candidate pool and read budget per case; answer accuracy, read efficiency, "
            "action divergence, and order sensitivity reported separately"
        ),
        "model_output_contract": (
            "observation descriptor, belief-delta descriptor, and pairwise preference only; "
            "no model-produced reward, probability, score, or utility"
        ),
        "human_review_required": False,
        "training_performed": False,
    }


def validate_l15_candidate_artifact(
    public: dict[str, Any], hidden: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if public.get("schema_version") != PUBLIC_SCHEMA:
        errors.append("unsupported public schema")
    if hidden.get("schema_version") != HIDDEN_SCHEMA:
        errors.append("unsupported hidden schema")
    if hidden.get("public_sha256") != _checksum(public):
        errors.append("hidden checksum mismatch")
    if _contains_key(public, {"answer_key", "answer_text", "clue_intervals", "hop_alignment",
                              "temporal_gt_relation", "semantic_negative"}):
        errors.append("public artifact leaks hidden GT alignment")
    case_ids: set[str] = set()
    split_by_video: dict[str, str] = {}
    hop_ids: set[str] = set()
    candidate_set_ids: set[str] = set()
    for set_index, candidate_set in enumerate(public.get("candidate_sets") or []):
        candidate_set_id = str(candidate_set.get("candidate_set_id") or "")
        if not candidate_set_id or candidate_set_id in candidate_set_ids:
            errors.append(f"candidate_sets.{set_index} has duplicate or empty ID")
        candidate_set_ids.add(candidate_set_id)
        for hop in candidate_set.get("candidate_hops") or []:
            hop_id = str(hop.get("hop_id") or "")
            if not hop_id or hop_id in hop_ids:
                errors.append(f"candidate_sets.{set_index} has duplicate or empty hop_id")
            hop_ids.add(hop_id)
            span = hop.get("time_span") or {}
            try:
                start, end = _span_tuple(span)
            except (KeyError, TypeError, ValueError):
                errors.append(f"candidate_sets.{set_index} has invalid candidate span")
                continue
            if start < 0 or end <= start:
                errors.append(f"candidate_sets.{set_index} has invalid candidate span")
    for index, case in enumerate(public.get("cases") or []):
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in case_ids:
            errors.append(f"cases.{index} has duplicate or empty case_id")
        case_ids.add(case_id)
        video_id = str(case.get("video_id") or "")
        split = str(case.get("split") or "")
        if video_id in split_by_video and split_by_video[video_id] != split:
            errors.append(f"video split leakage for {video_id}")
        split_by_video[video_id] = split
        if str(case.get("candidate_set_id") or "") not in candidate_set_ids:
            errors.append(f"cases.{index} references an unknown candidate set")
    hidden_ids = {str(row.get("case_id")) for row in hidden.get("cases") or []}
    if hidden_ids != case_ids:
        errors.append("hidden cases do not exactly match public cases")
    return errors


def _load_graph_nodes(
    graph_paths: Iterable[Path], wanted_videos: set[str]
) -> dict[str, list[dict[str, Any]]]:
    by_video: dict[str, list[dict[str, Any]]] = {}
    for path in sorted({Path(value).expanduser().resolve() for value in graph_paths}):
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        video_id = str(payload.get("video_id") or path.parent.name)
        if video_id not in wanted_videos or video_id in by_video:
            continue
        rows = payload.get("l1_observations") or payload.get("nodes") or []
        if payload.get("atomic_events"):
            rows = [*rows, *payload["atomic_events"]]
        nodes = []
        for row in rows:
            span = row.get("time_span") or {}
            try:
                start, end = _span_tuple(span)
            except (KeyError, TypeError, ValueError):
                continue
            if start < 0 or end <= start:
                continue
            node_id = str(row.get("node_id") or "")
            if not node_id:
                continue
            nodes.append(
                {
                    "node_id": node_id,
                    "time_span": {"start_s": start, "end_s": end},
                    "text": str(row.get("text") or ""),
                    "node_type": str(row.get("node_type") or "observation"),
                    "evidence_source": "persisted_l1_l15_graph",
                    "embedding_ref": row.get("embedding_ref"),
                }
            )
        if nodes:
            by_video[video_id] = sorted(nodes, key=lambda row: (*_span_tuple(row["time_span"]), row["node_id"]))
    return by_video


def _subtitle_candidate_nodes(video_id: str, path: Path) -> list[dict[str, Any]]:
    blocks = re.split(r"\r?\n\s*\r?\n", path.read_text(encoding="utf-8-sig").strip())
    nodes = []
    for index, block in enumerate(blocks):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None:
            continue
        bounds = [value.strip() for value in lines[timing_index].split("-->")]
        if len(bounds) != 2:
            continue
        try:
            start, end = _srt_time(bounds[0]), _srt_time(bounds[1])
        except ValueError:
            continue
        text = " ".join(lines[timing_index + 1 :]).strip()
        if end <= start or not text:
            continue
        nodes.append(
            {
                "node_id": f"subtitle:{video_id}:{index:06d}",
                "time_span": {"start_s": start, "end_s": end},
                "text": text,
                "node_type": "subtitle_observation",
                "evidence_source": "time_aligned_subtitle_l1_fallback",
                "embedding_ref": None,
            }
        )
    return nodes


def _public_hop(video_id: str, node: dict[str, Any]) -> dict[str, Any]:
    node_id = str(node["node_id"])
    return {
        "hop_id": "cghop:" + _digest(video_id + "\0" + node_id)[:20],
        "hop_type": "read_evidence_node",
        "node_id": node_id,
        "time_span": node["time_span"],
        "observation_preview": node.get("text") or "",
        "node_type": node.get("node_type"),
        "evidence_source": node.get("evidence_source"),
        "embedding_ref": node.get("embedding_ref"),
    }


def _srt_time(value: str) -> float:
    match = re.match(r"^(\d+):(\d+):(\d+)[,.](\d+)$", value)
    if not match:
        raise ValueError("invalid SRT timestamp")
    hours, minutes, seconds, millis = (int(item) for item in match.groups())
    return round(hours * 3600 + minutes * 60 + seconds + millis / (10 ** len(match.group(4))), 3)


def _span_tuple(value: dict[str, Any]) -> tuple[float, float]:
    return float(value["start_s"]), float(value["end_s"])


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(str(key) in forbidden or _contains_key(child, forbidden)
                   for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checksum(value: dict[str, Any]) -> str:
    return _digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hidden-input", required=True, type=Path)
    parser.add_argument("--graph", action="append", default=[], type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--subtitle-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--protocol-output", required=True, type=Path)
    parser.add_argument("--graph-generation-manifest-output", type=Path)
    args = parser.parse_args(argv)
    graph_paths = list(args.graph)
    if args.graph_root is not None:
        graph_paths.extend(args.graph_root.rglob("causal_temporal_overlay.json"))
        graph_paths.extend(args.graph_root.rglob("memory_graph.json"))
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    public, alignment, report = build_l15_candidate_artifact(
        dataset, hidden, graph_paths=graph_paths, subtitle_root=args.subtitle_root
    )
    _write_json(args.output, public)
    _write_json(args.hidden_output, alignment)
    _write_json(args.report, report)
    _write_json(args.protocol_output, build_closed_loop_protocol(public))
    if args.graph_generation_manifest_output is not None:
        _write_json(
            args.graph_generation_manifest_output,
            build_graph_generation_manifest(public, report),
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
