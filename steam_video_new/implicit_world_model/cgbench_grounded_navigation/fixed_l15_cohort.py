"""Freeze a held-out CG-Bench L1/L1.5 cohort and gate clue retention."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .builder import _write_json


SELECTION_SCHEMA = "steam-cgbench-fixed-l15-cohort-selection/v0.1"
PROTOCOL_SCHEMA = "steam-cgbench-fixed-l15-cohort-protocol/v0.1"
GATE_SCHEMA = "steam-cgbench-fixed-l15-cohort-gate/v0.1"
DETAIL_SCHEMA = "steam-cgbench-fixed-l15-cohort-gate-details/v0.1"
FORBIDDEN_GRAPH_KEYS = {
    "question",
    "choices",
    "answer",
    "answer_key",
    "answer_text",
    "clue_intervals",
    "hop_alignment",
}


def build_fixed_heldout_cohort(
    dataset: dict[str, Any],
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    splits: tuple[str, ...] = ("validation", "test"),
    minimum_cases: int = 30,
    maximum_cases: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select a deterministic held-out cohort without questions or clue labels.

    The current CG-Bench artifact has 49 validation/test cases on 36 videos, so
    the default keeps the complete held-out population rather than cherry-pick
    cases.  If a future artifact exceeds ``maximum_cases``, stable case-ID
    hashing is the only down-selection signal.
    """

    if minimum_cases < 1 or maximum_cases < minimum_cases:
        raise ValueError("case bounds must satisfy 1 <= minimum <= maximum")
    wanted_splits = set(splits)
    cases = [
        row
        for row in dataset.get("cases") or []
        if str(row.get("split")) in wanted_splits
    ]
    if len(cases) > maximum_cases:
        cases = sorted(cases, key=lambda row: _digest(str(row["case_id"])))[
            :maximum_cases
        ]
    cases = sorted(cases, key=lambda row: (str(row["split"]), str(row["case_id"])))
    if len(cases) < minimum_cases:
        raise ValueError(
            f"only {len(cases)} held-out cases are available; need {minimum_cases}"
        )

    manifest_by_video = {
        str(row["video_id"]): row for row in manifest.get("videos") or []
    }
    cases_by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        cases_by_video[str(case["video_id"])].append(case)

    root = dataset_root.expanduser().resolve()
    videos: list[dict[str, Any]] = []
    for video_id, video_cases in sorted(cases_by_video.items()):
        manifest_row = manifest_by_video.get(video_id)
        if manifest_row is None:
            raise ValueError(f"manifest lacks held-out video {video_id}")
        if _contains_key(manifest_row, FORBIDDEN_GRAPH_KEYS):
            raise ValueError(f"manifest row for {video_id} leaks QA or GT")
        split_values = {str(row["split"]) for row in video_cases}
        if len(split_values) != 1:
            raise ValueError(f"video {video_id} crosses dataset splits")
        durations = {round(float(row["video_duration_s"]), 3) for row in video_cases}
        if len(durations) != 1:
            raise ValueError(f"video {video_id} has inconsistent durations")
        video_ref = str(manifest_row["video_ref"])
        path = (root / video_ref).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"held-out video is unavailable: {path}")
        duration = durations.pop()
        videos.append(
            {
                "video_id": video_id,
                "video_ref": video_ref,
                "split": next(iter(split_values)),
                "fallback_source": str(
                    manifest_row.get("current_candidate_source") or "unavailable"
                ),
                "duration_s": duration,
                "observation_horizon_s": duration,
            }
        )

    case_rows = [
        {
            "case_id": str(row["case_id"]),
            "video_id": str(row["video_id"]),
            "split": str(row["split"]),
        }
        for row in cases
    ]
    selection = {
        "schema_version": SELECTION_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "selection_policy": (
            "complete_validation_and_test_population_or_stable_case_id_hash_cap"
        ),
        "selection_uses_question_or_gt": False,
        "full_video_required": True,
        "videos": videos,
        "forbidden_worker_inputs": sorted(FORBIDDEN_GRAPH_KEYS),
        "training_performed": False,
    }
    protocol = {
        "schema_version": PROTOCOL_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(case_rows),
        "video_count": len(videos),
        "splits": list(splits),
        "minimum_locked_cases": minimum_cases,
        "maximum_locked_cases": maximum_cases,
        "cases": case_rows,
        "case_selection_uses_question_text": False,
        "case_selection_uses_clue_or_answer_gt": False,
        "graph_builder_receives_case_protocol": False,
        "delayed_case_contract": (
            "structural candidates are identified only after graph freeze; actual "
            "delayed success requires an executed two-step rollout"
        ),
        "training_performed": False,
    }
    return selection, protocol


def evaluate_fixed_l15_cohort(
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    selection: dict[str, Any],
    protocol: dict[str, Any],
    *,
    graph_root: Path,
    maximum_path_hops: int = 8,
    compile_capacity: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate frozen graphs and lock only fully retained, connected cases."""

    if maximum_path_hops < 1:
        raise ValueError("maximum_path_hops must be positive")
    if compile_capacity is not None and compile_capacity < 1:
        raise ValueError("compile_capacity must be positive")
    public_by_case = {str(row["case_id"]): row for row in dataset.get("cases") or []}
    hidden_by_case = {str(row["case_id"]): row for row in hidden_key.get("cases") or []}
    selected_videos = {str(row["video_id"]): row for row in selection["videos"]}
    graph_root = graph_root.expanduser().resolve()
    graph_rows: list[dict[str, Any]] = []
    graph_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, dict[str, Any]] = {}
    for video_id, entry in sorted(selected_videos.items()):
        source_path = graph_root / video_id / "causal_temporal_overlay.json"
        graph_path = graph_root / video_id / "l1_l15_navigation_graph.json"
        row: dict[str, Any] = {
            "video_id": video_id,
            "split": entry["split"],
            "source_graph_available": source_path.is_file(),
            "compiled_graph_available": graph_path.is_file() or compile_capacity is not None,
            "status": "missing",
        }
        if source_path.is_file() and (graph_path.is_file() or compile_capacity is not None):
            source = _read_json(source_path)
            if compile_capacity is None:
                graph = _read_json(graph_path)
                graph_checksum = _file_checksum(graph_path)
            else:
                from steam_video_new.implicit_world_model.full_graph_iwm.graph_adapter import (
                    compile_l1_l15_navigation_graph,
                )
                from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
                    overlay_from_dict,
                )

                graph = compile_l1_l15_navigation_graph(
                    overlay_from_dict(source), capacity=compile_capacity
                ).graph.to_dict()
                graph_checksum = _payload_checksum(graph)
            source_metadata = source.get("metadata") or {}
            graph_metadata = graph.get("metadata") or {}
            leakage = sorted(
                _find_keys(source, FORBIDDEN_GRAPH_KEYS)
                | _find_keys(graph, FORBIDDEN_GRAPH_KEYS)
            )
            full_video = bool(source_metadata.get("full_video_scope"))
            question_independent = bool(
                source_metadata.get("question_independent_contract")
                and graph_metadata.get("question_independent")
            )
            row.update(
                {
                    "status": "loaded",
                    "full_video_scope": full_video,
                    "question_independent": question_independent,
                    "forbidden_key_paths": leakage,
                    "source_l1_node_count": len(source.get("l1_observations") or []),
                    "retained_l1_node_count": len(graph.get("nodes") or []),
                    "temporal_edge_count": len(graph.get("temporal_edges") or []),
                    "correlation_edge_count": len(graph.get("correlation_edges") or []),
                    "source_sha256": _file_checksum(source_path),
                    "graph_sha256": graph_checksum,
                    "compiled_in_memory": compile_capacity is not None,
                    "compile_capacity": compile_capacity,
                }
            )
            if full_video and question_independent and not leakage:
                graph_cache[video_id] = graph
                source_cache[video_id] = source
                row["status"] = "eligible"
        graph_rows.append(row)

    case_rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    for protocol_case in protocol.get("cases") or []:
        case_id = str(protocol_case["case_id"])
        video_id = str(protocol_case["video_id"])
        public = public_by_case.get(case_id)
        hidden = hidden_by_case.get(case_id)
        base = {
            "case_id": case_id,
            "video_id": video_id,
            "split": protocol_case["split"],
            "passed": False,
            "structural_delayed_candidate": False,
            "delayed_outcome_status": "requires_executed_rollout",
        }
        if public is None or hidden is None:
            base["failure_reason"] = "case_or_hidden_key_missing"
            failure_counts[base["failure_reason"]] += 1
            case_rows.append(base)
            continue
        graph = graph_cache.get(video_id)
        source = source_cache.get(video_id)
        if graph is None or source is None:
            base["failure_reason"] = "graph_not_eligible"
            failure_counts[base["failure_reason"]] += 1
            case_rows.append(base)
            continue
        clues = list(hidden.get("clue_intervals") or [])
        source_nodes = source.get("l1_observations") or []
        retained_nodes = graph.get("nodes") or []
        source_hits = [_overlapping_node_ids(source_nodes, clue) for clue in clues]
        retained_hits = [_overlapping_node_ids(retained_nodes, clue) for clue in clues]
        adjacency = _graph_adjacency(graph)
        bridge_hops = [
            _shortest_path_hops(set(left), set(right), adjacency, maximum_path_hops)
            for left, right in zip(retained_hits, retained_hits[1:])
        ]
        raw_complete = bool(clues) and all(source_hits)
        retained_complete = bool(clues) and all(retained_hits)
        connected = retained_complete and all(
            value is not None for value in bridge_hops
        )
        if not raw_complete:
            failure = "raw_l1_missing_clue"
        elif not retained_complete:
            failure = "bounded_consolidation_dropped_clue"
        elif not connected:
            failure = "l15_path_missing"
        else:
            failure = None
        delayed = bool(bridge_hops) and any(
            value is not None and value >= 2 for value in bridge_hops
        )
        base.update(
            {
                "clue_count": len(clues),
                "raw_l1_retained_clue_count": sum(bool(row) for row in source_hits),
                "bounded_l1_retained_clue_count": sum(
                    bool(row) for row in retained_hits
                ),
                "consecutive_clue_shortest_path_hops": bridge_hops,
                "passed": failure is None,
                "failure_reason": failure,
                "structural_delayed_candidate": failure is None and delayed,
            }
        )
        if failure is not None:
            failure_counts[failure] += 1
        case_rows.append(base)

    passed = [row for row in case_rows if row["passed"]]
    maximum_locked = int(protocol["maximum_locked_cases"])
    locked = sorted(passed, key=lambda row: (row["split"], row["case_id"]))[
        :maximum_locked
    ]
    minimum_locked = int(protocol["minimum_locked_cases"])
    delayed_count = sum(row["structural_delayed_candidate"] for row in locked)
    checks = {
        "all_requested_graphs_frozen": len(graph_cache) == len(selected_videos),
        "locked_case_count_at_least_minimum": len(locked) >= minimum_locked,
        "locked_case_count_at_most_maximum": len(locked) <= maximum_locked,
        "all_locked_cases_retain_every_clue": all(row["passed"] for row in locked),
        "structural_delayed_candidates_present": delayed_count > 0,
    }
    report = {
        "schema_version": GATE_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "selection_schema": selection.get("schema_version"),
        "protocol_schema": protocol.get("schema_version"),
        "graph_root": str(graph_root),
        "compile_capacity": compile_capacity,
        "requested_video_count": len(selected_videos),
        "requested_case_count": len(case_rows),
        "passed_case_count": len(passed),
        "locked_case_count": len(locked),
        "locked_case_ids": [row["case_id"] for row in locked],
        "locked_case_count_by_split": dict(
            sorted(Counter(row["split"] for row in locked).items())
        ),
        "structural_delayed_candidate_count": delayed_count,
        "structural_delayed_candidate_rate": (
            delayed_count / len(locked) if locked else None
        ),
        "failure_counts": dict(sorted(failure_counts.items())),
        "checks": checks,
        "gate_passed": all(checks.values()),
        "delayed_label_contract": (
            "path length >=2 marks a structural candidate only; it does not claim "
            "that the first read has zero utility or that the second read succeeds"
        ),
        "gt_used_for_graph_construction": False,
        "gt_joined_after_graph_freeze_for_evaluation": True,
        "top_k_applied": False,
        "training_performed": False,
    }
    details = {
        "schema_version": DETAIL_SCHEMA,
        "warning": "Evaluator-only: contains hidden clue-to-node alignment.",
        "graphs": graph_rows,
        "cases": [
            {
                **row,
                "clue_intervals": hidden_by_case.get(row["case_id"], {}).get(
                    "clue_intervals", []
                ),
                "source_l1_node_ids_by_clue": (
                    [
                        _overlapping_node_ids(
                            source_cache.get(row["video_id"], {}).get(
                                "l1_observations", []
                            ),
                            clue,
                        )
                        for clue in hidden_by_case.get(row["case_id"], {}).get(
                            "clue_intervals", []
                        )
                    ]
                ),
                "retained_l1_node_ids_by_clue": (
                    [
                        _overlapping_node_ids(
                            graph_cache.get(row["video_id"], {}).get("nodes", []),
                            clue,
                        )
                        for clue in hidden_by_case.get(row["case_id"], {}).get(
                            "clue_intervals", []
                        )
                    ]
                ),
            }
            for row in case_rows
        ],
        "locked_case_ids": [row["case_id"] for row in locked],
    }
    return report, details


def _graph_adjacency(graph: dict[str, Any]) -> dict[str, set[str]]:
    adjacency = {str(row["node_id"]): set() for row in graph.get("nodes") or []}
    for edge in [
        *(graph.get("temporal_edges") or []),
        *(graph.get("correlation_edges") or []),
    ]:
        src, dst = str(edge["src"]), str(edge["dst"])
        if src in adjacency and dst in adjacency:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
    return adjacency


def _shortest_path_hops(
    sources: set[str],
    targets: set[str],
    adjacency: dict[str, set[str]],
    maximum_hops: int,
) -> int | None:
    if not sources or not targets:
        return None
    if sources & targets:
        return 0
    queue = deque((node_id, 0) for node_id in sources)
    visited = set(sources)
    while queue:
        node_id, hops = queue.popleft()
        if hops >= maximum_hops:
            continue
        for neighbor in adjacency.get(node_id, ()):
            if neighbor in visited:
                continue
            next_hops = hops + 1
            if neighbor in targets:
                return next_hops
            visited.add(neighbor)
            queue.append((neighbor, next_hops))
    return None


def _overlapping_node_ids(
    nodes: Iterable[dict[str, Any]], clue: dict[str, Any]
) -> list[str]:
    start, end = float(clue["start_s"]), float(clue["end_s"])
    return sorted(
        str(node["node_id"])
        for node in nodes
        if float(node["time_span"]["start_s"]) < end
        and start < float(node["time_span"]["end_s"])
    )


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    return bool(_find_keys(value, forbidden))


def _find_keys(value: Any, forbidden: set[str], prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in forbidden:
                found.add(path)
            found.update(_find_keys(child, forbidden, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_find_keys(child, forbidden, f"{prefix}[{index}]"))
    return found


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload_checksum(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sweep_fixed_l15_capacities(
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    selection: dict[str, Any],
    protocol: dict[str, Any],
    *,
    graph_root: Path,
    capacities: Iterable[int],
    maximum_path_hops: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate capacities from frozen L1 without rewriting compiled graphs."""

    requested = tuple(sorted(set(int(value) for value in capacities)))
    if not requested or any(value < 1 for value in requested):
        raise ValueError("capacities must contain positive integers")
    reports: list[dict[str, Any]] = []
    hidden_reports: list[dict[str, Any]] = []
    for capacity in requested:
        report, details = evaluate_fixed_l15_cohort(
            dataset,
            hidden_key,
            selection,
            protocol,
            graph_root=graph_root,
            maximum_path_hops=maximum_path_hops,
            compile_capacity=capacity,
        )
        graph_rows = details["graphs"]
        retained_counts = [int(row.get("retained_l1_node_count") or 0) for row in graph_rows]
        edge_counts = [
            int(row.get("temporal_edge_count") or 0)
            + int(row.get("correlation_edge_count") or 0)
            for row in graph_rows
        ]
        reports.append(
            {
                "capacity": capacity,
                "gate_passed": report["gate_passed"],
                "passed_case_count": report["passed_case_count"],
                "locked_case_count": report["locked_case_count"],
                "structural_delayed_candidate_count": report[
                    "structural_delayed_candidate_count"
                ],
                "failure_counts": report["failure_counts"],
                "retained_node_count_total": sum(retained_counts),
                "retained_node_count_mean": (
                    sum(retained_counts) / len(retained_counts) if retained_counts else 0.0
                ),
                "navigation_edge_count_total": sum(edge_counts),
                "navigation_edge_count_mean": (
                    sum(edge_counts) / len(edge_counts) if edge_counts else 0.0
                ),
                "locked_cases_per_1000_retained_nodes": (
                    1000.0 * report["locked_case_count"] / sum(retained_counts)
                    if sum(retained_counts)
                    else None
                ),
            }
        )
        hidden_reports.append({"capacity": capacity, "gate": report, "details": details})
    selected = next((row["capacity"] for row in reports if row["gate_passed"]), None)
    public = {
        "schema_version": "steam-cgbench-fixed-l15-capacity-sweep/v0.1",
        "graph_root": str(graph_root.expanduser().resolve()),
        "capacities": list(requested),
        "selection_policy": "lowest_capacity_passing_fixed_cohort_gate",
        "selected_capacity": selected,
        "sweep_passed": selected is not None,
        "results": reports,
        "frozen_l1_reused": True,
        "vlm_calls": 0,
        "compiled_graphs_persisted": False,
        "training_performed": False,
    }
    hidden = {
        "schema_version": "steam-cgbench-fixed-l15-capacity-sweep-details/v0.1",
        "warning": "Evaluator-only: contains hidden clue-to-node alignment.",
        "capacities": hidden_reports,
    }
    return public, hidden


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select")
    select.add_argument("--dataset", required=True, type=Path)
    select.add_argument("--manifest", required=True, type=Path)
    select.add_argument("--dataset-root", required=True, type=Path)
    select.add_argument("--selection-output", required=True, type=Path)
    select.add_argument("--protocol-output", required=True, type=Path)
    select.add_argument("--split", action="append", choices=("validation", "test"))
    select.add_argument("--minimum-cases", type=int, default=30)
    select.add_argument("--maximum-cases", type=int, default=50)
    gate = subparsers.add_parser("gate")
    gate.add_argument("--dataset", required=True, type=Path)
    gate.add_argument("--hidden-key", required=True, type=Path)
    gate.add_argument("--selection", required=True, type=Path)
    gate.add_argument("--protocol", required=True, type=Path)
    gate.add_argument("--graph-root", required=True, type=Path)
    gate.add_argument("--output", required=True, type=Path)
    gate.add_argument("--details-output", required=True, type=Path)
    gate.add_argument("--maximum-path-hops", type=int, default=8)
    gate.add_argument(
        "--compile-capacity",
        type=int,
        help="Recompile from frozen L1 in memory without overwriting graph files.",
    )
    sweep = subparsers.add_parser("sweep")
    sweep.add_argument("--dataset", required=True, type=Path)
    sweep.add_argument("--hidden-key", required=True, type=Path)
    sweep.add_argument("--selection", required=True, type=Path)
    sweep.add_argument("--protocol", required=True, type=Path)
    sweep.add_argument("--graph-root", required=True, type=Path)
    sweep.add_argument("--capacity", action="append", required=True, type=int)
    sweep.add_argument("--output", required=True, type=Path)
    sweep.add_argument("--details-output", required=True, type=Path)
    sweep.add_argument("--maximum-path-hops", type=int, default=8)
    promote = subparsers.add_parser("promote-sweep")
    promote.add_argument("--sweep", required=True, type=Path)
    promote.add_argument("--sweep-details", required=True, type=Path)
    promote.add_argument("--output", required=True, type=Path)
    promote.add_argument("--details-output", required=True, type=Path)
    merge_sweeps = subparsers.add_parser("merge-sweeps")
    merge_sweeps.add_argument("--sweep", action="append", required=True, type=Path)
    merge_sweeps.add_argument(
        "--sweep-details", action="append", required=True, type=Path
    )
    merge_sweeps.add_argument("--output", required=True, type=Path)
    merge_sweeps.add_argument("--details-output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        selection, protocol = build_fixed_heldout_cohort(
            _read_json(args.dataset),
            _read_json(args.manifest),
            dataset_root=args.dataset_root,
            splits=tuple(args.split or ("validation", "test")),
            minimum_cases=args.minimum_cases,
            maximum_cases=args.maximum_cases,
        )
        _write_json(args.selection_output, selection)
        _write_json(args.protocol_output, protocol)
        print(
            json.dumps(
                {
                    "case_count": protocol["case_count"],
                    "video_count": protocol["video_count"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "promote-sweep":
        sweep = _read_json(args.sweep)
        details = _read_json(args.sweep_details)
        selected = sweep.get("selected_capacity")
        if selected is None:
            raise ValueError("capacity sweep has no passing selected capacity")
        row = next(
            (
                value
                for value in details.get("capacities") or []
                if int(value["capacity"]) == int(selected)
            ),
            None,
        )
        if row is None or row.get("gate", {}).get("gate_passed") is not True:
            raise ValueError("selected capacity lacks a passed detailed gate")
        _write_json(args.output, row["gate"])
        _write_json(args.details_output, row["details"])
        print(json.dumps({"selected_capacity": selected, "gate_passed": True}))
        return 0
    if args.command == "merge-sweeps":
        if len(args.sweep) != len(args.sweep_details):
            raise ValueError("--sweep and --sweep-details counts must match")
        public_rows: list[dict[str, Any]] = []
        hidden_rows: list[dict[str, Any]] = []
        graph_roots: set[str] = set()
        for public_path, hidden_path in zip(args.sweep, args.sweep_details):
            public = _read_json(public_path)
            hidden = _read_json(hidden_path)
            public_rows.extend(public.get("results") or [])
            hidden_rows.extend(hidden.get("capacities") or [])
            graph_roots.add(str(public.get("graph_root")))
        if len(graph_roots) != 1:
            raise ValueError("capacity sweeps do not share one frozen graph root")
        public_rows.sort(key=lambda row: int(row["capacity"]))
        hidden_rows.sort(key=lambda row: int(row["capacity"]))
        capacities = [int(row["capacity"]) for row in public_rows]
        if len(capacities) != len(set(capacities)):
            raise ValueError("capacity sweep inputs contain duplicate capacities")
        selected = next(
            (int(row["capacity"]) for row in public_rows if row["gate_passed"]),
            None,
        )
        merged = {
            "schema_version": "steam-cgbench-fixed-l15-capacity-sweep/v0.1",
            "graph_root": next(iter(graph_roots)),
            "capacities": capacities,
            "selection_policy": "lowest_capacity_passing_fixed_cohort_gate",
            "selected_capacity": selected,
            "sweep_passed": selected is not None,
            "results": public_rows,
            "frozen_l1_reused": True,
            "vlm_calls": 0,
            "compiled_graphs_persisted": False,
            "training_performed": False,
        }
        _write_json(args.output, merged)
        _write_json(
            args.details_output,
            {
                "schema_version": (
                    "steam-cgbench-fixed-l15-capacity-sweep-details/v0.1"
                ),
                "warning": "Evaluator-only: contains hidden clue-to-node alignment.",
                "capacities": hidden_rows,
            },
        )
        print(json.dumps({"selected_capacity": selected, "capacities": capacities}))
        return 0 if selected is not None else 2
    if args.command == "sweep":
        report, details = sweep_fixed_l15_capacities(
            _read_json(args.dataset),
            _read_json(args.hidden_key),
            _read_json(args.selection),
            _read_json(args.protocol),
            graph_root=args.graph_root,
            capacities=args.capacity,
            maximum_path_hops=args.maximum_path_hops,
        )
        _write_json(args.output, report)
        _write_json(args.details_output, details)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["sweep_passed"] else 2
    report, details = evaluate_fixed_l15_cohort(
        _read_json(args.dataset),
        _read_json(args.hidden_key),
        _read_json(args.selection),
        _read_json(args.protocol),
        graph_root=args.graph_root,
        maximum_path_hops=args.maximum_path_hops,
        compile_capacity=args.compile_capacity,
    )
    _write_json(args.output, report)
    _write_json(args.details_output, details)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
