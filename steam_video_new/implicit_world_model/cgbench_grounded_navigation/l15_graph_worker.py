"""Build question-independent visual L1/L1.5 graphs and evaluate them after freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from memory_graph.adaptive_windowing import (
    OpenCVPerceptualFeatureProvider,
    SelectStreamWindowProvider,
    SurpriseWindowConfig,
)
from memory_graph.embedding import Qwen3VLEmbeddingProvider, embed_memory_nodes
from memory_graph.pipeline import build_causal_temporal_overlay
from memory_graph.soft_correlation import (
    SoftCorrelationAdmissionPolicy,
    calibrate_soft_correlation_admission,
)
from memory_graph.video_l1 import (
    PayloadVideoL1Provider,
    QwenVideoL1Extractor,
    VideoL1AtomicEventExtractor,
    VideoL1Config,
)
from steam_video_new.implicit_world_model.full_graph_iwm.graph_adapter import (
    compile_l1_l15_navigation_graph,
)

from .builder import _write_json


SELECTION_SCHEMA = "steam-cgbench-l15-smoke-selection/v0.1"
BUILD_SCHEMA = "steam-cgbench-question-independent-l15-build/v0.1"
COVERAGE_SCHEMA = "steam-cgbench-frozen-l15-coverage/v0.1"
CORRELATION_EVALUATION_SCHEMA = "steam-cgbench-frozen-l15-correlation-evaluation/v0.1"
CORRELATION_COMPILE_SCHEMA = "steam-cgbench-frozen-l15-correlation-compile/v0.1"
FORBIDDEN_GRAPH_INPUTS = {
    "question",
    "choices",
    "answer",
    "answer_key",
    "answer_text",
    "clue_intervals",
    "hop_alignment",
}
FORBIDDEN_MODEL_NUMERIC_KEYS = {
    "confidence",
    "score",
    "probability",
    "reward",
    "utility",
}


def select_smoke_videos(
    manifest: dict[str, Any],
    *,
    dataset_root: Path,
    per_stratum: int = 2,
    observation_horizon_s: float = 120.0,
    duration_probe: Callable[[Path], float | None] | None = None,
) -> dict[str, Any]:
    """Select short videos by split and fallback availability without GT access."""

    if per_stratum < 1 or observation_horizon_s <= 0:
        raise ValueError("per_stratum and observation_horizon_s must be positive")
    probe = duration_probe or _video_duration
    strata: dict[tuple[str, str], list[dict[str, Any]]] = {}
    rejected: list[dict[str, str]] = []
    root = dataset_root.expanduser().resolve()
    for row in manifest.get("videos") or []:
        if _contains_key(row, FORBIDDEN_GRAPH_INPUTS):
            raise ValueError("graph-generation manifest leaks question or GT fields")
        splits = sorted({str(case["split"]) for case in row.get("cases") or []})
        if len(splits) != 1:
            raise ValueError(
                f"video {row.get('video_id')} does not have exactly one split"
            )
        path = (root / str(row["video_ref"])).resolve()
        if root not in path.parents or not path.is_file():
            rejected.append(
                {"video_id": str(row.get("video_id")), "reason": "video_missing"}
            )
            continue
        duration = probe(path)
        if duration is None:
            rejected.append(
                {"video_id": str(row.get("video_id")), "reason": "duration_unreadable"}
            )
            continue
        entry = {
            "video_id": str(row["video_id"]),
            "video_ref": str(row["video_ref"]),
            "split": splits[0],
            "fallback_source": str(
                row.get("current_candidate_source") or "unavailable"
            ),
            "duration_s": round(duration, 3),
        }
        strata.setdefault((splits[0], entry["fallback_source"]), []).append(entry)
    selected: list[dict[str, Any]] = []
    for stratum, entries in sorted(strata.items()):
        chosen = sorted(
            entries, key=lambda row: (row["duration_s"], _digest(row["video_id"]))
        )[:per_stratum]
        for row in chosen:
            selected.append({**row, "selection_stratum": list(stratum)})
    return {
        "schema_version": SELECTION_SCHEMA,
        "dataset_id": manifest.get("dataset_id"),
        "selection_policy": "shortest_by_video_duration_within_split_x_fallback_source",
        "selection_uses_question_or_gt": False,
        "observation_horizon_s": observation_horizon_s,
        "videos": sorted(
            selected,
            key=lambda row: (row["split"], row["fallback_source"], row["video_id"]),
        ),
        "rejected": rejected,
        "forbidden_worker_inputs": sorted(FORBIDDEN_GRAPH_INPUTS),
        "training_performed": False,
    }


def build_question_independent_graph(
    entry: dict[str, Any],
    *,
    dataset_root: Path,
    output_root: Path,
    video_l1_payload: dict[str, Any],
    observation_horizon_s: float,
) -> dict[str, Any]:
    """Promote persisted visual L1 into an L1.5 overlay without QA inputs."""

    if _contains_key(entry, FORBIDDEN_GRAPH_INPUTS):
        raise ValueError("graph worker entry contains question or hidden GT")
    video_id = str(entry["video_id"])
    root = dataset_root.expanduser().resolve()
    video_path = (root / str(entry["video_ref"])).resolve()
    if root not in video_path.parents or not video_path.is_file():
        raise ValueError(f"video unavailable: {video_path}")
    declared_duration = float(entry.get("duration_s") or observation_horizon_s)
    full_video = observation_horizon_s >= declared_duration - 1e-3
    canonical = {
        "schema_version": "steam-cgbench-question-independent-video/v0.1",
        "example_id": f"cgbench-memory:{video_id}",
        "dataset": "CG-Bench",
        "video": {"video_id": video_id, "primary_path": str(video_path)},
        "available_inputs": {"mode": "video_only"},
        "evidence_index": {
            "index_id": f"cgbench-video-index:{video_id}",
            "nodes": [],
            "edges": [],
            "clip_policy": {"observation_end_s": observation_horizon_s},
        },
        "metadata": {
            "video_regime": (
                "full_video_question_independent"
                if full_video
                else "streaming_prefix_smoke"
            ),
            "question_independent": True,
        },
    }
    result = build_causal_temporal_overlay(
        canonical,
        input_mode="video_only",
        event_extractor=VideoL1AtomicEventExtractor(),
        video_l1_provider=PayloadVideoL1Provider(video_l1_payload),
        allow_visual_l1_replacement=True,
        apply_hard_verifiers=True,
        memory_capacity=64,
    )
    payload = result.to_dict()
    payload["metadata"].update(
        {
            "question_independent_contract": True,
            "builder_inputs": ["raw_video_frames"],
            "forbidden_inputs_absent": sorted(FORBIDDEN_GRAPH_INPUTS),
            "l1_windowing": video_l1_payload.get("windowing")
            or {"mode": "legacy_unspecified"},
            "observation_end_s": observation_horizon_s,
            "observation_horizon_s": observation_horizon_s,
            "smoke_prefix_only": not full_video,
            "full_video_scope": full_video,
            "training_performed": False,
        }
    )
    sample_dir = output_root / video_id
    _write_json(sample_dir / "causal_temporal_overlay.json", payload)
    return payload


def evaluate_frozen_graph_coverage(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    selection: dict[str, Any],
    *,
    graph_root: Path,
    query_provider: Any,
    top_ks: tuple[int, ...] = (4, 8, 16, 32),
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join hidden clues only after graph freeze and evaluate embedding retrieval."""

    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("coverage evaluation requires numpy") from exc
    selected = {str(row["video_id"]): row for row in selection.get("videos") or []}
    hidden_by_case = {row["case_id"]: row for row in hidden.get("cases") or []}
    totals = {k: {"eligible": 0, "covered": 0} for k in top_ks}
    native_eligible = 0
    native_covered = 0
    details: list[dict[str, Any]] = []
    graph_checksums: dict[str, str] = {}
    for case in dataset.get("cases") or []:
        video_id = str(case["video_id"])
        if video_id not in selected:
            continue
        graph_path = graph_root / video_id / "causal_temporal_overlay.json"
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata") or {}
        if metadata.get("question_independent_contract") is not True:
            raise ValueError(f"graph {video_id} lacks question-independent contract")
        if _contains_key(
            {
                key: value
                for key, value in payload.items()
                if key not in {"build_report"}
            },
            FORBIDDEN_GRAPH_INPUTS,
        ):
            raise ValueError(f"graph {video_id} contains a forbidden QA/GT key")
        graph_checksums[video_id] = _file_checksum(graph_path)
        nodes = [
            *(payload.get("l1_observations") or []),
            *(payload.get("atomic_events") or []),
        ]
        matrix_path = graph_root / video_id / "node_embeddings.npy"
        matrix = np.load(matrix_path)
        manifest = json.loads(
            matrix_path.with_suffix(".manifest.json").read_text(encoding="utf-8")
        )
        rows = sorted(manifest.get("rows") or [], key=lambda row: int(row["row_index"]))
        node_by_id = {str(row["node_id"]): row for row in nodes}
        if matrix.shape != (len(rows), 2048) or any(
            str(row["node_id"]) not in node_by_id for row in rows
        ):
            raise ValueError(f"graph {video_id} embedding alignment is invalid")
        question = str(case["planner_input"]["question"])
        query = np.asarray(
            query_provider.encode([question], batch_size=1)[0], dtype=np.float32
        )
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        order = np.argsort(-(matrix @ query))
        ranked_ids = [str(rows[int(index)]["node_id"]) for index in order]
        key = hidden_by_case[str(case["case_id"])]
        horizon = float(
            selected[video_id].get("observation_horizon_s")
            or selection["observation_horizon_s"]
        )
        clue_rows = []
        for clue_index, clue in enumerate(key.get("clue_intervals") or []):
            clue_span = _span(clue)
            eligible = clue_span[1] <= horizon + 1e-6
            native_hits = [
                node_id
                for node_id, node in node_by_id.items()
                if _overlaps(_span(node["time_span"]), clue_span)
            ]
            if eligible:
                native_eligible += 1
                native_covered += bool(native_hits)
            topk_status: dict[str, str] = {}
            for top_k in top_ks:
                covered = bool(set(ranked_ids[:top_k]) & set(native_hits))
                if eligible:
                    totals[top_k]["eligible"] += 1
                    totals[top_k]["covered"] += covered
                topk_status[str(top_k)] = "covered" if covered else "not_covered"
            clue_rows.append(
                {
                    "clue_index": clue_index,
                    "clue_interval": clue,
                    "within_smoke_horizon": eligible,
                    "native_overlap_node_ids": native_hits,
                    "top_k_status": topk_status,
                }
            )
        details.append(
            {
                "case_id": case["case_id"],
                "video_id": video_id,
                "graph_sha256": graph_checksums[video_id],
                "ranked_node_ids": ranked_ids,
                "clues": clue_rows,
            }
        )
    top_k_report = {
        str(k): {
            **values,
            "recall": (
                values["covered"] / values["eligible"] if values["eligible"] else None
            ),
        }
        for k, values in totals.items()
    }
    report = {
        "schema_version": COVERAGE_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "graph_video_count": len(graph_checksums),
        "graph_checksums": graph_checksums,
        "eligible_clue_count_within_smoke_horizon": native_eligible,
        "native_candidate_covered_clue_count": native_covered,
        "native_candidate_recall": (
            native_covered / native_eligible if native_eligible else None
        ),
        "embedding_top_k_recall": top_k_report,
        "out_of_horizon_clues_excluded_from_recall": True,
        "nonoverlap_candidates_are_semantic_negatives": False,
        "query_embedding_model": query_provider.model_name,
        "human_review_required": False,
        "training_performed": False,
    }
    hidden_details = {
        "schema_version": "steam-cgbench-frozen-l15-coverage-details/v0.1",
        "warning": "Contains questions, GT clue intervals, and rankings; evaluator only.",
        "cases": details,
    }
    return report, hidden_details


def _extract_command(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selection = _select_video_shard(
        selection,
        video_limit=args.video_limit,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    video_skills_root = str(args.video_skills_root.expanduser().resolve())
    if video_skills_root not in sys.path:
        sys.path.insert(0, video_skills_root)
    from atomic_skills.skill_model_client import SkillModelClient

    client = _CategoricalGroundingClient(
        SkillModelClient.from_local(
            model=args.model,
            base_url=args.api_base,
            max_tokens=1400,
            timeout_s=180,
        )
    )
    extractor = QwenVideoL1Extractor(
        client=client,
        config=VideoL1Config(
            coarse_window_s=args.coarse_window_s,
            coarse_stride_s=args.coarse_stride_s,
            frames_per_coarse_window=args.coarse_frames,
            frames_per_fine_window=args.fine_frames,
            minimum_confidence=0.5,
        ),
        window_provider=(
            SelectStreamWindowProvider(
                OpenCVPerceptualFeatureProvider(),
                SurpriseWindowConfig(
                    sample_period_s=args.surprise_sample_period_s,
                    min_window_s=args.surprise_min_window_s,
                    max_window_s=args.surprise_max_window_s,
                    calibration_history=args.surprise_calibration_history,
                    high_surprise_quantile=args.surprise_quantile,
                ),
            )
            if args.windowing == "surprise-opencv-smoke"
            else None
        ),
    )
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in selection.get("videos") or []:
        video_id = str(entry["video_id"])
        observation_horizon_s = float(
            entry.get("observation_horizon_s")
            or selection["observation_horizon_s"]
        )
        sample_dir = args.output_root / video_id
        overlay_path = sample_dir / "causal_temporal_overlay.json"
        if overlay_path.is_file():
            existing = json.loads(overlay_path.read_text(encoding="utf-8"))
            metadata = existing.get("metadata") or {}
            if (
                metadata.get("question_independent_contract") is True
                and (metadata.get("l1_windowing") or {}).get("mode") == args.windowing
                and abs(
                    float(metadata.get("observation_end_s") or -1.0)
                    - observation_horizon_s
                )
                <= 1e-3
            ):
                completed.append(_graph_summary(existing, status="resumed"))
                continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        l1_path = sample_dir / "video_l1.json"
        try:
            l1_payload = (
                json.loads(l1_path.read_text(encoding="utf-8"))
                if l1_path.is_file()
                else None
            )
            if (
                not isinstance(l1_payload, dict)
                or (l1_payload.get("windowing") or {}).get("mode") != args.windowing
                or abs(float(l1_payload.get("duration_s") or -1.0) - observation_horizon_s)
                > 1e-3
            ):
                video_path = (args.dataset_root / str(entry["video_ref"])).resolve()
                result = extractor.extract(
                    video_path=video_path,
                    video_id=video_id,
                    observation_end_s=observation_horizon_s,
                )
                l1_payload = result.to_dict()
                l1_payload["windowing"] = {
                    "mode": args.windowing,
                    "provider": (
                        extractor.window_provider.provider_name
                        if extractor.window_provider is not None
                        else "fixed_coarse_windows_baseline"
                    ),
                    "question_independent": True,
                    "formal_learned_representation": False,
                }
                _write_json(l1_path, l1_payload)
            assert isinstance(l1_payload, dict)
            graph = build_question_independent_graph(
                entry,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                video_l1_payload=l1_payload,
                observation_horizon_s=observation_horizon_s,
            )
            completed.append(_graph_summary(graph, status="completed"))
        except Exception as exc:  # checkpoint every independent video
            errors.append(
                {"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        _write_json(
            args.report,
            _build_run_report(selection, completed, errors, status="running"),
        )
    report = _build_run_report(
        selection, completed, errors, status="completed" if not errors else "partial"
    )
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def _merge_extract_reports_command(args: argparse.Namespace) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    report = merge_extract_reports(
        selection,
        graph_root=args.graph_root,
        shard_report_dir=args.shard_report_dir,
        num_shards=args.num_shards,
        video_limit=args.video_limit,
    )
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _embed_evaluate_command(args: argparse.Namespace) -> int:
    from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
        overlay_from_dict,
    )

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    selection = _limit_selection(selection, args.video_limit)
    provider = Qwen3VLEmbeddingProvider(device=args.device)
    for entry in selection.get("videos") or []:
        sample_dir = args.graph_root / str(entry["video_id"])
        graph_path = sample_dir / "causal_temporal_overlay.json"
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        overlay = overlay_from_dict(payload)
        nodes = [*overlay.l1_observations, *overlay.atomic_events]
        embed_memory_nodes(
            nodes, provider, output_path=sample_dir / "node_embeddings.npy"
        )
        embedded = overlay.to_dict()
        embedded["build_report"] = payload.get("build_report") or {}
        _write_json(graph_path, embedded)
        compiled = compile_l1_l15_navigation_graph(
            overlay,
            capacity=args.memory_capacity,
        )
        _write_json(
            sample_dir / "l1_l15_navigation_graph.json",
            compiled.graph.to_dict(),
        )
        _write_json(
            sample_dir / "l1_l15_correlation_pair_audit.json",
            compiled.correlation_pair_audit,
        )
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    report, details = evaluate_frozen_graph_coverage(
        dataset, hidden, selection, graph_root=args.graph_root, query_provider=provider
    )
    _write_json(args.coverage_report, report)
    _write_json(args.coverage_details, details)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def compile_frozen_l15_correlation_artifacts(
    selection: dict[str, Any],
    *,
    graph_root: Path,
    memory_capacity: int,
    video_limit: int | None = None,
    admission_policy: SoftCorrelationAdmissionPolicy | None = None,
) -> dict[str, Any]:
    """Compile L1.5 and pair audits while proving persisted L1 is unchanged."""

    from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
        overlay_from_dict,
    )

    limited = _limit_selection(selection, video_limit)
    policy = admission_policy or SoftCorrelationAdmissionPolicy()
    samples: list[dict[str, Any]] = []
    for entry in limited.get("videos") or []:
        video_id = str(entry["video_id"])
        sample_dir = graph_root / video_id
        overlay_path = sample_dir / "causal_temporal_overlay.json"
        if not overlay_path.is_file():
            raise FileNotFoundError(f"missing frozen L1 overlay: {overlay_path}")
        source_checksum_before = _file_checksum(overlay_path)
        payload = json.loads(overlay_path.read_text(encoding="utf-8"))
        if (payload.get("metadata") or {}).get("question_independent_contract") is not True:
            raise ValueError(f"graph {video_id} lacks question-independent contract")
        if _contains_key(payload, FORBIDDEN_GRAPH_INPUTS):
            raise ValueError(f"graph {video_id} contains forbidden QA/GT input")
        compiled = compile_l1_l15_navigation_graph(
            overlay_from_dict(payload),
            capacity=memory_capacity,
            admission_policy=policy,
        )
        if _file_checksum(overlay_path) != source_checksum_before:
            raise RuntimeError(f"L1.5 compilation mutated frozen L1 for {video_id}")
        graph_path = sample_dir / "l1_l15_navigation_graph.json"
        audit_path = sample_dir / "l1_l15_correlation_pair_audit.json"
        multichannel_audit_path = sample_dir / "l1_l15_multichannel_pair_audit.json"
        _write_json(graph_path, compiled.graph.to_dict())
        _write_json(audit_path, compiled.correlation_pair_audit)
        _write_json(multichannel_audit_path, compiled.multichannel_pair_audit)
        correlation_build = compiled.graph.metadata.get("correlation_build") or {}
        samples.append(
            {
                "video_id": video_id,
                "split": entry.get("split"),
                "source_overlay_sha256": source_checksum_before,
                "source_l1_fingerprint": compiled.source_l1_fingerprint,
                "retained_l1_fingerprint": compiled.retained_l1_fingerprint,
                "retained_node_count": len(compiled.graph.nodes),
                "temporal_edge_count": len(compiled.graph.temporal_edges),
                "correlation_edge_count": len(compiled.graph.correlation_edges),
                "maximum_degree": correlation_build.get("maximum_degree"),
                "edge_density": correlation_build.get("edge_density"),
                "pair_audit_row_count": correlation_build.get("pair_audit_row_count"),
                "multichannel_pair_audit_path": str(multichannel_audit_path),
                "multichannel_signal_counts": compiled.multichannel_pair_audit.get(
                    "channel_signal_counts", {}
                ),
                "graph_path": str(graph_path),
                "graph_sha256": _file_checksum(graph_path),
                "pair_audit_path": str(audit_path),
                "pair_audit_sha256": _file_checksum(audit_path),
                "source_l1_unchanged": True,
            }
        )
    return {
        "schema_version": CORRELATION_COMPILE_SCHEMA,
        "dataset_id": limited.get("dataset_id"),
        "video_count": len(samples),
        "memory_capacity": memory_capacity,
        "admission_policy": policy.to_dict(),
        "admission_policy_fingerprint": policy.fingerprint,
        "l1_source_artifacts_unchanged": all(
            row["source_l1_unchanged"] for row in samples
        ),
        "question_or_gt_used_for_graph_building": False,
        "per_pair_llm_calls": 0,
        "top_k_applied": False,
        "samples": samples,
        "training_performed": False,
    }


def evaluate_frozen_l15_correlations(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    selection: dict[str, Any],
    *,
    graph_root: Path,
    target_positive_coverage: float = 0.9,
    max_path_hops: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate frozen topology with hidden clues after construction completes.

    GT clue pairs provide positive navigation bridges only.  Unmatched pairs
    remain unlabeled, so admitted-edge precision is intentionally unavailable
    until trusted negative controls exist.
    """

    if max_path_hops < 1:
        raise ValueError("max_path_hops must be positive")
    selected = {str(row["video_id"]): row for row in selection.get("videos") or []}
    hidden_by_case = {str(row["case_id"]): row for row in hidden.get("cases") or []}
    totals = _empty_bridge_metrics()
    split_totals: dict[str, dict[str, int]] = {}
    details: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    train_positive_similarities: list[float] = []
    heldout_positive_similarities: list[float] = []
    seen_graphs: set[str] = set()
    for case in dataset.get("cases") or []:
        video_id = str(case["video_id"])
        if video_id not in selected:
            continue
        case_id = str(case["case_id"])
        key = hidden_by_case.get(case_id)
        if key is None:
            raise ValueError(f"missing hidden clue key for {case_id}")
        graph_path = graph_root / video_id / "l1_l15_navigation_graph.json"
        audit_path = graph_root / video_id / "l1_l15_correlation_pair_audit.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if audit.get("contains_question_or_answer") is not False:
            raise ValueError(f"pair audit for {video_id} lacks leakage contract")
        metadata = graph.get("metadata") or {}
        if metadata.get("question_independent") is not True:
            raise ValueError(f"compiled graph {video_id} is not question-independent")
        nodes = {str(row["node_id"]): row for row in graph.get("nodes") or []}
        correlations = {
            frozenset((str(row["src"]), str(row["dst"])))
            for row in graph.get("correlation_edges") or []
        }
        temporal = {
            frozenset((str(row["src"]), str(row["dst"])))
            for row in graph.get("temporal_edges") or []
        }
        adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
        for pair in correlations | temporal:
            endpoints = tuple(pair)
            if len(endpoints) == 2:
                left_id, right_id = endpoints
                adjacency.setdefault(left_id, set()).add(right_id)
                adjacency.setdefault(right_id, set()).add(left_id)
        pair_rows = {
            frozenset((str(row["src"]), str(row["dst"]))): row
            for row in audit.get("pairs") or []
        }
        if video_id not in seen_graphs:
            seen_graphs.add(video_id)
            degree: dict[str, int] = {node_id: 0 for node_id in nodes}
            for pair in correlations:
                for node_id in pair:
                    degree[node_id] = degree.get(node_id, 0) + 1
            graph_rows.append(
                {
                    "video_id": video_id,
                    "node_count": len(nodes),
                    "correlation_edge_count": len(correlations),
                    "maximum_degree": max(degree.values(), default=0),
                    "mean_degree": (
                        sum(degree.values()) / len(degree) if degree else 0.0
                    ),
                    "source_l1_fingerprint": metadata.get("source_l1_fingerprint"),
                    "retained_l1_fingerprint": metadata.get("retained_l1_fingerprint"),
                    "graph_sha256": _file_checksum(graph_path),
                }
            )
        horizon = float(
            selected[video_id].get("observation_horizon_s")
            or selection.get("observation_horizon_s")
            or float("inf")
        )
        clue_rows: list[dict[str, Any]] = []
        for clue in key.get("clue_intervals") or []:
            clue_span = _span(clue)
            node_ids = sorted(
                node_id
                for node_id, node in nodes.items()
                if _overlaps(_span(node["time_span"]), clue_span)
            )
            clue_rows.append(
                {
                    "interval": clue,
                    "within_horizon": clue_span[1] <= horizon + 1e-6,
                    "node_ids": node_ids,
                }
            )
        case_bridges: list[dict[str, Any]] = []
        split = str(case.get("split") or "unspecified")
        split_metrics = split_totals.setdefault(split, _empty_bridge_metrics())
        for bridge_index, (left, right) in enumerate(zip(clue_rows, clue_rows[1:])):
            if not left["within_horizon"] or not right["within_horizon"]:
                totals["out_of_horizon"] += 1
                split_metrics["out_of_horizon"] += 1
                continue
            if not left["node_ids"] or not right["node_ids"]:
                totals["missing_l1_overlap"] += 1
                split_metrics["missing_l1_overlap"] += 1
                continue
            same_nodes = sorted(set(left["node_ids"]) & set(right["node_ids"]))
            cross_pairs = {
                frozenset((left_id, right_id))
                for left_id in left["node_ids"]
                for right_id in right["node_ids"]
                if left_id != right_id
            }
            correlation_hits = sorted(
                (tuple(sorted(pair)) for pair in cross_pairs & correlations)
            )
            temporal_hits = sorted((tuple(sorted(pair)) for pair in cross_pairs & temporal))
            similarities = [
                float(pair_rows[pair]["features"]["semantic_similarity"])
                for pair in cross_pairs
                if pair in pair_rows
            ]
            max_similarity = max(similarities) if similarities else None
            direct_covered = bool(same_nodes or correlation_hits or temporal_hits)
            shortest_path_hops = _shortest_path_hops(
                set(left["node_ids"]),
                set(right["node_ids"]),
                adjacency,
                max_hops=max_path_hops,
            )
            covered = shortest_path_hops is not None
            totals["eligible"] += 1
            split_metrics["eligible"] += 1
            totals["covered"] += int(covered)
            split_metrics["covered"] += int(covered)
            totals["direct_covered"] += int(direct_covered)
            split_metrics["direct_covered"] += int(direct_covered)
            totals["multi_hop_only"] += int(covered and not direct_covered)
            split_metrics["multi_hop_only"] += int(covered and not direct_covered)
            if shortest_path_hops is not None:
                totals["shortest_path_hop_sum"] += shortest_path_hops
                split_metrics["shortest_path_hop_sum"] += shortest_path_hops
            totals["soft_correlation"] += int(bool(correlation_hits))
            split_metrics["soft_correlation"] += int(bool(correlation_hits))
            totals["temporal"] += int(bool(temporal_hits))
            split_metrics["temporal"] += int(bool(temporal_hits))
            totals["same_node"] += int(bool(same_nodes))
            split_metrics["same_node"] += int(bool(same_nodes))
            if max_similarity is not None:
                if split == "train":
                    train_positive_similarities.append(max_similarity)
                else:
                    heldout_positive_similarities.append(max_similarity)
            case_bridges.append(
                {
                    "bridge_index": bridge_index,
                    "left_clue": left,
                    "right_clue": right,
                    "same_node_ids": same_nodes,
                    "soft_correlation_hits": correlation_hits,
                    "temporal_hits": temporal_hits,
                    "max_raw_similarity": max_similarity,
                    "direct_covered": direct_covered,
                    "shortest_path_hops": shortest_path_hops,
                    "max_path_hops": max_path_hops,
                    "covered": covered,
                }
            )
        details.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "split": split,
                "bridges": case_bridges,
            }
        )
    calibration: dict[str, Any]
    if train_positive_similarities:
        _, calibration = calibrate_soft_correlation_admission(
            train_positive_similarities,
            target_positive_coverage=target_positive_coverage,
            calibration_source="cgbench_train_gt_consecutive_clue_positive_pairs",
            calibration_split="train",
        )
        threshold = float(calibration["selected_threshold"])
        calibration["heldout_positive_example_count"] = len(
            heldout_positive_similarities
        )
        calibration["heldout_positive_coverage"] = (
            sum(value >= threshold for value in heldout_positive_similarities)
            / len(heldout_positive_similarities)
            if heldout_positive_similarities
            else None
        )
        calibration["application_status"] = "diagnostic_only_not_applied_to_graph"
    else:
        calibration = {
            "schema_version": "steam-soft-l1.5-calibration/v0.1",
            "status": "unavailable_no_train_positive_bridges_within_horizon",
            "application_status": "not_applied",
        }
    report = {
        "schema_version": CORRELATION_EVALUATION_SCHEMA,
        "dataset_id": dataset.get("dataset_id"),
        "graph_video_count": len(graph_rows),
        "maximum_evaluated_path_hops": max_path_hops,
        "bridge_metrics": _finalize_bridge_metrics(totals),
        "bridge_metrics_by_split": {
            split: _finalize_bridge_metrics(values)
            for split, values in sorted(split_totals.items())
        },
        "graph_structure": {
            "minimum_node_count": min((row["node_count"] for row in graph_rows), default=0),
            "maximum_node_count": max((row["node_count"] for row in graph_rows), default=0),
            "minimum_edge_count": min(
                (row["correlation_edge_count"] for row in graph_rows), default=0
            ),
            "maximum_edge_count": max(
                (row["correlation_edge_count"] for row in graph_rows), default=0
            ),
            "maximum_degree": max((row["maximum_degree"] for row in graph_rows), default=0),
            "mean_degree": (
                sum(float(row["mean_degree"]) for row in graph_rows) / len(graph_rows)
                if graph_rows
                else 0.0
            ),
        },
        "calibration": calibration,
        "admitted_edge_precision": None,
        "precision_status": "unavailable_no_trusted_negative_labels",
        "unmatched_pairs_treated_as_negative": False,
        "gt_used_for_graph_construction": False,
        "gt_joined_after_graph_freeze_for_evaluation": True,
        "top_k_applied": False,
        "training_performed": False,
    }
    hidden_details = {
        "schema_version": f"{CORRELATION_EVALUATION_SCHEMA}-details",
        "warning": "Evaluator-only: contains GT clue intervals and graph alignment.",
        "graphs": graph_rows,
        "cases": details,
    }
    return report, hidden_details


def _audit_correlations_command(args: argparse.Namespace) -> int:
    selection = _limit_selection(
        json.loads(args.selection.read_text(encoding="utf-8")), args.video_limit
    )
    policy = (
        SoftCorrelationAdmissionPolicy.from_dict(
            json.loads(args.correlation_policy.read_text(encoding="utf-8"))
        )
        if args.correlation_policy is not None
        else SoftCorrelationAdmissionPolicy()
    )
    compile_report = compile_frozen_l15_correlation_artifacts(
        selection,
        graph_root=args.graph_root,
        memory_capacity=args.memory_capacity,
        admission_policy=policy,
    )
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    evaluation, details = evaluate_frozen_l15_correlations(
        dataset,
        hidden,
        selection,
        graph_root=args.graph_root,
        target_positive_coverage=args.target_positive_coverage,
        max_path_hops=args.max_path_hops,
    )
    report = {**evaluation, "compile": compile_report}
    _write_json(args.report, report)
    _write_json(args.details, details)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _empty_bridge_metrics() -> dict[str, int]:
    return {
        "eligible": 0,
        "covered": 0,
        "direct_covered": 0,
        "multi_hop_only": 0,
        "shortest_path_hop_sum": 0,
        "soft_correlation": 0,
        "temporal": 0,
        "same_node": 0,
        "missing_l1_overlap": 0,
        "out_of_horizon": 0,
    }


def _finalize_bridge_metrics(values: dict[str, int]) -> dict[str, Any]:
    eligible = values["eligible"]
    return {
        **values,
        "coverage": values["covered"] / eligible if eligible else None,
        "direct_coverage": values["direct_covered"] / eligible if eligible else None,
        "multi_hop_only_coverage": (
            values["multi_hop_only"] / eligible if eligible else None
        ),
        "mean_shortest_path_hops": (
            values["shortest_path_hop_sum"] / values["covered"]
            if values["covered"]
            else None
        ),
        "soft_correlation_coverage": (
            values["soft_correlation"] / eligible if eligible else None
        ),
        "temporal_coverage": values["temporal"] / eligible if eligible else None,
        "same_node_coverage": values["same_node"] / eligible if eligible else None,
    }


def _shortest_path_hops(
    sources: set[str],
    targets: set[str],
    adjacency: dict[str, set[str]],
    *,
    max_hops: int,
) -> int | None:
    if sources & targets:
        return 0
    visited = set(sources)
    frontier = set(sources)
    for hops in range(1, max_hops + 1):
        frontier = {
            neighbor
            for node_id in frontier
            for neighbor in adjacency.get(node_id, ())
            if neighbor not in visited
        }
        if not frontier:
            return None
        if frontier & targets:
            return hops
        visited.update(frontier)
    return None


def _graph_summary(payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "video_id": payload.get("video_id"),
        "status": status,
        "l1_observation_count": len(payload.get("l1_observations") or []),
        "atomic_event_count": len(payload.get("atomic_events") or []),
        "relation_count": len(payload.get("relations") or []),
        "l1_windowing": (payload.get("metadata") or {}).get("l1_windowing"),
    }


def _limit_selection(
    selection: dict[str, Any], video_limit: int | None
) -> dict[str, Any]:
    if video_limit is None:
        return selection
    if video_limit < 1:
        raise ValueError("video_limit must be positive")
    return {**selection, "videos": list(selection.get("videos") or [])[:video_limit]}


def _select_video_shard(
    selection: dict[str, Any],
    *,
    video_limit: int | None,
    shard_index: int | None,
    num_shards: int | None,
) -> dict[str, Any]:
    """Choose a deterministic, disjoint video shard after applying the limit."""

    limited = _limit_selection(selection, video_limit)
    if shard_index is None and num_shards is None:
        return limited
    if shard_index is None or num_shards is None:
        raise ValueError("shard_index and num_shards must be provided together")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must satisfy 0 <= shard_index < num_shards")
    videos = list(limited.get("videos") or [])
    return {
        **limited,
        "videos": videos[shard_index::num_shards],
        "shard": {
            "index": shard_index,
            "count": num_shards,
            "assignment": "ordered_round_robin_after_video_limit",
            "unsharded_requested_video_count": len(videos),
        },
    }


def merge_extract_reports(
    selection: dict[str, Any],
    *,
    graph_root: Path,
    shard_report_dir: Path,
    num_shards: int,
    video_limit: int | None,
) -> dict[str, Any]:
    """Validate all shard reports and graph artifacts before embedding starts."""

    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    limited = _limit_selection(selection, video_limit)
    expected_ids = [str(row["video_id"]) for row in limited.get("videos") or []]
    samples_by_video: dict[str, dict[str, Any]] = {}
    shard_reports: list[dict[str, Any]] = []
    for shard_index in range(num_shards):
        path = (
            shard_report_dir / f"build_report.shard_{shard_index}_of_{num_shards}.json"
        )
        if not path.is_file():
            raise FileNotFoundError(f"missing shard report: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected_shard = _select_video_shard(
            selection,
            video_limit=video_limit,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        expected_shard_ids = [
            str(row["video_id"]) for row in expected_shard.get("videos") or []
        ]
        reported_ids = [
            str(row.get("video_id")) for row in payload.get("samples") or []
        ]
        if payload.get("schema_version") != BUILD_SCHEMA:
            raise ValueError(f"shard {shard_index} has an unexpected schema")
        if payload.get("status") != "completed" or payload.get("errors"):
            raise ValueError(f"shard {shard_index} did not complete cleanly")
        if reported_ids != expected_shard_ids:
            raise ValueError(
                f"shard {shard_index} videos differ: "
                f"expected {expected_shard_ids}, got {reported_ids}"
            )
        for sample in payload.get("samples") or []:
            video_id = str(sample["video_id"])
            if video_id in samples_by_video:
                raise ValueError(f"duplicate video across shards: {video_id}")
            graph_path = graph_root / video_id / "causal_temporal_overlay.json"
            l1_path = graph_root / video_id / "video_l1.json"
            if not graph_path.is_file() or not l1_path.is_file():
                raise FileNotFoundError(f"incomplete graph artifact for {video_id}")
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            metadata = graph.get("metadata") or {}
            if metadata.get("question_independent_contract") is not True:
                raise ValueError(
                    f"graph {video_id} lacks question-independent contract"
                )
            samples_by_video[video_id] = sample
        shard_reports.append(
            {
                "shard_index": shard_index,
                "path": str(path),
                "sha256": _file_checksum(path),
                "video_ids": reported_ids,
            }
        )
    if set(samples_by_video) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(samples_by_video))
        unexpected = sorted(set(samples_by_video) - set(expected_ids))
        raise ValueError(
            f"merged shard coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    return {
        "schema_version": BUILD_SCHEMA,
        "dataset_id": limited.get("dataset_id"),
        "status": "completed",
        "requested_video_count": len(expected_ids),
        "completed_video_count": len(expected_ids),
        "error_count": 0,
        "samples": [samples_by_video[video_id] for video_id in expected_ids],
        "errors": [],
        "parallel_extraction": {
            "num_shards": num_shards,
            "assignment": "ordered_round_robin_after_video_limit",
            "shard_reports": shard_reports,
        },
        "question_or_gt_used_for_graph_building": False,
        "model_numeric_confidence_requested": False,
        "model_numeric_confidence_rejected": True,
        "training_performed": False,
    }


def _build_run_report(
    selection: dict[str, Any],
    completed: list[dict[str, Any]],
    errors: list[dict[str, str]],
    *,
    status: str,
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_SCHEMA,
        "dataset_id": selection.get("dataset_id"),
        "status": status,
        "requested_video_count": len(selection.get("videos") or []),
        "completed_video_count": len(completed),
        "error_count": len(errors),
        "samples": completed,
        "errors": errors,
        "selection_shard": selection.get("shard"),
        "question_or_gt_used_for_graph_building": False,
        "model_numeric_confidence_requested": False,
        "model_numeric_confidence_rejected": True,
        "training_performed": False,
    }


def _video_duration(path: Path) -> float | None:
    try:
        import cv2
    except ImportError:
        return None
    capture = cv2.VideoCapture(str(path))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    return count / fps if fps > 0 and count > 0 else None


class _CategoricalGroundingClient:
    """Reject forbidden numeric judgment fields while retaining evidence timestamps."""

    def __init__(self, client: Any) -> None:
        self.client = client
        self.model = str(getattr(client, "model", "Qwen/Qwen3.5-9B"))

    def perceive(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        payload = self.client.perceive(*args, **kwargs)
        forbidden = _find_keys(payload, FORBIDDEN_MODEL_NUMERIC_KEYS)
        if forbidden:
            raise ValueError(
                "model returned forbidden numeric judgment fields: "
                + ", ".join(sorted(forbidden))
            )
        return payload


def _span(value: dict[str, Any]) -> tuple[float, float]:
    return float(value["start_s"]), float(value["end_s"])


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(
            str(key).casefold() in forbidden or _contains_key(child, forbidden)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def _find_keys(value: Any, forbidden: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).casefold()
            if normalized in forbidden:
                found.add(normalized)
            found.update(_find_keys(child, forbidden))
    elif isinstance(value, list):
        for child in value:
            found.update(_find_keys(child, forbidden))
    return found


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_checksum(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select")
    select.add_argument("--manifest", required=True, type=Path)
    select.add_argument("--dataset-root", required=True, type=Path)
    select.add_argument("--output", required=True, type=Path)
    select.add_argument("--per-stratum", type=int, default=2)
    select.add_argument("--observation-horizon-s", type=float, default=120.0)
    extract = sub.add_parser("extract")
    extract.add_argument("--selection", required=True, type=Path)
    extract.add_argument("--dataset-root", required=True, type=Path)
    extract.add_argument("--video-skills-root", required=True, type=Path)
    extract.add_argument("--output-root", required=True, type=Path)
    extract.add_argument("--report", required=True, type=Path)
    extract.add_argument("--model", default="Qwen/Qwen3.5-9B")
    extract.add_argument("--api-base", required=True)
    extract.add_argument("--coarse-window-s", type=float, default=8.0)
    extract.add_argument("--coarse-stride-s", type=float, default=6.0)
    extract.add_argument("--coarse-frames", type=int, default=8)
    extract.add_argument("--fine-frames", type=int, default=12)
    extract.add_argument(
        "--windowing",
        choices=("surprise-opencv-smoke", "fixed-window-baseline"),
        default="surprise-opencv-smoke",
        help=(
            "Question-independent L1 boundary policy. OpenCV surprise is an "
            "engineering smoke fallback, not the formal learned representation."
        ),
    )
    extract.add_argument("--surprise-sample-period-s", type=float, default=0.5)
    extract.add_argument("--surprise-min-window-s", type=float, default=2.0)
    extract.add_argument("--surprise-max-window-s", type=float, default=12.0)
    extract.add_argument("--surprise-calibration-history", type=int, default=8)
    extract.add_argument("--surprise-quantile", type=float, default=0.8)
    extract.add_argument("--video-limit", type=int)
    extract.add_argument("--shard-index", type=int)
    extract.add_argument("--num-shards", type=int)
    merge = sub.add_parser("merge-extract-reports")
    merge.add_argument("--selection", required=True, type=Path)
    merge.add_argument("--graph-root", required=True, type=Path)
    merge.add_argument("--shard-report-dir", required=True, type=Path)
    merge.add_argument("--report", required=True, type=Path)
    merge.add_argument("--num-shards", required=True, type=int)
    merge.add_argument("--video-limit", type=int)
    embed = sub.add_parser("embed-evaluate")
    embed.add_argument("--selection", required=True, type=Path)
    embed.add_argument("--graph-root", required=True, type=Path)
    embed.add_argument("--dataset", required=True, type=Path)
    embed.add_argument("--hidden-input", required=True, type=Path)
    embed.add_argument("--coverage-report", required=True, type=Path)
    embed.add_argument("--coverage-details", required=True, type=Path)
    embed.add_argument("--device", default="cuda")
    embed.add_argument("--memory-capacity", type=int, default=64)
    embed.add_argument("--video-limit", type=int)
    audit = sub.add_parser("audit-correlations")
    audit.add_argument("--selection", required=True, type=Path)
    audit.add_argument("--graph-root", required=True, type=Path)
    audit.add_argument("--dataset", required=True, type=Path)
    audit.add_argument("--hidden-input", required=True, type=Path)
    audit.add_argument("--report", required=True, type=Path)
    audit.add_argument("--details", required=True, type=Path)
    audit.add_argument("--correlation-policy", type=Path)
    audit.add_argument("--memory-capacity", type=int, default=64)
    audit.add_argument("--video-limit", type=int)
    audit.add_argument("--target-positive-coverage", type=float, default=0.9)
    audit.add_argument("--max-path-hops", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "select":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        selection = select_smoke_videos(
            manifest,
            dataset_root=args.dataset_root,
            per_stratum=args.per_stratum,
            observation_horizon_s=args.observation_horizon_s,
        )
        _write_json(args.output, selection)
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return 0
    if args.command == "extract":
        return _extract_command(args)
    if args.command == "merge-extract-reports":
        return _merge_extract_reports_command(args)
    if args.command == "audit-correlations":
        return _audit_correlations_command(args)
    return _embed_evaluate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
