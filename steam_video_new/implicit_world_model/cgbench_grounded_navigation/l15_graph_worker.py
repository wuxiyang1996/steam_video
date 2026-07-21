"""Build question-independent visual L1/L1.5 graphs and evaluate them after freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable

from memory_graph.embedding import Qwen3VLEmbeddingProvider, embed_memory_nodes
from memory_graph.pipeline import build_causal_temporal_overlay
from memory_graph.video_l1 import (
    PayloadVideoL1Provider,
    QwenVideoL1Extractor,
    VideoL1AtomicEventExtractor,
    VideoL1Config,
)

from .builder import _write_json


SELECTION_SCHEMA = "steam-cgbench-l15-smoke-selection/v0.1"
BUILD_SCHEMA = "steam-cgbench-question-independent-l15-build/v0.1"
COVERAGE_SCHEMA = "steam-cgbench-frozen-l15-coverage/v0.1"
FORBIDDEN_GRAPH_INPUTS = {
    "question", "choices", "answer", "answer_key", "answer_text",
    "clue_intervals", "hop_alignment",
}
FORBIDDEN_MODEL_NUMERIC_KEYS = {"confidence", "score", "probability", "reward", "utility"}


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
            raise ValueError(f"video {row.get('video_id')} does not have exactly one split")
        path = (root / str(row["video_ref"])).resolve()
        if root not in path.parents or not path.is_file():
            rejected.append({"video_id": str(row.get("video_id")), "reason": "video_missing"})
            continue
        duration = probe(path)
        if duration is None:
            rejected.append({"video_id": str(row.get("video_id")), "reason": "duration_unreadable"})
            continue
        entry = {
            "video_id": str(row["video_id"]),
            "video_ref": str(row["video_ref"]),
            "split": splits[0],
            "fallback_source": str(row.get("current_candidate_source") or "unavailable"),
            "duration_s": round(duration, 3),
        }
        strata.setdefault((splits[0], entry["fallback_source"]), []).append(entry)
    selected: list[dict[str, Any]] = []
    for stratum, entries in sorted(strata.items()):
        chosen = sorted(entries, key=lambda row: (row["duration_s"], _digest(row["video_id"])))[:per_stratum]
        for row in chosen:
            selected.append({**row, "selection_stratum": list(stratum)})
    return {
        "schema_version": SELECTION_SCHEMA,
        "dataset_id": manifest.get("dataset_id"),
        "selection_policy": "shortest_by_video_duration_within_split_x_fallback_source",
        "selection_uses_question_or_gt": False,
        "observation_horizon_s": observation_horizon_s,
        "videos": sorted(selected, key=lambda row: (row["split"], row["fallback_source"], row["video_id"])),
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
            "video_regime": "streaming_prefix_smoke",
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
            "observation_horizon_s": observation_horizon_s,
            "smoke_prefix_only": True,
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
            {key: value for key, value in payload.items() if key not in {"build_report"}},
            FORBIDDEN_GRAPH_INPUTS,
        ):
            raise ValueError(f"graph {video_id} contains a forbidden QA/GT key")
        graph_checksums[video_id] = _file_checksum(graph_path)
        nodes = [*(payload.get("l1_observations") or []), *(payload.get("atomic_events") or [])]
        matrix_path = graph_root / video_id / "node_embeddings.npy"
        matrix = np.load(matrix_path)
        manifest = json.loads(matrix_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        rows = sorted(manifest.get("rows") or [], key=lambda row: int(row["row_index"]))
        node_by_id = {str(row["node_id"]): row for row in nodes}
        if matrix.shape != (len(rows), 2048) or any(str(row["node_id"]) not in node_by_id for row in rows):
            raise ValueError(f"graph {video_id} embedding alignment is invalid")
        question = str(case["planner_input"]["question"])
        query = np.asarray(query_provider.encode([question], batch_size=1)[0], dtype=np.float32)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        order = np.argsort(-(matrix @ query))
        ranked_ids = [str(rows[int(index)]["node_id"]) for index in order]
        key = hidden_by_case[str(case["case_id"])]
        horizon = float(selected[video_id].get("observation_horizon_s") or selection["observation_horizon_s"])
        clue_rows = []
        for clue_index, clue in enumerate(key.get("clue_intervals") or []):
            clue_span = _span(clue)
            eligible = clue_span[1] <= horizon + 1e-6
            native_hits = [node_id for node_id, node in node_by_id.items() if _overlaps(_span(node["time_span"]), clue_span)]
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
            "recall": values["covered"] / values["eligible"] if values["eligible"] else None,
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
        "native_candidate_recall": native_covered / native_eligible if native_eligible else None,
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
    )
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for entry in selection.get("videos") or []:
        video_id = str(entry["video_id"])
        sample_dir = args.output_root / video_id
        overlay_path = sample_dir / "causal_temporal_overlay.json"
        if overlay_path.is_file():
            existing = json.loads(overlay_path.read_text(encoding="utf-8"))
            if (existing.get("metadata") or {}).get("question_independent_contract") is True:
                completed.append(_graph_summary(existing, status="resumed"))
                continue
        sample_dir.mkdir(parents=True, exist_ok=True)
        l1_path = sample_dir / "video_l1.json"
        try:
            if l1_path.is_file():
                l1_payload = json.loads(l1_path.read_text(encoding="utf-8"))
            else:
                video_path = (args.dataset_root / str(entry["video_ref"])).resolve()
                result = extractor.extract(
                    video_path=video_path,
                    video_id=video_id,
                    observation_end_s=float(selection["observation_horizon_s"]),
                )
                l1_payload = result.to_dict()
                _write_json(l1_path, l1_payload)
            graph = build_question_independent_graph(
                entry,
                dataset_root=args.dataset_root,
                output_root=args.output_root,
                video_l1_payload=l1_payload,
                observation_horizon_s=float(selection["observation_horizon_s"]),
            )
            completed.append(_graph_summary(graph, status="completed"))
        except Exception as exc:  # checkpoint every independent video
            errors.append({"video_id": video_id, "error": f"{type(exc).__name__}: {exc}"})
        _write_json(args.report, _build_run_report(selection, completed, errors, status="running"))
    report = _build_run_report(selection, completed, errors, status="completed" if not errors else "partial")
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def _embed_evaluate_command(args: argparse.Namespace) -> int:
    from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import overlay_from_dict

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    provider = Qwen3VLEmbeddingProvider(device=args.device)
    for entry in selection.get("videos") or []:
        sample_dir = args.graph_root / str(entry["video_id"])
        graph_path = sample_dir / "causal_temporal_overlay.json"
        payload = json.loads(graph_path.read_text(encoding="utf-8"))
        overlay = overlay_from_dict(payload)
        nodes = [*overlay.l1_observations, *overlay.atomic_events]
        embed_memory_nodes(nodes, provider, output_path=sample_dir / "node_embeddings.npy")
        embedded = overlay.to_dict()
        embedded["build_report"] = payload.get("build_report") or {}
        _write_json(graph_path, embedded)
    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    report, details = evaluate_frozen_graph_coverage(
        dataset, hidden, selection, graph_root=args.graph_root, query_provider=provider
    )
    _write_json(args.coverage_report, report)
    _write_json(args.coverage_details, details)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def _graph_summary(payload: dict[str, Any], *, status: str) -> dict[str, Any]:
    return {
        "video_id": payload.get("video_id"),
        "status": status,
        "l1_observation_count": len(payload.get("l1_observations") or []),
        "atomic_event_count": len(payload.get("atomic_events") or []),
        "relation_count": len(payload.get("relations") or []),
    }


def _build_run_report(
    selection: dict[str, Any], completed: list[dict[str, Any]], errors: list[dict[str, str]], *, status: str
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
        return any(str(key).casefold() in forbidden or _contains_key(child, forbidden)
                   for key, child in value.items())
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
    embed = sub.add_parser("embed-evaluate")
    embed.add_argument("--selection", required=True, type=Path)
    embed.add_argument("--graph-root", required=True, type=Path)
    embed.add_argument("--dataset", required=True, type=Path)
    embed.add_argument("--hidden-input", required=True, type=Path)
    embed.add_argument("--coverage-report", required=True, type=Path)
    embed.add_argument("--coverage-details", required=True, type=Path)
    embed.add_argument("--device", default="cuda")
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
    return _embed_evaluate_command(args)


if __name__ == "__main__":
    raise SystemExit(main())
