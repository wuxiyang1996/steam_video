"""Freeze a train-split video cohort for future runtime transition collection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


COLLECTION_SCHEMA = "steam-iwm-runtime-train-collection/v0.1"
GRAPH_SELECTION_SCHEMA = "steam-iwm-runtime-train-graph-selection/v0.1"
GRAPH_PROTOCOL_SCHEMA = "steam-iwm-runtime-train-graph-protocol/v0.1"
FORBIDDEN_GRAPH_KEYS = {
    "question",
    "choices",
    "answer",
    "answer_key",
    "answer_text",
    "clue_intervals",
    "hop_alignment",
}


def build_train_collection_manifest(
    dataset: dict[str, Any],
    *,
    collection_id: str,
    case_count: int = 40,
    exclude_video_ids: set[str] | None = None,
) -> dict[str, Any]:
    if case_count < 30 or case_count > 50:
        raise ValueError("runtime train collection must freeze 30 to 50 cases")
    excluded = exclude_video_ids or set()
    candidates = [
        row
        for row in dataset.get("cases") or ()
        if row.get("split") == "train"
        and row.get("case_id")
        and row.get("video_id")
        and row.get("video_ref")
        and str(row.get("video_id")) not in excluded
    ]
    candidates.sort(
        key=lambda row: hashlib.sha256(
            f"{collection_id}\0{row['case_id']}".encode()
        ).hexdigest()
    )
    selected: list[dict[str, Any]] = []
    videos: set[str] = set()
    for row in candidates:
        video_id = str(row["video_id"])
        if video_id in videos:
            continue
        videos.add(video_id)
        selected.append(row)
        if len(selected) == case_count:
            break
    if len(selected) != case_count:
        raise ValueError("not enough unique train videos for runtime collection")
    cases = [
        {
            "case_id": str(row["case_id"]),
            "video_id": str(row["video_id"]),
            "video_ref": str(row["video_ref"]),
            "video_duration_s": float(row.get("video_duration_s") or 0.0),
            "split": "train",
            "source_qid": str((row.get("source") or {}).get("qid") or ""),
            "public_executed_transition_count": len(
                row.get("executed_transitions") or ()
            ),
            "question_independent_l1_l15_status": "pending",
            "clue_retention_gate_status": "pending",
            "multi_trajectory_execution_status": "pending",
            "independent_transition_review_status": "pending",
        }
        for row in selected
    ]
    return {
        "schema_version": COLLECTION_SCHEMA,
        "collection_id": collection_id,
        "source_dataset_id": str(dataset.get("dataset_id") or "unknown"),
        "selection_policy": {
            "split": "train_only",
            "one_case_per_video": True,
            "stable_hash_sampling": True,
            "question_text_used_for_selection": False,
            "hidden_answer_used_for_selection": False,
            "graph_must_be_question_independent": True,
            "excluded_prior_video_count": len(excluded),
        },
        "target_failure_slices": [
            "weak_related_but_insufficient",
            "identity_unresolved_or_conflicting",
            "state_delta_incomplete",
            "hypothesis_discriminating_support",
            "counterevidence",
            "empty_or_inconclusive_control",
            "delayed_second_hop_completion",
        ],
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "video_count": len(videos),
            "total_video_duration_s": sum(
                float(row["video_duration_s"]) for row in cases
            ),
            "question_independent_graph_ready_count": 0,
            "executed_multi_trajectory_case_count": 0,
            "independently_reviewed_case_count": 0,
        },
        "training_ready": False,
        "training_performed": False,
    }


def build_question_independent_graph_inputs(
    collection: dict[str, Any],
    graph_manifest: dict[str, Any],
    *,
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Materialize worker selection and evaluator protocol from a frozen cohort.

    The worker-facing selection contains only video metadata. Case IDs stay in a
    separate evaluator protocol that is never passed to L1/L1.5 extraction.
    """

    manifest_by_video = {
        str(row["video_id"]): row for row in graph_manifest.get("videos") or ()
    }
    root = dataset_root.expanduser().resolve()
    videos: list[dict[str, Any]] = []
    protocol_cases: list[dict[str, str]] = []
    seen_videos: set[str] = set()
    for case in collection.get("cases") or ():
        case_id = str(case.get("case_id") or "")
        video_id = str(case.get("video_id") or "")
        if not case_id or not video_id or str(case.get("split")) != "train":
            raise ValueError("frozen collection contains an invalid train case")
        if video_id in seen_videos:
            raise ValueError("frozen collection must contain one case per video")
        seen_videos.add(video_id)
        source = manifest_by_video.get(video_id)
        if source is None:
            raise ValueError(f"graph manifest lacks frozen video {video_id}")
        leaked = _find_forbidden_keys(source)
        if leaked:
            raise ValueError(
                f"graph manifest row for {video_id} contains forbidden keys: "
                + ", ".join(sorted(leaked))
            )
        video_ref = str(source.get("video_ref") or case.get("video_ref") or "")
        path = (root / video_ref).resolve()
        if root not in path.parents or not path.is_file():
            raise FileNotFoundError(f"frozen train video is unavailable: {path}")
        duration = float(case.get("video_duration_s") or 0.0)
        videos.append(
            {
                "video_id": video_id,
                "video_ref": video_ref,
                "split": "train",
                "fallback_source": str(
                    source.get("current_candidate_source") or "unavailable"
                ),
                "duration_s": duration,
                "observation_horizon_s": duration,
            }
        )
        protocol_cases.append(
            {"case_id": case_id, "video_id": video_id, "split": "train"}
        )
    if not videos:
        raise ValueError("frozen collection contains no videos")
    selection = {
        "schema_version": GRAPH_SELECTION_SCHEMA,
        "dataset_id": collection.get("source_dataset_id"),
        "collection_id": collection.get("collection_id"),
        "selection_policy": "exact_frozen_train_collection_order",
        "selection_uses_question_or_gt": False,
        "full_video_required": True,
        "videos": videos,
        "forbidden_worker_inputs": sorted(FORBIDDEN_GRAPH_KEYS),
        "training_performed": False,
    }
    protocol = {
        "schema_version": GRAPH_PROTOCOL_SCHEMA,
        "dataset_id": collection.get("source_dataset_id"),
        "collection_id": collection.get("collection_id"),
        "case_count": len(protocol_cases),
        "video_count": len(videos),
        "splits": ["train"],
        "minimum_locked_cases": len(protocol_cases),
        "maximum_locked_cases": len(protocol_cases),
        "cases": protocol_cases,
        "case_selection_uses_question_text": False,
        "case_selection_uses_clue_or_answer_gt": False,
        "graph_builder_receives_case_protocol": False,
        "training_performed": False,
    }
    return selection, protocol


def _find_forbidden_keys(value: Any, prefix: str = "") -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if str(key).casefold() in FORBIDDEN_GRAPH_KEYS:
                found.add(path)
            found.update(_find_forbidden_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.update(_find_forbidden_keys(child, f"{prefix}[{index}]"))
    return found


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--collection-id", required=True)
    parser.add_argument("--case-count", type=int, default=40)
    parser.add_argument(
        "--exclude-selection",
        type=Path,
        help="Optional prior worker selection whose video IDs must be excluded.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--graph-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--protocol-output", type=Path)
    args = parser.parse_args(argv)
    excluded_video_ids: set[str] = set()
    if args.exclude_selection is not None:
        prior = _read(args.exclude_selection)
        excluded_video_ids = {
            str(row["video_id"])
            for row in prior.get("videos") or ()
            if row.get("video_id")
        }
    manifest = build_train_collection_manifest(
        _read(args.dataset),
        collection_id=args.collection_id,
        case_count=args.case_count,
        exclude_video_ids=excluded_video_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    graph_options = (
        args.graph_manifest,
        args.dataset_root,
        args.selection_output,
        args.protocol_output,
    )
    if any(value is not None for value in graph_options):
        if not all(value is not None for value in graph_options):
            raise ValueError(
                "graph input generation requires --graph-manifest, --dataset-root, "
                "--selection-output, and --protocol-output together"
            )
        selection, protocol = build_question_independent_graph_inputs(
            manifest,
            _read(args.graph_manifest),
            dataset_root=args.dataset_root,
        )
        for path, value in (
            (args.selection_output, selection),
            (args.protocol_output, protocol),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
                + "\n",
                encoding="utf-8",
            )
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
