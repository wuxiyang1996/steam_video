"""Build leakage-safe navigation supervision from CG-Bench clue intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable


SCHEMA_VERSION = "steam-cgbench-grounded-navigation/v0.2"
HIDDEN_SCHEMA_VERSION = "steam-cgbench-terminal-targets/v0.2"
PREFERENCE_LABELS = {"prefer_left", "prefer_right"}
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def build_cgbench_navigation_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    video_root: Path,
    dataset_id: str,
    min_clues: int = 2,
    max_cases: int | None = None,
    split_salt: str = "cgbench-grounded-navigation-v1",
    duration_probe: Callable[[Path], float | None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Create GT-clue transitions and ordinal coverage comparisons.

    Correct answers stay in a separate hidden key. A clue interval supervises
    evidence coverage, not identity, state transition, causality, or final
    answerability. Observation text and embeddings remain pending real VLM reads.
    """

    if min_clues < 1:
        raise ValueError("min_clues must be positive")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases must be positive")
    root = video_root.expanduser().resolve()
    probe = duration_probe or _probe_video_duration
    source_rows = sorted(
        (json.loads(json.dumps(row)) for row in rows),
        key=lambda row: _digest(
            f"{dataset_id}\0{row.get('video_uid')}\0{row.get('qid')}"
        ),
    )
    duration_cache: dict[str, float | None] = {}
    cases: list[dict[str, Any]] = []
    hidden_cases: list[dict[str, Any]] = []
    quarantined: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for row in source_rows:
        video_id = str(row.get("video_uid") or "").strip()
        qid = str(row.get("qid") or "").strip()
        if not video_id or not qid:
            skip("missing_video_or_qid")
            continue
        intervals = _normalize_intervals(row.get("clue_intervals") or [])
        if len(intervals) < min_clues:
            skip("insufficient_clue_intervals")
            continue
        if _has_overlap(intervals):
            skip("overlapping_clue_intervals")
            continue
        video_path = _resolve_video(root, video_id)
        if video_path is None:
            quarantined.append({"video_id": video_id, "reason": "video_missing"})
            continue
        if video_id not in duration_cache:
            duration_cache[video_id] = probe(video_path)
        duration_s = duration_cache[video_id]
        if duration_s is None:
            quarantined.append({"video_id": video_id, "reason": "duration_unreadable"})
            continue
        if any(end > duration_s + 0.5 for start, end in intervals):
            quarantined.append(
                {
                    "video_id": video_id,
                    "reason": "clue_window_exceeds_video_duration",
                }
            )
            continue
        case_id = "cgcase:" + _digest(f"{video_id}\0{qid}")[:20]
        split = _video_split(video_id, split_salt)
        clue_actions = [
            _action(case_id, "candidate", index, interval)
            for index, interval in enumerate(intervals)
        ]
        candidates = sorted(
            clue_actions,
            key=lambda action: _digest(f"{case_id}\0candidate-order\0{action['action_id']}"),
        )
        transitions = []
        acquired: list[str] = []
        for index, clue in enumerate(clue_actions):
            coverage_after = "complete" if index + 1 == len(clue_actions) else "partial"
            checkpoint = {
                "acquired_evidence_ids": list(acquired),
                "required_clue_coverage": "none" if not acquired else "partial",
                "answerability": "unknown",
            }
            transitions.append(
                _transition(
                    case_id, checkpoint, clue,
                    evidence_progress="advances_required_clue_coverage",
                    coverage_after=coverage_after,
                )
            )
            acquired.append(clue["action_id"])
        complete = {
            "trajectory_id": f"{case_id}:trajectory:{_digest(case_id + ':complete')[:12]}",
            "action_ids": [action["action_id"] for action in clue_actions],
        }
        omitted = int(_digest(case_id + ":coverage-omit")[:8], 16) % len(clue_actions)
        incomplete = {
            "trajectory_id": f"{case_id}:trajectory:{_digest(case_id + ':incomplete')[:12]}",
            "action_ids": [
                action["action_id"] for index, action in enumerate(clue_actions)
                if index != omitted
            ],
        }
        swap = int(_digest(f"{case_id}\0pair-order")[:2], 16) % 2 == 1
        left, right = (incomplete, complete) if swap else (complete, incomplete)
        preference = {
            "comparison_id": f"{case_id}:comparison:full-path",
            "left": left,
            "right": right,
            "label": "prefer_right" if swap else "prefer_left",
            "supervision_basis": (
                "manual_complete_clue_coverage_over_leave_one_clue_out;"
                "coverage_preference_not_numeric_reward_or_answer_correctness"
            ),
        }
        cases.append(
            {
                "case_id": case_id,
                "source": {"dataset": "CG-Bench", "qid": qid},
                "video_id": video_id,
                "video_ref": f"cg_videos/{video_path.name}",
                "video_duration_s": duration_s,
                "split": split,
                "planner_input": {
                    "question": str(row.get("question") or ""),
                    "choices": [str(value) for value in row.get("choices") or []],
                    "candidate_actions": candidates,
                },
                "executed_transitions": transitions,
                "trajectory_preference": preference,
            }
        )
        hidden_cases.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "qid": qid,
                "answer_text": str(row.get("answer") or ""),
                "answer_key": str(row.get("right_answer") or ""),
                "clue_intervals": [
                    {"start_s": start, "end_s": end} for start, end in intervals
                ],
                "clue_action_ids": [action["action_id"] for action in clue_actions],
            }
        )
        if max_cases is not None and len(cases) >= max_cases:
            break

    dataset = {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "annotation_status": "cgbench_human_ground_truth_anchored",
        "output_contract": "categorical_transitions_and_ordinal_preferences_only",
        "split_policy": "video_disjoint_deterministic_hash",
        "source_contract": {
            "clue_intervals": "manual_question_relevant_temporal_grounding",
            "outside_clue_intervals": (
                "unlabeled; excluded from supervision and never treated as negative"
            ),
            "does_not_imply": [
                "identity", "state_transition", "causality", "answer_sufficiency"
            ],
        },
        "cases": cases,
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
    }
    checksum = _checksum(dataset)
    hidden = {
        "schema_version": HIDDEN_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "dataset_sha256": checksum,
        "warning": "Terminal answers and clue identity; never expose to planner inputs or reviewers.",
        "cases": hidden_cases,
    }
    errors = validate_cgbench_navigation_dataset(dataset, hidden)
    if errors:
        raise ValueError("invalid CG-Bench navigation dataset: " + "; ".join(errors[:8]))
    split_counts: dict[str, int] = {}
    for case in cases:
        split_counts[case["split"]] = split_counts.get(case["split"], 0) + 1
    report = {
        "schema_version": "steam-cgbench-grounded-navigation-report/v0.2",
        "dataset_id": dataset_id,
        "source_row_count": len(source_rows),
        "case_count": len(cases),
        "video_count": len({case["video_id"] for case in cases}),
        "split_case_counts": dict(sorted(split_counts.items())),
        "transition_count": sum(len(case["executed_transitions"]) for case in cases),
        "trajectory_preference_count": len(cases),
        "multi_clue_case_count": sum(
            len(hidden_case["clue_intervals"]) > 1 for hidden_case in hidden_cases
        ),
        "skipped": dict(sorted(skipped.items())),
        "quarantined_video_count": len({row["video_id"] for row in quarantined}),
        "quarantined": _deduplicate_rows(quarantined),
        "answer_leakage_detected": False,
        "numeric_reward_present": False,
        "observation_descriptors": "pending_qwen_vl_grounded_reads",
        "embedding_model": EMBEDDING_MODEL,
        "formal_eligible": False,
        "training_ready": False,
        "training_performed": False,
        "next_gate": "ground real observations with Qwen-VL and run schema/coverage checks",
        "human_review_gate": "not_required; optional diagnostics only",
        "negative_supervision": "none",
    }
    return dataset, hidden, report


def validate_cgbench_navigation_dataset(
    dataset: dict[str, Any], hidden: dict[str, Any] | None = None
) -> list[str]:
    errors: list[str] = []
    if dataset.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    if dataset.get("training_ready") is not False or dataset.get("training_performed") is not False:
        errors.append("unverified dataset must not be training-ready or trained")
    case_ids: set[str] = set()
    split_by_video: dict[str, str] = {}
    for index, case in enumerate(dataset.get("cases") or []):
        case_id = str(case.get("case_id") or "")
        if not case_id or case_id in case_ids:
            errors.append(f"cases.{index} has duplicate or empty case_id")
        case_ids.add(case_id)
        video_id = str(case.get("video_id") or "")
        split = str(case.get("split") or "")
        if split not in {"train", "validation", "test"}:
            errors.append(f"cases.{index} has invalid split")
        if video_id in split_by_video and split_by_video[video_id] != split:
            errors.append(f"video split leakage for {video_id}")
        split_by_video[video_id] = split
        planner = case.get("planner_input") or {}
        if _contains_key(planner, {"answer", "answer_key", "right_answer", "clue_intervals"}):
            errors.append(f"cases.{index} leaks terminal or clue labels into planner_input")
        actions = {
            str(action.get("action_id")): action
            for action in planner.get("candidate_actions") or []
        }
        for action_id, action in actions.items():
            interval = action.get("interval") or {}
            start = interval.get("start_s")
            end = interval.get("end_s")
            if not action_id or not isinstance(start, (int, float)) or not isinstance(end, (int, float)) or start < 0 or end <= start:
                errors.append(f"cases.{index} has invalid candidate interval")
            elif end > float(case.get("video_duration_s") or 0) + 0.5:
                errors.append(f"cases.{index} candidate interval exceeds video duration")
        comparison = case.get("trajectory_preference") or {}
        if comparison.get("label") not in PREFERENCE_LABELS:
            errors.append(f"cases.{index} has invalid ordinal preference")
        positive: list[dict[str, Any]] = []
        for transition in case.get("executed_transitions") or []:
            if str((transition.get("action") or {}).get("action_id")) not in actions:
                errors.append(f"cases.{index} transition references unknown action")
            progress = str(
                ((transition.get("target") or {}).get("belief_delta") or {}).get(
                    "evidence_progress"
                )
            )
            if progress == "advances_required_clue_coverage":
                positive.append(transition)
            else:
                errors.append(f"cases.{index} has unknown categorical evidence progress")
            descriptor = (transition.get("target") or {}).get("observation_descriptor") or {}
            embedding = descriptor.get("embedding_ref") or {}
            if embedding.get("model") != EMBEDDING_MODEL:
                errors.append(f"cases.{index} has an invalid embedding model")
            embedding_status = embedding.get("status")
            grounding_status = descriptor.get("grounding_status")
            real_observation = transition.get("real_observation") or {}
            if embedding_status == "pending":
                if grounding_status not in {"pending", "grounded"}:
                    errors.append(f"cases.{index} has an invalid pending descriptor")
            elif embedding_status == "available":
                if grounding_status != "grounded":
                    errors.append(f"cases.{index} embeds an ungrounded descriptor")
                if not isinstance(embedding.get("row_index"), int):
                    errors.append(f"cases.{index} has no embedding row index")
                if not embedding.get("storage_uri") or not embedding.get("checksum"):
                    errors.append(f"cases.{index} has an incomplete embedding reference")
            else:
                errors.append(f"cases.{index} has an invalid embedding status")
            if grounding_status == "grounded":
                if real_observation.get("descriptor_status") != "grounded_qwen_vl_read":
                    errors.append(f"cases.{index} has inconsistent grounding status")
                if not isinstance(real_observation.get("descriptor"), dict):
                    errors.append(f"cases.{index} has no grounded descriptor")
        if not positive or len(positive) != len(actions):
            errors.append(f"cases.{index} must contain only GT-clue transitions")
        positive_ids = {str(row["action"]["action_id"]) for row in positive}
        left_ids = set((comparison.get("left") or {}).get("action_ids") or [])
        right_ids = set((comparison.get("right") or {}).get("action_ids") or [])
        preferred_side = "left" if comparison.get("label") == "prefer_left" else "right"
        preferred_ids = left_ids if preferred_side == "left" else right_ids
        if preferred_ids != positive_ids:
            errors.append(f"cases.{index} preference does not select the clue-grounded path")
        nonpreferred_ids = right_ids if preferred_side == "left" else left_ids
        if not nonpreferred_ids < positive_ids or len(positive_ids - nonpreferred_ids) != 1:
            errors.append(f"cases.{index} comparison must be complete versus leave-one-out")
    if _contains_key(dataset, {"answer", "answer_key", "right_answer"}):
        errors.append("terminal answer leaked into public dataset")
    if _contains_key(dataset, {"reward", "score", "probability", "utility"}):
        errors.append("numeric reward-like field is forbidden")
    if hidden is not None:
        if hidden.get("schema_version") != HIDDEN_SCHEMA_VERSION:
            errors.append("unsupported hidden schema_version")
        if hidden.get("dataset_id") != dataset.get("dataset_id"):
            errors.append("hidden dataset_id mismatch")
        if hidden.get("dataset_sha256") != _checksum(dataset):
            errors.append("hidden dataset checksum mismatch")
        hidden_ids = {str(row.get("case_id")) for row in hidden.get("cases") or []}
        if hidden_ids != case_ids:
            errors.append("hidden cases do not exactly match public cases")
    return errors


def _action(
    case_id: str, family: str, index: int, interval: tuple[float, float]
) -> dict[str, Any]:
    start, end = interval
    return {
        "action_id": f"{case_id}:read:{_digest(f'{case_id}:{family}:{index}')[:12]}",
        "action_type": "read_video_interval",
        "interval": {"start_s": start, "end_s": end},
        "read_budget_s": round(end - start, 3),
        "requested_modalities": ["vision", "audio", "subtitle"],
    }


def _transition(
    case_id: str,
    checkpoint: dict[str, Any],
    action: dict[str, Any],
    *,
    evidence_progress: str,
    coverage_after: str,
) -> dict[str, Any]:
    action_id = str(action["action_id"])
    return {
        "transition_id": f"{case_id}:transition:{_digest(action_id)[:12]}",
        "checkpoint": checkpoint,
        "action": action,
        "real_observation": {
            "video_interval": action["interval"],
            "descriptor_status": "pending_qwen_vl_grounded_read",
            "descriptor": None,
        },
        "target": {
            "observation_descriptor": {
                "role": "question_relevant_evidence_read",
                "grounding_status": "pending",
                "embedding_ref": {
                    "model": EMBEDDING_MODEL,
                    "status": "pending",
                    "dimension": 2048,
                    "dtype": "float32",
                    "normalized": True,
                    "storage_uri": None,
                    "row_index": None,
                    "checksum": None,
                },
            },
            "belief_delta": {
                "evidence_progress": evidence_progress,
                "required_clue_coverage_after": coverage_after,
                "answerability_after": "unknown",
                "predicted_only": False,
            },
        },
    }


def _normalize_intervals(values: Iterable[Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for value in values:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            continue
        try:
            start, end = float(value[0]), float(value[1])
        except (TypeError, ValueError):
            continue
        if start >= 0 and end > start:
            intervals.append((start, end))
    return sorted(set(intervals))


def _has_overlap(intervals: list[tuple[float, float]]) -> bool:
    return any(right[0] < left[1] for left, right in zip(intervals, intervals[1:]))


def _interval_tuple(value: dict[str, Any]) -> tuple[float, float]:
    return float(value["start_s"]), float(value["end_s"])


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _matched_negative_intervals(
    clues: list[tuple[float, float]], *, duration_s: float, salt: str
) -> list[tuple[float, float]] | None:
    protected = list(clues)
    selected: list[tuple[float, float]] = []
    for index, (start, end) in enumerate(clues):
        width = end - start
        gaps = _free_gaps([*protected, *selected], duration_s)
        viable = [(a, b) for a, b in gaps if b - a >= width]
        if not viable:
            return None
        digest = int(_digest(f"{salt}\0negative\0{index}"), 16)
        gap_start, gap_end = viable[digest % len(viable)]
        room_ms = max(0, int(round((gap_end - gap_start - width) * 1000)))
        offset_ms = digest % (room_ms + 1) if room_ms else 0
        negative = (round(gap_start + offset_ms / 1000, 3), round(gap_start + offset_ms / 1000 + width, 3))
        selected.append(negative)
    return selected


def _free_gaps(
    intervals: list[tuple[float, float]], duration_s: float
) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        start = max(0.0, start - 0.5)
        end = min(duration_s, end + 0.5)
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in merged:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration_s:
        gaps.append((cursor, duration_s))
    return gaps


def _resolve_video(root: Path, video_id: str) -> Path | None:
    for suffix in (".mp4", ".mkv", ".webm", ".avi"):
        path = (root / f"{video_id}{suffix}").resolve()
        if root in path.parents and path.is_file():
            return path
    return None


def _probe_video_duration(path: Path) -> float | None:
    try:
        import cv2  # type: ignore[import-not-found]

        capture = cv2.VideoCapture(str(path))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frames = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
    except (ImportError, ValueError):
        return None
    return frames / fps if fps > 0 and frames > 0 else None


def _video_split(video_id: str, salt: str) -> str:
    bucket = int(_digest(f"{salt}\0{video_id}")[:8], 16) % 10
    return "train" if bucket < 8 else ("validation" if bucket == 8 else "test")


def _contains_key(value: Any, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(str(key).casefold() in forbidden or _contains_key(child, forbidden) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


def _deduplicate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique = {json.dumps(row, sort_keys=True): row for row in rows}
    return [unique[key] for key in sorted(unique)]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _checksum(value: dict[str, Any]) -> str:
    return _digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--video-root", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--min-clues", type=int, default=2)
    parser.add_argument("--max-cases", type=int)
    args = parser.parse_args(argv)
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("CG-Bench input must be a JSON list")
    dataset, hidden, report = build_cgbench_navigation_dataset(
        rows,
        video_root=args.video_root,
        dataset_id=args.dataset_id,
        min_clues=args.min_clues,
        max_cases=args.max_cases,
    )
    _write_json(args.output, dataset)
    _write_json(args.hidden_output, hidden)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0
