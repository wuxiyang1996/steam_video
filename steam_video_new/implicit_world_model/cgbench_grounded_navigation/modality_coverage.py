"""Report subtitle/visual availability for GT clues without relabeling evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .builder import _write_json
from .l15_candidates import _overlaps, _span_tuple, _subtitle_candidate_nodes


def build_modality_coverage(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    *,
    subtitle_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hidden_by_case = {row["case_id"]: row for row in hidden.get("cases") or []}
    subtitle_cache: dict[str, list[dict[str, Any]]] = {}
    details: list[dict[str, Any]] = []
    category_counts: dict[str, int] = {}
    case_complete: dict[str, bool] = {}
    video_with_subtitles: set[str] = set()
    for case in dataset.get("cases") or []:
        case_id = str(case["case_id"])
        video_id = str(case["video_id"])
        subtitle_path = subtitle_root / f"{video_id}.srt"
        if video_id not in subtitle_cache:
            subtitle_cache[video_id] = (
                _subtitle_candidate_nodes(video_id, subtitle_path)
                if subtitle_path.is_file() else []
            )
        cues = subtitle_cache[video_id]
        if cues:
            video_with_subtitles.add(video_id)
        transitions = case.get("executed_transitions") or []
        key = hidden_by_case[case_id]
        statuses: list[bool] = []
        for clue_index, clue in enumerate(key.get("clue_intervals") or []):
            clue_span = _span_tuple(clue)
            overlapping_cues = [cue for cue in cues if _overlaps(clue_span, _span_tuple(cue["time_span"]))]
            transition = next(
                (row for row in transitions
                 if _same_span(_span_tuple(row["action"]["interval"]), clue_span)),
                None,
            )
            observation = (transition or {}).get("real_observation") or {}
            descriptor = observation.get("descriptor") or {}
            visual_grounded = observation.get("descriptor_status") == "grounded_qwen_vl_read"
            readable_text = bool(descriptor.get("readable_text"))
            subtitle_available = bool(overlapping_cues)
            if subtitle_available and visual_grounded:
                category = "subtitle_and_visual_descriptor"
            elif subtitle_available:
                category = "subtitle_available_visual_pending"
            elif visual_grounded:
                category = "visual_descriptor_no_subtitle"
            else:
                category = "no_observed_modality_yet"
            category_counts[category] = category_counts.get(category, 0) + 1
            statuses.append(subtitle_available or visual_grounded)
            details.append(
                {
                    "case_id": case_id,
                    "video_id": video_id,
                    "clue_index": clue_index,
                    "clue_interval": clue,
                    "subtitle_status": "available" if subtitle_available else "unavailable",
                    "overlapping_subtitle_cue_count": len(overlapping_cues),
                    "visual_descriptor_status": (
                        "grounded" if visual_grounded else "pending_or_failed"
                    ),
                    "in_frame_readable_text_observed": readable_text,
                    "direct_audio_observed": False,
                    "coverage_category": category,
                    "semantic_sufficiency_claimed": False,
                }
            )
        case_complete[case_id] = bool(statuses) and all(statuses)
    summary = {
        "schema_version": "steam-cgbench-modality-coverage-summary/v0.1",
        "dataset_id": dataset.get("dataset_id"),
        "case_count": len(dataset.get("cases") or []),
        "clue_count": len(details),
        "video_count": len({case["video_id"] for case in dataset.get("cases") or []}),
        "video_with_subtitle_count": len(video_with_subtitles),
        "coverage_category_counts": dict(sorted(category_counts.items())),
        "cases_with_observed_modality_for_every_clue": sum(case_complete.values()),
        "cases_still_missing_a_modality_for_at_least_one_clue": sum(not value for value in case_complete.values()),
        "audio_contract": (
            "SRT is reported as time-aligned text; direct audio is unobserved and no audio "
            "semantics are inferred"
        ),
        "ground_truth_contract": (
            "coverage reports availability only; it never changes CG-Bench clue labels"
        ),
        "human_review_required": False,
        "training_performed": False,
    }
    hidden_details = {
        "schema_version": "steam-cgbench-modality-coverage-details/v0.1",
        "dataset_id": dataset.get("dataset_id"),
        "warning": "Contains GT clue intervals; never expose to the planner.",
        "clues": details,
    }
    return summary, hidden_details


def _same_span(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return abs(left[0] - right[0]) < 0.002 and abs(left[1] - right[1]) < 0.002


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hidden-input", required=True, type=Path)
    parser.add_argument("--subtitle-root", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    args = parser.parse_args(argv)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    summary, details = build_modality_coverage(
        dataset, hidden, subtitle_root=args.subtitle_root.expanduser().resolve()
    )
    _write_json(args.summary, summary)
    _write_json(args.details, details)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
