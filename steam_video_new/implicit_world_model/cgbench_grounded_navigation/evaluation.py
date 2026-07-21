"""Create GT-only CG-Bench ablations and optional diagnostic audit packets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .builder import _write_json


def build_ablation_manifest(
    dataset: dict[str, Any], hidden: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build evaluation inputs; terminal targets remain in a separate key."""
    hidden_by_case = {row["case_id"]: row for row in hidden.get("cases") or []}
    cases: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    source_cases = list(dataset.get("cases") or [])
    for case in source_cases:
        clue_ids = list(hidden_by_case[case["case_id"]]["clue_action_ids"])
        shuffled = sorted(clue_ids, key=lambda value: _digest(case["case_id"] + ":shuffle:" + value))
        if shuffled == clue_ids and len(shuffled) > 1:
            shuffled = shuffled[1:] + shuffled[:1]
        omitted = int(_digest(case["case_id"] + ":omit")[:8], 16) % len(clue_ids)
        prediction_sources = sorted(
            clue_ids, key=lambda value: _digest(case["case_id"] + ":prediction-shuffle:" + value)
        )
        if prediction_sources == clue_ids and len(prediction_sources) > 1:
            prediction_sources = prediction_sources[-1:] + prediction_sources[:-1]
        foreign_pool: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for donor in source_cases:
            if donor["video_id"] == case["video_id"]:
                continue
            for action in donor["planner_input"]["candidate_actions"]:
                foreign_pool.append((donor, action))
        foreign_pool.sort(
            key=lambda row: _digest(case["case_id"] + ":foreign:" + row[1]["action_id"])
        )
        selected_foreign = foreign_pool[:len(clue_ids)]
        foreign_actions = [
            {
                **action,
                "source_video_id": donor["video_id"],
                "source_video_ref": donor["video_ref"],
                "evaluation_only": True,
            }
            for donor, action in selected_foreign
        ]
        arms = [
            _arm("all_clues", clue_ids, "complete_evidence_set_in_canonical_temporal_order"),
            _arm("leave_one_clue_out", clue_ids[:omitted] + clue_ids[omitted + 1:],
                 "incomplete_evidence_set"),
            _arm("shuffled_clue_order", shuffled, "complete_evidence_set_order_perturbed"),
            _arm(
                "transition_prediction_shuffle",
                clue_ids,
                "execute_gt_clues_but_permute_imagined_transition_descriptors",
                prediction_source_action_ids=prediction_sources,
            ),
            _arm(
                "cross_video_distractor",
                [action["action_id"] for action in foreign_actions],
                "video_disjoint_structural_distractor_not_semantic_negative_supervision",
                available=len(foreign_actions) == len(clue_ids),
            ),
        ]
        cases.append({
            "case_id": case["case_id"], "video_id": case["video_id"], "split": case["split"],
            "question": case["planner_input"]["question"], "choices": case["planner_input"]["choices"],
            "actions": [*case["planner_input"]["candidate_actions"], *foreign_actions],
            "arms": arms,
        })
        key = hidden_by_case[case["case_id"]]
        targets.append({"case_id": case["case_id"], "answer_text": key["answer_text"],
                        "answer_key": key["answer_key"], "omitted_clue_action_id": clue_ids[omitted]})
    public = {
        "schema_version": "steam-cgbench-gt-ablation/v0.2",
        "dataset_id": dataset.get("dataset_id"),
        "purpose": "no-training answer execution under GT evidence and causal interventions",
        "supervision_contract": (
            "CG-Bench answer and clue annotations are the only labels; cross-video distractors "
            "and shuffled predictions are evaluation interventions, never negative training labels"
        ),
        "human_review_required": False,
        "metric_contract": {
            "answer_accuracy": "exact multiple-choice correctness, computed only after hidden-key join",
            "read_efficiency": "correct answers per fixed number and duration of executed reads",
            "order_sensitivity": "answer or categorical preference changes under evidence-order shuffle",
            "report_separately": True,
        },
        "cases": cases, "training_performed": False,
    }
    key = {"schema_version": "steam-cgbench-gt-ablation-key/v0.2",
           "public_sha256": _checksum(public), "cases": targets}
    return public, key


def build_blinded_review_packet(
    dataset: dict[str, Any], hidden: dict[str, Any], *, sample_videos: int = 32,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build an optional visual-quality diagnostic; never a formal training gate."""
    hidden_by_case = {row["case_id"]: row for row in hidden.get("cases") or []}
    videos_by_split: dict[str, list[str]] = {}
    for case in dataset.get("cases") or []:
        videos_by_split.setdefault(case["split"], []).append(case["video_id"])
    selected: set[str] = set()
    all_videos = sorted({case["video_id"] for case in dataset.get("cases") or []})
    for split in ("train", "validation", "test"):
        candidates = sorted(set(videos_by_split.get(split, [])), key=lambda v: _digest("audit:" + v))
        quota = max(1, round(sample_videos * len(candidates) / max(1, len(all_videos))))
        selected.update(candidates[:quota])
    if len(selected) < sample_videos:
        selected.update(v for v in sorted(all_videos, key=lambda v: _digest("audit-fill:" + v))
                        if v not in selected and len(selected) < sample_videos)
    selected = set(sorted(selected, key=lambda v: _digest("audit-trim:" + v))[:sample_videos])
    items: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for case in dataset.get("cases") or []:
        if case["video_id"] not in selected:
            continue
        clue_ids = set(hidden_by_case[case["case_id"]]["clue_action_ids"])
        for transition in case.get("executed_transitions") or []:
            action = transition["action"]
            review_id = "cgaudit:" + _digest(transition["transition_id"])[:20]
            items.append({
                "review_id": review_id, "case_id": case["case_id"], "video_id": case["video_id"],
                "video_ref": case["video_ref"], "split": case["split"],
                "question": case["planner_input"]["question"], "choices": case["planner_input"]["choices"],
                "interval": action["interval"], "observation": transition["real_observation"],
                "review_fields": {
                    "media_matches_interval": None,
                    "descriptor_grounded": None,
                    "directly_question_relevant": None,
                    "label": None,
                    "notes": "",
                },
            })
            labels.append({"review_id": review_id, "case_id": case["case_id"],
                           "action_id": action["action_id"],
                           "gt_interval_role": "clue" if action["action_id"] in clue_ids else "control"})
    packet = {
        "schema_version": "steam-cgbench-blinded-visual-audit/v0.1",
        "dataset_id": dataset.get("dataset_id"), "annotation_status": "optional_diagnostic",
        "formal_gate": False,
        "sampling": "video-disjoint stratified deterministic sample",
        "labels": ["accept", "reject", "inconclusive"], "items": items,
    }
    key = {"schema_version": "steam-cgbench-blinded-visual-audit-key/v0.1",
           "packet_sha256": _checksum(packet), "selected_video_ids": sorted(selected), "items": labels}
    return packet, key


def _arm(
    name: str,
    action_ids: list[str],
    intervention: str,
    **extra: Any,
) -> dict[str, Any]:
    return {
        "arm": name,
        "action_ids": action_ids,
        "read_count": len(action_ids),
        "intervention": intervention,
        **extra,
    }


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _checksum(value: dict[str, Any]) -> str:
    return _digest(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hidden-input", required=True, type=Path)
    parser.add_argument("--ablation-output", required=True, type=Path)
    parser.add_argument("--ablation-key", required=True, type=Path)
    parser.add_argument("--review-output", type=Path,
                        help="Optional visual-QA diagnostic packet (not a formal gate).")
    parser.add_argument("--review-key", type=Path,
                        help="Hidden key for --review-output; both options must be supplied together.")
    parser.add_argument("--sample-videos", type=int, default=32)
    args = parser.parse_args(argv)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    ablation, ablation_key = build_ablation_manifest(dataset, hidden)
    if (args.review_output is None) != (args.review_key is None):
        parser.error("--review-output and --review-key must be supplied together")
    _write_json(args.ablation_output, ablation)
    _write_json(args.ablation_key, ablation_key)
    summary = {"ablation_cases": len(ablation["cases"]), "human_review_required": False}
    if args.review_output is not None and args.review_key is not None:
        review, review_key = build_blinded_review_packet(
            dataset, hidden, sample_videos=args.sample_videos
        )
        _write_json(args.review_output, review)
        _write_json(args.review_key, review_key)
        summary.update({"optional_review_items": len(review["items"]),
                        "optional_review_videos": len(review_key["selected_video_ids"])})
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
