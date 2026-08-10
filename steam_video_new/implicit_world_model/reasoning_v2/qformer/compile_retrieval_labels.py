"""Compile QF2 positives from evaluator-only clue intervals after graph freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .feature_store import FourSlotFeatureStore


def compile_retrieval_labels(
    *,
    feature_store: FourSlotFeatureStore,
    navigation_dataset: Mapping[str, Any],
    hidden_targets: Mapping[str, Any],
    same_video_negative_margin_s: float = 2.0,
    require_all_clue_groups: bool = False,
) -> dict[str, Any]:
    """Return question/node positives without copying answer fields into output."""

    if same_video_negative_margin_s < 0.0:
        raise ValueError("same-video negative margin must be non-negative")
    questions = {
        str(case.get("case_id")): case
        for case in navigation_dataset.get("cases") or ()
        if isinstance(case, Mapping)
    }
    nodes_by_video: dict[str, list[tuple[str, float, float]]] = {}
    # Feature rows intentionally do not contain grounded values. Time spans are
    # recovered from the source-contract companion supplied in target cases via
    # candidate actions when available; formal compilation should pass explicit
    # node spans through node_spans metadata in the feature manifest extension.
    node_spans = navigation_dataset.get("qformer_node_spans") or {}
    for row in feature_store.manifest.rows:
        span = node_spans.get(row.node_id)
        if not isinstance(span, Mapping):
            continue
        nodes_by_video.setdefault(row.video_id, []).append(
            (row.node_id, float(span["start_s"]), float(span["end_s"]))
        )

    records: list[dict[str, Any]] = []
    skips: dict[str, int] = {}
    for target in hidden_targets.get("cases") or ():
        if not isinstance(target, Mapping):
            continue
        case_id = str(target.get("case_id") or "")
        case = questions.get(case_id)
        if case is None:
            _increment(skips, "question_missing")
            continue
        video_id = str(target.get("video_id") or "")
        candidates = nodes_by_video.get(video_id) or []
        if not candidates:
            _increment(skips, "feature_nodes_or_spans_missing")
            continue
        clues = [
            (float(row["start_s"]), float(row["end_s"]))
            for row in target.get("clue_intervals") or ()
            if isinstance(row, Mapping)
        ]
        positive_groups = [
            sorted(
                node_id
                for node_id, start, end in candidates
                if _overlaps((start, end), clue)
            )
            for clue in clues
        ]
        covered_groups = [group for group in positive_groups if group]
        positives = sorted({node_id for group in covered_groups for node_id in group})
        if not positives:
            _increment(skips, "positive_node_missing")
            continue
        uncovered_group_count = len(positive_groups) - len(covered_groups)
        if require_all_clue_groups and uncovered_group_count:
            _increment(skips, "uncovered_clue_groups")
            continue
        planner_input = case.get("planner_input") or {}
        question = str(planner_input.get("question") or "").strip()
        if not question:
            _increment(skips, "question_text_missing")
            continue
        same_video_negatives = [
            (min(_interval_gap((start, end), clue) for clue in clues), node_id)
            for node_id, start, end in candidates
            if node_id not in positives
            and clues
            and all(
                _interval_gap((start, end), clue) >= same_video_negative_margin_s
                for clue in clues
            )
        ]
        same_video_negatives.sort(key=lambda row: (row[0], row[1]))
        trusted_same_video_ids = [node_id for _, node_id in same_video_negatives]
        records.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "split": str(case.get("split") or ""),
                "question": question,
                "positive_node_ids": positives,
                "positive_node_groups": [
                    {"clue_index": index, "node_ids": group}
                    for index, group in enumerate(positive_groups)
                    if group
                ],
                "clue_group_count": len(positive_groups),
                "uncovered_clue_group_count": uncovered_group_count,
                "candidate_node_ids": sorted(node_id for node_id, _, _ in candidates),
                "trusted_same_video_negative_node_ids": trusted_same_video_ids,
                "trusted_same_video_negative_margin_s": same_video_negative_margin_s,
                "same_video_nonpositive_semantics": (
                    "trusted_only_if_separated_from_every_clue_by_margin;"
                    "remaining_unlabeled_nodes_are_ignore"
                ),
            }
        )
    return {
        "schema_version": "steam-qformer-retrieval-labels/v0.3",
        "label_source": "dataset_gt_clue_interval_overlap_evaluator_only",
        "answer_fields_present": False,
        "require_all_clue_groups": require_all_clue_groups,
        "trusted_same_video_negative_margin_s": same_video_negative_margin_s,
        "same_video_nonpositive_semantics": (
            "trusted_only_if_separated_from_every_clue_by_margin;"
            "remaining_unlabeled_nodes_are_ignore"
        ),
        "record_count": len(records),
        "video_count": len({row["video_id"] for row in records}),
        "skips": skips,
        "records": records,
    }


def attach_node_spans(
    navigation_dataset: dict[str, Any], overlay_roots: Path | Sequence[Path]
) -> dict[str, Any]:
    """Attach address-only node spans without exposing grounded node values."""

    roots = [overlay_roots] if isinstance(overlay_roots, Path) else list(overlay_roots)
    spans: dict[str, dict[str, float]] = {}
    for root in roots:
        for path in sorted(root.expanduser().resolve().glob("**/causal_temporal_overlay.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for node in payload.get("l1_observations") or ():
                if not isinstance(node, Mapping):
                    continue
                node_id = str(node.get("node_id") or "")
                span = node.get("time_span") or {}
                value = {
                    "start_s": float(span["start_s"]),
                    "end_s": float(span["end_s"]),
                }
                if node_id in spans and spans[node_id] != value:
                    raise ValueError(f"conflicting time spans for node {node_id}")
                spans[node_id] = value
    copied = dict(navigation_dataset)
    copied["qformer_node_spans"] = spans
    return copied


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _interval_gap(left: tuple[float, float], right: tuple[float, float]) -> float:
    """Return zero for overlap/touching, otherwise the temporal separation."""

    if _overlaps(left, right):
        return 0.0
    return max(right[0] - left[1], left[0] - right[1], 0.0)


def _increment(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--navigation-dataset", required=True, type=Path)
    parser.add_argument("--hidden-targets", required=True, type=Path)
    parser.add_argument("--overlay-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--same-video-negative-margin-s", type=float, default=2.0)
    parser.add_argument("--require-all-clue-groups", action="store_true")
    args = parser.parse_args(argv)
    navigation = json.loads(args.navigation_dataset.read_text(encoding="utf-8"))
    navigation = attach_node_spans(navigation, args.overlay_root)
    result = compile_retrieval_labels(
        feature_store=FourSlotFeatureStore(args.feature_manifest),
        navigation_dataset=navigation,
        hidden_targets=json.loads(args.hidden_targets.read_text(encoding="utf-8")),
        same_video_negative_margin_s=args.same_video_negative_margin_s,
        require_all_clue_groups=args.require_all_clue_groups,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["record_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
