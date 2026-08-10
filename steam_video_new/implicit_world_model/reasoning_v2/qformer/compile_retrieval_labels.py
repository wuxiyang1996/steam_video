"""Compile QF2 positives from evaluator-only clue intervals after graph freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from .feature_store import FourSlotFeatureStore


def compile_retrieval_labels(
    *,
    feature_store: FourSlotFeatureStore,
    navigation_dataset: Mapping[str, Any],
    hidden_targets: Mapping[str, Any],
) -> dict[str, Any]:
    """Return question/node positives without copying answer fields into output."""

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
        positives = sorted(
            node_id
            for node_id, start, end in candidates
            if any(_overlaps((start, end), clue) for clue in clues)
        )
        if not positives:
            _increment(skips, "positive_node_missing")
            continue
        planner_input = case.get("planner_input") or {}
        question = str(planner_input.get("question") or "").strip()
        if not question:
            _increment(skips, "question_text_missing")
            continue
        records.append(
            {
                "case_id": case_id,
                "video_id": video_id,
                "split": str(case.get("split") or ""),
                "question": question,
                "positive_node_ids": positives,
                "candidate_node_ids": sorted(node_id for node_id, _, _ in candidates),
                "same_video_nonpositive_semantics": "ignore_unlabeled_not_negative",
            }
        )
    return {
        "schema_version": "steam-qformer-retrieval-labels/v0.1",
        "label_source": "dataset_gt_clue_interval_overlap_evaluator_only",
        "answer_fields_present": False,
        "same_video_nonpositive_semantics": "ignore_unlabeled_not_negative",
        "record_count": len(records),
        "video_count": len({row["video_id"] for row in records}),
        "skips": skips,
        "records": records,
    }


def attach_node_spans(
    navigation_dataset: dict[str, Any], overlay_root: Path
) -> dict[str, Any]:
    """Attach address-only node spans without exposing grounded node values."""

    spans: dict[str, dict[str, float]] = {}
    for path in sorted(overlay_root.expanduser().resolve().glob("**/causal_temporal_overlay.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for node in payload.get("l1_observations") or ():
            if not isinstance(node, Mapping):
                continue
            span = node.get("time_span") or {}
            spans[str(node.get("node_id") or "")] = {
                "start_s": float(span["start_s"]),
                "end_s": float(span["end_s"]),
            }
    copied = dict(navigation_dataset)
    copied["qformer_node_spans"] = spans
    return copied


def _overlaps(left: tuple[float, float], right: tuple[float, float]) -> bool:
    return left[0] < right[1] and right[0] < left[1]


def _increment(values: dict[str, int], key: str) -> None:
    values[key] = values.get(key, 0) + 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=Path)
    parser.add_argument("--navigation-dataset", required=True, type=Path)
    parser.add_argument("--hidden-targets", required=True, type=Path)
    parser.add_argument("--overlay-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    navigation = json.loads(args.navigation_dataset.read_text(encoding="utf-8"))
    navigation = attach_node_spans(navigation, args.overlay_root)
    result = compile_retrieval_labels(
        feature_store=FourSlotFeatureStore(args.feature_manifest),
        navigation_dataset=navigation,
        hidden_targets=json.loads(args.hidden_targets.read_text(encoding="utf-8")),
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    return 0 if result["record_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

