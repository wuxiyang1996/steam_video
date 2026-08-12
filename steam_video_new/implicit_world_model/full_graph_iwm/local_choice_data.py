"""Export grounded entry-anchor transition and preference supervision."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable


PUBLIC_SCHEMA = "steam-full-iwm-grounded-local-choice/v0.1"
HIDDEN_SCHEMA = "steam-full-iwm-grounded-local-choice-hidden/v0.1"


def build_grounded_local_choice_packet(
    *,
    runs: Iterable[dict[str, Any]],
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    graph_root: Path,
    transition_cache: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build blinded inputs and evaluator-only categorical labels.

    Absence of GT clue overlap never becomes a standalone negative label.
    Hidden clue intervals are consulted only after the model-backed run.
    """

    public_cases = {str(row["case_id"]): row for row in dataset.get("cases") or []}
    hidden_cases = {str(row["case_id"]): row for row in hidden_key.get("cases") or []}
    cached_predictions = _cache_predictions(transition_cache or {})
    graph_cache: dict[str, dict[str, dict[str, Any]]] = {}
    public_choices: list[dict[str, Any]] = []
    hidden_choices: list[dict[str, Any]] = []
    public_corrections: list[dict[str, Any]] = []
    hidden_corrections: list[dict[str, Any]] = []

    for run in runs:
        if run.get("arm") != "world_model_guided":
            continue
        case_id = str(run.get("case_id") or "")
        if case_id not in public_cases or case_id not in hidden_cases:
            raise ValueError(f"missing dataset case for local choice: {case_id}")
        public_case = public_cases[case_id]
        hidden_case = hidden_cases[case_id]
        video_id = str(hidden_case["video_id"])
        if video_id not in graph_cache:
            graph = _read_json(graph_root / video_id / "l1_l15_navigation_graph.json")
            graph_cache[video_id] = {
                str(node["node_id"]): node for node in graph.get("nodes") or []
            }
        nodes = graph_cache[video_id]
        clues = tuple(hidden_case.get("clue_intervals") or [])
        question = str((public_case.get("planner_input") or {}).get("question") or "")
        split = str(public_case.get("split") or "")
        localization = run.get("entry_localization") or {}
        anchor_ids = tuple(localization.get("selected_node_ids") or [])
        if len(anchor_ids) < 2:
            continue
        if any(node_id not in nodes for node_id in anchor_ids):
            raise ValueError(f"entry anchor missing from graph for {case_id}")

        first_step = (run.get("steps") or [{}])[0]
        belief = first_step.get("belief_before") or {}
        predictions = _initial_predictions(first_step)
        if not predictions:
            belief_id = str(belief.get("belief_id") or "")
            predictions = {
                target_id: cached_predictions[(belief_id, target_id)]
                for target_id in anchor_ids
                if (belief_id, target_id) in cached_predictions
            }
        if set(predictions) != set(anchor_ids):
            missing = sorted(set(anchor_ids) - set(predictions))
            raise ValueError(
                f"incomplete first-hop predictions for {case_id}: {missing}"
            )

        aliases = {
            node_id: f"candidate_{_letters(index)}"
            for index, node_id in enumerate(anchor_ids)
        }
        record_id = _stable_id("local_choice", case_id, "0")
        public_candidates = [
            {
                "candidate": aliases[node_id],
                "predicted_transition": _public_transition(predictions[node_id]),
            }
            for node_id in anchor_ids
        ]
        pair_labels: list[dict[str, Any]] = []
        grounded_endpoints = {
            node_id: _grounded_endpoint(nodes[node_id], clues) for node_id in anchor_ids
        }
        for left_id, right_id in combinations(anchor_ids, 2):
            left_clues = tuple(grounded_endpoints[left_id]["grounded_clue_ids"])
            right_clues = tuple(grounded_endpoints[right_id]["grounded_clue_ids"])
            label, basis = _categorical_label(left_clues, right_clues)
            pair_labels.append(
                {
                    "left": aliases[left_id],
                    "right": aliases[right_id],
                    "label": label,
                    "basis": basis,
                }
            )
        public_choices.append(
            {
                "record_id": record_id,
                "case_id": case_id,
                "split": split,
                "question": question,
                "belief": {
                    "answerability": belief.get("answerability"),
                    "required_roles": list(belief.get("required_roles") or []),
                    "missing_roles": list(belief.get("missing_roles") or []),
                    "contradictions": list(belief.get("contradictions") or []),
                },
                "candidates": public_candidates,
                "pairwise_labels": None,
            }
        )
        hidden_choices.append(
            {
                "record_id": record_id,
                "case_id": case_id,
                "split": split,
                "candidate_grounding": {
                    aliases[node_id]: grounded_endpoints[node_id]
                    for node_id in anchor_ids
                },
                "pairwise_labels": pair_labels,
                "label_source": (
                    "dataset_gt_clue_interval_overlap_after_model_planning"
                ),
            }
        )

        selected_action = first_step.get("selected_action") or {}
        target_id = str(selected_action.get("target_id") or "")
        realized = first_step.get("realized_label_evaluator_only")
        if target_id in predictions and isinstance(realized, dict):
            correction_id = _stable_id("transition_correction", case_id, "0")
            public_corrections.append(
                {
                    "record_id": correction_id,
                    "case_id": case_id,
                    "split": split,
                    "question": question,
                    "candidate": {
                        "address_descriptor": list(
                            (predictions[target_id].get("observation") or {}).get(
                                "descriptor"
                            )
                            or []
                        ),
                        "predicted_transition": _public_transition(
                            predictions[target_id]
                        ),
                    },
                    "grounded_navigation_label": None,
                }
            )
            hidden_corrections.append(
                {
                    "record_id": correction_id,
                    "case_id": case_id,
                    "split": split,
                    "target": _grounded_endpoint(nodes[target_id], clues),
                    "grounded_navigation_label": {
                        "navigation_outcome": realized.get("observation_outcome"),
                        "navigation_belief_delta": {
                            key: value
                            for key, value in (
                                realized.get("belief_delta") or {}
                            ).items()
                            if key != "predicted_only"
                        },
                    },
                    "prediction_matches_grounded_navigation_label": (
                        _transition_matches(predictions[target_id], realized)
                    ),
                    "label_source": (
                        "executed_read_plus_dataset_gt_clue_coverage_evaluator"
                    ),
                    "label_scope": (
                        "navigation_progress_only_not_general_semantic_truth"
                    ),
                }
            )

    public = {
        "schema_version": PUBLIC_SCHEMA,
        "local_choice_records": public_choices,
        "transition_correction_records": public_corrections,
        "record_counts": {
            "local_choice": len(public_choices),
            "transition_correction": len(public_corrections),
        },
        "label_visibility": "separate_hidden_key",
        "candidate_order_semantics": "none",
        "outside_clue_negative_labels_created": False,
        "numeric_reward_present": False,
        "top_k_applied": False,
        "training_ready": False,
        "training_performed": False,
    }
    hidden = {
        "schema_version": HIDDEN_SCHEMA,
        "local_choice_records": hidden_choices,
        "transition_correction_records": hidden_corrections,
        "fed_back_to_localizer_or_planner": False,
        "answer_text_stored": False,
        "outside_clue_grounding": "not_established_never_negative",
        "numeric_reward_present": False,
        "training_ready": False,
        "training_performed": False,
        "analysis": _analysis(hidden_choices, hidden_corrections),
    }
    return public, hidden


def _initial_predictions(step: dict[str, Any]) -> dict[str, dict[str, Any]]:
    decision = step.get("decision") or {}
    predictions: dict[str, dict[str, Any]] = {}
    for trajectory in decision.get("initial_trajectories") or []:
        transitions = trajectory.get("transitions") or []
        if not transitions:
            continue
        transition = transitions[0]
        action = transition.get("action") or {}
        target_id = str(action.get("target_id") or "")
        if action.get("kind") == "start_at" and target_id:
            predictions[target_id] = transition
    return predictions


def _cache_predictions(
    cache: dict[str, Any],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    entries = cache.get("entries") or {}
    if not isinstance(entries, dict):
        raise ValueError("transition cache entries must be an object")
    for row in entries.values():
        audit = row.get("request_audit") or {}
        if audit.get("action_kind") != "start_at" or audit.get("parent_action_ids"):
            continue
        belief_id = str(audit.get("belief_id") or "")
        target_id = str(audit.get("target_id") or "")
        if belief_id and target_id:
            result[(belief_id, target_id)] = row["transition"]
    return result


def _public_transition(value: dict[str, Any]) -> dict[str, Any]:
    observation = value.get("observation") or {}
    return {
        "observation": {
            "outcome": observation.get("outcome"),
            "descriptor": list(observation.get("descriptor") or []),
        },
        "belief_delta": {
            key: item
            for key, item in (value.get("belief_delta") or {}).items()
            if key != "predicted_only"
        },
    }


def _grounded_endpoint(
    node: dict[str, Any], clues: tuple[dict[str, Any], ...]
) -> dict[str, Any]:
    span = node.get("time_span") or {}
    clue_ids = [
        f"clue_{index}"
        for index, clue in enumerate(clues)
        if max(float(span["start_s"]), float(clue["start_s"]))
        < min(float(span["end_s"]), float(clue["end_s"]))
    ]
    return {
        "target_id": node.get("node_id"),
        "time_span": span,
        "real_evidence_descriptor": node.get("text"),
        "grounded_clue_ids": clue_ids,
        "grounded_relevance": "relevant" if clue_ids else "not_established",
        "provenance": node.get("provenance"),
    }


def _categorical_label(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[str, str]:
    if left and not right:
        return "prefer_left", "left_only_covers_a_dataset_grounded_clue"
    if right and not left:
        return "prefer_right", "right_only_covers_a_dataset_grounded_clue"
    if left and left == right:
        return "tie", "same_nonempty_dataset_grounded_clue_coverage"
    return "incomparable", "ground_truth_does_not_establish_strict_order"


def _transition_matches(predicted: dict[str, Any], realized: dict[str, Any]) -> bool:
    predicted_public = _public_transition(predicted)
    realized_public = {
        "observation": {
            "outcome": realized.get("observation_outcome"),
            "descriptor": list(predicted_public["observation"].get("descriptor") or []),
        },
        "belief_delta": {
            key: value
            for key, value in (realized.get("belief_delta") or {}).items()
            if key != "predicted_only"
        },
    }
    return predicted_public == realized_public


def _analysis(
    choices: list[dict[str, Any]], corrections: list[dict[str, Any]]
) -> dict[str, Any]:
    labels = Counter(
        pair["label"] for record in choices for pair in record["pairwise_labels"]
    )
    split_counts = Counter(record["split"] for record in choices)
    return {
        "choice_record_count": len(choices),
        "transition_correction_record_count": len(corrections),
        "pairwise_label_counts": dict(sorted(labels.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "strict_pairwise_preference_count": (
            labels["prefer_left"] + labels["prefer_right"]
        ),
        "transition_prediction_navigation_mismatch_count": sum(
            not row["prediction_matches_grounded_navigation_label"]
            for row in corrections
        ),
        "diagnostic": (
            "grounded local-choice supervision; no scalar reward and no "
            "outside-clue standalone negatives"
        ),
    }


def _letters(index: int) -> str:
    value = index + 1
    chars: list[str] = []
    while value:
        value, remainder = divmod(value - 1, 26)
        chars.append(chr(ord("a") + remainder))
    return "".join(reversed(chars))


def _stable_id(prefix: str, *parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return f"{prefix}:" + hashlib.sha256(payload).hexdigest()[:20]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--transition-cache", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-output", required=True, type=Path)
    args = parser.parse_args(argv)
    runs = [row for path in args.run for row in (_read_json(path).get("runs") or [])]
    public, hidden = build_grounded_local_choice_packet(
        runs=runs,
        dataset=_read_json(args.dataset),
        hidden_key=_read_json(args.hidden_key),
        graph_root=args.graph_root.expanduser().resolve(),
        transition_cache=(
            _read_json(args.transition_cache)
            if args.transition_cache is not None
            else None
        ),
    )
    _write_json(args.output, public)
    _write_json(args.hidden_output, hidden)
    print(json.dumps(hidden["analysis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
