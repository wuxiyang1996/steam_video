"""Export grounded categorical labels for tied full-IWM survivor pairs."""

from __future__ import annotations

import argparse
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA = "steam-full-iwm-grounded-survivor-preference/v0.1"
HIDDEN_SCHEMA = "steam-full-iwm-grounded-survivor-preference-hidden/v0.1"


def build_grounded_survivor_preference_packet(
    *,
    runs: Iterable[dict[str, Any]],
    dataset: dict[str, Any],
    hidden_key: dict[str, Any],
    graph_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create blinded inputs and separate GT-derived categorical labels.

    Hidden clue intervals are used only after planning. They never enter the
    public packet or any planner/world-model call.
    """

    hidden_cases = {
        str(row["case_id"]): row for row in hidden_key.get("cases") or []
    }
    public_cases = {
        str(row["case_id"]): row for row in dataset.get("cases") or []
    }
    graph_cache: dict[str, dict[str, Any]] = {}
    public_records: list[dict[str, Any]] = []
    hidden_records: list[dict[str, Any]] = []
    for run in runs:
        case_id = str(run.get("case_id") or "")
        if case_id not in hidden_cases:
            raise ValueError(f"missing hidden case for survivor packet: {case_id}")
        if case_id not in public_cases:
            raise ValueError(f"missing public case for survivor packet: {case_id}")
        hidden_case = hidden_cases[case_id]
        public_case = public_cases[case_id]
        question = str((public_case.get("planner_input") or {}).get("question") or "")
        video_id = str(hidden_case["video_id"])
        if video_id not in graph_cache:
            graph_path = graph_root / video_id / "l1_l15_navigation_graph.json"
            graph_cache[video_id] = _read_json(graph_path)
        nodes = {
            str(node["node_id"]): node
            for node in graph_cache[video_id].get("nodes") or []
        }
        clue_intervals = tuple(hidden_case.get("clue_intervals") or [])
        for step_index, step in enumerate(run.get("steps") or []):
            decision = step.get("decision") or {}
            survivor_ids = tuple(decision.get("undominated_trajectory_ids") or [])
            if not survivor_ids:
                continue
            by_id = {
                str(row["trajectory_id"]): row
                for row in decision.get("trajectories") or []
            }
            survivors = tuple(by_id[value] for value in survivor_ids)
            all_trajectories = tuple(by_id.values())
            pair_rows: list[tuple[dict[str, Any], dict[str, Any], str]] = [
                (left, right, "final_survivor_tie")
                for left, right in combinations(survivors, 2)
            ]
            for survivor in survivors:
                survivor_node = _optional_target_node(_first_action(survivor), nodes)
                if survivor_node is None:
                    continue
                survivor_clues = _overlapping_clues(survivor_node, clue_intervals)
                for candidate in all_trajectories:
                    if candidate["trajectory_id"] == survivor["trajectory_id"]:
                        continue
                    candidate_node = _optional_target_node(
                        _first_action(candidate), nodes
                    )
                    if candidate_node is None:
                        continue
                    candidate_clues = _overlapping_clues(
                        candidate_node, clue_intervals
                    )
                    if bool(survivor_clues) != bool(candidate_clues):
                        pair_rows.append(
                            (survivor, candidate, "executed_survivor_contrast")
                        )
            seen_action_pairs: set[tuple[str, str]] = set()
            for left, right, pair_source in pair_rows:
                left_action = _first_action(left)
                right_action = _first_action(right)
                if left_action.get("action_id") == right_action.get("action_id"):
                    continue
                action_pair = tuple(
                    sorted(
                        (str(left_action["action_id"]), str(right_action["action_id"]))
                    )
                )
                if action_pair in seen_action_pairs:
                    continue
                seen_action_pairs.add(action_pair)
                left_node = _target_node(left_action, nodes)
                right_node = _target_node(right_action, nodes)
                pair_id = _pair_id(case_id, step_index, left, right)
                public_records.append(
                    {
                        "pair_id": pair_id,
                        "case_id": case_id,
                        "arm": run.get("arm"),
                        "step_index": step_index,
                        "pair_source": pair_source,
                        "belief": _public_belief(
                            step.get("belief_before") or {}, question=question
                        ),
                        "left": _anonymous_trajectory(left),
                        "right": _anonymous_trajectory(right),
                        "label": None,
                    }
                )
                left_clues = _overlapping_clues(left_node, clue_intervals)
                right_clues = _overlapping_clues(right_node, clue_intervals)
                label, basis = _categorical_label(left_clues, right_clues)
                hidden_records.append(
                    {
                        "pair_id": pair_id,
                        "case_id": case_id,
                        "arm": run.get("arm"),
                        "step_index": step_index,
                        "pair_source": pair_source,
                        "label": label,
                        "basis": basis,
                        "left": _grounded_endpoint(left_node, left_clues),
                        "right": _grounded_endpoint(right_node, right_clues),
                        "outcome_belief_delta_collision": _prediction_signature(left)
                        == _prediction_signature(right),
                    }
                )
    public = {
        "schema_version": SCHEMA,
        "records": public_records,
        "record_count": len(public_records),
        "label_visibility": "separate_hidden_key",
        "hidden_clue_or_answer_present": False,
        "numeric_reward_present": False,
        "training_performed": False,
    }
    hidden = {
        "schema_version": HIDDEN_SCHEMA,
        "records": hidden_records,
        "record_count": len(hidden_records),
        "label_source": "dataset_gt_clue_interval_overlap_after_planning",
        "fed_back_to_planner": False,
        "answer_text_stored": False,
        "numeric_reward_present": False,
        "training_performed": False,
        "analysis": _analysis(hidden_records),
    }
    return public, hidden


def _first_action(trajectory: dict[str, Any]) -> dict[str, Any]:
    transitions = trajectory.get("transitions") or []
    if not transitions or not isinstance(transitions[0].get("action"), dict):
        raise ValueError("survivor trajectory has no first action")
    return transitions[0]["action"]


def _target_node(
    action: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    target_id = str(action.get("target_id") or "")
    if not action.get("reads_evidence") or target_id not in nodes:
        raise ValueError("grounded survivor pair requires two real read targets")
    return nodes[target_id]


def _optional_target_node(
    action: dict[str, Any], nodes: dict[str, dict[str, Any]]
) -> dict[str, Any] | None:
    target_id = str(action.get("target_id") or "")
    if not action.get("reads_evidence") or target_id not in nodes:
        return None
    return nodes[target_id]


def _public_belief(value: dict[str, Any], *, question: str) -> dict[str, Any]:
    return {
        "question": question,
        "answerability": value.get("answerability"),
        "required_roles": list(value.get("required_roles") or []),
        "missing_roles": list(value.get("missing_roles") or []),
        "contradictions": list(value.get("contradictions") or []),
    }


def _anonymous_trajectory(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "transitions": [
            {
                "observation_outcome": (transition.get("observation") or {}).get(
                    "outcome"
                ),
                "observation_descriptor": list(
                    (transition.get("observation") or {}).get("descriptor") or []
                ),
                "belief_delta": {
                    key: item
                    for key, item in (transition.get("belief_delta") or {}).items()
                    if key != "predicted_only"
                },
            }
            for transition in value.get("transitions") or []
        ]
    }


def _overlapping_clues(
    node: dict[str, Any], clue_intervals: tuple[dict[str, Any], ...]
) -> tuple[str, ...]:
    span = node.get("time_span") or {}
    start = float(span["start_s"])
    end = float(span["end_s"])
    return tuple(
        f"clue_{index}"
        for index, clue in enumerate(clue_intervals)
        if max(start, float(clue["start_s"])) < min(end, float(clue["end_s"]))
    )


def _categorical_label(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[str, str]:
    if left and not right:
        return "prefer_left", "left_only_has_dataset_grounded_clue_overlap"
    if right and not left:
        return "prefer_right", "right_only_has_dataset_grounded_clue_overlap"
    if left and left == right:
        return "tie", "same_nonempty_dataset_grounded_clue_coverage"
    return "incomparable", "ground_truth_does_not_establish_strict_pair_order"


def _grounded_endpoint(
    node: dict[str, Any], clue_ids: tuple[str, ...]
) -> dict[str, Any]:
    return {
        "target_id": node.get("node_id"),
        "time_span": node.get("time_span"),
        "real_evidence_descriptor": node.get("text"),
        "grounded_clue_ids": list(clue_ids),
        "grounded_relevance": "relevant" if clue_ids else "not_established",
        "provenance": node.get("provenance"),
    }


def _prediction_signature(trajectory: dict[str, Any]) -> str:
    value = [
        {
            "outcome": (row.get("observation") or {}).get("outcome"),
            "belief_delta": row.get("belief_delta"),
        }
        for row in trajectory.get("transitions") or []
    ]
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    labels: dict[str, int] = {}
    cases: dict[str, int] = {}
    sources: dict[str, int] = {}
    for row in records:
        label = str(row["label"])
        labels[label] = labels.get(label, 0) + 1
        case_id = str(row["case_id"])
        cases[case_id] = cases.get(case_id, 0) + 1
        source = str(row["pair_source"])
        sources[source] = sources.get(source, 0) + 1
    collision_count = sum(
        bool(row["outcome_belief_delta_collision"]) for row in records
    )
    return {
        "label_counts": labels,
        "case_counts": cases,
        "pair_source_counts": sources,
        "outcome_belief_delta_collision_count": collision_count,
        "strict_preferences_hidden_by_prediction_collision": sum(
            bool(row["outcome_belief_delta_collision"])
            and row["label"] in {"prefer_left", "prefer_right"}
            for row in records
        ),
        "strict_preferences_without_exact_collision": sum(
            not bool(row["outcome_belief_delta_collision"])
            and row["label"] in {"prefer_left", "prefer_right"}
            for row in records
        ),
        "diagnostic": (
            "grounded endpoints differ but imagined outcome/belief-delta descriptors "
            "collide"
            if collision_count
            else "no exact imagined-transition collision"
        ),
    }


def _pair_id(
    case_id: str,
    step_index: int,
    left: dict[str, Any],
    right: dict[str, Any],
) -> str:
    value = "\x1f".join(
        (
            case_id,
            str(step_index),
            str(left["trajectory_id"]),
            str(right["trajectory_id"]),
        )
    )
    return "survivor_pair:" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


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
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-output", required=True, type=Path)
    args = parser.parse_args(argv)
    run_rows = [
        row
        for path in args.run
        for row in (_read_json(path).get("runs") or [])
    ]
    public, hidden = build_grounded_survivor_preference_packet(
        runs=run_rows,
        dataset=_read_json(args.dataset),
        hidden_key=_read_json(args.hidden_key),
        graph_root=args.graph_root.expanduser().resolve(),
    )
    _write_json(args.output, public)
    _write_json(args.hidden_output, hidden)
    print(json.dumps(hidden["analysis"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
