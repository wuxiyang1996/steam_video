"""Collect independent, legal two-read delayed transitions from frozen graphs.

Ground truth is used only to select and label data-gathering paths after the
question-independent graph and question-localized entry frontier are frozen.
No hidden clue interval enters an IWM input and no model service is called.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from steam_video_new.implicit_world_model.full_graph_iwm.action_compiler import (
    GraphActionCompiler,
    execute_graph_action,
)
from steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot import (
    _build_graph_with_optional_caption_candidates,
    _clues,
    _covered_clue_indices_by_node_ids,
)
from steam_video_new.implicit_world_model.full_graph_iwm.closed_loop import (
    graph_fingerprint,
)
from steam_video_new.implicit_world_model.full_graph_iwm.contracts import (
    CursorBeliefState,
)
from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
    load_overlay_artifact,
)


SCHEMA_VERSION = "steam-iwm-targeted-delayed-collection/v0.1"


def collect_targeted_delayed_paths(
    dataset: Mapping[str, Any],
    hidden_key: Mapping[str, Any],
    localization_artifact: Mapping[str, Any],
    *,
    graph_root: Path,
    split: str = "validation",
    capacity: int = 256,
    minimum_units: int = 5,
    maximum_units: int = 6,
) -> dict[str, Any]:
    if minimum_units < 1 or maximum_units < minimum_units:
        raise ValueError("invalid delayed-unit limits")
    public = {str(row["case_id"]): row for row in dataset.get("cases") or ()}
    hidden = {str(row["case_id"]): row for row in hidden_key.get("cases") or ()}
    localizations, roles = _localization_state(localization_artifact)
    candidates: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for case_id in sorted(localizations):
        public_case = public.get(case_id)
        hidden_case = hidden.get(case_id)
        if (
            public_case is None
            or hidden_case is None
            or str(public_case.get("split") or "") != split
        ):
            continue
        video_id = str(public_case["video_id"])
        sample_dir = graph_root / video_id
        try:
            loaded = load_overlay_artifact(
                sample_dir / "causal_temporal_overlay.json", validate_schema=True
            )
            graph = _build_graph_with_optional_caption_candidates(
                loaded.overlay,
                sample_dir=sample_dir,
                capacity=capacity,
                include_caption_candidates=False,
            )
            anchors = tuple(
                node_id
                for node_id in localizations[case_id]
                if node_id in graph.node_by_id
            )
            if not anchors:
                raise ValueError("no frozen localized entry anchor survived")
            case_candidates = _case_candidates(
                case_id,
                public_case,
                hidden_case,
                graph,
                anchors=anchors,
                required_roles=roles.get(case_id, ()),
            )
            candidates.extend(case_candidates)
        except (KeyError, TypeError, ValueError) as exc:
            failures.append(
                {"case_id": case_id, "failure": f"{type(exc).__name__}: {exc}"}
            )

    selected: list[dict[str, Any]] = []
    used_videos: set[str] = set()
    for row in sorted(
        candidates,
        key=lambda value: (
            value["case_id"],
            value["first_action"]["action_id"],
            value["second_action"]["action_id"],
        ),
    ):
        if row["video_id"] in used_videos:
            continue
        selected.append(row)
        used_videos.add(row["video_id"])
        if len(selected) == maximum_units:
            break
    if len(selected) < minimum_units:
        raise ValueError(
            f"only {len(selected)} independent delayed units; need {minimum_units}"
        )
    runs = [_selected_path_to_run(row) for row in selected]
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": str(dataset.get("dataset_id") or "unknown"),
        "split": split,
        "graph_root": str(graph_root.resolve()),
        "capacity": capacity,
        "runs": runs,
        "selected_unit_count": len(runs),
        "selected_video_count": len({run["video_id"] for run in runs}),
        "selected_action_kind_counts": _action_kind_counts(runs),
        "candidate_path_count": len(candidates),
        "collection_failures": failures,
        "contract": {
            "question_independent_graph": True,
            "entry_frontier_frozen_before_gt_join": True,
            "first_read_has_no_dataset_clue_overlap": True,
            "second_read_has_dataset_clue_overlap": True,
            "both_actions_compiled_as_legal": True,
            "selected_nodes_have_materialized_qwen_embedding_refs": True,
            "hidden_clue_intervals_in_model_input": False,
            "gt_used_for_targeted_collection_and_evaluator_label_only": True,
            "model_service_called": False,
            "numeric_reward_present": False,
            "top_k_applied": False,
        },
        "training_performed": False,
    }


def _localization_state(
    artifact: Mapping[str, Any],
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    localizations: dict[str, tuple[str, ...]] = {}
    roles: dict[str, tuple[str, ...]] = {}
    for run in artifact.get("runs") or ():
        case_id = str(run.get("case_id") or "")
        selected = (run.get("entry_localization") or {}).get("selected_node_ids")
        if case_id and selected:
            localizations.setdefault(case_id, tuple(str(value) for value in selected))
        for step in run.get("steps") or ():
            pool = (step.get("pool_before") or {}).get("trajectories") or ()
            belief = step.get("belief_before") or (
                (pool[0].get("belief") or {}) if pool else {}
            )
            localized = belief.get("localized_entry_node_ids") or ()
            if case_id and localized:
                localizations.setdefault(
                    case_id, tuple(str(value) for value in localized)
                )
            required = belief.get("required_roles") or ()
            if case_id and required:
                roles.setdefault(case_id, tuple(str(value) for value in required))
                break
    return localizations, roles


def _case_candidates(
    case_id: str,
    public_case: Mapping[str, Any],
    hidden_case: Mapping[str, Any],
    graph: Any,
    *,
    anchors: Sequence[str],
    required_roles: Sequence[str],
) -> list[dict[str, Any]]:
    question = str((public_case.get("planner_input") or {}).get("question") or "")
    choices = tuple(
        str(value) for value in (public_case.get("planner_input") or {}).get("choices") or ()
    )
    clues = _clues(dict(hidden_case))
    initial = CursorBeliefState(
        belief_id=f"belief:{case_id}:targeted-delayed:initial",
        question=question,
        localized_entry_node_ids=tuple(anchors),
        required_roles=tuple(required_roles),
        missing_roles=tuple(required_roles),
        remaining_reads=2,
    )
    compiler = GraphActionCompiler()
    candidates: list[dict[str, Any]] = []
    for first in compiler.compile(initial, graph):
        if not first.reads_evidence or first.target_id is None:
            continue
        if _covered_clue_indices_by_node_ids(graph, clues, {first.target_id}):
            continue
        first_execution = execute_graph_action(first, initial, graph)
        if (
            first_execution.observation is None
            or first_execution.observation.embedding_ref is None
        ):
            continue
        for second in compiler.compile(first_execution.updated_belief, graph):
            if not second.reads_evidence or second.target_id is None:
                continue
            newly_covered = _covered_clue_indices_by_node_ids(
                graph, clues, {second.target_id}
            )
            if not newly_covered:
                continue
            second_execution = execute_graph_action(
                second, first_execution.updated_belief, graph
            )
            if (
                second_execution.observation is None
                or second_execution.observation.embedding_ref is None
            ):
                continue
            candidates.append(
                {
                    "case_id": case_id,
                    "video_id": str(public_case["video_id"]),
                    "split": str(public_case["split"]),
                    "question": question,
                    "choices": choices,
                    "required_roles": tuple(required_roles),
                    "anchors": tuple(anchors),
                    "graph_fingerprint": graph_fingerprint(graph),
                    "first_action": asdict(first),
                    "second_action": asdict(second),
                    "first_observation": first_execution.observation.to_dict(),
                    "second_observation": second_execution.observation.to_dict(),
                    "newly_covered": tuple(sorted(newly_covered)),
                    "belief_before": asdict(initial),
                    "belief_after_first": asdict(first_execution.updated_belief),
                    "belief_after_second": asdict(second_execution.updated_belief),
                }
            )
    return candidates


def _selected_path_to_run(row: Mapping[str, Any]) -> dict[str, Any]:
    before = dict(row["belief_before"])
    after_first = dict(row["belief_after_first"])
    after_second = dict(row["belief_after_second"])
    first_id = str(row["first_observation"]["node_id"])
    second_id = str(row["second_observation"]["node_id"])
    choices = tuple(row["choices"])
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": row["case_id"],
        "video_id": row["video_id"],
        "split": row["split"],
        "arm": "targeted_grounded_delayed_collection",
        "graph_fingerprint": row["graph_fingerprint"],
        "real_observations": [row["first_observation"], row["second_observation"]],
        "realized_labels_evaluator_only": [
            {"observation_id": first_id, "newly_covered_clue_indices": []},
            {
                "observation_id": second_id,
                "newly_covered_clue_indices": list(row["newly_covered"]),
            },
        ],
        "steps": [
            _step(
                row["first_action"],
                before,
                after_first,
                choices,
                step_index=0,
            ),
            _step(
                row["second_action"],
                after_first,
                after_second,
                choices,
                step_index=1,
            ),
        ],
        "entry_localization": {
            "selected_node_ids": list(row["anchors"]),
            "source": "frozen_model_localization_artifact",
            "hidden_supervision_used": False,
        },
        "method_audit": {
            "arm_runtime_failed": False,
            "targeted_delayed_collection": True,
            "gt_used_for_path_selection_evaluator_only": True,
            "hidden_gt_in_transition_input": False,
            "both_reads_executed_on_frozen_graph": True,
        },
    }


def _step(
    action: Mapping[str, Any],
    belief_before: Mapping[str, Any],
    belief_after: Mapping[str, Any],
    choices: Sequence[str],
    *,
    step_index: int,
) -> dict[str, Any]:
    observation_id = str(action["target_id"])
    return {
        "observation_id": observation_id,
        "decision": {
            "selected_action": dict(action),
            "planning_status": "targeted_grounded_collection_evaluator_only",
            "top_k_applied": False,
        },
        "pool_before": {
            "trajectories": _paths(choices, belief_before, step_index=step_index)
        },
        "pool_after": {
            "trajectories": _paths(choices, belief_after, step_index=step_index + 1)
        },
    }


def _paths(
    choices: Sequence[str], belief: Mapping[str, Any], *, step_index: int
) -> list[dict[str, Any]]:
    return [
        {
            "trajectory_id": f"trajectory:targeted:{index}",
            "hypothesis": hypothesis,
            "belief": deepcopy(dict(belief)),
            "step_index": step_index,
        }
        for index, hypothesis in enumerate(choices or ("",))
    ]


def _action_kind_counts(runs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        action = run["steps"][1]["decision"]["selected_action"]
        raw_kind = action["kind"]
        kind = str(getattr(raw_kind, "value", raw_kind))
        counts[kind] = counts.get(kind, 0) + 1
    return dict(sorted(counts.items()))


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--localization-artifact", required=True, type=Path)
    parser.add_argument("--graph-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--minimum-units", type=int, default=5)
    parser.add_argument("--maximum-units", type=int, default=6)
    args = parser.parse_args(argv)
    result = collect_targeted_delayed_paths(
        _read(args.dataset),
        _read(args.hidden_key),
        _read(args.localization_artifact),
        graph_root=args.graph_root,
        split=args.split,
        capacity=args.capacity,
        minimum_units=args.minimum_units,
        maximum_units=args.maximum_units,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "runs"},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
