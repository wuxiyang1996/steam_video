"""Build gate-controlled 9B data from grounded closed-loop executions.

The transition target uses only an executed L1 observation and evaluator-only
dataset clue coverage.  Model-generated belief corrections remain visible as
diagnostics but are never promoted to supervised fields.  Planner preferences
are derived ordinally from realized clue-set inclusion and remain ineligible
until the transition gate passes.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    PLANNER_TASK,
    SCHEMA_VERSION,
    TRANSITION_TASK,
    supervised_field,
    validate_sft_record,
)


RUNTIME_DATA_SCHEMA = "steam-grounded-runtime-iwm-data/v0.2"
EXECUTED_OBSERVATION = "executed_question_independent_l1_observation"
DATASET_CLUE_EVALUATOR = "dataset_gt_clue_overlap_after_graph_freeze"
NOT_SUPERVISED = "not_independently_supervised"
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def build_grounded_runtime_data(
    artifacts: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    hidden_key: Mapping[str, Any],
    *,
    minimum_train_videos: int = 20,
    minimum_train_records: int = 100,
    minimum_validation_videos: int = 5,
    minimum_validation_records: int = 20,
    minimum_delayed_positives: int = 5,
    minimum_delayed_positive_units: int = 5,
    minimum_semantic_hard_negatives: int = 10,
    minimum_validation_delayed_positives: int = 1,
    minimum_validation_delayed_positive_units: int = 3,
    minimum_validation_semantic_hard_negatives: int = 1,
    transition_calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    public = {str(row["case_id"]): row for row in dataset.get("cases") or ()}
    hidden = {str(row["case_id"]): row for row in hidden_key.get("cases") or ()}
    runs = _deduplicated_runs(artifacts)
    transition_records = _transition_records(runs, public, hidden)
    split_audit = _split_audit(transition_records)
    counts = _transition_counts(transition_records)
    gate = _transition_gate(
        counts,
        split_audit,
        minimum_train_videos=minimum_train_videos,
        minimum_train_records=minimum_train_records,
        minimum_validation_videos=minimum_validation_videos,
        minimum_validation_records=minimum_validation_records,
        minimum_delayed_positives=minimum_delayed_positives,
        minimum_delayed_positive_units=minimum_delayed_positive_units,
        minimum_semantic_hard_negatives=minimum_semantic_hard_negatives,
        minimum_validation_delayed_positives=(
            minimum_validation_delayed_positives
        ),
        minimum_validation_delayed_positive_units=(
            minimum_validation_delayed_positive_units
        ),
        minimum_validation_semantic_hard_negatives=(
            minimum_validation_semantic_hard_negatives
        ),
    )
    if gate["passed"]:
        transition_records = [
            {
                **row,
                "training_eligible": row["split"] == "train",
                "eligibility_reasons": (
                    ["grounded_runtime_transition_gate_passed"]
                    if row["split"] == "train"
                    else ["held_out_video_split"]
                ),
            }
            for row in transition_records
        ]
    calibration_passed = bool(
        transition_calibration
        and transition_calibration.get("schema_version")
        == "steam-iwm-transition-calibration/v0.1"
        and transition_calibration.get("split") == "validation"
        and transition_calibration.get("passed") is True
    )
    planner_records = _planner_preference_records(
        runs,
        public,
    )
    planner_gate = _planner_gate(
        planner_records,
        gate,
        transition_calibration_passed=calibration_passed,
    )
    if planner_gate["passed"]:
        planner_records = [
            {
                **row,
                "training_eligible": row["split"] == "train",
                "eligibility_reasons": (
                    ["planner_data_and_transition_calibration_gates_passed"]
                    if row["split"] == "train"
                    else ["held_out_video_split"]
                ),
            }
            for row in planner_records
        ]
    for row in (*transition_records, *planner_records):
        validate_sft_record(row)
        if _contains_numeric_scalar(row["target"]):
            raise ValueError(f"numeric supervised output found in {row['record_id']}")
    return {
        "schema_version": RUNTIME_DATA_SCHEMA,
        "source_dataset_id": str(dataset.get("dataset_id") or "unknown"),
        "source_artifact_count": len(artifacts),
        "transition_records": transition_records,
        "planner_preference_records": planner_records,
        "transition_gate": gate,
        "transition_calibration_gate": {
            "provided": transition_calibration is not None,
            "passed": calibration_passed,
            "required_for_planner_training": True,
        },
        "planner_gate": planner_gate,
        "split_audit": split_audit,
        "transition_counts": counts,
        "contracts": {
            "target_observation_is_executed_not_imagined": True,
            "model_correction_is_not_supervision": True,
            "unsupported_hypothesis_specific_belief_fields_are_masked": True,
            "negative_means_executed_no_new_clue_not_missing_label": True,
            "preference_is_ordinal_set_inclusion_not_numeric_reward": True,
            "planner_training_waits_for_transition_data_and_calibration_gates": True,
            "hidden_clue_or_answer_in_model_input": False,
            "numeric_reward_present": False,
            "numeric_model_target_present": False,
            "top_k_applied": False,
            "gtsam_ranks_actions": False,
        },
        "transition_training_ready": bool(gate["passed"]),
        "planner_training_ready": bool(planner_gate["passed"]),
        "training_ready": bool(gate["passed"] and planner_gate["passed"]),
        "training_performed": False,
    }


def hydrate_legacy_executed_observations(
    artifacts: Sequence[Mapping[str, Any]],
    dataset: Mapping[str, Any],
    compile_gates: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Replay legacy executed node addresses against their exact frozen graph.

    Older cohort artifacts serialized the executed ``observation_id`` but not
    the observation payload.  This migration is allowed only when the compiled
    graph fingerprint and the run fingerprint both match.  It performs no model
    inference and creates no counterfactual reads.
    """

    from steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot import (
        _build_graph_with_optional_caption_candidates,
    )
    from steam_video_new.implicit_world_model.full_graph_iwm.closed_loop import (
        graph_fingerprint,
    )
    from steam_video_new.implicit_world_model.l15_graph_navigator.overlay_io import (
        load_overlay_artifact,
    )

    public = {str(row["case_id"]): row for row in dataset.get("cases") or ()}
    gate_by_case: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for gate in compile_gates:
        for row in gate.get("cases") or ():
            if row.get("runnable") is True:
                gate_by_case[str(row["case_id"])] = (gate, row)
    graph_cache: dict[tuple[str, str], Any] = {}
    hydrated = deepcopy(list(artifacts))
    for artifact in hydrated:
        for run in artifact.get("runs") or ():
            if run.get("real_observations") or not run.get("steps"):
                continue
            node_ids = {
                str(step.get("observation_id"))
                for step in run.get("steps") or ()
                if step.get("observation_id")
            }
            if not node_ids:
                continue
            case_id = str(run.get("case_id") or "")
            if case_id not in gate_by_case or case_id not in public:
                continue
            gate, gate_case = gate_by_case[case_id]
            video_id = str(public[case_id].get("video_id") or "")
            cache_key = (str(gate.get("graph_root") or ""), video_id)
            graph = graph_cache.get(cache_key)
            if graph is None:
                graph_path = (
                    Path(str(gate["graph_root"])).expanduser().resolve()
                    / video_id
                    / "causal_temporal_overlay.json"
                )
                loaded = load_overlay_artifact(graph_path, validate_schema=True)
                graph_row = next(
                    (
                        row
                        for row in gate.get("graphs") or ()
                        if str(row.get("video_id")) == video_id
                    ),
                    {},
                )
                graph = _build_graph_with_optional_caption_candidates(
                    loaded.overlay,
                    sample_dir=graph_path.parent,
                    capacity=int(gate.get("capacity") or graph_row.get("capacity")),
                    include_caption_candidates=bool(
                        graph_row.get("caption_candidate_overlay_loaded")
                    ),
                )
                expected = str(gate_case.get("graph_fingerprint") or "")
                actual = graph_fingerprint(graph)
                if not expected or actual != expected:
                    raise ValueError(f"frozen graph fingerprint mismatch for {case_id}")
                graph_cache[cache_key] = graph
            if str(run.get("graph_fingerprint") or "") != graph_fingerprint(graph):
                raise ValueError(f"run graph fingerprint mismatch for {case_id}")
            unknown = node_ids - set(graph.node_by_id)
            if unknown:
                raise ValueError(
                    f"executed observations missing from frozen graph for {case_id}: "
                    + ", ".join(sorted(unknown))
                )
            run["real_observations"] = [
                graph.node_by_id[node_id].to_dict() for node_id in sorted(node_ids)
            ]
            run.setdefault("method_audit", {})[
                "legacy_observation_payload_hydration"
            ] = {
                "performed": True,
                "source": "executed_observation_id_replayed_on_exact_frozen_graph",
                "graph_fingerprint_verified": True,
                "model_called": False,
                "counterfactual_read_created": False,
            }
    return hydrated


def _deduplicated_runs(
    artifacts: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    by_key: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for artifact in artifacts:
        for run in artifact.get("runs") or ():
            if not isinstance(run, Mapping) or run.get("arm") == "oracle_clue_ceiling":
                continue
            if run.get("method_audit", {}).get("arm_runtime_failed") is True:
                continue
            key = (
                str(run.get("case_id") or ""),
                str(run.get("arm") or ""),
                str(run.get("graph_fingerprint") or ""),
            )
            if all(key):
                by_key[key] = run
    return [by_key[key] for key in sorted(by_key)]


def _transition_records(
    runs: Sequence[Mapping[str, Any]],
    public: Mapping[str, Mapping[str, Any]],
    hidden: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for run in runs:
        case_id = str(run["case_id"])
        public_case = public.get(case_id)
        hidden_case = hidden.get(case_id)
        if public_case is None or hidden_case is None:
            continue
        split = str(public_case.get("split") or "")
        video_id = str(public_case.get("video_id") or hidden_case.get("video_id") or "")
        clue_count = len(hidden_case.get("clue_intervals") or ())
        observations = {
            str(row.get("node_id")): row for row in run.get("real_observations") or ()
        }
        realized = {
            str(row.get("observation_id")): row
            for row in run.get("realized_labels_evaluator_only") or ()
        }
        covered: set[int] = set()
        prior_step_advanced = False
        for step_index, step in enumerate(run.get("steps") or ()):
            direct_observation = step.get("real_observation")
            if isinstance(direct_observation, Mapping):
                observation = direct_observation
                observation_id = str(observation.get("node_id") or "")
                label = step.get("realized_label_evaluator_only")
            else:
                observation_id = str(step.get("observation_id") or "")
                observation = observations.get(observation_id)
                label = realized.get(observation_id)
            if observation is None or label is None:
                continue
            newly_covered = {
                int(value) for value in label.get("newly_covered_clue_indices") or ()
            }
            before_covered = set(covered)
            covered.update(newly_covered)
            outcome = "support" if newly_covered else "inconclusive"
            progress = "advanced" if newly_covered else "unchanged"
            coverage_after = (
                "complete"
                if clue_count and len(covered) == clue_count
                else "partial" if covered else "none"
            )
            answerability = "ready" if coverage_after == "complete" else "not_ready"
            delayed = bool(step_index > 0 and newly_covered and not prior_step_advanced)
            action = (
                step.get("selected_action")
                or step.get("decision", {}).get("selected_action")
                or {}
            )
            semantic_hard_negative = bool(
                not newly_covered
                and (
                    label.get("control_type")
                    == "surface_correlation_hard_negative"
                    or
                    action.get("kind") == "follow_correlation"
                    or "correlation" in str(action.get("relation") or "")
                )
            )
            before_by_id = {
                str(row.get("trajectory_id")): row
                for row in step.get("pool_before", {}).get("trajectories") or ()
            }
            after_by_id = {
                str(row.get("trajectory_id")): row
                for row in step.get("pool_after", {}).get("trajectories") or ()
            }
            if not before_by_id and isinstance(direct_observation, Mapping):
                before_by_id, after_by_id = _legacy_hypothesis_paths(
                    public_case,
                    step.get("belief_before") or {},
                    step.get("belief_after_real_read") or {},
                )
            for trajectory_id, before in before_by_id.items():
                after = after_by_id.get(trajectory_id)
                if after is None:
                    continue
                hypothesis = str(before.get("hypothesis") or "")
                belief = before.get("belief") or {}
                identity = {
                    "case_id": case_id,
                    "video_id": video_id,
                    "hypothesis": hypothesis,
                    "acquired_evidence": belief.get("acquired_evidence") or [],
                    "target_id": action.get("target_id"),
                    "outcome": outcome,
                }
                digest = _digest(identity)
                record = {
                    "schema_version": SCHEMA_VERSION,
                    "task": TRANSITION_TASK,
                    "record_id": f"runtime-transition:{digest}",
                    "case_id": case_id,
                    "video_id": video_id,
                    "split": split,
                    "input": {
                        "question": belief.get("question"),
                        "hypothesis": hypothesis,
                        "persistent_path_state": {
                            "required_roles": belief.get("required_roles") or [],
                            "missing_roles": belief.get("missing_roles") or [],
                            "grounded_role_evidence": (
                                belief.get("grounded_role_evidence") or []
                            ),
                            "contradictions": belief.get("contradictions") or [],
                            "answerability": belief.get("answerability"),
                            "acquired_evidence_ids": (
                                belief.get("acquired_evidence") or []
                            ),
                        },
                        "action": _safe_action(action),
                        "target_node_safe_view": _safe_target_view(observation),
                        "local_temporal_context": (
                            [action.get("relation")]
                            if "temporal" in str(action.get("kind") or "")
                            else []
                        ),
                        "local_correlation_context": (
                            [action.get("relation") or "l1.5_correlation"]
                            if action.get("kind") == "follow_correlation"
                            else []
                        ),
                        "imagined_prefix": [],
                    },
                    "target": _transition_target(
                        observation,
                        outcome=outcome,
                        progress=progress,
                        coverage_after=coverage_after,
                        answerability=answerability,
                    ),
                    "label_source": (
                        "executed_l1_observation_plus_dataset_clue_overlap"
                    ),
                    "training_eligible": False,
                    "eligibility_reasons": ["awaiting_global_transition_gate"],
                    "target_is_real_not_imagined": True,
                    "hidden_supervision_in_input": False,
                    "numeric_reward_present": False,
                    "data_slices": {
                        "delayed_positive": delayed,
                        "semantic_neighbor_hard_negative": semantic_hard_negative,
                        "hard_negative_reason": (
                            "surface_correlation_without_realized_clue_gain"
                            if semantic_hard_negative
                            else None
                        ),
                        "executed_inconclusive_control": not newly_covered,
                    },
                    "audit": {
                        "source_arm": run.get("arm"),
                        "source_graph_fingerprint": run.get("graph_fingerprint"),
                        "step_index": step_index,
                        "covered_clue_state_before": (
                            "partial" if before_covered else "none"
                        ),
                        "model_corrected_belief_after_not_supervised": (
                            after.get("belief") or {}
                        ),
                        "hidden_clue_indices_stored_in_input": False,
                    },
                }
                records.setdefault(record["record_id"], record)
            prior_step_advanced = prior_step_advanced or bool(newly_covered)
    return [records[key] for key in sorted(records)]


def _legacy_hypothesis_paths(
    public_case: Mapping[str, Any],
    belief_before: Mapping[str, Any],
    belief_after: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Expose old executed-step artifacts through the current path boundary.

    Older runs stored one shared belief and the executed observation directly
    on each step.  Public answer choices are expanded only to restore the
    hypothesis-conditioned input interface; delayed readiness is counted by
    independent video/question/target units, so this expansion cannot inflate
    the data gate.
    """

    planner_input = public_case.get("planner_input") or {}
    question = str(planner_input.get("question") or "")
    hypotheses = [str(value) for value in planner_input.get("choices") or ()]
    if not hypotheses:
        hypotheses = [""]
    before: dict[str, dict[str, Any]] = {}
    after: dict[str, dict[str, Any]] = {}
    for index, hypothesis in enumerate(hypotheses):
        trajectory_id = f"legacy-hypothesis:{index}"
        before[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "hypothesis": hypothesis,
            "belief": {**belief_before, "question": question},
        }
        after[trajectory_id] = {
            "trajectory_id": trajectory_id,
            "hypothesis": hypothesis,
            "belief": {**belief_after, "question": question},
        }
    return before, after


def _transition_target(
    observation: Mapping[str, Any],
    *,
    outcome: str,
    progress: str,
    coverage_after: str,
    answerability: str,
) -> dict[str, Any]:
    metadata = observation.get("metadata") or {}
    participants = [
        {
            key: value
            for key in (
                "role",
                "entity_type",
                "surface",
                "visual_signature",
                "mention_id",
                "track_status",
            )
            if (value := row.get(key)) not in (None, "", [], {})
        }
        for row in metadata.get("participants") or ()
        if isinstance(row, Mapping)
    ]
    descriptor = {
        "summary": str(observation.get("text") or metadata.get("predicate") or ""),
        "predicate": metadata.get("predicate"),
        "participants": participants,
        "states": _without_numeric_scalars(metadata.get("states") or []),
        "state_change": _without_numeric_scalars(metadata.get("state_change")),
    }
    return {
        "observation_patch": {
            "descriptor": supervised_field(
                descriptor, supervised=True, provenance=EXECUTED_OBSERVATION
            ),
            "observed_modalities": supervised_field(
                [str(metadata.get("modality") or "visual")],
                supervised=True,
                provenance=EXECUTED_OBSERVATION,
            ),
            "entity_bindings": supervised_field(
                participants,
                supervised=True,
                provenance=EXECUTED_OBSERVATION,
            ),
            "temporal_binding": supervised_field(
                "executed_target_interval",
                supervised=True,
                provenance=EXECUTED_OBSERVATION,
            ),
            "evidence_role": supervised_field(
                "clue_support" if outcome == "support" else "non_clue_control",
                supervised=True,
                provenance=DATASET_CLUE_EVALUATOR,
            ),
        },
        "belief_patch": {
            name: supervised_field(None, supervised=False, provenance=NOT_SUPERVISED)
            for name in (
                "supported_claims",
                "contradicted_claims",
                "newly_bound_variables",
                "opened_dependencies",
                "resolved_dependencies",
            )
        },
        "categorical_audit": {
            "progress": supervised_field(
                progress, supervised=True, provenance=DATASET_CLUE_EVALUATOR
            ),
            "required_clue_coverage_after": supervised_field(
                coverage_after,
                supervised=True,
                provenance=DATASET_CLUE_EVALUATOR,
            ),
            "answerability_after": supervised_field(
                answerability,
                supervised=True,
                provenance=DATASET_CLUE_EVALUATOR,
            ),
            "observation_outcome": supervised_field(
                outcome, supervised=True, provenance=DATASET_CLUE_EVALUATOR
            ),
            "contradiction_change": supervised_field(
                None, supervised=False, provenance=NOT_SUPERVISED
            ),
            "frontier_change": supervised_field(
                None, supervised=False, provenance=NOT_SUPERVISED
            ),
        },
    }


def _planner_preference_records(
    runs: Sequence[Mapping[str, Any]],
    public: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        case_id = str(run["case_id"])
        public_case = public.get(case_id)
        if public_case is None or not run.get("steps"):
            continue
        candidate = _realized_candidate(run)
        if candidate is not None:
            candidate["split"] = str(public_case.get("split") or "")
            candidate["video_id"] = str(public_case.get("video_id") or "")
            by_case[case_id].append(candidate)
    records: list[dict[str, Any]] = []
    for case_id, candidates in sorted(by_case.items()):
        unique = {
            tuple(row["executed_action_ids"]): row for row in candidates
        }
        for left, right in combinations(unique.values(), 2):
            left_coverage = set(left["_coverage"])
            right_coverage = set(right["_coverage"])
            if left_coverage > right_coverage:
                label = "prefer_left"
            elif right_coverage > left_coverage:
                label = "prefer_right"
            elif left_coverage == right_coverage:
                label = "tie"
            else:
                label = "incomparable"
            digest = _digest(
                {
                    "case_id": case_id,
                    "left": left["executed_action_ids"],
                    "right": right["executed_action_ids"],
                }
            )
            split = str(left["split"])
            record = {
                "schema_version": SCHEMA_VERSION,
                "task": PLANNER_TASK,
                "record_id": f"runtime-preference:{digest}",
                "case_id": case_id,
                "video_id": left["video_id"],
                "split": split,
                "input": {
                    "question": left["question"],
                    "persistent_trajectory_pool": left["pool_before"],
                    "left": _public_candidate(left),
                    "right": _public_candidate(right),
                    "comparison_scope": "complete_executed_two_read_trajectory",
                },
                "target": {
                    "preference": supervised_field(
                        label,
                        supervised=True,
                        provenance=(
                            "realized_clue_set_inclusion_after_graph_freeze"
                        ),
                    )
                },
                "label_source": "ordinal_realized_clue_set_inclusion",
                "training_eligible": False,
                "eligibility_reasons": ["awaiting_global_planner_gate"],
                "target_is_real_not_imagined": True,
                "hidden_supervision_in_input": False,
                "numeric_reward_present": False,
            }
            validate_sft_record(record)
            records.append(record)
    return records


def _realized_candidate(run: Mapping[str, Any]) -> dict[str, Any] | None:
    steps = list(run.get("steps") or ())
    first = steps[0]
    selected_action_id = str(
        first.get("decision", {}).get("selected_action", {}).get("action_id") or ""
    )
    predicted = next(
        (
            row
            for row in first.get("decision", {}).get("imagined_paths") or ()
            if str(row.get("transitions", [{}])[0].get("action", {}).get("action_id"))
            == selected_action_id
        ),
        None,
    )
    if predicted is None:
        return None
    coverage: set[int] = set()
    labels = {
        str(row.get("observation_id")): row
        for row in run.get("realized_labels_evaluator_only") or ()
    }
    for step in steps:
        coverage.update(
            int(value)
            for value in labels.get(str(step.get("observation_id")), {}).get(
                "newly_covered_clue_indices", ()
            )
        )
    return {
        "question": str(
            first.get("pool_before", {})
            .get("trajectories", [{}])[0]
            .get("belief", {})
            .get("question", "")
        ),
        "pool_before": first.get("pool_before") or {},
        "source_arm": str(run.get("arm") or ""),
        "predicted_joint_tree": predicted,
        "executed_action_ids": [
            str(step.get("decision", {}).get("selected_action", {}).get("action_id"))
            for step in steps
        ],
        "_coverage": sorted(coverage),
    }


def _public_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_arm": row["source_arm"],
        "predicted_joint_tree": row["predicted_joint_tree"],
        "executed_action_ids": row["executed_action_ids"],
    }


def _transition_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, Counter[str]] = defaultdict(Counter)
    videos: dict[str, set[str]] = defaultdict(set)
    delayed_units: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    delayed_targets: dict[str, set[str]] = defaultdict(set)
    delayed_action_kinds: dict[str, set[str]] = defaultdict(set)
    for row in records:
        split = str(row["split"])
        video_id = str(row["video_id"])
        videos[split].add(video_id)
        outcome = str(
            row["target"]["categorical_audit"]["observation_outcome"]["value"]
        )
        by_split[split]["record_count"] += 1
        by_split[split][f"outcome:{outcome}"] += 1
        delayed_positive = bool(row["data_slices"]["delayed_positive"])
        by_split[split]["delayed_positive"] += int(delayed_positive)
        if delayed_positive:
            transition_input = row["input"]
            action = transition_input["action"]
            target_id = str(action.get("target_id") or "")
            delayed_units[split].add(
                (video_id, str(transition_input.get("question") or ""), target_id)
            )
            delayed_targets[split].add(target_id)
            delayed_action_kinds[split].add(str(action.get("kind") or "unknown"))
        by_split[split]["semantic_hard_negative"] += int(
            row["data_slices"]["semantic_neighbor_hard_negative"]
        )
        embedding_status = str(
            row["input"]["target_node_safe_view"]["embedding_ref"].get("status")
            or "unknown"
        )
        by_split[split][f"embedding:{embedding_status}"] += 1
    return {
        split: {
            **dict(sorted(counts.items())),
            "video_count": len(videos[split]),
            "delayed_positive_unit_count": len(delayed_units[split]),
            "delayed_positive_target_count": len(delayed_targets[split]),
            "delayed_positive_action_kind_count": len(delayed_action_kinds[split]),
        }
        for split, counts in sorted(by_split.items())
    }


def _split_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    videos: dict[str, set[str]] = defaultdict(set)
    for row in records:
        videos[str(row["split"])].add(str(row["video_id"]))
    overlaps = {
        f"{left}:{right}": sorted(videos[left] & videos[right])
        for left, right in combinations(sorted(videos), 2)
        if videos[left] & videos[right]
    }
    return {
        "video_ids_by_split": {
            split: sorted(values) for split, values in sorted(videos.items())
        },
        "cross_split_video_overlaps": overlaps,
        "video_disjoint": not overlaps,
        "hidden_supervision_in_input": False,
    }


def _transition_gate(
    counts: Mapping[str, Mapping[str, int]],
    split_audit: Mapping[str, Any],
    **minimums: int,
) -> dict[str, Any]:
    train = counts.get("train", {})
    validation = counts.get("validation", {})
    checks = {
        "video_disjoint": bool(split_audit.get("video_disjoint")),
        "train_video_count": train.get("video_count", 0)
        >= minimums["minimum_train_videos"],
        "train_record_count": train.get("record_count", 0)
        >= minimums["minimum_train_records"],
        "train_has_support_and_inconclusive": train.get("outcome:support", 0) > 0
        and train.get("outcome:inconclusive", 0) > 0,
        "train_delayed_positive_count": train.get("delayed_positive", 0)
        >= minimums["minimum_delayed_positives"],
        "train_delayed_positive_unit_count": train.get(
            "delayed_positive_unit_count", 0
        )
        >= minimums["minimum_delayed_positive_units"],
        "train_semantic_hard_negative_count": train.get(
            "semantic_hard_negative", 0
        )
        >= minimums["minimum_semantic_hard_negatives"],
        "validation_video_count": validation.get("video_count", 0)
        >= minimums["minimum_validation_videos"],
        "validation_record_count": validation.get("record_count", 0)
        >= minimums["minimum_validation_records"],
        "validation_has_support_and_inconclusive": validation.get(
            "outcome:support", 0
        )
        > 0
        and validation.get("outcome:inconclusive", 0) > 0,
        "validation_delayed_positive_count": validation.get(
            "delayed_positive", 0
        )
        >= minimums["minimum_validation_delayed_positives"],
        "validation_delayed_positive_unit_count": validation.get(
            "delayed_positive_unit_count", 0
        )
        >= minimums["minimum_validation_delayed_positive_units"],
        "validation_semantic_hard_negative_count": validation.get(
            "semantic_hard_negative", 0
        )
        >= minimums["minimum_validation_semantic_hard_negatives"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimums": minimums,
        "blockers": [name for name, passed in checks.items() if not passed],
        "training_performed": False,
    }


def _planner_gate(
    records: Sequence[Mapping[str, Any]],
    transition_gate: Mapping[str, Any],
    *,
    transition_calibration_passed: bool,
) -> dict[str, Any]:
    train_labels = Counter(
        str(row["target"]["preference"]["value"])
        for row in records
        if row["split"] == "train"
    )
    validation_count = sum(row["split"] == "validation" for row in records)
    checks = {
        "transition_gate_passed": bool(transition_gate.get("passed")),
        "transition_calibration_passed": transition_calibration_passed,
        "train_preference_pairs_present": sum(train_labels.values()) > 0,
        "train_preference_label_diversity": len(train_labels) >= 2,
        "validation_preference_pairs_present": validation_count > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "train_label_counts": dict(sorted(train_labels.items())),
        "validation_record_count": validation_count,
        "blockers": [name for name, passed in checks.items() if not passed],
        "training_performed": False,
    }


def _safe_action(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: action.get(key)
        for key in ("action_id", "kind", "source_id", "target_id", "edge_id", "relation")
    }


def _safe_target_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    metadata = observation.get("metadata") or {}
    embedding = observation.get("embedding_ref") or {}
    if embedding:
        embedding_slot = {
            "status": "available",
            **{
                key: embedding.get(key)
                for key in (
                    "model",
                    "dimension",
                    "dtype",
                    "normalized",
                    "row_index",
                    "checksum",
                )
                if embedding.get(key) is not None
            },
        }
    else:
        consolidation = metadata.get("consolidation") or {}
        embedding_slot = {
            "status": str(
                consolidation.get("embedding_status") or "not_materialized"
            ),
            "requested_model": EMBEDDING_MODEL,
            "source_node_ids": list(
                consolidation.get("lineage")
                or observation.get("provenance", {}).get("source_node_ids")
                or ()
            ),
        }
    return {
        "node_id": observation.get("node_id"),
        "node_type": observation.get("node_type"),
        "time_span": observation.get("time_span"),
        "semantic_key": metadata.get("predicate") or observation.get("text"),
        "structural_tags": [
            value
            for value in (metadata.get("modality"), metadata.get("action_kind"))
            if value
        ],
        "embedding_ref": embedding_slot,
        "evidence_value_visible": False,
    }


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()[:20]


def _without_numeric_scalars(value: Any) -> Any:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return None
    if isinstance(value, Mapping):
        return {
            str(key): cleaned
            for key, child in value.items()
            if (cleaned := _without_numeric_scalars(child)) not in (None, [], {})
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            cleaned
            for child in value
            if (cleaned := _without_numeric_scalars(child)) is not None
        ]
    return str(value)


def _contains_numeric_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_numeric_scalar(child) for child in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(_contains_numeric_scalar(child) for child in value)
    return False


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", default=[], type=Path)
    parser.add_argument(
        "--artifact-dir",
        action="append",
        default=[],
        type=Path,
        help="Read all case_*.json files directly below this legacy cohort directory.",
    )
    parser.add_argument(
        "--compile-gate",
        action="append",
        default=[],
        type=Path,
        help="Exact frozen graph gate used to hydrate legacy executed observation IDs.",
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--calibration-report",
        type=Path,
        help="Passing held-out transition report required to unlock Planner records.",
    )
    args = parser.parse_args(argv)
    artifact_paths = list(args.artifact)
    for directory in args.artifact_dir:
        artifact_paths.extend(sorted(directory.glob("case_*.json")))
    if not artifact_paths:
        parser.error("at least one --artifact or --artifact-dir is required")
    dataset = _read(args.dataset)
    artifacts = [_read(path) for path in artifact_paths]
    if args.compile_gate:
        artifacts = hydrate_legacy_executed_observations(
            artifacts,
            dataset,
            [_read(path) for path in args.compile_gate],
        )
    result = build_grounded_runtime_data(
        artifacts,
        dataset,
        _read(args.hidden_key),
        transition_calibration=(
            _read(args.calibration_report) if args.calibration_report else None
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.output_dir / "transition_records.jsonl", result["transition_records"])
    _write_jsonl(
        args.output_dir / "planner_preference_records.jsonl",
        result["planner_preference_records"],
    )
    combined = [*result["transition_records"], *result["planner_preference_records"]]
    _write_jsonl(args.output_dir / "records.jsonl", combined)
    report = {key: value for key, value in result.items() if not key.endswith("records")}
    (args.output_dir / "readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["transition_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
