"""Adapters from existing grounded artifacts to masked 9B SFT records."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
import hashlib
from typing import Any

from .schemas import (
    PLANNER_TASK,
    SCHEMA_VERSION,
    TRANSITION_TASK,
    supervised_field,
    validate_sft_record,
)


EXECUTED_OBSERVATION = "executed_grounded_observation"
DATASET_NAVIGATION = "dataset_gt_navigation_label_after_graph_freeze"
NOT_AVAILABLE = "not_available_in_source_artifact"


def adapt_grounded_transition_corpus(
    corpus: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Convert the positive-only CG-Bench corpus without inventing labels."""

    records: list[dict[str, Any]] = []
    for source in corpus.get("records") or []:
        context = source.get("training_context") or {}
        realized = source.get("real_transition_target") or {}
        descriptor = realized.get("observation_descriptor")
        delta = realized.get("categorical_belief_delta") or {}
        split = str(source.get("split") or "")
        progress = _progress_label(delta.get("evidence_progress"))
        coverage = delta.get("required_clue_coverage_after")
        answerability = delta.get("answerability_after")
        answerability_known = answerability in {"ready", "not_ready", "abstain"}
        record = {
            "schema_version": SCHEMA_VERSION,
            "task": TRANSITION_TASK,
            "record_id": f"iwm9b:{source['record_id']}",
            "case_id": source["case_id"],
            "video_id": source["video_id"],
            "split": split,
            "input": {
                "question": context.get("question"),
                "hypothesis": None,
                "persistent_path_state": {
                    "checkpoint": context.get("checkpoint") or {},
                    "hypothesis_conditioned": False,
                },
                "action": context.get("action") or {},
                "target_node_safe_view": None,
                "local_temporal_context": [],
                "local_correlation_context": [],
                "imagined_prefix": [],
            },
            "target": {
                "observation_patch": {
                    "descriptor": supervised_field(
                        descriptor,
                        supervised=isinstance(descriptor, Mapping),
                        provenance=EXECUTED_OBSERVATION,
                    ),
                    "observed_modalities": supervised_field(
                        realized.get("observed_modalities"),
                        supervised=isinstance(
                            realized.get("observed_modalities"), list
                        ),
                        provenance=EXECUTED_OBSERVATION,
                    ),
                    "entity_bindings": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "temporal_binding": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "evidence_role": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                },
                "belief_patch": {
                    "supported_claims": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "contradicted_claims": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "newly_bound_variables": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "opened_dependencies": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "resolved_dependencies": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                },
                "categorical_audit": {
                    "progress": supervised_field(
                        progress,
                        supervised=progress is not None,
                        provenance=DATASET_NAVIGATION,
                    ),
                    "required_clue_coverage_after": supervised_field(
                        coverage,
                        supervised=coverage in {"partial", "complete"},
                        provenance=DATASET_NAVIGATION,
                    ),
                    "answerability_after": supervised_field(
                        answerability if answerability_known else None,
                        supervised=answerability_known,
                        provenance=(
                            DATASET_NAVIGATION if answerability_known else NOT_AVAILABLE
                        ),
                    ),
                    "observation_outcome": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "contradiction_change": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                    "frontier_change": supervised_field(
                        None, supervised=False, provenance=NOT_AVAILABLE
                    ),
                },
            },
            "label_source": "executed_transition_plus_dataset_clue_coverage",
            "training_eligible": split == "train",
            "eligibility_reasons": (
                ["train_split_scoped_positive_transition"]
                if split == "train"
                else ["held_out_split"]
            ),
            "target_is_real_not_imagined": bool(
                source.get("target_is_real_not_imagined")
            ),
            "hidden_supervision_in_input": False,
            "numeric_reward_present": False,
        }
        validate_sft_record(record)
        records.append(record)
    return records


def adapt_local_choice_packet(
    public: Mapping[str, Any],
    hidden: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Join blinded candidates with offline labels while preserving split gates."""

    hidden_by_id = {
        str(row["record_id"]): row for row in hidden.get("local_choice_records") or []
    }
    records: list[dict[str, Any]] = []
    for public_row in public.get("local_choice_records") or []:
        record_id = str(public_row["record_id"])
        hidden_row = hidden_by_id.get(record_id)
        if hidden_row is None:
            raise ValueError(f"missing hidden Planner label row: {record_id}")
        candidates = {
            str(row["candidate"]): row for row in public_row.get("candidates") or []
        }
        video_id = _video_id(hidden_row)
        split = str(public_row.get("split") or "")
        for index, pair in enumerate(hidden_row.get("pairwise_labels") or []):
            left_id = str(pair.get("left") or "")
            right_id = str(pair.get("right") or "")
            if left_id not in candidates or right_id not in candidates:
                raise ValueError(f"unknown candidate alias in {record_id}")
            record = {
                "schema_version": SCHEMA_VERSION,
                "task": PLANNER_TASK,
                "record_id": f"iwm9b:{record_id}:pair:{index}",
                "case_id": public_row["case_id"],
                "video_id": video_id,
                "split": split,
                "input": {
                    "question": public_row.get("question"),
                    "persistent_trajectory_pool": {
                        "belief": public_row.get("belief") or {},
                        "hypothesis_coverage": "single_shared_belief_diagnostic",
                    },
                    "left": candidates[left_id],
                    "right": candidates[right_id],
                    "comparison_scope": "complete_available_first_hop_pair",
                },
                "target": {
                    "preference": supervised_field(
                        pair.get("label"),
                        supervised=True,
                        provenance=str(
                            hidden_row.get("label_source") or DATASET_NAVIGATION
                        ),
                    )
                },
                "label_source": hidden_row.get("label_source"),
                "training_eligible": split == "train",
                "eligibility_reasons": (
                    ["train_split_grounded_preference"]
                    if split == "train"
                    else ["held_out_split_diagnostic_only"]
                ),
                "target_is_real_not_imagined": True,
                "hidden_supervision_in_input": False,
                "numeric_reward_present": False,
            }
            validate_sft_record(record)
            records.append(record)
    return records


def build_readiness_report(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    for row in rows:
        validate_sft_record(row)
    tasks = Counter(str(row["task"]) for row in rows)
    splits = Counter(str(row["split"]) for row in rows)
    eligible = [row for row in rows if row.get("training_eligible") is True]
    eligible_tasks = Counter(str(row["task"]) for row in eligible)
    preference_labels = Counter(
        str(row["target"]["preference"]["value"])
        for row in eligible
        if row["task"] == PLANNER_TASK
    )
    all_preference_labels = Counter(
        str(row["target"]["preference"]["value"])
        for row in rows
        if row["task"] == PLANNER_TASK
    )
    supervised_fields: Counter[str] = Counter()
    unsupervised_fields: Counter[str] = Counter()
    supervised_scalar_values: dict[str, Counter[str]] = {}
    for row in rows:
        for path, field in _target_fields(row["target"]):
            destination = (
                supervised_fields if field["supervised"] else unsupervised_fields
            )
            destination[path] += 1
            value = field["value"]
            if field["supervised"] and isinstance(value, (str, int, float, bool)):
                supervised_scalar_values.setdefault(path, Counter())[str(value)] += 1
    transition_train = eligible_tasks.get(TRANSITION_TASK, 0)
    planner_train = eligible_tasks.get(PLANNER_TASK, 0)
    transition_ready = transition_train > 0
    planner_ready = planner_train > 0 and len(preference_labels) >= 2
    blockers: list[str] = []
    if not transition_ready:
        blockers.append("no train-split grounded transition records")
    if not planner_train:
        blockers.append("no train-split grounded Planner preference records")
    elif len(preference_labels) < 2:
        blockers.append("Planner train split has fewer than two preference classes")
    return {
        "schema_version": "steam-iwm-9b-readiness/v0.1",
        "record_count": len(rows),
        "task_counts": dict(sorted(tasks.items())),
        "split_counts": dict(sorted(splits.items())),
        "training_eligible_count": len(eligible),
        "eligible_task_counts": dict(sorted(eligible_tasks.items())),
        "eligible_preference_label_counts": dict(sorted(preference_labels.items())),
        "all_preference_label_counts": dict(sorted(all_preference_labels.items())),
        "supervised_field_counts": dict(sorted(supervised_fields.items())),
        "unsupervised_field_counts": dict(sorted(unsupervised_fields.items())),
        "supervised_scalar_value_counts": {
            path: dict(sorted(counts.items()))
            for path, counts in sorted(supervised_scalar_values.items())
        },
        "transition_sft_pipeline_ready": transition_ready,
        "planner_sft_pipeline_ready": planner_ready,
        "joint_multitask_sft_ready": transition_ready and planner_ready,
        "blockers": blockers,
        "numeric_reward_present": False,
        "training_performed": False,
    }


def _progress_label(value: Any) -> str | None:
    mapping = {
        "advances_required_clue_coverage": "advanced",
        "advances": "advanced",
        "unchanged": "unchanged",
        "regresses": "regressed",
    }
    return mapping.get(str(value or ""))


def _video_id(hidden_row: Mapping[str, Any]) -> str:
    grounding = hidden_row.get("candidate_grounding") or {}
    for value in grounding.values():
        target_id = str((value or {}).get("target_id") or "")
        parts = target_id.split(":")
        if len(parts) >= 3 and parts[1]:
            return parts[1]
        graph_id = str(
            ((value or {}).get("provenance") or {}).get("source_graph_id") or ""
        )
        if ":" in graph_id:
            return graph_id.split(":", 1)[1]
    digest = hashlib.sha256(str(hidden_row.get("case_id")).encode()).hexdigest()
    return f"unknown-video:{digest[:12]}"


def _target_fields(
    target: Mapping[str, Any],
    prefix: str = "",
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for key, value in target.items():
        if not isinstance(value, Mapping):
            raise ValueError("target tree must contain objects")
        path = f"{prefix}.{key}" if prefix else str(key)
        if "supervised" in value:
            yield path, value
        else:
            yield from _target_fields(value, path)
