"""Build the v4 observation-first IWM corpus from grounded executed reads.

The v3 corpus expanded one executed observation across every answer hypothesis
while assigning the same dataset-clue-overlap label to every expansion.  This
module restores the correct supervision boundary:

* an observation transition is hypothesis independent and appears once;
* clue acquisition remains evaluator-only metadata, never semantic support;
* hypothesis effects remain locked unless independently supervised;
* acquired source/path evidence values and typed L1/L1.5 context are visible;
* embedding sidecars are referenced for a future projector but are not claimed
  to be consumed by the text bridge.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schemas import (
    HYPOTHESIS_EFFECT_TASK,
    OBSERVATION_TASK,
    SCHEMA_VERSION,
    supervised_field,
    validate_sft_record,
)


V4_SCHEMA = "steam-grounded-runtime-iwm-data/v0.4"
EXECUTED_OBSERVATION = "executed_question_independent_l1_observation"
CLUE_EVALUATOR = "dataset_gt_clue_overlap_after_graph_freeze_evaluator_only"
NOT_ESTABLISHED = "not_independently_supervised"


def build_v4_records(
    legacy_records: Sequence[Mapping[str, Any]],
    *,
    minimum_train_units: int = 100,
    minimum_validation_units: int = 10,
    minimum_train_videos: int = 20,
    minimum_validation_videos: int = 5,
) -> dict[str, Any]:
    """Convert grounded v3 transitions without promoting hidden GT to input."""

    transition_rows = [
        row
        for row in legacy_records
        if row.get("task") == "iwm_transition"
        and row.get("split") in {"train", "validation", "test"}
    ]
    descriptor_index = _descriptor_index(transition_rows)
    grouped = _executed_read_groups(transition_rows)
    observation_records: list[dict[str, Any]] = []
    effect_candidates: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()

    for _, rows in sorted(grouped.items()):
        canonical = rows[0]
        model_input, input_audit = _observation_input(canonical, descriptor_index)
        exclusion_reasons = [
            name
            for name, passed in (
                ("target_semantic_descriptor_missing", input_audit["target_semantic_descriptor_present"]),
                ("source_evidence_descriptor_missing", input_audit["source_evidence_descriptor_present"]),
            )
            if not passed
        ]
        excluded.update(exclusion_reasons)
        record_id = f"runtime-observation-v4:{_digest(_unit_identity(canonical))}"
        target_descriptor = _compact_descriptor(
            _field_value(canonical, "observation_patch", "descriptor") or {}
        )
        clue_acquired = (
            "new_clue"
            if _field_value(canonical, "categorical_audit", "observation_outcome")
            == "support"
            else "no_new_clue"
        )
        observation_record = {
            "schema_version": SCHEMA_VERSION,
            "task": OBSERVATION_TASK,
            "record_id": record_id,
            "case_id": canonical["case_id"],
            "video_id": canonical["video_id"],
            "split": canonical["split"],
            "input": model_input,
            "target": {
                "observation_descriptor": supervised_field(
                    target_descriptor,
                    supervised=True,
                    provenance=EXECUTED_OBSERVATION,
                )
            },
            "evaluation_labels": {
                "clue_acquisition": {
                    "value": clue_acquired,
                    "provenance": CLUE_EVALUATOR,
                    "model_target": False,
                },
                "delayed_clue_acquisition": bool(
                    canonical.get("data_slices", {}).get("delayed_positive")
                ),
            },
            "representation_eligible": not exclusion_reasons,
            "training_eligible": not exclusion_reasons,
            "eligibility_reasons": (
                ["uniform_question_independent_text_bridge"]
                if not exclusion_reasons
                else exclusion_reasons
            ),
            "target_is_real_not_imagined": True,
            "hidden_supervision_in_input": False,
            "numeric_reward_present": False,
            "audit": {
                **input_audit,
                "legacy_hypothesis_expansion_count": len(rows),
                "legacy_record_ids": [str(row["record_id"]) for row in rows],
                "clue_label_is_evaluator_only": True,
                "embedding_values_consumed": False,
                "text_bridge_only": True,
            },
        }
        validate_sft_record(observation_record)
        observation_records.append(observation_record)

        for row in rows:
            effect = _hypothesis_effect_candidate(row, observation_record)
            validate_sft_record(effect)
            effect_candidates.append(effect)

    representation_gate = _representation_gate(
        observation_records,
        minimum_train_units=minimum_train_units,
        minimum_validation_units=minimum_validation_units,
        minimum_train_videos=minimum_train_videos,
        minimum_validation_videos=minimum_validation_videos,
    )
    observation_records = [
        {
            **row,
            "training_eligible": bool(
                row["training_eligible"]
                and row["split"] == "train"
                and representation_gate["passed"]
            ),
            "eligibility_reasons": (
                ["v4_observation_text_bridge_gate_passed"]
                if row["training_eligible"]
                and row["split"] == "train"
                and representation_gate["passed"]
                else (
                    ["held_out_video_split"]
                    if row["training_eligible"] and row["split"] != "train"
                    else row["eligibility_reasons"]
                )
            ),
        }
        for row in observation_records
    ]
    divergence = _hypothesis_divergence_audit(effect_candidates)
    hypothesis_effect_gate = {
        "passed": False,
        "checks": {
            "independent_effect_supervision_present": False,
            "hypothesis_conditioned_outcome_divergence_present": (
                divergence["groups_with_label_divergence"] > 0
            ),
        },
        "blockers": [
            "independent_effect_supervision_present",
            *(
                []
                if divergence["groups_with_label_divergence"] > 0
                else ["hypothesis_conditioned_outcome_divergence_present"]
            ),
        ],
        "planner_training_authorized": False,
    }
    return {
        "schema_version": V4_SCHEMA,
        "observation_records": observation_records,
        "hypothesis_effect_candidates": effect_candidates,
        "representation_gate": representation_gate,
        "hypothesis_effect_gate": hypothesis_effect_gate,
        "hypothesis_divergence_audit": divergence,
        "excluded_representation_counts": dict(sorted(excluded.items())),
        "contracts": {
            "one_observation_record_per_executed_read": True,
            "hypothesis_expansion_is_not_observation_supervision": True,
            "clue_acquisition_is_evaluator_only": True,
            "clue_acquisition_is_not_hypothesis_support": True,
            "source_evidence_values_visible_after_acquisition": True,
            "target_full_evidence_hidden_before_execution": True,
            "target_compact_question_independent_semantics_visible": True,
            "embedding_sidecar_referenced_for_future_projector": True,
            "embedding_values_consumed_by_text_bridge": False,
            "numeric_reward_present": False,
            "top_k_applied": False,
        },
        "observation_training_ready": bool(representation_gate["passed"]),
        "hypothesis_effect_training_ready": False,
        "planner_training_ready": False,
        "training_performed": False,
    }


def _executed_read_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[Any, ...], list[Mapping[str, Any]]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_unit_identity(row)].append(row)
    return groups


def _unit_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    payload = row["input"]
    action = payload.get("action") or {}
    state = payload.get("persistent_path_state") or {}
    return (
        str(row.get("split") or ""),
        str(row.get("case_id") or ""),
        str(row.get("video_id") or ""),
        tuple(str(value) for value in state.get("acquired_evidence_ids") or ()),
        str(action.get("action_id") or ""),
        str(action.get("source_id") or ""),
        str(action.get("target_id") or ""),
    )


def _descriptor_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        target_id = str(row.get("input", {}).get("action", {}).get("target_id") or "")
        descriptor = _compact_descriptor(
            _field_value(row, "observation_patch", "descriptor") or {}
        )
        if target_id and _descriptor_has_semantics(descriptor):
            result[(str(row.get("case_id") or ""), target_id)] = descriptor
    return result


def _observation_input(
    row: Mapping[str, Any],
    descriptor_index: Mapping[tuple[str, str], Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    legacy_input = row["input"]
    action = dict(legacy_input.get("action") or {})
    state = legacy_input.get("persistent_path_state") or {}
    case_id = str(row.get("case_id") or "")
    acquired_ids = [str(value) for value in state.get("acquired_evidence_ids") or ()]
    acquired = [
        {
            "node_id": node_id,
            "descriptor": dict(descriptor_index[(case_id, node_id)]),
        }
        for node_id in acquired_ids
        if (case_id, node_id) in descriptor_index
    ]
    source_id = str(action.get("source_id") or "")
    source_descriptor = descriptor_index.get((case_id, source_id))
    target_view = legacy_input.get("target_node_safe_view") or {}
    target_semantic_key = target_view.get("semantic_key")
    source_required = bool(source_id)
    edge_channel = (
        "correlation"
        if action.get("kind") == "follow_correlation"
        else "temporal"
        if "temporal" in str(action.get("kind") or "")
        else "entry"
    )
    payload = {
        "question": legacy_input.get("question"),
        "persistent_read_state": {
            "required_roles": list(state.get("required_roles") or ()),
            "missing_roles": list(state.get("missing_roles") or ()),
            "grounded_role_evidence": list(state.get("grounded_role_evidence") or ()),
            "contradictions": list(state.get("contradictions") or ()),
            "answerability": state.get("answerability"),
            "acquired_evidence": acquired,
        },
        "action": action,
        "source_evidence": (
            {"node_id": source_id, "descriptor": dict(source_descriptor)}
            if source_descriptor is not None
            else None
        ),
        "target_node_key": {
            "node_id": target_view.get("node_id"),
            "node_type": target_view.get("node_type"),
            "time_span": target_view.get("time_span"),
            "semantic_key": target_semantic_key,
            "structural_tags": list(target_view.get("structural_tags") or ()),
            "embedding_ref": target_view.get("embedding_ref"),
        },
        "edge_context": {
            "channel": edge_channel,
            "relation": action.get("relation"),
            "source_node_id": action.get("source_id"),
            "target_node_id": action.get("target_id"),
        },
        "representation_contract": {
            "question_independent_l1_l15": True,
            "acquired_evidence_values_visible": True,
            "unread_target_full_evidence_visible": False,
            "embedding_values_consumed": False,
            "text_bridge_only": True,
        },
    }
    return payload, {
        "target_semantic_descriptor_present": bool(target_semantic_key),
        "source_evidence_descriptor_present": (
            not source_required or source_descriptor is not None
        ),
        "all_acquired_evidence_descriptors_present": (
            len(acquired) == len(acquired_ids)
        ),
    }


def _hypothesis_effect_candidate(
    legacy: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    candidate = {
        "schema_version": SCHEMA_VERSION,
        "task": HYPOTHESIS_EFFECT_TASK,
        "record_id": f"runtime-hypothesis-effect-v4:{_digest(str(legacy['record_id']))}",
        "case_id": legacy["case_id"],
        "video_id": legacy["video_id"],
        "split": legacy["split"],
        "input": {
            "observation_record_id": observation["record_id"],
            "question": legacy["input"].get("question"),
            "hypothesis": legacy["input"].get("hypothesis"),
            "belief_before": legacy["input"].get("persistent_path_state"),
            "executed_observation": _field_value(
                legacy, "observation_patch", "descriptor"
            ),
            "action_context": observation["input"],
        },
        "target": {
            "hypothesis_effect": supervised_field(
                None, supervised=False, provenance=NOT_ESTABLISHED
            )
        },
        "evaluation_labels": {
            "clue_acquisition": observation["evaluation_labels"]["clue_acquisition"],
            "clue_acquisition_is_not_hypothesis_effect": True,
        },
        "training_eligible": False,
        "eligibility_reasons": ["independent_hypothesis_effect_not_established"],
        "target_is_real_not_imagined": True,
        "hidden_supervision_in_input": False,
        "numeric_reward_present": False,
    }
    return candidate


def _representation_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_train_units: int,
    minimum_validation_units: int,
    minimum_train_videos: int,
    minimum_validation_videos: int,
) -> dict[str, Any]:
    eligible = [row for row in rows if row["representation_eligible"]]
    by_split: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in eligible:
        by_split[str(row["split"])].append(row)
    counts = {
        split: {
            "independent_unit_count": len(values),
            "video_count": len({str(row["video_id"]) for row in values}),
            "new_clue_count": sum(
                row["evaluation_labels"]["clue_acquisition"]["value"] == "new_clue"
                for row in values
            ),
            "no_new_clue_count": sum(
                row["evaluation_labels"]["clue_acquisition"]["value"] == "no_new_clue"
                for row in values
            ),
        }
        for split, values in sorted(by_split.items())
    }
    train = counts.get("train", {})
    validation = counts.get("validation", {})
    checks = {
        "minimum_train_units": train.get("independent_unit_count", 0)
        >= minimum_train_units,
        "minimum_validation_units": validation.get("independent_unit_count", 0)
        >= minimum_validation_units,
        "minimum_train_videos": train.get("video_count", 0) >= minimum_train_videos,
        "minimum_validation_videos": validation.get("video_count", 0)
        >= minimum_validation_videos,
        "train_clue_controls_present": train.get("new_clue_count", 0) > 0
        and train.get("no_new_clue_count", 0) > 0,
        "validation_clue_controls_present": validation.get("new_clue_count", 0) > 0
        and validation.get("no_new_clue_count", 0) > 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "counts": counts,
        "blockers": [name for name, passed in checks.items() if not passed],
        "training_scope": "observation_transition_text_bridge_only",
        "embedding_projector_ready": False,
    }


def _hypothesis_divergence_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        key = str(row["input"]["observation_record_id"])
        groups[key].append(row)
    labeled = [
        values
        for values in groups.values()
        if any(
            row["target"]["hypothesis_effect"]["supervised"] is True
            for row in values
        )
    ]
    divergent = 0
    for values in labeled:
        labels = {
            str(row["target"]["hypothesis_effect"]["value"])
            for row in values
            if row["target"]["hypothesis_effect"]["supervised"] is True
        }
        divergent += int(len(labels) > 1)
    return {
        "executed_read_group_count": len(groups),
        "hypothesis_candidate_count": len(rows),
        "independently_labeled_group_count": len(labeled),
        "groups_with_label_divergence": divergent,
        "legacy_clue_label_promoted_to_effect": False,
    }


def _compact_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "event": str(value.get("predicate") or value.get("summary") or ""),
        "entities": [
            {
                key: entity[key]
                for key in ("role", "entity_type", "surface", "visual_signature")
                if entity.get(key) not in (None, "", [], {})
            }
            for entity in value.get("participants") or ()
            if isinstance(entity, Mapping)
        ],
        "states": value.get("states") or [],
        "state_change": value.get("state_change"),
    }


def _descriptor_has_semantics(value: Mapping[str, Any]) -> bool:
    return bool(
        value.get("event")
        or value.get("entities")
        or value.get("states")
        or value.get("state_change")
    )


def _field_value(row: Mapping[str, Any], branch: str, name: str) -> Any:
    return row.get("target", {}).get(branch, {}).get(name, {}).get("value")


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=list)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-records", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    result = build_v4_records(_read_jsonl(args.legacy_records))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        args.output_dir / "observation_records.jsonl", result["observation_records"]
    )
    _write_jsonl(
        args.output_dir / "hypothesis_effect_candidates.jsonl",
        result["hypothesis_effect_candidates"],
    )
    _write_jsonl(args.output_dir / "records.jsonl", result["observation_records"])
    report = {key: value for key, value in result.items() if not key.endswith("records") and not key.endswith("candidates")}
    (args.output_dir / "readiness.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if result["observation_training_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
