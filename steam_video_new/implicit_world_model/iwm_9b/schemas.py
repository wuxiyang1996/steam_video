"""Strict, masked SFT contracts for IWM transition and Planner preference."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


SCHEMA_VERSION = "steam-iwm-9b-sft/v0.1"
TRANSITION_TASK = "iwm_transition"
PLANNER_TASK = "planner_preference"
TASKS = {TRANSITION_TASK, PLANNER_TASK}
PREFERENCE_LABELS = {
    "prefer_left",
    "prefer_right",
    "tie",
    "incomparable",
}
SPLITS = {"train", "validation", "test"}


def supervised_field(
    value: Any,
    *,
    supervised: bool,
    provenance: str,
) -> dict[str, Any]:
    """Construct one auditable target field.

    Missing supervision is represented as ``value=None`` and never converted to
    a negative label.
    """

    if not provenance:
        raise ValueError("supervision provenance must not be empty")
    if not supervised and value is not None:
        raise ValueError("an unsupervised field must have value=None")
    return {
        "value": value,
        "supervised": supervised,
        "provenance": provenance,
    }


def validate_sft_record(record: Mapping[str, Any]) -> None:
    """Validate one task record without importing a training framework."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported 9B SFT schema")
    task = str(record.get("task") or "")
    if task not in TASKS:
        raise ValueError(f"unsupported SFT task: {task}")
    if not record.get("record_id") or not record.get("case_id"):
        raise ValueError("SFT record requires record_id and case_id")
    if not record.get("video_id"):
        raise ValueError("SFT record requires video_id for split-safe training")
    split = str(record.get("split") or "")
    if split not in SPLITS:
        raise ValueError(f"unsupported split: {split}")
    if not isinstance(record.get("input"), Mapping):
        raise ValueError("SFT input must be an object")
    target = record.get("target")
    if not isinstance(target, Mapping):
        raise ValueError("SFT target must be an object")
    if record.get("numeric_reward_present") is not False:
        raise ValueError("numeric reward is forbidden")
    if record.get("hidden_supervision_in_input") is not False:
        raise ValueError("hidden supervision leaked into model input")

    _validate_masked_tree(target)
    if task == PLANNER_TASK:
        label = target.get("preference")
        if not isinstance(label, Mapping) or label.get("supervised") is not True:
            raise ValueError("Planner preference must be supervised")
        if str(label.get("value") or "") not in PREFERENCE_LABELS:
            raise ValueError("invalid Planner preference label")
        planner_input = record["input"]
        if not isinstance(planner_input.get("left"), Mapping) or not isinstance(
            planner_input.get("right"), Mapping
        ):
            raise ValueError("Planner record requires left and right candidates")


def _validate_masked_tree(value: Any, path: str = "target") -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be an object")
    if "supervised" in value or "provenance" in value:
        if set(value) != {"value", "supervised", "provenance"}:
            raise ValueError(f"{path} is not a strict supervised field")
        if not isinstance(value["supervised"], bool):
            raise ValueError(f"{path}.supervised must be boolean")
        if not value["provenance"]:
            raise ValueError(f"{path}.provenance must not be empty")
        if not value["supervised"] and value["value"] is not None:
            raise ValueError(f"{path} unsupervised value must be null")
        return
    if not value:
        raise ValueError(f"{path} must contain supervised fields")
    for key, child in value.items():
        _validate_masked_tree(child, f"{path}.{key}")


def supervised_target(target: Mapping[str, Any]) -> dict[str, Any]:
    """Drop unsupervised fields before rendering the assistant target."""

    result: dict[str, Any] = {}
    for key, value in target.items():
        if not isinstance(value, Mapping):
            raise ValueError("target tree must contain only objects")
        if "supervised" in value:
            if value["supervised"]:
                result[key] = value["value"]
            continue
        child = supervised_target(value)
        if child:
            result[key] = child
    return result
