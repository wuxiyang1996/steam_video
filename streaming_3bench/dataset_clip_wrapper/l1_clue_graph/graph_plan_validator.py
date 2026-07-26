"""Validate and resolve constrained (multiple-choice) L1 skill plans."""

from __future__ import annotations

import re
from typing import Any

ALLOWED_MEMORY_EDGES = {
    "temporal_next",
    "entity_mention",
    "derived_from",
    "causal_hint",
    "same_entity",
    "state_of",
    "dialogue_speaker",
    "located_in",
}

_PLACEHOLDER_ID_PREFIXES = ("obs:", "mention:", "entity:", "edge:", "clip:s", "state:")


def executable_skill_ids(skill_executors: dict[str, Any]) -> list[str]:
    return sorted(skill_executors.keys())


def build_skill_plan_response_schema(allowed_skill_ids: list[str]) -> dict[str, Any]:
    """OpenRouter JSON schema: skill_id is multiple-choice enum, not free text."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "clue_memory_skill_plan",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "skill_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "step_id": {"type": "string"},
                                "skill_id": {"type": "string", "enum": allowed_skill_ids},
                                "args": {
                                    "type": "object",
                                    "additionalProperties": True,
                                },
                                "depends_on": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                            },
                            "required": ["step_id", "skill_id", "args", "depends_on"],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["skill_plan", "notes"],
            },
        },
    }


def validate_skill_plan(
    skill_plan: list[dict[str, Any]],
    *,
    allowed_skill_ids: set[str],
    clip_schemas: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Reject free-form skill ids and placeholder node references before execution."""
    errors: list[str] = []
    known_clip_ids = {schema.get("clip_id") for schema in clip_schemas or [] if schema.get("clip_id")}
    step_ids: set[str] = set()

    for index, step in enumerate(skill_plan):
        step_id = step.get("step_id")
        skill_id = step.get("skill_id")
        if not step_id:
            errors.append(f"step[{index}] missing step_id")
            continue
        if step_id in step_ids:
            errors.append(f"duplicate step_id: {step_id}")
        step_ids.add(step_id)

        if skill_id not in allowed_skill_ids:
            errors.append(f"{step_id}: skill_id {skill_id!r} not in executable allowlist")

        args = step.get("args") or {}
        if skill_id == "link_graph_relation":
            edge_type = args.get("edge_type")
            if edge_type not in ALLOWED_MEMORY_EDGES:
                errors.append(f"{step_id}: edge_type {edge_type!r} not allowed")

        if skill_id == "assign_provenance_trust" and not isinstance(args.get("trust_policy"), dict):
            errors.append(f"{step_id}: trust_policy must be an object, not free-form text")

        errors.extend(_validate_arg_values(step_id, args, known_clip_ids=known_clip_ids))

    return errors


def _validate_arg_values(step_id: str, value: Any, *, known_clip_ids: set[str], path: str = "args") -> list[str]:
    errors: list[str] = []
    if isinstance(value, str):
        if value.startswith("$step.") or value.startswith("$bindings."):
            return errors
        if any(value.startswith(prefix) for prefix in _PLACEHOLDER_ID_PREFIXES):
            errors.append(f"{step_id}: placeholder id {value!r} at {path}; use $step.<step_id>.<field>")
        if path.endswith("_ref") or path.endswith("_refs") or "node" in path or path.endswith("ref"):
            if value.startswith("evidence.") or value.startswith("clip:") or value in known_clip_ids:
                return errors
            if re.match(r"^[a-z_]+:s\d+$", value):
                errors.append(f"{step_id}: invented id {value!r} at {path}; use $step references")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            errors.extend(_validate_arg_values(step_id, item, known_clip_ids=known_clip_ids, path=f"{path}[{i}]"))
    elif isinstance(value, dict):
        for key, item in value.items():
            errors.extend(_validate_arg_values(step_id, item, known_clip_ids=known_clip_ids, path=f"{path}.{key}"))
    return errors


def resolve_plan_value(value: Any, bindings: dict[str, Any], step_outputs: dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$bindings."):
        return bindings.get(value.split(".", 1)[1])
    if isinstance(value, str) and value.startswith("$step."):
        resolved = resolve_step_reference(value[len("$step.") :], step_outputs)
        return _coerce_node_ref(resolved)
    if isinstance(value, list):
        return [resolve_plan_value(item, bindings, step_outputs) for item in value]
    if isinstance(value, dict):
        return {key: resolve_plan_value(item, bindings, step_outputs) for key, item in value.items()}
    return value


def _coerce_node_ref(value: Any) -> Any:
    """If value is a node/edge dict, extract the id string for use as a ref arg."""
    if isinstance(value, dict):
        if "node_id" in value:
            return value["node_id"]
        if "edge_id" in value:
            return value["edge_id"]
    return value


def resolve_step_reference(path: str, step_outputs: dict[str, Any]) -> Any:
    normalized = re.sub(r"\[(\d+)\]", r".\1", path.strip())
    parts = [part for part in normalized.split(".") if part]
    if not parts:
        return None
    current: Any = step_outputs.get(parts[0])
    for part in parts[1:]:
        if current is None:
            return None
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            elif part == "claim" and ("claim_text" in current or "text" in current):
                current = current
            elif part == "support_evidence":
                current = current.get("support_refs") or current.get("evidence_refs") or current.get("supported_by_refs")
            elif part.isdigit() and isinstance(current.get("evidence_refs"), list):
                current = current["evidence_refs"][int(part)]
            else:
                return None
        elif isinstance(current, list) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current
