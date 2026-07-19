"""Validate serialized memory-graph artifacts against local JSON Schemas."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_overlay_artifact(payload: dict[str, Any]) -> list[str]:
    """Return stable, local-only validation errors for a serialized overlay."""

    try:
        import jsonschema
        from referencing import Registry, Resource
    except ImportError as exc:  # pragma: no cover - packaging/environment failure
        raise RuntimeError("jsonschema and referencing are required") from exc

    root = Path(__file__).resolve().parent
    overlay_schema = json.loads(
        (root / "causal_temporal_overlay.schema.json").read_text(encoding="utf-8")
    )
    memory_schema = json.loads(
        (root / "memory_graph.schema.json").read_text(encoding="utf-8")
    )
    registry = Registry().with_resources(
        [
            (overlay_schema["$id"], Resource.from_contents(overlay_schema)),
            (memory_schema["$id"], Resource.from_contents(memory_schema)),
        ]
    )
    validator = jsonschema.Draft202012Validator(
        overlay_schema,
        registry=registry,
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    messages: list[str] = []
    for error in errors:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        messages.append(f"{path}: {error.message}")
    return messages


def require_valid_overlay_artifact(payload: dict[str, Any]) -> None:
    errors = validate_overlay_artifact(payload)
    if errors:
        preview = "; ".join(errors[:5])
        raise ValueError(f"serialized overlay failed JSON Schema validation: {preview}")
