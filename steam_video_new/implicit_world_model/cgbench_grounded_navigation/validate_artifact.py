"""Validate a completed GT-only grounded-navigation artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .builder import EMBEDDING_MODEL, _write_json, validate_cgbench_navigation_dataset


def validate_completed_artifact(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    manifest: dict[str, Any],
    *,
    matrix_path: Path,
) -> dict[str, Any]:
    errors = list(validate_cgbench_navigation_dataset(dataset, hidden))
    transitions = [
        transition
        for case in dataset.get("cases") or []
        for transition in case.get("executed_transitions") or []
    ]
    expected = len(transitions)
    grounded = [
        row for row in transitions
        if (row.get("real_observation") or {}).get("descriptor_status")
        == "grounded_qwen_vl_read"
    ]
    failed = [
        row for row in transitions
        if (row.get("real_observation") or {}).get("descriptor_status") == "grounding_failed"
    ]
    available = [
        row for row in transitions
        if (((row.get("target") or {}).get("observation_descriptor") or {})
            .get("embedding_ref") or {}).get("status") == "available"
    ]
    if len(grounded) != expected:
        errors.append(f"grounded transition count is {len(grounded)}, expected {expected}")
    if failed:
        errors.append(f"grounding failure count is {len(failed)}")
    if len(available) != expected:
        errors.append(f"available embedding count is {len(available)}, expected {expected}")
    if manifest.get("model") != EMBEDDING_MODEL:
        errors.append("embedding manifest model mismatch")
    if manifest.get("row_count") != expected or manifest.get("dimension") != 2048:
        errors.append("embedding manifest shape mismatch")
    if not matrix_path.is_file():
        errors.append("embedding matrix is missing")
        matrix_checksum = None
        matrix_shape = None
    else:
        matrix_checksum = _file_checksum(matrix_path)
        if matrix_checksum != manifest.get("checksum"):
            errors.append("embedding matrix checksum mismatch")
        try:
            import numpy as np

            matrix_shape = list(np.load(matrix_path, mmap_mode="r").shape)
            if matrix_shape != [expected, 2048]:
                errors.append(f"embedding matrix shape is {matrix_shape}, expected [{expected}, 2048]")
        except (ImportError, OSError, ValueError) as exc:
            matrix_shape = None
            errors.append(f"embedding matrix cannot be inspected: {type(exc).__name__}")
    row_indices = sorted(
        int((((row["target"]["observation_descriptor"])["embedding_ref"])["row_index"]))
        for row in available
    )
    if row_indices != list(range(expected)):
        errors.append("embedding row indices are not a complete contiguous range")
    if dataset.get("training_ready") is not False or dataset.get("training_performed") is not False:
        errors.append("data-only artifact unexpectedly enables or records training")
    split_counts: dict[str, int] = {}
    for case in dataset.get("cases") or []:
        split = str(case.get("split"))
        split_counts[split] = split_counts.get(split, 0) + 1
    return {
        "schema_version": "steam-cgbench-grounded-artifact-validation/v0.1",
        "dataset_id": dataset.get("dataset_id"),
        "status": "pass" if not errors else "fail",
        "errors": errors,
        "case_count": len(dataset.get("cases") or []),
        "split_case_counts": dict(sorted(split_counts.items())),
        "transition_count": expected,
        "grounded_transition_count": len(grounded),
        "failed_transition_count": len(failed),
        "available_embedding_count": len(available),
        "embedding_model": manifest.get("model"),
        "embedding_shape": matrix_shape,
        "embedding_checksum": matrix_checksum,
        "human_review_required": False,
        "training_performed": False,
    }


def _file_checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--hidden-input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = validate_completed_artifact(dataset, hidden, manifest, matrix_path=args.matrix)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
