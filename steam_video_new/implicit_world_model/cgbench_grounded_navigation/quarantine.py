"""Quarantine asset/grounding failures without consulting answer or clue labels."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from .builder import EMBEDDING_MODEL, _checksum, _write_json


SCHEMA = "steam-cgbench-grounding-quarantine/v0.1"


def quarantine_failed_grounding_cases(
    dataset: dict[str, Any],
    hidden: dict[str, Any],
    manifest: dict[str, Any],
    matrix: Any,
    *,
    matrix_output: Path,
    matrix_uri: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Remove whole cases containing failed reads and rebuild embedding alignment."""

    import numpy as np

    failed_case_ids = {
        str(case["case_id"])
        for case in dataset.get("cases") or []
        if any(
            (transition.get("real_observation") or {}).get("descriptor_status")
            != "grounded_qwen_vl_read"
            for transition in case.get("executed_transitions") or []
        )
    }
    if not failed_case_ids:
        raise ValueError("quarantine requires at least one failed grounding case")
    result = deepcopy(dataset)
    result["cases"] = [
        case
        for case in result.get("cases") or []
        if str(case["case_id"]) not in failed_case_ids
    ]
    hidden_result = deepcopy(hidden)
    hidden_result["cases"] = [
        case
        for case in hidden_result.get("cases") or []
        if str(case["case_id"]) not in failed_case_ids
    ]

    old_row_by_transition = {
        str(row["transition_id"]): int(row["row_index"])
        for row in manifest.get("rows") or []
    }
    kept: list[tuple[dict[str, Any], str, int]] = []
    for case in result["cases"]:
        for transition in case.get("executed_transitions") or []:
            transition_id = str(transition["transition_id"])
            observation = transition.get("real_observation") or {}
            if observation.get("descriptor_status") != "grounded_qwen_vl_read":
                raise ValueError("a failed transition survived case quarantine")
            if transition_id not in old_row_by_transition:
                raise ValueError(f"embedding manifest lacks {transition_id}")
            kept.append(
                (transition, transition_id, old_row_by_transition[transition_id])
            )
    source = np.asarray(matrix, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 2048:
        raise ValueError("source embedding matrix must have shape [N, 2048]")
    if any(index < 0 or index >= source.shape[0] for _, _, index in kept):
        raise ValueError("embedding manifest row index is out of range")
    subset = source[[index for _, _, index in kept]]
    matrix_output.parent.mkdir(parents=True, exist_ok=True)
    np.save(matrix_output, subset)
    saved = (
        matrix_output
        if matrix_output.suffix == ".npy"
        else matrix_output.with_suffix(".npy")
    )
    checksum = hashlib.sha256(saved.read_bytes()).hexdigest()
    uri = matrix_uri or saved.name
    rows: list[dict[str, Any]] = []
    for new_index, (transition, transition_id, _) in enumerate(kept):
        transition["target"]["observation_descriptor"]["embedding_ref"] = {
            "model": EMBEDDING_MODEL,
            "status": "available",
            "dimension": 2048,
            "dtype": "float32",
            "normalized": True,
            "storage_uri": uri,
            "row_index": new_index,
            "checksum": checksum,
        }
        rows.append({"row_index": new_index, "transition_id": transition_id})
    result["annotation_status"] = "qwen_grounded_case_quarantine_complete"
    result["grounding_complete"] = True
    result["formal_eligible"] = False
    result["training_ready"] = False
    result["next_gate"] = (
        "freeze a video-disjoint L1/L1.5 graph set with retained multi-hop clues"
    )
    result["quarantine_contract"] = {
        "excluded_case_ids": sorted(failed_case_ids),
        "selection_signal": "grounding_or_asset_failure_only",
        "answer_or_clue_label_used_for_exclusion": False,
    }
    hidden_result["dataset_sha256"] = _checksum(result)
    new_manifest = {
        "schema_version": "steam-cgbench-embedding-manifest/v0.1",
        "model": EMBEDDING_MODEL,
        "matrix_uri": uri,
        "row_count": len(rows),
        "dimension": 2048,
        "dtype": "float32",
        "normalized": True,
        "checksum": checksum,
        "rows": rows,
    }
    report = {
        "schema_version": SCHEMA,
        "source_case_count": len(dataset.get("cases") or []),
        "retained_case_count": len(result["cases"]),
        "excluded_case_count": len(failed_case_ids),
        "excluded_case_ids": sorted(failed_case_ids),
        "retained_transition_count": len(rows),
        "embedding_shape": list(subset.shape),
        "embedding_checksum": checksum,
        "selection_signal": "grounding_or_asset_failure_only",
        "answer_or_clue_label_used_for_exclusion": False,
        "training_performed": False,
    }
    return result, hidden_result, new_manifest, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--hidden-key", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-output", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    parser.add_argument("--matrix-output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    import numpy as np

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_key.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    result, hidden_result, new_manifest, report = quarantine_failed_grounding_cases(
        dataset,
        hidden,
        manifest,
        np.load(args.matrix),
        matrix_output=args.matrix_output,
    )
    _write_json(args.output, result)
    _write_json(args.hidden_output, hidden_result)
    _write_json(args.manifest_output, new_manifest)
    _write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
