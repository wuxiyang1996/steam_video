"""Retry only the held-out audit for an already persisted overlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .openrouter_validation import GPTOSSGraphValidator
from .reverify_overlay import _node
from .types import MemoryGraph, RelationBelief, RelationStatus
from .validate_video_holmes import _load_annotation, _load_qa_rows


def _relation(payload: dict[str, Any]) -> RelationBelief:
    return RelationBelief(
        edge_id=str(payload["edge_id"]),
        src=str(payload["src"]),
        dst=str(payload["dst"]),
        relation_probabilities={
            str(key): float(value)
            for key, value in (payload.get("relation_probabilities") or {}).items()
        },
        status=RelationStatus(str(payload["status"])),
        direction_confidence=float(payload.get("direction_confidence") or 0.0),
        evidence_refs=[str(value) for value in payload.get("evidence_refs") or []],
        warrant=payload.get("warrant"),
        provenance=dict(payload.get("provenance") or {}),
        features={
            str(key): float(value)
            for key, value in (payload.get("features") or {}).items()
        },
    )


def event_graph_from_overlay(payload: dict[str, Any]) -> MemoryGraph:
    """Restore the auditor's atomic-event compatibility view from JSON."""
    return MemoryGraph(
        graph_id=str(payload["overlay_id"]),
        example_id=str(payload["example_id"]),
        video_id=str(payload["video_id"]),
        nodes=[_node(value) for value in payload.get("atomic_events") or []],
        relations=[_relation(value) for value in payload.get("relations") or []],
        metadata={
            **dict(payload.get("metadata") or {}),
            "compatibility_view": "atomic_events_only",
            "restored_from_persisted_overlay": True,
        },
    )


def update_run_summary(
    summary: dict[str, Any],
    *,
    video_id: str,
    audit: dict[str, Any],
) -> dict[str, Any]:
    """Replace one audit result and clear only its graph-audit error."""
    updated = dict(summary)
    samples = [dict(value) for value in updated.get("samples") or []]
    matched = False
    for sample in samples:
        if str(sample.get("video_id")) != video_id:
            continue
        matched = True
        sample["audit_summary"] = audit.get("summary")
        sample["computed_audit_summary"] = audit.get("computed_summary")
        sample["temporal_consistency"] = audit.get("temporal_consistency")
    if not matched:
        raise ValueError(f"summary does not contain video_id {video_id}")
    updated["samples"] = samples
    errors = [
        value
        for value in updated.get("errors") or []
        if not (
            str(value.get("video_id")) == video_id
            and value.get("stage") == "graph_audit"
        )
    ]
    updated["errors"] = errors
    updated["video_count_failed"] = len(errors)
    return updated


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    workspace = Path("/fs/gamma-projects/vlm-robot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    parser.add_argument("--audit-output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--dataset-root", default=str(workspace / "datasets"), type=Path)
    parser.add_argument("--video-skills-root", default=str(workspace / "Video_Skills"))
    parser.add_argument("--keys-py", default=str(workspace / "keys.py"))
    parser.add_argument("--timeout-s", default=240, type=int)
    parser.add_argument("--audit-max-tokens", default=8000, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overlay = json.loads(args.overlay.read_text(encoding="utf-8"))
    graph = event_graph_from_overlay(overlay)
    audit_output = args.audit_output or args.overlay.with_name("audit.json")
    if audit_output.exists():
        previous = json.loads(audit_output.read_text(encoding="utf-8"))
        failed_output = audit_output.with_name("audit.failed.json")
        if previous.get("error") and not failed_output.exists():
            _write_json(failed_output, previous)

    validator = GPTOSSGraphValidator(
        keys_py_path=args.keys_py,
        video_skills_root=args.video_skills_root,
        timeout_s=args.timeout_s,
        audit_max_tokens=args.audit_max_tokens,
    )
    audit = validator.audit_graph(
        graph,
        held_out_annotation=_load_annotation(args.dataset_root, graph.video_id),
        held_out_questions=_load_qa_rows(args.dataset_root, (graph.video_id,)),
    )
    _write_json(audit_output, audit)
    computed = dict(audit.get("computed_summary") or {})
    audit_complete = computed.get("audit_complete") is True

    if args.summary and audit_complete:
        summary = json.loads(args.summary.read_text(encoding="utf-8"))
        _write_json(
            args.summary,
            update_run_summary(summary, video_id=graph.video_id, audit=audit),
        )
    print(
        json.dumps(
            {
                "video_id": graph.video_id,
                "audit_output": str(audit_output),
                "computed_summary": computed,
            },
            indent=2,
        )
    )
    return 0 if audit_complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
