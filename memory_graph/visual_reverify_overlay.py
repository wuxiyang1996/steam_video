"""Apply raw-video VLM rereads to persisted relation candidates without embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .pipeline import _apply_visual_rereads, _verify_proposals
from .reverify_overlay import _node
from .types import RelationBelief, RelationStatus
from .visual_verifier import QwenVisualRereadProvider


def _relation(payload: dict[str, Any]) -> RelationBelief:
    return RelationBelief(
        edge_id=str(payload["edge_id"]),
        src=str(payload["src"]),
        dst=str(payload["dst"]),
        relation_probabilities={
            str(name): float(value)
            for name, value in (payload.get("relation_probabilities") or {}).items()
        },
        status=RelationStatus(str(payload["status"])),
        direction_confidence=float(payload.get("direction_confidence") or 0.0),
        evidence_refs=[str(value) for value in payload.get("evidence_refs") or []],
        warrant=payload.get("warrant"),
        provenance=dict(payload.get("provenance") or {}),
        features={
            str(name): float(value)
            for name, value in (payload.get("features") or {}).items()
        },
    )


def reverify_with_video(
    payload: dict[str, Any],
    *,
    provider: QwenVisualRereadProvider,
    minimum_probability: float = 0.5,
) -> dict[str, Any]:
    events = [_node(value) for value in payload.get("atomic_events") or []]
    l1_nodes = [_node(value) for value in payload.get("l1_observations") or []]
    build_report = dict(payload.get("build_report") or {})
    raw_candidates = build_report.get("candidate_relations") or []
    candidates = [
        _relation(value) for value in raw_candidates if isinstance(value, dict)
    ]
    video_path = str((payload.get("metadata") or {}).get("raw_video_path") or "")
    reread_candidates, reread_count = _apply_visual_rereads(
        candidates,
        event_nodes=events,
        video_path=video_path or None,
        provider=provider,
    )
    accepted, rejected, summary = _verify_proposals(
        reread_candidates,
        event_nodes=events,
        l1_nodes=l1_nodes,
        minimum_probability=minimum_probability,
        relation_thresholds={},
        enabled=True,
        require_visual_verification=True,
    )
    deterministic = [
        value
        for value in payload.get("relations") or []
        if isinstance(value, dict) and value.get("status") == "deterministic"
    ]
    result = dict(payload)
    result["relations"] = deterministic + [relation.to_dict() for relation in accepted]
    metadata = dict(result.get("metadata") or {})
    metadata.update(
        {
            "visual_verification_enabled": True,
            "visual_verification_required": True,
            "visual_reread_model": provider.model,
            "visual_reread_count": reread_count,
            "visual_reverified_from_persisted_candidates": True,
        }
    )
    result["metadata"] = metadata
    build_report["candidate_relations"] = [
        relation.to_dict() for relation in reread_candidates
    ]
    build_report["rejected_relations"] = rejected
    build_report["verifier_summary"] = summary
    result["build_report"] = build_report
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument(
        "--video-skills-root",
        default=Path("/fs/gamma-projects/vlm-robot/Video_Skills"),
        type=Path,
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-9B")
    parser.add_argument(
        "--api-base",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--frames-per-window", type=int, default=3)
    args = parser.parse_args(argv)

    root = str(args.video_skills_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from atomic_skills.skill_model_client import SkillModelClient

    provider = QwenVisualRereadProvider(
        client=SkillModelClient.from_local(
            model=args.model,
            base_url=args.api_base,
            max_tokens=900,
            timeout_s=180,
        ),
        frames_per_window=args.frames_per_window,
    )
    totals = {
        "video_count": 0,
        "visual_reread_count": 0,
        "accepted_labels": 0,
        "rejected_labels": 0,
    }
    for path in sorted(args.run_dir.glob("*/causal_temporal_overlay.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        updated = reverify_with_video(payload, provider=provider)
        backup = path.with_name("causal_temporal_overlay.pre_visual_reverify.json")
        if not backup.exists():
            backup.write_text(
                json.dumps(payload, indent=2) + "\n",
                encoding="utf-8",
            )
        path.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
        metadata = updated.get("metadata") or {}
        summary = (updated.get("build_report") or {}).get("verifier_summary") or {}
        totals["video_count"] += 1
        totals["visual_reread_count"] += int(
            metadata.get("visual_reread_count") or 0
        )
        totals["accepted_labels"] += int(summary.get("accepted_labels") or 0)
        totals["rejected_labels"] += int(summary.get("rejected_labels") or 0)
    report = args.run_dir / "visual_reverification_summary.json"
    report.write_text(json.dumps(totals, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
