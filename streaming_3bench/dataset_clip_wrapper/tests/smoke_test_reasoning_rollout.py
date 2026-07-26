#!/usr/bin/env python3
"""Validate layer-2 deterministic reasoning rollout executes the 19 core skills."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.schemas import RuntimeMode, VideoRegime, WrapperConfig

EXPECTED_SKILLS = {
    "parse_question_target",
    "propose_evidence_roles",
    "retrieve_by_event",
    "retrieve_by_entity",
    "retrieve_by_time",
    "retrieve_by_relation",
    "localize_clue",
    "extract_claim",
    "assign_evidence_role",
    "compose_evidence_chain",
    "detect_missing_role",
    "search_counterevidence",
    "infer_temporal_relation",
    "infer_state_change",
    "infer_causal_relation",
    "infer_intention_or_motive",
    "infer_social_contradiction",
    "verify_claim_support",
    "commit_answer",
}


def main() -> int:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    cases = [
        ("video_holmes", VideoRegime.SHORT),
        ("cg_bench", VideoRegime.LONG),
        ("cg_bench", VideoRegime.STREAMING),
    ]
    report = []
    for dataset, regime in cases:
        config = WrapperConfig(
            dataset_root=dataset_root,
            dataset=dataset,  # type: ignore[arg-type]
            regime=regime,
            mode=RuntimeMode.VIDEO_ONLY,
            split="train",
            limit=1,
        )
        example = next(iter_canonical_examples(config))
        rollout = example["metadata"]["reasoning_rollout"]
        executed = set(rollout.get("metadata", {}).get("executed_skill_ids") or [])
        node_skills = {node.get("skill_id") for node in rollout.get("nodes", []) if node.get("skill_id")}
        missing = sorted(EXPECTED_SKILLS - executed)
        errors: list[str] = []
        if missing:
            errors.append(f"missing skills: {missing}")
        if len(rollout.get("nodes", [])) < 19:
            errors.append(f"expected >=19 rollout nodes, got {len(rollout.get('nodes', []))}")
        if not rollout.get("claims"):
            errors.append("missing claims")
        if rollout.get("layer") != "reasoning":
            errors.append("rollout missing layer=reasoning")
        if node_skills != executed:
            errors.append("executed_skill_ids mismatch with node skill_ids")
        if regime == VideoRegime.STREAMING and dataset == "cg_bench":
            clip_count = example["metadata"].get("clip_count")
            if clip_count and clip_count > 200:
                errors.append(f"cg streaming should use coarse index, got clip_count={clip_count}")
        report.append(
            {
                "dataset": dataset,
                "regime": regime.value,
                "executed_skill_count": len(executed),
                "rollout_nodes": len(rollout.get("nodes", [])),
                "acceptance_status": rollout.get("acceptance_status"),
                "passed": not errors,
                "errors": errors,
            }
        )
    print(json.dumps(report, indent=2))
    return 0 if all(item["passed"] for item in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
