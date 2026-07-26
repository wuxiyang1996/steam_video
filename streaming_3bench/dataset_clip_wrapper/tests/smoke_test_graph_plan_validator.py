#!/usr/bin/env python3
"""Offline tests for constrained multiple-choice L1 skill plan validation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.l1_clue_graph.graph_composer import EXECUTABLE_SKILL_IDS, SKILL_EXECUTORS
from dataset_clip_wrapper.graph_plan_validator import (
    build_skill_plan_response_schema,
    resolve_plan_value,
    validate_skill_plan,
)


def main() -> int:
    allowed = set(EXECUTABLE_SKILL_IDS)
    bad_plan = [
        {
            "step_id": "s6",
            "skill_id": "resolve_entity_coreference",
            "args": {"mention_nodes": ["mention:s3", "mention:s5"]},
            "depends_on": [],
        },
        {
            "step_id": "s7",
            "skill_id": "create_state_node",
            "args": {},
            "depends_on": [],
        },
        {
            "step_id": "s8",
            "skill_id": "link_graph_relation",
            "args": {"source_node": "entity:s3", "target_node": "entity:s4", "edge_type": "on_top_of"},
            "depends_on": [],
        },
    ]
    good_plan = [
        {
            "step_id": "s1",
            "skill_id": "segment_video_or_select_clip",
            "args": {"video_id": "$bindings.video_id", "clip_policy": "$bindings.clip_policy"},
            "depends_on": [],
        },
        {
            "step_id": "s2",
            "skill_id": "extract_observation",
            "args": {
                "clip_or_text_ref": "$step.s1.evidence_refs.0",
                "modality": "visual",
                "text": "A person stands by a fence.",
            },
            "depends_on": ["s1"],
        },
    ]

    bad_errors = validate_skill_plan(bad_plan, allowed_skill_ids=allowed)
    good_errors = validate_skill_plan(good_plan, allowed_skill_ids=allowed)
    schema = build_skill_plan_response_schema(EXECUTABLE_SKILL_IDS)
    resolved = resolve_plan_value(
        {"video_id": "$bindings.video_id"},
        {"video_id": "vid123"},
        {},
    )

    report = {
        "executable_skill_count": len(EXECUTABLE_SKILL_IDS),
        "schema_skill_enum": schema["json_schema"]["schema"]["properties"]["skill_plan"]["items"]["properties"]["skill_id"]["enum"],
        "bad_plan_rejected": len(bad_errors) >= 3,
        "bad_plan_errors_sample": bad_errors[:5],
        "good_plan_accepted": not good_errors,
        "bindings_resolve": resolved,
        "passed": len(bad_errors) >= 3 and not good_errors and resolved == {"video_id": "vid123"},
    }
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
