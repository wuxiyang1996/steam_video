#!/usr/bin/env python3
"""Validate two-layer graph shells across supported datasets and video regimes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.l1_clue_graph.clue_memory import extract_clue_memory_graph
from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.schemas import RuntimeMode, VideoRegime, WrapperConfig

CLUE_SCHEMA = Draft202012Validator(json.loads((REPO_ROOT / "schemas" / "clue_memory_graph.schema.json").read_text()))
ROLL_SCHEMA = Draft202012Validator(json.loads((REPO_ROOT / "schemas" / "skill_graph_rollout.schema.json").read_text()))


def _schema_errors(validator: Draft202012Validator, payload: dict) -> list[str]:
    return [f"{'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in validator.iter_errors(payload)]


def main() -> int:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    cases = [
        ("video_holmes", VideoRegime.SHORT),
        ("video_holmes", VideoRegime.STREAMING),
        ("siv_bench", VideoRegime.SHORT),
        ("cg_bench", VideoRegime.LONG),
        ("vrbench", VideoRegime.LONG),
        ("cg_bench", VideoRegime.STREAMING),
        ("ovo_bench", VideoRegime.STREAMING),
        ("videomme", VideoRegime.SHORT),
    ]
    report = []
    for dataset, regime in cases:
        for mode in (RuntimeMode.EXPERT_DEMO, RuntimeMode.VIDEO_ONLY):
            config = WrapperConfig(
                dataset_root=dataset_root,
                dataset=dataset,  # type: ignore[arg-type]
                regime=regime,
                mode=mode,
                split="train",
                limit=1,
            )
            example = next(iter_canonical_examples(config))
            clue = example["metadata"]["clue_memory_graph"]
            rollout = example["metadata"]["reasoning_rollout"]
            errors = _schema_errors(CLUE_SCHEMA, clue) + _schema_errors(ROLL_SCHEMA, rollout)
            if rollout.get("clue_memory_ref", {}).get("graph_id") != clue.get("graph_id"):
                errors.append("reasoning rollout clue_memory_ref mismatch")
            if clue.get("layer") != "clue_memory":
                errors.append("clue graph missing layer=clue_memory")
            if rollout.get("layer") != "reasoning":
                errors.append("rollout missing layer=reasoning")
            skill_count = rollout.get("metadata", {}).get("executed_skill_count", 0)
            if skill_count < 19:
                errors.append(f"reasoning rollout executed {skill_count}/19 skills")
            if mode == RuntimeMode.VIDEO_ONLY:
                leaked = [n.get("source_type") for n in clue.get("nodes", []) if n.get("source_type") in {
                    "segment_description", "inference_shot", "clue_interval", "reasoning_process_step"
                }]
                if leaked:
                    errors.append(f"video_only clue graph leaked hidden nodes: {sorted(set(leaked))}")
            if regime == VideoRegime.STREAMING:
                obs_end = clue.get("observation_end_s")
                if obs_end is None:
                    errors.append("streaming clue graph missing observation_end_s")
                else:
                    for node in clue.get("nodes", []):
                        span = node.get("time_span")
                        if span and span["end_s"] > obs_end + 1e-6:
                            errors.append("streaming clue graph node exceeds observation_end_s")
                            break
            report.append(
                {
                    "dataset": dataset,
                    "regime": regime.value,
                    "mode": mode.value,
                    "video_regime": clue.get("video_regime"),
                    "clue_nodes": len(clue.get("nodes", [])),
                    "index_clip_count": clue.get("index_stats", {}).get("index_clip_count"),
                    "passed": not errors,
                    "errors": errors,
                }
            )
    print(json.dumps(report, indent=2))
    return 0 if all(item["passed"] for item in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
