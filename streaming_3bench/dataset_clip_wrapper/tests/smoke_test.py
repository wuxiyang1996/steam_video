#!/usr/bin/env python3
"""Smoke test for dataset clip wrappers without API calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator

PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PKG_ROOT.parent
sys.path.insert(0, str(REPO_ROOT))

from dataset_clip_wrapper.pipeline import iter_canonical_examples
from dataset_clip_wrapper.schemas import BackboneConfig, RuntimeMode, VideoRegime, WrapperConfig
from dataset_clip_wrapper.l1_clue_graph.skill_graph_bridge import HIDDEN_SUPERVISION_SOURCE_TYPES, canonical_example_to_skill_graph

SCHEMA_PATH = REPO_ROOT / "schemas" / "canonical_video_example.schema.json"
SCHEMA_VALIDATOR = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))


def _check_example(example: dict) -> list[str]:
    errors: list[str] = []
    required = ["schema_version", "example_id", "dataset", "video", "question", "evidence_candidates", "evidence_index"]
    for key in required:
        if key not in example:
            errors.append(f"missing key: {key}")
    clips = example.get("video", {}).get("derived_clips", [])
    if not clips:
        errors.append("no derived_clips")
    policy = example.get("evidence_index", {}).get("clip_policy", {})
    if not policy.get("strategy"):
        errors.append("missing clip_policy.strategy")
    schema_errors = sorted(SCHEMA_VALIDATOR.iter_errors(example), key=lambda err: list(err.path))
    for error in schema_errors:
        path = ".".join(str(part) for part in error.path) or "<root>"
        errors.append(f"schema {path}: {error.message}")
    return errors


def _check_video_only_no_hidden_leakage(example: dict) -> list[str]:
    errors: list[str] = []
    if example.get("hidden_supervision", {}).get("available_for_inference") is not False:
        errors.append("hidden_supervision.available_for_inference must be false")
    hidden_sources = {
        "segment_description",
        "inference_shot",
        "key_relationship",
        "clue_interval",
        "clue_clip",
        "reasoning_process_step",
        "video_summary",
        "qa_answer",
    }
    leaked = [
        evidence.get("source_type")
        for evidence in example.get("evidence_candidates", [])
        if evidence.get("source_type") in hidden_sources
    ]
    if leaked:
        errors.append(f"hidden evidence leaked in video_only: {sorted(set(leaked))}")
    if example.get("video", {}).get("segments"):
        errors.append("video_only should not expose dataset annotation segments")
    return errors


def _check_skill_graph(example: dict) -> list[str]:
    errors: list[str] = []
    graph = canonical_example_to_skill_graph(example)
    if not graph.get("nodes"):
        errors.append("skill graph has no nodes")
    if not any(node.get("node_type") == "clip" for node in graph.get("nodes", [])):
        errors.append("skill graph has no clip nodes")
    for node in graph.get("nodes", []):
        if not node.get("node_id"):
            errors.append("skill graph node missing node_id")
        if not node.get("node_type"):
            errors.append(f"skill graph node missing node_type: {node}")
        if node.get("node_type") != "clip" and not node.get("source_ids"):
            errors.append(f"skill graph evidence node missing source_ids: {node.get('node_id')}")
    if graph.get("mode") == RuntimeMode.VIDEO_ONLY.value:
        leaked = [
            node.get("source_type")
            for node in graph.get("nodes", [])
            if node.get("source_type") in HIDDEN_SUPERVISION_SOURCE_TYPES
        ]
        if leaked:
            errors.append(f"hidden source nodes leaked into skill graph: {sorted(set(leaked))}")
    return errors


def main() -> int:
    dataset_root = "/mnt/is_data/xwu/video_skills/data/datasets"
    cases = [
        ("video_holmes", VideoRegime.SHORT, "train"),
        ("cg_bench", VideoRegime.LONG, "train"),
        ("vrbench", VideoRegime.LONG, "train"),
        ("siv_bench", VideoRegime.SHORT, "train"),
        ("ovo_bench", VideoRegime.STREAMING, "train"),
        ("videomme", VideoRegime.SHORT, "train"),
    ]
    report = []
    for dataset, regime, split in cases:
        config = WrapperConfig(
            dataset_root=dataset_root,
            dataset=dataset,  # type: ignore[arg-type]
            regime=regime,
            mode=RuntimeMode.EXPERT_DEMO,
            split=split,
            limit=1,
            backbone=BackboneConfig(name="annotation_only"),
            run_backbone=False,
        )
        example = next(iter_canonical_examples(config))
        errors = _check_example(example) + _check_skill_graph(example)
        report.append(
            {
                "dataset": dataset,
                "regime": regime.value,
                "mode": "expert_demo",
                "example_id": example["example_id"],
                "clip_count": len(example["video"]["derived_clips"]),
                "coarse_clip_count": example["metadata"].get("coarse_clip_count"),
                "fine_clip_count": example["metadata"].get("fine_clip_count"),
                "evidence_count": len(example["evidence_candidates"]),
                "strategy": example["evidence_index"]["clip_policy"]["strategy"],
                "index_fine_expansion": example["metadata"].get("index_fine_expansion"),
                "passed": not errors,
                "errors": errors,
            }
        )

    for dataset, regime, split in cases:
        config = WrapperConfig(
            dataset_root=dataset_root,
            dataset=dataset,  # type: ignore[arg-type]
            regime=regime,
            mode=RuntimeMode.VIDEO_ONLY,
            split=split,
            limit=1,
            backbone=BackboneConfig(name="annotation_only"),
            run_backbone=False,
        )
        example = next(iter_canonical_examples(config))
        errors = _check_example(example) + _check_video_only_no_hidden_leakage(example) + _check_skill_graph(example)
        report.append(
            {
                "dataset": dataset,
                "regime": regime.value,
                "mode": "video_only",
                "example_id": example["example_id"],
                "clip_count": len(example["video"]["derived_clips"]),
                "evidence_count": len(example["evidence_candidates"]),
                "strategy": example["evidence_index"]["clip_policy"]["strategy"],
                "passed": not errors,
                "errors": errors,
            }
        )

    streaming_config = WrapperConfig(
        dataset_root=dataset_root,
        dataset="ovo_bench",
        regime=VideoRegime.STREAMING,
        mode=RuntimeMode.VIDEO_ONLY,
        split="train",
        limit=1,
        backbone=BackboneConfig(name="annotation_only"),
        run_backbone=False,
    )
    streaming = next(iter_canonical_examples(streaming_config))
    max_end = max(c["source_span"]["end_s"] for c in streaming["video"]["derived_clips"])
    obs_end = streaming["evidence_index"]["clip_policy"].get("observation_end_s")
    streaming_errors = _check_example(streaming) + _check_video_only_no_hidden_leakage(streaming) + _check_skill_graph(streaming)
    report.append(
        {
            "dataset": "ovo_bench",
            "regime": "streaming",
            "mode": "video_only",
            "clip_count": len(streaming["video"]["derived_clips"]),
            "observation_end_s": obs_end,
            "max_clip_end_s": max_end,
            "passed": obs_end is not None and max_end <= obs_end + 1e-6 and not streaming_errors,
            "errors": streaming_errors,
        }
    )

    print(json.dumps(report, indent=2))
    return 0 if all(item["passed"] for item in report) else 2


if __name__ == "__main__":
    raise SystemExit(main())
