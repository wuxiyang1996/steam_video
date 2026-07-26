#!/usr/bin/env python3
"""Smoke test: SkillExecutor dispatches to rule vs LLM backend correctly."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from atomic_skills.common import SCHEMA_VERSION
from atomic_skills.skill_backends import SkillBackendConfig, SkillBackendMode
from atomic_skills.skill_model_client import SkillModelClient
from atomic_skills.skill_executor import SkillExecutor


def _sample_graph():
    return {
        "schema_version": SCHEMA_VERSION,
        "nodes": [
            {"node_id": "event:e1", "node_type": "event", "text": "A man leaves the iron fence.", "time_span": {"start_s": 10, "end_s": 12}},
            {"node_id": "event:e2", "node_type": "event", "text": "The man returns to the fence.", "time_span": {"start_s": 120, "end_s": 125}},
            {"node_id": "state:s1", "node_type": "state", "text": "Iron fence marks original position.", "time_span": {"start_s": 10, "end_s": 125}},
        ],
        "edges": [
            {"edge_id": "edge:1", "src": "event:e1", "dst": "state:s1", "edge_type": "same_location"},
            {"edge_id": "edge:2", "src": "state:s1", "dst": "event:e2", "edge_type": "temporal_next"},
        ],
    }


def test_rule_mode():
    """Rule mode should work without any LLM client."""
    config = SkillBackendConfig(default_mode=SkillBackendMode.RULE)
    executor = SkillExecutor(config=config)

    result = executor.execute(
        "parse_question_target",
        args={"question_text": "Where did the man go?", "options": [{"label": "A", "text": "Away"}]},
        graph=None,
    )
    assert result.ok, f"parse_question_target failed: {result.failure_code}"
    assert "question_focus" in result.outputs

    result = executor.execute(
        "retrieve_by_event",
        args={"event_description": "man leaves fence"},
        graph=_sample_graph(),
    )
    assert result.ok, f"retrieve_by_event failed: {result.failure_code}"
    print(f"  rule mode: retrieve_by_event found {len(result.evidence_refs)} refs")

    result = executor.execute(
        "infer_causal_relation",
        args={"candidate_cause": "man leaves", "candidate_effect": "man returns", "evidence_chain": {"evidence_refs": ["event:e1"]}},
        graph=_sample_graph(),
    )
    assert result.ok is not None
    print(f"  rule mode: infer_causal_relation ok={result.ok}")

    print("PASS: rule mode")


def test_llm_mode_with_mock():
    """LLM mode should call the model and parse response."""
    config = SkillBackendConfig(default_mode=SkillBackendMode.LLM)
    mock_client = MagicMock(spec=SkillModelClient)

    mock_client.reason.return_value = {
        "causal": True,
        "confidence": 0.85,
        "reasoning": "The man leaving and then returning implies a round trip."
    }

    executor = SkillExecutor(llm_client=mock_client, config=config)

    result = executor.execute(
        "infer_causal_relation",
        args={
            "candidate_cause": "man leaves fence",
            "candidate_effect": "man returns to fence",
            "evidence_chain": {"evidence_refs": ["event:e1", "event:e2"]},
        },
        graph=_sample_graph(),
    )
    assert result.ok, f"infer_causal_relation LLM mode failed: {result.failure_code}"
    assert result.outputs.get("backend") == "llm"
    assert mock_client.reason.called
    print(f"  llm mode: infer_causal_relation ok, confidence={result.confidence}")

    mock_client.reason.return_value = {
        "supported": True,
        "score": 0.92,
        "reasoning": "Evidence directly supports the claim."
    }
    result = executor.execute(
        "verify_claim_support",
        args={
            "claim": "The man walked back to his original position.",
            "evidence_chain": {"evidence_refs": ["event:e2", "state:s1"]},
        },
        graph=_sample_graph(),
    )
    assert result.ok, f"verify_claim_support LLM mode failed: {result.failure_code}"
    assert result.outputs.get("backend") == "llm"
    print(f"  llm mode: verify_claim_support ok, score={result.outputs.get('verification_score')}")

    mock_client.reason.return_value = {
        "relevant_ids": ["event:e1", "event:e2"],
        "scores": {"event:e1": 0.9, "event:e2": 0.8}
    }
    result = executor.execute(
        "retrieve_by_event",
        args={"event_description": "man at fence"},
        graph=_sample_graph(),
    )
    assert result.ok
    assert result.outputs.get("backend") == "llm"
    assert "event:e1" in result.evidence_refs
    print(f"  llm mode: retrieve_by_event found {result.evidence_refs}")

    print("PASS: llm mode (mock)")


def test_rule_only_skill_stays_rule():
    """Rule-only skills should never dispatch to LLM even in LLM mode."""
    config = SkillBackendConfig(default_mode=SkillBackendMode.LLM)
    mock_client = MagicMock(spec=SkillModelClient)
    executor = SkillExecutor(llm_client=mock_client, config=config)

    result = executor.execute(
        "parse_question_target",
        args={"question_text": "What happened next?"},
        graph=None,
    )
    assert result.ok
    assert not mock_client.reason.called, "rule-only skill should NOT call LLM"
    print("PASS: rule-only skills stay rule")


def test_vlm_mode_video_clips():
    """VLM mode should handle video clip perception with frame sampling."""
    config = SkillBackendConfig(default_mode=SkillBackendMode.LLM)
    mock_vlm = MagicMock(spec=SkillModelClient)

    mock_vlm.perceive.return_value = {
        "observations": [
            {"text": "A man walks away from an iron fence.", "modality": "visual", "confidence": 0.9},
            {"text": "The scene shows a residential street.", "modality": "visual", "confidence": 0.85},
        ],
        "scene_description": "Outdoor residential area with iron fence."
    }

    executor = SkillExecutor(vlm_client=mock_vlm, config=config)

    graph = _sample_graph()
    graph["video_path"] = "/fake/video.mp4"
    graph["nodes"].append({
        "node_id": "clip:c1",
        "node_type": "clip",
        "time_span": {"start_s": 10.0, "end_s": 15.0},
        "representative_frame": {"time_s": 12.0, "image_url": "data:image/jpeg;base64,FAKE"},
    })

    result = executor.execute(
        "extract_observation",
        args={
            "clip_or_text_ref": "clip:c1",
            "modality": "visual",
            "text": "",
            "observation_query": "What is the man doing?",
        },
        graph=graph,
    )
    assert result.ok, f"extract_observation VLM failed: {result.failure_code}"
    assert result.outputs.get("backend") == "vlm"
    assert len(result.outputs.get("observation_nodes", [])) == 2
    assert mock_vlm.perceive.called

    call_args = mock_vlm.perceive.call_args
    prompt_text = call_args[0][0] if call_args[0] else call_args.kwargs.get("prompt", "")
    assert "10.0s" in prompt_text or "video clip" in prompt_text
    image_urls = call_args.kwargs.get("image_urls") or (call_args[1] if len(call_args) > 1 else [])
    print(f"  vlm mode: extract_observation ok, 2 obs nodes, frames passed: {bool(image_urls)}")

    mock_vlm.perceive.return_value = {
        "entities": [
            {"surface_form": "man in blue jacket", "entity_type": "person", "first_appearance_s": 10.5, "confidence": 0.88},
            {"surface_form": "iron fence", "entity_type": "object", "first_appearance_s": 10.0, "confidence": 0.95},
        ]
    }
    result = executor.execute(
        "detect_entity_mention",
        args={"observation_ref": "clip:c1", "entity_type": "person"},
        graph=graph,
    )
    assert result.ok
    assert result.outputs.get("backend") == "vlm"
    assert len(result.outputs.get("mention_nodes", [])) == 2
    print(f"  vlm mode: detect_entity_mention ok, 2 entities")

    mock_vlm.perceive.return_value = {
        "dialogue_spans": [
            {"speaker": "man", "utterance": "I'll be right back.", "start_s": 11.0, "end_s": 12.0, "confidence": 0.82}
        ]
    }
    result = executor.execute(
        "extract_dialogue_span",
        args={"subtitle_or_asr_ref": "clip:c1", "text": "", "time_span": {"start_s": 10.0, "end_s": 15.0}},
        graph=graph,
    )
    assert result.ok
    assert result.outputs.get("backend") == "vlm"
    assert result.outputs.get("speaker_mention") == "man"
    print(f"  vlm mode: extract_dialogue_span ok, speaker='man'")

    print("PASS: vlm mode (video clips)")


def test_planner_integration_with_executor():
    """execute_reasoning_plan with skill_executor dispatches LLM skills via executor."""
    from dataset_clip_wrapper.l2_reasoning_graph.reasoning_planner import execute_reasoning_plan

    config = SkillBackendConfig(default_mode=SkillBackendMode.LLM)
    mock_client = MagicMock(spec=SkillModelClient)
    mock_client.reason.return_value = {
        "causal": True,
        "confidence": 0.9,
        "reasoning": "test"
    }
    executor = SkillExecutor(llm_client=mock_client, config=config)

    plan = [
        {"step_id": "r1", "skill_id": "parse_question_target", "args": {"question_text": "$bindings.question_text"}, "depends_on": []},
        {"step_id": "r2", "skill_id": "infer_causal_relation", "args": {"candidate_cause": "left", "candidate_effect": "returned", "evidence_chain": {"evidence_refs": ["event:e1"]}}, "depends_on": ["r1"]},
    ]

    trace, outputs = execute_reasoning_plan(
        reasoning_plan=plan,
        clue_memory_graph=_sample_graph(),
        question={"question_text": "Why did the man return?", "options": []},
        skill_executor=executor,
    )

    assert trace[0]["ok"], f"r1 failed: {trace[0]}"
    assert trace[1]["ok"], f"r2 failed: {trace[1]}"
    assert trace[1].get("backend") == "llm"
    assert mock_client.reason.called
    print(f"PASS: planner integration (rule r1 + llm r2)")


def main() -> int:
    errors = []
    for name, fn in [
        ("rule_mode", test_rule_mode),
        ("llm_mode_mock", test_llm_mode_with_mock),
        ("rule_only_stays_rule", test_rule_only_skill_stays_rule),
        ("vlm_mode_video_clips", test_vlm_mode_video_clips),
        ("planner_integration", test_planner_integration_with_executor),
    ]:
        try:
            fn()
        except Exception as exc:
            errors.append(f"{name}: {exc}")
            import traceback
            traceback.print_exc()

    if errors:
        print(f"\nFAILED ({len(errors)} tests):")
        for e in errors:
            print(f"  - {e}")
        return 2
    print(f"\nAll tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
