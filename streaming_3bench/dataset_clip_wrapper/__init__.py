"""Dataset clip wrappers for core and streaming video benchmarks."""

from __future__ import annotations

import importlib
import sys


_COMPAT_MODULE_ALIASES = {
    "l1": "l1_clue_graph",
    "l2": "l2_reasoning_graph",
    "backbone": "perception.backbone",
    "clip_policy": "perception.clip_policy",
    "clip_schema": "perception.clip_schema",
    "openrouter_client": "perception.openrouter_client",
    "subtitles": "perception.subtitles",
    "video_probe": "perception.video_probe",
    "video_tool_backend": "perception.video_tool_backend",
    "clip_retrieval": "l1_clue_graph.clip_retrieval",
    "clue_memory": "l1_clue_graph.clue_memory",
    "gate_l1_for_l2": "l1_clue_graph.gate_l1_for_l2",
    "graph_composer": "l1_clue_graph.graph_composer",
    "graph_plan_validator": "l1_clue_graph.graph_plan_validator",
    "skill_graph_bridge": "l1_clue_graph.skill_graph_bridge",
    "fault_repair": "l2_reasoning_graph.fault_repair",
    "l2_recursive_trace": "l2_reasoning_graph.l2_recursive_trace",
    "reasoning_planner": "l2_reasoning_graph.reasoning_planner",
    "reasoning_rollout": "l2_reasoning_graph.reasoning_rollout",
    "evaluate_l1_query_memory": "verification.evaluate_l1_query_memory",
    "evaluate_vrbench_video_only_graph": "verification.evaluate_vrbench_video_only_graph",
    "runtime_verifier": "verification.runtime_verifier",
    "llm_pipeline": "runners.llm_pipeline",
}


for _old_name, _new_name in _COMPAT_MODULE_ALIASES.items():
    sys.modules.setdefault(
        f"{__name__}.{_old_name}",
        importlib.import_module(f".{_new_name}", __name__),
    )

from .l1_clue_graph.clue_memory import extract_clue_memory_graph, make_reasoning_rollout_shell
from .dataset_graph_presets import (
    DATASET_DEFAULT_REGIME,
    DATASET_HIDDEN_SOURCES,
    DATASET_LAYER1_PROFILE,
    clip_policy_for,
    default_regime_for_dataset,
    retrieval_for,
)
from .pipeline import build_canonical_example, iter_canonical_examples
from .runners.llm_pipeline import build_llm_enriched_example, iter_llm_enriched_examples
from .l1_clue_graph.skill_graph_bridge import canonical_example_to_skill_graph
from .perception.video_tool_backend import VideoToolConfig, VideoToolPerceptionBackend
from .schemas import (
    BackboneConfig,
    ClipPolicyConfig,
    ClipRetrievalConfig,
    ClipSchemaConfig,
    GraphComposerConfig,
    RuntimeMode,
    VideoRegime,
    WrapperConfig,
)

__all__ = [
    "BackboneConfig",
    "ClipPolicyConfig",
    "ClipRetrievalConfig",
    "ClipSchemaConfig",
    "GraphComposerConfig",
    "RuntimeMode",
    "VideoRegime",
    "VideoToolConfig",
    "VideoToolPerceptionBackend",
    "WrapperConfig",
    "DATASET_DEFAULT_REGIME",
    "DATASET_HIDDEN_SOURCES",
    "DATASET_LAYER1_PROFILE",
    "build_canonical_example",
    "build_llm_enriched_example",
    "canonical_example_to_skill_graph",
    "clip_policy_for",
    "default_regime_for_dataset",
    "extract_clue_memory_graph",
    "iter_canonical_examples",
    "iter_llm_enriched_examples",
    "make_reasoning_rollout_shell",
    "retrieval_for",
]
