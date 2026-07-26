"""Canonical schema builders for dataset clip wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

SCHEMA_VERSION = "video-skills-relaunch/v0.1"


class VideoRegime(str, Enum):
    SHORT = "short"
    LONG = "long"
    STREAMING = "streaming"


class RuntimeMode(str, Enum):
    EXPERT_DEMO = "expert_demo"
    VIDEO_ONLY = "video_only"


class BenchmarkProfile(str, Enum):
    DEFAULT = "default"
    SHORT_MULTI_HOP = "short_multi_hop"
    LONG_COARSE_FINE = "long_coarse_fine"


DatasetName = Literal[
    "video_holmes",
    "cg_bench",
    "vrbench",
    "siv_bench",
    "ovo_bench",
    "videomme",
    "streaming_bench",
]


@dataclass
class ClipRetrievalConfig:
    """Coarse-clip retrieval gate (M3-style top-k before fine expansion)."""

    enabled: bool = True
    topk: int = 2
    threshold: float = 0.0
    mode: Literal["lexical", "sequential"] = "lexical"
    query_in_video_only: bool = False
    expand_time_anchors: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "topk": self.topk,
            "threshold": self.threshold,
            "mode": self.mode,
            "query_in_video_only": self.query_in_video_only,
            "expand_time_anchors": self.expand_time_anchors,
        }


@dataclass
class ClipPolicyConfig:
    """Clip segmentation hyperparameters."""

    strategy: str = "whole_video"
    window_s: float = 4.0
    overlap_s: float = 1.0
    coarse_window_s: float = 45.0
    fine_window_s: float = 8.0
    index_fine_expansion: Literal["none", "all", "retrieval_gated"] = "retrieval_gated"
    online: bool = False
    observation_end_s: float | None = None
    duration_s: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "strategy": self.strategy,
            "window_s": self.window_s,
            "overlap_s": self.overlap_s,
            "online": self.online,
        }
        if self.coarse_window_s is not None:
            payload["coarse_window_s"] = self.coarse_window_s
        if self.fine_window_s is not None:
            payload["fine_window_s"] = self.fine_window_s
        payload["index_fine_expansion"] = self.index_fine_expansion
        if self.observation_end_s is not None:
            payload["observation_end_s"] = self.observation_end_s
        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s
        return payload

    @classmethod
    def for_regime(
        cls,
        regime: VideoRegime,
        *,
        observation_end_s: float | None = None,
        duration_s: float | None = None,
    ) -> ClipPolicyConfig:
        if regime == VideoRegime.SHORT:
            return cls(
                strategy="whole_video",
                window_s=4.0,
                overlap_s=1.0,
                online=False,
                observation_end_s=observation_end_s,
                duration_s=duration_s,
            )
        if regime == VideoRegime.LONG:
            return cls(
                strategy="hierarchical",
                coarse_window_s=30.0,
                fine_window_s=8.0,
                overlap_s=2.0,
                index_fine_expansion="retrieval_gated",
                online=False,
                observation_end_s=observation_end_s,
                duration_s=duration_s,
            )
        return cls(
            strategy="fixed_window",
            window_s=4.0,
            overlap_s=1.0,
            online=True,
            observation_end_s=observation_end_s,
            duration_s=duration_s,
        )

    @classmethod
    def dataset_default(cls, dataset: DatasetName, regime: VideoRegime | None = None) -> ClipPolicyConfig:
        inferred = regime or {
            "video_holmes": VideoRegime.SHORT,
            "siv_bench": VideoRegime.SHORT,
            "cg_bench": VideoRegime.LONG,
            "vrbench": VideoRegime.LONG,
            "ovo_bench": VideoRegime.STREAMING,
            "videomme": VideoRegime.SHORT,
            "streaming_bench": VideoRegime.STREAMING,
        }[dataset]
        return cls.for_regime(inferred)


@dataclass
class BackboneConfig:
    """Perception backbone hyperparameters."""

    name: str = "annotation_only"
    model: str | None = None
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    max_clips: int | None = None
    temperature: float = 0.0
    request_frames: int = 4

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "max_clips": self.max_clips,
            "temperature": self.temperature,
            "request_frames": self.request_frames,
        }


@dataclass
class ClipSchemaConfig:
    """Multimodal clip-schema producer hyperparameters (Qwen via OpenRouter)."""

    backend: Literal["qwen", "video_tools"] = "qwen"
    model: str = "qwen/qwen3.5-9b"
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    temperature: float = 0.0
    request_frames: int = 4
    max_clips: int | None = None
    max_tokens: int | None = 1200
    reasoning_effort: str | None = "none"
    timeout_s: int = 180

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "temperature": self.temperature,
            "request_frames": self.request_frames,
            "max_clips": self.max_clips,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "timeout_s": self.timeout_s,
        }


@dataclass
class GraphComposerConfig:
    """Graph composer hyperparameters.

    In teacher mode (expert demo generation): uses gpt-oss-120b for high-quality plans.
    In student mode (inference): uses qwen3.5-9b for all planning + execution.
    """

    model: str = "openai/gpt-oss-120b"
    api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    api_key_env: str = "OPENROUTER_API_KEY"
    keys_py_path: str | None = None
    temperature: float = 0.0
    use_llm_planner: bool = True
    composer_mode: Literal["neighbor_vlm_l1", "vlm_l1", "skill_plan", "deterministic"] = "neighbor_vlm_l1"
    max_tokens: int | None = 1800
    reasoning_effort: str | None = "minimal"
    timeout_s: int = 180
    neighbor_workers: int = 1
    neighbor_cache_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "keys_py_path": self.keys_py_path,
            "temperature": self.temperature,
            "use_llm_planner": self.use_llm_planner,
            "composer_mode": self.composer_mode,
            "max_tokens": self.max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "timeout_s": self.timeout_s,
            "neighbor_workers": self.neighbor_workers,
            "neighbor_cache_path": self.neighbor_cache_path,
        }


@dataclass
class SkillExecutionConfig:
    """Model allocation for atomic skill execution.

    Architecture:
    - teacher mode (expert_demo): gpt-oss-120b generates expert trajectories as
      supervision signal. Skill execution can also use gpt-oss for highest quality.
    - student mode (inference): Qwen3.5-9B handles everything — planning, reasoning,
      and perception. Single-model deployment after distillation.

    The planner model (GraphComposerConfig.model) controls L1/L2 plan generation.
    This config controls the model used for actual skill-level execution.
    """

    skill_model: str = "qwen/qwen3.5-9b"
    skill_api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    skill_max_tokens_llm: int = 512
    skill_max_tokens_vlm: int = 1024
    skill_temperature: float = 0.0
    skill_timeout_s: int = 120
    enable_llm_skills: bool = True
    enable_vlm_skills: bool = True
    llm_skill_scope: Literal["all", "verifier"] = "all"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_model": self.skill_model,
            "skill_api_base": self.skill_api_base,
            "skill_max_tokens_llm": self.skill_max_tokens_llm,
            "skill_max_tokens_vlm": self.skill_max_tokens_vlm,
            "skill_temperature": self.skill_temperature,
            "skill_timeout_s": self.skill_timeout_s,
            "enable_llm_skills": self.enable_llm_skills,
            "enable_vlm_skills": self.enable_vlm_skills,
            "llm_skill_scope": self.llm_skill_scope,
        }


@dataclass
class ClipSpan:
    start_s: float
    end_s: float
    granularity: Literal["whole", "coarse", "fine"] = "whole"
    parent_index: int | None = None
    clip_index: int = 0

    def to_dict(self) -> dict[str, float]:
        return {"start_s": self.start_s, "end_s": self.end_s}


@dataclass
class WrapperConfig:
    dataset_root: str
    dataset: DatasetName
    regime: VideoRegime = VideoRegime.SHORT
    benchmark_profile: BenchmarkProfile = BenchmarkProfile.DEFAULT
    mode: RuntimeMode = RuntimeMode.EXPERT_DEMO
    clip_policy: ClipPolicyConfig | None = None
    retrieval: ClipRetrievalConfig = field(default_factory=ClipRetrievalConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    clip_schema: ClipSchemaConfig = field(default_factory=ClipSchemaConfig)
    graph_composer: GraphComposerConfig = field(default_factory=GraphComposerConfig)
    skill_execution: SkillExecutionConfig = field(default_factory=SkillExecutionConfig)
    split: str = "train"
    limit: int | None = None
    run_backbone: bool = False
    run_clip_schema: bool = False
    run_graph_compose: bool = False
    run_l2_llm_planner: bool = False

    def resolved_clip_policy(self, duration_s: float | None = None) -> ClipPolicyConfig:
        from .dataset_graph_presets import clip_policy_for

        policy = self.clip_policy or clip_policy_for(self.dataset, self.regime, duration_s=duration_s)
        if duration_s is not None:
            policy.duration_s = duration_s
        if self.regime == VideoRegime.STREAMING and policy.observation_end_s is None and duration_s:
            policy.observation_end_s = duration_s
        policy.online = self.regime == VideoRegime.STREAMING or policy.online
        return policy


def make_canonical_shell(
    *,
    example_id: str,
    dataset: DatasetName,
    task_family: str,
    split: str,
    video: dict[str, Any],
    question: dict[str, Any],
    mode: RuntimeMode,
    clip_policy: ClipPolicyConfig,
    backbone: BackboneConfig,
    hidden_sources: list[str] | None = None,
    video_regime: VideoRegime | None = None,
    retrieval: ClipRetrievalConfig | None = None,
) -> dict[str, Any]:
    hidden = hidden_sources or []
    visible = ["video", "question"]
    if mode == RuntimeMode.EXPERT_DEMO:
        visible.extend(["subtitles", "captions", "dataset_annotations", "ground_truth_clues"])
    else:
        visible.extend(["automatic_clips", "automatic_segments"])

    return {
        "schema_version": SCHEMA_VERSION,
        "example_id": example_id,
        "dataset": dataset,
        "split": split,
        "task_family": task_family,
        "video": video,
        "question": question,
        "evidence_candidates": [],
        "available_inputs": {
            "mode": mode.value,
            "visible_to_agent": visible,
            "notes": "Generated by dataset_clip_wrapper",
        },
        "hidden_supervision": {
            "available_for_training": True,
            "available_for_inference": False,
            "sources": hidden,
        },
        "evidence_index": {
            "index_id": f"{dataset}:{example_id}:clip_index:v0",
            "index_type": "clip_memory_graph",
            "layer": "clue_memory",
            "visible_in_modes": ["expert_demo", "video_only"],
            "clip_policy": {**clip_policy.to_dict(), "video_regime": (video_regime or VideoRegime.SHORT).value},
            "retrieval": (retrieval or ClipRetrievalConfig()).to_dict(),
            "backbone": backbone.to_dict(),
            "node_types": ["clip", "observation", "subtitle_span", "caption_span", "dialogue_span", "event", "entity"],
            "edge_types": ["temporal_next", "derived_from", "entity_mention", "same_entity"],
            "nodes": [],
            "edges": [],
        },
        "raw_source_refs": [],
        "trust_policy": {
            "gold_sources": [],
            "strong_sources": [],
            "weak_sources": [],
            "model_labeled_sources": [],
        },
        "metadata": {
            "video_regime": (video_regime or VideoRegime.SHORT).value,
            "retrieval": (retrieval or ClipRetrievalConfig()).to_dict(),
            "wrapper_version": "dataset_clip_wrapper/v0.1",
        },
    }
