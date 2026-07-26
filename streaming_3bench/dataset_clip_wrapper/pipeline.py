"""Build canonical video examples with clip segmentation and optional backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from .adapters.base import RawDatasetItem
from .adapters import get_adapter
from .backbone import PerceptionBackbone, build_backbone
from .clip_policy import FineExpansion, segment_video
from .clue_memory import extract_clue_memory_graph
from .dataset_graph_presets import task_family_for
from .reasoning_rollout import build_reasoning_rollout
from .schemas import (
    BackboneConfig,
    ClipPolicyConfig,
    RuntimeMode,
    WrapperConfig,
    make_canonical_shell,
)
from .subtitles import parse_srt
from .video_probe import probe_duration_s


def _clip_id(video_id: str, index: int, granularity: str) -> str:
    return f"clip:{video_id}:{granularity}:{index:04d}"


def _index_fine_expansion(policy: ClipPolicyConfig) -> FineExpansion:
    if policy.strategy != "hierarchical":
        return "all"
    if policy.index_fine_expansion == "all":
        return "all"
    return "none"


def _omit_none_values(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def build_canonical_example(
    item: RawDatasetItem,
    *,
    config: WrapperConfig,
    backbone: PerceptionBackbone | None = None,
) -> dict[str, Any]:
    duration_s = item.duration_s
    if (duration_s is None or duration_s <= 0) and item.video_path and item.video_path.exists():
        duration_s = probe_duration_s(item.video_path)
    if not duration_s or duration_s <= 0:
        duration_s = 120.0 if item.dataset in {"cg_bench", "vrbench"} else 60.0

    clip_policy = config.resolved_clip_policy(duration_s)
    spans = segment_video(
        duration_s,
        clip_policy,
        regime=config.regime,
        fine_expansion=_index_fine_expansion(clip_policy),
    )

    primary_path = str(item.video_path) if item.video_path else ""
    derived_clips: list[dict[str, Any]] = []
    clip_evidence: list[dict[str, Any]] = []
    index_nodes: list[dict[str, Any]] = []
    index_edges: list[dict[str, Any]] = []

    question_context = item.question.get("question_text")
    backbone = backbone or build_backbone(config.backbone)
    caption_budget = config.backbone.max_clips

    for clip in spans:
        clip_id = _clip_id(item.video_id, clip.clip_index, clip.granularity)
        derived = {
            "clip_id": clip_id,
            "path": primary_path,
            "source_span": clip.to_dict(),
            "granularity": clip.granularity,
            "parent_index": clip.parent_index,
        }
        derived_clips.append(derived)
        clip_evidence.append(
            {
                "evidence_id": f"ev:clip:{clip_id}",
                "source_type": "video_segment",
                "time_span": clip.to_dict(),
                "text": f"{clip.granularity} clip [{clip.start_s:.2f}, {clip.end_s:.2f}]",
                "trust_level": "derived",
                "provenance": {
                    "created_by": "dataset_clip_wrapper.segment_video",
                    "clip_policy": clip_policy.to_dict(),
                },
                "media_ref": {"video_id": item.video_id, "path": primary_path, "clip_id": clip_id},
            }
        )
        index_nodes.append(
            {
                "node_id": clip_id,
                "node_type": "clip",
                "video_id": item.video_id,
                "time_span": clip.to_dict(),
                "granularity": clip.granularity,
                "clip_policy": clip_policy.strategy,
            }
        )

    for left, right in zip(index_nodes, index_nodes[1:]):
        index_edges.append(
            {
                "edge_id": f"edge:{left['node_id']}->{right['node_id']}",
                "src": left["node_id"],
                "dst": right["node_id"],
                "edge_type": "temporal_next",
            }
        )

    visible_segments: list[dict[str, Any]] = []
    hidden_segments: list[dict[str, Any]] = list(item.annotation_segments)
    if config.mode == RuntimeMode.EXPERT_DEMO:
        visible_segments.extend(hidden_segments)
        for subtitle_path in item.subtitle_paths:
            if subtitle_path.exists():
                visible_segments.extend(parse_srt(subtitle_path))

    evidence_candidates = list(item.evidence_seeds)
    if config.mode == RuntimeMode.VIDEO_ONLY:
        evidence_candidates = []
    evidence_candidates = [_omit_none_values(evidence) for evidence in evidence_candidates]

    evidence_candidates.extend(clip_evidence)

    for seg in visible_segments:
        if seg.get("text"):
            evidence_candidates.append(
                _omit_none_values(
                    {
                        "evidence_id": f"ev:seg:{seg['segment_id']}",
                        "source_type": seg.get("source_type", "segment_description"),
                        "time_span": seg.get("time_span"),
                        "text": seg.get("text"),
                        "trust_level": "gold" if config.mode == RuntimeMode.EXPERT_DEMO else "weak",
                        "provenance": seg.get("provenance", {}),
                    }
                )
            )
            index_nodes.append(
                _omit_none_values(
                    {
                        "node_id": seg["segment_id"],
                        "node_type": "observation",
                        "text": seg.get("text"),
                        "time_span": seg.get("time_span"),
                        "source_type": seg.get("source_type"),
                    }
                )
            )

    if config.run_backbone and item.video_path and item.video_path.exists():
        for i, (clip, derived) in enumerate(zip(spans, derived_clips)):
            if caption_budget is not None and i >= caption_budget:
                break
            observation = backbone.describe_clip(
                video_path=item.video_path,
                clip=clip,
                question_context=question_context,
            )
            if observation.get("skipped") or not observation.get("text"):
                continue
            obs_id = f"obs:{derived['clip_id']}"
            evidence_candidates.append(
                {
                    "evidence_id": f"ev:{obs_id}",
                    "source_type": "caption_span",
                    "time_span": clip.to_dict(),
                    "text": observation["text"],
                    "trust_level": observation.get("trust_level", "model_labeled"),
                    "provenance": {
                        "created_by": "dataset_clip_wrapper.perception.backbone",
                        "backbone": config.backbone.to_dict(),
                    },
                    "discovery_status": observation.get("discovery_status", "discovered_runtime"),
                }
            )
            index_nodes.append(
                {
                    "node_id": obs_id,
                    "node_type": "observation",
                    "text": observation["text"],
                    "time_span": clip.to_dict(),
                    "modality": observation.get("modality", "visual_caption"),
                }
            )
            index_edges.append(
                {
                    "edge_id": f"edge:{obs_id}->{derived['clip_id']}",
                    "src": obs_id,
                    "dst": derived["clip_id"],
                    "edge_type": "derived_from",
                }
            )

    subtitle_tracks = [
        {
            "track_id": f"sub:{path.stem}",
            "path": str(path),
            "format": path.suffix.lstrip(".") or "srt",
            "language": "en",
        }
        for path in item.subtitle_paths
        if path.exists()
    ]

    example = make_canonical_shell(
        example_id=item.example_id,
        dataset=item.dataset,
        task_family=task_family_for(
            item.dataset,
            regime=config.regime,
            profile=config.benchmark_profile,
            adapter_task_family=item.task_family,
        ),
        split=item.split,
        video={
            "video_id": item.video_id,
            "primary_path": primary_path,
            "duration_s": duration_s,
            "fps": None,
            "resolution": None,
            "language": "en",
            "subtitle_tracks": subtitle_tracks,
            "caption_tracks": [],
            "derived_clips": derived_clips,
            "segments": [_omit_none_values(segment) for segment in visible_segments],
        },
        question=item.question,
        mode=config.mode,
        clip_policy=clip_policy,
        backbone=config.backbone,
        hidden_sources=item.hidden_supervision_sources,
        video_regime=config.regime,
        retrieval=config.retrieval,
    )
    example["evidence_candidates"] = evidence_candidates
    example["raw_source_refs"] = item.raw_source_refs
    example["metadata"].update(item.metadata or {})
    example["metadata"]["video_regime"] = config.regime.value
    example["metadata"]["retrieval"] = config.retrieval.to_dict()
    example["metadata"]["clip_count"] = len(derived_clips)
    example["metadata"]["index_clip_count"] = len(derived_clips)
    example["metadata"]["coarse_clip_count"] = sum(1 for c in derived_clips if c.get("granularity") == "coarse")
    example["metadata"]["fine_clip_count"] = sum(1 for c in derived_clips if c.get("granularity") == "fine")
    example["metadata"]["index_fine_expansion"] = clip_policy.index_fine_expansion
    example["metadata"]["hidden_segment_count"] = len(hidden_segments)
    example["evidence_index"]["nodes"] = index_nodes
    example["evidence_index"]["edges"] = index_edges
    example["evidence_index"]["retrieval"] = config.retrieval.to_dict()
    clue_graph = extract_clue_memory_graph(example, mode=config.mode)
    example["metadata"]["clue_memory_graph"] = clue_graph
    reasoning_rollout = build_reasoning_rollout(example, clue_graph)
    example["metadata"]["reasoning_rollout"] = reasoning_rollout
    example["metadata"]["reasoning_rollout_shell"] = reasoning_rollout
    return example


def iter_canonical_examples(config: WrapperConfig) -> Iterator[dict[str, Any]]:
    adapter = get_adapter(config.dataset, Path(config.dataset_root), split=config.split)
    backbone = build_backbone(config.backbone if config.run_backbone else BackboneConfig(name="annotation_only"))
    for item in adapter.iter_items(limit=config.limit):
        yield build_canonical_example(item, config=config, backbone=backbone)
