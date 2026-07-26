"""Compose clue-memory graphs with gpt-oss-120B over frozen graph-crafting atomic skills."""

from __future__ import annotations

import json
import hashlib
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from atomic_skills import export_skill_ontology  # noqa: E402
from atomic_skills.evidence_graph_construction import (  # noqa: E402
    assign_provenance_trust,
    create_event_node,
    create_state_node,
    detect_entity_mention,
    extract_dialogue_span,
    extract_observation,
    link_graph_relation,
    resolve_entity_coreference,
    segment_video_or_select_clip,
)

from .graph_plan_validator import (
    ALLOWED_MEMORY_EDGES,
    build_skill_plan_response_schema,
    executable_skill_ids,
    resolve_plan_value,
    validate_skill_plan,
)
from ..perception.openrouter_client import OpenRouterClient
from ..schemas import GraphComposerConfig, RuntimeMode

SKILL_EXECUTORS = {
    "segment_video_or_select_clip": segment_video_or_select_clip,
    "extract_observation": extract_observation,
    "extract_dialogue_span": extract_dialogue_span,
    "detect_entity_mention": detect_entity_mention,
    "resolve_entity_coreference": resolve_entity_coreference,
    "create_event_node": create_event_node,
    "create_state_node": create_state_node,
    "link_graph_relation": link_graph_relation,
    "assign_provenance_trust": assign_provenance_trust,
}

EXECUTABLE_SKILL_IDS = executable_skill_ids(SKILL_EXECUTORS)
_ALLOWED_EDGE_TYPES = ", ".join(sorted(ALLOWED_MEMORY_EDGES))
VLM_L1_EDGE_TYPES = {
    "temporal_next",
    "entity_mention",
    "derived_from",
    "same_entity",
    "same_object",
    "same_place",
    "reappears",
    "before_after",
    "state_change",
    "supports_observation",
    "contrasts_observation",
    "located_in",
    "causal_hint",
    "social_cue",
}
VLM_L1_NODE_TYPES = {"observation", "entity_mention", "event", "state", "dialogue_span", "clue"}
SHORT_RECURRENCE_GROUPS = {
    "iron_fence_or_gate": {
        "terms": ("iron fence", "rusty iron fence", "metal gate", "metal bars", "metal bar", "fence", "gate"),
        "label": "iron fence/gate",
        "edge_type": "reappears",
    },
    "red_shirt_person": {
        "terms": ("red shirt", "red-shirt", "red garment", "person in red", "red top"),
        "label": "red-shirted person",
        "edge_type": "same_entity",
    },
    "infinity_paper": {
        "terms": ("infinity symbol", "infinity drawing", "crumpled paper", "piece of paper"),
        "label": "paper with infinity symbol",
        "edge_type": "same_object",
    },
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _summarize_llm_usage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usages = [row.get("llm_usage") or {} for row in rows if isinstance(row, dict)]
    return {
        "calls": len(usages),
        "prompt_chars": sum(int(usage.get("prompt_chars") or 0) for usage in usages),
        "prompt_approx_tokens": sum(int(usage.get("prompt_approx_tokens") or 0) for usage in usages),
        "output_chars": sum(int(usage.get("output_chars") or 0) for usage in usages),
        "malformed_json_count": sum(int(usage.get("malformed_json") or 0) for usage in usages),
        "timeout_count": sum(int(usage.get("timeout_count") or 0) for usage in usages),
        "compact_retry_count": sum(int(usage.get("compact_retry_count") or 0) for usage in usages),
        "cache_hits": sum(1 for usage in usages if usage.get("cache_hit")),
        "cache_misses": sum(1 for usage in usages if not usage.get("cache_hit")),
    }

GRAPH_COMPOSE_PROMPT = f"""You compose a clue-memory / perception graph using ONLY the executable
Evidence Graph Construction atomic skills listed in allowed_skill_ids.

This is a MULTIPLE-CHOICE skill selection task:
- skill_id MUST be exactly one value from allowed_skill_ids (no new names, no synonyms).
- Do NOT output free-form skill names or invented skills such as create_relation_node.

Return JSON only:
{{
  "skill_plan": [
    {{
      "step_id": "s1",
      "skill_id": "segment_video_or_select_clip",
      "args": {{
        "video_id": "$bindings.video_id",
        "clip_policy": "$bindings.clip_policy"
      }},
      "depends_on": []
    }},
    {{
      "step_id": "s2",
      "skill_id": "extract_observation",
      "args": {{
        "clip_or_text_ref": "$step.s1.evidence_refs.0",
        "modality": "visual",
        "text": "grounded observation text from clip schema"
      }},
      "depends_on": ["s1"]
    }}
  ],
  "notes": "short summary"
}}

Rules:
1. skill_id is multiple-choice from allowed_skill_ids only.
2. Reference prior step outputs ONLY with $step.<step_id>.<field> or $step.<step_id>.evidence_refs.N
   Never invent placeholder ids like obs:s2, mention:s3, entity:s4, edge:s8.
3. For node ID references (observation_ref, entity_ref, source_node, target_node, clip_or_text_ref,
   node_or_edge_ref, subtitle_or_asr_ref), ALWAYS use $step.<step_id>.evidence_refs.N which resolves
   to a node_id string. Do NOT use $step.X.observation_nodes.0 or $step.X.mention_nodes.0 (these
   resolve to full node objects, not strings).
4. For mention_nodes (list of node IDs), use [$step.<step_id>.evidence_refs.0, $step.<other>.evidence_refs.0].
5. link_graph_relation edge_type MUST be one of: {_ALLOWED_EDGE_TYPES}.
6. assign_provenance_trust trust_policy MUST be an object with gold_sources / model_labeled_sources lists.
7. Prefer: segment -> extract_observation / extract_dialogue_span -> detect_entity_mention ->
   create_event_node -> link_graph_relation(temporal_next) -> assign_provenance_trust.
8. Keep the plan compact; skip coreference unless clearly needed.
9. Do not output chain-of-thought.
"""

VLM_L1_GRAPH_PROMPT = """You build a video-only L1 clue-memory graph directly from clip schemas.

Return JSON only:
{
  "nodes": [
    {
      "node_id": "optional stable local id",
      "node_type": "observation|entity_mention|event|state|dialogue_span|clue",
      "clip_id": "clip id from input",
      "time_span": {"start_s": number, "end_s": number},
      "text": "grounded clue text",
      "modality": "visual|audio|subtitle|ocr|mixed",
      "confidence": 0.0
    }
  ],
  "edges": [
    {
      "edge_id": "optional stable local id",
      "src": "node_id",
      "dst": "node_id",
      "edge_type": "same_object|same_place|reappears|before_after|state_change|supports_observation|contrasts_observation|located_in|causal_hint|social_cue|temporal_next|derived_from",
      "evidence_refs": ["node_id"],
      "text": "short grounded reason"
    }
  ],
  "reason_short": "one short quality note"
}

Rules:
1. Use only visible/spoken/subtitle/OCR evidence from clip_schemas.
2. Do not use official answers, hidden clues, labels, or dataset supervision.
3. Prefer reasoning-useful clues over generic captions.
4. Cross-clip semantic edges must be model-judged from the clip evidence, not string matching.
5. If evidence is weak, include uncertainty in node text and keep confidence low.
6. Every node must reference an input clip_id and time_span when possible.
7. Every edge src/dst must refer to nodes you output.
8. Do not output chain-of-thought.
"""

NEIGHBOR_VLM_L1_PROMPT = """You build a small local L1 clue graph for one target video clip.

Input contains one target_clip digest and a few neighbor clip digests.

Return JSON only:
{
  "target_nodes": [
    {
      "node_id": "optional local id",
      "node_type": "observation|entity_mention|event|state|dialogue_span|clue",
      "text": "short grounded clue for the target clip",
      "modality": "visual|audio|subtitle|ocr|mixed",
      "confidence": 0.0
    }
  ],
  "neighbor_edges": [
    {
      "src_clip_id": "neighbor or target clip id",
      "dst_clip_id": "neighbor or target clip id",
      "src_node_id": "optional local target node id",
      "dst_node_id": "optional local target node id",
      "edge_type": "same_object|same_place|reappears|before_after|state_change|supports_observation|contrasts_observation|located_in|causal_hint|social_cue|temporal_next",
      "text": "short grounded reason",
      "confidence": 0.0
    }
  ],
  "notes": "short quality note"
}

Rules:
1. Use only target_clip and neighbor_clips evidence.
2. Create target_nodes only for the target clip, not for neighbors.
3. neighbor_edges may connect the target clip to neighbor clips, or target nodes to target nodes.
4. Prefer sparse, reasoning-useful output: at most 4 target_nodes and 4 neighbor_edges.
5. Cross-clip edges must be model-judged from evidence, not string matching.
6. If the relationship is weak or generic, omit the edge.
7. Do not use answer labels, gold answers, hidden clues, or dataset supervision.
8. Do not output chain-of-thought.
9. Keep every text/reason field under 120 characters.
"""


def build_vlm_l1_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "vlm_l1_clue_graph",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "node_id": {"type": "string"},
                                "node_type": {"type": "string"},
                                "clip_id": {"type": "string"},
                                "time_span": {"type": "object", "additionalProperties": True},
                                "text": {"type": "string"},
                                "modality": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["node_type", "text"],
                        },
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "edge_id": {"type": "string"},
                                "src": {"type": "string"},
                                "dst": {"type": "string"},
                                "edge_type": {"type": "string"},
                                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                                "text": {"type": "string"},
                            },
                            "required": ["src", "dst", "edge_type"],
                        },
                    },
                    "notes": {"type": "string"},
                },
                "required": ["nodes", "edges", "notes"],
            },
        },
    }


def build_neighbor_vlm_l1_response_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "neighbor_vlm_l1_clip_graph",
            "strict": False,
            "schema": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "target_nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "node_id": {"type": "string"},
                                "node_type": {"type": "string"},
                                "text": {"type": "string"},
                                "modality": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["node_type", "text"],
                        },
                    },
                    "neighbor_edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": True,
                            "properties": {
                                "src_clip_id": {"type": "string"},
                                "dst_clip_id": {"type": "string"},
                                "src_node_id": {"type": "string"},
                                "dst_node_id": {"type": "string"},
                                "edge_type": {"type": "string"},
                                "text": {"type": "string"},
                                "confidence": {"type": "number"},
                            },
                            "required": ["src_clip_id", "dst_clip_id", "edge_type"],
                        },
                    },
                    "reason_short": {"type": "string"},
                },
                "required": ["target_nodes", "neighbor_edges"],
            },
        },
    }


def _neighbor_vlm_l1_worker(job: dict[str, Any]) -> dict[str, Any]:
    client = OpenRouterClient(**job["client_kwargs"])
    response = client.chat_json(job["messages"], response_format=job["response_format"])
    response["model"] = job["client_kwargs"]["model"]
    response["composer"] = "neighbor_vlm_l1_graph_composer"
    response["target_clip_id"] = job["target_clip_id"]
    response["llm_usage"] = client.last_response_metadata
    return response


class GraphComposer:
    def __init__(self, config: GraphComposerConfig, client: OpenRouterClient):
        self.config = config
        self.client = client
        full_ontology = export_skill_ontology()["evidence_graph_construction"]
        executable = set(EXECUTABLE_SKILL_IDS)
        self.ontology = [item for item in full_ontology if item["skill_id"] in executable]
        self.allowed_skill_ids = executable

    @staticmethod
    def _stable_id(prefix: str, *parts: Any) -> str:
        payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
        return f"{prefix}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"

    @staticmethod
    def _clip_digest(schema: dict[str, Any]) -> dict[str, Any]:
        def _strings(items: Any, *, key: str | None = None, limit: int = 5) -> list[str]:
            rows: list[str] = []
            for item in items or []:
                if isinstance(item, str):
                    text = item
                elif isinstance(item, dict) and key:
                    text = str(item.get(key) or "")
                elif isinstance(item, dict):
                    text = str(item.get("text") or item.get("description") or item.get("surface_form") or "")
                else:
                    text = str(item)
                text = " ".join(text.split())
                if text:
                    rows.append(text[:160])
            return rows[:limit]

        objects = []
        for item in schema.get("salient_objects") or []:
            if not isinstance(item, dict):
                continue
            surface = str(item.get("surface_form") or "").strip()
            attrs = ", ".join(str(attr) for attr in item.get("attributes") or [])
            phrase = ", ".join(str(text) for text in item.get("searchable_phrases") or [])
            text = " ".join(part for part in [surface, attrs, phrase] if part).strip()
            if text:
                objects.append(text[:160])

        return {
            "clip_id": schema.get("clip_id"),
            "time_span": schema.get("time_span"),
            "granularity": schema.get("granularity"),
            "scene": str(schema.get("scene_description") or "")[:220],
            "facts": _strings(schema.get("observable_facts"), key="text", limit=6),
            "objects": objects[:6],
            "entities": _strings(schema.get("entity_mentions"), key="surface_form", limit=6),
            "events": _strings(schema.get("events"), key="description", limit=4),
            "visual_social_cues": _strings(schema.get("visual_social_cues"), key="description", limit=4),
            "dialogue": _strings(schema.get("dialogue_spans"), key="text", limit=4),
            "searchable_phrases": _strings(schema.get("searchable_phrases"), limit=6),
            "uncertainty": str(schema.get("uncertainty") or "")[:160],
        }

    def _schema_anchor_text(self, schema: dict[str, Any]) -> str:
        """Build a grounded clip anchor from the VLM clip schema, not labels."""
        digest = self._clip_digest(schema)
        parts: list[str] = []
        scene = str(digest.get("scene") or "").strip()
        if scene:
            parts.append(scene)
        for key in ("facts", "objects", "entities", "events", "visual_social_cues", "dialogue", "searchable_phrases"):
            for value in digest.get(key) or []:
                text = str(value).strip()
                if text and text not in parts:
                    parts.append(text)
        return " | ".join(parts)[:500]

    def _ensure_schema_anchor_node(
        self,
        graph: dict[str, Any],
        *,
        schema: dict[str, Any],
        mode: RuntimeMode,
        primary_node_by_clip: dict[str, str],
    ) -> str | None:
        """Create a VLM-schema endpoint when a clip-level edge needs one."""
        clip_id = str(schema.get("clip_id") or "").strip()
        if not clip_id or schema.get("model_error"):
            return None
        if clip_id in primary_node_by_clip:
            return primary_node_by_clip[clip_id]

        text = self._schema_anchor_text(schema).strip()
        if not text:
            return None

        node_id = self._stable_id("evidence.observation", clip_id, schema.get("time_span"), text, "schema_anchor")
        if not any(node.get("node_id") == node_id for node in graph.get("nodes", [])):
            graph.setdefault("nodes", []).append(
                {
                    "node_id": node_id,
                    "node_type": "observation",
                    "clip_id": clip_id,
                    "time_span": schema.get("time_span"),
                    "text": text,
                    "modality": "mixed",
                    "confidence": 0.72,
                    "source_type": "qwen_clip_schema_anchor",
                    "producer": "neighbor_vlm_l1_schema_anchor",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
        primary_node_by_clip[clip_id] = node_id
        return node_id

    def plan_skill_graph(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        allowed = sorted(self.allowed_skill_ids)
        payload = {
            "task": "compose_clue_memory_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "layer": "clue_memory",
            "clip_policy": clip_policy,
            "clip_schemas": clip_schemas,
            "segments": segments,
            "allowed_skill_ids": allowed,
            "allowed_edge_types": sorted(ALLOWED_MEMORY_EDGES),
            "ontology": self.ontology,
            "instructions": GRAPH_COMPOSE_PROMPT,
        }
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert graph-crafting planner. "
                        "Choose skills only from allowed_skill_ids (multiple choice). "
                        "Never invent skill ids or placeholder node ids."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=build_skill_plan_response_schema(allowed),
        )
        response["model"] = self.config.model
        response["composer"] = "gpt_oss_graph_composer"
        validation_errors = validate_skill_plan(
            response.get("skill_plan") or [],
            allowed_skill_ids=self.allowed_skill_ids,
            clip_schemas=clip_schemas,
        )
        if validation_errors:
            response["validation_errors"] = validation_errors
            response["skill_plan"] = []
            response["notes"] = (
                (response.get("notes") or "")
                + " Plan rejected: skill selection must be multiple-choice from allowed_skill_ids with $step refs."
            ).strip()
        return response

    def execute_skill_plan(
        self,
        *,
        graph: dict[str, Any] | None,
        skill_plan: list[dict[str, Any]],
        bindings: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        graph = graph or {"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []}
        trace: list[dict[str, Any]] = []
        step_outputs: dict[str, Any] = {}

        for step in skill_plan:
            step_id = step.get("step_id")
            skill_id = step.get("skill_id")
            args = dict(step.get("args") or {})
            if skill_id not in SKILL_EXECUTORS:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "unknown_skill_id",
                    }
                )
                continue

            try:
                resolved_args = resolve_plan_value(args, bindings, step_outputs)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "invalid_step_reference",
                        "messages": [str(exc)],
                    }
                )
                continue
            try:
                result = SKILL_EXECUTORS[skill_id](graph, **resolved_args)
            except TypeError as exc:
                trace.append(
                    {
                        "step_id": step_id,
                        "skill_id": skill_id,
                        "ok": False,
                        "failure_code": "invalid_skill_args",
                        "messages": [str(exc)],
                    }
                )
                continue
            graph = result.outputs.get("graph", graph)
            trace.append(
                {
                    "step_id": step_id,
                    "skill_id": skill_id,
                    "ok": result.ok,
                    "failure_code": result.failure_code,
                    "evidence_refs": result.evidence_refs,
                }
            )
            if step_id:
                step_outputs[step_id] = {**result.outputs, "evidence_refs": result.evidence_refs}
        return graph, trace

    def compose_vlm_l1_graph(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        visible_schemas = [schema for schema in clip_schemas if not schema.get("model_error")]
        payload = {
            "task": "compose_video_only_vlm_l1_clue_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "clip_policy": clip_policy,
            "clip_schemas": visible_schemas,
            "segments": segments,
            "allowed_edge_types": sorted(VLM_L1_EDGE_TYPES),
            "instructions": VLM_L1_GRAPH_PROMPT,
        }
        response = self.client.chat_json(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a grounded video clue-graph composer. "
                        "Create semantic L1 nodes and edges only from the supplied clip schemas."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            response_format=build_vlm_l1_response_schema(),
        )
        response["model"] = self.config.model
        response["composer"] = "vlm_l1_graph_composer"
        return response

    def compose_neighbor_vlm_l1_clip(
        self,
        *,
        example_id: str,
        video_id: str,
        target_schema: dict[str, Any],
        neighbor_schemas: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> dict[str, Any]:
        messages, response_format = self._neighbor_vlm_l1_request(
            example_id=example_id,
            video_id=video_id,
            target_schema=target_schema,
            neighbor_schemas=neighbor_schemas,
            mode=mode,
        )
        response = self.client.chat_json(messages, response_format=response_format)
        response["model"] = self.config.model
        response["composer"] = "neighbor_vlm_l1_graph_composer"
        response["target_clip_id"] = target_schema.get("clip_id")
        response["llm_usage"] = getattr(self.client, "last_response_metadata", {})
        return response

    def _neighbor_vlm_l1_request(
        self,
        *,
        example_id: str,
        video_id: str,
        target_schema: dict[str, Any],
        neighbor_schemas: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        payload = {
            "task": "compose_neighbor_vlm_l1_clip_graph",
            "example_id": example_id,
            "video_id": video_id,
            "mode": mode.value,
            "target_clip": self._clip_digest(target_schema),
            "neighbor_clips": [self._clip_digest(schema) for schema in neighbor_schemas if not schema.get("model_error")],
            "allowed_edge_types": sorted(VLM_L1_EDGE_TYPES - {"entity_mention", "derived_from"}),
            "instructions": NEIGHBOR_VLM_L1_PROMPT,
        }
        return [
            {
                "role": "system",
                "content": (
                    "You are a grounded local video clue-graph composer. "
                    "Only create target clip nodes and sparse semantic edges to neighboring clips."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], build_neighbor_vlm_l1_response_schema()

    def _compose_neighbor_vlm_l1_graph(
        self,
        *,
        graph: dict[str, Any],
        example_id: str,
        video_id: str,
        clip_schemas: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
        visible_schemas = [schema for schema in clip_schemas if schema.get("clip_id") and not schema.get("model_error")]
        schema_by_clip = {str(schema.get("clip_id")): schema for schema in visible_schemas if schema.get("clip_id")}
        responses: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = []
        local_id_map: dict[str, str] = {}
        primary_node_by_clip: dict[str, str] = {}
        pending_edges: list[tuple[dict[str, Any], str]] = []

        indexed_jobs: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
        for index, schema in enumerate(visible_schemas):
            neighbors = visible_schemas[max(0, index - 2) : index] + visible_schemas[index + 1 : index + 3]
            indexed_jobs.append((index, schema, neighbors))

        cache_path = Path(self.config.neighbor_cache_path) if self.config.neighbor_cache_path else None
        cached_rows = _read_jsonl(cache_path) if cache_path else []
        cached_by_clip = {
            str(row.get("target_clip_id")): row
            for row in cached_rows
            if isinstance(row, dict)
            and row.get("target_clip_id")
            and row.get("model") == self.config.model
            and row.get("response")
        }

        def _checkpoint_cache(new_rows: list[dict[str, Any]]) -> None:
            if not cache_path:
                return
            merged = dict(cached_by_clip)
            for row in new_rows:
                if row.get("target_clip_id"):
                    merged[str(row["target_clip_id"])] = row
            _write_jsonl(cache_path, list(merged.values()))
            cached_by_clip.clear()
            cached_by_clip.update(merged)

        def _error_response(index: int, target_clip_id: str, exc: Exception) -> tuple[int, dict[str, Any], dict[str, Any]]:
            response = {
                "target_nodes": [],
                "neighbor_edges": [],
                "reason_short": "neighbor VLM L1 clip compose failed",
                "composer": "neighbor_vlm_l1_graph_composer",
                "target_clip_id": target_clip_id,
                "planner_error": str(exc),
                "llm_usage": {
                    "timeout_count": int(isinstance(exc, TimeoutError)),
                    "malformed_json": 0,
                    "cache_hit": False,
                },
            }
            return index, response, {
                "skill_id": "neighbor_vlm_l1_clip_failed",
                "ok": False,
                "clip_id": target_clip_id,
                "failure_code": "api_or_parse_error",
                "messages": [str(exc)],
            }

        def _run_job(job: tuple[int, dict[str, Any], list[dict[str, Any]]]) -> tuple[int, dict[str, Any], dict[str, Any] | None]:
            index, schema, neighbors = job
            target_clip_id = str(schema.get("clip_id"))
            try:
                response = self.compose_neighbor_vlm_l1_clip(
                    example_id=example_id,
                    video_id=video_id,
                    target_schema=schema,
                    neighbor_schemas=neighbors,
                    mode=mode,
                )
                return index, response, None
            except Exception as exc:
                return _error_response(index, target_clip_id, exc)

        indexed_responses: list[tuple[int, dict[str, Any], dict[str, Any] | None]]
        workers = max(1, int(self.config.neighbor_workers or 1))
        indexed_responses = []
        pending_jobs = []
        for job in indexed_jobs:
            index, schema, _neighbors = job
            target_clip_id = str(schema.get("clip_id"))
            cached = cached_by_clip.get(target_clip_id)
            if cached:
                response = dict(cached.get("response") or {})
                usage = dict(response.get("llm_usage") or {})
                usage["cache_hit"] = True
                response["llm_usage"] = usage
                indexed_responses.append((index, response, None))
            else:
                pending_jobs.append(job)

        cache_hit_count = len(indexed_responses)
        if workers == 1 or len(pending_jobs) <= 1:
            cache_updates: list[dict[str, Any]] = []
            for job in pending_jobs:
                index, schema, _neighbors = job
                target_clip_id = str(schema.get("clip_id"))
                result = _run_job(job)
                indexed_responses.append(result)
                _idx, response, error_trace = result
                if not error_trace:
                    cache_updates.append(
                        {
                            "target_clip_id": target_clip_id,
                            "model": self.config.model,
                            "response": response,
                        }
                    )
                    _checkpoint_cache(cache_updates)
        else:
            client_kwargs = {
                "model": self.client.model,
                "api_key": self.client.api_key,
                "api_base": self.client.api_base,
                "temperature": self.client.temperature,
                "max_tokens": self.client.max_tokens,
                "reasoning": self.client.reasoning,
                "timeout_s": self.client.timeout_s,
            }
            cache_updates: list[dict[str, Any]] = []
            with ProcessPoolExecutor(max_workers=min(workers, len(pending_jobs))) as executor:
                future_to_job = {}
                for index, schema, neighbors in pending_jobs:
                    messages, response_format = self._neighbor_vlm_l1_request(
                        example_id=example_id,
                        video_id=video_id,
                        target_schema=schema,
                        neighbor_schemas=neighbors,
                        mode=mode,
                    )
                    future = executor.submit(
                        _neighbor_vlm_l1_worker,
                        {
                            "client_kwargs": client_kwargs,
                            "messages": messages,
                            "response_format": response_format,
                            "target_clip_id": schema.get("clip_id"),
                        },
                    )
                    future_to_job[future] = (index, schema)
                for future in as_completed(future_to_job):
                    index, schema = future_to_job[future]
                    target_clip_id = str(schema.get("clip_id"))
                    try:
                        response = future.result()
                        indexed_responses.append((index, response, None))
                        cache_updates.append(
                            {
                                "target_clip_id": target_clip_id,
                                "model": self.config.model,
                                "response": response,
                            }
                        )
                        _checkpoint_cache(cache_updates)
                    except Exception as exc:
                        indexed_responses.append(_error_response(index, target_clip_id, exc))
        indexed_responses.sort(key=lambda item: item[0])

        for index, response, error_trace in indexed_responses:
            schema = visible_schemas[index]
            target_clip_id = str(schema.get("clip_id"))
            if error_trace:
                trace.append(error_trace)
            responses.append(response)

            target_node_ids: list[str] = []
            response_nodes = response.get("target_nodes") or response.get("nodes") or []
            response_edges = response.get("neighbor_edges") or response.get("edges") or []
            response["target_nodes"] = response_nodes
            response["neighbor_edges"] = response_edges

            for node_index, raw_node in enumerate(response_nodes):
                if not isinstance(raw_node, dict):
                    continue
                text = str(raw_node.get("text") or raw_node.get("description") or "").strip()
                if not text:
                    continue
                raw_node_type = str(raw_node.get("node_type") or "observation").strip() or "observation"
                node_type = raw_node_type if raw_node_type in VLM_L1_NODE_TYPES else "observation"
                local_id = str(raw_node.get("node_id") or f"target_node_{node_index}").strip()
                node_id = self._stable_id("evidence." + node_type, target_clip_id, schema.get("time_span"), text, node_index)
                local_id_map[f"{target_clip_id}:{local_id}"] = node_id
                target_node_ids.append(node_id)
                graph.setdefault("nodes", []).append(
                    {
                        **raw_node,
                        "node_id": node_id,
                        "node_type": node_type,
                        "clip_id": target_clip_id,
                        "time_span": schema.get("time_span"),
                        "text": text,
                        "modality": raw_node.get("modality") or "mixed",
                        "source_type": raw_node.get("source_type") or raw_node_type or "neighbor_vlm_l1",
                        "producer": "neighbor_vlm_l1_graph_composer",
                        "visibility": {"hidden_supervision": False, "mode": mode.value},
                    }
                )
                trace.append({"skill_id": "neighbor_vlm_l1_create_node", "ok": True, "node_id": node_id})

            if target_node_ids:
                primary_node_by_clip[target_clip_id] = target_node_ids[0]
            else:
                anchor = self._ensure_schema_anchor_node(
                    graph,
                    schema=schema,
                    mode=mode,
                    primary_node_by_clip=primary_node_by_clip,
                )
                if anchor:
                    target_node_ids.append(anchor)
                    trace.append(
                        {
                            "skill_id": "neighbor_vlm_l1_create_schema_anchor",
                            "ok": True,
                            "node_id": anchor,
                            "clip_id": target_clip_id,
                            "reason": "no_target_nodes_from_model",
                        }
                    )
            for edge in response_edges:
                if isinstance(edge, dict):
                    pending_edges.append((edge, target_clip_id))

        valid_node_ids = {node.get("node_id") for node in graph.get("nodes", [])}
        for edge_index, (raw_edge, target_clip_id) in enumerate(pending_edges):
            src_clip = str(raw_edge.get("src_clip_id") or target_clip_id)
            dst_clip = str(raw_edge.get("dst_clip_id") or target_clip_id)
            for endpoint_clip in (src_clip, dst_clip):
                if endpoint_clip not in primary_node_by_clip and endpoint_clip in schema_by_clip:
                    anchor = self._ensure_schema_anchor_node(
                        graph,
                        schema=schema_by_clip[endpoint_clip],
                        mode=mode,
                        primary_node_by_clip=primary_node_by_clip,
                    )
                    if anchor:
                        valid_node_ids.add(anchor)
                        trace.append(
                            {
                                "skill_id": "neighbor_vlm_l1_create_schema_anchor",
                                "ok": True,
                                "node_id": anchor,
                                "clip_id": endpoint_clip,
                            }
                        )
            src = local_id_map.get(f"{src_clip}:{raw_edge.get('src_node_id')}", primary_node_by_clip.get(src_clip))
            dst = local_id_map.get(f"{dst_clip}:{raw_edge.get('dst_node_id')}", primary_node_by_clip.get(dst_clip))
            if src not in valid_node_ids or dst not in valid_node_ids or src == dst:
                trace.append(
                    {
                        "skill_id": "neighbor_vlm_l1_skip_edge",
                        "ok": True,
                        "reason": "missing_endpoint",
                        "src_clip_id": src_clip,
                        "dst_clip_id": dst_clip,
                    }
                )
                continue
            edge_type = str(raw_edge.get("edge_type") or "supports_observation").strip()
            if edge_type not in VLM_L1_EDGE_TYPES:
                edge_type = "supports_observation"
            edge_id = self._stable_id("edge", src, dst, edge_type, raw_edge.get("text"), edge_index)
            graph.setdefault("edges", []).append(
                {
                    **raw_edge,
                    "edge_id": edge_id,
                    "src": src,
                    "dst": dst,
                    "edge_type": edge_type,
                    "evidence_refs": [src, dst],
                    "producer": "neighbor_vlm_l1_graph_composer",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
            trace.append({"skill_id": "neighbor_vlm_l1_create_edge", "ok": True, "edge_id": edge_id, "edge_type": edge_type})

        if not any(node.get("producer") == "neighbor_vlm_l1_graph_composer" for node in graph.get("nodes", [])):
            raise ValueError("neighbor VLM L1 composer produced no usable graph nodes")

        plan_payload = {
            "composer": "neighbor_vlm_l1_graph_composer",
            "model": self.config.model,
            "neighbor_workers": workers,
            "neighbor_cache_path": self.config.neighbor_cache_path,
            "neighbor_cache_hits": cache_hit_count,
            "neighbor_cache_misses": len(pending_jobs),
            "llm_budget_summary": _summarize_llm_usage(responses),
            "clip_results": responses,
            "notes": "local target-clip graph construction with neighbor semantic edges",
        }
        return graph, trace, plan_payload

    def _add_short_video_recurrence_clues(
        self,
        *,
        graph: dict[str, Any],
        mode: RuntimeMode,
    ) -> list[dict[str, Any]]:
        """Link repeated VLM-observed objects across non-adjacent short-video clips."""
        trace: list[dict[str, Any]] = []
        nodes = [
            node
            for node in graph.get("nodes", [])
            if node.get("node_type") in VLM_L1_NODE_TYPES and isinstance(node.get("time_span"), dict)
        ]
        existing_edges = {(edge.get("src"), edge.get("dst"), edge.get("edge_type")) for edge in graph.get("edges", [])}
        for group_id, spec in SHORT_RECURRENCE_GROUPS.items():
            terms = tuple(str(term).lower() for term in spec["terms"])
            matches = []
            for node in nodes:
                text = " ".join(
                    str(node.get(key) or "")
                    for key in ("text", "description", "surface_form")
                ).lower()
                if any(term in text for term in terms):
                    matches.append(node)
            matches.sort(key=lambda node: float((node.get("time_span") or {}).get("start_s", 0.0)))
            if len(matches) < 2:
                continue
            early = matches[0]
            late_candidates = [
                node
                for node in matches[1:]
                if float((node.get("time_span") or {}).get("start_s", 0.0))
                - float((early.get("time_span") or {}).get("end_s", 0.0))
                >= 12.0
            ]
            if not late_candidates:
                continue
            late = late_candidates[-1]
            edge_type = str(spec["edge_type"])
            if (early.get("node_id"), late.get("node_id"), edge_type) not in existing_edges:
                edge_id = self._stable_id("edge", early.get("node_id"), late.get("node_id"), edge_type, group_id)
                graph.setdefault("edges", []).append(
                    {
                        "edge_id": edge_id,
                        "src": early.get("node_id"),
                        "dst": late.get("node_id"),
                        "edge_type": edge_type,
                        "evidence_refs": [early.get("node_id"), late.get("node_id")],
                        "text": (
                            f"The {spec['label']} appears in an earlier clip and reappears later, "
                            "linking non-adjacent moments in the same short video."
                        ),
                        "producer": "short_video_recurrence_linker",
                        "visibility": {"hidden_supervision": False, "mode": mode.value},
                    }
                )
                existing_edges.add((early.get("node_id"), late.get("node_id"), edge_type))
                trace.append({"skill_id": "short_video_recurrence_link", "ok": True, "edge_id": edge_id})

            clue_text = (
                f"Repeated {spec['label']} clue: it is visible around "
                f"{float((early.get('time_span') or {}).get('start_s', 0.0)):.1f}s and again around "
                f"{float((late.get('time_span') or {}).get('start_s', 0.0)):.1f}s, suggesting the later moment "
                "returns to or echoes a previously seen place/object."
            )
            clue_id = self._stable_id("evidence.clue", group_id, early.get("node_id"), late.get("node_id"), clue_text)
            if not any(node.get("node_id") == clue_id for node in graph.get("nodes", [])):
                graph.setdefault("nodes", []).append(
                    {
                        "node_id": clue_id,
                        "node_type": "clue",
                        "time_span": {
                            "start_s": float((early.get("time_span") or {}).get("start_s", 0.0)),
                            "end_s": float((late.get("time_span") or {}).get("end_s", 0.0)),
                        },
                        "text": clue_text,
                        "source_type": "short_video_recurrence_clue",
                        "producer": "short_video_recurrence_linker",
                        "visibility": {"hidden_supervision": False, "mode": mode.value},
                    }
                )
                trace.append({"skill_id": "short_video_recurrence_create_clue", "ok": True, "node_id": clue_id})
            for source in (early, late):
                edge_id = self._stable_id("edge", clue_id, source.get("node_id"), "supports_observation")
                if (clue_id, source.get("node_id"), "supports_observation") in existing_edges:
                    continue
                graph.setdefault("edges", []).append(
                    {
                        "edge_id": edge_id,
                        "src": clue_id,
                        "dst": source.get("node_id"),
                        "edge_type": "supports_observation",
                        "evidence_refs": [clue_id, source.get("node_id")],
                        "text": "The recurrence clue is grounded in this observed clip evidence.",
                        "producer": "short_video_recurrence_linker",
                        "visibility": {"hidden_supervision": False, "mode": mode.value},
                    }
                )
                existing_edges.add((clue_id, source.get("node_id"), "supports_observation"))
        return trace

    def _graph_from_vlm_l1_response(
        self,
        *,
        response: dict[str, Any],
        base_graph: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        mode: RuntimeMode,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        graph = base_graph
        clip_by_id = {schema.get("clip_id"): schema for schema in clip_schemas if schema.get("clip_id")}
        existing_node_ids = {node.get("node_id") for node in graph.get("nodes", [])}
        id_map: dict[str, str] = {}

        for index, raw_node in enumerate(response.get("nodes") or []):
            if not isinstance(raw_node, dict):
                continue
            text = str(raw_node.get("text") or raw_node.get("description") or "").strip()
            if not text:
                continue
            raw_node_type = str(raw_node.get("node_type") or "observation").strip() or "observation"
            node_type = raw_node_type if raw_node_type in VLM_L1_NODE_TYPES else "observation"
            clip_id = str(raw_node.get("clip_id") or "").strip()
            schema = clip_by_id.get(clip_id) or {}
            time_span = raw_node.get("time_span") if isinstance(raw_node.get("time_span"), dict) else schema.get("time_span")
            proposed_id = str(raw_node.get("node_id") or "").strip()
            node_id = proposed_id if proposed_id and proposed_id not in existing_node_ids else self._stable_id(
                f"evidence.{node_type}",
                clip_id,
                time_span,
                text,
                index,
            )
            id_map[proposed_id] = node_id
            existing_node_ids.add(node_id)
            graph.setdefault("nodes", []).append(
                {
                    **raw_node,
                    "node_id": node_id,
                    "node_type": node_type,
                    "text": text,
                    "clip_id": clip_id or schema.get("clip_id"),
                    "time_span": time_span,
                    "modality": raw_node.get("modality") or "mixed",
                    "source_type": raw_node.get("source_type") or raw_node_type or "vlm_l1",
                    "producer": "vlm_l1_graph_composer",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
            trace.append({"skill_id": "vlm_l1_create_node", "ok": True, "node_id": node_id})

        valid_node_ids = {node.get("node_id") for node in graph.get("nodes", [])}
        for index, raw_edge in enumerate(response.get("edges") or []):
            if not isinstance(raw_edge, dict):
                continue
            src = id_map.get(str(raw_edge.get("src") or ""), raw_edge.get("src"))
            dst = id_map.get(str(raw_edge.get("dst") or ""), raw_edge.get("dst"))
            if src not in valid_node_ids or dst not in valid_node_ids:
                trace.append(
                    {
                        "skill_id": "vlm_l1_skip_edge",
                        "ok": True,
                        "reason": "missing_endpoint",
                        "src": src,
                        "dst": dst,
                    }
                )
                continue
            edge_type = str(raw_edge.get("edge_type") or "supports_observation").strip()
            if edge_type not in VLM_L1_EDGE_TYPES:
                edge_type = "supports_observation"
            refs = [
                id_map.get(str(ref), str(ref))
                for ref in raw_edge.get("evidence_refs") or [src, dst]
                if id_map.get(str(ref), str(ref)) in valid_node_ids
            ]
            edge_id = str(raw_edge.get("edge_id") or "").strip() or self._stable_id(
                "edge",
                src,
                dst,
                edge_type,
                index,
            )
            graph.setdefault("edges", []).append(
                {
                    **raw_edge,
                    "edge_id": edge_id,
                    "src": src,
                    "dst": dst,
                    "edge_type": edge_type,
                    "evidence_refs": refs or [src, dst],
                    "producer": "vlm_l1_graph_composer",
                    "visibility": {"hidden_supervision": False, "mode": mode.value},
                }
            )
            trace.append({"skill_id": "vlm_l1_create_edge", "ok": True, "edge_id": edge_id, "edge_type": edge_type})

        if not any(node.get("producer") == "vlm_l1_graph_composer" for node in graph.get("nodes", [])):
            raise ValueError("VLM L1 composer produced no usable graph nodes")
        return graph, trace

    def compose_from_clip_schemas(
        self,
        *,
        example_id: str,
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        mode: RuntimeMode,
        duration_s: float,
        observation_end_s: float | None = None,
    ) -> dict[str, Any]:
        composer_mode = self.config.composer_mode
        if not self.config.use_llm_planner:
            composer_mode = "deterministic"

        plan_payload: dict[str, Any]
        if composer_mode == "neighbor_vlm_l1":
            plan_payload = {"clip_results": [], "notes": "neighbor_vlm_l1 graph compose not attempted"}
            skill_plan = []
        elif composer_mode == "vlm_l1":
            plan_payload = {"nodes": [], "edges": [], "notes": "vlm_l1 graph compose not attempted"}
            skill_plan = []
        elif composer_mode == "skill_plan":
            try:
                plan_payload = self.plan_skill_graph(
                    example_id=example_id,
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    mode=mode,
                )
                skill_plan = plan_payload.get("skill_plan") or []
            except Exception as exc:
                plan_payload = {
                    "skill_plan": [],
                    "notes": "planner failed; deterministic fallback used",
                    "planner_error": str(exc),
                    "model": self.config.model,
                    "composer": "gpt_oss_graph_composer",
                }
                skill_plan = []
        else:
            plan_payload = {"skill_plan": [], "notes": "deterministic debug composer"}
            skill_plan = []

        bindings = {
            "video_id": video_id,
            "clip_policy": {**clip_policy, "duration_s": duration_s},
            "observation_end_s": observation_end_s,
            "mode": mode.value,
        }

        graph: dict[str, Any] = {"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []}
        trace: list[dict[str, Any]] = []
        used_deterministic = False

        if composer_mode in {"neighbor_vlm_l1", "vlm_l1"}:
            seg_result = segment_video_or_select_clip(
                graph,
                video_id=video_id,
                clip_policy={**clip_policy, "duration_s": duration_s},
                observation_end_s=observation_end_s,
            )
            graph = seg_result.outputs["graph"]
            trace.append(
                {
                    "skill_id": "segment_video_or_select_clip",
                    "ok": seg_result.ok,
                    "source": "vlm_l1_plumbing",
                }
            )
            if composer_mode == "neighbor_vlm_l1":
                try:
                    graph, neighbor_trace, plan_payload = self._compose_neighbor_vlm_l1_graph(
                        graph=graph,
                        example_id=example_id,
                        video_id=video_id,
                        clip_schemas=clip_schemas,
                        mode=mode,
                    )
                    trace.extend(neighbor_trace)
                except Exception as exc:
                    plan_payload = {
                        "clip_results": [],
                        "notes": "neighbor VLM L1 composer failed; deterministic debug fallback used",
                        "planner_error": str(exc),
                        "model": self.config.model,
                        "composer": "neighbor_vlm_l1_graph_composer",
                    }
                    graph, deterministic_trace = self._compose_deterministically(
                        graph={"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []},
                        video_id=video_id,
                        clip_policy=clip_policy,
                        clip_schemas=clip_schemas,
                        segments=segments,
                        duration_s=duration_s,
                        observation_end_s=observation_end_s,
                        mode=mode,
                    )
                    trace.append(
                        {
                            "skill_id": "deterministic_fallback",
                            "ok": True,
                            "reason": "neighbor_vlm_l1_failed",
                        }
                    )
                    trace.extend(deterministic_trace)
                    used_deterministic = True
            else:
                try:
                    plan_payload = self.compose_vlm_l1_graph(
                        example_id=example_id,
                        video_id=video_id,
                        clip_policy=clip_policy,
                        clip_schemas=clip_schemas,
                        segments=segments,
                        mode=mode,
                    )
                    graph, vlm_trace = self._graph_from_vlm_l1_response(
                        response=plan_payload,
                        base_graph=graph,
                        clip_schemas=clip_schemas,
                        mode=mode,
                    )
                    trace.extend(vlm_trace)
                except Exception as exc:
                    plan_payload = {
                        "nodes": [],
                        "edges": [],
                        "notes": "VLM L1 composer failed; deterministic debug fallback used",
                        "planner_error": str(exc),
                        "model": self.config.model,
                        "composer": "vlm_l1_graph_composer",
                    }
                    graph, deterministic_trace = self._compose_deterministically(
                        graph={"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []},
                        video_id=video_id,
                        clip_policy=clip_policy,
                        clip_schemas=clip_schemas,
                        segments=segments,
                        duration_s=duration_s,
                        observation_end_s=observation_end_s,
                        mode=mode,
                    )
                    trace.append(
                        {
                            "skill_id": "deterministic_fallback",
                            "ok": True,
                            "reason": "vlm_l1_failed",
                        }
                    )
                    trace.extend(deterministic_trace)
                    used_deterministic = True
            if clip_policy.get("strategy") == "whole_video":
                trace.extend(self._add_short_video_recurrence_clues(graph=graph, mode=mode))
        elif skill_plan:
            graph, trace = self.execute_skill_plan(graph=graph, skill_plan=skill_plan, bindings=bindings)
            failed_steps = [step for step in trace if step.get("ok") is False]
            successful_graph_steps = [
                step
                for step in trace
                if step.get("ok") and step.get("skill_id") != "segment_video_or_select_clip"
            ]
            if failed_steps or not successful_graph_steps:
                graph, deterministic_trace = self._compose_deterministically(
                    graph={"schema_version": "video-skills-relaunch/v0.1", "nodes": [], "edges": []},
                    video_id=video_id,
                    clip_policy=clip_policy,
                    clip_schemas=clip_schemas,
                    segments=segments,
                    duration_s=duration_s,
                    observation_end_s=observation_end_s,
                    mode=mode,
                )
                trace.append(
                    {
                        "skill_id": "deterministic_fallback",
                        "ok": True,
                        "reason": "llm_plan_invalid_or_failed",
                        "failed_steps": len(failed_steps),
                    }
                )
                trace.extend(deterministic_trace)
                used_deterministic = True
        else:
            graph, deterministic_trace = self._compose_deterministically(
                graph=graph,
                video_id=video_id,
                clip_policy=clip_policy,
                clip_schemas=clip_schemas,
                segments=segments,
                duration_s=duration_s,
                observation_end_s=observation_end_s,
                mode=mode,
            )
            trace.extend(deterministic_trace)
            used_deterministic = True

        return {
            "graph": graph,
            "skill_plan": plan_payload,
            "execution_trace": trace,
            "composer_model": self.config.model,
            "composer_mode": composer_mode,
            "used_deterministic_fallback": used_deterministic,
        }

    def _compose_deterministically(
        self,
        *,
        graph: dict[str, Any],
        video_id: str,
        clip_policy: dict[str, Any],
        clip_schemas: list[dict[str, Any]],
        segments: list[dict[str, Any]],
        duration_s: float,
        observation_end_s: float | None,
        mode: RuntimeMode,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        trace: list[dict[str, Any]] = []
        seg_result = segment_video_or_select_clip(
            graph,
            video_id=video_id,
            clip_policy={**clip_policy, "duration_s": duration_s},
            observation_end_s=observation_end_s,
        )
        graph = seg_result.outputs["graph"]
        trace.append({"skill_id": "segment_video_or_select_clip", "ok": seg_result.ok})

        clip_ref_by_id = {node["node_id"]: node for node in seg_result.outputs.get("clip_nodes", [])}
        clip_nodes = list(clip_ref_by_id.keys())
        for schema in clip_schemas:
            if schema.get("model_error"):
                trace.append(
                    {
                        "skill_id": "skip_failed_clip_schema",
                        "ok": True,
                        "clip_id": schema.get("clip_id"),
                        "source": "model_error",
                    }
                )
                continue
            clip_id = schema.get("clip_id")
            clip_ref = clip_id if clip_id in clip_ref_by_id else (clip_nodes[0] if clip_nodes else clip_id)
            scene = schema.get("scene_description") or ""
            if scene:
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="visual_caption",
                    text=scene,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "scene_description"})
                mentions = detect_entity_mention(graph, observation_ref=obs.evidence_refs[0])
                graph = mentions.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mentions.ok})

            for fact in schema.get("observable_facts") or []:
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality=fact.get("modality", "visual"),
                    text=fact.get("text", ""),
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "observable_fact"})

            for obj in schema.get("salient_objects") or []:
                if not isinstance(obj, dict):
                    continue
                attributes = ", ".join(str(item) for item in obj.get("attributes") or [])
                phrases = ", ".join(str(item) for item in obj.get("searchable_phrases") or [])
                text = " ".join(
                    part
                    for part in [
                        str(obj.get("surface_form") or "").strip(),
                        attributes,
                        phrases,
                    ]
                    if part
                )
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="object_clue",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "salient_object"})
                mention = detect_entity_mention(
                    graph,
                    observation_ref=obs.evidence_refs[0],
                    text=str(obj.get("surface_form") or text),
                    entity_type="object",
                )
                graph = mention.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mention.ok, "source": "salient_object"})

            place = schema.get("place") or {}
            if isinstance(place, dict):
                phrases = place.get("searchable_phrases") or []
                text = " ".join(
                    part
                    for part in [
                        str(place.get("description") or "").strip(),
                        " ".join(str(item) for item in phrases),
                    ]
                    if part
                )
                if text:
                    obs = extract_observation(
                        graph,
                        clip_or_text_ref=clip_ref,
                        modality="place_clue",
                        text=text,
                        time_span=schema.get("time_span"),
                    )
                    graph = obs.outputs["graph"]
                    trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "place"})

            for phrase in schema.get("searchable_phrases") or []:
                text = str(phrase).strip()
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="searchable_phrase",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "searchable_phrase"})

            for cue in schema.get("cross_clip_cues") or []:
                if not isinstance(cue, dict):
                    continue
                text = " ".join(
                    part
                    for part in [
                        str(cue.get("cue_type") or "").strip(),
                        str(cue.get("description") or "").strip(),
                    ]
                    if part
                )
                if not text:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="cross_clip_cue",
                    text=text,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "cross_clip_cue"})

            for mention_payload in schema.get("entity_mentions") or []:
                if not isinstance(mention_payload, dict):
                    continue
                surface = str(mention_payload.get("surface_form") or "").strip()
                if not surface:
                    continue
                obs = extract_observation(
                    graph,
                    clip_or_text_ref=clip_ref,
                    modality="entity_schema",
                    text=surface,
                    time_span=schema.get("time_span"),
                )
                graph = obs.outputs["graph"]
                trace.append({"skill_id": "extract_observation", "ok": obs.ok, "source": "entity_schema"})
                mention = detect_entity_mention(
                    graph,
                    observation_ref=obs.evidence_refs[0],
                    text=surface,
                    entity_type=mention_payload.get("entity_type"),
                )
                graph = mention.outputs["graph"]
                trace.append({"skill_id": "detect_entity_mention", "ok": mention.ok, "source": "entity_schema"})

            for dialogue in schema.get("dialogue_spans") or []:
                dia = extract_dialogue_span(
                    graph,
                    subtitle_or_asr_ref=clip_ref,
                    text=dialogue.get("text", ""),
                    time_span=dialogue.get("time_span") or schema.get("time_span") or {"start_s": 0.0, "end_s": 1.0},
                    speaker_hint=dialogue.get("speaker"),
                )
                graph = dia.outputs["graph"]
                trace.append({"skill_id": "extract_dialogue_span", "ok": dia.ok})

            for event in schema.get("events") or []:
                obs_refs = [node["node_id"] for node in graph.get("nodes", []) if node.get("node_type") == "observation"][-1:]
                if not obs_refs:
                    continue
                ev = create_event_node(
                    graph,
                    observation_refs=obs_refs[:1],
                    event_description=event.get("description", ""),
                    time_span=event.get("time_span") or schema.get("time_span") or {"start_s": 0.0, "end_s": 1.0},
                )
                graph = ev.outputs["graph"]
                trace.append({"skill_id": "create_event_node", "ok": ev.ok})

        event_nodes = [node for node in graph.get("nodes", []) if node.get("node_type") == "event" and node.get("time_span")]
        event_nodes.sort(key=lambda node: node["time_span"]["start_s"])
        for left, right in zip(event_nodes, event_nodes[1:]):
            rel = link_graph_relation(
                graph,
                source_node=left["node_id"],
                target_node=right["node_id"],
                edge_type="temporal_next",
                evidence_refs=[left["node_id"], right["node_id"]],
            )
            graph = rel.outputs["graph"]
            trace.append({"skill_id": "link_graph_relation", "ok": rel.ok})

        mentions_by_surface: dict[str, list[dict[str, Any]]] = {}
        for node in graph.get("nodes", []):
            if node.get("node_type") != "entity_mention":
                continue
            surface = str(node.get("surface_form") or node.get("text") or "").strip().lower()
            if len(surface) < 2:
                continue
            mentions_by_surface.setdefault(surface, []).append(node)
        for mentions in mentions_by_surface.values():
            mentions.sort(key=lambda node: (node.get("time_span") or {}).get("start_s", 0.0))
            for left, right in zip(mentions, mentions[1:]):
                if left["node_id"] == right["node_id"]:
                    continue
                rel = link_graph_relation(
                    graph,
                    source_node=left["node_id"],
                    target_node=right["node_id"],
                    edge_type="same_entity",
                    evidence_refs=[left["node_id"], right["node_id"]],
                )
                graph = rel.outputs["graph"]
                trace.append({"skill_id": "link_graph_relation", "ok": rel.ok, "source": "same_entity_surface"})

        trust_policy = {
            "gold_sources": ["segment_description", "inference_shot", "clue_interval", "reasoning_process_step"],
            "strong_sources": ["video_summary", "subtitle_span"],
            "weak_sources": [],
            "model_labeled_sources": ["visual_caption", "scene_description"],
        }
        for node in graph.get("nodes", []):
            if node.get("node_type") in {"observation", "event", "dialogue_span"}:
                prov = assign_provenance_trust(
                    graph,
                    node_or_edge_ref=node["node_id"],
                    source_ref=node.get("modality") or node.get("source_type") or "model_labeled_span",
                    mode=mode.value,
                    trust_policy=trust_policy,
                )
                graph = prov.outputs["graph"]
        trace.append({"skill_id": "assign_provenance_trust", "ok": True})
        return graph, trace
