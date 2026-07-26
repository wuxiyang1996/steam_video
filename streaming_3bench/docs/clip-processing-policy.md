# Clip Processing Policy

Last updated: 2026-07-01

This document is the canonical reference for how short, long, and streaming
videos are segmented before evidence indexing and skill-graph reasoning. It
consolidates design decisions from `unified-video-skill-schema.md`, atomic
skills v1, and prior project discussions.

## 1. Core Pipeline

All video regimes share the same evidence interface. Only the clip policy and
the thickness of the evidence layer change:

```text
video
  -> clip segmentation (clip_policy)
  -> per-clip captions / ASR / objects / events
  -> evidence index / clue-memory graph
  -> agent-composed SkillGraphRollout
  -> verifier checks cited evidence supports the answer
```

The reasoning layer (`SkillGraphRollout`) stays stable across short, long, and
streaming settings. What changes is how `EvidenceGraph` / `evidence_index` is
built.

## 2. Clip Policy Schema

Persisted on `CanonicalVideoExample.evidence_index.clip_policy` and passed to
the atomic skill `segment_video_or_select_clip`.

| Field | Type | Meaning |
|-------|------|---------|
| `strategy` | enum | `whole_video`, `fixed_window`, `hierarchical`, `shot_boundary`, `scene_boundary`, `adaptive` |
| `window_s` | number | Sliding window length in seconds |
| `overlap_s` | number | Overlap between adjacent windows; must be `< window_s` |
| `coarse_window_s` | number | Coarse retrieval window for `hierarchical` (default **30s**, M3-aligned) |
| `fine_window_s` | number | Fine evidence window inside coarse candidates |
| `index_fine_expansion` | enum | `retrieval_gated` (default long), `none` (coarse index only), `all` (legacy full expand) |
| `online` | boolean | When true, enforce causal visibility (streaming) |
| `observation_end_s` | number | Latest observable time `t`; clips with `end_s > t` are hidden |

JSON schema: `schemas/canonical_video_example.schema.json`.

## 3. Regime Defaults

### 3.1 Short Video

Use when the full video can fit in context or needs only light indexing.

| Item | Recommendation |
|------|----------------|
| Primary strategy | `whole_video` plus small `fixed_window` clips |
| Window size | 2–5s with light overlap; MVP default `window_s=4`, `overlap_s=1` |
| Evidence layer | Lightweight: whole-video clip, subtitles, captions, entities |
| Visibility | All clips retrievable (`online=false`) |

Example:

```json
{
  "strategy": "whole_video",
  "window_s": 4,
  "overlap_s": 1,
  "online": false,
  "observation_end_s": null
}
```

Typical datasets: Video-Holmes, SIV-Bench.

### 3.2 Long Video

Use when full-video reasoning cost is too high.

| Item | Recommendation |
|------|----------------|
| Primary strategy | `hierarchical` |
| Coarse windows | 30–60s for retrieval and coarse summaries |
| Fine windows | 5–10s inside top retrieval candidates |
| Evidence layer | Rich M3-style index: fixed-window clips, episodic/semantic text, entity links |
| Visibility | All clips retrievable (`online=false`) |

Example:

```json
{
  "strategy": "hierarchical",
  "coarse_window_s": 30,
  "fine_window_s": 8,
  "overlap_s": 2,
  "online": false,
  "observation_end_s": null
}
```

Long-video flow:

```text
full-video coarse graph
  -> coarse fixed_window(30-60s)
  -> baseline video_only: question-blind coarse neighborhoods
  -> query-time QA: visible question / timestamp anchors retrieve coarse neighborhoods
  -> fine graph inside selected coarse parents
  -> fine fixed_window(5-10s)
  -> fine --refines--> parent coarse
  -> final EvidenceCandidate must cite a concrete timestamp
```

For CG-Bench and VRBench, "entire video" means the coarse graph covers the full
video. Fine-grained graph crafting is intentionally retrieve-gated; full
fine-window expansion over every 8s span is a stress test, not the default
video-only path.

### 3.2.1 L1 Quality Fix: Tool / Prompt / Agent Split

Recent L1 query-memory probes showed that a structurally valid graph is not
automatically a useful reasoning memory. The failing pattern is:

```text
generic clip caption
  -> shallow observation nodes
  -> lexical retrieval misses the question clue
  -> L2 planner guesses from option text or falls back to weak evidence
```

The repair is a three-part split:

| Layer | Required refinement |
|-------|---------------------|
| Tool set | Object/place cues, time anchors, ASR/subtitle cues, coarse summaries, repeated-object links |
| Prompt | Graph-ready observations, salient objects, places, cross-clip cues, searchable phrases, uncertainty |
| Agent setting | Question-blind L1 building separated from query-time memory retrieval and no-gold-answer L2 reasoning |

The default `video_only` L1 pass remains question-blind. QA experiments may
turn on query-time retrieval because the question is visible input, but hidden
clue intervals, official reasoning, and official answers remain unavailable.
Timestamp anchors in the visible question, such as `2:04`, can force expansion
of the matching coarse window and adjacent context.

Typical datasets: CG-Bench, VRBench, M3-Bench.

### 3.3 Streaming Video

Use for realtime or partial-video QA where future frames must stay hidden.

| Item | Recommendation |
|------|----------------|
| Primary strategy | `fixed_window` or `hierarchical` with `online=true` |
| Window size | Share MVP default `window_s=4`, `overlap_s=1` with short video |
| Low-latency option | `window_s=2`, `overlap_s=0.5–1` |
| Visibility rule | Only clips with `time_span.end_s <= observation_end_s` |
| Hard invariant | Every cited evidence span must satisfy the visibility rule |

Example:

```json
{
  "strategy": "fixed_window",
  "window_s": 5,
  "overlap_s": 1,
  "online": true,
  "observation_end_s": 32.0
}
```

Streaming QA mapping (e.g. OVO-Bench / StreamBridge-style records):

```text
video: full video path
question: QA at timestamp t
observation_window: [0, t]
hidden_future: (t, video_end]
evidence_candidates: only spans with end_s <= t
task_family: streaming_realtime_qa
```

Paper-1 scope note: streaming is supported in schema and clip policy, but the
first paper intentionally defers streaming memory update / writer-reasoner
split experiments to a later phase. Offline short/long QA is the default MVP.

## 4. Benchmark-Specific Presets

Recommended `clip_policy` and index mode per local dataset. Current verified
cluster paths are listed in
[cluster dataset inventory](cluster-dataset-inventory.md); the default adapter
root is `/mnt/is_data/xwu/video_skills/data/datasets`.

| Dataset | Length regime | Recommended `clip_policy` | Index mode | Notes |
|---------|---------------|---------------------------|------------|-------|
| Video-Holmes | Short | `whole_video` + `fixed_window(4s, 1s overlap)` | Lightweight | Strong segment/inference annotations often seed graph offline |
| SIV-Bench | Very short | `whole_video` + subtitle-aligned spans | Lightweight | Weak evidence; model-labeled spans |
| CG-Bench | Medium/long | `hierarchical(30s coarse, retrieve-gated 8s fine)` | Rich retrieval | Coarse index in `evidence_index`; fine only after `retrieve_coarse_clips` top-k |
| VRBench | Long | `hierarchical(30s coarse, retrieve-gated 8s fine)` | Rich retrieval | Same M3-style gate; timestamped `reasoning_process` in expert_demo |
| M3-Bench | Long + memory graph | `fixed_window(30s)` or M3 graph clips | Rich M3-style | Deferred until memory graph reader exists |

### Legacy `Video_Skills` mapping

The earlier `Video_Skills/visual_grounding` repo used per-benchmark segmentation
presets that align with this policy:

| Legacy `segmentation` | Relaunch `clip_policy` | Legacy benchmarks |
|----------------------|------------------------|-------------------|
| `scene` | `whole_video` + scene-change windows (~5s) | Video-Holmes |
| `subtitle` | subtitle-aligned `fixed_window` | SIV-Bench |
| `long_hierarchical` | `hierarchical` coarse + fine | VRBench, CG-Bench, M3-Bench |
| `fixed` | `fixed_window` | generic short clips |

Legacy defaults in `Video_Skills/visual_grounding/segmenter.py`:

- Short: `DEFAULT_WINDOW_SECONDS_SHORT = 5.0`, fps `0.5` (about 1 frame / 2s)
- Long: `DEFAULT_WINDOW_SECONDS_LONG = 15.0`, fps `0.2` (about 1 frame / 5s)

The relaunch MVP prefers the shared `4s / 1s overlap` default for simpler
ablations across short and streaming regimes.

## 5. M3-Agent Borrowing

For long-video evidence indexing, borrow from M3-Agent:

```text
video
  -> 30s clips (or configured fixed_window)
  -> face detection + speaker diarization
  -> episodic memory per clip
  -> semantic memory per clip/entity
  -> multimodal graph
  -> iterative search/answer control loop
```

Map into our schema as:

| M3 object | Our target |
|-----------|------------|
| 30s clip | `derived_clips`, `source_type=video_segment` |
| Episodic memory | `caption_span` evidence |
| Semantic memory | `model_labeled_span` (lower trust unless verified) |
| Face/voice node | `entities[]` with modality |
| Retrieval score | `provenance.retrieval_score` (ranking only, not verification) |

Do **not** use unrestricted semantic memory or cross-video persistent facts as
final answer evidence without lower-level clip support.

## 6. Atomic Skill Entry Point

Graph construction starts with:

```text
segment_video_or_select_clip(video_id, clip_policy, observation_end_s?)
  -> clip_nodes, time_spans
```

Followed by `extract_observation`, `extract_dialogue_span`, entity/event/state
nodes, and `link_graph_relation`.

For the first `expert_demo` pass, graph construction is an **offline graph
builder**. The controller-visible action set is Reasoning Graph Assembly Skills
only. Selected graph-construction skills become tool-mediated actions in
`video_only` Stage C.

## 7. expert_demo vs video_only

| Mode | Clip/index inputs | GT clues visible? |
|------|-------------------|-------------------|
| `expert_demo` | video, subtitles, captions, annotations, clue intervals | Yes for labeling; provenance must be explicit |
| `video_only` | video, automatic clips/captions, tool-produced evidence | No; GT clues become hidden supervision |

In `video_only`, citing a gold clue interval without rediscovering it through
retrieval/segmentation is a leakage failure.

## 8. Implementation Status

| Capability | Doc status | Code status |
|------------|------------|-------------|
| `whole_video` | Documented | Implemented in `atomic_skills/evidence_graph_construction/skills.py` |
| `fixed_window` | Documented | Implemented |
| `online` + `observation_end_s` | Documented | Implemented (causal clip filtering) |
| `hierarchical` coarse→fine | Documented | **Implemented**: coarse index + `retrieve_coarse_clips` top-k + fine inside selected parents (`clip_retrieval.py`, `llm_pipeline.py`) |
| `shot_boundary` / `scene_boundary` / `adaptive` | Schema enum only | Not implemented in relaunch code |
| Legacy `long_hierarchical` segmenter | Mapped above | Exists in `Video_Skills/visual_grounding/segmenter.py`, not ported |
| Local raw-video tool backend | Stage C seed | Implemented as optional `--clip-schema-backend video_tools`: frame sampling, frame-change signals, optional OCR; no detector/tracker/ASR yet |
| Raw VLM caption / ASR / detector perception | Planned Stage C | Partial: Qwen clip-schema backend and local video tools exist; full ASR/object/tracking stack not implemented |

## 9. Verifier Invariants

Regardless of regime:

1. Final answers cite `EvidenceCandidate` records, not the raw index alone.
2. Streaming/partial QA: `time_span.end_s <= observation_end_s` for every cited span.
3. Semantic summaries may guide retrieval but final commits need clip/caption/subtitle support unless the task explicitly allows summary-level evidence.
4. Retrieval scores rank candidates; they do not prove answer support.

## 10. Related Documents

- [Unified video skill schema](unified-video-skill-schema.md) §4.1, §5.4, §15
- [Atomic skills v1](../atomic-skill-decomposition-and-assembly/atomic-skills-v1.md)
- [Expert demo rollouts from datasets](../atomic-skill-decomposition-and-assembly/expert-demo-rollouts-from-datasets.md)
- [Implementation status](implementation-status.md)
