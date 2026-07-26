# STEAM Video Memory Graph

This repository develops an evidence-grounded memory graph for long-video
understanding and bounded graph navigation. The active implementation lives in
`memory_graph/`; the IWM-guided navigation research path lives in
`steam_video_new/`. `legacy/` and older Video-Holmes artifacts remain
historical or engineering-only.

**Primary testbed: CG-Bench.** Supervision, frozen cohorts, matched IWM/planner
arms, and held-out gates use CG-Bench multi-clue cases with human
`clue_intervals`. Video-Holmes remains only as a historical graph-construction
and edge-audit smoke; it is not the active evaluation protocol.

The implemented graph is deliberately not presented as a fully validated
causal graph. It is a two-layer temporal-state memory graph with a sparse,
strictly gated dependency overlay.

## Architecture

```text
Video_Skills canonical L1 graph
  |
  +-- L1 observations and native structural candidates
  |     event / entity mention / state evidence
  |
  +-- L1.5 atomic-event overlay
        +-- grounded participants and visible states
        +-- deterministic temporal relations
        +-- conflict-aware identity tracks
        +-- same-track state transitions
        +-- Qwen event embedding references
                         |
                         +-- bounded graph navigation
```

The layers have separate node-ID spaces and responsibilities:

- **L1 is immutable evidence.** Native identity and support edges may guide a
  read, but candidate edges are not automatically answer evidence or causal
  claims.
- **L1.5 contains atomic event hypotheses.** Every event points back to its L1
  evidence through `source_segments`.
- **Identity is conflict-aware.** Track construction rejects entity-type,
  stable-attribute, simultaneous-instance, and impossible-motion conflicts.
- **State transitions are strict.** A transition requires explicit grounded
  states with the same normalized attribute on the same accepted identity
  track and a real before/after value change.
- **Causality remains gated.** Temporal succession, native L1 labels, and
  embedding similarity are not sufficient to admit `explains` or `enables`.

## Embeddings

Atomic events retain an `embedding_ref` generated with
`Qwen/Qwen3-VL-Embedding-2B`:

- 2048 dimensions, `float32`, L2 normalized;
- text contract `event+participants+states/v1`;
- `.npy` matrix path and row index;
- model, dimension, dtype, normalization flag, and SHA-256 checksum.

Vectors stay in matrix files rather than being duplicated in graph JSON.
Navigation questions are encoded with the same model and receive equivalent
query references and manifests. This contract is intended to support a future
persistent vector index and learned navigation without regenerating the graph.

## Current CG-Bench status

CG-Bench is the locked navigation and IWM testbed:

| Item | Value |
| --- | --- |
| Pilot protocol | `cgbench-gt-navigation-pilot-v2` |
| Case pool | 256 video-disjoint multi-clue cases / 205 videos |
| GT clue transitions | 672 (670/672 successful Qwen-VL grounding reads) |
| Formal labels | human `clue_intervals`, question/choices, hidden terminal answer |
| Not auto-labeled | identity, state transition, causality, outside-clue negatives |

Active evaluation path:

- question-independent L1/L1.5 graphs from CG-Bench videos;
- legal cursor-local temporal/correlation hops (no Top-K action ranking);
- matched closed-loop arms: intact IWM / no-WM / shuffled-IWM / immediate-only / oracle;
- hidden clue overlap for evaluator-only scoring.

A prior 34-case/27-video zero-shot matched run is provisional engineering
evidence for the older single-belief loop (intact IWM clue recall 0.201 vs
immediate-only 0.137). It does **not** establish the newer multi-trajectory
planner. The repaired-graph diagnostic is currently negative for method claim
(IWM ≈ no-WM; shuffled can look better), so the active gate remains full-video
held-out L1/L1.5 freeze plus five-arm re-evaluation.

Historical Video-Holmes structured-state smokes and six-case dependency
ablations are preserved under `memory_graph/outputs/` and
`l15_graph_navigator/baselines/`; they are not the current benchmark.

## Quick start

Build an overlay from a canonical Video_Skills example and precomputed atomic
events:

```bash
python -m memory_graph.cli \
  --canonical /path/to/canonical_example.json \
  --atomic-events /path/to/atomic_events.json \
  --input-mode video_only \
  --output /path/to/causal_temporal_overlay.json
```

Persist event embeddings with Qwen3-VL-Embedding-2B:

```bash
python -m memory_graph.cli \
  --canonical /path/to/canonical_example.json \
  --atomic-events /path/to/atomic_events.json \
  --input-mode video_only \
  --embedding-output /path/to/event_embeddings.npy \
  --device cuda:0 \
  --output /path/to/causal_temporal_overlay.json
```

Run the four-strategy navigation ablation with persisted query embeddings:

```bash
python -m memory_graph.navigation_ablation \
  --overlay /path/to/causal_temporal_overlay.json \
  --cases /path/to/fixed_navigation_cases.json \
  --query-embedding-output /path/to/query_embeddings.npy \
  --embedding-device cuda:0 \
  --output /path/to/navigation_report.json
```

Run the preference-only L1.5 navigator with real Video_Skills retrieval calls:

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What happened after the anchor event?" \
  --seed-event event:anchor \
  --missing-role temporal \
  --belief-backend factor_graph \
  --output-dir /path/to/preference_navigation_run
```

The default production-compatible backend is currently a discrete Python
sum-product graph. The canonical factor-graph design, strict LLM/numeric
boundary, real GTSAM discrete pilot, Active-SLAM correction experiment, and
migration plan now live in [`factor_graph/README.md`](factor_graph/README.md).
The factorized local-update backend remains an ablation, and sibling labels
still require independent review before training.

For fixed multi-case generation, blinded four-way preference annotation,
training-record export, lightweight categorical-model training, and the
matched-budget ablation plus destructive controls, use:

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow --help
```

Formal CG-Bench evaluation and training export require frozen
question-independent graphs, hidden clue/answer keys, and content-locked
preference annotations. GPT / model-provisional outputs cannot pass those gates
without an explicit engineering-only opt-in. Historical Video-Holmes provisional
status remains in
[`video_holmes_engineering_status_v3.json`](steam_video_new/implicit_world_model/l15_graph_navigator/baselines/video_holmes_engineering_status_v3.json);
current CG-Bench builders and cohorts live under
[`steam_video_new/implicit_world_model/cgbench_grounded_navigation/`](steam_video_new/implicit_world_model/cgbench_grounded_navigation/README.md)
and
[`datasets/cgbench_gt_navigation_pilot_v2/`](steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/README.md).

Run the regression suite:

```bash
python -m unittest memory_graph.tests.test_memory_graph
```

The embedding path requires `sentence-transformers`, `torch`,
`transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, and `numpy`.

## Trust boundary and next gates

CG-Bench clue coverage can supervise reads and ordinal trajectory preference, but
it does not certify identity, state-transition, or causal edges. Deterministic
graph construction may still run; stronger relation claims remain untrusted
without an independent audit.

The grounded GT-interval transition corpus (670 available records; video-disjoint
train/validation/test = 516/74/80) is authorized now for loader/schema smoke,
intentional small-set overfitting, descriptor distillation, and split-safe
offline data checks. Those uses do not require completion of the full-video
L1/L1.5 or five-arm navigation gates.

It is not, by itself, sufficient training or validation for the runtime
categorical IWM: all 670 records are positive clue advances, answerability is
always `unknown`, timestamp-only actions omit the runtime L1/L1.5 semantic
target key, and the target schema differs from the full-graph categorical IWM
contract. Runtime-aligned categorical verification requires full-video train
graphs, post-freeze node alignment, categorical controls, and an adapter plus
held-out evaluator. The corpus may not be presented as closed-loop navigation
training, used as same-case runtime lookup, or promoted into identity,
state-transition, causal, or outside-clue negative labels.

The active CG-Bench milestones below gate full L1.5 closed-loop training,
preference/GRPO training, and method claims—not scoped data-pipeline smoke:

1. finish question-independent full-video validation/test L1/L1.5 builds;
2. freeze graph fingerprints and compile the held-out gate
   (target 30--50 fully retained cases before matched IWM runs);
3. rerun the five matched arms at the two-read stress budget and a larger
   budget justified by oracle clue count;
4. require shuffled-IWM to underperform intact IWM, with separate reports for
   transition confusion, action divergence, coverage, abstention, and latency;
5. only then treat closed-loop gains as method evidence or begin full
   navigation-policy training. Do not invent outside-clue negatives or
   question-conditioned Top-K.

## Documentation

- [World-model-guided navigation research (CG-Bench)](steam_video_new/README.md)
- [CG-Bench grounded navigation builder](steam_video_new/implicit_world_model/cgbench_grounded_navigation/README.md)
- [Memory graph design and implementation](memory_graph/README.md)
- [L1 reliability contract](memory_graph/L1_RELIABILITY.md)
- [Independent L1 audit protocol](memory_graph/INDEPENDENT_L1_AUDIT_PROTOCOL.md)
- [Validation history and staged protocol](memory_graph/VALIDATION.md)
- [Preference-only implicit world-model navigator](steam_video_new/implicit_world_model/l15_graph_navigator/README.md)
