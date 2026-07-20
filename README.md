# STEAM Video Memory Graph

This repository develops an evidence-grounded memory graph for long-video
understanding and bounded graph navigation. The active implementation lives in
`memory_graph/`; `dynamic_navigation/`, `legacy/`, and parts of
`steam_video_new/` contain separate experiments or earlier work.

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

## Current validated engineering smoke

The latest structured-state Video-Holmes smoke completed 55/55 clip schemas
and 55/55 graph-composition targets with zero final integrity errors. Its
provisional strict overlay contains:

| Item | Count |
| --- | ---: |
| L1 observations | 295 |
| L1 structural relations | 59 |
| Atomic events | 30 |
| Events with grounded participants | 30 |
| Events with visible states | 25 |
| Grounded state assertions | 67 |
| Event relations | 140 |
| Qwen event embedding references | 30 |
| Strict state transitions | 1 |

The admitted provisional transition is the same tracked man's expression
changing from `serious` to `focused`.

Under the same two-read budget on an unchanged six-case provisional gold set,
the Qwen embedding navigation ablation produced:

| Strategy | Answer accuracy | Mean evidence recall |
| --- | ---: | ---: |
| Semantic only | 66.7% | 0.833 |
| Event only | 66.7% | 0.833 |
| Native L1 candidate | 66.7% | 0.833 |
| Verified dependency | **83.3%** | **0.917** |

This is positive engineering evidence that a verified state dependency can
improve navigation. It is not a formal benchmark result: the audit decisions
are GPT-5.6 visual provisional labels, and the result currently covers one
video, six questions, and one admitted transition.

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

The default backend is a discrete sum-product factor graph that maintains
exploration belief, performs global conflict correction, and prioritizes which
constraint to inspect. The world model remains a separate rule-based baseline
that emits categorical observation/belief deltas; the planner uses ordinal
trajectory preferences, never numeric reward or utility. GTSAM is reserved for
future continuous latent time/track variables, and the factorized local-update
backend remains available as an ablation. Sibling labels require independent
review before training.

For fixed multi-case generation, blinded four-way preference annotation,
training-record export, and the eight-policy matched-budget ablation, use:

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow --help
```

Formal evaluation and training export require content-locked human case and
preference annotations. GPT-5.6 outputs remain explicitly `ai_provisional` and
cannot pass those gates without an opt-in intended only for engineering runs.

Run the regression suite:

```bash
python -m unittest memory_graph.tests.test_memory_graph
```

The embedding path requires `sentence-transformers`, `torch`,
`transformers>=4.57.0`, `qwen-vl-utils>=0.0.14`, and `numpy`.

## Trust boundary and next gates

Without an independent human audit, deterministic graph construction still
runs, but video-only probabilistic identity, state, dependency, and causal
claims must not be reported as trusted. The next required evidence milestones
are:

1. independently review the identity-relation packet and reach at least 90%
   strict precision for admitted identity and state-transition edges;
2. freeze 30--50 navigation questions across multiple videos;
3. test state-change paraphrases and hard negatives where no verified
   transition exists;
4. report accepted-track, grounded-delta, and usable-dependency coverage;
5. build a persistent embedding index and expand to learned or multi-hop
   navigation only after those gates pass.

## Documentation

- [Memory graph design and implementation](memory_graph/README.md)
- [L1 reliability contract](memory_graph/L1_RELIABILITY.md)
- [Independent L1 audit protocol](memory_graph/INDEPENDENT_L1_AUDIT_PROTOCOL.md)
- [Validation history and staged protocol](memory_graph/VALIDATION.md)
- [Preference-only implicit world-model navigator](steam_video_new/implicit_world_model/l15_graph_navigator/README.md)
