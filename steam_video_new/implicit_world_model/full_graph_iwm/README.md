# Full-Graph IWM Navigation

This package is the new main-method path. It navigates a fixed-capacity,
question-independent L1/L1.5 evidence graph with one active cursor. It does not
replace the existing `l15_graph_navigator` data/review workflows; that package
remains the legacy baseline and dataset pipeline.

## Implemented flow

```text
visual representations
  -> surprise-driven adaptive L1 write windows
  -> priority-preserving materialized consolidation
  -> retained L1 nodes + temporal edges
  -> all-pair categorical L1.5 correlations
  -> one-cursor legal action compiler (no Top-K)
  -> batched categorical IWM predictions
  -> complete horizon-1/2 trajectory preferences
  -> execute one real read, correct belief, replan
```

The graph adapter consumes an existing `CausalTemporalOverlay`, consolidates
its L1 observations, projects legacy L1/L1.5 relation evidence categorically,
and returns a `RetainedEvidenceGraph`. Evidence nodes are reused; they are not
copied into a second memory store.

## Contracts that are enforced

- Legal actions are deterministic functions of the current cursor, retained
  graph, visibility, relation status, provenance, executor availability, and
  remaining budget. The question never filters the legal set.
- The initial virtual root emits `START_AT` for every visible retained node.
- From a real cursor, semantic probes cover every visible unread retained node;
  temporal and correlation actions use only that cursor as source.
- `BACKTRACK` changes focus to acquired evidence without rereading it.
- Unread nodes expose time/type/structural keys and a Qwen embedding sidecar
  reference, never the evidence text/value.
- Imagined evidence remains address-only. A horizon-two rollout cannot reveal
  the real value of its imagined first-hop target.
- GPT-OSS outputs only categorical observation/belief deltas and four-way
  pairwise preferences. Numeric output is rejected. Public action, trajectory,
  and pair IDs are replaced with pure-alphabetic aliases in model outputs and
  mapped back to graph IDs only after strict coverage validation.
- The main planner applies no embedding Top-K, score, weighted utility,
  lexicographic winner, or first-item tie break. Multiple undominated first
  hops cause explicit abstention.

## Components

```text
contracts.py              strict graph, action, transition, and preference types
graph_adapter.py          legacy overlay -> retained L1/categorical L1.5 graph
action_compiler.py        single-cursor legal actions and real graph execution
model_input.py            full retained graph with unread-value leakage guard
planner.py                batched horizon-1/2 rollout and partial-order selection
gpt_oss.py                strict GPT-OSS-120B categorical batch adapters
correlation_evaluator.py  all-pair GPT-OSS categorical L1.5 proposal adapter
```

The L1 primitives live in `memory_graph/`:

- `adaptive_windowing.py`: representation-surprise window writer with a
  pluggable learned feature provider and an explicit OpenCV smoke fallback;
- `consolidation.py`: materialized keep/merge/evict, lineage, embedding refresh
  invalidation, relation rewiring, and temporal-chain rebuilding;
- `correlation_overlay.py`: categorical correlation schema, structural
  candidates, all-pair evaluator contract, and legacy relation projection.

## Minimal construction

```python
from steam_video_new.implicit_world_model.full_graph_iwm import (
    CursorBeliefState,
    FullGraphIWMPlanner,
    GraphActionCompiler,
    build_retained_graph_from_legacy_overlay,
)

graph = build_retained_graph_from_legacy_overlay(overlay, capacity=8)
belief = CursorBeliefState(
    belief_id="belief:0",
    question=question,
    remaining_reads=8,
)
legal_actions = GraphActionCompiler().compile(belief, graph)
decision = FullGraphIWMPlanner(world_model, preference_model, horizon=2).plan(
    belief,
    graph,
)
```

The learned batched IWM can use the experiment's frozen capacity. For the
provisional GPT-OSS API smoke, begin with capacity 8 and horizon 1; horizon 2
is exhaustive and should be enabled only after checking the compiled action
and trajectory counts. The planner abstains on ambiguity and never silently
falls back to Top-K.

For GPT-OSS-120B, instantiate the existing strict OpenRouter client from
`l15_graph_navigator.gpt_oss.OpenAICompatibleCategoricalClient`, then pass it
to `GPTOSSFullGraphWorldModel`, `GPTOSSFullGraphPreferenceModel`, and optionally
`GPTOSSCategoricalCorrelationEvaluator`. The adapters are inference/data-
gathering only; no GPT-OSS training is performed here.

A compile-only smoke does not call any external model:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What evidence connects the two events?" \
  --capacity 8 \
  --mode compile-only \
  --output /tmp/full_graph_compile.json
```

One GPT-OSS planning step through OpenRouter is:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What evidence connects the two events?" \
  --capacity 8 \
  --mode gpt-oss-120b \
  --keys-py /fs/gamma-projects/vlm-robot/keys.py \
  --output /tmp/full_graph_gpt_oss_plan.json
```

Add `--gpt-correlation-proposals` only when all retained pairs should also be
sent to GPT-OSS for provisional categorical relation proposals. Model proposals
cannot self-admit a `verified` edge.

## Current validation

`memory_graph/tests/test_full_graph_iwm.py` covers adaptive boundaries,
materialized fixed capacity, edge rewiring, all-pair correlation coverage,
single-cursor legality, hidden-node filtering, unread-value isolation, delayed
two-hop selection, and abstention under ties. These are implementation tests,
not a scientific performance result.

On 2026-07-21, a real OpenRouter `openai/gpt-oss-120b` smoke over a persisted
Video-Holmes overlay also passed the strict adapter: two retained nodes, four
legal initial actions, four horizon-one trajectories, all six pairwise
comparisons, zero unread evidence-value leaks, and no Top-K. The categorical
partial order was non-unique, so the planner correctly returned `abstain`.
This validates execution and failure semantics; it is not a navigation-quality
claim.
