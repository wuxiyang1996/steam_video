# World-Model-Guided Multi-Hop Video Reasoning

`steam_video_new` is the research workspace for **world-model-guided reasoning over
grounded video evidence memory**. The central question is not whether a graph can
store video facts. It is whether an implicit world model (IWM) can predict how a
candidate reasoning hop will change future belief, and whether a planner can use
that prediction to choose better multi-hop evidence trajectories.

Detailed formulations: [English](problem-formulation-en.html) ·
[中文](problem-formulation-zh.html)

## 1. Current research thesis

```text
question-independent L1/L1.5 evidence memory
        ↓ retrieve a bounded candidate set
current latent belief z_t + grounded evidence read so far
        ↓
IWM imagines each legal reasoning action
        ├─ categorical observation descriptor
        └─ categorical future belief delta
        ↓
pairwise trajectory preference (no scalar reward)
        ↓
planner executes only the preferred trajectory's first hop
        ↓
real evidence observation → correct belief → replan
```

The novelty is the middle of this loop: **action-conditioned prediction of
reasoning dynamics in belief space and model-predictive selection of the next
reasoning hop**. L1/L1.5 memory is the grounded evidence substrate. It is useful
infrastructure, but not the main novelty.

## 2. Responsibilities and boundaries

| Component | Responsibility | It must not do |
|---|---|---|
| L1/L1.5 evidence memory | Store question-independent observations, events, time spans, provenance, typed candidate relations, and embedding references | Read the future question/answer while building the graph; turn unmatched candidates into negatives |
| Candidate retrieval | Use embedding/structure to expose a bounded set of legal reasoning hops | Select the final winner by a hand-written score or stable ordering |
| Latent belief state | Summarize acquired evidence, competing interpretations, missing links, contradictions, answerability, and budget | Be confused with the explicit L1.5 graph or imagined evidence |
| Implicit world model | Predict categorical observation and belief-delta descriptors for actions; imagine one- or two-hop futures | Emit reward, utility, Q-value, probability, confidence, or evidence used directly in the answer |
| Preference planner | Compare complete candidate trajectories ordinally and execute the first hop of a uniquely preferred trajectory | Sum model-generated scores; silently break ties by candidate order |
| Real observation update | Replace imagined consequences with an executed evidence read and update belief | Persist an imagined rollout as fact |
| GTSAM/factor graph | Optional correction backup, diagnostic baseline, teacher, and visualization for conflicts and persistent belief consistency | Generate actions, rank candidates, replace the IWM, or become required by the main method |
| L2 trace | Record executed actions, real observations, realized belief deltas, and decisions for audit/training data | Act as the belief model or evidence source |

The default paper method **requires the shared L1/L1.5 evidence graph for
navigation**, while its question-conditioned reasoning belief and IWM state are
latent rather than an explicit graph posterior. GTSAM remains available through
explicit backup/baseline modes; it is not silently enabled.

### 2.1 What the L1 graph stores

L1 is a question-independent `ClueMemoryGraph` compatible with Video_Skills. A
node stores one local, time-scoped piece of grounded evidence—not a reasoning
conclusion. Common fields are:

```text
node_id, node_type, video_id
text / grounded descriptor, modality
time_span, clip_id, local/mention ID
entity attributes or structured state fields when applicable
evidence_refs, source_type, producer, provenance
visibility / hidden-supervision flag
optional Qwen3-VL-Embedding-2B sidecar reference
```

The main node types are `clip`, `observation`, `event`, `entity_mention`,
`state`, and `dialogue_span`/OCR. Entity mentions are local observations; they
do not assert cross-clip identity. A state should identify its subject,
attribute, value, polarity, time span, and evidence rather than merely say that
something changed. Subtitle, OCR, direct audio, and visual descriptions remain
separate modalities.

L1 also stores composition structure over the same node IDs:

```text
temporal_next, derived_from, entity_mention, state_of, located_in
```

Native Video_Skills relations such as `same_entity`, `same_object`,
`reappears`, `before_after`, `state_change`, `supports_observation`,
`contrasts_observation`, `causal_hint`, and `social_cue` are retained with
provenance, but unverified semantic labels are downgraded to candidates. In
particular, `state_change` is not automatically an accepted state transition,
and `causal_hint` is not causality.

### 2.2 What the L1.5 overlay adds

L1.5 does not duplicate the evidence nodes. It connects the existing L1 node
IDs with question-independent temporal and cross-node navigation correlations:

```text
semantic/entity recurrence
state continuity or candidate state transition
transition support, response, and missing-bridge candidates
contradiction candidates
sparse verified explains/enables witnesses
```

Each correlation edge records `edge_id`, `src`, `dst`, relation type,
`candidate|verified|rejected|inconclusive` status, evidence refs, candidate
sources, mention/state alignment, verifier result, and provenance. Embedding
similarity may propose a pair but does not prove its relation. Candidate edges
may create inspect/verify actions; only admitted relations constrain belief.

L1/L1.5 never stores the correct answer, hidden clue identity, question-
conditioned belief, IWM imagined observation, predicted belief delta, planner
trajectory, reward/utility/Q-value, GTSAM posterior, or final claim. Those
belong to hidden evaluation, latent belief, or the executed L2 audit trace.

## 3. Reasoning actions

An action is a reasoning/evidence-acquisition hop, not a physical robot action:

- semantic evidence lookup;
- temporal before/after expansion;
- same-entity or state-continuity check;
- candidate-cause/effect or missing-bridge lookup;
- counterevidence or relation verification;
- stop/answer/abstain.

The main method uses a **single active current-node cursor**. Ordinary move,
follow, inspect, and verify actions always use that cursor as their source; all
previously acquired nodes remain in belief/frontier history but are not expanded
simultaneously. This avoids the invalid `all frontier sources × all targets`
action product. A recorded `BACKTRACK`/`SHIFT_FOCUS` operation can return the
cursor to an acquired node without rereading evidence.

Legal actions are compiled deterministically from executability, not proposed
or ranked by the IWM:

```text
current-node temporal outgoing edges
+ current-node correlation inspect/follow/verify edges
+ semantic probes from the cursor to every visible retained node
+ backtrack to acquired frontier-history nodes
+ stop / answer / abstain
```

At the initial step, a virtual query root provides `START_AT(node)` for every
visible retained node. Because L1 memory has fixed capacity, this action set is
bounded. The main arm does not apply embedding Top-K: embedding is an action
feature/key, not a gate. Structural legality may remove hidden, out-of-horizon,
rejected, provenance-free, non-executable, or over-budget actions; it may not
remove actions for low question relevance. Top-K remains an explicit retrieval
baseline only.

For a current belief `z_t` and legal action `a`, the IWM predicts:

```text
T(z_t, a) → (predicted observation descriptor, predicted belief delta)
Pref(trajectory_left, trajectory_right)
  ∈ {prefer_left, tie, prefer_right, incomparable}
```

The model may have internal logits during optimization, but its public contract
is categorical. Numeric evaluation metrics remain outside the model.

## 4. Why planning must depend on the world model

A reactive retriever can favor the most immediately relevant hop. The intended
IWM handles **delayed reasoning effects**: a first hop may have little immediate
answer value but reveal a bridge that makes the second hop decisive.

The planner therefore performs horizon-1/2 model-predictive control:

1. compile all legal actions from the single current-node cursor and the fixed L1.5 graph;
2. imagine action-conditioned future belief transitions;
3. compare full candidate trajectories pairwise;
4. execute only the first hop of the selected trajectory;
5. read real evidence, update belief, and replan.

Candidate order, lexical overlap, graph priority, factor posterior, and embedding
similarity cannot filter or determine the winner in the main arm. The IWM
predicts all legal actions in one batched graph/tensor forward. For horizon two,
strictly dominated first hops may be removed before expanding every second hop
from the remaining partial-order set; this is not fixed-K beam pruning.
`tie` or `incomparable` remains available and causes further evidence,
backtracking, another comparison, or abstention—not an implicit first-item
fallback.

## 5. Bounded input contract

The IWM/planner never receives the whole video, raw embedding matrix, hidden
evaluator key, or unconsolidated history. Because SelectStream-style L1 memory
has fixed capacity, the main method can consume the complete **retained** graph
without Top-K truncation. Each step receives:

- the question and compact categorical belief summary;
- the current cursor node's key and full acquired evidence value;
- every retained node key plus all retained temporal/correlation edges;
- every legal action compiled from the current cursor, plus short recent-hop history;
- remaining categorical budget/status;
- at most one- or two-hop trajectory descriptors.

`Qwen/Qwen3-VL-Embedding-2B` embeddings are stored as sidecars and referenced by
row/checksum. Unread targets expose compact keys/addresses rather than full
evidence values; the value is revealed only after execution. Embeddings provide
semantic action features, not main-arm ranking, reward, or preference
supervision.

## 6. Supervision and post-training

The cleanest current supervision source is CG-Bench:

- human `clue_intervals` supervise whether a real read acquires a new required clue;
- partial/complete clue coverage supplies categorical belief progress;
- the terminal answer is kept in a hidden evaluator key;
- complete-clue versus leave-one-clue-out trajectories supply ordinal preference;
- unmatched L1.5 candidates remain **unlabeled**, never automatic negatives.

Executed sibling branches from the same immutable checkpoint provide additional
observation and belief-delta targets. Identity, state-transition, causal, and
counterevidence labels require their own trusted source; clue relevance must not
be promoted into those stronger relations.

The staged training plan is:

1. collect and validate grounded executed transitions;
2. supervised learning for categorical observation/belief-delta prediction;
3. pairwise preference optimization on locked sibling trajectories;
4. optionally study group-relative preference optimization only after the
   categorical ordering is reliable.

There is no model-produced scalar reward. A future GRPO-style experiment may
compute optimization advantages internally from group ordering, but it must not
invent weighted heuristic rewards or expose numeric reward targets. At the
current stop point GPT-OSS-120B is used only for inference/data gathering; it is
not being trained.

## 7. Evaluation that can support the claim

The main matched-budget arms keep question, belief, candidates, planner, and read
budget fixed, changing only the world-model condition:

1. full delayed-belief IWM;
2. no world model;
3. shuffled IWM predictions;
4. frozen IWM;
5. immediate-effect-only;
6. oracle-clue ceiling (diagnostic only).

Report separately:

- terminal answer accuracy;
- evidence/clue completeness;
- read efficiency and latency;
- first-action and full-trajectory divergence;
- categorical observation/belief-delta prediction accuracy;
- delayed two-hop success and recovery after contradiction;
- candidate recall before planning.

The claim fails if perturbing or removing the IWM does not systematically change
actions and outcomes, or if gains come from candidate retrieval/order heuristics.

## 8. Current implementation status (2026-07-21)

Implemented:

- L1/L1.5 overlay adapter, legal graph-read actions, bounded reasoning context,
  horizon-1/2 trajectory expansion, pairwise partial-order planning, real-read
  execution, belief snapshots, and L2-compatible audit traces;
- strict categorical GPT-OSS-120B observation/belief and trajectory-preference
  adapters; numeric model output is rejected;
- executed-transition, sibling, blinded review, targeted gathering, and
  matched-ablation workflows;
- optional Python factor backend and isolated GTSAM correction pilots/backups;
- CG-Bench v2 with 256 video-disjoint multi-clue cases, 205 videos, 672 GT clue
  transitions, hidden terminal targets, and no fabricated outside-clue negatives;
- question-independent 12-video L1/L1.5 prefix-smoke worker and frozen hidden
  evaluator for native and embedding Recall@K.

Not yet established:

- a trained learned IWM;
- production-quality identity/state/dependency relation precision;
- full-video L1.5 candidate coverage;
- a locked, multi-video closed-loop result showing causal dependence on the IWM;
- production readiness.

The active pipeline is grounding all 672 CG-Bench reads with Qwen-VL, validating
embeddings/leakage/splits, then running the 12-video question-independent L1/L1.5
smoke. Runtime artifacts remain data-only and are not committed as results.

## 9. Directory map

```text
steam_video_new/
├── README.md
├── problem-formulation-en.html
├── problem-formulation-zh.html
└── implicit_world_model/
    ├── l15_graph_navigator/        # IWM/planner contracts and data workflows
    ├── cgbench_grounded_navigation/ # CG-Bench grounding and L1.5 smoke
    └── datasets/                    # versioned local data artifacts/reports

../factor_graph/                     # canonical GTSAM backup and experiments
../memory_graph/                     # L1/L1.5 graph construction primitives
../../Video_Skills/                  # external execution/runtime substrate
```

See component documentation:

- [`implicit_world_model/l15_graph_navigator/README.md`](implicit_world_model/l15_graph_navigator/README.md)
- [`implicit_world_model/cgbench_grounded_navigation/README.md`](implicit_world_model/cgbench_grounded_navigation/README.md)
- [`../factor_graph/README.md`](../factor_graph/README.md)
- [`../memory_graph/README.md`](../memory_graph/README.md)

## 10. Immediate next gate

Do not begin model training until the current data-only chain passes:

1. 672/672 grounded-read completion and schema validation;
2. embedding shape/checksum, hidden-answer leakage, and video-disjoint split checks;
3. 12-video frozen-graph native and embedding Recall@4/8/16/32;
4. failure-slice inspection for empty graphs, missed temporal segments, visual
   grounding failures, and clues outside the 120-second smoke horizon;
5. a decision to scale L1.5 construction or first repair candidate generation.

Only after candidate recall is adequate should the matched-budget closed-loop
IWM interventions be treated as meaningful.
