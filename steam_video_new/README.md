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
        ↓ expose the retained graph and cursor-local legal hops
current latent belief z_t + grounded evidence read so far
        ↓
IWM imagines each legal reasoning action
        ├─ backend-bound target observation descriptor
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
| L1/L1.5 evidence memory | Store question-independent semantic observations, temporal edges, soft navigation correlations, provenance, and embedding references | Read the future question/answer; present similarity as probability or verified fact |
| Legal-action compiler | Expose executable root, temporal, correlation, backtrack, and terminal actions | Rank actions by a hand-written score or silently apply Top-K |
| Latent belief state | Summarize acquired evidence, competing interpretations, missing links, contradictions, answerability, and budget | Be confused with the explicit L1.5 graph or imagined evidence |
| Implicit world model | Predict categorical outcome and belief-delta descriptors for target-bound actions; imagine one- or two-hop futures | Rewrite the selected target identity; emit reward, utility, Q-value, probability, confidence, or evidence used directly in the answer |
| Preference planner | Compare complete candidate trajectories ordinally and execute the first hop of a uniquely preferred trajectory | Sum model-generated scores; silently break ties by candidate order |
| Real observation update | Replace imagined consequences with an executed evidence read and update belief | Persist an imagined rollout as fact |
| GTSAM/factor graph | Optional correction backup, diagnostic baseline, teacher, and visualization for conflicts and persistent belief consistency | Generate actions, rank candidates, replace the IWM, or become required by the main method |
| L2 trace | Record executed actions, real observations, realized belief deltas, and decisions for audit/training data | Act as the belief model or evidence source |

The default paper method **requires the shared L1/L1.5 evidence graph for
navigation**, while its question-conditioned reasoning belief and IWM state are
latent rather than an explicit graph posterior. GTSAM remains available through
explicit backup/baseline modes; it is not silently enabled.

Learned navigation must not be replaced by accumulated routing rules. Structural
code may enforce only executability, leakage isolation, graph direction,
already-read state, and budget. Question relevance, expected belief change, and
trajectory preference remain learned categorical decisions. In particular,
role count, keyword overlap, cosine value, edge density, timestamp proximity,
compiler order, and fixed-K membership are not action utilities.

### Architecture invariant: correlation is not preference

L1.5 correlation and planner preference answer different questions and must not
be conflated:

```text
L1.5 correlation: which L1 node pairs have a supported navigable connection?
IWM prediction:   what categorical belief effect may follow a legal graph hop?
Planner choice:   which legal hop should be executed next under those predictions?
```

L1.5 is therefore a question-independent node-to-node correlation overlay. It
may store measured embedding similarity, directional affinity, calibration,
and provenance because these are graph-construction features. The prohibition
on model-produced numbers applies to invented reward, utility, Q-value,
probability, or confidence used to choose an action; it does not prohibit
measured graph features. Similarity or affinity still cannot be presented as a
verified identity/state/causal fact or directly determine the winning action.

The legal-action compiler reads the fixed topology and exposes only executable
root, cursor-incident temporal/correlation, backtrack, and terminal hops. It
does not ask the IWM or planner to discover correlations. Conversely, the IWM
and planner may consume correlation edges as navigation context but may not
create, verify, or reclassify those edges during action selection. Optional
strict identity/state/causal relations remain a separately admitted layer.

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

The main L1 navigation structure over these nodes is the deterministic temporal
backbone:

```text
temporal_next / before / overlaps / during
```

Native composition links (`derived_from`, `entity_mention`, `state_of`,
`located_in`) remain provenance/audit structure, not L1.5 correlation.
Video_Skills labels such as `same_entity`, `same_object`,
`reappears`, `before_after`, `state_change`, `supports_observation`,
`contrasts_observation`, `causal_hint`, and `social_cue` are retained with
provenance only in the optional strict-relation path. They do not automatically
become navigation edges. In particular, `state_change` is not an accepted state
transition, and `causal_hint` is not causality.

### 2.2 What the L1.5 overlay adds

L1.5 does not duplicate evidence nodes or try to name every relation. It adds
question-independent, embedding-derived **soft nonlocal navigation
correlations** over existing L1 node IDs. An edge records:

```text
src / dst
endpoint cosine similarity
src→dst and dst→src navigation affinity
embedding model/checksum provenance
semantic or semantic_recurrence channel
```

All non-temporal pairs are scored. Adjacent near-duplicate semantic nodes are
coalesced during L1 consolidation; remaining near-identical embeddings form a
time-ordered recurrence chain rather than a clique. Correlations between
semantic equivalence classes use standardized sparsemax, not fixed Top-K.
Directional affinity controls hop legality. Similarity and affinity are learned
representation features, not calibrated probability, confidence, identity,
state transition, support, or causality.

Strict categorical identity/state/causal relations are a separate optional
layer. Only independently verified relations enter it; categorical candidates
never become generic navigation edges merely because a verifier proposed them.

L1/L1.5 never stores the correct answer, hidden clue identity, question-
conditioned belief, IWM imagined observation, predicted belief delta, planner
trajectory, reward/utility/Q-value, GTSAM posterior, or final claim. Those
belong to hidden evaluation, latent belief, or the executed L2 audit trace.

## 3. Reasoning actions

An action is a reasoning/evidence-acquisition hop, not a physical robot action:

- initial read of one visible retained semantic node;
- temporal before/after expansion;
- follow a positive-direction soft L1.5 correlation;
- backtrack to an acquired node without rereading it;
- stop/answer/abstain.

The main method uses a **single active current-node cursor**. Ordinary move,
follow actions always use that cursor as their source; all
previously acquired nodes remain in belief/frontier history but are not expanded
simultaneously. This avoids the invalid `all frontier sources × all targets`
action product. A recorded `BACKTRACK`/`SHIFT_FOCUS` operation can return the
cursor to an acquired node without rereading evidence.

Legal actions are compiled deterministically from executability, not proposed
or ranked by the IWM:

```text
current-node temporal outgoing edges
+ current-node positive-direction soft-correlation edges
+ backtrack to acquired frontier-history nodes
+ stop / answer / abstain
```

At the initial step, a virtual query root provides `START_AT(node)` for every
node in a frozen categorical entry frontier. The entry localizer examines all
visible retained semantic addresses once, without unread evidence values,
numeric scores, or Top-K; it repairs or fails when the frontier exceeds its
declared bound. After entry, the compiler exposes only cursor-incident
temporal/correlation hops. Embedding remains an address/edge feature rather
than a direct action-ranking rule. Top-K remains an explicit retrieval
baseline only.

For a current belief `z_t` and legal action `a`, the IWM predicts:

```text
address(a) → backend-bound target descriptor
T(z_t, a, address(a)) → (predicted categorical outcome, predicted belief delta)
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

### Multi-trajectory reasoning state

The main IWM/planner contract maintains a pool of competing reasoning
trajectories rather than committing the complete belief to one cursor path:

```text
shared grounded evidence + read budget
        ├─ trajectory A: hypothesis, cursor, frontier, missing roles
        ├─ trajectory B: hypothesis, cursor, frontier, missing roles
        └─ trajectory C: hypothesis, cursor, frontier, contradictions
                              ↓
IWM jointly compares every legal hypothesis-conditioned expansion
                              ↓
categorical partial preference over expansions
                              ↓
planner executes one shared real evidence action
                              ↓
observation is broadcast to every active trajectory; update and replan
```

Trajectories are preserved when tied or incomparable. They are removed only by
grounded contradiction, explicit abandonment, budget invalidity, or exact
structural consolidation with an equivalent trajectory. There is no score-based
beam or fixed-K trajectory pool. If several preferred expansions correspond to
the same executable graph action, that action may be executed once; preferred
expansions with different first actions cause abstention.

The IWM directly receives the trajectory pool plus temporal, semantic,
correlation and current-belief context and returns categorical preference. An
imagined transition descriptor remains an auxiliary supervision/audit target,
not the only information available to the preference decision. The planner is
responsible only for legality, shared execution, lifecycle bookkeeping and
replanning.

The belief backend is interchangeable. A latent updater is the default method;
GTSAM/factor graph may maintain competing hypotheses and persistent corrections
behind the same interface, but it does not rank expansions or choose actions.

The experiment-facing implementation now represents horizon-two imagination as
one complete action tree per possible next shared evidence read. Every tree
contains its categorical first transition and all legal categorical second-hop
transitions across the competing trajectory context. The planner exhaustively
compares first-action trees with categorical pairwise labels, executes only the
first real read, broadcasts the observation, and replans. This preserves every
hypothesis and delayed continuation without multiplying the final comparison
set by hypothesis count and without using Top-K.

CG-Bench public answer choices initialize the trajectory pool. Hidden answers
and clue intervals are evaluator-only. The runnable CLI and five matched arms
live in `implicit_world_model/full_graph_iwm/multi_trajectory_cgbench.py`; the
optional executed-read-only GTSAM adapter lives in
`implicit_world_model/full_graph_iwm/gtsam_backup.py`.

## 5. Bounded input contract

The IWM/planner never receives the whole video, raw embedding matrix, hidden
evaluator key, unconsolidated history, or the complete retained graph at every
step. Entry localization consumes all safe semantic addresses once. Each
reasoning step then receives:

- the question and compact categorical belief summary;
- required evidence roles and each real role-to-node provenance binding;
- the current cursor node's key and full acquired evidence value;
- only acquired/current/legal-endpoint node keys and their induced
  temporal/correlation edges;
- every legal action compiled from the current cursor, plus short recent-hop history;
- remaining categorical budget/status;
- at most one- or two-hop trajectory descriptors.

For supervision, the system records every initial anchor prediction before
setwise selection. A separate exporter forms complete local-anchor comparisons
and executed transition corrections. Dataset GT is applied after planning and
kept in a hidden label file; lack of clue overlap remains `not_established`,
not a semantic negative. No scalar reward is synthesized.

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

- pluggable representation-surprise L1 windowing and a materialized fixed-
  capacity consolidation path with lineage, embedding invalidation, relation
  rewiring, and retained temporal-chain rebuilding;
- soft Qwen-embedding L1.5 navigation correlations with all non-temporal pairs
  scored, near-duplicate semantic equivalence classes, recurrence chains instead
  of cliques, and class-level standardized sparsemax without fixed Top-K;
- categorical identity/state/causal verification retained as a separate optional
  strict-relation layer rather than the generic navigation edge definition;
- a separate `full_graph_iwm` main path with categorical entry localization, a
  virtual query root, one active cursor, frozen entry anchors, cursor-local
  temporal/correlation hops, no Top-K pruning,
  batched horizon-1/2 prediction, and explicit abstention for a non-unique
  partial order;
- unread-value and imagined-rollout leakage guards: unread nodes expose only
  compact keys/embedding references, and imagined reads never reveal real text;
- fixed-case CG-Bench compile gate and matched closed-loop arms for full IWM,
  no-WM, prediction shuffle, frozen WM, immediate-only, and oracle ceiling;
  hidden clue overlap is evaluator-only and never fed into planner belief;
- L1/L1.5 overlay adapter, legal graph-read actions, bounded reasoning context,
  horizon-1/2 trajectory expansion, pairwise partial-order planning, real-read
  execution, belief snapshots, and L2-compatible audit traces;
- strict categorical GPT-OSS-120B observation/belief and trajectory-preference
  adapters; numeric model output is rejected, output IDs use pure-alphabetic
  aliases, and a real OpenRouter full-graph smoke passed strict transition and
  exhaustive pairwise coverage (returning explicit abstention on ambiguity);
- executed-transition, sibling, blinded review, targeted gathering, and
  matched-ablation workflows;
- optional Python factor backend and isolated GTSAM correction pilots/backups;
- CG-Bench v2 with 256 video-disjoint multi-clue cases, 205 videos, 672 GT clue
  transitions, hidden terminal targets, and no fabricated outside-clue negatives;
- question-independent L1/L1.5 prefix-smoke worker with default
  surprise-adaptive OpenCV smoke boundaries; the current fixed pilot runs the
  first 8 videos from the 12-video stratified selection, with a frozen hidden
  evaluator for native and embedding Recall@K;
- a fixed held-out cohort protocol covering the complete 49-case validation/test
  population on 36 videos, plus a post-freeze gate that separates raw-L1 clue
  loss, bounded-consolidation clue loss, and missing L1.5 paths. Structural
  two-hop candidates remain distinct from executed delayed-success labels;
- a grounded single-pass L1 mode that folds coarse detection and frame-evidence
  localization into one VLM request per surprise window. A real 120-second
  smoke reduced requests from roughly 115 to 9 while rejecting outputs without
  sampled-frame provenance. The aggressive window configuration subsequently
  missed one full-video clue, so formal runs use a denser balanced configuration
  and retain full-video clue retention as the promotion gate.
- an explicit Transformers/vLLM serving ablation for Qwen3.5-9B. Backend
  provenance is part of the L1 resume contract, and vLLM is promoted only after
  a matched full-video speed, node-quality, clue-retention, and L1.5-path check;
  serving changes do not alter the L1/L1.5 method or provide supervision.
- bounded within-video request concurrency with isolated clients and ordered
  result collection, allowing vLLM continuous batching without changing the
  frozen surprise windows, sampled evidence, prompts, parsers, or node order.
- a complete 34-case/27-video, 170-slot zero-shot matched evaluation. Intact
  horizon-two IWM reaches 0.201 clue recall versus 0.137 immediate-only, 0.098
  shuffled-IWM and 0.000 no-WM. The paired gains are provisional: only 3 cases
  improve over immediate-only, transition outcome exact match is 1/31, and
  exact belief-delta match is 0/31. All localization/schema failures remain in
  the denominator as fail-closed abstentions. This result validates the earlier
  single-persistent-belief loop, not the newer persistent multi-trajectory
  planner.

The canonical V1 responsibility boundary is now explicit: the Planner owns and
tracks persistent competing reasoning trajectories; the IWM predicts the
future observation/belief outcome of each proposed short reasoning chain; a
categorical preference model compares complete predicted outcomes; and the
Planner executes one first hop, applies the real observation to the persistent
trajectory pool, discards stale imagined continuations and replans. Imagined
belief is never copied into persistent belief.

### Engineering completion status

The scoped V1 IWM/planner implementation is **code-complete**. It includes:

- question-independent L1/L1.5 evidence graphs and legal graph actions;
- persistent competing answer-hypothesis trajectories;
- categorical, action-conditioned horizon-one/two imagined transitions;
- complete-coverage setwise/pairwise trajectory preference without numeric
  rewards, heuristic Top-K, candidate-order tie breaking or imagined-to-real
  belief leakage;
- execution of one shared real evidence read, hypothesis-conditioned belief
  correction, stale-rollout invalidation and replanning;
- matched intact/no-WM/shuffled/immediate/oracle arms;
- optional isolated GTSAM belief-correction backup that never ranks actions;
- persistent response caches, fail-closed schemas, coverage/calibration audits
  and a resumable case-isolated frozen-cohort runner.

Real model-backed smoke tests have exercised these interfaces end to end. This
means the remaining gate is empirical model quality, not an unimplemented
planner or belief-update path. “Code-complete” does not mean that the learned
IWM is trained, calibrated, statistically better than the matched baselines or
production-ready.

Not yet established:

- a trained learned IWM;
- production-quality identity/state/dependency relation precision;
- full-video L1.5 candidate coverage;
- a strong and statistically stable horizon-two advantage over immediate-only;
- calibrated IWM observation/belief transitions and robust entry localization;
- a frozen-cohort five-arm validation of persistent multi-trajectory planning;
- terminal answer accuracy on the locked cohort;
- production readiness.

A real `qwen/qwen3.6-flash` multi-trajectory smoke now closes two full
`IWM -> setwise planner -> real read -> hypothesis-conditioned correction ->
replan` steps. Joint-chain construction retained all 369 generated conditioned
outcomes, and both reads split the six answer hypotheses into two distinct
belief groups. The five matched arms also complete without runtime errors:
no-WM diverges at the first read, while shuffled-IWM and immediate-only diverge
from the intact IWM at the second read. This establishes mechanism dependence,
not navigation quality. Every model-backed arm has zero clue recall and zero
answer accuracy on this case, while the two-read oracle covers two of three
clues. Selected-action transition outcome and progress calibration for the
intact IWM are both zero on the hidden clue-overlap evaluator. The frozen
34-case run is therefore the active empirical gate, not a formality.

Transition transport is now losslessly grouped by exact shared action sequence:
the L1/L1.5 address descriptor is sent once, while every hypothesis retains an
independent belief/prefix-conditioned categorical outcome. This leaves the
horizon-two, two-read and five-arm protocol unchanged. OpenRouter Qwen is stable
with one shared-action group per request; two groups exceeded its structured
output envelope. The IWM prompt also now treats “no acquired evidence yet” as
the pre-read state rather than predicting an `empty` observation by default.
A post-fix model-backed smoke retained all 246 hypothesis-conditioned outcomes,
formed 41 complete joint chains and reduced the preferred frontier to two
different first hops. The model returned a tie, so the planner correctly
abstained instead of introducing an ungrounded tie-break. This further verifies
the code path while isolating first-hop preference identifiability as an
empirical model limitation.

The latest grounding validation has 670/672 successful Qwen-VL reads and two
failed reads. Eight question-independent CG-Bench smoke graphs and their Qwen
embedding sidecars are present. Under the repaired L1/L1.5 builder they retain
15–64 nodes and emit 22–133 soft correlation edges; the former dense climbing
case is now 15 nodes/22 edges instead of a near-clique. The previous matched
pilot used the obsolete dense graph fingerprint and comparison budget, so it is
not evidence about the repaired method. A local repaired 8-video capacity-64
compile gate now passes graph availability, embedding build, clue retention,
hidden-key, and unread-value checks.

The new frozen-correlation audit writes one decomposed row for every retained
node pair, records source/retained L1 fingerprints, makes zero per-pair LLM
calls, and confirms that all eight source L1 overlays remain checksum-identical.
For the six consecutive clue bridges inside the 120-second prefix, direct graph
coverage is 5/6 and at-most-eight-hop coverage is 6/6; the missing direct edge
is replaced by a three-hop sparse path. These six cases are train-only. The
selected test clues are outside the prefix and no trusted negative edge labels
exist, so held-out coverage and admitted-edge precision are still unavailable;
the positive-only diagnostic threshold is not applied. The next runtime step is
the 36-video full-length extraction launched by
`cgbench_grounded_navigation/submit_l15_fixed_cohort.sh`. It freezes the
question-independent graph before hidden clue evaluation and requires 30–50
fully retained cases before any matched IWM run. Runtime artifacts remain
data-only and are not committed as scientific results.

## 9. Directory map

```text
steam_video_new/
├── README.md
├── problem-formulation-en.html
├── problem-formulation-zh.html
└── implicit_world_model/
    ├── l15_graph_navigator/        # IWM/planner contracts and data workflows
    ├── full_graph_iwm/             # single-cursor no-Top-K main-method path
    ├── cgbench_grounded_navigation/ # CG-Bench grounding and L1.5 smoke
    └── datasets/                    # versioned local data artifacts/reports

../factor_graph/                     # canonical GTSAM backup and experiments
../memory_graph/                     # L1/L1.5 graph construction primitives
../../Video_Skills/                  # external execution/runtime substrate
```

See component documentation:

- [`implicit_world_model/l15_graph_navigator/README.md`](implicit_world_model/l15_graph_navigator/README.md)
- [`implicit_world_model/full_graph_iwm/README.md`](implicit_world_model/full_graph_iwm/README.md)
- [`implicit_world_model/cgbench_grounded_navigation/README.md`](implicit_world_model/cgbench_grounded_navigation/README.md)
- [`../factor_graph/README.md`](../factor_graph/README.md)
- [`../memory_graph/README.md`](../memory_graph/README.md)

## 10. Immediate next gate

The repaired-graph two-video horizon-two diagnostic has now completed the fixed
`IWM / no-WM / shuffled-IWM / immediate-only / oracle` protocol and a strict
cache-only replay. It applies no heuristic Top-K and reports metrics separately.
The result is negative: IWM recall equals no-WM, while shuffled-IWM is higher;
action trajectories do diverge, so WM dependence exists but is not useful yet.

Do not begin model training until the remaining data-only chain is complete:

1. finish the question-independent full-video validation/test L1/L1.5 builds;
2. freeze their graph fingerprints and compile the held-out gate;
3. run the same five arms at both the two-read stress budget and a larger matched
   budget justified by the oracle clue count;
4. report transition prediction/realization confusion, action divergence,
   coverage, read efficiency, abstention and latency separately;
5. turn the observed false `support/advanced` and large-tie slices into grounded
   training/evaluation examples, without adding a question-conditioned Top-K.

Only after held-out candidate recall is adequate and shuffled-IWM is worse than
the intact IWM should the closed-loop interventions be treated as method evidence.
