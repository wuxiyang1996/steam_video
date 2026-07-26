# Full-Graph IWM Navigation

This package implements the main reasoning path over a fixed-capacity,
question-independent video memory. The current layer contract is deliberately
simple:

```text
L1   = grounded semantic evidence nodes + deterministic temporal backbone
L1.5 = embedding-derived soft nonlocal correlations plus sparse, grounded,
       categorical candidate hops over L1 nodes
L1.5-strict (optional) = separately verified identity/state/causal relations

question + current belief + current cursor + legal graph hops
  -> IWM predicts categorical observation / belief delta for each hop
  -> preference model compares imagined trajectories ordinally
  -> planner executes one real hop
  -> real evidence corrects belief, then the system replans
```

L1/L1.5 is the evidence and navigation substrate. It is not the learned
reasoning belief, the reward, or the main research contribution. GTSAM remains
an optional correction/diagnostic backend and is not required by this path.

## Non-negotiable boundary: graph correlation versus planner preference

These are two distinct relations:

```text
Correlation(node_i, node_j)
  -> constructs question-independent L1.5 navigation topology

Preference(trajectory_i, trajectory_j | belief, imagined transitions)
  -> chooses a reasoning action at runtime
```

Correlation determines which node-to-node hops exist; it does not decide which
hop wins for a question. The action compiler turns the fixed cursor-incident
topology into legal actions. The IWM predicts the categorical observation and
belief effect of each legal action, and only then may the planner choose the
next hop. The IWM/planner must not invent, verify, or reclassify L1.5 edges, and
the graph builder must not use planner preference or future questions to build
them.

Numeric embedding similarity and directional affinity are allowed graph
features because they are measured or calibrated construction signals. They
are not model-generated reward/utility/confidence, are not verified facts, and
must not directly rank the planner's winner. Strict identity/state/causal
relations remain a separately verified optional layer.

## L1 and L1.5 construction

`build_l1_l15_navigation_graph` consumes a persisted
`CausalTemporalOverlay` and performs the following question-independent steps:

1. Reuse its grounded L1 observations as semantic evidence nodes.
2. Derive local embedding redundancy from the persisted
   `Qwen/Qwen3-VL-Embedding-2B` sidecar.
3. Coalesce only adjacent near-duplicate semantic observations and apply
   priority-preserving fixed-capacity consolidation.
4. Rebuild a deterministic temporal chain over retained nodes.
5. Score every non-temporal retained-node pair with cosine similarity.
6. Collapse near-identical embeddings into semantic equivalence classes,
   represent recurrence as a temporal chain rather than a clique, and apply
   standardized sparsemax between classes. This is not Top-K.
7. Optionally run one question-independent caption/structured-node pass that
   proposes sparse categorical `caption_bridge`, `entity_candidate`,
   `change_candidate`, or `contrast_candidate` hops. Both endpoint evidence
   snippets must be exact substrings of their persisted L1 descriptors, and
   existing temporal/semantic pairs are removed.
8. Keep explicitly verified fact relations in `verified_relations`. Candidate
   hops remain navigation affordances and never become verified facts.

Every soft correlation stores endpoint cosine similarity plus navigation
affinities. These values come from embeddings, not an LLM. They are
similarity/attention features, not calibrated probabilities or verified facts.
Because cosine is symmetric, an admitted semantic correlation exposes both
directions; only genuinely asymmetric temporal/strict evidence may claim a
one-way relation. A frozen global admission policy may reject edges by a
calibrated threshold, but it cannot apply per-node Top-K.

Compilation also emits `l1_l15_correlation_pair_audit.json`. It records every
retained pair's decomposed features, structural exclusion/admission reason,
policy fingerprint, and source/retained L1 fingerprints without storing a
question or answer. The compiler verifies that the source overlay checksum and
source L1 fingerprint are unchanged.

Compilation now also emits `l1_l15_multichannel_pair_audit.json`. It is a
schema/audit extension, not a learned navigator. It may record independently
sourced `entity_correspondence`, `change`, and `contrast` descriptors, but only
when structured L1 metadata grounds them. Candidate/local track IDs, endpoint-
local state-change fields, or caption keywords are insufficient. These
descriptors never admit an edge by themselves, never become identity/state or
causal facts, and never rank actions.

This audit-only descriptor layer is distinct from the optional
`l1_l15_caption_candidates.json` runtime overlay. The latter is built once per
frozen video graph without question, answer, or clue access. It admits only
scoreless, evidence-backed categorical hops; it does not emit probability,
confidence, causal truth, identity truth, or planner preference. The graph
fingerprint covers the resulting candidate edges, so they cannot change after
the evaluation gate.

We intentionally do **not** train a separate L1.5 bridge encoder. The learned
model is the IWM itself:

```text
static L1/L1.5 hop + question + current belief
  -> IWM predicts categorical observation and belief delta
  -> planner compares imagined action trajectories
```

GT clue chains and grounded executed transitions are therefore reserved as IWM
supervision rather than used to learn a second graph selector.

The old `build_retained_graph_from_legacy_overlay` name remains only as a
compatibility alias. Passing a categorical relation evaluator to it raises an
error, because fact verification and generic navigation correlation are now
separate concerns.

After node embedding, the CG-Bench worker also persists this compiled view as
`l1_l15_navigation_graph.json` next to the source overlay and embedding
sidecar. Runtime compilation remains deterministic and the closed-loop gate
fingerprints the resulting graph.

## Legal actions

The action compiler is structural and deterministic; the IWM does not invent
actions. A separate categorical entry localizer inspects every visible
retained node's safe semantic address once and establishes a small set of
initial cursors. It emits no score or ranking, applies no Top-K, and must repair
or fail if its declared maximum frontier is exceeded. At the virtual root the
compiler exposes `START_AT(node)` only for these frozen anchors. At a real
cursor it exposes only:

- temporal forward/backward hops incident to the cursor;
- positive-direction L1.5 correlation hops incident to the cursor;
- permitted-direction categorical candidate hops incident to the cursor;
- backtracking to already acquired nodes;
- typed terminal controls, handled separately from evidence acquisition.

When unread navigation actions exist, `stop`/`abstain` do not compete inside
the IWM rollout. `answer` becomes active only from a corrected real belief that
is already `ready`; `abstain` remains a fail-closed controller outcome. This is
an action-lifecycle invariant, not a ranking heuristic.

There is no global all-node root action set, flat `current node x every unread
node` semantic-probe product, or candidate `inspect/verify` action in the main
navigation path. After the first real read, reasoning remains on the current
node's close temporal/L1.5 neighborhood.

## Multi-trajectory IWM and planner

`multi_trajectory.py` adds the forward-compatible main interface while keeping
the earlier single-cursor planner as an ablation. A `TrajectoryPool` stores
multiple competing hypotheses. Every active trajectory has its own cursor,
frontier, missing roles, contradictions and action history, while all active
trajectories share acquired real evidence and the remaining read budget.

For every active trajectory, the structural compiler emits all legal temporal,
correlation, backtrack and terminal expansions. No heuristic Top-K or beam is
applied. `GPTOSSMultiTrajectoryIWM` receives the complete pool and all
hypothesis-conditioned candidate contexts, including source/target semantic
keys, temporal span/relation, correlation channel and current belief. It
directly returns `unique`, `tie`, or `incomparable` preference plus predicted
lifecycle suggestions. Lifecycle suggestions are audit-only until a real
observation arrives.

The planner preserves the complete preferred trajectory frontier. If it contains
several first hops, a separate categorical evidence scheduler selects the one
real observation predicted to best discriminate the surviving hypotheses or
unlock their next grounded step. Scheduling one read does not delete any
trajectory. The observation is then broadcast to every active
trajectory. A pluggable categorical evidence assessor may mark each trajectory
supported, contradicted, inconclusive or completed; a belief updater may be
latent or factor-graph-backed. Exact duplicate hypotheses are structurally
merged, while tied or incomparable non-equivalent hypotheses remain active.
Pool size is never controlled by a numeric score or fixed K.

```text
TrajectoryPool + L1/L1.5 graph
  -> all legal hypothesis-conditioned expansions
  -> direct categorical IWM preference
  -> one shared real read
  -> broadcast correction across active trajectories
  -> exact consolidation / preserve alternatives
  -> replan
```

## Leakage and model-output contracts

- Entry localization sees all compact semantic addresses once, but not unread
  evidence values. Each IWM step sees only the legal local closure: acquired
  nodes, current cursor, legal action endpoints, and induced edges. A real
  executor reveals an evidence value only after the hop.
- Imagined evidence remains predicted-only and never enters persistent belief.
- The backend binds each observation descriptor exactly to the selected
  action target's visible semantic key. A proxy LLM predicts only the
  categorical outcome/belief delta and emits ordinal pairwise labels or
  setwise `unique`, `tie`, and
  `incomparable` decisions.
  Model-generated reward, utility, probability, confidence, and Q-values are
  rejected.
- The preference model receives anonymous predicted consequences rather than
  action/node/timestamp IDs, preventing lexical or construction-order shortcuts.
- All retained addresses are considered by the categorical localizer; all
  legal local actions are considered by the IWM/planner. Neither stage uses an
  embedding Top-K or hand-written score to select a winner. Pair comparisons
  are request-batched to keep model outputs bounded without sampling pairs.
- A non-unique trajectory frontier remains alive. When configured, the
  categorical evidence scheduler chooses one observation action without a
  score or construction-order tie-break; a genuinely incomparable scheduler
  result still fails closed. Imagined consequence is never used as
  final-answer evidence.
- A real role resolution is persisted as `(role, acquired_node_id)`. It cannot
  cite imagined evidence. Multi-role questions cannot become answerable from
  only one real observation, even if the categorical updater over-resolves its
  first read; readiness requires complete role lineage and at least two real
  observations.
- Actions without a real read (`stop`, `answer`, `abstain`, `backtrack`) have
  deterministic `inconclusive + unchanged` dynamics, bypass model inference,
  and cannot hallucinate an observation.
- Repeated hypothesis-conditioned outcomes are serialized as exact descriptor
  equivalence classes. Every member alias and hypothesis is retained, so this
  reduces context without Top-K, sampling, or information loss.

## Components

```text
contracts.py       graph/action/transition/preference contracts
graph_adapter.py   persisted overlay -> retained semantic L1 + soft L1.5
action_compiler.py cursor-local legal hops and real evidence execution
localization.py    scoreless categorical global-address entry localization
local_choice_data.py blinded entry-anchor predictions + separate grounded labels
model_input.py     leakage-safe legal-local-closure IWM view
planner.py         horizon-1/2 IWM rollout and ordinal partial-order selection
multi_trajectory.py direct multi-hypothesis IWM, shared execution and pool lifecycle
gpt_oss.py         categorical GPT-OSS IWM and batched preference adapters
caption_candidates.py question-independent grounded categorical hop builder
reactive.py        matched no-world-model direct-policy baseline
interventions.py   shuffled/frozen world-model controls
closed_loop.py     execute/read/correct/replan loop and evaluator-only metrics
cgbench_pilot.py   fixed-case compile gate and matched-budget experiment
```

## Canonical V1 responsibility boundary

The V1 method assigns persistent state and imagined dynamics to different
owners:

```text
Planner-owned persistent trajectory pool
  -> generate every legal short reasoning chain from every active trajectory
  -> IWM predicts the outcome of each proposed chain
  -> Planner compares predicted chain outcomes
  -> execute one uniquely preferred first hop
  -> broadcast the real observation to affected trajectories
  -> correct/support/reject/merge/branch persistent trajectories
  -> discard the old imagined chains and replan
```

The **Planner**, not the IWM, owns competing reasoning trajectories. A
persistent trajectory contains its hypothesis, grounded belief, acquired real
evidence, executed action history, parent lineage and lifecycle status. The
Planner may retain multiple unresolved trajectories even when only one real
read is executed in a step.

The **IWM** receives a proposed horizon-one/two reasoning chain together with
its current grounded trajectory context. It predicts that chain's future
observation and belief outcome. V1 represents this outcome with categorical
descriptors; it emits no reward, probability, confidence or utility. The IWM
does not delete trajectories, persist imagined evidence, execute graph actions
or choose a final answer.

The **preference model** maps predicted complete-chain outcomes to
`prefer_left`, `prefer_right`, `tie` or `incomparable`. The deterministic
Planner aggregates that partial order, executes only a unique shared first hop,
and otherwise abstains. Preference is therefore a Planner input, not a
replacement for IWM dynamics prediction.

Imagined and persistent state must never be merged by averaging or by copying
an imagined result into real belief:

```text
persistent belief + predicted chain outcome -> temporary imagined belief
real read + persistent belief              -> corrected persistent belief
```

After a real read, every stale imagined continuation is discarded. New chains
are generated from the corrected persistent trajectory pool.

### V1 implementation status and remaining gap (2026-07-23)

The repository now connects both sides of this design:

- `multi_trajectory.py` owns a persistent hypothesis pool, shared real reads,
  lifecycle updates, exact-equivalence merging and closed-loop replanning;
- `multi_trajectory_rollout.py` predicts and compares complete horizon-one/two
  reasoning chains;
- the multi-trajectory CG-Bench runner initializes one trajectory from each
  public answer choice and supports latent or optional GTSAM real correction;
- the runner calls the scoreless entry localizer once per case and injects the
  same `localized_entry_node_ids` into every matched model-backed arm. A
  localization contract error produces an explicit fail-closed abstention;
- real belief correction receives the trajectory hypothesis, so the same
  executed observation may produce different persistent corrections for
  competing interpretations;
- the IWM predicts every legal chain separately under every persistent
  trajectory. All continuations with the same executable first hop are then
  represented as one joint first-hop action tree containing every
  hypothesis-conditioned one-/two-hop outcome. This removes duplicate
  execution candidates without dropping a hypothesis, outcome or action and
  without applying Top-K;
- the focused multi-trajectory suite reports 23 passed tests. The combined
  focused IWM, multi-trajectory and GTSAM-backup suite reports 53 passed and 1
  dependency-gated skip.

The frozen 34-case result below validates the earlier single-persistent-belief
closed loop. It does **not** validate persistent multi-trajectory reasoning.
The remaining promotion gap is empirical: run a real model-backed
multi-trajectory smoke, inspect chain/outcome coverage and correction traces,
then repeat the five matched arms on the frozen cohort. Transition calibration,
answer accuracy and a stable delayed-planning advantage remain unestablished.

The direct multi-trajectory scaffolding provides:

- deterministic initialization from all supplied hypotheses or answer
  interpretations, without selecting a subset;
- dynamic branching from a parent trajectory after grounded evidence, retaining
  every supplied new interpretation and its parent lineage;
- one complete legal expansion set per active trajectory;
- direct GPT-compatible categorical preference over the joint expansion set;
- one shared read when preferred expansions agree on the same action;
- batched real-observation assessment across every active hypothesis;
- independent trajectory support/counterevidence/inconclusive/completed states;
- exact structural consolidation with merged records retained for audit;
- a closed-loop trace that preserves the full pool at every decision.

A July 2026 GPT-5-mini schema smoke used two competing door-opening
hypotheses and six joint expansions. The first response used an invalid
`incomparable + preferred aliases` combination, the strict repair call converted
it to a valid tie, and both hypotheses selected the same real read. One executed
observation was broadcast once; the real assessor marked the orange-person
hypothesis `supported` and the alternative `contradicted`, with identical shared
evidence and read budget in both trajectory records. No Top-K was applied.

Hypothesis generation is deliberately outside this layer: callers may supply
all answer choices, externally enumerated interpretations, or hypotheses from a
separate categorical proposer. The pool does not silently reduce that input.
GTSAM can implement the same real-belief update boundary, but is not required by
the direct IWM or planner.

Minimal direct-IWM usage:

```python
pool = initialize_trajectory_pool(
    initial_belief,
    hypotheses=("identity hypothesis A", "identity hypothesis B"),
)
planner = MultiTrajectoryIWMPlanner(GPTOSSMultiTrajectoryIWM(client))
trace = run_multi_trajectory_closed_loop(
    pool,
    graph,
    planner,
    belief_updater=real_belief_backend,  # latent or optional factor graph adapter
    assessor=GPTOSSRealTrajectoryEvidenceAssessor(client),
)
```

## Complete multi-trajectory rollout implementation

`multi_trajectory_rollout.py` is the experiment-facing planner. It treats the
next executed evidence read as the decision unit. Route-specific actions that
read the same L1 target are one shared action, while their temporal/correlation
provenance remains in the audit trace.

For every trajectory-specific legal first action, the IWM predicts a
categorical observation and belief delta. At horizon two it projects a
temporary imagined belief, compiles every legal second hop and predicts the
complete chain outcome. The Planner groups all continuations under their shared
executable first hop, retaining every hypothesis-conditioned chain in that
first-hop action tree:

```text
every competing trajectory + its current real belief
  -> every legal first action
  -> one categorical first transition per trajectory/action
  -> every legal second action under that imagined trajectory belief
  -> one categorical outcome per complete trajectory/chain
  -> group all continuations into shared-first-hop action trees without pruning
  -> exhaustive categorical partial order over complete first-hop trees
  -> execute one first action only when the preferred execution key is unique
  -> broadcast real evidence, correct every trajectory, and replan
```

This representation compares the decision that is actually executed now while
preserving distinct hypothesis-conditioned second-hop branches. It is not
Top-K: every legal first and second action remains visible, every
hypothesis-conditioned outcome remains attached to its joint tree, and every
joint tree participates in the partial order. Transport batching only splits
complete prediction or comparison sets across requests. If a configured
complete-pair budget is too small, the planner explicitly abstains with
`rollout_abstain_complete_comparison_budget_exceeded`; it never runs a partial
tournament.

One real read also updates the cursor and action history of every persistent
path, because all paths executed the same shared navigation action. Subsequent
semantic support, counterevidence and role correction remain independently
hypothesis-conditioned. The earlier behavior updated only the source path's
cursor; after one read that produced different legal action spaces and partial
joint chains. A two-cycle structured-9B regression test now guards this
boundary.

`multi_trajectory_cgbench.py` provides the end-to-end CG-Bench CLI. Public
answer choices initialize the trajectory pool. Hidden answer and clue fields
are joined only after execution for evaluation. A separate categorical terminal
selector sees only acquired real evidence and the final trajectory pool.

The matched arms are:

- `world_model_guided`: complete horizon-two action trees;
- `no_world_model`: reactive preference with imagined transitions removed;
- `shuffled_world_model_prediction`: the same actions and planner with
  categorical transition descriptors permuted;
- `immediate_effect_only`: horizon-one categorical transitions;
- `oracle_clue_ceiling`: hidden-alignment evaluator upper bound only.

Metrics are reported separately: answer accuracy, clue recall, reads, read
efficiency, action divergence, delayed success, correct-hypothesis survival,
false correct-hypothesis elimination, abstention, and complete-comparison budget
failure. No scalar reward or aggregate pass score is created.

Compile the frozen gate without a model call:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cgbench \
  --dataset /path/to/navigation_dataset.json \
  --hidden-key /path/to/l15_candidate_alignment.hidden_key.json \
  --selection /path/to/frozen_selection.json \
  --graph-root /path/to/frozen_graphs \
  --mode compile-gate \
  --output /path/to/multi_trajectory_gate.json
```

Run the matched pilot with GPT-OSS-120B:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cgbench \
  --dataset /path/to/navigation_dataset.json \
  --hidden-key /path/to/l15_candidate_alignment.hidden_key.json \
  --selection /path/to/frozen_selection.json \
  --graph-root /path/to/frozen_graphs \
  --mode run \
  --keys-py /fs/gamma-projects/vlm-robot/keys.py \
  --model openai/gpt-oss-120b \
  --response-cache /path/to/responses.json \
  --question-role-cache /path/to/roles.json \
  --output /path/to/multi_trajectory_matched.json
```

`--belief-backend latent` is the main method. `gtsam_backup` and
`gtsam_always` use `gtsam_backup.py` only after executed real reads. GTSAM
numeric state never enters a prompt or preference, and missing GTSAM raises an
explicit error rather than changing the planner.

The July 2026 real GPT-OSS-120B schema smoke used two hypotheses, two L1 nodes,
four shared first-action trees, ten complete transition predictions and six
pairwise comparisons. All three API responses passed categorical coverage with
no Top-K. The model retained multiple preferred first actions and the planner
correctly abstained instead of using compiler order. On an actual 64-node
CG-Bench smoke graph with eight public choices, deterministic scale validation
produced 66 shared first-action trees, 566 horizon-two transition predictions,
and all 2,145 first-action pairs.

The 8-video frozen smoke gate currently has two runnable cases. A newer
2-video held-out build is not runnable at capacity 64 because consolidation
discarded part of its evaluator-only clue coverage. The gate blocks it; this is
a remaining data-selection issue, not a planner fallback.

Supporting L1 code lives in `memory_graph/`:

- `adaptive_windowing.py`: surprise-driven write boundaries; the present
  OpenCV representation is explicitly a smoke fallback;
- `selectstream_policy.py` and `consolidation.py`: priority-preserving
  keep/merge/evict with lineage and temporal rebuilding;
- `soft_correlation.py`: Qwen embedding soft L1.5 construction;
- `correlation_overlay.py`: optional categorical fact-relation audit and legacy
  relation projection, not the main navigation edge builder.

## Minimal use

```python
from steam_video_new.implicit_world_model.full_graph_iwm import (
    CursorBeliefState,
    FullGraphIWMPlanner,
    GraphActionCompiler,
    build_l1_l15_navigation_graph,
)

graph = build_l1_l15_navigation_graph(overlay, capacity=64)
belief = CursorBeliefState(
    belief_id="belief:0",
    question=question,
    remaining_reads=8,
)
legal_actions = GraphActionCompiler().compile(belief, graph)
decision = FullGraphIWMPlanner(
    world_model,
    preference_model,
    horizon=1,
    max_trajectory_pairs=4096,
).plan(belief, graph)
```

Compile one graph without calling an external model:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What evidence connects the events?" \
  --capacity 64 --mode compile-only \
  --output /tmp/full_graph_compile.json
```

Run the fixed CG-Bench gate and GPT-OSS pilot with:

```bash
bash steam_video_new/implicit_world_model/full_graph_iwm/run_cgbench_pilot_after_graph.sh
```

For long horizon-two data-gathering runs, use all three persistence layers:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot \
  ... --rollout-horizon 2 --setwise-preference \
  --transition-cache /path/to/transitions.json \
  --response-cache /path/to/categorical-responses.json \
  --progress-output /path/to/run.progress.json \
  --arm world_model_guided --arm no_world_model \
  --arm shuffled_world_model_prediction --arm immediate_effect_only \
  --arm oracle_clue_ceiling --output /path/to/run.json
```

If interrupted, rerun the identical command with `--resume-progress`. Keep cache
mode `record` while gathering missing safe states; switch to `replay` only after
the frozen evaluation state space is complete. Replay is fail-closed on any
unseen transition, question-role decomposition or categorical response.

## July 2026 proxy result

The first fixed puppy case establishes reachability but not IWM readiness:

- the repaired evaluator-only shortest-path oracle covers 4/4 clues with four
  reads via temporal and L1.5 correlation hops;
- `openai/gpt-5-mini` completes a six-read categorical closed loop but covers
  0/4 clues, repeatedly preferring semantically related pre-event washing
  nodes over the requested post-drench state;
- its matched no-WM direct-policy arm uniquely chooses `abstain` before any
  read under a two-read budget; the IWM changes behavior, but has not yet shown
  a coverage or answer-quality gain;
- `qwen/qwen3.7-max` covers 1/4 clues in a two-read diagnostic run, but takes
  about 332 seconds, so it is not an online implementation as invoked here;
- no answer head was evaluated and no training was performed.

Therefore L1/L1.5 is executable and contains a valid route on this case, while
the zero-shot large-model IWM is only a data-gathering proxy. The data/runtime
step is now implemented without a heuristic question-conditioned Top-K:

- `transition_descriptor_cache.py` exports 670 grounded executed transitions
  as a video-disjoint train/validation/test corpus. It retains real observation
  descriptors, categorical belief deltas, embeddings and provenance, while
  excluding question-conditioned relevance prose and hidden evaluator fields;
- `PersistentTransitionCacheWorldModel` records action-conditioned categorical
  predictions keyed by the complete safe visible graph, real belief, action and
  imagined prefix. Replay fails closed on a miss and cannot perform same-case
  GT lookup;
- `PersistentCategoricalResponseCacheClient` additionally freezes categorical
  model decisions across interrupted runs. It stores request hashes and strict
  JSON outputs, not prompts, questions, clues or answers;
- long experiments atomically checkpoint after each case/arm and can resume from
  that journal. A hard wall-clock timeout also covers slowly chunked provider
  responses;
- bounded setwise comparison now batches several independent complete candidate
  groups in one request. Every legal trajectory is still judged; grouping is a
  transport optimization, not candidate pruning, and ties/incomparability are
  preserved.

The fixed matched protocol is `IWM / no-WM / shuffled-IWM / immediate-only /
oracle`, with the same graph fingerprint, read budget and legal-action compiler.
Metrics remain separate: clue coverage, read efficiency, action divergence,
abstention, delayed success and latency. No scalar reward or aggregate boolean
is synthesized. The present two-video run is a development smoke while the
video-disjoint validation/test full-video graphs are being built; it is not a
formal method result.

For capacity-256 full graphs, horizon-two planning must not materialize the
complete first-hop × second-hop Cartesian product before preference. The current
planner therefore uses a two-stage learned frontier: the IWM predicts every
legal first hop, categorical setwise preference identifies the non-dominated
first-hop frontier, and only that frontier is expanded for the second imagined
step. This is not heuristic Top-K: every first hop is judged, no fixed number of
survivors is imposed, ties remain ties, and an unresolved frontier that exceeds
the explicit rollout budget causes fail-closed abstention. The planner then
executes only the preferred first real action, corrects belief from real
evidence, and replans. Transport batches only split the same complete candidate
set; they do not remove actions.

The formal runner accepts `--fixed-cohort-gate`; only locked case IDs at the
gate's selected capacity can enter an experiment. A first capacity-256 transport
attempt with 48 actions/request failed strict JSON coverage and was retained as
a protocol-failure artifact. The corrected pilot uses 12 actions/request and at
most two graph contexts/request; this changes serving granularity only.

### Capacity-256 delayed-case proxy diagnostic (2026-07-22)

The first complete five-arm run uses the lowest passing cohort capacity (256),
one locked structural-delayed case, horizon two, a matched two-read budget and
`qwen/qwen3.6-flash` as a categorical proxy IWM. The transport validation used
24 actions/request; every recorded transition batch had complete action
coverage. The explicit imagined-transition budget is 512 and never removes a
candidate: an oversized non-dominated frontier must abstain.

The intact IWM reduced 258 initial first hops to 82 (31.8% retained), while the
shuffled-IWM retained all 258. Thus IWM predictions measurably affect the
planner frontier, but the intact frontier is still too ambiguous to expand
within budget. IWM, no-WM, shuffled-IWM and immediate-only all abstained before
a real read; the evaluator-only oracle read 2 nodes and covered 1/2 clues. This
is a clean negative method result, not a transport or graph-gate failure. It
localizes the next data target to categorical preference among the surviving
first-hop ties/incomparables; heuristic Top-K or order-based tie breaking must
not be added.

### Local-neighborhood repair diagnostic (2026-07-22)

The repaired path separates one-time global address localization from
multi-hop reasoning. On the same delayed case, the localizer examined all 256
safe addresses and returned three frozen anchors. The initial legal action set
therefore fell from 258 to five (three reads plus stop/abstain); after one real
read the cursor exposed nine local actions. Per-step IWM input contained only
the legal local closure. No action was discarded by a heuristic Top-K.

This repair made the WM intervention causal and inspectable: intact IWM chose a
real read, while no-WM abstained and shuffled-IWM stopped. It did not yet
improve clue recall. The localizer covered 1/2 hidden clues, including the
1263-second missile node, but the IWM predicted that an unrelated 105-second
firearm node would resolve `agent/action/victim` and selected it. The executed
read was inconclusive. Horizon two and immediate-only made the same first-hop
choice, so there is no delayed-planning advantage yet.

This is a transition-calibration failure, not a graph-density or transport
failure. The next supervision target is grounded executed comparisons between
the remaining local anchors, with predicted-versus-realized role changes kept
explicit. We must not hide the error using clue-aware routing, numeric reward,
manual role-count ranking, Top-K, or deterministic tie breaking.

`local_choice_data.py` implements that boundary without adding a routing
heuristic. Future planner traces retain every initial anchor prediction in
`FullGraphPlanDecision.initial_trajectories`; older traces can be reconstructed
from their frozen transition cache. The exporter writes two separate tasks:

1. blinded categorical comparisons among every localized anchor;
2. executed predicted-versus-grounded navigation-delta corrections.

GT clue overlap is applied only after planning and stored in a separate hidden
key. A clue-overlapping anchor may be strictly preferred to a non-overlapping
anchor for the dataset navigation task, but the latter remains
`not_established`, never a standalone semantic negative. The correction target
is explicitly scoped to clue-coverage navigation progress; it is not an
identity, causal, or general observation-truth label.

The current one-case packet has three anchors, three complete pair comparisons
(two strict and one incomparable), and one executed transition mismatch. It is
a test-split diagnostic with `training_ready=false`; it must not be used for
post-training. A train-split multi-video packet is required before any SFT,
DPO/OPD, or RL experiment.

### Frozen 34-case zero-shot verification (2026-07-22)

The first complete validation/test verification of the earlier
single-persistent-belief planner uses 34 locked cases from 27 videos, capacity
256, two real reads, horizon two, and
`qwen/qwen3.6-flash` as an untrained proxy IWM. It contains all 170 case-arm
slots and compares intact IWM, no-WM, shuffled predictions, immediate-only and
the evaluator-only oracle under the same graph fingerprints and read budget.

| Arm | Mean clue recall | Complete coverage | Mean reads | Abstain | Delayed success |
|---|---:|---:|---:|---:|---:|
| IWM horizon two | 0.201 | 0.118 | 0.912 | 0.618 | 0.088 |
| no-WM | 0.000 | 0.000 | 0.029 | 1.000 | 0.000 |
| shuffled-IWM | 0.098 | 0.059 | 0.941 | 0.471 | 0.059 |
| immediate-only | 0.137 | 0.088 | 0.676 | 0.765 | 0.059 |
| oracle ceiling | 0.730 | 0.471 | 1.853 | 0.000 | 0.324 |

The paired intact-minus-control clue-recall deltas are +0.201 versus no-WM,
+0.103 versus shuffled and +0.064 versus immediate-only. Intact is better/worse
on 7/3 cases versus shuffled and 3/0 versus immediate-only; the remaining cases
tie. Action divergence is 64.7% versus shuffled and 14.7% versus
immediate-only. This establishes a **provisional navigation effect** and real
dependence on IWM outputs, but the small number of non-tied improvements does
not establish a strong delayed-planning advantage.

The bottlenecks are now measured directly:

- entry localization exposes at least one clue in 55.9% of cases and has mean
  clue recall 0.432 after counting contract failures as zero;
- conditional on a clue being present among anchors, intact IWM's first real
  read hits one in 47.4% of cases;
- only 1/31 executed intact transitions has an exact categorical outcome match
  (3.2%), and exact belief-delta match is 0/31;
- two cases fail the localization contract and fail closed; two individual
  arms return invalid setwise schemas and are conservatively scored as
  abstentions rather than removed from the denominator.

Validation and test independently retain an intact mean clue recall of 0.200
and 0.202, respectively. No hidden clue or answer field enters model requests,
no scalar reward is created, no Top-K is applied, and `training_performed`
remains false.

The correct conclusion is therefore: the single-belief IWM + Planner has passed
full-cohort integration, causal-dependence and provisional navigation-effect
verification, but has **not** passed transition calibration, robust
localization, strong horizon-two advantage, answer accuracy, persistent
multi-trajectory validation or production gates.

The same run is reproducibly replayable from the categorical response and
action-conditioned transition caches. GPT-OSS-120B remains a compatible data
gathering backend, but its measured OpenRouter latency was about one minute per
12-action transition batch, so it is not the default rapid proxy loop.

### Two-video development diagnostic

The current-code record and fail-closed replay both completed 10/10 case-arm
runs with zero errors. Replay served all 44 categorical calls from the frozen
response cache and every transition request was a cache hit. With two real
reads per case, mean clue recall was:

| Arm | Recall | Mean reads | Read efficiency | Abstain |
|---|---:|---:|---:|---:|
| IWM horizon two | 0.125 | 2.0 | 0.25 | 0.0 |
| no-WM direct policy | 0.125 | 1.0 | 0.50 | 0.5 |
| shuffled-IWM | 0.250 | 2.0 | 0.50 | 0.0 |
| immediate-only | 0.125 | 2.0 | 0.25 | 0.0 |
| evaluator-only oracle | 0.625 | 2.0 | 1.25 | 0.0 |

This is a negative IWM diagnostic, not a success claim. IWM actions diverged
from every comparison arm, so the planner genuinely depends on imagined
transitions; however, that dependence did not improve retrieval. Failure
inspection found predicted `support/advanced` transitions whose executed reads
were inconclusive, plus a large categorical tie in one horizon-two tournament.
Shuffling predictions improved one case, which directly fails the desired
causal ordering. The two-read oracle ceiling also shows that full clue coverage
is impossible under this diagnostic budget. The next evidential gate is the
same five-arm protocol on frozen full-video validation/test graphs, with the
low-budget result and a larger matched read budget reported separately.

The executed-read confusion slice explains the failure more directly. For the
intact IWM's four selected reads, predicted progress was `advanced` four times,
but realized progress advanced once and was unchanged three times. Its selected
observation categories were one `identity_evidence` and three `support`; the
real evaluator observed one support and three inconclusive reads. Immediate-only
also predicted `advanced` on all four reads but realized only one advance, and
predicted `ready` twice while the corrected belief remained `not_ready` both
times. These categorical false positives, rather than missing numeric score
calibration, are the first training/data target.

Post-run inspection found three implementation-level confounds in that result:

1. batched model rows could emit an observation descriptor inconsistent with
   the row's action target;
2. a setwise tie could be executed by stable compiler order when the diagnostic
   flag was enabled;
3. the real-belief updater could mark every question role resolved without
   retaining which executed observation grounded each role.

The current contract fixes all three. Descriptors are target-bound by the
backend and invariant to transport batch/order; tied first hops abstain; and
real belief carries auditable evidence lineage with a multi-observation
readiness gate. Transition-cache schema `v0.3` invalidates the earlier
action-descriptor cache. Therefore the table above remains a historical failure
diagnostic and must not be reused as a post-fix result.

A one-case post-fix safety smoke (`openai/gpt-5-mini`, puppy case, horizon one,
two-read budget) completed with zero runtime errors. All 34 cached descriptors
were backend-bound and the raw model response contained no descriptor field.
The categorical tournament retained two distinct first hops, so the planner
abstained with zero reads instead of executing compiler order. This confirms the
integrity fixes, but it is not a navigation improvement: the remaining failure
is now cleanly isolated to non-unique zero-shot preference. A larger matched
evaluation should proceed only after the transition/preference model can resolve
such cases from grounded supervision rather than an order fallback.

The grounded survivor exporter in `survivor_preference_data.py` now separates
blinded model inputs from post-planning GT labels. On the two fixed development
videos it produced 44 strict categorical contrasts: one final survivor tie and
43 exhaustive survivor-versus-grounded counterfactual pairs. Labels are balanced
between left and right by construction order (22 each), so there is no fixed-side
shortcut. Two pairs have different grounded relevance but exactly the same
imagined outcome and belief delta. The other 42 strict contrasts show that exact
collision is not the only issue: the IWM often over-credits an isolated role
such as generic signage or vehicle motion without satisfying joint identity and
temporal requirements.

This isolates four current failure sources:

1. action-conditioned transition descriptors can still be semantically
   under-differentiated even though their target identity is now correct;
2. role resolution lacks learned joint prerequisites, especially
   identity-before-attribute and anchor-before-after-state constraints;
3. a preference comparator cannot recover distinctions already erased by the
   imagined transition;
4. the bounded setwise tournament is complete-coverage but not yet proven
   permutation/bracket invariant.

The next data gate is therefore transition/preference accuracy on the frozen
blinded packet plus a tournament permutation audit. It is not a Top-K or
stable-tie change.

GPT-5-mini was then evaluated on all 44 blinded pairs. It reached 21/44
(47.7%). The original packet elicited 41 left preferences; after mechanically
swapping every pair it elicited 42 right preferences. Once mapped back to the
same semantic orientation, 43/44 predictions were consistent. This rules out a
simple left-position explanation: the comparator follows the original IWM
survivor regardless of side. The main error is the survivor's overstated
transition descriptor, while the preference stage consistently propagates that
upstream error. Training or replacing only the preference head is therefore not
the next priority.

Compile and audit frozen L1.5 graphs without calling an LLM:

```bash
python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker \
  audit-correlations \
  --selection /path/to/l15_graph_smoke_selection.json \
  --graph-root /path/to/l15_graph_smoke_v1 \
  --dataset /path/to/navigation_dataset.qwen_grounded_embedded.json \
  --hidden-input /path/to/terminal_targets.hidden_key.json \
  --report /path/to/l15_correlation_evaluation.json \
  --details /path/to/l15_correlation_evaluation.hidden_key.json \
  --memory-capacity 64 --max-path-hops 8
```

The shell entry point uses `/fs/gamma-projects/vlm-robot/keys.py` for the
OpenRouter client by default, horizon one, capacity 64, read budget 8, and an
exhaustive 4096-pair resource limit. Pairwise calls are chunked; no pair is
silently dropped. Horizon two should be enabled only after inspecting the
compiled trajectory count because exhaustive cross-trajectory comparison can
still be expensive.

## Validation status

### Code-complete versus empirically validated

The scoped V1 implementation is code-complete: persistent competing
hypotheses, complete legal-action expansion, horizon-one/two categorical IWM
rollouts, multi-trajectory preference, one-hop execution, real-evidence
correction, stale-rollout invalidation, replanning, five matched intervention
arms, isolated GTSAM backup, persistent caches and resumable frozen-cohort
evaluation are all implemented and covered by end-to-end model-backed smoke
tests. No remaining interface placeholder prevents the intended
`IWM -> Planner -> real read -> corrected belief -> IWM -> Planner` loop from
running.

The scientific claim is not complete. The remaining work is to measure and
improve transition calibration, preference identifiability, delayed-planning
advantage, clue recall and answer accuracy on the locked cohort. In this
document, **code-complete** must not be interpreted as trained, baseline-beating
or production-ready.

The frozen train collection now has 31/31 completed runnable cases across the
capacity-192 and capacity-384 graph cohorts, with zero runtime errors and zero
fail-closed method failures. This clears the infrastructure
gate but is a negative scientific result: GPT-5-mini WM-guided answer accuracy
is 0.129 versus 0.161 for no-WM, and clue recall is 0.145 versus 0.266. Of the
763 executed WM-conditioned transition rows, 79% of realized outcomes are
`inconclusive`; micro exact accuracy therefore overstates calibration. The
pooled outcome/progress balanced accuracies are 0.332/0.419, while categorical
over-credit false-discovery rates are 0.759/0.857. Predicted `ready` is not
confirmed by the corrected belief in 0.974 of predicted-ready rows. These
figures identify action-conditioned transition over-crediting as the immediate
data/model target; they do not justify a planner tie-break, heuristic Top-K or
hand-written utility.

Runtime reports now keep micro exact accuracy only as a descriptive statistic
and additionally emit pooled confusion matrices, macro-recall balanced
accuracy, and categorical over-credit false-discovery rates for observation,
progress and answerability. This prevents a shuffled model that mostly predicts
the dominant `inconclusive/not_ready` controls from appearing calibrated.
The stricter entry-localization repair was exercised by three of the nine
capacity-384 cases (12 repeated arm audits), each requiring exactly one repair;
all cases then completed without weakening the alias/role contract.

The latest no-training GPT-5-mini H2 smoke also makes the evaluator boundary
explicit. A compile-time oracle that may start at any retained node measures the
L1/L1.5 substrate ceiling, not the ceiling available to the runtime method.
The executed oracle now shares the exact model-localized entry frontier and
read budget with the learned arms. Across three cases its mean clue ceiling is
0.833 rather than the compile-time 1.0; one case is entry-unreachable under the
two-read contract. Such cases must fail the localized-entry preflight before
being used to claim an IWM navigation advantage.

All five arms then completed without runtime or method failures. The intact IWM
retained every hypothesis-conditioned joint outcome, scheduled real reads, and
produced post-read belief divergence, but its mean clue recall was 0.333, tied
with no-WM and shuffled-IWM and below immediate-only at 0.667. Its categorical
observation/progress calibration was 0.305/0.218. This is a working closed-loop
implementation and a negative zero-shot result: coarse address descriptors are
often over-credited (for example, `hand holds object` is imagined to resolve
all roles). The next intervention is grounded transition and discrimination
supervision, not Top-K, a fixed tie-break, or another hand-written utility.

The model-backed transition transport now losslessly factors requests by exact
action sequence. One shared legal action and address-level L1/L1.5 descriptor
is sent once, while every answer hypothesis keeps its own belief, imagined
prefix and categorical outcome row. The planner restores the original request
order and audits both the conditioned-outcome count and shared-action-group
count. This changes neither horizon, legal actions, hypotheses nor the matched
read budget, and applies no Top-K. On OpenRouter `qwen/qwen3.6-flash`, two
shared-action groups in one response exceeded the provider's stable structured
output envelope; one group per request is therefore the measured safe default.
The prompt also explicitly distinguishes “no real evidence acquired yet” from
an imagined read returning `empty`: an unread target's semantic key is an
address-level clue for predicting the result of the read, not acquired
evidence and not evidence of emptiness.

A post-fix grouped smoke retained 246/246 hypothesis-conditioned transition
outcomes, consolidated them into 41 complete joint chains and left a preferred
frontier of two chains with different first hops. The categorical preference
model returned a tie, and the planner therefore abstained. This is the intended
fail-closed behavior: it verifies that the planner consumes the IWM frontier
without silently adding Top-K or candidate-order tie breaking, while showing
that the remaining two-way preference must be resolved through better grounded
model supervision or a better model rather than another planner rule.

The first real `qwen/qwen3.6-flash` persistent multi-trajectory smoke now
completes two horizon-two planning/read/correction cycles. It retains all
369 generated hypothesis-conditioned outcomes across 77 joint chains and
produces two distinct corrected belief groups after each real read. A cached
five-arm replay confirms action dependence: no-WM diverges at step zero, and
shuffled-IWM plus immediate-only diverge from the intact IWM at step one.
However, all model-backed arms miss every hidden clue and abstain; the oracle
covers two of three clues with the same two-read budget. Thus the mechanism is
wired correctly but the zero-shot transition model is badly miscalibrated. The
resumable frozen 34-case runner records each case independently and aggregates
transition confusion, answer accuracy, delayed success, belief divergence, and
action divergence without applying Top-K.

The regression suite covers semantic duplicate coalescing, recurrence-chain
sparsification, temporal-pair exclusion, directional affinity legality,
fixed-capacity materialization, hidden-node and unread-value isolation,
world-model-dependent delayed-hop selection, intervention arms, real-read
replanning, and resource abstention.

The current eight persisted CG-Bench smoke graphs compile with 15–64 retained
nodes, 22–133 L1.5 edges, and a maximum emitted degree of 7–17. The previously
pathological climbing video fell from a dense near-clique to 15 retained nodes
and 22 correlation edges. All source L1 artifacts remain checksum-identical and
the build makes zero per-pair LLM calls. Of six consecutive clue bridges that
fall inside the 120-second prefix, five have a direct graph connection and all
six are reachable within eight hops; one depends on a three-hop sparse path.
All six eligible bridges are from the train split. The four selected test
videos contribute only out-of-horizon clues, and there are no trusted negative
edge labels, so held-out coverage and admitted-edge precision remain
unavailable. The diagnostic train-positive threshold is not applied to the
graph. These are structural/evaluator smoke results, not evidence of navigation
quality; the matched closed-loop experiment must still be rerun on the new
graph fingerprint.

## Zero-shot information-regime diagnostic

Zero-shot IWM evaluation now separates three questions that were previously
confounded:

- `semantic_address`: the pre-read model receives the legal action, temporal or
  correlation relation, address-level semantic key, structural tags and time
  span. The unread evidence value remains hidden.
- `rich_question_independent`: the model additionally receives the frozen L1
  text and an allowlisted structured descriptor. The graph must be
  question-independent; hidden clue intervals, terminal answers and post-read
  VLM observations remain unavailable.
- post-read oracle observation: an evaluator-only analysis of the existing real
  belief updater and trajectory assessor after the action has executed. It is
  never an action-selection arm and is never fed back to the planner.

The runtime records the regime, visible-field contract, forbidden-field audit,
question-role fingerprint, transition-contract repairs and response-cache hit
or miss counts. Separate regime runs are matched only when they reuse the same
question-role cache and a shared sequential response cache. Running two fresh
role initializers is not a valid regime comparison: the different categorical
role decompositions can change every later belief, trajectory assessment and
terminal answer.

Imagined transitions now pass a semantic state-integrity gate before they can
affect a second hop. In particular, `empty` or `inconclusive` cannot resolve
roles, claim progress or resolve a contradiction, and `ready` is accepted only
when projected missing roles, contradictions and evidence lineage permit it.
The gate only blocks optimistic inconsistency; it does not force a conservative
`not_ready` or `abstain` prediction to become ready. One repair is attempted
with exact violations. A persistently invalid conditioned row becomes a
neutral `inconclusive/unchanged` consequence while preserving its legal action
and hypothesis. It cannot erase the rest of the batch or create a planner
preference.

The historical 31-case GPT-5-mini traces contain 384/11,460 conditioned read
transitions with obvious cross-field contradictions, including 248 cases where
non-evidence resolves roles and 180 where it claims progress. Post-read oracle
analysis is materially better but still imperfect: on the nine executed reads
that overlap a hidden clue, the correct hypothesis is supported or completed
in 0.667, survives in 1.0 and reduces missing roles in 0.889. This localizes the
primary failure before the read, while retaining a separate correction-model
error slice.

A strict matched diagnostic on `cgcase:b2101769a6b2902e04ee` used one frozen
384-node graph, the same role cache, the same real-read budget and shared cached
responses. The final semantic-address IWM read one of two clues and tied no-WM
at clue recall 0.5; both abstained. The rich IWM read `pick_up` followed by
`hand_lifts_object`, covered no clues and abstained, while the localized oracle
covered both clues. The rich first-hop tree predicted `empty/unchanged` for all
28 hypothesis-conditioned first-hop outcomes, but two of its 28 imagined
second-hop outcomes predicted support/progress, and the setwise planner selected
the chain. This is a grounded delayed-value prediction error and lexical anchor
shortcut, not a wiring failure. Richer descriptors alone therefore do not
establish a zero-shot IWM advantage.

Generate a post-hoc diagnostic without model calls:

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm.zero_shot_diagnostic \
  --artifact /path/to/cohort.final.json \
  --hidden-key /path/to/terminal_targets.hidden_key.json \
  --output /path/to/zero_shot_diagnostic.json
```

For a strict semantic-versus-rich comparison, run the regimes sequentially
with identical `--question-role-cache` and `--response-cache` paths. Use
separate output artifacts, and never run concurrent writers against the shared
cache. The next scientific gate is a multi-video, video-disjoint comparison of
semantic-address, rich descriptor, no-WM, shuffled-IWM, immediate-only and the
post-read oracle diagnostic. The present one-case result is sufficient to
reject the assumption that adding L1 text automatically fixes zero-shot IWM,
but not to estimate an average effect.
