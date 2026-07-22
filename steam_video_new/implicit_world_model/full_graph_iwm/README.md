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
actions. At the virtual root it exposes `START_AT(node)` for every visible
retained node. At a real cursor it exposes only:

- temporal forward/backward hops incident to the cursor;
- positive-direction L1.5 correlation hops incident to the cursor;
- permitted-direction categorical candidate hops incident to the cursor;
- backtracking to already acquired nodes;
- stop, answer (when ready), and abstain.

There is no flat `current node x every unread node` semantic-probe product and
no candidate `inspect/verify` action in the main navigation path. After the
first real read, the eight current smoke graphs expose about four to five real
read actions per cursor on average, rather than dozens or hundreds.

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

The planner executes exactly one real action only when all preferred expansions
share that action. The observation is then broadcast to every active
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

- The complete retained graph is visible, but unread evidence values are not.
  Unread nodes expose a compact semantic key, timestamp/type, and embedding
  reference. A real executor reveals the evidence value only after the hop.
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
- All retained candidates are considered; the main planner does not use an
  embedding Top-K or a hand-written score to select a winner. Pair comparisons
  are request-batched to keep model outputs bounded without sampling pairs.
- Every mode abstains on a non-unique order. The legacy
  `--execute-stable-ties` option is retained only as an audited request flag;
  it cannot restore construction-order execution. Imagined consequence is
  never used as final-answer evidence.
- A real role resolution is persisted as `(role, acquired_node_id)`. It cannot
  cite imagined evidence. Multi-role questions cannot become answerable from
  only one real observation, even if the categorical updater over-resolves its
  first read; readiness requires complete role lineage and at least two real
  observations.
- Actions without a real read (`stop`, `answer`, `abstain`, `backtrack`) have
  deterministic `inconclusive + unchanged` dynamics and cannot hallucinate an
  observation.

## Components

```text
contracts.py       graph/action/transition/preference contracts
graph_adapter.py   persisted overlay -> retained semantic L1 + soft L1.5
action_compiler.py cursor-local legal hops and real evidence execution
model_input.py     leakage-safe full retained-graph model view
planner.py         horizon-1/2 IWM rollout and ordinal partial-order selection
multi_trajectory.py direct multi-hypothesis IWM, shared execution and pool lifecycle
gpt_oss.py         categorical GPT-OSS IWM and batched preference adapters
caption_candidates.py question-independent grounded categorical hop builder
reactive.py        matched no-world-model direct-policy baseline
interventions.py   shuffled/frozen world-model controls
closed_loop.py     execute/read/correct/replan loop and evaluator-only metrics
cgbench_pilot.py   fixed-case compile gate and matched-budget experiment
```

The direct multi-trajectory implementation currently provides:

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
