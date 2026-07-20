# Implicit World Model for Grounded Evidence Navigation

## 1. Goal

### Architecture decision: fixed L1/L1.5 memory, IWM reasoning novelty

The target research method is **implicit-world-model-guided multi-hop reasoning
over a fixed grounded L1/L1.5 Memory Graph**. The Memory Graph is shared input
infrastructure, not the claimed novelty. It provides L1 observations, L1.5
atomic events and relations, timestamps, provenance, acquired status, and Qwen
embedding references. The research contribution is action-conditioned
future-belief prediction and planning that depends on those predictions.

The default target loop is:

```text
persistent grounded L1/L1.5 Memory Graph
  → current question-conditioned belief
  → unordered admissible evidence-query / reasoning operations
  → action-conditioned imagined belief transitions
  → blinded categorical trajectory preference
  → execute the preferred first operation
  → read real evidence and correct latent belief
  → replan
```

Factor graph / GTSAM is retained as an **optional belief-correction backup**, a
baseline, and a diagnostic tool. It is not part of the main planner claim. The
same frozen L1/L1.5 Memory Graph, retrieved candidate set, and evidence budget
must be shared across reasoning-policy comparisons so gains cannot be
attributed to graph construction.

The backup boundary is strict:

- only a real executed read followed by a categorical verifier outcome may add
  a GTSAM measurement;
- imagined observations or predicted belief deltas never enter persistent
  GTSAM state;
- GTSAM may preserve competing hypotheses, contradictions, identity
  consistency, and correction persistence;
- GTSAM does not generate candidates, traverse the memory, rank actions,
  predict transitions, compare trajectories, or supply final-answer evidence;
- the Planner still selects the next hop from IWM-predicted future-belief
  trajectories, whether or not backup correction was used on the current real
  belief.

The intended correction modes are:

```text
iwm_belief_only
  default main method

iwm_with_gtsam_backup
  activate only for an explicit categorical correction condition

gtsam_always
  graph-based baseline / ablation
```

Allowed backup triggers are categorical, for example
`contradiction_detected`, `identity_ambiguous`, `state_history_conflict`,
`multiple_competing_hypotheses`, `correction_failed`, or
`long_dependency_unresolved`. They are not hand-authored numeric thresholds.
Backup output exposed outside the solver is limited to categories such as
`accepted`, `rejected`, `unresolved`, and `conflicted`; numeric posteriors remain
solver-internal. Backup unavailability must be explicit and must never silently
change the planner or fall back to a heuristic ranking policy.

This directory contains the runnable IWM-guided L1/L1.5 navigation prototype,
its graph-based baselines, and the optional GTSAM correction experiments.

Core division of labor:

```text
L1 grounded observations
  → provide the final citable video evidence

L1.5 causal-temporal overlay
  → provide atomic events, temporal edges, and candidate explain / enable relations
  → define the structured hypothesis space for navigation

factor-graph belief backend (optional backup / graph baseline)
  → maintain question-conditioned relation posteriors and global consistency
  → expose categorical conflicts and corrected hypotheses after real reads

implicit world model
  → predict categorical observation descriptors and belief deltas

navigator
  → choose the next evidence-acquisition / reasoning hop for multi-hop reasoning
  → compare complete candidate reasoning trajectories by pairwise preference
  → execute only the first planned hop
  → read a real L1 / L1.5 node
  → update the belief and replan
```

The L1.5 graph is not the world model. It is the fixed structured memory over
which every compared reasoning policy predicts and reads. Graph relations may
define legal candidates and retrieval context, but cannot rank the final
winner in the main IWM planner.

The factor graph is neither L2 nor the learned world model. In `gtsam_always`
it is the baseline belief state; in `iwm_with_gtsam_backup` it is only an
optional real-evidence corrector. L2 remains the audited execution trace.

### 1.1 Runnable graph-based baseline

The first preference-only closed loop is implemented in this package:

| File | Responsibility |
|---|---|
| `contracts.py` | Backend-neutral belief, categorical transition, four-way preference, plan, and L2-trace contracts |
| `belief.py` | `FactorizedBeliefBackend`, retained as a local-update ablation |
| `factor_graph.py` | Production-compatible discrete sum-product adapter; canonical design and GTSAM pilot are in [`factor_graph/`](../../../factor_graph/) |
| `continuous.py` | Compatibility protocol for optional continuous smoothing; canonical boundary is documented in [`factor_graph/`](../../../factor_graph/) |
| `world_model.py` | Categorical observation/belief-delta baseline and ordinal trajectory comparator |
| `planner.py` | Horizon-one/two next-hop expansion, pairwise partial-order selection, real evidence read, belief update, and replanning |
| `context.py` | Bounded local-subgraph retrieval, categorical belief projection, candidate-hop pruning, and context audit |
| `gpt_oss.py` | OpenAI-compatible `gpt-oss-120B` categorical world-model and pairwise-preference adapters; numeric model outputs are rejected |
| `interventions.py` | Normal/null/shuffled transition controls and frozen-belief WM wrapper for causal dependence ablations |
| `realized.py` | Recompute complete categorical deltas from persisted before/after belief; never reuse imagined deltas |
| `executed_transitions.py` | Immutable sibling execution, normalized transition dataset, review locking, and categorical training export |
| `balanced_cases.py` | Correction-sensitive case mining with per-category quotas, no cross-category backfill, and independent review/export gates |
| `evidence_packets.py` | Outcome-blinded visible-evidence packets, categorical review locking/import, hidden provenance keys, and leakage checks |
| `data_inspection.py` | Data-only executed-transition coverage inspection; reports missing actions/roles and never trains a model |
| `overlay_io.py` | Strict `steam-causal-overlay/v0.2` loader with embedding-reference checks |
| `video_skills_adapter.py` | Real Video_Skills retrieval execution and schema-compatible L2 rollout export |
| `siblings.py` | Real one-step action branching with provisional four-way preference labels |
| `case_miner.py` | Stratified draft-case mining, duplicate-overlay removal, video-disjoint splits, and coverage deficits |
| `preference_data.py` | Locked case validation, blinded annotation packets, trust gates, and train-record export |
| `matched_ablation.py` | Eight primary matched-checkpoint policies plus frozen-posterior and shuffled-relation diagnostics |
| `train_models.py` | Leakage-aware categorical transition heads and four-way preference baseline |
| `workflow.py` | Batch generation, annotation, export, and evaluation CLI |
| `run.py` | CLI that writes navigation, L2, belief, sibling, and summary artifacts |

Minimal baseline use with an existing `CausalTemporalOverlay`:

```python
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    ClosedLoopNavigator,
    FactorGraphBeliefBackend,
    PreferenceOnlyPlanner,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)

backend = FactorGraphBeliefBackend()
belief = backend.initialize(
    question,
    overlay,
    seed_evidence=(initial_event_id,),
    missing_roles=("dependency",),
)
planner = PreferenceOnlyPlanner(
    RuleBasedObservationBeliefModel(),
    RuleBasedTrajectoryPreferenceModel(),
    horizon=2,
)
run = ClosedLoopNavigator(backend, planner).run(belief, overlay)
l2_trace = run.to_l2_rollout()
```

Run the focused tests from the repository root:

```bash
pytest -q memory_graph/tests/test_preference_navigation.py
```

The older `memory_graph.navigation.plan_next_read` and
`RuleBasedDependencyWorldModel` remain scalar legacy baselines for existing
ablation compatibility. The new closed loop does not call them.

Relation probabilities and correlation features are preserved in each
`RelationState` as structural priors and audit metadata. Reading both endpoints
of a candidate edge yields `endpoints_observed`; it does not resolve a missing
dependency role. Promotion to `verified` requires a passed hard verifier (or an
accepted identity track for the relevant identity/state relations). Exported
L2 records contain real observation IDs, realized belief deltas, and ordinal
trajectory labels, but no model-generated reward or utility.

### 1.2 Run on a persisted artifact

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What happened after the person opened the door?" \
  --seed-event event:anchor \
  --missing-role temporal \
  --belief-backend factor_graph \
  --horizon 2 \
  --graph-read-budget 4 \
  --output-dir /path/to/preference_navigation_run
```

By default the CLI discovers the sibling `Video_Skills` repository and calls
its real `retrieve_by_event`, `retrieve_by_relation`,
`bridge_evidence_hops`, and `search_counterevidence` functions. Use
`--video-skills-root` when the repository is elsewhere. The explicit
`--no-video-skills-runtime` option exists only for isolated tests and replay.

The output directory contains:

| Artifact | Contract |
|---|---|
| `navigation_run.json` | Plans, categorical predictions, real observations, and realized belief deltas |
| `l2_rollout.json` | Video_Skills `SkillGraphRollout`-compatible execution record |
| `belief_snapshots.jsonl` | Initial and post-observation backend-neutral beliefs |
| `sibling_checkpoint.json` | Real branches from one immutable checkpoint and provisional pairwise labels |
| `run_summary.json` | Counts, final answerability, embedding status, and output-contract audit |

`sibling_checkpoint.json` is validated against
`sibling_trajectory.schema.json`. Rule-derived labels are marked
`rule_based_provisional/v0.1`; GPT-OSS labels are marked
`gpt-oss-120b_categorical_provisional/v0.1`. Both remain
`requires_independent_annotation` and are not training gold until independently
reviewed. Numeric retrieval similarity,
edge probability, and runtime cost may remain audit metadata, but no numeric
reward, Q-value, or utility is emitted.

### 1.3 Fixed-case annotation and evaluation workflow

Formal experiments use `navigation_case_set.schema.json`. Each case fixes the
overlay ID, question, initial evidence, missing roles, real-read budget, gold
evidence events, and acceptable first actions. Formal evaluation requires a
`human_locked` case set plus a content checksum. `ai_provisional` is accepted
only with an explicit command flag and must be reported as provisional.

### 1.4 Real executed-transition supervision

World-model supervision is now generated independently of the provisional
world model and preference labels. For every case, the generator reconstructs
the same immutable checkpoint for every legal action, executes one real
Video_Skills-compatible read, applies backend correction, and recomputes the
complete categorical delta from the persisted before/after beliefs.

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  generate-transitions \
  --cases /path/to/cases.human_locked.json \
  --dataset-id executed-transitions-v1 \
  --output /path/to/executed_transitions.unreviewed.json
```

The normalized dataset stores each checkpoint once and references it from
action records. Each record retains a bounded endpoint-local context and an
optional `Qwen/Qwen3-VL-Embedding-2B` reference, never a raw vector. Targets
contain only real observation descriptors and categorical belief deltas; a
numeric value in a target is rejected.

Generated data is always `unreviewed` and `formal_eligible=false`, even when its
source case set is human locked. An independent reviewer must set every record
to `accept` or `reject` with a rationale before `lock-transitions` succeeds.
Only `human_locked` transition datasets export by default; provisional export
requires the explicit `--allow-ai-provisional` flag.

The current persisted-replay 8-video/29-case provisional pilot produced 609 grounded records,
but it is not balanced: only one action resolved a state-transition role and
only two counterevidence actions were available. It is useful for pipeline and
coverage diagnosis, not a live post-read-verifier, production, or research
accuracy claim.

### 1.5 Balanced correction-sensitive review queue

`mine-balanced-cases` targets delayed two-hop, identity/state
support/reject/inconclusive, contradiction open/resolve, useful/empty
counterevidence, blocked-path recovery, and ambiguity/abstention separately.
When a category is unavailable its quota remains a reported deficit; easy
temporal cases never fill the gap.

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  mine-balanced-cases \
  --overlay-root memory_graph/outputs \
  --case-set-id phase-e-balanced-reasoning-candidates-v1 \
  --output /path/to/balanced_cases.draft.json \
  --report /path/to/balanced_case_mining_report.json \
  --review-queue /path/to/balanced_case_review.unreviewed.json
```

Every row requires an accept/reject decision, categorical verifier outcome,
evidence-chain validity, first-action validity, delayed-effect label, and
rationale. The review rows use opaque IDs, broad relation strata, sanitized
questions/tags, deterministic hash ordering, and never expose the miner's
expected support/reject/inconclusive outcome. Pre-admission hard-verifier failures are marked
`offline_verifier_challenge`; they are diagnostic candidates, not legal online
edges, unless independent review explicitly restores them.

The current pilot selected 54 candidates without cross-category backfill:
15 delayed, 10 identity-support, 1 state-support, 4 offline state-reject,
10 empty-counterevidence, 4 blocked-path-recovery, and 10 ambiguity cases.
Identity negatives, state inconclusive, contradiction, and useful
counterevidence remain explicit deficits.

The Video_Skills runtime availability arm completed all 609 actions and exposed
236 post-read categorical claim-support results, but all 236 were `supports`.
Runtime wiring is available, while relation-level verifier validity and
negative coverage are not demonstrated. A failed support check maps to
`inconclusive`, never automatically to `rejects`.

### 1.6 Evidence gathering stop point

The reviewed-transition workflow now has an explicit data-only stop before any
GPT-OSS training. `build-balanced-evidence` converts the outcome-blinded queue
into bounded visible evidence containing endpoint events, local temporal
context, participants, states, evidence refs, and opaque
`Qwen/Qwen3-VL-Embedding-2B` references. Raw vectors, teacher probabilities,
hard-verifier output, confidence fields, and expected labels are excluded. A
separate hidden key retains overlay and embedding paths.

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  build-balanced-evidence --queue /path/to/review.unreviewed.json \
  --packet-id balanced-evidence-v1 --output /path/to/evidence.unreviewed.json \
  --key-output /private/path/to/evidence.hidden_key.json

python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  apply-balanced-evidence-review --packet /path/to/evidence.unreviewed.json \
  --review /path/to/categorical_review.json \
  --output /path/to/evidence.locked_ai_provisional.json

python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  inspect-transition-gathering --dataset /path/to/transitions.unreviewed.json \
  --cases /path/to/cases.locked_ai_provisional.json \
  --output /path/to/gathering_inspection.json
```

Every accepted annotation must cite an ID or evidence ref visible in its own
packet item. Annotation fields reject numeric model output. GPT-5.6 review is
always `ai_provisional`; generated executed-transition targets remain
`unreviewed` and cannot be exported as formal training data.

The first data-gathering run reviewed 54 items, accepted 52, and collected 994
live Video_Skills transition records, of which 978 were grounded. Inspection
found that only identity was resolved; `inspect_state_change`,
`search_counterevidence`, and `find_bridge` never executed, and no record
resolved state-transition or counterevidence. The artifact is therefore useful
for diagnosing candidate/executor coverage but is explicitly
`training_ready=false`; no GPT-OSS model was trained.

```bash
# Mine draft cases and report missing relation strata. This never creates gold.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  mine-cases --overlay-root /path/to/outputs --case-set-id diagnostic-v1 \
  --output /path/to/cases.draft.json --report /path/to/coverage.json

# Validate and independently lock the fixed cases.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  validate-cases --cases /path/to/cases.json
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  lock-cases --cases /path/to/cases.json --output /path/to/cases.locked.json \
  --status human_locked --annotator reviewer-id

# Execute every legal sibling with real Video_Skills reads.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  generate --cases /path/to/cases.locked.json --output-dir /path/to/runs

# Produce a blinded four-class packet with no rule label or posterior.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  make-annotation --runs-dir /path/to/runs --output /path/to/preferences.json \
  --packet-id preference-round-001

# After labels and rationales are filled, lock and export JSONL records.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  lock-annotation --packet /path/to/preferences.filled.json \
  --output /path/to/preferences.locked.json --status human_locked \
  --annotator reviewer-id
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  export-training --packet /path/to/preferences.locked.json \
  --output /path/to/preference_training.jsonl

# Train categorical transition heads and an ordinal-label-only preference model.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  train --training-jsonl /path/to/preference_training.jsonl \
  --output-dir /path/to/models

# Run all policies from the same checkpoint and real-read-call budget.
python -m steam_video_new.implicit_world_model.l15_graph_navigator.workflow \
  evaluate --cases /path/to/cases.locked.json \
  --output /path/to/matched_ablation.json
```

The primary report contains semantic-only, event-only, native-L1-candidate,
verified-dependency, factorized direct preference, factor-graph direct
preference, factor-graph rule-world-model lookahead, and oracle rows. It also
contains frozen-posterior and deterministically shuffled-relation diagnostic
controls. The current world-model row is an engineering baseline, not a
learned result. When a case has `query_embedding_ref`, static retrieval uses persisted
Qwen3-VL-Embedding-2B vectors with dimension and checksum checks; otherwise it
uses lexical retrieval.

GPT-5.6 may fill the blinded packet using only supplied branch outcomes, but
those labels must be locked as `ai_provisional`. The exporter rejects them by
default; `--allow-ai-provisional` exists only for clearly marked engineering
experiments. No command converts a preference into a numeric reward.

### 1.4 Current engineering evidence, not a formal result

The latest reproducible provisional status is recorded in
`baselines/video_holmes_engineering_status_v3.json`. Scanning 15 overlays from
11 unique videos produced 40 draft cases over 8 selected videos, with 20/10/10
video-disjoint train/dev/test cases. It also exposed decisive coverage gaps:
zero admitted `state_transition`, one admitted verified dependency, and zero
admitted counterevidence candidates. Forty real batch runs produced 480
blinded comparisons with no posterior or rule-label leakage.

The provisional ablation currently fails two of three engineering gates:
rule-world-model lookahead trails direct preference by 0.05 answer accuracy,
and freezing posterior correction changes nothing. Relation shuffling does
hurt by 0.20, indicating that graph structure carries signal. These numbers
must not be cited as formal results because the cases and preferences are not
human locked. The next data-generation work must fill state/dependency/counter
coverage before model training can test the central claim.

## 2. Why Use the L1.5 Graph

Relative to flat memory retrieval, the L1.5 overlay explicitly provides:

- atomic events with L1 provenance;
- temporal constraints such as `before`, `overlaps`, `during`, and `temporal_next`;
- continuity hypotheses such as `same_entity` and `state_transition`;
- candidate causal support such as `explains` and `enables`;
- competing evidence such as `contradicts`;
- a structured frontier usable by `bridge`, `counter`, and `verify`.

The navigator can therefore choose which relation to follow and which kind of evidence to seek, rather than scoring all memory once by similarity.

## 3. Semantic Boundaries

### 3.1 L1 vs L1.5

- **L1** stores only directly observable and localizable content. It is the evidence source for final answers.
- **L1.5** stores revisable atomic-event and relation hypotheses. It is the navigation space.
- The navigator may use L1.5 to choose what to read, but final answers must ground back to L1 evidence references.

### 3.2 Two meanings of "causal" must not be conflated

In L1.5:

- `explains`: an earlier event helps answer why a later event occurs;
- `enables`: an earlier event creates a necessary or materially facilitating condition.

These are candidate support relations over observational video, not identified true causal effects.

In world-model training, a counterfactual means:

> from the same belief checkpoint, execute different evidence-acquisition actions and compare the resulting observations, belief updates, and future preference outcomes.

It does not mean generating a physical counterfactual video, and it is not a do-intervention on the events themselves.

## 4. Closed-Loop Navigation

Here, an `action` is not a physical robot action. It is a next-hop
evidence-acquisition or reasoning operation in a multi-hop chain, such as
reading a supporting event, following a temporal or dependency edge, resolving
an identity, comparing before/after state, seeking counter-evidence, or stopping
to answer. If the term `skill` is retained in an interface, it means a
**reasoning skill**.

At step \(t\):

```text
z_t = Enc(q, E_1:t, H_t, g_t, B_t, frontier_t)

for each admissible action a:
    (predicted_observation_role,
     predicted_belief_delta) = WorldModel(z_t, a, G)

preference = Compare(candidate_action_or_trajectory_pairs)
a_t = PlanByPreference(preference, horizon=1 or 2)
o_t+1 = ReadGraph(G, a_t)
z_t+1 = Update(z_t, a_t, o_t+1)
```

The planner executes only the first reasoning hop of the preferred trajectory.
Subsequent imagined observations do not enter the evidence set and must not
enter the answer directly.

### 4.1 Bounded reasoning context

The planner and world model do not receive the complete Memory Graph, all
GTSAM variables/factors, every past hop, or every expanded trajectory. That
input grows too quickly for multi-video, multi-hop navigation and exposes solver
details that the LLM must not interpret.

Each planning round constructs a compact `ReasoningContext` containing
only:

- the query and current answer constraints;
- unresolved or missing evidence roles;
- a retrieved local subgraph relevant to the candidate hop;
- a categorical summary of accepted, rejected, and unresolved hypotheses;
- references to acquired evidence, plus only the most recent local history;
- the two candidate trajectories currently being compared.

The recommended input path is:

```text
complete Memory Graph
  → embedding / structural retrieval
  → candidate local subgraph
  → query-, missing-role-, and conflict-conditioned pruning
  → bounded ReasoningContext
  → categorical world-model prediction
  → pairwise trajectory preference
```

Embeddings, including the `Qwen/Qwen3-VL-Embedding-2B` field, are retrieval
features and references; raw embedding vectors must not be placed in the LLM
prompt. Likewise, GTSAM priors, likelihoods, marginals, covariance, and factor
weights remain solver-internal. The LLM sees categorical belief descriptors,
for example `accepted`, `rejected`, `unresolved`, `missing`, or
`answerability=insufficient`, and never produces numeric confidence, reward,
probability, or utility.

The runtime enforces and audits three independent budgets:

- **retrieval budget:** maximum local nodes and edges retrieved per round;
- **candidate budget:** maximum legal next hops retained after deterministic
  filtering;
- **comparison budget:** maximum pairwise trajectory comparisons, preferably
  using a tournament or staged selection instead of a full all-pairs expansion.

Long history must be folded into the current belief snapshot, acquired-evidence
references, unresolved questions, and a short recent-hop window. Every run
records retrieved/dropped nodes and edges, candidate count, comparison budget,
estimated prompt tokens, and real evidence reads. Context-budget ablations must
report final-answer accuracy, evidence-chain completeness, and read efficiency
at fixed budgets; increasing the model context window is not a substitute for
this retrieval boundary.

This boundary is implemented by `ReasoningContextBuilder`. `GraphReadAction`
remains the execution compatibility type, while every `PlanDecision` exposes a
semantic `ReasoningHop` (`read_event`, `follow_temporal`,
`follow_dependency`, `resolve_identity`, `inspect_state_delta`,
`seek_counter_evidence`, `verify_relation`, or `stop_and_answer`). The planner
retains the largest candidate set whose complete all-pairs comparison fits the
comparison budget. It executes a first hop only when the undominated set has
one semantic first-hop meaning; otherwise it emits an audited abstention. The
complete context audit is recorded in the navigation and L2 artifacts.

The optional embedding-score input is computed outside the LLM. When supplied,
it ranks structurally legal candidates before pruning; the context stores only
the embedding model name and retrieved IDs, never raw vectors. The current
artifact contract retains `Qwen/Qwen3-VL-Embedding-2B` references for this path.

### 4.2 GPT-OSS-120B provisional reasoning backend

For now, both categorical transition prediction and trajectory comparison may
use `openai/gpt-oss-120b` through an OpenAI-compatible endpoint:

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator \
  --overlay /path/to/causal_temporal_overlay.json \
  --question "What happened after the person opened the door?" \
  --seed-event event:anchor \
  --missing-role temporal \
  --reasoning-model-backend gpt-oss-120b \
  --reasoning-model openai/gpt-oss-120b \
  --reasoning-keys-py /fs/gamma-projects/vlm-robot/keys.py \
  --context-max-nodes 24 \
  --context-max-edges 32 \
  --candidate-hop-budget 8 \
  --comparison-budget 32 \
  --transition-intervention normal \
  --output-dir /path/to/navigation_run
```

For matched dependence controls, use `--transition-intervention null` or
`shuffled`, `--freeze-world-model`, and horizon one/two for immediate-only vs
delayed-belief prediction. These flags alter imagined transitions only; they
do not change legal candidates or permit imagined evidence to be persisted.

With `--reasoning-keys-py`, the client reads `OPENROUTER_API_KEY` without
serializing it and uses `https://openrouter.ai/api/v1` by default.
`--reasoning-api-base` may override that endpoint. Alternatively,
`OPENAI_BASE_URL` plus the environment named by `--reasoning-api-key-env` may
be used. Selecting GPT-OSS without either a keys file or environment endpoint
fails explicitly and never falls back silently to the rule baseline.

The adapter permits only categorical observation descriptors, categorical
belief deltas, and one of `prefer_left`, `prefer_right`, `tie`, or
`incomparable`. A numeric value anywhere in the model JSON output is rejected.
GTSAM probabilities/posteriors and raw embeddings are excluded from the model
payload. GPT-OSS outputs remain provisional model judgments rather than
independent gold annotations or acquired evidence.

The evidence role is not generated: it is a deterministic type contract of the
legal reasoning operation. GPT-OSS predicts only unknown descriptor fields,
which avoids redundant role disagreement without adding a winner heuristic.

`stop_and_answer` is a deterministic no-observation transition and never calls
the model. For every other hop, predicted relation updates must copy an exact
edge ID touched by that hop; cross-edge or invented updates are rejected.
Transport token counts and finish reasons may be logged separately to audit
context cost, but they are never used as reward, confidence, or belief values.

A live OpenRouter smoke on the grounded Phase D fixture passed on 2026-07-20.
With a budget of 3 local nodes, 2 local edges, 2 candidate hops, and 1
comparison, GPT-OSS selected `inspect_state_delta`; the real read resolved the
missing role and the final belief became answer-ready. The run made two model
requests (one non-stop transition and one preference comparison). Their prompt
token counts were 804 and 445; both completed with `finish_reason=stop`. This is
an integration smoke, not an accuracy result. The deterministic stop trajectory
made no request and produced no resolved role or relation update.

### 4.3 Non-heuristic planning boundary

The research claim is **world-model-guided multi-hop reasoning in belief
space**, not factor-priority or hand-scored graph traversal. Deterministic rules
may enforce legality, provenance, safety, deduplication, and compute budgets;
they must not decide which admissible reasoning direction wins.

The intended boundary is:

```text
graph rules / factor graph
  → reject illegal, blocked, repeated, or ungrounded hops
  → produce an unordered admissible candidate set

embedding retrieval
  → improve candidate recall within the retrieval budget
  → never assign the final winner

world model
  → predict categorical future-belief transitions for candidate trajectories

preference planner
  → compare only those predicted trajectory outcomes
  → execute the preferred first reasoning hop
```

Question-direction ordering, lexical overlap, factor priority, stable structural
order, and rule-based projected progress are provisional candidate-generation
or baseline mechanisms. They are not acceptable evidence that the learned
planner selected an action. In particular, `tie` or `incomparable` must trigger
additional evidence, another comparison, or abstention; production code must
not silently choose the first candidate.

The production planner must satisfy these invariants:

- permuting candidate order does not change the selected semantic hop;
- removing lexical cues or shuffling graph presentation order does not create
  an implicit ranking policy;
- factor probabilities/posteriors remain solver-internal and never become
  hand-authored action scores;
- failure of GPT-OSS causes an explicit failure or abstention, never a silent
  rule-planner fallback;
- the rule-based world model remains an explicitly named experimental baseline;
- learned supervision comes from real post-read categorical belief deltas and
  independently reviewed trajectory preferences, not action bonuses.

The current implementation now enforces the first-stage anti-shortcut boundary:

- candidate truncation is stable under input permutation and does not use
  question-direction, lexical, or factor-priority winner rules;
- every retained trajectory pair is compared within the comparison budget;
- a unique undominated first-hop meaning is required; otherwise planning emits
  an audited `abstain` instead of selecting the first item;
- GPT-OSS trajectory comparison receives anonymous imagined outcomes, without
  operator names, target/edge/trajectory identifiers, or action rationales;
- imagined rollouts carry categorical hypothesis, frontier, contradiction,
  path, recovery, uncertainty, and answerability changes.

These are implementation invariants, not evidence of research success. The
remaining gap is empirical: train/validate the learned WM and preference model,
construct delayed-effect and ambiguity cases, run the interventions below on a
fixed multi-video gold set, and show that WM corruption changes actions and
reduces end-task quality under a matched read budget. The categorical GTSAM
backup wrapper is now engineering-complete: the main CLI exposes all three
belief modes, categorical activation is audited separately from factor
activation, stateful sibling branches are checkpoint-forked, and categorical
belief can be checksum-restored without exporting solver marginals. This does
not establish production calibration or navigation benefit.

The required dependence experiment keeps the query, current factor-graph
belief, candidate set, planner, and budget fixed, changing only the world-model
prediction arm:

```text
normal learned WM
null WM
shuffled-transition WM
frozen WM
immediate-only WM
delayed-belief trajectory WM
```

The runtime primitives for these arms are implemented as
`TransitionIntervention.NORMAL/NULL/SHUFFLED`, `FrozenBeliefWorldModel`, and
planner horizon one/two. A benchmark result is not claimed until the arms are
run on the same locked cases, candidates, graph-read budget, and model budget.

Report action divergence, categorical transition accuracy, evidence-chain
completion, reads, and final-answer accuracy separately. The main claim is
supported only if corrupting or removing world-model predictions systematically
changes trajectories and degrades reasoning outcomes, while the delayed-belief
arm succeeds on cases whose first hop has no immediate categorical progress
but opens the second-hop evidence path. If actions remain unchanged, the system
must not be described as world-model-guided.

## 5. Belief State

The latent state \(z_t\) need not decode into a fixed number of reasoning chains, but its inputs and auxiliary probes should cover:

- the current question and answer candidates;
- acquired L1 evidence;
- currently or recently visited L1.5 events;
- relevant relation posteriors;
- competing explanations and contradictions;
- the missing reasoning role;
- the graph frontier and visited mask;
- residual uncertainty;
- remaining read and planning budget.

An interpretable auxiliary stage variable may be:

```text
GROUND
TEMPORAL_LINK
EXPLANATORY_LINK
FIND_BRIDGE
SEEK_COUNTER
VERIFY_CHAIN
ANSWER
ABSTAIN
```

These variables are for supervision and auditing. They should not be treated as the full world state.

## 6. Graph Action Space

An action should at least contain an operator and a frontier target:

```text
GraphAction:
  operator: semantic
          | temporal_back
          | temporal_forward
          | same_entity
          | candidate_cause
          | effect
          | bridge
          | counter
          | verify
          | stop
  source_event_id: string | null
  target_event_id: string | null
  frontier_slot: string | null
```

Suggested execution semantics:

| Action | Graph execution |
|---|---|
| `semantic` | question-conditioned semantic retrieval among unvisited events |
| `temporal_back` | follow temporal edges to earlier events |
| `temporal_forward` | follow temporal edges to later events |
| `same_entity` | expand along entity-continuity edges |
| `candidate_cause` | read earlier events that may explain or enable the current event |
| `effect` | read later events that the current event may explain or enable |
| `bridge` | seek a missing intermediate event connecting known evidence to a target hypothesis |
| `counter` | seek contradictions or competing explanations |
| `verify` | return to the L1 provenance of a relation for evidence checking |
| `stop` | stop when evidence is sufficient or the budget is exhausted |

`before` only denotes temporal order. It must not be automatically upgraded to `candidate_cause`.

## 7. Minimal Interfaces

### 7.1 Overlay tensorizer

Inputs:

- `CausalTemporalOverlay`;
- event embeddings;
- L1 observation embeddings;
- event-to-L1 provenance index.

Outputs:

```text
node_embeddings[N, D]
node_types[N]
time_spans[N, 2]
relation_edges[2, E]
relation_types[E]
relation_probabilities[E]
relation_status[E]
event_to_l1
```

Uncalibrated probabilistic relations must retain `uncalibrated_prior` status and must not be treated as equivalent to deterministic temporal edges.

### 7.2 ReadGraph

`ReadGraph(G, action)` returns a real observation, not model-imagined node text:

```text
GraphObservation:
  reached_event_ids
  l1_evidence_refs
  node_embeddings
  grounded_text
  traversed_relations
  verifier_status
  execution_cost
```

### 7.3 Update

The update step must at least:

- add actually read nodes to the acquired / visited set;
- refresh the frontier;
- adjust relation beliefs from verifier outcomes;
- update missing roles, contradictions, and uncertainty;
- retain the gap between predicted and corrected belief as a training signal.

## 8. World-Model Prediction Targets

Given \((z_t, a_t, G)\), the model predicts:

1. **Observation role**: which kind of evidence an action may return, not imagined evidence that can directly answer the question.
2. **Belief delta**: changes in relation posteriors, entity / event bindings, missing roles, contradictions, and uncertainty.
3. **Pairwise preference**: whether one realized or predicted action trajectory
   is preferred, tied, worse, or incomparable to another.
4. **Calibration**: uncertainty over observation and belief-delta predictions,
   plus preference reliability.

### 8.1 Preference-only output contract

The model must not emit a numeric reward, value, Q-value, or delayed-utility
estimate. It may emit internal comparison logits, but those logits are used
only to produce an ordinal decision and must never be interpreted or reported
as utility magnitudes:

```text
Preference(a_i, a_j | z_t) ∈ {
  prefer_left,
  tie,
  prefer_right,
  incomparable
}
```

Sibling labels come from real executed outcomes. Verified answer support and
required-role completion dominate merely relevant retrieval; no progress
dominates an invalid action; unsupported commitment or hidden-supervision
leakage is worst. When two outcomes trade off in a way the evaluator cannot
reliably order, the target is `incomparable`, not an invented scalar reward.

Numeric accuracy, evidence recall, latency, and cost may remain offline audit
metrics. They may help the evaluator derive a preference label, but they are
not model prediction targets. Two-step planning compares complete predicted
trajectory descriptors pairwise; it does not add predicted rewards across
steps.

```text
L_WM =
    λ_obs L_observation
  + λ_belief L_belief_delta
  + λ_pref L_pairwise_preference
  + λ_tie L_tie_or_incomparable
  + λ_cal L_calibration
```

## 9. Sibling Counterfactual Trajectories

Training data should fork from the same checkpoint:

```text
checkpoint C_t
  ├─ semantic         → observation 1 → update 1 → outcome descriptor 1
  ├─ temporal_forward → observation 2 → update 2 → outcome descriptor 2
  ├─ candidate_cause  → observation 3 → update 3 → outcome descriptor 3
  └─ bridge           → observation 4 → update 4 → outcome descriptor 4

pairwise labels:
  trajectory 2 ≻ trajectory 1
  trajectory 2 ≻ trajectory 3
  trajectory 3 ∥ trajectory 4  (incomparable)
```

At least retain:

- the expert action;
- the current-policy action;
- one high-scoring alternative action;
- one structurally executable but inefficient action.

A single behavior trajectory alone cannot reliably supervise which action is better under the same belief.

## 10. Staged Implementation

### Phase A: Graph-aware retrieval baseline

- tensorize the L1.5 overlay;
- implement `visited`, `frontier`, and `ReadGraph`;
- use heuristic / greedy graph actions;
- compare L1.5 graph navigation against flat semantic retrieval.

This phase does not claim a world-model contribution.

### Phase B: Learned next-action policy

- learn discrete graph-action ranking;
- update in closed loop from real graph reads;
- compare direct pairwise ranking, beam search, and learned stopping;
- verify that shuffling actions significantly degrades performance.

### Phase C: Implicit world model

- train observation-role, belief-delta, and trajectory-preference heads;
- build sibling trajectories;
- rank one- or two-step predicted trajectories without numeric value summation;
- execute only the first action and replan after each real observation.

Only this phase can support the main claim of an implicit belief-transition world model.

## 11. Required Ablations

- flat semantic retrieval;
- deterministic temporal graph only;
- no `explains` / `enables`;
- direct action ranking without rollout;
- behavior trajectories only, without sibling branching;
- shuffled action / operator;
- predicted transitions without test-time planning;
- planning without posterior correction;
- L1 evidence verification on / off.

If shuffling learned actions leaves results essentially unchanged, the model is mostly relying on recurrent reads or semantic similarity rather than a useful graph-navigation policy.

## 12. Success Criteria

Phase A should first establish:

1. whether the L1.5 graph improves multi-hop evidence recall and bridge recall;
2. whether graph actions outperform flat retrieval under the same read budget;
3. whether candidate-causal edges yield net benefit rather than error propagation;
4. whether all answer evidence can be traced back to L1;
5. whether uncalibrated relations are correctly isolated.

The full world model must additionally show:

1. sibling supervision beats behavior-only supervision;
2. rollout preference beats a direct action-preference ranker trained on the
   same data;
3. delayed bridge preferences are accurate and ties / incomparable outcomes
   are calibrated;
4. MPC beats greedy, beam, and ordinary iterative retrieval under the same budget.

## 13. Non-Goals

This design does not attempt to:

- identify true causal effects from observational video;
- generate physical counterfactual videos;
- answer questions from imagined evidence;
- treat L1.5 hypotheses as L1 facts;
- replace the existing `memory_graph` or Video_Skills stack with a new graph system;
- enlarge the world-model claim before graph-aware retrieval is shown to help.

## 14. Factor-Graph Belief Backend and GTSAM Boundary

The canonical specification has moved to
[`factor_graph/README.md`](../../../factor_graph/README.md). It covers the
belief/L2/world-model separation, reasoning-as-Active-SLAM mapping, variables
and factors, preference-only LLM contract, GTSAM 4.2.1 binding limits, runnable
correction pilot, four-arm destructive-control experiment, embedding boundary,
and production migration gates.

This package retains `FactorGraphBeliefBackend` as the default
production-compatible Python sum-product adapter and
`FactorizedBeliefBackend` as its local-update ablation. Installing GTSAM does
not silently change navigator inference. Use `--factor-iterations N` only for
the compatibility backend; use the commands in the canonical document for the
GTSAM pilot.

## 15. Phase E Provisional Evidence

The 2026-07-20 pilot locks 29 AI-provisional cases across 8 videos, creates a
side-randomized blinded packet with 29 categorical GPT-5.6 preferences, trains
categorical transition/preference pilots, and runs the matched-budget arms.
No LLM-produced numeric reward, confidence, probability, or utility enters the
training data. Full metrics and provenance are recorded in
[`factor_graph/experiments/phase_e_gpt56_provisional_v1/status.json`](../../../factor_graph/experiments/phase_e_gpt56_provisional_v1/status.json).

This is engineering evidence, not gold: independent human identity/state
labels and an actual post-read verifier confusion matrix are still absent.
The strict v2 `evaluate --gtsam-closed-loop` follow-up adds a
verifier-direct-only arm and removes the former aggregate lexicographic gate.
On the 29-case set, GTSAM correction and verifier-direct-only have identical
accuracy proxies, mean reads, and action sequences, so no GTSAM navigation
benefit is claimed. Three derived coupled mechanism cases separately show that
support/reject propagation changes the next action while inconclusive remains
a no-factor negative control. These mechanism cases are not independent gold.

The review-anchored v2 collector fixes a separate data-generation gap:
accepted reviewed actions are now executed in addition to native graph
candidates. A restored action may read only its reviewed, already-persisted
endpoints and never inserts an edge. Targetless counterevidence search has an
auditable grounded-empty outcome, and state resolution requires a strict
post-read categorical state-delta check. On the 52 provisional cases, all 42
non-STOP reviewed actions are executed and grounded; the dataset remains
unreviewed and no GPT-OSS training is performed. See the canonical factor
graph README section 13.5 for counts and remaining gates.

## 16. Targeted Failure Slices and Human Review

The transition-review workflow now separates model-provisional inspection from
independent human locking. `inspect-transition-failures` partitions every
provisional `inconclusive` row into one exclusive primary failure slice using
only public categorical evidence. `build-targeted-gathering` then samples
already executed records across videos for strict state deltas, identity hard
negatives, counterevidence adjudication, empty/reject controls, inconclusive
controls, and delayed two-hop cases. Target strata and consistency groups stay
in ignored `*.hidden_key.json` files and never enter the public packet.

The local review application is started with:

```bash
python -m steam_video_new.implicit_world_model.l15_graph_navigator.human_review_server \
  --packet steam_video_new/implicit_world_model/datasets/targeted_transition_review_v1/human_review_packet.unreviewed.json
```

It saves drafts in browser localStorage, renders only public evidence, validates
against the production Python contract, and exports an `independent_human`
response. Applying that response produces `human_locked`; GPT-5.6 responses
remain `ai_provisional`. Neither path exports training records automatically.

## 17. Qwen IWM Post-Training: Grounded Dynamics plus Preference GRPO

The intended IWM is a Qwen-family LLM/VLM, not a rule model, graph neural
network, or weighted collection of hand-designed scores. Given a compact
belief state, grounded evidence memory, and candidate reasoning action, it
predicts an observation descriptor, a categorical belief delta, and a short
imagined continuation:

```text
Qwen-IWM(belief_t, grounded_memory_t, action_t)
    -> observation_descriptor_hat
    -> categorical_belief_delta_hat
    -> imagined_continuation
```

The post-training design separates two objectives that should not be
conflated:

1. **Grounded world-dynamics learning.** Human-locked executed transitions
   supervise the actual observation descriptor and categorical belief delta.
   SFT, categorical preference optimization, or on-policy distillation may be
   used here. This objective prevents the model from learning a successful
   action shortcut while producing inaccurate imagined transitions.
2. **Reasoning-policy improvement.** From one fixed belief checkpoint, the
   current Qwen policy samples a group of short sibling trajectories. After
   real execution or auditable replay, an independent reviewer returns only a
   partial order: `prefer_left`, `prefer_right`, `tie`, or `incomparable`.
   Ordinal preference GRPO then improves action selection and stopping.

GRPO is therefore part of the main method, but GRPO alone is not treated as
evidence that the IWM learned world dynamics. A policy optimized only from the
final trajectory outcome could exploit retrieval shortcuts, answer priors, or
language patterns while ignoring its own imagined belief transitions.

### 17.1 No model-produced scalar reward

The reviewer never emits a reward, probability, confidence, utility, or
weighted subscore. In particular, the method must not introduce a reward such
as a weighted combination of information gain, graph progress, answerability,
and read efficiency. The only supervision exposed by the reviewer is an
ordinal trajectory relation:

```text
same initial belief + same execution budget

trajectory A preferred to trajectory B
trajectory A tied with trajectory C
trajectory D incomparable
```

The optimizer may deterministically convert an ordinal group ranking into a
centered group-relative advantage. Those internal numerical values are an
implementation detail of optimization, not an LLM/VLM judgment or an
annotation target. Ties retain equal rank; incomparable trajectories do not
receive an invented ordering.

Evidence grounding is a validity constraint rather than a reward component.
A trajectory that cites imagined evidence as observed evidence is invalid; an
undecidable comparison is `incomparable`, and equal-quality trajectories are
`tie`.

### 17.2 Alternating training loop

The clean training loop alternates grounded transition batches and on-policy
trajectory groups:

```text
human-locked executed transition
    -> update Qwen observation/belief-transition behavior

current Qwen samples sibling reasoning trajectories
    -> execute the first action or perform auditable replay
    -> obtain real evidence and corrected belief
    -> collect categorical group preference
    -> ordinal preference GRPO update
```

On-policy distillation is an optional dynamics-stabilization method or
ablation. It is not the primary claim because a larger teacher may transfer
reasoning style or answer priors without establishing that the student learned
grounded action-conditioned dynamics. A text-only teacher also cannot replace
visual transition supervision unless it receives an independently grounded
observation description.

### 17.3 Required causal ablations

The minimum matched-data comparison is:

- executed-transition SFT only;
- ordinal preference GRPO only;
- executed-transition SFT plus ordinal preference GRPO;
- DPO/IPO on the same sibling preferences;
- shuffled IWM transition predictions with the same GRPO planner;
- no imagined transition with the same GRPO planner;
- optional on-policy distillation with the same teacher and data budget.

The main IWM claim requires both better navigation and evidence that planner
behavior depends on learned transition predictions. If shuffling or removing
imagined belief transitions leaves action selection unchanged, GRPO learned a
policy shortcut rather than world-model-guided reasoning.

The current targeted packet supplies only action-level transition supervision.
Before GRPO, it still needs independent human locking, Qwen on-policy sibling
rollouts from identical checkpoints, real executed corrections, and blinded
group-level ordinal preferences. No GRPO or GPT-OSS/Qwen training should begin
from the current model-provisional packet alone.
