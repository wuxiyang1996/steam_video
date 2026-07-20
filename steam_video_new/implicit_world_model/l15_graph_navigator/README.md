# Implicit World Model over the L1.5 Graph

## 1. Goal

This directory describes how to run the implicit graph-navigation world model on the L1 / L1.5 structure in [`memory_graph`](../../../memory_graph/).

Core division of labor:

```text
L1 grounded observations
  → provide the final citable video evidence

L1.5 causal-temporal overlay
  → provide atomic events, temporal edges, and candidate explain / enable relations
  → define the structured hypothesis space for navigation

implicit world model
  → predict observation and belief change, then compare candidate trajectories by preference

navigator
  → execute only the first planned action
  → read a real L1 / L1.5 node
  → update the belief and replan
```

The L1.5 graph is not the world model. It is the environment and structural prior over which the world model predicts, acts, and receives real observations.

### 1.1 Runnable v0.1 implementation

The first preference-only closed loop is implemented in this package:

| File | Responsibility |
|---|---|
| `contracts.py` | Backend-neutral belief, categorical transition, four-way preference, plan, and L2-trace contracts |
| `belief.py` | `FactorizedBeliefBackend`, which updates relation grounding only after real graph observations |
| `world_model.py` | Categorical observation/belief-delta baseline and ordinal trajectory comparator |
| `planner.py` | Horizon-one/two expansion, pairwise partial-order selection, real read, belief update, and replanning |

Minimal use with an existing `CausalTemporalOverlay`:

```python
from steam_video_new.implicit_world_model.l15_graph_navigator import (
    ClosedLoopNavigator,
    FactorizedBeliefBackend,
    PreferenceOnlyPlanner,
    RuleBasedObservationBeliefModel,
    RuleBasedTrajectoryPreferenceModel,
)

backend = FactorizedBeliefBackend()
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

The planner executes only the first action of the preferred trajectory.
Subsequent imagined observations do not enter the evidence set and must not
enter the answer directly.

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

## 14. Factor-Graph Upgrade Boundary

The v0.1 backend is deliberately factorized, not a complete factor graph. A
future `FactorGraphBeliefBackend` should implement the same two operations:

```text
initialize(question, overlay, seed_evidence, missing_roles, budget)
  → BeliefSnapshot

update(belief, real_action, real_observations, overlay)
  → BeliefUpdateResult
```

The factor-graph implementation may keep a GTSAM, Pyro, or custom
message-passing object behind `BeliefSnapshot.backend_ref`. It must still
project its posterior into the shared snapshot fields: acquired evidence,
frontier, missing roles, contradictions, relation grounding, uncertainty
category, and answerability category. The following components must remain
unchanged when that backend is introduced:

- L1 evidence and L1.5 graph schemas;
- graph action generation and real `ReadGraph` execution;
- categorical observation/belief-delta world-model outputs;
- four-way pairwise trajectory preference;
- first-action execution and L2 rollout logging.

Adopt the full factor graph only if the factorized backend fails a targeted
global-consistency test—for example, later evidence cannot revise an earlier
identity assignment, mutually exclusive hypotheses retain incompatible mass,
or loop closure materially improves navigation. It is an evidence-triggered
backend upgrade, not a prerequisite for the v0.1 closed loop.
