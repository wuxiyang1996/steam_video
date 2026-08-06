# Reasoning v2: Multi-step Evidence World Model

This package is the clean implementation path for world-model-guided evidence
acquisition. The target architecture predicts the consequences of *unexecuted*
multi-step read trajectories, then lets a separate Reasoner execute grounded
reads and answer from real evidence.

> **Status:** this document is the target research and implementation contract.
> The checked-in runtime still contains graph-navigation and categorical
> single-step components inherited from earlier iterations. Those components
> are migration baselines, not claims about the final method.

## 1. Research question

Long videos contain many candidate evidence nodes, but full evidence reads and
large-model reasoning calls are expensive. The central question is:

> Can a lightweight world model predict the joint reasoning consequences of
> alternative multi-step, multi-clue evidence-acquisition trajectories before
> execution, so that a grounded Reasoner avoids wasteful or myopic search?

The method targets cases where static node relevance is insufficient:

- one clue is insufficient but several clues jointly answer the question;
- an early clue changes the meaning or retrievability of a later clue;
- individually relevant clues are redundant when acquired together;
- a new clue contradicts the current reasoning direction;
- a temporarily unhelpful read completes a necessary evidence chain later.

The IWM therefore models delayed, conditional, and non-additive acquisition
consequences rather than static relevance.

## 2. Final module boundary

```text
question + current grounded reasoning state
                    |
                    v
L1 grounded evidence memory: candidate nodes and hidden-until-read values
                    |
                    v
Q-Former: fixed-size, node-aligned pre-read proposal tokens for every unread node
                    |
                    v
IWM: counterfactual multi-step rollout over alternative clue combinations
                    |
                    v
Reasoner: choose/deviate/stop, execute real READs, answer or abstain
                    |
                    v
L1 update and next grounded planning state
```

The boundary is summarized by three rules:

1. **Q-Former represents candidate evidence.**
2. **IWM predicts multi-step clue-combination consequences.**
3. **Reasoner performs grounded execution and QA.**

Imagined IWM states never become evidence. Only an executed read can expose a
node's grounded value or support the final answer.

## 3. L1: grounded evidence memory

L1 stores real, traceable video evidence nodes. Each node retains at least:

- stable node ID;
- source interval and timestamp;
- grounded visual/text content;
- provenance;
- visibility and read status.

L1 is the factual source for the final answer. Before `READ(i)`, the system may
expose a bounded, low-bandwidth address/proposal view of node `i`; the full
grounded evidence remains hidden until execution.

The explicit L1.5 semantic/correlation graph is removed from the main method.
Deterministic temporal, visibility, provenance, availability, and budget
constraints remain valid execution masks, but the target architecture does not
construct static semantic adjacency, cursor rows, or a graph-action compiler.
Existing `navigation/` and `full_graph_iwm` code remains available only for
historical reproduction and matched graph-based baselines.

## 4. Q-Former: node-aligned proposal encoder

For every unread node `i`, Q-Former emits a fixed number of proposal tokens:

```text
p_i = Q_phi(q, X_i, t_i) in R^(m x d)
```

`p_i` is a low-bandwidth, node-aligned *pre-read* representation. It may encode:

- coarse node content available under the pre-read contract;
- relation to the current question;
- temporal position;
- a possible evidence role.

Q-Former does not select the next hop, predict a multi-step consequence, expose
the hidden source value, or answer the question. It prepares the candidate set

```text
P = {p_1, ..., p_N}
```

for the IWM. Proposal generation must preserve node identity, use explicit masks
for unavailable/padding slots, and be tested for permutation consistency.

## 5. IWM: multi-step compositional consequence model

The IWM receives the current grounded reasoning state `z_0` and proposal set
`P`. It rolls out imagined evidence interventions:

```text
z_(h+1) = T_theta(z_h, READ(S_h), {p_i : i in S_h})
```

where `S_h` contains one or more imagined clue nodes. A candidate trajectory is

```text
tau = (S_1, S_2, ..., S_H).
```

For each trajectory, the IWM predicts how the combined clues would change:

- the latent reasoning state;
- remaining evidence needs;
- expected answerability;
- the decision to continue, answer, or abstain;
- contradiction or insufficiency status.

The primary object is the composition, not independent node utility:

```text
C(A, B) != C(A) + C(B).
```

The rollout output contains a proposed evidence trajectory, a lightweight
latent consequence at each step, and a predicted terminal status. It does not
contain imagined captions, factual memory writes, a final answer, scalar
reward, or evidence that can be cited by the Reasoner.

## 6. Reasoner: grounded execution and QA

The Reasoner receives current real evidence plus one or more IWM rollout
contexts. It may:

- execute several real reads along a predicted trajectory;
- stop early after a decisive observation;
- deviate when an observation invalidates the rollout;
- replan after prediction mismatch;
- produce a grounded answer;
- abstain when the acquired evidence is insufficient.

The Reasoner owns the execution policy. The IWM predicts multi-step futures;
the Reasoner performs multi-step grounded execution. Every answer claim must be
traceable to evidence actually read by the Reasoner.

## 7. Runtime loop

```text
1. Build z_0 from the question and already-read grounded evidence.
2. Encode every eligible unread node into node-aligned proposal tokens p_i.
3. Roll out several counterfactual trajectories with the IWM.
4. Give compact rollout contexts and candidate trajectories to the Reasoner.
5. Execute READ(S_h) under the real visibility/provenance/budget constraints.
6. Compare the observation with the predicted consequence.
7. Stop, continue, deviate, or replan.
8. Answer only from executed evidence, otherwise abstain.
```

This is not a requirement to execute an entire imagined trajectory blindly.
Model-predictive replanning after every real observation remains permitted.

## 8. Why a world model is necessary

A reactive Reasoner can also read and replan. The IWM is justified only if it
can cheaply preview several counterfactual trajectories, for example
`A -> B`, `A -> C`, and `D -> E`, without paying for every real read and every
large Reasoner call.

The intended gains are:

- fewer full evidence reads;
- fewer large-model Reasoner calls;
- higher multi-clue chain completion;
- better acquisition of delayed clues;
- better answerability and abstention decisions;
- higher grounded QA performance at a matched reasoning budget.

If a reactive Reasoner produces the same trajectories and results at the same
budget, the IWM is unnecessary. This is the project's explicit Go/No-Go rule.

## 9. Difference from iterative reasoning

A deep Q-Former or iterative module may repeatedly refine representations that
have already been supplied:

```text
z_(h+1) = F(z_h, P).
```

Depth alone does not make it a world model. Here, every rollout step corresponds
to an explicit evidence intervention `a_h = READ(S_h)`, and supervision comes
from the consequence of executing the same clue combination on real evidence:

```text
READ(S_(1:h)) -> r*(S_(1:h)).
```

Ordinary iterative reasoning learns to compute over available features. The IWM
learns what an unexecuted multi-step acquisition would do to later reasoning.

## 10. Training contract

Final QA loss alone is insufficient evidence that the IWM learned dynamics.
Training examples must branch from the same immutable initial state and execute
real clue combinations such as:

```text
empty, A, B, A+B, A+C
```

A frozen teacher Reasoner reads the real evidence and produces a functional
target state:

```text
r*(S) = R_teacher(q, E_0 union E_S).
```

Targets may include answerability, next evidence need, subsequent read
distribution, answer distribution, `continue | answer | abstain`, contradiction,
and insufficiency. The IWM predicts the same target using only pre-read proposal
tokens and the imagined intervention sequence.

The dataset must deliberately cover:

- clue synergy and leave-one-clue-out examples;
- redundancy and contradiction;
- conditional relevance;
- different prefixes of the same trajectory;
- exchangeable and order-sensitive clue sequences;
- missing-key-evidence abstention;
- hard negatives with surface similarity but no realized clue gain.

Splits must be video-disjoint. Candidate order must be randomized, hidden
grounded values must never leak into proposal inputs, and all teacher targets
must be generated from real reads at immutable checkpoints.

## 11. Decisive evaluation

Under the same Q-Former, Reasoner, memory, real-read budget, and large-model call
budget, compare:

1. Q-Former proposal tokens directly to the Reasoner;
2. a parameter-matched deep Transformer/iterative module;
3. a reactive Reasoner that replans after every real read;
4. a single-step consequence model;
5. additive independent-node consequences;
6. the multi-step compositional IWM;
7. shuffled IWM rollouts;
8. oracle real post-read consequences.

Report:

- multi-clue evidence-chain completion;
- grounded answer accuracy;
- answerability and abstention quality;
- combination-consequence prediction;
- synergy/redundancy discrimination;
- reads per question and Reasoner forward calls;
- rollout error versus horizon;
- causal effect of IWM predictions on executed trajectories.

Graph-based navigation, Qwen listwise readers, and previous categorical planners
may be retained as additional historical baselines, but they are not substitutes
for the matched ablations above.

## 12. Required gates

Do not claim a successful IWM or start expensive scaling until all of the
following hold on a fixed multi-video cohort:

1. L1 nodes retain complete grounded clue coverage with valid provenance.
2. The pre-read proposal contract passes leakage audits.
3. Q-Former preserves candidate/clue recall and node alignment.
4. Real clue combinations yield measurable non-additive teacher consequences.
5. The IWM beats single-step, additive, iterative, and shuffled controls on
   held-out consequence prediction.
6. Oracle trajectories have budget-feasible headroom.
7. Intact IWM predictions change real execution decisions.
8. The full system improves grounded accuracy or read/call efficiency over the
   reactive Reasoner at matched budget.

Failed gates must fail closed. They must not be repaired by hidden relevance
Top-K, heuristic tie-breaking, or treating imagined consequences as evidence.

## 13. Migration from the current code

The migration should preserve frozen artifacts while replacing the main method
in stages:

1. keep `evidence/` as the grounded L1 substrate and formalize its pre-read/read
   boundary;
2. add a node-aligned Q-Former proposal interface for all eligible unread nodes;
3. add real combination rollouts and teacher consequence collection;
4. train/evaluate a compositional multi-step IWM;
5. adapt the Reasoner to consume rollout contexts and execute grounded reads;
6. demote `navigation/`, cursor-local graph rows, and graph action compilation to
   explicit legacy/baseline modes;
7. run the decisive matched-budget evaluation before scaling.

Existing single-case GPT-5-mini and complete-graph outputs remain useful for
substrate and evaluator regression tests. They do not establish IWM value under
the new definition because they did not train or test multi-step compositional
consequence prediction.
