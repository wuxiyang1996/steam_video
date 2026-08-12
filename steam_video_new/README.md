# World-Model-Guided Multi-Clue Video Reasoning

`steam_video_new` is the research workspace for grounded reasoning over long
video with a lightweight Evidence World Model (IWM). The active research target
is no longer explicit graph navigation. It is counterfactual prediction of how
alternative multi-step evidence-acquisition trajectories would change later
reasoning.

> **Architecture status (2026-08-05):** the target design is specified in
> [`implicit_world_model/reasoning_v2`](implicit_world_model/reasoning_v2/README.md).
> Existing L1.5 graph, cursor, Qwen listwise, categorical planner, and
> `full_graph_iwm` paths remain reproducible historical baselines. They must not
> be described as the final method.

## 1. Problem definition

Long videos contain many candidate evidence nodes. Reading source clips and
invoking a large Reasoner for every branch is expensive. The project asks:

> Can a lightweight world model predict the joint reasoning consequences of
> several possible multi-step, multi-clue acquisition trajectories before they
> are executed, and thereby help a grounded Reasoner avoid wasteful or myopic
> search?

The target setting includes cases where per-node relevance fails:

- no single clue answers the question, but a clue combination does;
- one clue changes the interpretation or retrievability of another;
- superficially relevant clues are mutually redundant;
- a new clue contradicts the current hypothesis;
- an early read has delayed value because it enables a later evidence link.

Accordingly, the IWM models delayed, conditional, and non-additive consequences
of acquisition. It is not a relevance ranker, a static graph scorer, a KV-cache
memory abstraction, or an answer generator.

**Primary testbed: CG-Bench.** Active supervision and held-out evaluation use
multi-clue cases with grounded `clue_intervals`. Video-Holmes artifacts are
historical engineering references rather than the current evaluation protocol.

## 2. Research thesis

```text
question-independent grounded L1 evidence memory
        ├─ bounded pre-read node view
        └─ full grounded value, hidden until READ
        ↓
node-aligned Q-Former proposals for every eligible unread node
        ↓
multi-step compositional IWM rollouts over clue combinations
        ↓
Reasoner compares trajectories and performs grounded execution
        ↓
real observation replaces prediction; stop, deviate, or replan
        ↓
answer from executed evidence, otherwise abstain
```

The core claim is deliberately narrow:

> We learn a lightweight evidence world model that predicts the non-additive
> reasoning consequences of multi-step, multi-clue acquisition trajectories
> from node-aligned Q-Former proposals. A separate Reasoner uses these
> predictions to perform grounded multi-step evidence acquisition and QA.

## 3. Component responsibilities

| Component | Responsibility | Must not do |
|---|---|---|
| L1 grounded memory | Store traceable evidence nodes, timestamps, source intervals, provenance, visibility, and read state | Expose a hidden grounded value before execution; treat imagined content as fact |
| Q-Former | Produce fixed-size, node-aligned pre-read proposal tokens for every eligible unread node | Choose the next read; predict a trajectory; answer the question |
| IWM | Predict latent consequences of alternative multi-step clue combinations | Generate factual captions; write memory; provide answer evidence or a final answer |
| Reasoner | Compare rollout contexts, execute real reads, replan, answer, or abstain | Cite imagined consequences as evidence |
| Deterministic controls | Enforce time, visibility, provenance, availability, and budget constraints | Introduce semantic adjacency or a hidden relevance policy |
| L2 trace | Record executed reads, observations, prediction errors, decisions, and outcomes | Become the factual evidence source or the world model |

The decisive invariant is:

```text
IWM:      multi-step prediction
Reasoner: multi-step grounded execution
```

## 4. L1 grounded evidence memory

L1 remains the factual substrate. A node contains at least:

- node ID;
- video interval and timestamp;
- grounded visual/text content;
- provenance;
- visibility and read status.

The complete value is available to the answer context only after `READ(i)`.
Before that action, downstream models receive only the explicitly permitted
low-bandwidth node view.

The explicit L1.5 semantic/correlation graph is removed from the main method.
The target system does not require static semantic adjacency, cursor-local
`A_corr/A_temp` rows, or a graph action compiler. Temporal ordering and other
deterministic execution constraints may still be applied as masks. Graph-based
artifacts remain useful for reproducibility and ablations but no longer define
the paper architecture.

## 5. Q-Former proposal interface

For question `q`, candidate node input `X_i`, and time `t_i`, Q-Former emits:

```text
p_i = Q_phi(q, X_i, t_i) in R^(m x d)
```

The fixed number of tokens is a node-aligned pre-read proposal representation.
It can encode coarse permitted content, relation to the question, temporal
position, and a possible evidence role. The full proposal set is

```text
P = {p_1, ..., p_N}.
```

Node IDs and masks remain explicit so that the IWM can compose proposals without
losing their executable targets. Candidate order is randomized during training,
and output alignment/permutation consistency is evaluated directly.

The main proposal path must not silently drop candidates with a semantic score,
threshold, or fixed Top-K. If bounded proposal selection is later necessary for
efficiency, it must be evaluated as a separate recall-audited component against
the all-eligible-node reference.

## 6. Multi-step compositional IWM

Let `z_0` be the reasoning state formed only from the question and already-read
grounded evidence. The IWM imagines interventions over proposal subsets:

```text
z_(h+1) = T_theta(z_h, READ(S_h), {p_i : i in S_h})
```

for a trajectory `tau = (S_1, ..., S_H)`. Each `S_h` may contain one or several
candidate clue nodes.

The IWM predicts how the sequence would jointly change:

- the reasoning state;
- the remaining evidence need;
- expected answerability;
- the choice to continue, answer, or abstain;
- contradiction or insufficiency.

Independent node scores are not sufficient because the target obeys

```text
C(A, B) != C(A) + C(B).
```

The IWM output is a compact rollout context: candidate trajectory, stepwise
latent consequences, and predicted terminal state. It contains no imagined
caption, grounded fact, scalar action utility, or final answer.

## 7. Grounded Reasoner loop

The Reasoner consumes current real evidence and IWM rollout contexts. It may
follow the suggested trajectory, stop early, deviate, or replan after a mismatch.

```text
for each decision round:
    z_0 <- question + executed grounded evidence
    P   <- Q-Former proposals for eligible unread nodes
    R   <- IWM rollouts for alternative trajectories
    a   <- Reasoner decision from grounded state + R
    if a is READ(S):
        observation <- execute real read
        append observation to grounded memory/trace
        compare predicted and realized consequences
    else if a is ANSWER:
        answer using executed evidence only
    else:
        abstain or wait
```

The Reasoner is not obligated to execute an imagined trajectory end to end.
Real observations have priority over predictions at every step.

## 8. Why the IWM is not just iterative reasoning

Repeated attention or representation refinement over supplied features has the
form

```text
z_(h+1) = F(z_h, P).
```

That alone is not a world model. In this project, every predicted step
corresponds to an explicit evidence intervention `READ(S_h)`, and its target is
the state produced after a teacher Reasoner actually reads the same combination:

```text
READ(S_(1:h)) -> r*(S_(1:h)).
```

The distinction is the training object, not network depth. An iterative module
computes over existing features; the IWM predicts the consequences of evidence
that has not yet been acquired.

## 9. Training supervision

Final QA loss cannot establish that the IWM learned acquisition dynamics. From
the same immutable initial state, the data pipeline executes real branches such
as:

```text
empty, A, B, A+B, A+C
```

A frozen teacher Reasoner produces a functional state for each branch:

```text
r*(S) = R_teacher(q, E_0 union E_S).
```

Targets can include:

- answerability;
- next evidence need;
- later read distribution;
- answer distribution;
- `continue | answer | abstain`;
- contradiction or insufficiency.

The IWM sees only pre-read proposal tokens and the imagined intervention
sequence. Training minimizes the discrepancy between predicted and real
post-read consequences.

Required data slices include clue synergy, leave-one-out, redundancy,
contradiction, conditional relevance, different clue prefixes, exchangeable and
order-sensitive sequences, and missing-key-evidence abstention. Surface-similar
reads with no realized clue gain are required hard negatives.

## 10. Matched evaluation

All arms share the same Q-Former, Reasoner, L1 memory, real-read budget, and
large-model call budget:

1. proposal tokens directly to the Reasoner;
2. parameter-matched deep Transformer/iterative reasoning;
3. reactive read-then-replan Reasoner;
4. single-step consequence model;
5. additive independent-node consequences;
6. multi-step compositional IWM;
7. shuffled IWM rollouts;
8. oracle real post-read consequences.

The evaluation reports:

- multi-clue evidence-chain completion;
- grounded answer accuracy;
- answerability and abstention;
- combination-consequence prediction;
- synergy/redundancy discrimination;
- reads per question;
- Reasoner forward calls;
- rollout error versus horizon;
- the causal effect of predictions on executed paths.

Strict clue-interval/node recall, grounded belief change, and final answer
accuracy remain separate metrics. A correct answer from evidence outside an
annotated interval is not relabeled as a localization success, and a localization
miss is not automatically described as an end-to-end QA failure.

## 11. Go/No-Go criterion

The IWM is valuable only if cheap counterfactual rollout improves real decisions.
At a matched budget it must deliver at least one of:

- higher grounded answer accuracy;
- higher multi-clue chain completion;
- better calibrated answerability/abstention;
- fewer full evidence reads;
- fewer expensive Reasoner calls.

If a reactive Reasoner produces the same trajectories and outcomes at the same
budget, the IWM is unnecessary. Shuffled, additive, iterative, and single-step
controls must also rule out extra parameters or generic computation as the
explanation.

## 12. Implementation status and migration

The repository contains substantial v1/v2 infrastructure for grounded nodes,
graph navigation, single-step categorical effects, planners, and matched replay.
Those artifacts exposed useful substrate issues, including incomplete grounded
values, surface-correlation hard negatives, budget-infeasible oracle routes, and
the need to separate localization from belief change. They remain regression
assets.

They do **not** yet implement or validate the architecture described here. In
particular, prior model-backed smokes did not train a node-aligned Q-Former plus
multi-step compositional consequence model from real sibling executions.

Migration order:

1. freeze and audit L1 grounded evidence plus the pre-read/read boundary;
2. implement the node-aligned Q-Former proposal interface;
3. collect real multi-combination teacher trajectories from immutable states;
4. train and calibrate the multi-step IWM;
5. connect rollout contexts to grounded Reasoner execution;
6. move explicit graph navigation to named legacy/baseline modes;
7. run the fixed-cohort matched evaluation before larger-model scaling.

Training and scientific claims remain blocked until the full gates in the
[Reasoning v2 specification](implicit_world_model/reasoning_v2/README.md#12-required-gates)
pass.

## 13. Repository map

| Path | Role |
|---|---|
| `implicit_world_model/reasoning_v2/` | Active target specification and migration path |
| `implicit_world_model/datasets/` | Grounded transition, runtime, audit, and pilot artifacts |
| `implicit_world_model/cgbench_grounded_navigation/` | CG-Bench substrate and evaluation tooling |
| `implicit_world_model/full_graph_iwm/` | Historical explicit-graph IWM baseline |
| `implicit_world_model/l15_graph_navigator/` | Historical L1.5 navigation implementation |
| `implicit_world_model/iwm_9b/` | Earlier scaling/training experiments; not the target architecture |
| `memory_graph/` | L1 evidence-memory reliability and validation work |
| `factor_graph/` | Optional correction/diagnostic baselines |

## 14. Non-negotiable scientific boundaries

- Unread grounded values never enter proposal tokens, IWM prompts, or answers.
- Imagined consequences never become memory facts.
- The final answer cites only executed evidence.
- Deterministic constraints may enforce legality but not semantic preference.
- Candidate dropping must be explicit and recall-audited.
- Teacher targets come from real reads at immutable checkpoints.
- Evaluation uses video-disjoint splits and matched budgets.
- Failed gates fail closed; heuristic Top-K or forced tie-breaking cannot repair
  missing model evidence.
