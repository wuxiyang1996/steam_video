# Reasoning v2

This package is the clean implementation path for world-model-guided reasoning.
It does not modify frozen v1 artifacts and it does not train a model.

## Data flow

```text
question-independent video
  -> evidence/     L1 address + hidden-until-read grounded value
  -> navigation/   L1.5 typed local proposals, never factual confidence
  -> world_model/  hypothesis-independent observation prediction
                  + separate hypothesis-conditioned belief effect
  -> planner/      persistent forest of different reasoning/action histories
  -> belief/       optional GTSAM correction after one real shared read
  -> evaluation/   independent substrate, reachability, oracle, and model gates
```

## Non-negotiable contracts

- An unread address does not expose its grounded predicate, entities, states, or
  attributes.
- The same physical action has one observation prediction. Answer hypotheses
  may affect belief interpretation, never the observation itself.
- L1.5 correlation is a navigation proposal, not a probability, causal edge,
  identity claim, relevance label, or action score.
- The planner compares joint action trees. Each tree groups every available
  hypothesis-conditioned future under one shared physical first read; it may
  not select an isolated single-hypothesis trajectory. There is no second
  scheduler that silently becomes the real planner.
- Unselected reasoning alternatives remain suspended with their original
  action histories. One real read enters shared evidence memory without
  rewriting every path into the selected trajectory.
- Normal replanning expands active paths only. Suspended paths remain
  recoverable state and may be reactivated only by an explicit recovery
  decision; retaining a path does not mean expanding it every round.
- No fixed Top-K pruning and no numeric value emitted by an LLM. Numeric metrics
  exist only in offline evaluators.
- GTSAM remains an optional real-belief correction backend, not the main method.
  Its adapter receives an executed action and grounded overlay node only after
  the read; one shared measurement is projected categorically to all paths.

## Migration

`evidence.legacy_adapter` and `navigation.adapter` read the existing
`RetainedEvidenceGraph`. The adapters deliberately downgrade legacy semantic
correlations to uncalibrated proposals and expose only coarse `action_kind` as
the pre-read event family. Existing `full_graph_iwm` remains available for
historical artifact reproduction but is not the v2 research implementation.

## Required gates before 9B training

1. Learned-representation L1 windowing and adequate grounded descriptor quality.
2. Complete clue retention on a frozen cohort.
3. Trusted hard-negative calibration for non-temporal L1.5 proposals.
4. Legal reachability plus explicit delayed cases requiring at least two hops.
5. Oracle-observation/oracle-effect headroom.
6. Independent observation calibration.
7. Independent hypothesis-effect supervision and divergent effects.
8. Grounded categorical trajectory preferences.

Until these pass, conservative baselines fail closed instead of manufacturing
belief progress.

## First real model-backed smoke (GPT-5-mini)

The first frozen CG-Bench case smoke is stored outside the training datasets at
`outputs/reasoning_v2/gpt5mini_case_03ae/model_smoke.json`. It exercised five
matched read-budget arms (`iwm`, `no_wm`, `shuffled_iwm`, `immediate_only`, and
`oracle`) plus one executed read and replan.

Confirmed infrastructure properties:

- all 192 safe semantic addresses were inspected without Top-K;
- physical observations were deduplicated across six hypotheses;
- every joint action tree covered all six hypothesis-conditioned outcomes;
- one real read was executed and a second planning round produced candidates;
- short request/tree aliases eliminated long-ID copy corruption;
- suspended paths were retained but not recursively expanded during normal
  replanning;
- strict categorical responses were cached, and no training was performed.

The scientific gates did **not** pass. The localizer selected nodes `0045` and
`0060`, while the hidden evaluator's delayed first clue was node `0021`. IWM,
no-WM, and shuffled-IWM consequently selected the same incorrect first read;
the immediate-only arm tied. The executed evidence was inconclusive for every
hypothesis, so real beliefs did not diverge. Answer accuracy, independently
calibrated transitions, and a multi-video fixed cohort were not available.

This failure must not be repaired with heuristic Top-K or forced tie-breaking.
It identifies the next data/model requirement: supervision for delayed entry
localization and grounded hypothesis-conditioned effects, with semantic-neighbor
hard negatives. Each smoke writes a sibling
`<artifact>.training_export.blocked.json`, so experiments cannot overwrite one
another's failed gates; no 9B training export was produced.

## Complete-graph dual-entry verification

The model-backed evaluator now runs two protocols over the exact same complete,
question-independent L1/L1.5 artifact:

- `oracle_entry` changes only the initial `entry_node_ids` to the hidden
  evaluator's first-clue nodes.  It is a downstream diagnostic: after the
  common first real read, IWM/no-WM/shuffled-IWM/immediate-only/oracle each
  correct belief and independently replan toward the next clue.
- `learned_entry` obtains the initial frontier by inspecting every safe address,
  then runs the same arms, horizon, two-read budget, correction, and replanning
  loop.  It is the end-to-end diagnostic.

All nodes and proposals remain present in both protocols.  An entry protocol
does not dump the graph into one prompt and does not expose unread evidence
values.  Legal actions are still compiled from entry nodes on round zero and
from graph adjacency after a real read.  Hidden clue IDs and shortest-path next
hops exist only in evaluator metrics and the evaluator-only oracle arm; they are
never sent to the IWM, learned localizer, or learned planner.

The output reports each arm's two round decisions, executed target sequence,
remaining matched read budget, hypothesis divergence after real correction,
and whether the replan selected a first hop on a shortest route to the later
clue.  Infrastructure, localization, transition, planning, intervention, and
answer metrics remain separate; they are not collapsed into a claimed model
improvement.  Failed scientific gates continue to block training export.

The first dual-entry result localized the evidence bottleneck more precisely.
Even oracle entry at node `0021` produced no real belief change because its L1
value was only `person holding object`; IWM then followed surface-color/package
associations to `0176` instead of the evaluator route target `0034`. The
substrate repair therefore adds time-aligned, question-independent subtitle
content to both the bounded address key and hidden-until-read value, regenerates
Qwen3-VL-2B embeddings, and adds a hidden evaluator audit requiring every clue
group to have a materially enriched grounded value. Temporal overlap alone no
longer passes this substrate gate.

Each real read now also reports whether correction changed belief, separately
from whether answer hypotheses diverged. A partial clue may legitimately
advance a shared missing role without distinguishing answers, so these metrics
must not be conflated. Delayed transition collection marks executed
correlation reads with no realized clue gain as the categorical slice
`surface_correlation_without_realized_clue_gain`; these are the required hard
negatives for surface matches such as color or packaging recurrence.

## Repaired complete-graph result and honest headroom

After subtitle enrichment and Qwen3-VL-Embedding-2B regeneration, the same
complete graph contains 256 nodes and 957 typed proposals, and all three hidden
clue groups have non-placeholder grounded values and current embedding
references. In the two-read GPT-5-mini replay, oracle entry now changes and
diverges real beliefs, although its second IWM action still misses the shortest-
path hop. Learned entry selects a more direct subtitle-bearing node and
produces the correct unique answer after its first real read. This shows that
the repaired substrate can support the closed loop; it does not yet show that
IWM improves navigation.

The evaluator prevents two false conclusions exposed by this replay:

- Dataset interval/node-ID recall is a strict localization metric, separate
  from grounded belief change and final answer accuracy. A direct evidence node
  outside the annotated interval remains a strict miss, but its correct answer
  is not described as an end-to-end failure.
- Route headroom uses exact graph distance. For this case, the nearest node in
  the first annotated clue group is two proposal edges from the next clue
  group, so reaching it requires three real reads including entry. A two-read arm can test correction and
  replanning but cannot pass the later-clue/answer oracle gate. Merely executing
  two oracle reads is no longer called “route available.”

This remains a single-case substrate diagnostic. IWM changes the oracle action
sequence relative to no-WM and shuffled-IWM but does not turn that divergence
into clue reach or a correct answer; learned-entry IWM and immediate-only both
obtain the same answer in one read. Training stays blocked until a fixed multi-video cohort has
budget-feasible oracle paths, independent transition calibration, and matched-
arm evidence that intact IWM predictions improve accuracy or read efficiency.
