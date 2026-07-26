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
hard negatives. `training_export.blocked.json` records the failed gate; no 9B
training export was produced.
