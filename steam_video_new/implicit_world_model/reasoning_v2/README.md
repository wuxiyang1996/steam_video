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
