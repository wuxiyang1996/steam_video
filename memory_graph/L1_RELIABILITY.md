# L1 Reliability and the L1.5 Belief Overlay

## Decision

The existing Video_Skills L1 graph should remain the grounded source of truth. It is a suitable container and execution interface, but its perceptual reliability has not yet been demonstrated.

The architecture is therefore:

```text
Video_Skills L1 ClueMemoryGraph
  - clips and grounded observations
  - time spans
  - provenance and visibility
  - explicit observational facts
              |
              | references immutable L1 node IDs
              v
L1.5 Causal-Temporal Belief Overlay
  - atomic event hypotheses
  - entity and state hypotheses
  - uncertain explanatory support
  - contradictions and missing bridges
              |
              v
L2 evidence actions and belief-transition planning
```

L1 is not replaced or copied into another graph. L1.5 stores hypotheses that reference L1 evidence.

## What L1 May Claim

L1 should contain high-precision observations:

- clip and observation boundaries;
- visible entities, actions, objects, and dialogue;
- explicit state assertions;
- source modality;
- provenance;
- visibility and hidden-supervision status;
- uncertainty and grounding references.

L1 may store a `causal_hint` as an unverified proposal, but it must not expose `explains`, `enables`, motivation, hidden identity, or inferred outcomes as facts.

## What Belongs in L1.5

L1.5 contains revisable hypotheses:

- event decomposition inside a coarse L1 observation;
- cross-node entity identity;
- state transitions;
- explanatory-support proposals;
- contradictions;
- missing or uninstantiated bridge events;
- question-conditioned posterior mass.

Every L1.5 object must cite one or more L1 node IDs. Removing its L1 evidence must invalidate or reduce the hypothesis.

## Current Reliability Assessment

The current implementation has reliable structural interfaces:

- canonical-to-L1 conversion;
- node IDs and references;
- explicit time spans;
- provenance and visibility fields;
- deterministic interval calculations.

The following are not yet proven reliable:

- video-only factuality;
- event atomicity;
- key-event coverage;
- cross-clip entity identity;
- model-generated causal hints;
- robustness against wrong captions and hidden-supervision leakage.

The first Video-Holmes validation used two examples and gold segment descriptions because cropped videos were unavailable. It validated the pipeline, not deployable video-only L1 reliability.

## Reliability Gates

Belief or navigation training must not start until a video-only L1 audit reaches:

| Metric | Gate |
|---|---:|
| Provenance completeness | 100% |
| Hidden-supervision leakage | 0 |
| Timestamp validity | at least 95% |
| Grounded event precision | at least 90% |
| Key-event coverage | at least 90% |
| Entity-link precision | at least 90% |
| Compound-event rate | at most 10% |

Metrics that require semantic judgment must be measured with independent human labels. Deterministic checks may flag failures but cannot estimate factual precision.

## Why This Matters

If L1 is unreliable, the world model can learn to plan around teacher hallucinations instead of evidence. More elaborate belief representations, factorization, or model-predictive control cannot recover distinctions that were never grounded correctly.

The implementation order is:

```text
L1 reliability benchmark
  → atomic event hypotheses grounded in L1
  → constrained causal-temporal overlay
  → belief-transition model
  → delayed-utility planning
```

## Novelty Boundary

The L1 graph, embedding retrieval, event extraction, and causal-edge labeling are infrastructure rather than the main paper claim.

The intended contribution is:

> Action-conditioned belief-transition learning and delayed-utility evidence planning over an uncertain causal-temporal overlay grounded in fixed-capacity video memory.

The novelty requires:

- sibling epistemic-action trajectories from the same belief checkpoint;
- prediction of how a graph read changes future belief and answerability;
- one- or two-step imagined rollouts;
- execution of only the first action;
- posterior correction from a real L1 observation;
- evidence that bridge actions outperform greedy retrieval under equal graph-read budgets.
