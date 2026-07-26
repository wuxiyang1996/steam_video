# Implicit world model workspace

The active architecture is [reasoning_v2](reasoning_v2/README.md). It separates
the research method into six independently testable packages:

```text
reasoning_v2/evidence/     L1 address/value boundary and substrate audits
reasoning_v2/navigation/   L1.5 typed navigation proposals and calibration
reasoning_v2/world_model/  observation prediction and hypothesis effects
reasoning_v2/planner/      persistent multi-reasoning-path planning
reasoning_v2/belief/       optional post-read GTSAM categorical correction
reasoning_v2/evaluation/   clue, reachability, delayed, and oracle gates
```

Historical components remain in place for artifact reproduction:

- `full_graph_iwm/`: v1 single-cursor, answer-hypothesis runtime;
- `iwm_9b/`: existing data adapters and halted LoRA pilots;
- `l15_graph_navigator/`: earlier graph/factor-graph experiments;
- `cgbench_grounded_navigation/`: frozen graph workers and dataset tooling.

New method work must not add more responsibilities to `full_graph_iwm`. Frozen
v1 graphs enter v2 through explicit evidence and navigation adapters. GTSAM is
isolated in `reasoning_v2/belief`: it receives one shared executed-read
measurement and returns only per-path categorical corrections. It is not part
of imagined observation prediction or trajectory preference, and its numeric
solver state is never model-visible.

The v2 planner decision unit is a joint action tree, not one answer-hypothesis
trajectory. Every legal shared first read is evaluated together with all of its
hypothesis-conditioned one/two-hop futures before the categorical choice.

No 9B training is authorized until the gates listed in
`reasoning_v2/README.md` pass on a fixed multi-video cohort.

The first real GPT-5-mini v2 smoke completed the full execute/correct/replan
loop but failed delayed entry retention and intervention-divergence gates. See
the v2 README for the exact result. This is an infrastructure validation, not
evidence that zero-shot IWM improves QA, and it did not release training data.
