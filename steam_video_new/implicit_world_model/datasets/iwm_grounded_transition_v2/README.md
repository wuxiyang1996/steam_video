# Grounded IWM Transition Data v2

This export fixes the delayed-validation defect found by the first A6000 LoRA
pilot. It is built from real reads on frozen, question-independent L1/L1.5
graphs. Ground truth is joined only after execution to label clue-coverage
change; hidden clue intervals and answers never enter model inputs.

Current transition counts:

- train: 31 videos, 1165 records, 188 support, 977 inconclusive, 7 independent
  delayed `(video, question, target)` units;
- validation: 10 disjoint videos, 253 records, 77 support, 176 inconclusive,
  6 independent delayed units and 29 semantic-neighbor hard negatives;
- validation delayed records: 41 hypothesis-conditioned expansions over those
  6 independent units;
- cross-split video overlap: zero;
- validation embedding coverage: 253/253 materialized
  `Qwen/Qwen3-VL-Embedding-2B` references.

The six targeted delayed units execute legal two-read paths. The first action
starts at a frozen question-localized node with no dataset-clue overlap; the
second follows an actual temporal or L1.5 correlation edge to a node with clue
overlap. Five second hops are correlation actions and one is temporal. This is
targeted data gathering, not a runtime oracle: GT selects and labels collection
paths but is absent from the exported IWM input.

`targeted_delayed_validation.json` preserves the executed paths and the data
collection audit. `readiness.json` uses the v0.2 gate, which counts independent
delayed units rather than allowing hypothesis expansion to inflate readiness.
Planner preference training remains locked until an independently trained IWM
passes held-out categorical calibration.
