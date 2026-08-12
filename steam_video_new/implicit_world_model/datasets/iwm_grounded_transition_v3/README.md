# Grounded IWM Transition Data v3

This export adds train-side delayed diversity after the v1 adapter failed to
generalize to the corrected v2 held-out split. It preserves the v2 validation
set unchanged.

Counts:

- train: 31 videos, 1336 records, 290 support, 1046 inconclusive, 22
  independent delayed units and 380 semantic-neighbor hard negatives;
- validation: 10 disjoint videos, 253 records, 77 support, 176 inconclusive,
  6 independent delayed units and 29 semantic-neighbor hard negatives;
- cross-split video overlap: zero.

The additional train paths use the same frozen-frontier, legal two-read
collection contract as v2 validation. They add independent executed examples;
no record duplication, class weighting, numeric reward or heuristic runtime
ranking is used. Ground truth selects and labels collection paths only after
the question-independent graph and public-question localization are frozen.

`targeted_delayed_train_192.json` contains 16 units from capacity-192 graphs.
`targeted_delayed_train_384.json` contains four units from capacity-384 graphs.
The combined export has 22 unique delayed `(video, question, target)` units
after record deduplication. Planner training remains locked until held-out
transition calibration passes at both record and independent-unit levels.

## A6000 LoRA result

A one-epoch Qwen3.5-9B LoRA run completed on one 48 GB A6000. Training used
1336 records, micro-batch 2, gradient accumulation 8 and 84 optimizer steps.
Training took 1819 seconds; the complete job, including 253 held-out
generations, took 50 minutes. The run is stored outside the repository at:

```text
/fs/gamma-projects/vlm-robot/steam_video_runs/iwm_9b_transition_lora_a6000_v3_b2/
```

The corrected held-out gate failed:

- 252/253 generations were valid categorical JSON and none emitted a numeric
  scalar;
- outcome/progress balanced accuracy was `0.494`;
- all 77 support records were missed;
- delayed record recall was `0/41`, and strict independent-unit recall was
  `0/6`;
- semantic-hard-negative false support remained `0/29`.

Increasing independent delayed train coverage therefore did not establish
cross-video transition learning. The adapter learned the conservative
inconclusive/unchanged majority behavior.

Post-run inspection found an input-representation mismatch. The text-only
LoRA receives an embedding *reference* (model, dimension, row and checksum),
not the Qwen embedding values. It also receives no consumable L1 caption when
`semantic_key` is absent. All 1336 train records expose a semantic key, versus
only 75/253 validation records; 36 validation support records even have an
empty executed descriptor. At the same time, the target asks the model to
generate the complete observation descriptor. In the only invalid generation,
the model repeated a participant list until it exhausted the inference budget.

Do not address this failure by adding epochs, duplicating positives, weighting
classes, adding heuristic Top-K, or relaxing calibration. The next data/model
version must first make the frozen question-independent L1/L1.5 representation
actually consumable by the IWM: either project the materialized Qwen embedding
into model tokens, or expose an audited question-independent L1 semantic
descriptor. The observation-prediction target and belief-delta target should
then be evaluated separately before Planner preference training is unlocked.

An exact tokenizer audit confirmed that none of the 1336 training examples was
truncated at `max_length=1792` (the longest complete prompt plus target was
1313 tokens). The invalid held-out row is therefore counted as a model-output
failure, not hidden training-data damage.
