# Grounded IWM Transition Data v1

This directory is a data-gathering and readiness artifact. No 9B model was
trained.

The export contains 1273 executed transition records and 83 ordinal trajectory
comparison records. Transition train/validation videos are disjoint. Every
target observation was actually read from a frozen, question-independent
L1/L1.5 graph; dataset clue intervals are used only after execution to label
categorical clue-coverage change.

Key files:

- `transition_records.jsonl`: 1165 train and 108 validation transitions;
- `planner_preference_records.jsonl`: generated comparisons, all locked until
  transition calibration passes;
- `records.jsonl`: combined SFT input;
- `readiness.json`: counts, split audit, contracts and blockers;
- `heldout_collection/`: two fresh validation no-WM collection-policy runs and
  their current-code compile gate.

Current gate state:

```text
transition_training_ready = true
planner_training_ready = false
transition_calibration_provided = false
training_performed = false
```

The dataset intentionally masks unsupported hypothesis-specific logical belief
deltas. Model-generated post-read corrections remain audit-only. A missing
label is never interpreted as counterevidence or another negative class.

All records contain an embedding slot for future navigation. Existing nodes
reference Qwen3-VL-Embedding-2B manifest rows. Consolidated nodes without a
materialized vector are marked `refresh_required` and retain source lineage;
the export does not fabricate embedding values.
