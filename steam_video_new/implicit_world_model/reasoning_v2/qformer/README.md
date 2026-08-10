# Four-slot dual Q-Former baseline

This package implements the staged baseline described in
`plan/qformer-four-slot-baseline-plan.md`:

```text
caption/entity-state/visual/time
  -> QF1 (8 independent learned queries)
  -> frozen node-token cache
  -> QF2 (8 independent question-conditioned queries)
  -> trusted-negative multi-positive retrieval
```

QF1 reconstructs the four **frozen raw source embeddings** through four
heterogeneous output heads. This prevents a trainable target projector from
collapsing together with the reconstruction decoder. Missing slots are masked.

QF2 uses only cross-video nodes as automatic trusted negatives. Same-video
non-positive nodes remain unlabeled and are excluded from the loss denominator.
The pooled frozen question embedding is the single question token for v0.1;
token-level question encoding is a later matched extension.

## Data boundary

Feature materialization consumes question-independent L1 overlays and raw video
only. Retrieval labels are compiled after graph freeze by an evaluator-only
interval-overlap step. The compiled output contains question text and node IDs,
but no answer field or grounded post-READ value.

The current train collection has 40 public-train videos. The fixed 36-video
cohort contains validation/test videos only. QF1/QF2 fitting must use the former;
the latter is held out. Training on the held-out cohort is at most a transductive
engineering diagnostic and must not be reported as an inductive result.

## Entry points

```bash
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.audit_features ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.materialize_cohort ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.train_qf1 ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.export_qf1 ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.compile_retrieval_labels ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.materialize_questions ...
python -m steam_video_new.implicit_world_model.reasoning_v2.qformer.train_qf2 ...
```

Cluster launchers live in `/fs/gamma-projects/vlm-robot/cluster` and write all
large arrays/checkpoints under `/fs/gamma-projects/vlm-robot/steam_video_runs`.
