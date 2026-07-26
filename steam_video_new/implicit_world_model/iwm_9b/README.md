# Shared 9B IWM and Multi-Path Planner

This package implements the first training boundary described in
`plan/README.md`:

- strict V2 masked-supervision records;
- an adapter for the existing 670 grounded transition records;
- an adapter for blinded/hidden local-choice preference packets;
- split-safe and label-diversity readiness gates;
- stable chat serialization with no numeric reward;
- separate LoRA SFT entry points for `iwm_transition` and
  `planner_preference` on a shared Qwen 9B base model.

It does not treat missing labels as negatives. It also refuses Planner SFT when
the train split has fewer than two preference classes.

## Recommended grounded training order

The positive-only 670-record adapter below is retained for reproducibility,
but it is not the recommended training source. It lacks executed
inconclusive controls, delayed positives and semantic-neighbor hard negatives.

`grounded_runtime_data.py` builds the current source from closed-loop real
reads:

```text
persistent hypothesis belief + executed legal action + safe L1/L1.5 context
→ executed L1 observation descriptor
→ evaluator-grounded categorical clue-coverage delta
```

The observation payload comes from the actual read. The model-generated belief
correction is stored only under `audit` and is masked from supervision. Missing
hypothesis-specific logical deltas are null/masked, never converted into
negative labels. Each record preserves the Qwen3-VL-Embedding-2B reference for
future navigation without placing the embedding vector in JSON.

The data gate requires video-disjoint train/validation splits, both support and
inconclusive outcomes, delayed positives and executed correlation-neighbor hard
negatives. Delayed coverage is counted both as records and as independent
`(video, question, target)` units, so hypothesis expansions of one read cannot
inflate readiness. It unlocks only transition SFT. Planner records require a
second, independent held-out transition-calibration report, so the intended
order is:

```text
grounded transition data gate
→ 9B transition SFT/distillation
→ held-out categorical calibration gate
→ categorical trajectory-preference SFT
→ small matched IWM/no-WM evaluation
→ full frozen cohort only after IWM wins
```

No stage uses numeric model rewards, heuristic Top-K, forced tie-breaking or
GTSAM action ranking. GTSAM remains an optional persistent-belief backend only.

Legacy cohort files that stored an executed `observation_id` without its
payload can be migrated only against their frozen compile gate. The migration
recomputes the exact graph fingerprint and refuses a mismatch; it never invents
a counterfactual read.

Build the dataset without training:

```bash
python -m steam_video_new.implicit_world_model.iwm_9b.grounded_runtime_data \
  --artifact /path/to/train/cohort.final.json \
  --artifact-dir /path/to/legacy/cases \
  --compile-gate /path/to/frozen/compile_gate.json \
  --dataset /path/to/navigation_dataset.json \
  --hidden-key /path/to/terminal_targets.hidden_key.json \
  --output-dir /tmp/iwm_grounded_transition_v1
```

The builder exits nonzero and leaves all `training_eligible` flags false when
held-out data are insufficient. Once transition SFT has produced categorical
predictions for every validation record, evaluate them with:

```bash
python -m steam_video_new.implicit_world_model.iwm_9b.calibration_gate \
  --records /tmp/iwm_grounded_transition_v1/records.jsonl \
  --predictions /path/to/validation_predictions.jsonl \
  --output /tmp/iwm_grounded_transition_v1/calibration.json
```

The model prediction file contains labels only. Accuracy, balanced accuracy,
confusion matrices, delayed-positive recall, semantic-hard-negative support
rate and ready false-discovery rate are computed offline by the evaluator.
Re-run the data builder with `--calibration-report` only after this gate passes
to unlock train-split Planner preference records.

The supervised transition target is numeric-free: confidence values, frame
indices and timestamp regression are removed. The target address/time span is
already part of the legal action input, so the model predicts the categorical
binding `executed_target_interval` rather than seconds. Natural-language
descriptors plus categorical deltas remain; numeric metrics are evaluator-only.

### Current grounded export

`datasets/iwm_grounded_transition_v1/` passed the original record-count
transition-data gate used for the A6000 pilot:

- train: 31 videos, 1165 transitions, 188 support, 977 inconclusive, 50 delayed
  positives and 380 semantic-neighbor hard negatives;
- validation: 5 disjoint videos, 108 transitions, 30 support, 78 inconclusive,
  8 delayed positives and 14 semantic-neighbor hard negatives;
- cross-split video overlap: zero;
- transition dry-run: 1165 records were eligible under that original gate;
- Planner dry-run: correctly blocked because transition calibration has not
  been performed.

The post-pilot v0.2 gate also counts independent delayed units. Under this
corrected gate, train has seven independent delayed units but validation has
only one, below the default minimum of three. The checked-in readiness artifact
is retained as the exact historical pre-pilot v0.1 result; it must not be used
to authorize a new run. Rebuild the dataset after collecting independent
validation units.

Embedding slots are present on every transition. 1227 point to available
Qwen3-VL-Embedding-2B manifest rows; 46 consolidated-node slots are explicitly
marked `refresh_required` with their source lineage and requested model. No
vector or row index is fabricated.

### A6000 transition-LoRA pilot

A one-epoch Qwen3.5-9B LoRA pilot was run on one 48 GB A6000 with the grounded
export above. It trained on all 1165 train transitions for 73 optimizer steps
(`r=16`, `alpha=32`, effective batch size 16, maximum length 1792). The adapter
is about 116 MB. Training and held-out inference completed in 56 minutes; all
108 generations were valid JSON, contained no numeric scalar, and respected
the categorical-output contract.

The held-out result is promising but does **not** pass the transition gate:

- observation outcome and progress accuracy: `0.852`;
- observation outcome and progress balanced accuracy: `0.733`, versus `0.500`
  for the train-majority baseline;
- support confusion: 14 true positives, 16 false negatives and zero false
  positives;
- semantic-neighbor hard-negative false-support rate: `0.000`;
- delayed-positive recall: `0/8`, so Planner training remains blocked;
- coverage balanced accuracy: `0.417`; answerability balanced accuracy:
  `0.500`.

The delayed failure exposes a data-split problem rather than a reason to add
epochs or heuristic routing. The 50 train delayed records cover seven videos
and seven targets, all with available embeddings and correlation/temporal
actions. The eight validation records are eight hypothesis expansions of only
one video/question/target: a consolidated `start_at` node whose embedding is
`refresh_required` and whose local temporal/correlation context is empty. That
input does not contain enough observable structure to distinguish the desired
delayed transition. Before another training run, refresh consolidated-node
embeddings/correlation context and rebuild a validation delayed slice with
multiple independent videos, questions, targets and action types. Do not
oversample the duplicated records or relax the delayed-recall gate.

Run artifacts are outside the repository at:

```text
/fs/gamma-projects/vlm-robot/steam_video_runs/iwm_9b_transition_lora_a6000_v1/
```

### Corrected v2 export

`datasets/iwm_grounded_transition_v2/` keeps the same 31-video/1165-record
training split and replaces the defective held-out slice with 10 disjoint
videos and 253 records. It contains six independent delayed units, expanded to
41 hypothesis-conditioned records, plus 29 semantic-neighbor hard negatives.
All 253 validation target nodes have materialized Qwen3-VL-Embedding-2B
references; the old consolidated `refresh_required` delayed target is absent.

The delayed collector freezes the question-independent graph and the existing
question-localized entry frontier before joining GT. It then executes a legal
non-clue first read followed by a clue-overlapping temporal or L1.5 correlation
neighbor. GT is used only for targeted collection and evaluator labels, never
as model input or runtime action ranking. The v0.2 independent-unit data gate
passes. A second matched A6000 LoRA/calibration run uses this export; Planner
training remains locked until that report passes.

The reused v1 adapter does not pass this corrected split. All 253 batched
generations are valid and numeric-free, but it recalls 0/77 support records and
0/6 delayed units; outcome/progress balanced accuracy is `0.491`. The earlier
v1 score therefore did not demonstrate cross-video transition generalization.
Calibration now reports both record recall and strict independent-unit recall;
all hypothesis expansions of a delayed unit must predict support for that unit
to count as recalled.

`datasets/iwm_grounded_transition_v3/` addresses the corresponding train-side
coverage gap without loss weighting or record duplication. It adds legal,
executed targeted paths from 20 train videos, producing 1336 train records,
290 support outcomes and 22 independent delayed units after deduplication. The
v2 validation split is unchanged.

The new one-epoch Qwen3.5-9B LoRA run completed on an A6000 in 50 minutes
(84 optimizer steps; training time 1819 seconds). It did **not** pass the
held-out gate: 252/253 generations were valid categorical JSON with no numeric
scalar, but outcome/progress balanced accuracy was `0.494`; all 77 support
records were missed; delayed recall was `0/41` records and `0/6` strict
independent units. Hard-negative false support remained `0/29`. Planner
training therefore remains locked.

The result exposes an information-regime mismatch, not a reason to add epochs
or heuristic routing. This text-only LoRA sees embedding references but never
the materialized Qwen embedding values. All 1336 train inputs expose a semantic
key, whereas only 75/253 validation inputs do, and 36 validation support
records have an empty executed descriptor. Nevertheless the supervised target
requires the complete observation descriptor. In the sole invalid generation,
the model repeated a participant list until it exhausted the inference budget.
An exact tokenizer audit found zero truncated train targets at
`max_length=1792` (maximum complete length: 1313 tokens), so this is not silent
training-target truncation. The next version must
make the frozen, question-independent L1/L1.5 representation consumable through
an embedding projector or an audited L1 semantic descriptor, and separately
calibrate observation prediction and belief delta. Positive duplication, loss
weighting, relaxed gates and heuristic Top-K are not valid fixes.

Run artifacts are outside the repository at:

```text
/fs/gamma-projects/vlm-robot/steam_video_runs/iwm_9b_transition_lora_a6000_v3_b2/
```

## Export the existing data

```bash
cd /fs/gamma-projects/vlm-robot/steam_video

python -m steam_video_new.implicit_world_model.iwm_9b.export \
  --transition-corpus steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_supervision_phase2/grounded_action_transition_corpus.json \
  --local-choice-public steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/blinded_local_choices.json \
  --local-choice-hidden steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/grounded_labels.hidden_key.json \
  --output-dir /tmp/steam_iwm_9b_existing
```

The expected result for this historical export is:

- transition serialization ready from train-split scoped positive records,
  but not sufficient for the recommended grounded transition gate;
- Planner SFT blocked because the available grounded comparisons are test-only;
- joint multi-task SFT blocked;
- `training_performed=false`.

## Preflight either adapter

```bash
python -m steam_video_new.implicit_world_model.iwm_9b.train_sft \
  --records /tmp/steam_iwm_9b_existing/records.jsonl \
  --task iwm_transition \
  --output-dir /tmp/steam_iwm_9b_transition \
  --dry-run
```

Running the equivalent `planner_preference` command must fail closed until a
multi-video train-split grounded preference packet exists.

## LoRA training

Create an isolated environment with `requirements-sft.txt`, then omit
`--dry-run`. The A6000 pilot used the existing Qwen3.5 serving environment;
`peft_compat.py` narrowly disables a stale, unused TorchAO dispatch path only
after confirming that the loaded model contains no TorchAO tensor subclasses.
The two tasks write separate adapters:

```text
iwm_transition_adapter
planner_preference_adapter
```

A successful dry-run validates data and serialization only; it does not claim
model training or navigation quality. A completed LoRA run likewise does not
authorize Planner training until the independent calibration gate passes.

## Multi-path runtime

`runtime.py` implements the trained-model boundary for the existing
`MultiTrajectoryRolloutPlanner`. A serving client only needs the existing
`complete_json(task, payload)` interface, so it can be backed by a local
OpenAI-compatible vLLM endpoint or a scripted test client.

The runtime:

- predicts a structured observation patch, belief patch and categorical audit
  for every hypothesis-conditioned legal action;
- attaches the rich patch to `ImaginedTransition`;
- carries the patch through horizon-two rollouts and persistent transition
  caches;
- presents every hypothesis-conditioned continuation to the Planner;
- groups continuations as complete shared-first-hop action trees;
- requires every joint tree to cover every active hypothesis;
- outputs only categorical partial preference;
- never copies a structured imagined patch into persistent belief.

One executed shared read now updates the graph cursor and action history of
every persistent reasoning path. Semantic support, counterevidence and role
resolution remain hypothesis-conditioned. This distinction is required for
the next replan: every path took the same real navigation action, but each path
may interpret the resulting evidence differently.

The scripted end-to-end test closes two horizon-two cycles:

```text
IWM rich multi-path rollout
→ Planner shared-first-hop tree preference
→ one real read
→ broadcast navigation-state update
→ hypothesis-conditioned belief correction boundary
→ invalidate imagined state
→ IWM and Planner replan
```

This is an engineering verification of the intended control flow. It is not a
claim that the untrained 9B model has learned useful transition dynamics.

Run the same two-cycle boundary with a real OpenAI-compatible model:

```bash
python -m steam_video_new.implicit_world_model.iwm_9b.model_smoke \
  --keys-py /fs/gamma-projects/vlm-robot/keys.py \
  --model openai/gpt-5-mini \
  --output /tmp/steam_iwm_9b_gpt5mini_smoke.json
```

The artifact records rich-patch coverage, joint-tree hypothesis coverage,
executed real reads, persistent-path retention and imagined-state leakage. It
remains an engineering smoke with `training_performed=false` and
`scientific_validation=false`.

### Real GPT-5-mini result

A real OpenRouter `openai/gpt-5-mini` run completed the two-cycle smoke:

```text
step zero: start_at l1:entry
step one:  temporal_forward l1:cabinet
termination: read_budget_exhausted
```

Both planning steps selected a unique preferred first hop. Each step retained
three complete shared-first-hop trees, every tree covered all active
hypotheses, every imagined transition carried a rich patch, both persistent
reasoning paths survived, and no imagined evidence entered persistent belief.
The second planning cycle required one strict complete-alias repair.

The live transport exposed and now tests these canonical variations:

- a sole `predictions` object flattened to top-level aliases;
- empty categorical lists serialized as `none`, `unbound` or an empty string;
- categorical enums wrapped in singleton arrays;
- comma-delimited categorical string lists;
- entity bindings serialized as `key:value` or a short binding description;
- `incomparable` accompanied by its complete undominated frontier.

Normalization is lossless and never creates a preference, reward, evidence
claim or missing alias. Unknown roles, incomplete alias coverage after repair,
numeric output and invalid frontier membership still fail closed.

The smoke uses a small grounded synthetic graph and an executed-fixture belief
updater to isolate GPT-5-mini as the IWM and Planner. It establishes that the
new direction can execute with a real model; it does not establish CG-Bench
accuracy, transition calibration, delayed-planning advantage or 9B
distillation quality.
