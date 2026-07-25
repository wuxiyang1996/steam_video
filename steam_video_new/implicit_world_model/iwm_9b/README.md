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

## Export the existing data

```bash
cd /fs/gamma-projects/vlm-robot/steam_video

python -m steam_video_new.implicit_world_model.iwm_9b.export \
  --transition-corpus steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_supervision_phase2/grounded_action_transition_corpus.json \
  --local-choice-public steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/blinded_local_choices.json \
  --local-choice-hidden steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/grounded_labels.hidden_key.json \
  --output-dir /tmp/steam_iwm_9b_existing
```

The expected current result is:

- transition SFT pipeline ready from train-split scoped positive records;
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
`--dry-run`. The two tasks write separate adapters:

```text
iwm_transition_adapter
planner_preference_adapter
```

The current repository environment intentionally does not carry the heavy
PyTorch/Transformers training dependencies. A successful dry-run validates
data and serialization only; it does not claim model training or navigation
quality.

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
