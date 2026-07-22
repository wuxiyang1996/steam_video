# Grounded Survivor Preference Slice v1

This is a post-planning diagnostic dataset for the fixed full-graph IWM. It
does not train a model and is never fed back into the planner.

## Files

- `blinded_pairs.json`: question/belief plus anonymous imagined trajectory
  consequences. It contains no target node IDs, timestamps, hidden clue
  intervals, answers, labels, or numeric rewards.
- `proxy_eval_summary.json`: aggregate original/side-swapped GPT-5-mini results
  without pair-level labels, target IDs, clue intervals, or answers.
- `grounded_labels.hidden_key.json` (local, intentionally gitignored):
  categorical labels and endpoint provenance derived from CG-Bench clue
  intervals after planning.
- `gpt5mini_proxy_eval*.hidden_key.json` (local, intentionally gitignored):
  pair-level GPT-5-mini decisions joined with labels only after inference.

The packet contains 44 strict contrasts from two fixed videos: one final
survivor tie and 43 exhaustive survivor-versus-grounded counterfactual pairs.
There are 22 `prefer_left` and 22 `prefer_right` labels. Two strict preferences
are hidden by exact imagined outcome/belief-delta collisions.

The label policy is deliberately conservative:

- prefer the only endpoint with dataset-grounded clue overlap;
- tie only when both endpoints have the same non-empty clue coverage;
- otherwise mark the pair incomparable.

Clue counts are not converted into reward, utility, confidence, or probability.
Different non-empty clue sets remain incomparable rather than being ranked.

## Proxy result

GPT-5-mini obtains 21/44 correct (47.7%). On the original orientation it emits
41 `prefer_left`, two `tie`, and one `incomparable`; after swapping every pair it
emits 42 `prefer_right` and two `tie`. Mapping the swapped predictions back to
their original semantic orientation gives 43/44 consistency (97.7%).

This is not a simple left-position shortcut. The packet places the original IWM
survivor on the left, and the comparator consistently follows that survivor's
overstated imagined consequence even after it moves to the right. The principal
error is upstream transition semantics; the preference head amplifies it.

## Rebuild

```bash
python -m steam_video_new.implicit_world_model.full_graph_iwm.survivor_preference_data \
  --run /path/to/postfix_puppy_run.json \
  --run /path/to/postfix_driving_run.json \
  --dataset ../navigation_dataset.qwen_grounded_embedded.json \
  --hidden-key ../terminal_targets.qwen_grounded_embedded.hidden_key.json \
  --graph-root ../l15_graph_smoke_v1 \
  --output blinded_pairs.json \
  --hidden-output grounded_labels.hidden_key.json
```

## What this slice establishes

The graph contains grounded alternatives that are better than the chosen or
tied endpoints. The current failure is therefore downstream of candidate
availability:

1. some distinct grounded actions collapse to the same imagined transition;
2. the IWM over-credits isolated lexical roles without enforcing joint
   identity/temporal evidence requirements;
3. the anonymous comparator cannot recover information already erased by the
   transition descriptor;
4. the bounded setwise tournament may introduce bracket sensitivity and needs
   a separate permutation audit before it is used as method evidence.

The labels supervise categorical action-conditioned reasoning effects. They do
not justify a heuristic Top-K, a scalar reward, or forced tie execution.
