# Grounded local-choice diagnostic

This directory records the first scoreless local-anchor supervision packet
from the capacity-256 CG-Bench delayed-case pilot.

- `blinded_local_choices.json` contains the question, initial categorical
  belief, anonymous anchor predictions, and no labels.
- `grounded_labels.hidden_key.json` contains post-planning GT clue overlap,
  complete pairwise categorical labels, and the executed navigation-delta
  correction.

The packet contains one test-split case, three anchors, three pair comparisons
(two strict preferences and one incomparable), and one transition-prediction
navigation mismatch. It is diagnostic only:
`training_ready=false` and `training_performed=false`.

No scalar reward, Top-K selection, manual role-count preference, or
outside-clue standalone negative is created. A non-overlapping endpoint is
`not_established`; CG-Bench does not make it a general semantic negative.

Regenerate:

```bash
cd /fs/gamma-projects/vlm-robot/steam_video

python -m steam_video_new.implicit_world_model.full_graph_iwm.local_choice_data \
  --run steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/l15_fixed_cohort_v2_vllm_c8/iwm_delayed1_localnav_qwen36flash_v4_replay.json \
  --dataset steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/navigation_dataset.gt_only.json \
  --hidden-key steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/terminal_targets.hidden_key.json \
  --graph-root steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/l15_fixed_cohort_v2_vllm_c8 \
  --transition-cache steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/l15_fixed_cohort_v2_vllm_c8/iwm_delayed1_localnav_qwen36flash.transition_cache.json \
  --output steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/blinded_local_choices.json \
  --hidden-output steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2/iwm_local_choice_v1/grounded_labels.hidden_key.json
```
