#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"

cd "${REPO_ROOT}"
python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.validate_artifact \
  --input "${ARTIFACT_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
  --hidden-input "${ARTIFACT_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json" \
  --manifest "${ARTIFACT_ROOT}/qwen3_vl_2b_descriptor_embeddings.manifest.json" \
  --matrix "${ARTIFACT_ROOT}/qwen3_vl_2b_descriptor_embeddings.npy" \
  --report "${ARTIFACT_ROOT}/grounded_validation_report.json"

python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.modality_coverage \
  --input "${ARTIFACT_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
  --hidden-input "${ARTIFACT_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json" \
  --subtitle-root "/fs/gamma-projects/vlm-robot/datasets/CG-Bench/cg_subtitles/cg_subtitles" \
  --summary "${ARTIFACT_ROOT}/modality_coverage_summary.json" \
  --details "${ARTIFACT_ROOT}/modality_coverage_details.hidden_key.json"
