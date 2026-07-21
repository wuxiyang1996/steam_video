#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_grounded_navigation_pilot_v1}"
QWEN_ROOT="${QWEN_ROOT:-${SOURCE_ROOT}/qwen_audit_subset_v1}"
PYTHON="${PYTHON:-/fs/gamma-projects/vlm-robot/conda/bin/python}"
cd "${REPO_ROOT}"

"${PYTHON}" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.audit_assets \
  --packet "${SOURCE_ROOT}/blinded_visual_audit.with_assets.unreviewed.json" \
  --video-root /fs/gamma-projects/vlm-robot/datasets/CG-Bench/cg_videos \
  --output-dir "${SOURCE_ROOT}/blinded_visual_assets" \
  --grounded-dataset "${QWEN_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
  --terminal-key "${SOURCE_ROOT}/terminal_targets.hidden_key.json" \
  --output-packet "${QWEN_ROOT}/blinded_visual_audit.with_assets.qwen_grounded.json" \
  --report "${QWEN_ROOT}/grounded_audit_packet_report.json"

"${PYTHON}" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.gpt56_audit \
  --packet "${QWEN_ROOT}/blinded_visual_audit.with_assets.qwen_grounded.json" \
  --hidden-key "${SOURCE_ROOT}/blinded_visual_audit.hidden_key.json" \
  --keys /fs/gamma-projects/vlm-robot/keys.py \
  --output "${QWEN_ROOT}/visual_audit.gpt56.provisional.json" \
  --inspection "${QWEN_ROOT}/visual_audit.gpt56.inspection.json" \
  --asset-root "${SOURCE_ROOT}" \
  --workers 4

"${PYTHON}" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.filter_ablation \
  --ablation "${SOURCE_ROOT}/matched_ablation.unexecuted.json" \
  --review-packet "${QWEN_ROOT}/blinded_visual_audit.with_assets.qwen_grounded.json" \
  --audit "${QWEN_ROOT}/visual_audit.gpt56.provisional.json" \
  --hidden-review-key "${SOURCE_ROOT}/blinded_visual_audit.hidden_key.json" \
  --output "${QWEN_ROOT}/matched_ablation.gpt56_filtered.provisional.json" \
  --report "${QWEN_ROOT}/matched_ablation.gpt56_filter_report.json"
