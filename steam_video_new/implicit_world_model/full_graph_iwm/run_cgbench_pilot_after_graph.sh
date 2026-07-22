#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
PYTHON_BIN="${PYTHON_BIN:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python}"
KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
GRAPH_ROOT="${GRAPH_ROOT:-${ARTIFACT_ROOT}/l15_graph_smoke_v1}"
GRAPH_CAPACITY="${GRAPH_CAPACITY:-64}"
GRAPH_READ_BUDGET="${GRAPH_READ_BUDGET:-8}"

cd "${REPO_ROOT}"
COMMON_ARGS=(
  --dataset "${ARTIFACT_ROOT}/navigation_dataset.qwen_grounded_embedded.json"
  --hidden-key "${ARTIFACT_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json"
  --selection "${ARTIFACT_ROOT}/l15_graph_smoke_selection.json"
  --graph-root "${GRAPH_ROOT}"
  --capacity "${GRAPH_CAPACITY}"
  --graph-read-budget "${GRAPH_READ_BUDGET}"
  --video-limit 8
)

"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot \
  --mode compile-gate "${COMMON_ARGS[@]}" \
  --output "${GRAPH_ROOT}/full_graph_iwm_gate_8v.json"

"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.full_graph_iwm.cgbench_pilot \
  --mode gpt-oss-120b "${COMMON_ARGS[@]}" \
  --keys-py "${KEYS_PY}" \
  --rollout-horizon 1 \
  --max-trajectory-pairs 4096 \
  --output "${GRAPH_ROOT}/full_graph_iwm_horizon1_pilot.json"
