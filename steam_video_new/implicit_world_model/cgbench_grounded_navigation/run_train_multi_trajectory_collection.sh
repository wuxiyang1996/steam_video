#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
PYTHON_BIN="${PYTHON_BIN:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python}"
KEYS_PY="${KEYS_PY:-/fs/gamma-projects/vlm-robot/keys.py}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/iwm_runtime_train_collection_v1}"
GRAPH_ROOT="${GRAPH_ROOT:-${COLLECTION_ROOT}/l15_fixed_train_v1_vllm}"
SELECTION="${SELECTION:-${COLLECTION_ROOT}/l15_train_selection.json}"
PROTOCOL="${PROTOCOL:-${COLLECTION_ROOT}/l15_train_protocol.json}"
COMPILED_GATE="${COMPILED_GATE:-${GRAPH_ROOT}/multi_trajectory_compile_gate.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${COLLECTION_ROOT}/multi_trajectory_v1/run}"
MODEL="${MODEL:-openai/gpt-5-mini}"
CAPACITY="${CAPACITY:-192}"
READ_BUDGET="${READ_BUDGET:-2}"
WORKERS="${WORKERS:-8}"
SKIP_COMPILE_GATE="${SKIP_COMPILE_GATE:-0}"
FORCE="${FORCE:-0}"
CASE_IDS_CSV="${CASE_IDS_CSV:-}"

cd "${REPO_ROOT}"
if [[ "${SKIP_COMPILE_GATE}" == "1" ]]; then
  mapfile -t RUNNABLE_CASE_IDS < <("${PYTHON_BIN}" -c \
    'import json,sys; print("\n".join(json.load(open(sys.argv[1]))["runnable_case_ids"]))' \
    "${COMPILED_GATE}")
else
  mapfile -t RUNNABLE_CASE_IDS < <("${PYTHON_BIN}" -c \
    'import json,sys; print("\n".join(x["case_id"] for x in json.load(open(sys.argv[1]))["cases"]))' \
    "${PROTOCOL}")
fi
CASE_IDS=("${RUNNABLE_CASE_IDS[@]}")
if [[ -n "${CASE_IDS_CSV}" ]]; then
  declare -A RUNNABLE_CASE_SET=()
  for case_id in "${RUNNABLE_CASE_IDS[@]}"; do
    RUNNABLE_CASE_SET["${case_id}"]=1
  done
  IFS=',' read -r -a CASE_IDS <<< "${CASE_IDS_CSV}"
  for case_id in "${CASE_IDS[@]}"; do
    if [[ -z "${case_id}" || -z "${RUNNABLE_CASE_SET[${case_id}]:-}" ]]; then
      echo "Requested retry case is not in the frozen runnable gate: ${case_id}" >&2
      exit 2
    fi
  done
fi
if [[ "${#CASE_IDS[@]}" -lt 1 ]]; then
  echo "No runnable frozen train cases were selected" >&2
  exit 2
fi
CASE_ARGS=()
for case_id in "${CASE_IDS[@]}"; do
  CASE_ARGS+=(--case-id "${case_id}")
done
FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS+=(--force)
fi

if [[ "${SKIP_COMPILE_GATE}" != "1" ]]; then
  "${PYTHON_BIN}" -m steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cgbench \
    --mode compile-gate \
    --dataset "${SOURCE_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
    --hidden-key "${SOURCE_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json" \
    --selection "${SELECTION}" \
    --graph-root "${GRAPH_ROOT}" \
    --capacity "${CAPACITY}" \
    --graph-read-budget "${READ_BUDGET}" \
    --video-limit 40 \
    --cases-per-video 1 \
    --disable-caption-candidates \
    "${CASE_ARGS[@]}" \
    --output "${COMPILED_GATE}"
fi

"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.full_graph_iwm.multi_trajectory_cohort \
  --repo-root "${REPO_ROOT}" \
  --dataset "${SOURCE_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
  --hidden-key "${SOURCE_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json" \
  --selection "${SELECTION}" \
  --graph-root "${GRAPH_ROOT}" \
  --compiled-gate "${COMPILED_GATE}" \
  --keys-py "${KEYS_PY}" \
  --output-dir "${OUTPUT_DIR}" \
  --model "${MODEL}" \
  --workers "${WORKERS}" \
  --capacity "${CAPACITY}" \
  --read-budget "${READ_BUDGET}" \
  --transition-batch-size 1 \
  --comparison-batch-size 12 \
  --max-complete-pairs 4096 \
  --timeout-s 180 \
  --max-tokens 8000 \
  --reasoning-effort low \
  --disable-caption-candidates \
  "${CASE_ARGS[@]}" \
  "${FORCE_ARGS[@]}"
