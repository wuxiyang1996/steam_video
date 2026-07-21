#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
VENV_ROOT="${VENV_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/fs/gamma-projects/vlm-robot/hf_cache}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets/CG-Bench}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
GRAPH_ROOT="${GRAPH_ROOT:-${ARTIFACT_ROOT}/l15_graph_smoke_v1}"
SELECTION="${SELECTION:-${ARTIFACT_ROOT}/l15_graph_smoke_selection.json}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
PORT="${PORT:-$((21000 + (${SLURM_JOB_ID:-0} % 1000)))}"

export HF_HOME="${HF_CACHE_ROOT}"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/hub"
export TOKENIZERS_PARALLELISM=false
export SETUPTOOLS_USE_DISTUTILS=stdlib

mkdir -p "${GRAPH_ROOT}"
cd "${REPO_ROOT}"
"${VENV_ROOT}/bin/transformers" serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  --device cuda:0 --dtype bfloat16 --reasoning off --attn-implementation sdpa \
  >"${GRAPH_ROOT}/qwen_l15_server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    tail -100 "${GRAPH_ROOT}/qwen_l15_server.log" >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

"${VENV_ROOT}/bin/python" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker extract \
  --selection "${SELECTION}" \
  --dataset-root "${DATASET_ROOT}" \
  --video-skills-root "/fs/gamma-projects/vlm-robot/Video_Skills" \
  --output-root "${GRAPH_ROOT}" \
  --report "${GRAPH_ROOT}/build_report.json" \
  --model "${MODEL}" \
  --api-base "http://127.0.0.1:${PORT}/v1/chat/completions"

cleanup
trap - EXIT INT TERM

"${VENV_ROOT}/bin/python" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker embed-evaluate \
  --selection "${SELECTION}" \
  --graph-root "${GRAPH_ROOT}" \
  --dataset "${ARTIFACT_ROOT}/navigation_dataset.gt_only.json" \
  --hidden-input "${ARTIFACT_ROOT}/terminal_targets.hidden_key.json" \
  --coverage-report "${GRAPH_ROOT}/coverage_report.json" \
  --coverage-details "${GRAPH_ROOT}/coverage_details.hidden_key.json" \
  --device cuda
