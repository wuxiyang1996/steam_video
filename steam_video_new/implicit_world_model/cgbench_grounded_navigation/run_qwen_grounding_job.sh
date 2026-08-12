#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
VENV_ROOT="${VENV_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-/fs/gamma-projects/vlm-robot/hf_cache}"
VIDEO_ROOT="${VIDEO_ROOT:-/fs/gamma-projects/vlm-robot/datasets/CG-Bench/cg_videos}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
SOURCE_ROOT="${SOURCE_ROOT:-${ARTIFACT_ROOT}}"
SOURCE_DATASET_NAME="${SOURCE_DATASET_NAME:-navigation_dataset.gt_only.json}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
CASE_LIMIT="${CASE_LIMIT-2}"
REVIEW_PACKET="${REVIEW_PACKET:-}"
PORT="${PORT:-$((20000 + (${SLURM_JOB_ID:-0} % 1000)))}"

export HF_HOME="${HF_CACHE_ROOT}"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/hub"
export TOKENIZERS_PARALLELISM=false
export SETUPTOOLS_USE_DISTUTILS=stdlib

mkdir -p "${ARTIFACT_ROOT}"
SOURCE_DATASET="${SOURCE_ROOT}/${SOURCE_DATASET_NAME}"
SOURCE_KEY="${SOURCE_ROOT}/terminal_targets.hidden_key.json"
GROUNDED_DATASET="${ARTIFACT_ROOT}/navigation_dataset.qwen_grounded.json"
GROUNDED_KEY="${ARTIFACT_ROOT}/terminal_targets.qwen_grounded.hidden_key.json"
GROUNDING_REPORT="${ARTIFACT_ROOT}/qwen_grounding_report.json"
SERVER_LOG="${ARTIFACT_ROOT}/qwen_grounding_server.log"
GROUND_INPUT="${SOURCE_DATASET}"
GROUND_KEY_INPUT="${SOURCE_KEY}"
if [[ -f "${GROUNDED_DATASET}" && -f "${GROUNDED_KEY}" ]]; then
  GROUND_INPUT="${GROUNDED_DATASET}"
  GROUND_KEY_INPUT="${GROUNDED_KEY}"
fi

cd "${REPO_ROOT}"
"${VENV_ROOT}/bin/transformers" serve "${MODEL}" --host 127.0.0.1 --port "${PORT}" \
  --device cuda:0 --dtype bfloat16 --reasoning off --attn-implementation sdpa \
  >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then break; fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then tail -100 "${SERVER_LOG}" >&2; exit 1; fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

LIMIT_ARGS=()
if [[ -n "${CASE_LIMIT}" ]]; then LIMIT_ARGS+=(--case-limit "${CASE_LIMIT}"); fi
if [[ -n "${REVIEW_PACKET}" ]]; then LIMIT_ARGS+=(--review-packet "${REVIEW_PACKET}"); fi
"${VENV_ROOT}/bin/python" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.grounding ground \
  --input "${GROUND_INPUT}" --hidden-input "${GROUND_KEY_INPUT}" --video-root "${VIDEO_ROOT}" \
  --output "${GROUNDED_DATASET}" --hidden-output "${GROUNDED_KEY}" \
  --report "${GROUNDING_REPORT}" --endpoint "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --model "${MODEL}" --frames-per-interval 6 "${LIMIT_ARGS[@]}"

cleanup
trap - EXIT INT TERM

"${VENV_ROOT}/bin/python" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.grounding embed \
  --input "${GROUNDED_DATASET}" --hidden-input "${GROUNDED_KEY}" \
  --output "${ARTIFACT_ROOT}/navigation_dataset.qwen_grounded_embedded.json" \
  --hidden-output "${ARTIFACT_ROOT}/terminal_targets.qwen_grounded_embedded.hidden_key.json" \
  --matrix "${ARTIFACT_ROOT}/qwen3_vl_2b_descriptor_embeddings.npy" \
  --manifest "${ARTIFACT_ROOT}/qwen3_vl_2b_descriptor_embeddings.manifest.json" \
  --device cuda --batch-size 8
