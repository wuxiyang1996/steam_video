#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
COLLECTION_ROOT="${COLLECTION_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/iwm_runtime_train_collection_v1}"
GRAPH_ROOT="${GRAPH_ROOT:-${COLLECTION_ROOT}/l15_fixed_train_v1_vllm}"
SELECTION="${SELECTION:-${COLLECTION_ROOT}/l15_train_selection.json}"
PROTOCOL="${PROTOCOL:-${COLLECTION_ROOT}/l15_train_protocol.json}"
NUM_SHARDS=40
MAX_PARALLEL_PER_GPU_TYPE="${MAX_PARALLEL_PER_GPU_TYPE:-4}"
QOS="${QOS:-gamma-huge-long}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEMORY_PER_TASK="${MEMORY_PER_TASK:-64G}"
EXTRACT_TIME_LIMIT="${EXTRACT_TIME_LIMIT:-1-00:00:00}"
FINALIZE_TIME_LIMIT="${FINALIZE_TIME_LIMIT:-08:00:00}"
COLLECTION_TIME_LIMIT="${COLLECTION_TIME_LIMIT:-3-00:00:00}"
SKIP_COLLECTION="${SKIP_COLLECTION:-0}"
VLLM_ROOT="${VLLM_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-vllm}"

mkdir -p "${COLLECTION_ROOT}/slurm_logs" "${GRAPH_ROOT}/shard_reports"
cd "${REPO_ROOT}"
ACTUAL_VIDEO_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["videos"]))' "${SELECTION}")"
if [[ "${ACTUAL_VIDEO_COUNT}" -ne "${NUM_SHARDS}" ]]; then
  echo "Expected ${NUM_SHARDS} question-independent videos, found ${ACTUAL_VIDEO_COUNT}" >&2
  exit 2
fi

COMMON_EXPORT="ALL,REPO_ROOT=${REPO_ROOT},ARTIFACT_ROOT=${SOURCE_ROOT},GRAPH_ROOT=${GRAPH_ROOT},SELECTION=${SELECTION},RUN_STAGE=extract,NUM_SHARDS=${NUM_SHARDS},VIDEO_LIMIT=${NUM_SHARDS},MEMORY_CAPACITY=192,L1_LOCALIZATION_MODE=grounded_single_pass,L1_REQUEST_CONCURRENCY=8,SERVER_BACKEND=vllm,VLLM_ROOT=${VLLM_ROOT},SURPRISE_SAMPLE_PERIOD_S=0.75,SURPRISE_MIN_WINDOW_S=4.0,SURPRISE_MAX_WINDOW_S=20.0,SURPRISE_CALIBRATION_HISTORY=12,SURPRISE_QUANTILE=0.85"

EVEN_JOB_ID="$(sbatch --parsable \
  --job-name=cg-train-l15-l40s \
  --partition=gamma --account=gamma --qos="${QOS}" \
  --array="0-38:2%${MAX_PARALLEL_PER_GPU_TYPE}" \
  --gres=gpu:l40s:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${EXTRACT_TIME_LIMIT}" \
  --output="${COLLECTION_ROOT}/slurm_logs/l15-train-%A_%a.out" \
  --error="${COLLECTION_ROOT}/slurm_logs/l15-train-%A_%a.err" \
  --export="${COMMON_EXPORT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

ODD_JOB_ID="$(sbatch --parsable \
  --job-name=cg-train-l15-a6k \
  --partition=gamma --account=gamma --qos="${QOS}" \
  --array="1-39:2%${MAX_PARALLEL_PER_GPU_TYPE}" \
  --gres=gpu:rtxa6000:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${EXTRACT_TIME_LIMIT}" \
  --output="${COLLECTION_ROOT}/slurm_logs/l15-train-%A_%a.out" \
  --error="${COLLECTION_ROOT}/slurm_logs/l15-train-%A_%a.err" \
  --export="${COMMON_EXPORT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EVEN_JOB_ID}:${ODD_JOB_ID}" \
  --job-name=cg-train-l15-final \
  --partition=gamma --account=gamma --qos="${QOS}" \
  --gres=gpu:rtxa6000:1 --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${FINALIZE_TIME_LIMIT}" \
  --output="${COLLECTION_ROOT}/slurm_logs/l15-train-final-%j.out" \
  --error="${COLLECTION_ROOT}/slurm_logs/l15-train-final-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${SOURCE_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",RUN_STAGE=finalize,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${NUM_SHARDS}",MEMORY_CAPACITY=192,COHORT_PROTOCOL="${PROTOCOL}",COHORT_GATE_OUTPUT="${GRAPH_ROOT}/train_clue_retention_gate.json",COHORT_GATE_DETAILS="${GRAPH_ROOT}/train_clue_retention_gate.hidden_key.json" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

COLLECTION_JOB_ID="skipped"
if [[ "${SKIP_COLLECTION}" != "1" ]]; then
  COLLECTION_JOB_ID="$(sbatch --parsable \
    --dependency="afterok:${FINALIZE_JOB_ID}" \
    --job-name=cg-train-iwm-collect \
    --partition=gamma --account=gamma --qos=default \
    --cpus-per-task=4 --mem=32G --time="${COLLECTION_TIME_LIMIT}" \
    --output="${COLLECTION_ROOT}/slurm_logs/multi-trajectory-%j.out" \
    --error="${COLLECTION_ROOT}/slurm_logs/multi-trajectory-%j.err" \
    --export=ALL,REPO_ROOT="${REPO_ROOT}",SOURCE_ROOT="${SOURCE_ROOT}",COLLECTION_ROOT="${COLLECTION_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",PROTOCOL="${PROTOCOL}" \
    "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_train_multi_trajectory_collection.sh")"
fi

printf 'l40s_extract_array_job_id=%s\n' "${EVEN_JOB_ID}"
printf 'a6000_extract_array_job_id=%s\n' "${ODD_JOB_ID}"
printf 'finalize_job_id=%s\n' "${FINALIZE_JOB_ID}"
printf 'multi_trajectory_collection_job_id=%s\n' "${COLLECTION_JOB_ID}"
printf 'graph_root=%s\n' "${GRAPH_ROOT}"
