#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"
VIDEO_LIMIT="${VIDEO_LIMIT:-8}"
GPU_TYPE="${GPU_TYPE:-rtxa6000}"
mkdir -p "${ARTIFACT_ROOT}/slurm_logs"

if [[ "${NUM_SHARDS}" -lt 1 ]]; then
  echo "NUM_SHARDS must be positive" >&2
  exit 2
fi

DEPENDENCY_ARGS=()
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  DEPENDENCY_ARGS+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
fi

EXTRACT_JOB_ID="$(sbatch --parsable "${DEPENDENCY_ARGS[@]}" \
  --job-name=cg-l15-extract \
  --partition=gamma --account=gamma --qos=default \
  --array="0-$((NUM_SHARDS - 1))%${NUM_SHARDS}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-extract-%A_%a.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-extract-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",RUN_STAGE=extract,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${VIDEO_LIMIT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EXTRACT_JOB_ID}" \
  --job-name=cg-l15-finalize \
  --partition=gamma --account=gamma --qos=default \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task=4 --mem=32G --time=02:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-finalize-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-finalize-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",RUN_STAGE=finalize,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${VIDEO_LIMIT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

printf 'extract_array_job_id=%s\n' "${EXTRACT_JOB_ID}"
printf 'finalize_job_id=%s\n' "${FINALIZE_JOB_ID}"
