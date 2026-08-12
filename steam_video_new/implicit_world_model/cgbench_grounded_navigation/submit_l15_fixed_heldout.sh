#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
GRAPH_ROOT="${GRAPH_ROOT:-${ARTIFACT_ROOT}/l15_fixed_heldout_v1}"
SELECTION="${SELECTION:-${ARTIFACT_ROOT}/l15_fixed_heldout_selection.json}"
NUM_SHARDS="${NUM_SHARDS:-2}"
GPU_TYPE="${GPU_TYPE:-rtxa6000}"
mkdir -p "${ARTIFACT_ROOT}/slurm_logs" "${GRAPH_ROOT}/shard_reports"

EXTRACT_JOB_ID="$(sbatch --parsable \
  --job-name=cg-l15-heldout \
  --partition=gamma --account=gamma --qos=default \
  --array="0-$((NUM_SHARDS - 1))%${NUM_SHARDS}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task=4 --mem=32G --time=12:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-heldout-%A_%a.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-heldout-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",RUN_STAGE=extract,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT=2 \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EXTRACT_JOB_ID}" \
  --job-name=cg-l15-heldout-final \
  --partition=gamma --account=gamma --qos=default \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task=4 --mem=32G --time=04:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-heldout-final-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-heldout-final-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",RUN_STAGE=finalize,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT=2 \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

printf 'extract_array_job_id=%s\n' "${EXTRACT_JOB_ID}"
printf 'finalize_job_id=%s\n' "${FINALIZE_JOB_ID}"
