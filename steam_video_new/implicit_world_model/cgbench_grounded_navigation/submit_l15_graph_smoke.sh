#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
DEPENDENCY_JOB_ID="${DEPENDENCY_JOB_ID:-}"
mkdir -p "${ARTIFACT_ROOT}/slurm_logs"

DEPENDENCY_ARGS=()
if [[ -n "${DEPENDENCY_JOB_ID}" ]]; then
  DEPENDENCY_ARGS+=(--dependency="afterok:${DEPENDENCY_JOB_ID}")
fi

sbatch --parsable "${DEPENDENCY_ARGS[@]}" \
  --job-name=cg-l15-graph-smoke \
  --partition=gamma --account=gamma --qos=default \
  --gres=gpu:rtxa6000:1 --cpus-per-task=4 --mem=32G --time=06:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-smoke-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-smoke-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh"
