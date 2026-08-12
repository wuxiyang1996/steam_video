#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
CASE_LIMIT="${CASE_LIMIT:-2}"
mkdir -p "${ARTIFACT_ROOT}/slurm_logs"

sbatch --parsable --job-name=cg-qwen-ground-smoke \
  --partition=scavenger --account=scavenger --qos=scavenger \
  --gres=gpu:a100:1 --cpus-per-task=8 --mem=96G --time=01:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/ground-smoke-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/ground-smoke-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",CASE_LIMIT="${CASE_LIMIT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_qwen_grounding_job.sh"
