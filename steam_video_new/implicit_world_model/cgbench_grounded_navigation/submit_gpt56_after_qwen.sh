#!/usr/bin/env bash
set -euo pipefail

QWEN_JOB_ID="${QWEN_JOB_ID:?set QWEN_JOB_ID to the active grounding job}"
REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_grounded_navigation_pilot_v1}"
QWEN_ROOT="${QWEN_ROOT:-${SOURCE_ROOT}/qwen_audit_subset_v1}"
mkdir -p "${QWEN_ROOT}/slurm_logs"

sbatch --parsable --job-name=cg-gpt56-audit \
  --dependency="afterok:${QWEN_JOB_ID}" --partition=scavenger --account=scavenger --qos=scavenger \
  --cpus-per-task=4 --mem=16G --time=02:00:00 \
  --output="${QWEN_ROOT}/slurm_logs/gpt56-audit-%j.out" \
  --error="${QWEN_ROOT}/slurm_logs/gpt56-audit-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",SOURCE_ROOT="${SOURCE_ROOT}",QWEN_ROOT="${QWEN_ROOT}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_gpt56_audit_after_qwen.sh"
