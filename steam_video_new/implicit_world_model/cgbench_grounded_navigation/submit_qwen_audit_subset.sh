#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
SOURCE_ROOT="${SOURCE_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_grounded_navigation_pilot_v1}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SOURCE_ROOT}/qwen_audit_subset_v1}"
REVIEW_PACKET="${REVIEW_PACKET:-${SOURCE_ROOT}/blinded_visual_audit.unreviewed.json}"
PARTITION="${PARTITION:-scavenger}"
ACCOUNT="${ACCOUNT:-scavenger}"
QOS="${QOS-scavenger}"
GRES="${GRES:-gpu:a100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-96G}"
mkdir -p "${ARTIFACT_ROOT}/slurm_logs"

QOS_ARGS=()
if [[ -n "${QOS}" ]]; then QOS_ARGS+=(--qos="${QOS}"); fi
sbatch --parsable --job-name=cg-qwen-audit-subset \
  --partition="${PARTITION}" --account="${ACCOUNT}" "${QOS_ARGS[@]}" \
  --gres="${GRES}" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY}" --time=04:00:00 \
  --output="${ARTIFACT_ROOT}/slurm_logs/ground-audit-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/ground-audit-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",SOURCE_ROOT="${SOURCE_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",REVIEW_PACKET="${REVIEW_PACKET}",CASE_LIMIT= \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_qwen_grounding_job.sh"
