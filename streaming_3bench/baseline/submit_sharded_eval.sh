#!/usr/bin/env bash
# Submit sharded Qwen streaming eval to a requested GPU model.
#
# This queues real evaluation work. It should not be used for idle placeholder
# reservations.

set -euo pipefail

GPU_MODEL=${1:-GPUMODEL_A6000}
NUM_SHARDS=${2:-4}
LIMIT_PER_DATASET=${3:--1}
ANSWER_MODE=${ANSWER_MODE:-json_rationale}
PARTITION=${PARTITION:-prod}
PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}

mkdir -p "$PROJECT/outputs/atomic_skills_for_video/logs"

cd "$REPO"

NUM_SHARDS="$NUM_SHARDS" \
ANSWER_MODE="$ANSWER_MODE" \
LIMIT_PER_DATASET="$LIMIT_PER_DATASET" \
sbatch --parsable \
  --partition="$PARTITION" \
  --constraint="$GPU_MODEL" \
  --array="0-$((NUM_SHARDS - 1))" \
  baseline/slurm_qwen35_sharded_eval.sbatch
