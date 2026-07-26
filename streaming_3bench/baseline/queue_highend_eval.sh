#!/usr/bin/env bash
# Queue the same real sharded eval on multiple high-end GPU types.
#
# Slurm will start whichever constrained array becomes schedulable first. Cancel
# duplicate queued/running arrays after one finishes if only one result set is
# needed.

set -euo pipefail

NUM_SHARDS=${NUM_SHARDS:-4}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}
ANSWER_MODE=${ANSWER_MODE:-json_rationale}

for gpu_model in GPUMODEL_H200-SXM5 GPUMODEL_H100-SXM5 GPUMODEL_A100-SXM4; do
  job_id=$(ANSWER_MODE="$ANSWER_MODE" baseline/submit_sharded_eval.sh "$gpu_model" "$NUM_SHARDS" "$LIMIT_PER_DATASET")
  printf '%s\t%s\n' "$gpu_model" "$job_id"
done
