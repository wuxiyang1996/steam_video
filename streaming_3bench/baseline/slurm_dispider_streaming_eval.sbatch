#!/bin/bash
#SBATCH --job-name=dispider-stream
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/dispider-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/dispider-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/dispider_venv}
DISPIDER_REPO=${DISPIDER_REPO:-$PROJECT/code/Dispider}
MODEL=${MODEL:?Set MODEL to the Dispider checkpoint path}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/atomic_skills_for_video/dispider_streaming_eval/${SLURM_ARRAY_JOB_ID:-manual}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
DATASETS=${DATASETS:-ovo_bench videomme}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
mkdir -p "$OUT_ROOT" "$PROJECT/outputs/atomic_skills_for_video/logs"

"$VENV/bin/python" -m baseline.dispider_streaming_eval \
  --model "$MODEL" \
  --dispider-repo "$DISPIDER_REPO" \
  --repo-root "$REPO" \
  --dataset-root "$PROJECT/data/datasets" \
  --output-dir "$OUT_ROOT/shard_${SHARD_INDEX}" \
  --datasets $DATASETS \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS"
