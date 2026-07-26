#!/bin/bash
#SBATCH --job-name=qwen3vl-merge
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_merge-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_merge-%j.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
OUT_DIR=${OUT_DIR:-$PROJECT/outputs/atomic_skills_for_video/baseline_faiss/full_3bench_qwen3_vl_embedding_4frames}
NUM_SHARDS=${NUM_SHARDS:-4}

cd "$REPO"

SHARD_DIRS=()
for shard_index in $(seq 0 $((NUM_SHARDS - 1))); do
  SHARD_DIRS+=("$OUT_DIR/shards/shard_${shard_index}")
done

"$VENV/bin/python" -m baseline.merge_faiss_indexes \
  --shard-dirs "${SHARD_DIRS[@]}" \
  --output-dir "$OUT_DIR"
