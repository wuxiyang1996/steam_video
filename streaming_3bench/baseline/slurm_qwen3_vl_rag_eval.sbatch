#!/bin/bash
#SBATCH --job-name=qwen3vl-rag
#SBATCH --partition=prod_long
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_rag-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_rag-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
INDEX_DIR=${INDEX_DIR:-$PROJECT/outputs/atomic_skills_for_video/baseline_faiss/full_3bench_qwen3_vl_embedding_4frames}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/atomic_skills_for_video/iterative_rag_memory_query/full_3bench_qwen3_vl_embedding_4frames/${SLURM_ARRAY_JOB_ID}}
MODEL=${MODEL:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
QWEN3_VL_MODEL=${QWEN3_VL_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
DATASETS=${DATASETS:-ovo_bench videomme streaming_bench}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
# Dual-GPU pin: Embedding-2B on GPU0, Qwen3.5-9B on GPU1 (sequential per example).
EMBED_DEVICE=${EMBED_DEVICE:-cuda:0}
ANSWER_DEVICE=${ANSWER_DEVICE:-cuda:1}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
mkdir -p "$OUT_ROOT"

"$VENV/bin/python" -m baseline.iterative_rag_memory_query \
  --index-dir "$INDEX_DIR" \
  --output-dir "$OUT_ROOT/shard_${SHARD_INDEX}" \
  --datasets $DATASETS \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --iterations 3 \
  --per-iteration-top-k 4 \
  --final-top-k 8 \
  --pool-k 128 \
  --embedding-backend qwen3_vl \
  --qwen3-vl-model "$QWEN3_VL_MODEL" \
  --qwen3-vl-dtype bfloat16 \
  --embed-device "$EMBED_DEVICE" \
  --answer-device "$ANSWER_DEVICE" \
  --answer-backend local_qwen \
  --model "$MODEL" \
  --max-new-tokens 128
