#!/bin/bash
#SBATCH --job-name=pervid-rag
#SBATCH --partition=prod_long
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/pervid_rag-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/pervid_rag-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MODEL=${MODEL:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/atomic_skills_for_video/per_video_embedding_rag/${SLURM_ARRAY_JOB_ID}}
QWEN3_VL_MODEL=${QWEN3_VL_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
EMBEDDING_BACKEND=${EMBEDDING_BACKEND:-qwen3_vl}
DATASETS=${DATASETS:-ovo_bench videomme streaming_bench}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
# Embedding retrieves top-k clips; answerer sees those video clips (M3-aligned).
TOP_K=${TOP_K:-3}
MAX_EMBED_CANDIDATES=${MAX_EMBED_CANDIDATES:-16}
CANDIDATE_STRATEGY=${CANDIDATE_STRATEGY:-uniform}
FRAMES_PER_CLIP=${FRAMES_PER_CLIP:-4}
CLIP_ENCODE_MODE=${CLIP_ENCODE_MODE:-video}
if [[ "$CLIP_ENCODE_MODE" == "video" ]]; then
  IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE:-4}
  ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-16}
else
  IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE:-32}
  ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-64}
fi
VIDEO_FPS=${VIDEO_FPS:-2.0}
VIDEO_MAX_FRAMES_PER_CLIP=${VIDEO_MAX_FRAMES_PER_CLIP:-8}
ANSWER_MODE=${ANSWER_MODE:-label_only}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
# Single-GPU: embedder + 9B answerer share cuda:0 (sequential per example).
EMBED_DEVICE=${EMBED_DEVICE:-cuda:0}
ANSWER_DEVICE=${ANSWER_DEVICE:-cuda:0}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
mkdir -p "$OUT_ROOT"

"$VENV/bin/python" -m baseline.per_video_embedding_rag \
  --repo-root "$REPO" \
  --dataset-root "$PROJECT/data/datasets" \
  --output-dir "$OUT_ROOT/shard_${SHARD_INDEX}" \
  --datasets $DATASETS \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --embedding-backend "$EMBEDDING_BACKEND" \
  --qwen3-vl-model "$QWEN3_VL_MODEL" \
  --qwen3-vl-dtype bfloat16 \
  --frames-per-clip "$FRAMES_PER_CLIP" \
  --clip-encode-mode "$CLIP_ENCODE_MODE" \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --decode-workers 2 \
  --decode-strategy scan \
  --encode-chunk-size "$ENCODE_CHUNK_SIZE" \
  --max-embed-candidates "$MAX_EMBED_CANDIDATES" \
  --candidate-strategy "$CANDIDATE_STRATEGY" \
  --top-k "$TOP_K" \
  --video-fps "$VIDEO_FPS" \
  --video-max-frames-per-clip "$VIDEO_MAX_FRAMES_PER_CLIP" \
  --answer-mode "$ANSWER_MODE" \
  --embed-device "$EMBED_DEVICE" \
  --answer-device "$ANSWER_DEVICE" \
  --model "$MODEL" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --progress-every 5
