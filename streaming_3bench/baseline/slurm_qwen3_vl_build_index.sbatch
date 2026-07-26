#!/bin/bash
#SBATCH --job-name=qwen3vl-index
#SBATCH --partition=prod_long
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=48:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_index-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/qwen3vl_index-%j.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
CANONICAL_JSONL=${CANONICAL_JSONL:-$PROJECT/outputs/atomic_skills_for_video/canonical_merged/full_3bench_fix2_20260708_193641.canonical_schemas.jsonl}
OUT_DIR=${OUT_DIR:-$PROJECT/outputs/atomic_skills_for_video/baseline_faiss/full_3bench_qwen3_vl_embedding_4frames}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
QWEN3_VL_MODEL=${QWEN3_VL_MODEL:-Qwen/Qwen3-VL-Embedding-2B}
FRAMES_PER_CLIP=${FRAMES_PER_CLIP:-4}
# Keep decode/embed batches small: previous OOM was host RAM from decoding a full shard at once.
IMAGE_BATCH_SIZE=${IMAGE_BATCH_SIZE:-8}
DECODE_WORKERS=${DECODE_WORKERS:-2}
DECODE_STRATEGY=${DECODE_STRATEGY:-scan}
ENCODE_CHUNK_SIZE=${ENCODE_CHUNK_SIZE:-64}
DEVICE=${DEVICE:-cuda:0}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
SHARD_OUT_DIR="$OUT_DIR/shards/shard_${SHARD_INDEX}"
mkdir -p "$SHARD_OUT_DIR"

"$VENV/bin/python" -m baseline.build_faiss_index \
  --canonical-jsonl "$CANONICAL_JSONL" \
  --output-dir "$SHARD_OUT_DIR" \
  --embedding-backend qwen3_vl \
  --qwen3-vl-model "$QWEN3_VL_MODEL" \
  --qwen3-vl-dtype bfloat16 \
  --device "$DEVICE" \
  --frames-per-clip "$FRAMES_PER_CLIP" \
  --image-batch-size "$IMAGE_BATCH_SIZE" \
  --decode-workers "$DECODE_WORKERS" \
  --decode-strategy "$DECODE_STRATEGY" \
  --encode-chunk-size "$ENCODE_CHUNK_SIZE" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --no-store-embeddings
