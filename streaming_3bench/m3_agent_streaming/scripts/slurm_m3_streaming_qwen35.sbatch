#!/bin/bash
#SBATCH --job-name=m3-qwen35-stream
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/m3-agent/logs/m3_qwen35_stream-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/m3-agent/logs/m3_qwen35_stream-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/m3-agent}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MODEL=${MODEL:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/m3-agent/qwen35_streaming_eval_sharded/${SLURM_ARRAY_JOB_ID}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SLURM_ARRAY_TASK_ID:-0}
ANSWER_MODE=${ANSWER_MODE:-json_rationale}
DATASETS=${DATASETS:-ovo_bench videomme}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}
RETRIEVAL_STRATEGY=${RETRIEVAL_STRATEGY:-latest}
CANONICAL_JSONL=${CANONICAL_JSONL:-}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
mkdir -p "$OUT_ROOT" "$PROJECT/outputs/m3-agent/logs"

CANONICAL_ARGS=()
if [ -n "$CANONICAL_JSONL" ]; then
  CANONICAL_ARGS=(--canonical-jsonl "$CANONICAL_JSONL")
fi

"$VENV/bin/python" -m m3_agent.streaming_eval \
  --model "$MODEL" \
  --atomic-repo-root "$ATOMIC_REPO" \
  --dataset-root "$PROJECT/data/datasets" \
  "${CANONICAL_ARGS[@]}" \
  --output-dir "$OUT_ROOT/shard_${SHARD_INDEX}" \
  --datasets $DATASETS \
  --limit-per-dataset "$LIMIT_PER_DATASET" \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$NUM_SHARDS" \
  --input-mode video_clip \
  --retrieval-topk "$RETRIEVAL_TOPK" \
  --retrieval-strategy "$RETRIEVAL_STRATEGY" \
  --video-fps 2.0 \
  --video-max-frames-per-clip 8 \
  --frame-workers 8 \
  --max-new-tokens 160 \
  --answer-mode "$ANSWER_MODE"
