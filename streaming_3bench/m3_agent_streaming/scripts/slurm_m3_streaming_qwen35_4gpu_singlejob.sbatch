#!/bin/bash
#SBATCH --job-name=m3-qwen35-4gpu
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/m3-agent/logs/m3_qwen35_4gpu-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/m3-agent/logs/m3_qwen35_4gpu-%j.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/m3-agent}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MODEL=${MODEL:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/m3-agent/qwen35_streaming_eval_4gpu_singlejob/${SLURM_JOB_ID}}
ANSWER_MODE=${ANSWER_MODE:-json_rationale}
DATASETS=${DATASETS:-ovo_bench videomme}
LIMIT_PER_DATASET=${LIMIT_PER_DATASET:--1}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-3}
RETRIEVAL_STRATEGY=${RETRIEVAL_STRATEGY:-latest}
NUM_WORKERS=${NUM_WORKERS:-4}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

cd "$REPO"
mkdir -p "$OUT_ROOT" "$PROJECT/outputs/m3-agent/logs"

if [ "$NUM_WORKERS" -gt 4 ]; then
  echo "NUM_WORKERS=$NUM_WORKERS exceeds this script's 4-GPU allocation" >&2
  exit 2
fi

echo "Starting $NUM_WORKERS single-GPU shards inside one 4-GPU Slurm allocation"
echo "SLURM_JOB_ID=$SLURM_JOB_ID CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

pids=()
for SHARD_INDEX in $(seq 0 $((NUM_WORKERS - 1))); do
  (
    export CUDA_VISIBLE_DEVICES="$SHARD_INDEX"
    "$VENV/bin/python" -m m3_agent.streaming_eval \
      --model "$MODEL" \
      --atomic-repo-root "$ATOMIC_REPO" \
      --dataset-root "$PROJECT/data/datasets" \
      --output-dir "$OUT_ROOT/shard_${SHARD_INDEX}" \
      --datasets $DATASETS \
      --limit-per-dataset "$LIMIT_PER_DATASET" \
      --shard-index "$SHARD_INDEX" \
      --num-shards "$NUM_WORKERS" \
      --input-mode video_clip \
      --retrieval-topk "$RETRIEVAL_TOPK" \
      --retrieval-strategy "$RETRIEVAL_STRATEGY" \
      --video-fps 2.0 \
      --video-max-frames-per-clip 8 \
      --frame-workers 8 \
      --max-new-tokens 160 \
      --answer-mode "$ANSWER_MODE"
  ) > "$OUT_ROOT/shard_${SHARD_INDEX}.out" 2> "$OUT_ROOT/shard_${SHARD_INDEX}.err" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done

if [ "$failed" -ne 0 ]; then
  echo "At least one shard failed; inspect $OUT_ROOT/shard_*.err" >&2
  exit 1
fi

"$VENV/bin/python" -m m3_agent.merge_streaming_eval_shards \
  --shards-root "$OUT_ROOT" \
  --output-dir "${OUT_ROOT}_merged"

echo "Merged metrics: ${OUT_ROOT}_merged/metrics_summary.json"
