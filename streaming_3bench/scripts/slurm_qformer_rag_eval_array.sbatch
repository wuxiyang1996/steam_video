#!/bin/bash
#SBATCH --job-name=qf-rag-3bench
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --array=0-71%12
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%A_%a.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%A_%a.err

set -euo pipefail

ROOT=/fs/gamma-projects/vlm-robot
PROJECT=$ROOT/steam_video-streaming-3bench/streaming_3bench
STEAM=$ROOT/steam_video
RUN=$ROOT/steam_video_runs/qformer_threebench_v01
EMBED=$ROOT/hf_cache/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
REASONER=$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
QF2=$ROOT/steam_video_runs/qformer_combined_v09/experiments/7230385_1/preinit_hard4_seed17/best.pt
SHARDS=8

ARMS=(uniform visual qformer)
DATASETS=(ovo_bench videomme streaming_bench)
ARM_INDEX=$((SLURM_ARRAY_TASK_ID / 24))
REMAINDER=$((SLURM_ARRAY_TASK_ID % 24))
DATASET_INDEX=$((REMAINDER / SHARDS))
SHARD_INDEX=$((REMAINDER % SHARDS))
ARM=${ARMS[$ARM_INDEX]}
DATASET=${DATASETS[$DATASET_INDEX]}
OUTPUT=$RUN/eval/$ARM/$DATASET/shard_$SHARD_INDEX

source "$PROJECT/scripts/activate_gamma_eval.sh"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
python -m baseline.qformer_rag_eval \
  --repo-root "$PROJECT" \
  --steam-root "$STEAM" \
  --dataset-root "$VIDEO_SKILLS_DATASET_ROOT" \
  --feature-manifest "$RUN/features/merged/manifest.json" \
  --projected-manifest "$RUN/projected_slots/manifest.json" \
  --clip-sidecar "$RUN/features/merged/clips.jsonl" \
  --qformer-checkpoint "$QF2" \
  --embedding-model "$EMBED" \
  --model "$REASONER" \
  --output-dir "$OUTPUT" \
  --datasets "$DATASET" \
  --arm "$ARM" \
  --split test \
  --limit-per-dataset -1 \
  --shard-index "$SHARD_INDEX" \
  --num-shards "$SHARDS" \
  --top-k 3 \
  --device cuda:0 \
  --video-fps 1.0 \
  --video-max-frames-per-clip 8 \
  --max-new-tokens 128

echo "QFORMER_RAG_EVAL_DONE arm=$ARM dataset=$DATASET shard=$SHARD_INDEX"
