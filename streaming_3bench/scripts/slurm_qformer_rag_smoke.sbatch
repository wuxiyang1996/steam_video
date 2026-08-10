#!/bin/bash
#SBATCH --job-name=qf-rag-smoke
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.err

set -euo pipefail

ROOT=/fs/gamma-projects/vlm-robot
PROJECT=$ROOT/steam_video-streaming-3bench/streaming_3bench
RUN=$ROOT/steam_video_runs/qformer_threebench_v01
EMBED=$ROOT/hf_cache/hub/models--Qwen--Qwen3-VL-Embedding-2B/snapshots/9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda
REASONER=$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
QF2=$ROOT/steam_video_runs/qformer_combined_v09/experiments/7230385_1/preinit_hard4_seed17/best.pt

source "$PROJECT/scripts/activate_gamma_eval.sh"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
python -m baseline.qformer_rag_eval \
  --repo-root "$PROJECT" \
  --steam-root "$ROOT/steam_video" \
  --dataset-root "$VIDEO_SKILLS_DATASET_ROOT" \
  --feature-manifest "$RUN/features/merged/manifest.json" \
  --projected-manifest "$RUN/projected_slots/manifest.json" \
  --clip-sidecar "$RUN/features/merged/clips.jsonl" \
  --qformer-checkpoint "$QF2" \
  --embedding-model "$EMBED" \
  --model "$REASONER" \
  --output-dir "$RUN/smoke/qformer" \
  --datasets ovo_bench videomme streaming_bench \
  --arm qformer \
  --split test \
  --limit-per-dataset 2 \
  --top-k 3 \
  --device cuda:0 \
  --video-fps 1.0 \
  --video-max-frames-per-clip 8 \
  --max-new-tokens 128 \
  --fail-on-error
