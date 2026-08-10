#!/bin/bash
#SBATCH --job-name=qf-rag-stats
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.err

set -euo pipefail

ROOT=/fs/gamma-projects/vlm-robot
PROJECT=$ROOT/steam_video-streaming-3bench/streaming_3bench
RUN=$ROOT/steam_video_runs/qformer_threebench_v01

source "$PROJECT/scripts/activate_gamma_eval.sh"
cd "$PROJECT"
python -m baseline.qformer_rag_stats \
  --uniform-root "$RUN/eval/uniform" \
  --visual-root "$RUN/eval/visual" \
  --qformer-root "$RUN/eval/qformer" \
  --bootstrap-samples 10000 \
  --seed 20260810 \
  --output "$RUN/eval/paired_statistics.json"
