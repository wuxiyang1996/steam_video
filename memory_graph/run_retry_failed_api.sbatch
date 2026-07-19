#!/bin/bash
#SBATCH --job-name=mg-vh-retry
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
OUT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_50

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "$WORKSPACE/steam_video"
hostname
date -Is

"$PY" -u -m memory_graph.retry_failed_api \
  --output-dir "$OUT" \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --top-k 2 \
  --api-workers 2 \
  --max-retries 2 \
  --timeout-s 180

"$PY" -u -m memory_graph.evaluate_validation_outcomes \
  --summary "$OUT/summary.json"

date -Is
echo done
