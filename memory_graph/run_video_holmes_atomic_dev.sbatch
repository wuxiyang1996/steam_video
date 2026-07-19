#!/bin/bash
#SBATCH --job-name=mg-vh-atomic10
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
OUT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_atomic_dev_10

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills:${PYTHONPATH:-}"
export HF_HOME="$WORKSPACE/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT" "$WORKSPACE/cluster/logs"
cd "$WORKSPACE/steam_video"

"$PY" -u -m memory_graph.validate_video_holmes_parallel \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --output-dir "$OUT" \
  --video-ids-json "$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_50/video_ids.json" \
  --limit 10 \
  --device cuda:0 \
  --batch-size 4 \
  --top-k 2 \
  --input-mode expert_demo \
  --allow-provisional-expert-demo
