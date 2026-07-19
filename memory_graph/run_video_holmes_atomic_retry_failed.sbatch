#!/bin/bash
#SBATCH --job-name=mg-vh-atomic-retry
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

"$PY" -u -m memory_graph.validate_video_holmes \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --output-dir "$OUT" \
  --video-id=033fKtGdpPc \
  --video-id=0CPoD4ZRTUQ \
  --video-id=1IzpG2kJPdg \
  --video-id=1iHavvOahH0 \
  --video-id=21mDekTZwsw \
  --video-id=2REkk9SCRn0 \
  --device cuda:0 \
  --batch-size 4 \
  --top-k 2 \
  --input-mode expert_demo \
  --allow-provisional-expert-demo
