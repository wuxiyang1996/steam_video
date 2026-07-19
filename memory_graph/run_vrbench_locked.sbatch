#!/bin/bash
#SBATCH --job-name=mg-vrb-100
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
CANONICAL_DIR=${VRBENCH_CANONICAL_DIR:?Set VRBENCH_CANONICAL_DIR to locked video-only canonical outputs}
OUT=$WORKSPACE/steam_video/memory_graph/outputs/vrbench_locked_100.json

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills:${PYTHONPATH:-}"
mkdir -p "$(dirname "$OUT")" "$WORKSPACE/cluster/logs"
cd "$WORKSPACE/steam_video"

"$PY" -u -m memory_graph.validate_vrbench \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --output "$OUT" \
  --mode video_only \
  --phase locked \
  --canonical-dir "$CANONICAL_DIR"
