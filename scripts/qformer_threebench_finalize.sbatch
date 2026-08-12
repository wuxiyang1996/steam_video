#!/bin/bash
#SBATCH --job-name=qf-3b-final
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%j.err

set -euo pipefail

PROJECT=/fs/gamma-projects/vlm-robot/steam_video
RUN=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01
PYTHON=/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python
QF1=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_combined_v09/7230384/qf1/qf1_best.pt

export PYTHONPATH=$PROJECT
cd "$PROJECT"
"$PYTHON" -m steam_video_new.implicit_world_model.reasoning_v2.qformer.merge_threebench \
  --inputs "$RUN/features/ovo" "$RUN/features/videomme" "$RUN/features/streaming" \
  --destination "$RUN/features/merged" \
  --report "$RUN/features/merged_report.json"
"$PYTHON" -m steam_video_new.implicit_world_model.reasoning_v2.qformer.export_qf1 \
  --checkpoint "$QF1" \
  --feature-manifest "$RUN/features/merged/manifest.json" \
  --destination "$RUN/projected_slots" \
  --projector-only \
  --device cpu \
  --batch-size 256
