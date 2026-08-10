#!/bin/bash
#SBATCH --job-name=qf-3b-feat
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%A_%a.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01/logs/%x-%A_%a.err

set -euo pipefail

PROJECT=/fs/gamma-projects/vlm-robot/steam_video
RUN=/fs/gamma-projects/vlm-robot/steam_video_runs/qformer_threebench_v01
PYTHON=/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python
SOURCE=/gammascratch/wuxiyang/streaming_3bench_outputs/caption_rag_2b_full
DATASETS=(ovo videomme streaming)
DATASET=${DATASETS[$SLURM_ARRAY_TASK_ID]}

export PYTHONPATH=$PROJECT
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

cd "$PROJECT"
"$PYTHON" -m steam_video_new.implicit_world_model.reasoning_v2.qformer.materialize_threebench \
  --clips "$SOURCE/$DATASET/index_merged/clips.jsonl" \
  --output-dir "$RUN/features/$DATASET" \
  --report "$RUN/features/$DATASET/report.json" \
  --device cuda \
  --batch-size 8 \
  --frames-per-node 3
