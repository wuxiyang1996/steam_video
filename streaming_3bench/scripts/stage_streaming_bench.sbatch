#!/bin/bash
#SBATCH --job-name=stage-streamingbench
#SBATCH --partition=prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/stage_streamingbench-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs/stage_streamingbench-%j.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
DATASET_ROOT=${DATASET_ROOT:-$PROJECT/data/datasets}
STAGE_WORKERS=${STAGE_WORKERS:-4}
CLEANUP_ARCHIVES=${CLEANUP_ARCHIVES:-1}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}

mkdir -p "$PROJECT/outputs/atomic_skills_for_video/logs"
cd "$REPO"

"$VENV/bin/python" scripts/stage_streaming_bench.py \
  --dataset-root "$DATASET_ROOT" \
  --include-videos \
  --workers "$STAGE_WORKERS" \
  $([ "$CLEANUP_ARCHIVES" = "1" ] && echo "--cleanup-archives")
