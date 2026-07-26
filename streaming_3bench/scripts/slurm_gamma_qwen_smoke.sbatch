#!/bin/bash
#SBATCH --job-name=qwen35-3bench-smoke
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/slurm_logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/fs/gamma-projects/vlm-robot
EVAL_ROOT=$ROOT/steam_video-streaming-3bench/streaming_3bench
MODEL=$ROOT/hf_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a

source "$EVAL_ROOT/scripts/activate_gamma_eval.sh"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi -L
python - <<'PY'
import faiss
import flash_attn
import torch
import transformers
import vllm

print(
    f"torch={torch.__version__} cuda={torch.version.cuda} "
    f"transformers={transformers.__version__} flash_attn={flash_attn.__version__} "
    f"vllm={vllm.__version__} faiss={faiss.__version__}"
)
PY

python -m baseline.qwen35_streaming_eval \
  --model "$MODEL" \
  --repo-root "$EVAL_ROOT" \
  --dataset-root "$VIDEO_SKILLS_DATASET_ROOT" \
  --output-dir "$VIDEO_SKILLS_OUTPUT_ROOT/qwen35_smoke/$SLURM_JOB_ID" \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 1 \
  --input-mode video_clip \
  --video-fps 1.0 \
  --video-max-frames-per-clip 4 \
  --max-new-tokens 32 \
  --answer-mode label_only
