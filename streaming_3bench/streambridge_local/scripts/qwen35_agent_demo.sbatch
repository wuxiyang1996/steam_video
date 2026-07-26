#!/bin/bash
#SBATCH --job-name=streambridge-qwen35-agent
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_agent-%j.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_agent-%j.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/ml-streambridge}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
QWEN35_CKPT=${QWEN35_CKPT:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
VIDEO_PATH=${VIDEO_PATH:-$REPO/assets/__an5-LdjFY.mp4}
ACTIVATE_PRETRAINED=${ACTIVATE_PRETRAINED:-/mnt/haibo/streambridge-opensource/llava-onevision-qwen2-0.5b-ov-hf-seperated}
ACTIVATE_CKPT=${ACTIVATE_CKPT:-/mnt/haibo/streambridge-opensource/activation_0.5_ratio_anet_coin_yc2_s2s_fa_mhego_hacs_cha_et_llava-ov_epoch_5.pth}
QUESTION=${QUESTION:-Recognize and highlight specific sequences of actions.}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

mkdir -p "$PROJECT/outputs/ml-streambridge/logs"
cd "$REPO"

"$VENV/bin/python" demo_qwen35_agent.py \
  --video-path "$VIDEO_PATH" \
  --qwen35-ckpt "$QWEN35_CKPT" \
  --activate-pretrained "$ACTIVATE_PRETRAINED" \
  --activate-ckpt "$ACTIVATE_CKPT" \
  --question "$QUESTION" \
  --clip-window-s "${CLIP_WINDOW_S:-4.0}" \
  --video-fps "${VIDEO_FPS:-2.0}" \
  --video-max-frames-per-clip "${VIDEO_MAX_FRAMES_PER_CLIP:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-128}"
