#!/bin/bash
#SBATCH --job-name=mg-vh-vs-l1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
L1_JSONL=$WORKSPACE/Video_Skills/dataset_clip_wrapper/output/sft_auto_20260713_full_retrieval/video_holmes/start_100_limit_25/examples.jsonl
OUT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_video_skills_l1_graph_smoke

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills:${PYTHONPATH:-}"
export HF_HOME="$WORKSPACE/Multi-hop-Reasoning-VLM-Agent/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
mkdir -p "$WORKSPACE/cluster/logs"

cd "$WORKSPACE/steam_video"
"$PY" -u -m memory_graph.validate_video_holmes \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --video-id TNYeYwYiAag \
  --input-mode video_only \
  --video-skills-l1-jsonl "$L1_JSONL" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --batch-size 4 \
  --top-k 2 \
  --memory-capacity 24
