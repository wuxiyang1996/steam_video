#!/bin/bash
# Run Week 0.5 multihop experiment on an existing Slurm allocation.
# Usage: srun --jobid=JOBID --overlap bash run_on_gpu.sh [encoder]
set -euo pipefail
ROOT=/fs/gamma-projects/vlm-robot/steam_video/dynamic_navigation/experiments/multihop
PY=/fs/gamma-projects/vlm-robot/conda/envs/swift/bin/python
ENC=${1:-qwen3-emb-2b}
export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TOKENIZERS_PARALLELISM=false
cd "$ROOT"
hostname
nvidia-smi -L
"$PY" -u train_eval.py \
  --encoder "$ENC" \
  --policy mlp \
  --n-train 512 \
  --n-val 256 \
  --epochs 5 \
  --batch-size 8 \
  --k-steps 4 \
  --n-tokens 16 \
  --out "outputs/week05_${ENC}.json"
