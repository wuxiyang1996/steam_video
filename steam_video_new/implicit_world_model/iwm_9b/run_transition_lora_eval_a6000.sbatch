#!/usr/bin/env bash
#SBATCH --job-name=iwm9b-lora-eval
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/slurm-%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
PYTHON_BIN="${PYTHON_BIN:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python}"
RECORDS="${RECORDS:?RECORDS is required}"
ADAPTER="${ADAPTER:?ADAPTER is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
BATCH_SIZE="${BATCH_SIZE:-1}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_ROOT}/evaluation"
cd "${REPO_ROOT}"

nvidia-smi
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.categorical_baseline \
  --records "${RECORDS}" \
  --output "${RUN_ROOT}/evaluation/majority_predictions.jsonl"

set +e
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.calibration_gate \
  --records "${RECORDS}" \
  --predictions "${RUN_ROOT}/evaluation/majority_predictions.jsonl" \
  --output "${RUN_ROOT}/evaluation/majority_calibration.json"
BASELINE_STATUS=$?
set -e
if [[ "${BASELINE_STATUS}" -gt 2 ]]; then
  exit "${BASELINE_STATUS}"
fi

set +e
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.predict_validation \
  --records "${RECORDS}" \
  --model "${MODEL}" \
  --adapter "${ADAPTER}" \
  --split validation \
  --output "${RUN_ROOT}/evaluation/lora_predictions.jsonl" \
  --report "${RUN_ROOT}/evaluation/lora_inference_report.json" \
  --max-input-length 1024 \
  --max-new-tokens 1200 \
  --batch-size "${BATCH_SIZE}"
INFERENCE_STATUS=$?
if [[ "${INFERENCE_STATUS}" -gt 2 ]]; then
  exit "${INFERENCE_STATUS}"
fi
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.calibration_gate \
  --records "${RECORDS}" \
  --predictions "${RUN_ROOT}/evaluation/lora_predictions.jsonl" \
  --output "${RUN_ROOT}/evaluation/lora_calibration.json"
CALIBRATION_STATUS=$?
set -e
if [[ "${CALIBRATION_STATUS}" -gt 2 ]]; then
  exit "${CALIBRATION_STATUS}"
fi

"${PYTHON_BIN}" -c \
  'import json,sys; print(json.dumps({"baseline_exit": int(sys.argv[1]), "inference_exit": int(sys.argv[2]), "calibration_exit": int(sys.argv[3]), "adapter_reused": True, "training_performed": False}, indent=2))' \
  "${BASELINE_STATUS}" "${INFERENCE_STATUS}" "${CALIBRATION_STATUS}" \
  > "${RUN_ROOT}/pilot_status.json"
