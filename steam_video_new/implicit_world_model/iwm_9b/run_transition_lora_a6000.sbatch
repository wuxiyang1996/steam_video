#!/usr/bin/env bash
#SBATCH --job-name=iwm9b-lora-v1
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video_runs/slurm-%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/steam_video_runs/slurm-%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
PYTHON_BIN="${PYTHON_BIN:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python}"
RECORDS="${RECORDS:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/iwm_grounded_transition_v1/records.jsonl}"
RUN_ROOT="${RUN_ROOT:-/fs/gamma-projects/vlm-robot/steam_video_runs/iwm_9b_transition_lora_a6000_v1}"
MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
HF_HOME="${HF_HOME:-/fs/gamma-projects/vlm-robot/Multi-hop-Reasoning-VLM-Agent/.hf_cache}"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

mkdir -p "${RUN_ROOT}/adapter" "${RUN_ROOT}/evaluation"
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

"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.train_sft \
  --records "${RECORDS}" \
  --task iwm_transition \
  --model "${MODEL}" \
  --output-dir "${RUN_ROOT}/adapter" \
  --max-length 1792 \
  --epochs 1 \
  --learning-rate 2e-4 \
  --batch-size "${TRAIN_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"

set +e
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.iwm_9b.predict_validation \
  --records "${RECORDS}" \
  --model "${MODEL}" \
  --adapter "${RUN_ROOT}/adapter" \
  --split validation \
  --output "${RUN_ROOT}/evaluation/lora_predictions.jsonl" \
  --report "${RUN_ROOT}/evaluation/lora_inference_report.json" \
  --max-input-length 1024 \
  --max-new-tokens 1200 \
  --batch-size "${INFERENCE_BATCH_SIZE}"
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
  'import json,sys; print(json.dumps({"baseline_exit": int(sys.argv[1]), "inference_exit": int(sys.argv[2]), "calibration_exit": int(sys.argv[3]), "training_performed": True}, indent=2))' \
  "${BASELINE_STATUS}" "${INFERENCE_STATUS}" "${CALIBRATION_STATUS}" \
  > "${RUN_ROOT}/pilot_status.json"
