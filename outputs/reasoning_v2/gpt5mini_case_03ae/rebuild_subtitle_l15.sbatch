#!/usr/bin/env bash
#SBATCH --job-name=subtitle-l15-rebuild
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/steam_video/outputs/reasoning_v2/gpt5mini_case_03ae/rebuild_subtitle_l15.log

set -euo pipefail

REPO_ROOT=/fs/gamma-projects/vlm-robot/steam_video
PYTHON_BIN=/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python
ARTIFACT_ROOT="${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2"
GRAPH_ROOT="${ARTIFACT_ROOT}/l15_fixed_cohort_v2_vllm_c8"
OUTPUT_ROOT="${REPO_ROOT}/outputs/reasoning_v2/gpt5mini_case_03ae"

export HF_HOME=/fs/gamma-projects/vlm-robot/hf_cache
export TRANSFORMERS_CACHE=/fs/gamma-projects/vlm-robot/hf_cache/hub
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.l15_graph_worker embed-evaluate \
  --selection "${OUTPUT_ROOT}/target_video_selection.json" \
  --graph-root "${GRAPH_ROOT}" \
  --dataset "${ARTIFACT_ROOT}/navigation_dataset.gt_only.json" \
  --hidden-input "${ARTIFACT_ROOT}/terminal_targets.hidden_key.json" \
  --coverage-report "${OUTPUT_ROOT}/subtitle_l1_coverage.json" \
  --coverage-details "${OUTPUT_ROOT}/subtitle_l1_coverage.hidden_key.json" \
  --memory-capacity 256 \
  --device cuda
