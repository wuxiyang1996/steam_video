#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets/CG-Bench}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/cgbench_gt_navigation_pilot_v2}"
GRAPH_ROOT="${GRAPH_ROOT:-${ARTIFACT_ROOT}/l15_fixed_cohort_v2_vllm_c8}"
SELECTION="${SELECTION:-${ARTIFACT_ROOT}/l15_fixed_cohort_selection.json}"
PROTOCOL="${PROTOCOL:-${ARTIFACT_ROOT}/l15_fixed_cohort_protocol.json}"
NUM_SHARDS="${NUM_SHARDS:-36}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
GPU_TYPE="${GPU_TYPE:-rtxa6000}"
QOS="${QOS:-gamma-huge-long}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY_PER_TASK="${MEMORY_PER_TASK:-64G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gammagpu13}"
EXTRACT_TIME_LIMIT="${EXTRACT_TIME_LIMIT:-12:00:00}"
FINALIZE_TIME_LIMIT="${FINALIZE_TIME_LIMIT:-04:00:00}"
MEMORY_CAPACITY="${MEMORY_CAPACITY:-192}"
VIDEO_LIMIT="${VIDEO_LIMIT:-36}"
L1_LOCALIZATION_MODE="${L1_LOCALIZATION_MODE:-grounded_single_pass}"
L1_REQUEST_CONCURRENCY="${L1_REQUEST_CONCURRENCY:-8}"
SERVER_BACKEND="${SERVER_BACKEND:-vllm}"
VLLM_ROOT="${VLLM_ROOT:-/fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-vllm}"
SURPRISE_SAMPLE_PERIOD_S="${SURPRISE_SAMPLE_PERIOD_S:-0.75}"
SURPRISE_MIN_WINDOW_S="${SURPRISE_MIN_WINDOW_S:-4.0}"
SURPRISE_MAX_WINDOW_S="${SURPRISE_MAX_WINDOW_S:-20.0}"
SURPRISE_CALIBRATION_HISTORY="${SURPRISE_CALIBRATION_HISTORY:-12}"
SURPRISE_QUANTILE="${SURPRISE_QUANTILE:-0.85}"

mkdir -p "${ARTIFACT_ROOT}/slurm_logs" "${GRAPH_ROOT}/shard_reports"
cd "${REPO_ROOT}"
python -m steam_video_new.implicit_world_model.cgbench_grounded_navigation.fixed_l15_cohort select \
  --dataset "${ARTIFACT_ROOT}/navigation_dataset.gt_only.json" \
  --manifest "${ARTIFACT_ROOT}/l15_graph_generation_manifest.json" \
  --dataset-root "${DATASET_ROOT}" \
  --selection-output "${SELECTION}" \
  --protocol-output "${PROTOCOL}" \
  --minimum-cases 30 \
  --maximum-cases 50

ACTUAL_VIDEO_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["videos"]))' "${SELECTION}")"
if [[ "${VIDEO_LIMIT}" -ne "${ACTUAL_VIDEO_COUNT}" ]]; then
  echo "VIDEO_LIMIT=${VIDEO_LIMIT} differs from frozen cohort video count ${ACTUAL_VIDEO_COUNT}" >&2
  exit 2
fi
if [[ "${NUM_SHARDS}" -ne "${ACTUAL_VIDEO_COUNT}" ]]; then
  echo "NUM_SHARDS must equal the video count so each checkpoint owns one video" >&2
  exit 2
fi

EXCLUDE_ARGS=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  EXCLUDE_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

EXTRACT_JOB_ID="$(sbatch --parsable \
  --job-name=cg-l15-cohort \
  --partition=gamma --account=gamma --qos="${QOS}" \
  "${EXCLUDE_ARGS[@]}" \
  --array="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${EXTRACT_TIME_LIMIT}" \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-cohort-%A_%a.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-cohort-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",RUN_STAGE=extract,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${VIDEO_LIMIT}",MEMORY_CAPACITY="${MEMORY_CAPACITY}",L1_LOCALIZATION_MODE="${L1_LOCALIZATION_MODE}",L1_REQUEST_CONCURRENCY="${L1_REQUEST_CONCURRENCY}",SERVER_BACKEND="${SERVER_BACKEND}",VLLM_ROOT="${VLLM_ROOT}",SURPRISE_SAMPLE_PERIOD_S="${SURPRISE_SAMPLE_PERIOD_S}",SURPRISE_MIN_WINDOW_S="${SURPRISE_MIN_WINDOW_S}",SURPRISE_MAX_WINDOW_S="${SURPRISE_MAX_WINDOW_S}",SURPRISE_CALIBRATION_HISTORY="${SURPRISE_CALIBRATION_HISTORY}",SURPRISE_QUANTILE="${SURPRISE_QUANTILE}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EXTRACT_JOB_ID}" \
  --job-name=cg-l15-cohort-final \
  --partition=gamma --account=gamma --qos="${QOS}" \
  "${EXCLUDE_ARGS[@]}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${FINALIZE_TIME_LIMIT}" \
  --output="${ARTIFACT_ROOT}/slurm_logs/l15-cohort-final-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/l15-cohort-final-%j.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",RUN_STAGE=finalize,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${VIDEO_LIMIT}",MEMORY_CAPACITY="${MEMORY_CAPACITY}",COHORT_PROTOCOL="${PROTOCOL}",COHORT_GATE_OUTPUT="${GRAPH_ROOT}/fixed_cohort_gate.json",COHORT_GATE_DETAILS="${GRAPH_ROOT}/fixed_cohort_gate.hidden_key.json" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

printf 'case_protocol=%s\n' "${PROTOCOL}"
printf 'selection=%s\n' "${SELECTION}"
printf 'extract_array_job_id=%s\n' "${EXTRACT_JOB_ID}"
printf 'finalize_job_id=%s\n' "${FINALIZE_JOB_ID}"
