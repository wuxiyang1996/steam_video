#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/fs/gamma-projects/vlm-robot/steam_video}"
DATASET_ROOT="${DATASET_ROOT:-/fs/gamma-projects/vlm-robot/datasets}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/steam_video_new/implicit_world_model/datasets/streamingbench_ovo_l15_sft_v1}"
GRAPH_ROOT="${GRAPH_ROOT:-${ARTIFACT_ROOT}/graphs}"
SELECTION="${SELECTION:-${ARTIFACT_ROOT}/selection.public.json}"
NUM_SHARDS="${NUM_SHARDS:-96}"
MAX_PARALLEL="${MAX_PARALLEL:-8}"
GPU_TYPE="${GPU_TYPE:-rtxa6000}"
QOS="${QOS:-gamma-huge-long}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY_PER_TASK="${MEMORY_PER_TASK:-64G}"
EXCLUDE_NODES="${EXCLUDE_NODES:-gammagpu13}"
EXTRACT_TIME_LIMIT="${EXTRACT_TIME_LIMIT:-12:00:00}"
FINALIZE_TIME_LIMIT="${FINALIZE_TIME_LIMIT:-04:00:00}"
MEMORY_CAPACITY="${MEMORY_CAPACITY:-192}"
VIDEO_LIMIT="${VIDEO_LIMIT:-96}"
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

if [[ ! -f "${SELECTION}" ]]; then
  python -m steam_video_new.implicit_world_model.streaming_l15_data.build_manifest \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${ARTIFACT_ROOT}" \
    --max-videos "${VIDEO_LIMIT}" \
    --horizon-policy full_video
fi

ACTUAL_VIDEO_COUNT="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["videos"]))' "${SELECTION}")"
if [[ "${VIDEO_LIMIT}" -ne "${ACTUAL_VIDEO_COUNT}" ]]; then
  echo "VIDEO_LIMIT=${VIDEO_LIMIT} differs from selection video count ${ACTUAL_VIDEO_COUNT}" >&2
  exit 2
fi
if [[ "${NUM_SHARDS}" -ne "${ACTUAL_VIDEO_COUNT}" ]]; then
  echo "NUM_SHARDS must equal selection video count so each checkpoint owns one video" >&2
  exit 2
fi

EXCLUDE_ARGS=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  EXCLUDE_ARGS+=(--exclude="${EXCLUDE_NODES}")
fi

EXTRACT_JOB_ID="$(sbatch --parsable \
  --job-name=sb-ovo-l15 \
  --partition=gamma --account=gamma --qos="${QOS}" \
  "${EXCLUDE_ARGS[@]}" \
  --array="0-$((NUM_SHARDS - 1))%${MAX_PARALLEL}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${EXTRACT_TIME_LIMIT}" \
  --output="${ARTIFACT_ROOT}/slurm_logs/extract-%A_%a.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/extract-%A_%a.err" \
  --export=ALL,REPO_ROOT="${REPO_ROOT}",ARTIFACT_ROOT="${ARTIFACT_ROOT}",GRAPH_ROOT="${GRAPH_ROOT}",SELECTION="${SELECTION}",DATASET_ROOT="${DATASET_ROOT}",RUN_STAGE=extract,NUM_SHARDS="${NUM_SHARDS}",VIDEO_LIMIT="${VIDEO_LIMIT}",MEMORY_CAPACITY="${MEMORY_CAPACITY}",L1_LOCALIZATION_MODE="${L1_LOCALIZATION_MODE}",L1_REQUEST_CONCURRENCY="${L1_REQUEST_CONCURRENCY}",SERVER_BACKEND="${SERVER_BACKEND}",VLLM_ROOT="${VLLM_ROOT}",SURPRISE_SAMPLE_PERIOD_S="${SURPRISE_SAMPLE_PERIOD_S}",SURPRISE_MIN_WINDOW_S="${SURPRISE_MIN_WINDOW_S}",SURPRISE_MAX_WINDOW_S="${SURPRISE_MAX_WINDOW_S}",SURPRISE_CALIBRATION_HISTORY="${SURPRISE_CALIBRATION_HISTORY}",SURPRISE_QUANTILE="${SURPRISE_QUANTILE}" \
  "${REPO_ROOT}/steam_video_new/implicit_world_model/cgbench_grounded_navigation/run_l15_graph_smoke_job.sh")"

FINALIZE_JOB_ID="$(sbatch --parsable \
  --dependency="afterok:${EXTRACT_JOB_ID}" \
  --job-name=sb-ovo-l15-final \
  --partition=gamma --account=gamma --qos="${QOS}" \
  "${EXCLUDE_ARGS[@]}" \
  --gres="gpu:${GPU_TYPE}:1" --cpus-per-task="${CPUS_PER_TASK}" --mem="${MEMORY_PER_TASK}" --time="${FINALIZE_TIME_LIMIT}" \
  --output="${ARTIFACT_ROOT}/slurm_logs/final-%j.out" \
  --error="${ARTIFACT_ROOT}/slurm_logs/final-%j.err" \
  --wrap="cd '${REPO_ROOT}' && /fs/gamma-projects/vlm-robot/Video_Skills/.venv-qwen35-serve/bin/python -m steam_video_new.implicit_world_model.streaming_l15_data.compile_graphs --selection '${SELECTION}' --graph-root '${GRAPH_ROOT}' --report '${GRAPH_ROOT}/compile_report.json' --memory-capacity '${MEMORY_CAPACITY}' --device cuda --video-limit '${VIDEO_LIMIT}'")"

printf 'artifact_root=%s\n' "${ARTIFACT_ROOT}"
printf 'selection=%s\n' "${SELECTION}"
printf 'hidden_key=%s\n' "${ARTIFACT_ROOT}/targets.hidden_key.json"
printf 'graph_root=%s\n' "${GRAPH_ROOT}"
printf 'extract_array_job_id=%s\n' "${EXTRACT_JOB_ID}"
printf 'finalize_job_id=%s\n' "${FINALIZE_JOB_ID}"
