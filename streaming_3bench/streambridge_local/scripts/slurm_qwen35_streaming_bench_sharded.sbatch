#!/bin/bash
#SBATCH --job-name=sb-qwen35-sbench
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_sbench-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_sbench-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/ml-streambridge}
ATOMIC_REPO=${ATOMIC_REPO:-/home/xwu/atomic_skills_for_video}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MODEL_CKPT=${MODEL_CKPT:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/ml-streambridge/qwen35_streaming_bench_sharded/$RUN_ID}
RESULT_DIR=${RESULT_DIR:-$OUT_ROOT/shard_${SHARD_INDEX}}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

export MODEL=qwen35vl
export CKPT="$MODEL_CKPT"
export RESULT_DIR
export ATOMIC_REPO
export DATASET_ROOT=${DATASET_ROOT:-$PROJECT/data/datasets}
export NUM_GPUS=1
export STREAMBRIDGE_NUM_SHARDS="$NUM_SHARDS"
export STREAMBRIDGE_SHARD_INDEX="$SHARD_INDEX"
export CLIP_WINDOW_S=${CLIP_WINDOW_S:-4.0}
export QWEN35_VIDEO_FPS=${QWEN35_VIDEO_FPS:-2.0}
export QWEN35_VIDEO_MAX_FRAMES_PER_CLIP=${QWEN35_VIDEO_MAX_FRAMES_PER_CLIP:-8}

mkdir -p "$RESULT_DIR" "$PROJECT/outputs/ml-streambridge/logs"
cd "$REPO"

cat > "$OUT_ROOT/shard_${SHARD_INDEX}/run_config.json" <<EOF
{
  "repo": "$REPO",
  "atomic_repo": "$ATOMIC_REPO",
  "model": "$MODEL",
  "ckpt": "$CKPT",
  "result_dir": "$RESULT_DIR",
  "dataset_root": "$DATASET_ROOT",
  "num_shards": "$NUM_SHARDS",
  "shard_index": "$SHARD_INDEX",
  "clip_window_s": "$CLIP_WINDOW_S",
  "qwen35_video_fps": "$QWEN35_VIDEO_FPS",
  "qwen35_video_max_frames_per_clip": "$QWEN35_VIDEO_MAX_FRAMES_PER_CLIP",
  "streambridge_limit_records": "${STREAMBRIDGE_LIMIT_RECORDS:-}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
  "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}"
}
EOF

"$VENV/bin/python" eval/eval_benchmarks/parallel_streaming_bench.py

echo "Shard results: $RESULT_DIR"
