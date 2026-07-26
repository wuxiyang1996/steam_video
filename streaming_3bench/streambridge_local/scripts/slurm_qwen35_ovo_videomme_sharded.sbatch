#!/bin/bash
#SBATCH --job-name=sb-qwen35-shard
#SBATCH --partition=prod
#SBATCH --constraint=GPUMODEL_A6000
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_shard-%A_%a.out
#SBATCH --error=/mnt/is_data/xwu/video_skills/outputs/ml-streambridge/logs/qwen35_shard-%A_%a.err

set -euo pipefail

PROJECT=${PROJECT:-/mnt/is_data/xwu/video_skills}
REPO=${REPO:-/home/xwu/ml-streambridge}
VENV=${VENV:-$PROJECT/code/vllm_qwen_cu124_venv}
MODEL_CKPT=${MODEL_CKPT:-$PROJECT/data/models/qwen35_9b/Qwen3.5-9B}
RUN_ID=${RUN_ID:-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}}
NUM_SHARDS=${NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}
SHARD_INDEX=${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}
OUT_ROOT=${OUT_ROOT:-$PROJECT/outputs/ml-streambridge/qwen35_ovo_videomme_sharded/$RUN_ID}
RESULT_DIR=${RESULT_DIR:-$OUT_ROOT/shard_${SHARD_INDEX}/results}

export HF_HOME=${HF_HOME:-$PROJECT/data/models/hf_cache}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$PROJECT/data/models/hf_cache/hub}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$PROJECT/data/models/hf_cache/transformers}

export MODEL=qwen35vl
export CKPT="$MODEL_CKPT"
export RESULT_DIR
export NUM_GPUS=1
export STREAMBRIDGE_NUM_SHARDS="$NUM_SHARDS"
export STREAMBRIDGE_SHARD_INDEX="$SHARD_INDEX"
export CLIP_WINDOW_S=${CLIP_WINDOW_S:-4.0}
export QWEN35_VIDEO_FPS=${QWEN35_VIDEO_FPS:-2.0}
export QWEN35_VIDEO_MAX_FRAMES_PER_CLIP=${QWEN35_VIDEO_MAX_FRAMES_PER_CLIP:-8}
export VIDEOMME_OBSERVATION_END_S=${VIDEOMME_OBSERVATION_END_S:-60.0}

mkdir -p "$RESULT_DIR" "$PROJECT/outputs/ml-streambridge/logs"
cd "$REPO"

cat > "$OUT_ROOT/shard_${SHARD_INDEX}/run_config.json" <<EOF
{
  "repo": "$REPO",
  "model": "$MODEL",
  "ckpt": "$CKPT",
  "result_dir": "$RESULT_DIR",
  "num_shards": "$NUM_SHARDS",
  "shard_index": "$SHARD_INDEX",
  "clip_window_s": "$CLIP_WINDOW_S",
  "qwen35_video_fps": "$QWEN35_VIDEO_FPS",
  "qwen35_video_max_frames_per_clip": "$QWEN35_VIDEO_MAX_FRAMES_PER_CLIP",
  "videomme_observation_end_s": "$VIDEOMME_OBSERVATION_END_S",
  "streambridge_limit_videos": "${STREAMBRIDGE_LIMIT_VIDEOS:-}",
  "streambridge_limit_records": "${STREAMBRIDGE_LIMIT_RECORDS:-}",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_job_id": "${SLURM_ARRAY_JOB_ID:-}",
  "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}"
}
EOF

export ANNO_PATH=${OVO_ANNO_PATH:-$REPO/assets/ovo_bench.json}
export VIDEO_PATH=${OVO_VIDEO_PATH:-/net/mlfs01/export/users/dpatel/OVO-Bench/src_videos}
"$VENV/bin/python" eval/eval_benchmarks/parallel_ovo_bench.py

export ANNO_PATH=${VIDEOMME_ANNO_PATH:-$REPO/assets/videomme.json}
export VIDEO_PATH=${VIDEOMME_VIDEO_PATH:-/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme/videos}
"$VENV/bin/python" eval/eval_benchmarks/parallel_videomme.py

"$VENV/bin/python" eval/metric_report.py | tee "$RESULT_DIR/metric_report.txt"

echo "Shard results: $RESULT_DIR"
