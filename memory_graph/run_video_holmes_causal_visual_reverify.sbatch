#!/bin/bash
#SBATCH --job-name=mg-vh-vlm-reverify
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
TRANSFORMERS=$WORKSPACE/Video_Skills/.venv-qwen35-serve/bin/transformers
MODEL=Qwen/Qwen3.5-9B
OUT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_causal_visual_smoke_2
PORT=$((20000 + SLURM_JOB_ID % 1000))
SERVER_LOG=$WORKSPACE/cluster/logs/mg-vh-vlm-reverify-server-${SLURM_JOB_ID}.log

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills:${PYTHONPATH:-}"
export HF_HOME="$WORKSPACE/Multi-hop-Reasoning-VLM-Agent/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
mkdir -p "$WORKSPACE/cluster/logs"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVE_BIN="$(dirname "$TRANSFORMERS")"
env -u PYTHONPATH -u PYTHONHOME -u SETUPTOOLS_USE_DISTUTILS \
  PATH="${SERVE_BIN}:/usr/bin:/bin" \
  HF_HOME="$HF_HOME" \
  TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
  TOKENIZERS_PARALLELISM=false \
  VLLM_USE_DEEP_GEMM=0 \
  "$TRANSFORMERS" serve "$MODEL" \
  --host 127.0.0.1 \
  --port "$PORT" \
  --device cuda:0 \
  --dtype bfloat16 \
  --reasoning off \
  --attn-implementation sdpa >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 300); do
  if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null; then
    break
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    printf 'Qwen server exited before becoming ready\n' >&2
    exit 1
  fi
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null

cd "$WORKSPACE/steam_video"
"$PY" -u -m memory_graph.visual_reverify_overlay \
  --run-dir "$OUT" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --model "$MODEL" \
  --api-base "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --frames-per-window 3
