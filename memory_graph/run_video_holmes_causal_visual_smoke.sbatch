#!/bin/bash
#SBATCH --job-name=mg-vh-causal-vlm2
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
TRANSFORMERS=$WORKSPACE/Video_Skills/.venv-qwen35-serve/bin/transformers
MODEL=Qwen/Qwen3.5-9B
OUT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_causal_visual_smoke_2
PORT=$((19000 + SLURM_JOB_ID % 1000))
SERVER_LOG=$WORKSPACE/cluster/logs/mg-vh-causal-vlm-server-${SLURM_JOB_ID}.log

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$WORKSPACE/Multi-hop-Reasoning-VLM-Agent/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
mkdir -p "$OUT" "$WORKSPACE/cluster/logs"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$WORKSPACE/steam_video"
"$PY" -u -m memory_graph.validate_video_holmes \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --output-dir "$OUT" \
  --video-id=0CPoD4ZRTUQ \
  --video-id=1TR9riaDzY8 \
  --device cuda:0 \
  --batch-size 4 \
  --top-k 2 \
  --input-mode expert_demo \
  --allow-provisional-expert-demo \
  --memory-capacity 32

# Clean env for serve — avoid conda/swift distutils clash with serve venv.
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

"$PY" -u -m memory_graph.visual_reverify_overlay \
  --run-dir "$OUT" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --model "$MODEL" \
  --api-base "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --frames-per-window 3
