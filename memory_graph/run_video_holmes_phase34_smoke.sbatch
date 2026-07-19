#!/bin/bash
#SBATCH --job-name=mg-vh-phase34
#SBATCH --partition=gamma
#SBATCH --account=gamma
#SBATCH --qos=default
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.out
#SBATCH --error=/fs/gamma-projects/vlm-robot/cluster/logs/%x-%j.err

set -euo pipefail

WORKSPACE=/fs/gamma-projects/vlm-robot
PY=$WORKSPACE/conda/envs/cosplay-candy-a100/bin/python
TRANSFORMERS=$WORKSPACE/Video_Skills/.venv-qwen35-serve/bin/transformers
MODEL=Qwen/Qwen3.5-9B
RUN_ROOT=$WORKSPACE/steam_video/memory_graph/outputs/video_holmes_phase34_smoke
L1_ROOT=$RUN_ROOT/video_skills_l1
GRAPH_ROOT=$RUN_ROOT/memory_graph
PORT=$((23000 + SLURM_JOB_ID % 1000))
SERVER_LOG=$WORKSPACE/cluster/logs/mg-vh-phase34-server-${SLURM_JOB_ID}.log

export HF_HOME=$WORKSPACE/Multi-hop-Reasoning-VLM-Agent/.hf_cache
export TRANSFORMERS_CACHE=$HF_HOME/hub
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_DEEP_GEMM=0
mkdir -p "$WORKSPACE/cluster/logs" "$L1_ROOT" "$GRAPH_ROOT"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

SERVE_BIN=$(dirname "$TRANSFORMERS")
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

export PYTHONPATH="$WORKSPACE/steam_video:$WORKSPACE/Video_Skills${PYTHONPATH:+:$PYTHONPATH}"
cd "$WORKSPACE/Video_Skills"
"$PY" -u -m dataset_clip_wrapper.run_staged_llm_pipeline \
  --dataset video_holmes \
  --dataset-root "$WORKSPACE/datasets" \
  --split train \
  --mode video_only \
  --unique-videos \
  --start-index 100 \
  --limit 1 \
  --skip-l2-planner \
  --disable-llm-skills \
  --disable-vlm-skills \
  --clip-schema-model "$MODEL" \
  --clip-schema-api-base "http://127.0.0.1:${PORT}/v1/chat/completions" \
  --clip-schema-frames 2 \
  --clip-schema-max-tokens 1200 \
  --clip-schema-timeout-s 300 \
  --clip-schema-workers 1 \
  --graph-model openai/gpt-oss-120b \
  --graph-timeout-s 300 \
  --graph-neighbor-workers 1 \
  --output "$L1_ROOT/examples.jsonl" \
  --stage-dir "$L1_ROOT/stages"

kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true
unset SERVER_PID

cd "$WORKSPACE/steam_video"
"$PY" -u -m memory_graph.validate_video_holmes \
  --dataset-root "$WORKSPACE/datasets" \
  --video-skills-root "$WORKSPACE/Video_Skills" \
  --keys-py "$WORKSPACE/keys.py" \
  --input-mode video_only \
  --video-skills-l1-jsonl "$L1_ROOT/examples.jsonl" \
  --output-dir "$GRAPH_ROOT" \
  --device cuda:0 \
  --batch-size 4 \
  --top-k 2 \
  --memory-capacity 24
