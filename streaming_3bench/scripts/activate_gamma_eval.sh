#!/usr/bin/env bash
# Source this file before running the GAMMA streaming evaluation jobs.

export VLM_ROBOT_ROOT="${VLM_ROBOT_ROOT:-/fs/gamma-projects/vlm-robot}"
export STREAMING_EVAL_ROOT="${STREAMING_EVAL_ROOT:-$VLM_ROBOT_ROOT/steam_video-streaming-3bench/streaming_3bench}"
export STREAMING_EVAL_ENV="${STREAMING_EVAL_ENV:-$VLM_ROBOT_ROOT/conda/envs/video-streaming-eval}"

export PATH="$STREAMING_EVAL_ENV/bin:$PATH"
export PYTHONPATH="$STREAMING_EVAL_ROOT:$VLM_ROBOT_ROOT/Video_Skills${PYTHONPATH:+:$PYTHONPATH}"
export VIDEO_SKILLS_PROJECT_ROOT="$VLM_ROBOT_ROOT"
export VIDEO_SKILLS_DATASET_ROOT="$VLM_ROBOT_ROOT/datasets"
export VIDEO_SKILLS_OUTPUT_ROOT="${VIDEO_SKILLS_OUTPUT_ROOT:-/gammascratch/$USER/streaming_3bench_outputs}"
export HF_HOME="${HF_HOME:-$VLM_ROBOT_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$VLM_ROBOT_ROOT/hf_cache/hub}"
export QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-flash_attention_2}"

mkdir -p "$VIDEO_SKILLS_OUTPUT_ROOT"
