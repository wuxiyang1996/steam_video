# GAMMA runtime

## Environments

- Qwen evaluation: `/fs/gamma-projects/vlm-robot/conda/envs/video-streaming-eval`
  - PyTorch 2.6.0 + CUDA 12.4
  - Transformers 5.14.1
  - FlashAttention 2.7.4.post1
  - Flash Linear Attention 0.5.1
  - vLLM 0.8.5.post1
  - FAISS CPU 1.14.3
  - Qwen VL utils, PyAV, and Decord
- VERL training: `/fs/gamma-projects/vlm-robot/conda/envs/video-skills-grpo`
  - FlashAttention 2.7.4.post1
  - VERL 0.8.0
  - Ray 2.56.1

VERL is intentionally separate from the evaluation environment. VERL 0.8 pulls
Ray's current telemetry stack, which conflicts with the older telemetry versions
required by vLLM 0.8.5.

The current vLLM build imports and detects CUDA, but its model registry predates
`Qwen3_5ForConditionalGeneration`. Qwen3.5 evaluation therefore uses the tested
Transformers + FlashAttention path. A vLLM Qwen3.5 backend requires a newer
CUDA/PyTorch/vLLM stack and is not used by the baseline jobs yet.

## Activation

```bash
cd /fs/gamma-projects/vlm-robot/steam_video-streaming-3bench/streaming_3bench
source scripts/activate_gamma_eval.sh
```

The activation script selects the full GAMMA benchmark data, shared Hugging Face
cache, scratch output root, and `flash_attention_2`.

## Smoke test

```bash
sbatch scripts/slurm_gamma_qwen_smoke.sbatch
```

The job validates CUDA imports and evaluates one full-data example from each of
OVO-Bench, VideoMME, and StreamingBench.
