# Baseline 3 — M3-Agent

M3-style streaming memory retrieve + local Qwen answer on
**OVO-Bench / VideoMME / StreamingBench**.

## Code

- Runner: `../../m3_agent_streaming/streaming_eval.py`
- Merge: `../../m3_agent_streaming/merge_streaming_eval_shards.py`
- Slurm: `../../m3_agent_streaming/scripts/slurm_m3_streaming_qwen35.sbatch`
- Slurm (4-GPU single job): `../../m3_agent_streaming/scripts/slurm_m3_streaming_qwen35_4gpu_singlejob.sbatch`

## Contract

- Reuses `dataset_clip_wrapper` canonical examples (same schema as FAISS / Qwen / Dispider).
- Visible clips form a causal memory bank; retrieve top-k, then answer with local Qwen3.5-9B.
- Supported datasets: `ovo_bench`, `videomme`, `streaming_bench`.

## Smoke

```bash
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench
export PYTHONPATH=$PWD:$PYTHONPATH

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  m3_agent_streaming/streaming_eval.py \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 2
```
