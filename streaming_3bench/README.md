# Streaming 3-Benchmark Implementations

Port of local cluster implementations for streaming video QA on three benchmarks:

| Benchmark | Adapter / CLI id | Role |
| --- | --- | --- |
| **OVO-Bench** | `ovo_bench` | Online / chunked realtime QA |
| **VideoMME** | `videomme` | Prefix / StreamBridge-compatible video QA |
| **StreamingBench** | `streaming_bench` | Official StreamingBench CSV-backed streaming QA |

This tree was collected from working code on CIPR under `/home/xwu` and
`/mnt/is_data/xwu/video_skills` on 2026-07-26. Large datasets, indexes,
checkpoints, and eval outputs stay on shared storage and are **not** vendored
here.

## Layout

```text
streaming_3bench/
  baseline/                 # atomic_skills_for_video streaming baselines
  dataset_clip_wrapper/     # canonical schema + OVO/VideoMME/StreamingBench adapters
  scripts/                  # StreamingBench staging jobs
  docs/                     # cluster paths + status notes
  m3_agent_streaming/       # local M3-style streaming runner deltas
  streambridge_local/       # local StreamBridge eval deltas (not full upstream)
```

## Source trees on this machine

| Source | Path | What was taken |
| --- | --- | --- |
| Video Skills relaunch | `/home/xwu/atomic_skills_for_video` | `baseline/`, `dataset_clip_wrapper/`, staging scripts, docs |
| M3-Agent local fork | `/home/xwu/m3-agent` | `m3_agent/streaming_eval.py`, merge helper, Slurm scripts |
| StreamBridge local fork | `/home/xwu/ml-streambridge` | Local 3-bench eval patches + Qwen3.5 streaming model wrapper |
| Cluster outputs | `/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video` | Not copied; see [INVENTORY.md](INVENTORY.md) |

## Runnable baselines (same schema)

All primary runners consume canonical examples from `dataset_clip_wrapper` and
support `ovo_bench`, `videomme`, and `streaming_bench`.

1. **Direct local VLM** — `baseline/qwen35_streaming_eval.py`  
   Local Qwen3.5-9B, `INPUT_MODE=video_clip` preferred. Sharded via
   `baseline/slurm_qwen35_sharded_eval.sbatch`.

2. **Iterative text-memory RAG** — `baseline/iterative_rag_memory_query.py`  
   FAISS clip-text memory, streaming cutoff, no answer-time frame decode.

3. **Per-video embedding RAG + video-clip answer** — `baseline/per_video_embedding_rag.py`  
   Clip embeddings + retrieved visible clips fed to Qwen.

4. **Dispider end-to-end** — `baseline/dispider_streaming_eval.py`  
   External Dispider checkpoint; prefix-video no-future-leak boundary.

5. **M3-style memory retrieve** — `m3_agent_streaming/streaming_eval.py`  
   Reuses the same canonical schema; causal visible-clip memory bank.

6. **StreamBridge local eval** — `streambridge_local/eval/eval_benchmarks/`  
   `parallel_ovo_bench.py`, `parallel_videomme.py`, `parallel_streaming_bench.py`
   plus `eval/streaming_models/online_qwen35_vl.py`.

## Cluster defaults

```text
Project root:   /mnt/is_data/xwu/video_skills
Code/venv:      /mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
Qwen3.5-9B:     /mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
OVO videos:     /net/mlfs01/export/users/dpatel/OVO-Bench
VideoMME:       /net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme
StreamBridge annotations:
  /mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/{ovo_bench,videomme}.json
StreamingBench staging:
  /mnt/is_data/xwu/video_skills/data/datasets/StreamingBench
```

## Quick start (from this package)

```bash
export PYTHONPATH=/mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench:$PYTHONPATH
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench

# Canonical schema smoke (3 benches x N examples)
python -m dataset_clip_wrapper.cli --dataset ovo_bench --limit 2
python -m dataset_clip_wrapper.cli --dataset videomme --limit 2
python -m dataset_clip_wrapper.cli --dataset streaming_bench --limit 2

# Direct Qwen streaming smoke
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python -m baseline.qwen35_streaming_eval \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 2 \
  --input-mode video_clip
```

See `baseline/README.md` for FAISS build/query, sharded eval, Dispider, and
merge helpers. See [INVENTORY.md](INVENTORY.md) for completed runs and metrics
already present on shared storage.
