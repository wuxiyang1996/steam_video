# Baseline 5 — Qwen3.5-9B + FAISS

Shared **clip FAISS index** plus local **Qwen3.5-9B** answering on
**OVO-Bench / VideoMME / StreamingBench**.

## Code

| Role | Path |
| --- | --- |
| Build FAISS | `../../baseline/build_faiss_index.py` |
| Query FAISS | `../../baseline/query_faiss_index.py` |
| Merge indexes | `../../baseline/merge_faiss_indexes.py` |
| Store / embeddings | `../../baseline/faiss_store.py`, `embeddings.py`, `schemas.py` |
| Retrieve + video-clip Qwen answer | `../../baseline/per_video_embedding_rag.py` |
| Direct visible-clip Qwen QA (same schema) | `../../baseline/qwen35_streaming_eval.py` |
| Slurm | `slurm_qwen3_vl_*.sbatch`, `slurm_per_video_embedding_rag.sbatch`, `slurm_qwen35_sharded_eval.sbatch` |

## Contract

```text
canonical wrapper clips
  -> CLIP / Qwen3-VL embeddings
  -> FAISS index (per-video refs preferred)
  -> streaming visibility filter
  -> local Qwen3.5-9B answer
```

Two answer modes under this baseline family:

1. **FAISS retrieve → Qwen video answer** — `per_video_embedding_rag.py`
2. **Direct visible-clip Qwen** (no retrieve) — `qwen35_streaming_eval.py` with
   `INPUT_MODE=video_clip`

Iterative text-only RAG is **baseline 4**, not this baseline.

## Smoke

```bash
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench
export PYTHONPATH=$PWD:$PYTHONPATH
VENV=/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python

# Direct Qwen on 3 benches
$VENV -m baseline.qwen35_streaming_eval \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 2 \
  --input-mode video_clip

# Per-video embedding RAG + Qwen
$VENV -m baseline.per_video_embedding_rag \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 2
```
