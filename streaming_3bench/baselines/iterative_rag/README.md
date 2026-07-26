# Baseline 4 — Iterative RAG

Multi-round **text-memory** RAG over the shared FAISS clip index on
**OVO-Bench / VideoMME / StreamingBench**.

## Code

- Runner: `../../baseline/iterative_rag_memory_query.py`
- Index: built by baseline 5 (`build_faiss_index.py` / `faiss_store.py`)
- Slurm helpers: reuse FAISS index jobs under `../../baseline/slurm_qwen3_vl_*.sbatch`

## Contract

- Iteratively queries FAISS (prefer per-video refs).
- Applies streaming cutoff (`clip.start_s <= visible_until_s`).
- Answers from retrieved **clip-memory text only** (no answer-time frame decode).
- Distinct from baseline 5’s retrieve-then-**video**-answer path.

## Smoke

```bash
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench
export PYTHONPATH=$PWD:$PYTHONPATH

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.iterative_rag_memory_query \
  --datasets ovo_bench videomme streaming_bench \
  --limit-per-dataset 2 \
  --index-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/<index>
```
