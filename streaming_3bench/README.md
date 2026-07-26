# Streaming 3-Benchmark × 5 Baselines

Canonical local eval package for streaming video QA.

## Benchmarks (3)

| Benchmark | CLI id | Role |
| --- | --- | --- |
| **OVO-Bench** | `ovo_bench` | Online / chunked realtime QA |
| **VideoMME** | `videomme` | Prefix / StreamBridge-compatible video QA |
| **StreamingBench** | `streaming_bench` | Official StreamingBench streaming QA |

## Baselines (5)

| # | Baseline | Entry points | What it is |
| --- | --- | --- | --- |
| 1 | **Dispider** | `baselines/dispider/` → `baseline/dispider_streaming_eval.py` | End-to-end streaming VideoLLM; visible-prefix video, no future leak |
| 2 | **StreamBridge** | `baselines/streambridge/` → `streambridge_local/` | StreamBridge eval harness + local Qwen3.5 streaming wrapper |
| 3 | **M3-Agent** | `baselines/m3_agent/` → `m3_agent_streaming/streaming_eval.py` | M3-style causal visible-clip memory retrieve + local Qwen answer |
| 4 | **Iterative RAG** | `baselines/iterative_rag/` → `baseline/iterative_rag_memory_query.py` | Multi-round FAISS **text-memory** RAG; answers from retrieved clip text only |
| 5 | **Qwen3.5-9B + FAISS** | `baselines/qwen9b_faiss/` → FAISS build/query + `per_video_embedding_rag.py` / `qwen35_streaming_eval.py` | Clip FAISS index + local Qwen3.5-9B (retrieve-then-answer and/or direct visible-clip QA) |

All five share the same canonical schema from `dataset_clip_wrapper` and the same
three dataset ids. Large media, indexes, checkpoints, and metrics stay under
`/mnt/is_data/xwu/video_skills` (see [INVENTORY.md](INVENTORY.md)).

```text
                    ovo_bench   videomme   streaming_bench
Dispider               ✓           ✓             ✓
StreamBridge           ✓           ✓             ✓
M3-Agent               ✓           ✓             ✓
Iterative RAG          ✓           ✓             ✓
Qwen3.5-9B + FAISS     ✓           ✓             ✓
```

## Layout

```text
streaming_3bench/
  README.md                 # this file (5×3 contract)
  INVENTORY.md              # cluster paths + existing runs
  baselines/                # one folder per baseline (pointers + how-to)
    dispider/
    streambridge/
    m3_agent/
    iterative_rag/
    qwen9b_faiss/
  baseline/                 # shared Python package (FAISS, Qwen, Dispider, iterative RAG)
  dataset_clip_wrapper/     # adapters + canonical clip schema
  m3_agent_streaming/       # M3 runner package
  streambridge_local/       # StreamBridge local deltas
  scripts/                  # StreamingBench staging
  docs/                     # cluster dataset notes
```

## Cluster defaults

```text
Project:     /mnt/is_data/xwu/video_skills
Venv:        /mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
Qwen3.5-9B:  /mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
OVO:         /net/mlfs01/export/users/dpatel/OVO-Bench
VideoMME:    /net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme
Annotations: /mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/{ovo_bench,videomme}.json
StreamingBench staging: /mnt/is_data/xwu/video_skills/data/datasets/StreamingBench
```

## Quick start

```bash
export PYTHONPATH=/mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench:$PYTHONPATH
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench

# Shared schema smoke on all 3 benches
python -m dataset_clip_wrapper.cli --dataset ovo_bench --limit 2
python -m dataset_clip_wrapper.cli --dataset videomme --limit 2
python -m dataset_clip_wrapper.cli --dataset streaming_bench --limit 2
```

Per-baseline commands live under `baselines/*/README.md`.
