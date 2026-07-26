# Inventory: 5 Baselines × 3 Benchmarks

Checked: 2026-07-26 on CIPR.

## Benchmarks

`ovo_bench` · `videomme` · `streaming_bench`

## Baselines

| # | Baseline | Primary tree on machine | Ported path |
| --- | --- | --- | --- |
| 1 | Dispider | `atomic_skills_for_video/baseline/dispider_streaming_eval.py` | `baseline/` + `baselines/dispider/` |
| 2 | StreamBridge | `/home/xwu/ml-streambridge` local deltas | `streambridge_local/` + `baselines/streambridge/` |
| 3 | M3-Agent | `/home/xwu/m3-agent/m3_agent/streaming_eval.py` | `m3_agent_streaming/` + `baselines/m3_agent/` |
| 4 | Iterative RAG | `baseline/iterative_rag_memory_query.py` | `baseline/` + `baselines/iterative_rag/` |
| 5 | Qwen3.5-9B + FAISS | `build_faiss_index.py`, `per_video_embedding_rag.py`, `qwen35_streaming_eval.py` | `baseline/` + `baselines/qwen9b_faiss/` |

## Dataset paths

| Dataset | Media | Annotations |
| --- | --- | --- |
| OVO-Bench | `/net/mlfs01/export/users/dpatel/OVO-Bench` | `ml-streambridge/assets/ovo_bench.json` (1640) |
| VideoMME | `/net/nj-storage02/.../videomme` | `ml-streambridge/assets/videomme.json` (2700) |
| StreamingBench | `/mnt/is_data/xwu/video_skills/data/datasets/StreamingBench` | Official CSVs; stage via `scripts/stage_streaming_bench.sbatch` |

## Existing output roots (not in git)

`/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/`

| Baseline family | Example artifacts |
| --- | --- |
| Qwen3.5-9B (+ direct / FAISS stack) | `qwen35_streaming_eval_sharded/290921_merged` (OVO+VideoMME overall acc ≈ 0.487); `baseline_faiss/`; `per_video_embedding_rag/full_3bench_*` |
| Iterative RAG | `iterative_rag_memory_query/full_3bench_*` |
| Dispider | `dispider_streaming_eval/smoke_3bench_*` |
| StreamingBench adapter | `streaming_bench_adapter_smoke/` |

## Runtime

```text
Qwen3.5-9B: /mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
Venv:       /mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
Dispider:   /mnt/is_data/xwu/video_skills/code/Dispider (+ dispider_venv)
```
