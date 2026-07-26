# Machine Inventory: Streaming Video × 3 Benchmarks

Checked: 2026-07-26 on CIPR (`cipr-js01`).

Benchmarks: **OVO-Bench**, **VideoMME**, **StreamingBench**.

## 1. Implementation locations

### 1.1 Primary (ported here)

`/home/xwu/atomic_skills_for_video` (branch `video_skills_relaunched`)

| Component | Path | 3-bench support |
| --- | --- | --- |
| Adapters | `dataset_clip_wrapper/adapters/streaming_video.py` | `ovo_bench`, `videomme`, `streaming_bench` |
| Direct Qwen eval | `baseline/qwen35_streaming_eval.py` | yes |
| Iterative RAG | `baseline/iterative_rag_memory_query.py` | yes |
| Per-video embedding RAG | `baseline/per_video_embedding_rag.py` | yes (default all 3) |
| Dispider eval | `baseline/dispider_streaming_eval.py` | yes |
| FAISS index build/query | `baseline/build_faiss_index.py`, `query_faiss_index.py`, … | schema-aligned |
| Sharded Slurm | `baseline/slurm_*.sbatch`, `submit_sharded_eval.sh` | yes |
| StreamingBench stage | `scripts/stage_streaming_bench.{py,sbatch}` | staging only |

### 1.2 M3-Agent local streaming runner (ported)

`/home/xwu/m3-agent`

- `m3_agent/streaming_eval.py`
- `m3_agent/merge_streaming_eval_shards.py`
- `scripts/slurm_m3_streaming_qwen35.sbatch`
- `scripts/slurm_m3_streaming_qwen35_4gpu_singlejob.sbatch`

Depends on `atomic_skills_for_video.dataset_clip_wrapper` for canonical examples.

### 1.3 StreamBridge local deltas (ported)

`/home/xwu/ml-streambridge` (upstream Apple StreamBridge + local patches)

Local/untracked additions:

- `eval/eval_benchmarks/parallel_streaming_bench.py`
- `eval/streaming_models/online_qwen35_vl.py`
- `eval/merge_streambridge_shards.py`
- `scripts/slurm_qwen35_ovo_videomme*.sbatch`
- `scripts/slurm_qwen35_streaming_bench_sharded.sbatch`
- `demo_qwen35_agent.py`

Modified upstream eval helpers for local Qwen / sharding.

### 1.4 Related but not the 3-bench eval harness

| Path | Notes |
| --- | --- |
| `/mnt/is_data/xwu/video_skills/code/steam_video` | This repo: formulations + `memory_graph` |
| `/home/xwu/continuous_memory_query` | StreamQA / continuous-memory experiments |
| `/home/xwu/NEC_intern/Steam_Video_Agents` | Older formulation notes |
| `/mnt/is_data/xwu/video_skills/code/Dispider` | External Dispider checkout for baseline 4 |

## 2. Dataset / annotation paths

| Dataset | Media | Annotations |
| --- | --- | --- |
| OVO-Bench | `/net/mlfs01/export/users/dpatel/OVO-Bench` (`chunked_videos/`) | `/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/ovo_bench.json` (1640 rows) |
| VideoMME | `/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme` | `.../ml-streambridge/assets/videomme.json` (2700 rows) |
| StreamingBench | `/mnt/is_data/xwu/video_skills/data/datasets/StreamingBench` | Official CSVs staged; full video archive via `scripts/stage_streaming_bench.sbatch` |

## 3. Notable eval artifacts (not in git)

Root: `/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/`

| Run family | Example dir | Notes |
| --- | --- | --- |
| Direct Qwen full (OVO+VideoMME) | `qwen35_streaming_eval_sharded/290921_merged` | overall acc ≈ **0.487** (OVO 0.465 / VideoMME 0.499), 4340 examples |
| Direct Qwen smoke | `qwen35_streaming_eval/290877` | 5+5 smoke, `INPUT_MODE=video_clip` |
| Iterative RAG 3-bench | `iterative_rag_memory_query/full_3bench_*` | full 3-bench jobs present |
| Per-video embedding RAG | `per_video_embedding_rag/full_3bench_clip_per_video_4frames_scan` | 3-bench CLIP scan index/eval |
| Dispider smoke | `dispider_streaming_eval/smoke_3bench_*` | 3-bench smoke shards |
| StreamingBench adapter smoke | `streaming_bench_adapter_smoke/`, `streaming_bench_adapter_live_smoke/` | adapter/schema smoke |

FAISS index smoke:

```text
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames_scan
```

## 4. Runtime stack

```text
Model:   Qwen3.5-9B @ /mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
Env:     /mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
Stack:   HF Transformers, AutoProcessor, AutoModelForImageTextToText, bfloat16
Policy:  streaming visibility via visible_until_s / observation_end_s; no future clips
```

## 5. What this branch intentionally excludes

- Datasets, videos, FAISS indexes, checkpoints, bulk logs
- `keys.py` / API secrets
- Full upstream StreamBridge / M3-Agent / Dispider repositories
- Non-streaming five-dataset L1/L2 graph controller outputs
