# Baseline 2 — StreamBridge

StreamBridge streaming eval harness (local deltas) on **OVO-Bench / VideoMME / StreamingBench**.

## Code

Local deltas live in `../../streambridge_local/` (apply on top of upstream
`apple/ml-streambridge`; full upstream is at `/home/xwu/ml-streambridge` and
`/mnt/is_data/xwu/video_skills/code/ml-streambridge`).

Key files:

- `streambridge_local/eval/eval_benchmarks/parallel_ovo_bench.py`
- `streambridge_local/eval/eval_benchmarks/parallel_videomme.py`
- `streambridge_local/eval/eval_benchmarks/parallel_streaming_bench.py`
- `streambridge_local/eval/streaming_models/online_qwen35_vl.py`
- `streambridge_local/scripts/slurm_qwen35_ovo_videomme_sharded.sbatch`
- `streambridge_local/scripts/slurm_qwen35_streaming_bench_sharded.sbatch`

## Contract

- Official StreamBridge-style multi-GPU / sharded eval path.
- Local addition: StreamingBench parallel runner + Qwen3.5 online VL wrapper.
- Annotations: `ml-streambridge/assets/{ovo_bench,videomme}.json`.

## Smoke

```bash
# From a StreamBridge checkout with these local files overlaid:
sbatch scripts/slurm_qwen35_ovo_videomme_sharded.sbatch
sbatch scripts/slurm_qwen35_streaming_bench_sharded.sbatch
```
