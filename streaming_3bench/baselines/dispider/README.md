# Baseline 1 — Dispider

End-to-end streaming VideoLLM baseline on **OVO-Bench / VideoMME / StreamingBench**.

## Code

- Runner: `../../baseline/dispider_streaming_eval.py`
- Slurm: `../../baseline/slurm_dispider_streaming_eval.sbatch`
- External repo (not vendored): `/mnt/is_data/xwu/video_skills/code/Dispider`

## Contract

- Uses the same canonical schema as the other four baselines.
- Materializes a **visible-prefix video** per example before calling Dispider so
  future frames beyond `visible_until_s` cannot leak.
- Writes `run_config.json`, `canonical_schemas.jsonl`, `records.jsonl`,
  `metrics_summary.json`.

## Smoke

```bash
cd /mnt/is_data/xwu/video_skills/code/steam_video/streaming_3bench
export PYTHONPATH=$PWD:$PYTHONPATH

PROJECT=/mnt/is_data/xwu/video_skills \
VENV=$PROJECT/code/dispider_venv \
DISPIDER_REPO=$PROJECT/code/Dispider \
MODEL=$PROJECT/data/models/dispider/<checkpoint> \
DATASETS="ovo_bench videomme streaming_bench" \
LIMIT_PER_DATASET=5 \
sbatch --array=0-0 baseline/slurm_dispider_streaming_eval.sbatch
```
