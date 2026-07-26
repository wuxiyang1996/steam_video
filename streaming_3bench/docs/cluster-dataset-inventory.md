# Cluster Dataset Inventory

Last checked: 2026-07-06

This page records the dataset paths verified from the CIPR-accessible cluster
workspace. Keep large datasets on shared storage; do not copy them into this
repo or `/home/xwu`.

## Primary Five-Dataset Track

| Dataset | Verified location | Current adapter status |
| --- | --- | --- |
| `Video-Holmes` | `/mnt/is_data/xwu/video_skills/data/video_holmes` | Adapter-ready through `/mnt/is_data/xwu/video_skills/data/datasets/Video-Holmes/Benchmark`. |
| `CG-Bench` | `/mnt/is_data/xwu/video_skills/data/cg_bench/raw` | Raw data is present, about 396G. Adapter-ready through `/mnt/is_data/xwu/video_skills/data/datasets/CG-Bench`; the adapter skips annotation rows whose media are absent locally. |
| `VRBench` | `/mnt/is_data/xwu/video_skills/data/vrbench/raw` | Raw data is present, about 793G. Adapter-ready for smoke tests through `/mnt/is_data/xwu/video_skills/data/datasets/VRBench`; the first eval video is extracted, while the rest of `v001_360p` remains in the multipart archive. |
| `OVO-Bench` | `/net/mlfs01/export/users/dpatel/OVO-Bench` | Full videos are accessible. The adapter can now pair this path with StreamBridge annotations at `/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/ovo_bench.json`. |
| `VideoMME` | `/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme` | Videos/subtitles are accessible. The adapter can now pair this path with StreamBridge annotations at `/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/videomme.json`. |

`SIV-Bench` is also staged at `/mnt/is_data/xwu/video_skills/data/siv_bench`
and is adapter-ready, but it is treated as a secondary answerability/repair
stress-test dataset rather than part of the primary five-dataset track.

## StreamQA-120K

StreamQA-120K is available as a training source rather than the primary
evaluation benchmark:

```text
/net/mlfs01/export/users/dpatel/StreamingVideoLLM/data/streamqa-120k/train.jsonl
```

Verified properties:

- `128292` JSONL rows.
- File size is about `219M`.
- Rows contain `video_ids`, `video_files`, `captions`, `questions`,
  `options`, `answers`, and `types`.
- The referenced video files are relative to:

```text
/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/
```

For example, the first checked row references:

```text
webvid-703k/videos/034101_034150/1012406735.mp4
```

and the corresponding file exists at:

```text
/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/webvid-703k/videos/034101_034150/1012406735.mp4
```

## StreamingBench

StreamingBench is part of the StreamBridge paper's streaming evaluation suite
alongside OVO-Bench. The official CSV annotations are staged locally; the video
archives are large and should be downloaded through the provided shared-storage
staging job.

Expected location once staged:

```text
/mnt/is_data/xwu/video_skills/data/datasets/StreamingBench
```

Expected annotation/media layouts supported by the adapter:

```text
StreamingBench/questions_real.json
StreamingBench/questions_omni.json
StreamingBench/questions_sqa.json
StreamingBench/questions_proactive.json
StreamingBench/questions_proactive_50.json
StreamingBench/src/data/questions_*.json
StreamingBench/questions.json
StreamingBench/*.jsonl
StreamingBench/*.parquet
StreamingBench/videos/
StreamingBench/src/data/videos/
StreamingBench/{real,omni,sqa,proactive}/
StreamingBench/data/{real,omni,sqa,proactive}/
```

Current status:

- Official CSV annotations are staged under
  `/mnt/is_data/xwu/video_skills/data/datasets/StreamingBench/StreamingBench`.
- Adapter code is wired as `streaming_bench` and reads the official CSV files.
- CLI choices include `streaming_bench`.
- `baseline/qwen35_streaming_eval.py` can accept `--datasets streaming_bench`.
- Full video staging can be launched with:

```bash
cd /home/xwu/atomic_skills_for_video
sbatch scripts/stage_streaming_bench.sbatch
```

- Tiny synthetic adapter smoke passed for both root-level and `src/data`
  question files at:

```text
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/streaming_bench_adapter_smoke/streaming_bench_canonical_2files.jsonl
```

Official/HF dataset reference:

```text
https://huggingface.co/datasets/mjuicem/StreamingBench
```

The HF page reports about `4,550` rows and about `203GB` total file size, so
download/staging should be done explicitly on shared storage rather than in this
repo or `/home/xwu`.

## OVO-Bench Notes

Verified OVO-Bench path:

```text
/net/mlfs01/export/users/dpatel/OVO-Bench
```

The local tree contains `chunked_videos/` with `3035` `.mp4` files. The
StreamBridge annotation file in this workspace has `1640` rows:

```text
/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/ovo_bench.json
```

Many rows map directly by `id` to `chunked_videos/{id}.mp4`; for example,
`id=0` maps to `chunked_videos/0.mp4`.

## VideoMME Notes

Verified VideoMME media path:

```text
/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme
```

The local tree contains `videos/` and `subtitle/`. The StreamBridge annotation
file in this workspace has `2700` rows:

```text
/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets/videomme.json
```

The sample row with `videoID=fFjv93ACGo8` has both:

```text
/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme/videos/fFjv93ACGo8.mp4
/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme/subtitle/fFjv93ACGo8.srt
```

## Latest Five-Dataset Probe

Checked on 2026-07-06 with:

```bash
python3 -m dataset_clip_wrapper.cli --dataset <dataset> --limit 1
```

All five primary datasets wrote one canonical example with a real media path:

| Dataset | Example id | Media status |
| --- | --- | --- |
| `video_holmes` | `video_holmes:train:oZ4pa_5R0nY:q1` | Video exists under `Video-Holmes/Benchmark/videos_cropped`. |
| `videomme` | `videomme:001-1` | Video and subtitle exist under Grounded-VideoLLM `videomme`. |
| `ovo_bench` | `ovo_bench:0` | Video exists under OVO-Bench `chunked_videos/0.mp4`. |
| `cg_bench` | `cg_bench:69` | Video exists under local CG-Bench root as `69.mp4`. |
| `vrbench` | `vrbench:TZk_p-q8Fzo:qa1` | Smoke video exists under local VRBench `v001_360p/TZk_p-q8Fzo.mp4`. |
