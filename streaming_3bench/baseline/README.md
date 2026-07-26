# Baseline Video QA Memory

This folder contains the standard baseline path for streaming video QA:

```text
dataset_clip_wrapper canonical schema
  -> normalized video QA examples
  -> clip records
  -> clip embeddings
  -> FAISS retrieval
  -> local Qwen3.5-9B answer generation
```

The baseline is intentionally separate from `dataset_clip_wrapper/`. The wrapper
owns dataset adapters and canonical examples; this folder owns a simple,
comparable retrieval baseline over those canonical examples.

## 中文更新：Streaming Clip 加速

我们已经把 baseline 的视频 clip 预处理改成更适合 streaming video QA 的方式：

- 仍然以 wrapper 产生的 `VideoClipRecord` 作为 retrieval 单位，不改变标准 schema。
- 同一个视频下的 clips 会按 `video_path` 分组，避免每个 clip 都重新打开同一个视频文件。
- `--decode-strategy scan` 会按时间顺序读取视频流，读到目标 timestamp 时采帧，比反复 random seek 更符合 streaming 设置。
- `--decode-workers` 用来并行处理不同视频文件；`--image-batch-size` 用来把采到的 frames 批量送进 CLIP，在 A6000 上减少小 batch overhead。
- 每个 clip 的 embedding 仍然会写回 `clips.jsonl` 的 `embedding` 字段，FAISS index 和 schema records 都保留，方便后续复用和复现实验。

当前推荐参数：

```bash
--embedding-backend clip \
--clip-model openai/clip-vit-base-patch32 \
--frames-per-clip 4 \
--image-batch-size 64 \
--decode-workers 4 \
--decode-strategy scan
```

5 OVO-Bench + 5 VideoMME 的 smoke test 对比：

```text
原始逐 clip seek + 小 batch: 00:07:46
按视频分组 seek + CLIP batch: 00:05:03
按视频分组 scan + CLIP batch: 00:01:39
```

最新验证输出：

```text
job_id: 290918
state: COMPLETED
node: cipr-gpu16
indexed clips: 420
OVO-Bench clips: 325
VideoMME clips: 95
embedding_dim: 512
output: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames_scan
```

## 中文更新：多单卡 Sharded Eval

Qwen3.5-9B 单卡可以跑 OVO-Bench 和 VideoMME。为了更快跑完整 streaming
evaluation，我们采用 multi single-GPU shards：

```text
dataset examples
  -> shard by row_id % num_shards
  -> each shard runs one Slurm array task on one GPU
  -> each shard writes records.jsonl + metrics_summary.json
  -> merge_eval_shards.py merges records and recomputes metrics
```

推荐先用 4 或 8 个 shards。当前集群有可用 A6000 资源时，完整 OVO-Bench +
VideoMME 粗略估计：

```text
1 GPU: 5-6 hours for direct video QA, longer if top-k RAG sends more clips
4 GPUs: 1.3-1.6 hours for direct video QA
8 GPUs: 40-60 minutes for direct video QA
```

提交 4 个单卡 shards：

```bash
cd /home/xwu/atomic_skills_for_video
mkdir -p /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/logs

NUM_SHARDS=4 \
ANSWER_MODE=json_rationale \
LIMIT_PER_DATASET=-1 \
sbatch --array=0-3 baseline/slurm_qwen35_sharded_eval.sbatch
```

提交 8 个单卡 shards：

```bash
cd /home/xwu/atomic_skills_for_video

NUM_SHARDS=8 \
ANSWER_MODE=json_rationale \
LIMIT_PER_DATASET=-1 \
sbatch --array=0-7 baseline/slurm_qwen35_sharded_eval.sbatch
```

`LIMIT_PER_DATASET=-1` 表示全量。做 smoke test 时可以改成：

```bash
NUM_SHARDS=2 ANSWER_MODE=json_rationale LIMIT_PER_DATASET=20 \
sbatch --array=0-1 baseline/slurm_qwen35_sharded_eval.sbatch
```

合并 shards：

```bash
PROJECT=/mnt/is_data/xwu/video_skills
VENV=$PROJECT/code/vllm_qwen_cu124_venv
JOB_ID=<array_job_id>

$VENV/bin/python -m baseline.merge_eval_shards \
  --shards-root $PROJECT/outputs/atomic_skills_for_video/qwen35_streaming_eval_sharded/$JOB_ID \
  --output-dir $PROJECT/outputs/atomic_skills_for_video/qwen35_streaming_eval_sharded/${JOB_ID}_merged
```

Current full 4-shard run:

```text
job_id: 290921
state at launch check: RUNNING
shards: 0-3
nodes: cipr-gpu16, cipr-gpu17
datasets: ovo_bench videomme
answer_mode: json_rationale
limit_per_dataset: -1 (full)
output root: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval_sharded/290921
```

### Metrics and Records

Primary metrics:

- `accuracy`: correct multiple-choice label among successful examples.
- `accuracy_on_parsed`: correct label among examples where the output was parsed.
- `parse_rate`: fraction of successful examples with a parsed A/B/C/D label.
- `avg_generate_s`: average Qwen generation time per example.
- `failed`: preparation/model failures.

For RAG runs, we should also add retrieval metrics:

- `retrieval_recall@k`: whether retrieved clips cover the gold/time-anchor region.
- `retrieval_mrr`: rank of the first relevant retrieved clip.
- `no_future_leak_rate`: retrieved clips must satisfy `clip.end_s <= visible_until_s`.

Each `records.jsonl` row stores:

```text
dataset
example_id
video_id
visible_until_s
media_records
prompt
response
prediction_label
gold_label
correct
evidence_summary
timing_s
```

### CoT / Rationale Choice

We should not use full free-form CoT as the default metric path. It increases
generation length, can reduce parse stability, and is hard to verify. The
default sharded eval instead uses:

```text
ANSWER_MODE=json_rationale
```

This asks Qwen3.5-9B for valid JSON:

```json
{"answer_label": "C", "evidence_summary": "short grounded sentence"}
```

So we still record a useful rationale/evidence note for debugging and later
analysis, while the main metric remains the parsed answer label. If we truly
want to probe Qwen thinking behavior, `baseline/qwen35_streaming_eval.py` also
has `--enable-thinking`, but that should be a separate ablation rather than the
main benchmark run.

## Local Model Boundary

Current evaluation should use the locally deployed Qwen3.5-9B model:

```text
/mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
```

This baseline should not call OpenRouter or any hosted API for model inference.
The local A6000-compatible environment is:

```text
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
```

## Dispider Baseline

Dispider should be treated as an end-to-end streaming VideoLLM baseline, not as
part of the atomic-skill graph controller. Its result is comparable at the
answer-label level, while our method is additionally evaluated on evidence refs,
skill rollouts, verifier acceptance, and repair traces.

Keep the external repo, environment, checkpoints, and generated prefix videos
under `/mnt/is_data/xwu/video_skills`, not under `/home/xwu`:

```bash
cd /mnt/is_data/xwu/video_skills/code
git clone https://github.com/Mark12Ding/Dispider.git
```

Create a Dispider-compatible environment following the upstream repo, for
example at:

```text
/mnt/is_data/xwu/video_skills/code/dispider_venv
```

Run a small smoke test once `MODEL` points to the Dispider checkpoint:

```bash
cd /home/xwu/atomic_skills_for_video

PROJECT=/mnt/is_data/xwu/video_skills \
VENV=/mnt/is_data/xwu/video_skills/code/dispider_venv \
DISPIDER_REPO=/mnt/is_data/xwu/video_skills/code/Dispider \
MODEL=/mnt/is_data/xwu/video_skills/data/models/dispider/<checkpoint> \
DATASETS="ovo_bench videomme" \
LIMIT_PER_DATASET=5 \
sbatch --array=0-0 baseline/slurm_dispider_streaming_eval.sbatch
```

`baseline/dispider_streaming_eval.py` writes the same top-level artifacts as the
Qwen baseline:

```text
run_config.json
canonical_schemas.jsonl
records.jsonl
metrics_summary.json
visible_prefix_videos/
```

The runner materializes one visible-prefix video per example before calling
Dispider. This is important because the upstream inference API accepts a single
video path; clipping first prevents future-frame leakage beyond
`visible_until_s`.

## Standard Records

`schemas.py` defines three JSONL-friendly records:

- `VideoQAExample`: one question over one video, including streaming visibility.
- `VideoClipRecord`: one wrapper-derived video clip span.
- `RetrievedClip`: one retrieval result with score and source clip metadata.

The key idea is that all retrieval, RAG, and Qwen calls refer back to stable
clip IDs and source spans:

```json
{
  "clip_id": "clip:video_id:fine:0007",
  "video_path": "/path/to/video.mp4",
  "start_s": 21.0,
  "end_s": 25.0,
  "visible_until_s": 60.0
}
```

## FAISS Plan

`faiss_store.py` is a small wrapper around FAISS. It stores:

- `index.faiss`: vector index
- `clips.jsonl`: clip metadata aligned to FAISS row IDs, including the clip
  embedding in `embedding`
- `manifest.json`: model/dimension/count metadata

Two embedding backends are currently available:

- `hashing_text`: deterministic text hashing over clip schema text. This is a
  plumbing smoke backend, not a strong semantic retriever.
- `clip`: cross-modal CLIP retrieval. Video clips are embedded by sampling one
  or more frames from the wrapper clip span and averaging CLIP image embeddings.
  Questions are embedded with the matching CLIP text encoder.

The recommended smoke/default setting is currently:

```text
--embedding-backend clip
--clip-model openai/clip-vit-base-patch32
--frames-per-clip 4
--image-batch-size 64
--decode-workers 4
--decode-strategy scan
```

This keeps the retrieval unit as a wrapper video clip while representing each
clip with multiple sampled frames rather than a single still image.

For speed, CLIP video embedding now groups records by `video_path`, opens each
video once, samples all wrapper clip frames for that video, and then runs CLIP
image encoding in batches. `--decode-workers` parallelizes independent videos;
`--image-batch-size` controls GPU batch size for frame embedding.
`--decode-strategy scan` reads each video in timestamp order and captures frames
as the stream advances, which matches the streaming-video setting better than
repeated random seeks. `--decode-strategy seek` remains available for short
local files where direct timestamp seeking is faster.

At query time, the question text is embedded with the same model and searched
against the FAISS index. The query CLI hides full vectors by default to keep
terminal output readable; pass `--include-embeddings` to print the question and
clip embedding arrays.

## Example Commands

Build a clip index from canonical examples:

```bash
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke \
  --embedding-backend hashing_text
```

Build a cross-modal CLIP clip index with 4 sampled frames per wrapper clip:

```bash
export HF_HOME=/mnt/is_data/xwu/video_skills/data/models/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/hub
export TRANSFORMERS_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/transformers

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip \
  --embedding-backend clip \
  --clip-model openai/clip-vit-base-patch32 \
  --frames-per-clip 4 \
  --image-batch-size 64 \
  --decode-workers 4 \
  --decode-strategy scan
```

Build a Qwen3-VL-Embedding-2B clip index:

```bash
export HF_HOME=/mnt/is_data/xwu/video_skills/data/models/hf_cache
export HUGGINGFACE_HUB_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/hub
export TRANSFORMERS_CACHE=/mnt/is_data/xwu/video_skills/data/models/hf_cache/transformers

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_qwen3_vl_embedding \
  --embedding-backend qwen3_vl \
  --qwen3-vl-model Qwen/Qwen3-VL-Embedding-2B \
  --frames-per-clip 4 \
  --image-batch-size 16 \
  --decode-workers 4 \
  --decode-strategy scan
```

For this baseline, do not use clip captions in the main retrieval path. The
Qwen3-VL embedding model is already a multimodal embedding model: questions are
embedded as text, sampled clip frames are embedded as images, and FAISS compares
them in the shared vector space. Caption-based retrieval should be a separate
ablation:

```bash
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.build_faiss_index \
  --canonical-jsonl /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_qwen3_text_caption \
  --embedding-backend qwen3_text_caption \
  --qwen3-vl-model Qwen/Qwen3-VL-Embedding-2B \
  --clip-text-mode caption_metadata
```

Query the index:

```bash
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.query_faiss_index \
  --index-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke \
  --query "Who did I communicate to when chopping eggplants?" \
  --topk 5
```

FAISS is installed in the current A6000-compatible environment:

```text
/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv
```

Verified smoke outputs:

```text
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke             # hashing_text, 420 clips
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_smoke        # CLIP, 10 clips, 1 frame/clip
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5          # CLIP, 5+5 examples, 1 frame/clip
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames  # CLIP, 5+5 examples, 4 frames/clip
/mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames_scan  # CLIP, 5+5 examples, 4 frames/clip, streaming scan decode
```

Latest 5+5 smoke:

```text
job_id: 290918
state: COMPLETED
elapsed: 00:01:39
node: cipr-gpu16
canonical input: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/qwen35_streaming_eval/290877/canonical_schemas.jsonl
output: /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames_scan
```

Summary:

```text
examples: 5 OVO-Bench + 5 VideoMME
indexed clips: 420
OVO-Bench clips: 325
VideoMME clips: 95
embedding_backend: clip
embedding_model: openai/clip-vit-base-patch32
embedding_dim: 512
frames_per_clip: 4
image_batch_size: 64
decode_workers: 4
decode_strategy: scan
```

The output directory contains:

```text
clips.jsonl                     # VideoClipRecord schema + embedding per clip
index.faiss                     # FAISS index over clip embeddings
manifest.json                   # embedding/index metadata
query_clip_4frames_scan_compact.json # query embedding -> top-k clip retrieval smoke
```

Each `clips.jsonl` row stores future-reference metadata:

```text
row_id
example_id
dataset
video_id
clip_id
video_path
start_s
end_s
granularity
visible_until_s
text
embedding
metadata
```

The retrieved clip rows are ready to become local Qwen3.5-9B direct video
inputs:

```python
{
    "type": "video",
    "video": clip["video_path"],
    "video_start": clip["start_s"],
    "video_end": clip["end_s"],
}
```

## Iterative RAG Memory Query Baseline

`iterative_rag_memory_query.py` adds a text-memory RAG baseline on top of the
same FAISS clip store. For each canonical QA example, it repeatedly queries the
index, filters retrieved clips to the same example/video and visible memory,
deduplicates evidence, then answers from retrieved memory text only.

Smoke test with the hashing-text FAISS index:

```bash
cd /home/xwu/atomic_skills_for_video

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.iterative_rag_memory_query \
  --index-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_smoke \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/iterative_rag_memory_query/smoke \
  --datasets ovo_bench \
  --limit-per-dataset 1 \
  --iterations 2 \
  --per-iteration-top-k 2 \
  --final-top-k 3 \
  --answer-backend heuristic
```

Full local-Qwen text-memory answer generation:

```bash
cd /home/xwu/atomic_skills_for_video

/mnt/is_data/xwu/video_skills/code/vllm_qwen_cu124_venv/bin/python \
  -m baseline.iterative_rag_memory_query \
  --index-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/baseline_faiss/ovo_videomme_clip_5x5_4frames_scan \
  --output-dir /mnt/is_data/xwu/video_skills/outputs/atomic_skills_for_video/iterative_rag_memory_query/qwen_text_smoke \
  --datasets ovo_bench videomme \
  --limit-per-dataset 5 \
  --iterations 3 \
  --per-iteration-top-k 4 \
  --final-top-k 8 \
  --answer-backend local_qwen \
  --model /mnt/is_data/xwu/video_skills/data/models/qwen35_9b/Qwen3.5-9B
```

Outputs:

```text
run_config.json        # index/model/retrieval configuration
records.jsonl          # per-example retrieval trace, memory evidence, prediction
metrics_summary.json   # accuracy, parse rate, failure count, latency
```
