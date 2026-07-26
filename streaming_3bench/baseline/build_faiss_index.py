#!/usr/bin/env python3
"""Build a FAISS clip index from wrapper canonical examples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import CLIPVideoTextEmbedder, HashingTextEmbedder, Qwen3VLVLLMEmbedder
from .faiss_store import FaissClipStore
from .schemas import canonical_video_key, clip_records_from_canonical


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _keep_embedded_clips(clips: list, kept: list[int]) -> tuple[list, int]:
    if len(kept) == len(clips):
        return clips, 0
    skipped = len(clips) - len(kept)
    kept_clips = [clips[i] for i in kept]
    for row_id, clip in enumerate(kept_clips):
        object.__setattr__(clip, "row_id", row_id)
    return kept_clips, skipped


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--videomme-observation-end-s", type=float, default=60.0)
    parser.add_argument(
        "--index-granularity",
        default="video",
        choices=["video", "example"],
        help="video=one shared ref per unique video; example=one ref per QA (legacy).",
    )
    parser.add_argument(
        "--embedding-backend",
        default="hashing_text",
        choices=["hashing_text", "clip", "qwen3_vl", "qwen3_text_caption"],
    )
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--qwen3-vl-model", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--qwen3-vl-dtype", default="bfloat16")
    parser.add_argument("--qwen3-vl-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the embedding model (e.g. cuda:0). Defaults to cuda:0/cpu.",
    )
    parser.add_argument(
        "--qwen3-instruction",
        default="Represent the input for retrieving relevant video clips for a question.",
    )
    parser.add_argument(
        "--clip-text-mode",
        default=None,
        choices=["metadata_question", "metadata", "caption", "caption_metadata"],
        help="Text stored on VideoClipRecord; per-video defaults to metadata/caption_metadata.",
    )
    parser.add_argument("--frames-per-clip", type=int, default=1)
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=64,
        help="Image batch size when using visual embedding backends.",
    )
    parser.add_argument(
        "--decode-workers",
        type=int,
        default=1,
        help="Number of per-video decode workers when using visual embedding backends.",
    )
    parser.add_argument(
        "--decode-strategy",
        default="seek",
        choices=["seek", "scan"],
        help="Use timestamp seeks or one-pass streaming scan per video for CLIP frame sampling.",
    )
    parser.add_argument(
        "--encode-chunk-size",
        type=int,
        default=64,
        help="Max clips decoded+embedded at once. Bounds host RAM for visual backends.",
    )
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument(
        "--store-embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Store clip embeddings directly in clips.jsonl schema records.",
    )
    parser.add_argument(
        "--skip-failed-clips",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip clips whose frames cannot be decoded/sampled instead of aborting the shard.",
    )
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.encode_chunk_size <= 0:
        parser.error("--encode-chunk-size must be positive")

    text_mode = args.clip_text_mode
    if text_mode is None:
        if args.embedding_backend == "hashing_text":
            text_mode = "metadata" if args.index_granularity == "video" else "metadata_question"
        else:
            text_mode = "caption_metadata"

    seen_examples = 0
    if args.index_granularity == "video":
        best: dict[str, tuple[tuple[int, float], dict]] = {}
        for example in iter_jsonl(args.canonical_jsonl):
            seen_examples += 1
            key = canonical_video_key(example)
            clips_meta = ((example.get("video") or {}).get("derived_clips") or [])
            max_end = 0.0
            for clip in clips_meta:
                span = clip.get("source_span") or {}
                try:
                    max_end = max(max_end, float(span.get("end_s", 0.0) or 0.0))
                except (TypeError, ValueError):
                    continue
            score = (len(clips_meta), max_end)
            prev = best.get(key)
            if prev is None or score > prev[0]:
                best[key] = (score, example)
        units = sorted((key, payload[1]) for key, payload in best.items())
        apply_visible_cutoff = False
        bind_example_id = False
    else:
        units = []
        for example_index, example in enumerate(iter_jsonl(args.canonical_jsonl)):
            seen_examples += 1
            units.append((str(example_index), example))
        apply_visible_cutoff = True
        bind_example_id = True

    clips = []
    indexed_units = 0
    for unit_index, (_key, example) in enumerate(units):
        if unit_index % args.num_shards != args.shard_index:
            continue
        indexed_units += 1
        clips.extend(
            clip_records_from_canonical(
                example,
                start_row_id=len(clips),
                default_videomme_cutoff_s=args.videomme_observation_end_s,
                text_mode=text_mode,
                apply_visible_cutoff=apply_visible_cutoff,
                bind_example_id=bind_example_id,
            )
        )
        if args.max_clips is not None and len(clips) >= args.max_clips:
            clips = clips[: args.max_clips]
            break

    candidate_clips = len(clips)
    skipped_clips = 0
    if args.embedding_backend == "clip":
        embedder = CLIPVideoTextEmbedder(
            model_name=args.clip_model,
            device=args.device,
            frames_per_clip=args.frames_per_clip,
            image_batch_size=args.image_batch_size,
            decode_workers=args.decode_workers,
            decode_strategy=args.decode_strategy,
            encode_chunk_size=args.encode_chunk_size,
        )
        embeddings, kept = embedder.encode_clip_records_with_keep(
            clips, skip_failed_clips=args.skip_failed_clips
        )
        clips, skipped_clips = _keep_embedded_clips(clips, kept)
    elif args.embedding_backend == "qwen3_vl":
        embedder = Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model,
            dtype=args.qwen3_vl_dtype,
            device=args.device,
            frames_per_clip=args.frames_per_clip,
            image_batch_size=args.image_batch_size,
            decode_workers=args.decode_workers,
            decode_strategy=args.decode_strategy,
            encode_chunk_size=args.encode_chunk_size,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
        embeddings, kept = embedder.encode_clip_records_with_keep(
            clips, skip_failed_clips=args.skip_failed_clips
        )
        clips, skipped_clips = _keep_embedded_clips(clips, kept)
    elif args.embedding_backend == "qwen3_text_caption":
        embedder = Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model,
            dtype=args.qwen3_vl_dtype,
            device=args.device,
            image_batch_size=args.image_batch_size,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
        embeddings = embedder.encode([clip.text for clip in clips])
    else:
        embedder = HashingTextEmbedder(dim=args.hash_dim)
        embeddings = embedder.encode([clip.text for clip in clips])

    if not clips or getattr(embeddings, "size", 0) == 0:
        raise RuntimeError(
            f"no clips embedded for shard {args.shard_index}/{args.num_shards} "
            f"(candidates={candidate_clips}, skipped={skipped_clips})"
        )
    if args.store_embeddings:
        for clip, embedding in zip(clips, embeddings.tolist()):
            object.__setattr__(clip, "embedding", embedding)
    store = FaissClipStore.build(
        embeddings,
        clips,
        embedding_model=embedder.name,
        embedding_backend=args.embedding_backend,
    )
    store.manifest["index_granularity"] = args.index_granularity
    store.save(args.output_dir)
    print(
        json.dumps(
            {
                "clips": len(clips),
                "candidate_clips": candidate_clips,
                "skipped_clips": skipped_clips,
                "seen_examples": seen_examples,
                "unique_units": len(units),
                "indexed_units": indexed_units,
                "index_granularity": args.index_granularity,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "dim": embedder.dim,
                "embedding_backend": args.embedding_backend,
                "clip_text_mode": text_mode,
                "device": getattr(embedder, "device", None),
                "frames_per_clip": args.frames_per_clip if args.embedding_backend in {"clip", "qwen3_vl"} else None,
                "image_batch_size": args.image_batch_size if args.embedding_backend in {"clip", "qwen3_vl"} else None,
                "decode_workers": args.decode_workers if args.embedding_backend in {"clip", "qwen3_vl"} else None,
                "decode_strategy": args.decode_strategy if args.embedding_backend in {"clip", "qwen3_vl"} else None,
                "encode_chunk_size": args.encode_chunk_size if args.embedding_backend in {"clip", "qwen3_vl"} else None,
                "skip_failed_clips": args.skip_failed_clips,
                "store_embeddings": args.store_embeddings,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
