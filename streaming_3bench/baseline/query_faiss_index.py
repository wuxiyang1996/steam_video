#!/usr/bin/env python3
"""Query a baseline FAISS clip index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import CLIPVideoTextEmbedder, HashingTextEmbedder, Qwen3VLVLLMEmbedder
from .faiss_store import FaissClipStore


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--embedding-backend",
        default=None,
        choices=["hashing_text", "clip", "qwen3_vl", "qwen3_text_caption"],
    )
    parser.add_argument("--clip-model", default=None)
    parser.add_argument("--qwen3-vl-model", default=None)
    parser.add_argument("--qwen3-vl-dtype", default="bfloat16")
    parser.add_argument("--qwen3-vl-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the query embedder (e.g. cuda:0).",
    )
    parser.add_argument(
        "--qwen3-instruction",
        default="Represent the input for retrieving relevant video clips for a question.",
    )
    parser.add_argument("--include-embeddings", action="store_true")
    args = parser.parse_args()

    store = FaissClipStore.load(args.index_dir)
    backend = args.embedding_backend or store.manifest.get("embedding_backend") or "hashing_text"
    if backend == "clip":
        embedder = CLIPVideoTextEmbedder(
            model_name=args.clip_model or store.manifest["embedding_model"],
            device=args.device,
        )
    elif backend == "qwen3_vl":
        embedder = Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model or store.manifest["embedding_model"],
            dtype=args.qwen3_vl_dtype,
            device=args.device,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
    elif backend == "qwen3_text_caption":
        embedder = Qwen3VLVLLMEmbedder(
            model_name=args.qwen3_vl_model or store.manifest["embedding_model"],
            dtype=args.qwen3_vl_dtype,
            device=args.device,
            instruction=args.qwen3_instruction,
            gpu_memory_utilization=args.qwen3_vl_gpu_memory_utilization,
        )
    else:
        embedder = HashingTextEmbedder(dim=int(store.manifest["dim"]))
    query_embedding = embedder.encode([args.query])
    results = store.search(query_embedding, topk=args.topk)
    result_payloads = [result.to_dict() for result in results]
    if not args.include_embeddings:
        for result in result_payloads:
            result.get("clip", {}).pop("embedding", None)
    payload = {
        "query": args.query,
        "question_embedding": query_embedding[0].tolist() if args.include_embeddings else None,
        "question_embedding_dim": int(query_embedding.shape[1]),
        "embedding_model": embedder.name,
        "results": result_payloads,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
