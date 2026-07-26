#!/usr/bin/env python3
"""Merge sharded FAISS clip indexes into one row-aligned store."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .faiss_store import FaissClipStore, _import_faiss
from .schemas import VideoClipRecord


def load_embeddings(index_path: Path) -> np.ndarray:
    faiss = _import_faiss()
    index = faiss.read_index(str(index_path))
    if not hasattr(index, "reconstruct_n"):
        raise RuntimeError(f"FAISS index does not support reconstruct_n: {index_path}")
    vectors = index.reconstruct_n(0, index.ntotal)
    return np.asarray(vectors, dtype=np.float32)


def load_clips(clips_path: Path) -> list[VideoClipRecord]:
    clips = []
    with clips_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                clips.append(VideoClipRecord(**json.loads(line)))
    return clips


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_vectors = []
    all_clips = []
    manifests = []
    for shard_dir in args.shard_dirs:
        manifest = json.loads((shard_dir / "manifest.json").read_text(encoding="utf-8"))
        vectors = load_embeddings(shard_dir / "index.faiss")
        clips = load_clips(shard_dir / "clips.jsonl")
        if vectors.shape[0] != len(clips):
            raise ValueError(f"{shard_dir}: vectors {vectors.shape[0]} != clips {len(clips)}")
        all_vectors.append(vectors)
        all_clips.extend(clips)
        manifests.append(manifest)

    if not all_vectors:
        raise ValueError("no shard indexes provided")
    embedding_model = manifests[0].get("embedding_model")
    embedding_backend = manifests[0].get("embedding_backend")
    for manifest in manifests[1:]:
        if manifest.get("embedding_model") != embedding_model:
            raise ValueError("shard embedding_model mismatch")
        if manifest.get("embedding_backend") != embedding_backend:
            raise ValueError("shard embedding_backend mismatch")

    embeddings = np.concatenate(all_vectors, axis=0).astype(np.float32)
    reset_clips = []
    for row_id, clip in enumerate(all_clips):
        object.__setattr__(clip, "row_id", row_id)
        object.__setattr__(clip, "embedding", None)
        reset_clips.append(clip)

    store = FaissClipStore.build(
        embeddings,
        reset_clips,
        embedding_model=str(embedding_model),
        embedding_backend=embedding_backend,
    )
    store.manifest["source_shards"] = [str(path) for path in args.shard_dirs]
    store.save(args.output_dir)
    print(
        json.dumps(
            {
                "shards": len(args.shard_dirs),
                "clips": len(reset_clips),
                "dim": int(embeddings.shape[1]),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
