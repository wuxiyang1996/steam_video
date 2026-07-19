"""Qwen3-VL-Embedding-2B integration with lazy optional dependencies."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Protocol, Sequence

from .types import DEFAULT_EMBEDDING_DIM, DEFAULT_EMBEDDING_MODEL, EmbeddingRef, MemoryNode


DEFAULT_NODE_PROMPT = (
    "Represent this video event for temporal, entity-state, and explanatory relation retrieval."
)


class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def encode(self, texts: Sequence[str], *, batch_size: int = 8) -> Sequence[Sequence[float]]:
        """Return one normalized embedding for every input text."""


class Qwen3VLEmbeddingProvider:
    """Lazy SentenceTransformers wrapper for Qwen3-VL-Embedding-2B.

    The official 2B embedding checkpoint is multimodal and produces 2048
    dimensional vectors. Phase 1 encodes the semantic event text assembled
    from Video_Skills. Direct video-span encoding can be added without changing
    the graph schema because vectors are referenced through ``EmbeddingRef``.
    """

    model_name = DEFAULT_EMBEDDING_MODEL
    dimension = DEFAULT_EMBEDDING_DIM

    def __init__(
        self,
        *,
        device: str | None = None,
        prompt: str = DEFAULT_NODE_PROMPT,
        model_kwargs: dict[str, object] | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL-Embedding-2B requires sentence-transformers, torch, "
                "transformers>=4.57.0, and qwen-vl-utils>=0.0.14"
            ) from exc

        kwargs = dict(model_kwargs or {})
        if device is not None:
            kwargs["device"] = device
        self._model = SentenceTransformer(self.model_name, **kwargs)
        self.prompt = prompt

    def encode(self, texts: Sequence[str], *, batch_size: int = 8) -> Sequence[Sequence[float]]:
        if not texts:
            return []
        embeddings = self._model.encode(
            list(texts),
            batch_size=batch_size,
            prompt=self.prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        if embeddings.ndim != 2 or int(embeddings.shape[1]) != self.dimension:
            raise ValueError(
                f"{self.model_name} returned shape {embeddings.shape}; "
                f"expected [N, {self.dimension}]"
            )
        return embeddings


def embed_memory_nodes(
    nodes: list[MemoryNode],
    provider: EmbeddingProvider,
    *,
    output_path: str | Path,
    batch_size: int = 8,
) -> list[list[float]]:
    """Encode node text, persist one matrix, and attach row-based references."""
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("persisting embeddings requires numpy") from exc

    texts = [node.text or _fallback_text(node) for node in nodes]
    raw = provider.encode(texts, batch_size=batch_size)
    matrix = np.asarray(raw, dtype=np.float32)
    expected = (len(nodes), provider.dimension)
    if matrix.shape != expected:
        raise ValueError(f"embedding provider returned {matrix.shape}; expected {expected}")

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if len(nodes):
        matrix = matrix / np.clip(norms, 1e-12, None)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, matrix)
    saved_path = destination if destination.suffix == ".npy" else destination.with_suffix(".npy")
    checksum = hashlib.sha256(saved_path.read_bytes()).hexdigest()

    for row_index, node in enumerate(nodes):
        node.embedding_ref = EmbeddingRef(
            path=str(saved_path),
            model=provider.model_name,
            dimension=provider.dimension,
            dtype="float32",
            normalized=True,
            row_index=row_index,
            checksum=checksum,
        )

    manifest_path = saved_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "model": provider.model_name,
                "dimension": provider.dimension,
                "normalized": True,
                "matrix_path": str(saved_path),
                "checksum": checksum,
                "rows": [{"row_index": index, "node_id": node.node_id} for index, node in enumerate(nodes)],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return matrix.tolist()


def _fallback_text(node: MemoryNode) -> str:
    return (
        f"Video event from {node.time_span.start_s:.2f} to "
        f"{node.time_span.end_s:.2f} seconds."
    )
