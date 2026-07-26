"""Small embedding interfaces for the FAISS baseline."""

from __future__ import annotations

import gc
import hashlib
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

DEFAULT_ENCODE_CHUNK_SIZE = 64


def resolve_torch_device(device: str | None = None) -> str:
    """Resolve a concrete torch device string (e.g. cuda:0 / cpu)."""

    import torch

    if device:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class EmbeddingModel:
    name: str
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


@dataclass
class HashingTextEmbedder(EmbeddingModel):
    """Deterministic no-download text embedder for plumbing smoke tests.

    This is not a strong semantic model. It exists so schema, metadata, and FAISS
    mechanics are runnable before choosing the production embedding model.
    """

    dim: int = 384
    name: str = "hashing-text-v0"

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _tokens(text):
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if int.from_bytes(digest[4:], "little") % 2 == 0 else -1.0
                vectors[row, bucket] += sign
            norm = float(np.linalg.norm(vectors[row]))
            if norm > 0:
                vectors[row] /= norm
        return vectors


class Qwen3VLVLLMEmbedder(EmbeddingModel):
    """Qwen3-VL multimodal embedder for text-query to video-frame retrieval."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        *,
        dtype: str = "bfloat16",
        device: str | None = None,
        frames_per_clip: int = 1,
        image_batch_size: int = 16,
        decode_workers: int = 1,
        decode_strategy: str = "seek",
        encode_chunk_size: int = DEFAULT_ENCODE_CHUNK_SIZE,
        clip_encode_mode: str = "image_mean",
        instruction: str = "Represent the input for retrieving relevant video clips for a question.",
        gpu_memory_utilization: float | None = None,
        dim: int = 2048,
    ) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        if clip_encode_mode not in {"image_mean", "video"}:
            raise ValueError("clip_encode_mode must be 'image_mean' or 'video'")
        self.name = model_name
        self.frames_per_clip = frames_per_clip
        self.image_batch_size = image_batch_size
        self.decode_workers = decode_workers
        if decode_strategy not in {"seek", "scan"}:
            raise ValueError("decode_strategy must be 'seek' or 'scan'")
        self.decode_strategy = decode_strategy
        if encode_chunk_size <= 0:
            raise ValueError("encode_chunk_size must be positive")
        self.encode_chunk_size = encode_chunk_size
        self.clip_encode_mode = clip_encode_mode
        self.instruction = instruction
        self.dim = dim
        self.device = resolve_torch_device(device)
        # SentenceTransformer does not implement vLLM's gpu_memory_utilization.
        # Keep the flag for CLI compatibility and surface it for dual-GPU runners.
        self.gpu_memory_utilization = gpu_memory_utilization
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) and hasattr(torch, dtype) else dtype
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": torch_dtype},
            device=self.device,
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            prompt=self.instruction,
            batch_size=self.image_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        self.dim = int(vectors.shape[1])
        return vectors

    def encode_clip_records(self, clips: list[Any], *, skip_failed_clips: bool = False) -> np.ndarray:
        """Decode+embed clips in chunks so host RAM does not hold the full shard."""

        vectors, _kept = self.encode_clip_records_with_keep(clips, skip_failed_clips=skip_failed_clips)
        return vectors

    def encode_clip_records_with_keep(
        self, clips: list[Any], *, skip_failed_clips: bool = True
    ) -> tuple[np.ndarray, list[int]]:
        def _encode_chunk_image_mean(image_groups: list[list[Image.Image]]) -> np.ndarray:
            flat_images = []
            spans: list[tuple[int, int]] = []
            for images in image_groups:
                start = len(flat_images)
                flat_images.extend({"image": image} for image in images)
                spans.append((start, len(flat_images)))
            frame_vectors = self.model.encode(
                flat_images,
                prompt=self.instruction,
                batch_size=self.image_batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            frame_vectors = np.asarray(frame_vectors, dtype=np.float32)
            clip_vectors = [frame_vectors[start:end].mean(axis=0) for start, end in spans]
            return cosine_normalize(np.asarray(clip_vectors, dtype=np.float32))

        def _encode_chunk_video(image_groups: list[list[Image.Image]]) -> np.ndarray:
            # One native video input per clip: model does temporal pooling internally.
            documents = [{"video": list(images)} for images in image_groups]
            vectors = self.model.encode(
                documents,
                prompt=self.instruction,
                batch_size=max(1, min(self.image_batch_size, 8)),
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return cosine_normalize(np.asarray(vectors, dtype=np.float32))

        encode_fn = _encode_chunk_video if self.clip_encode_mode == "video" else _encode_chunk_image_mean
        vectors, kept = encode_clip_records_in_chunks(
            clips,
            frames_per_clip=self.frames_per_clip,
            decode_workers=self.decode_workers,
            decode_strategy=self.decode_strategy,
            encode_chunk_size=self.encode_chunk_size,
            encode_image_groups=encode_fn,
            skip_failed_clips=skip_failed_clips,
        )
        if vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors, kept


class CLIPVideoTextEmbedder(EmbeddingModel):
    """Cross-modal CLIP embedder for text queries and video clip records.

    Video clips are represented by one or more sampled frames, averaged in CLIP
    image-embedding space. Questions are encoded in CLIP text space, so FAISS can
    do cross-modal retrieval.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        device: str | None = None,
        frames_per_clip: int = 1,
        image_batch_size: int = 64,
        decode_workers: int = 1,
        decode_strategy: str = "seek",
        encode_chunk_size: int = DEFAULT_ENCODE_CHUNK_SIZE,
    ) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.name = model_name
        self.frames_per_clip = frames_per_clip
        self.image_batch_size = image_batch_size
        self.decode_workers = decode_workers
        if decode_strategy not in {"seek", "scan"}:
            raise ValueError("decode_strategy must be 'seek' or 'scan'")
        self.decode_strategy = decode_strategy
        if encode_chunk_size <= 0:
            raise ValueError("encode_chunk_size must be positive")
        self.encode_chunk_size = encode_chunk_size
        self.device = resolve_torch_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        self.dim = int(self.model.config.projection_dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        import torch

        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            vectors = self.model.get_text_features(**inputs)
        vectors = _feature_tensor(vectors)
        return cosine_normalize(vectors.detach().cpu().numpy())

    def encode_clip_records(self, clips: list[Any], *, skip_failed_clips: bool = False) -> np.ndarray:
        """Encode video clip records with chunked decode and batched CLIP calls."""

        vectors, _kept = self.encode_clip_records_with_keep(clips, skip_failed_clips=skip_failed_clips)
        return vectors

    def encode_clip_records_with_keep(
        self, clips: list[Any], *, skip_failed_clips: bool = True
    ) -> tuple[np.ndarray, list[int]]:
        def _encode_chunk(image_groups: list[list[Image.Image]]) -> np.ndarray:
            flat_images: list[Image.Image] = []
            spans: list[tuple[int, int]] = []
            for images in image_groups:
                start = len(flat_images)
                flat_images.extend(images)
                spans.append((start, len(flat_images)))
            image_vectors = self._encode_images(flat_images)
            clip_vectors = [image_vectors[start:end].mean(axis=0) for start, end in spans]
            return cosine_normalize(np.asarray(clip_vectors, dtype=np.float32))

        vectors, kept = encode_clip_records_in_chunks(
            clips,
            frames_per_clip=self.frames_per_clip,
            decode_workers=self.decode_workers,
            decode_strategy=self.decode_strategy,
            encode_chunk_size=self.encode_chunk_size,
            encode_image_groups=_encode_chunk,
            skip_failed_clips=skip_failed_clips,
        )
        if vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors, kept

    def _encode_images(self, images: list[Image.Image]) -> np.ndarray:
        import torch

        vectors = []
        for start in range(0, len(images), self.image_batch_size):
            batch = images[start : start + self.image_batch_size]
            inputs = self.processor(images=batch, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_vectors = self.model.get_image_features(**inputs)
            image_vectors = _feature_tensor(image_vectors)
            vectors.append(image_vectors.detach().cpu().numpy())
        return np.concatenate(vectors, axis=0).astype(np.float32)


def encode_clip_records_in_chunks(
    clips: list[Any],
    *,
    frames_per_clip: int,
    decode_workers: int,
    decode_strategy: str,
    encode_chunk_size: int,
    encode_image_groups: Callable[[list[list[Image.Image]]], np.ndarray],
    skip_failed_clips: bool = False,
) -> tuple[np.ndarray, list[int]]:
    """Decode and embed clips in fixed-size chunks to bound host RAM.

    Returns ``(vectors, kept_indices)``. When ``skip_failed_clips`` is True, each
    chunk is batched first; only on chunk failure do we fall back to per-clip
    encode so one bad video cannot abort the whole example.
    """

    if encode_chunk_size <= 0:
        raise ValueError("encode_chunk_size must be positive")
    if not clips:
        return np.zeros((0, 0), dtype=np.float32), []

    outputs: list[np.ndarray] = []
    kept_indices: list[int] = []
    total = len(clips)

    def _encode_one(clip: Any, abs_index: int) -> bool:
        try:
            image_groups = sample_image_groups_for_clips(
                [clip],
                frames_per_clip=frames_per_clip,
                decode_workers=1,
                decode_strategy=decode_strategy,
            )
            if not image_groups:
                return False
            chunk_vectors = encode_image_groups(image_groups)
        except Exception:
            return False
        outputs.append(np.asarray(chunk_vectors, dtype=np.float32))
        kept_indices.append(abs_index)
        return True

    for start in range(0, total, encode_chunk_size):
        chunk = list(clips[start : start + encode_chunk_size])
        absolute = list(range(start, start + len(chunk)))
        try:
            image_groups = sample_image_groups_for_clips(
                chunk,
                frames_per_clip=frames_per_clip,
                decode_workers=decode_workers,
                decode_strategy=decode_strategy,
            )
            chunk_vectors = encode_image_groups(image_groups)
            if len(chunk_vectors) != len(absolute):
                raise RuntimeError(
                    f"encode produced {len(chunk_vectors)} vectors for {len(absolute)} clips"
                )
            outputs.append(np.asarray(chunk_vectors, dtype=np.float32))
            kept_indices.extend(absolute)
            del image_groups, chunk_vectors
        except Exception:
            if not skip_failed_clips:
                raise
            for clip, abs_index in zip(chunk, absolute):
                _encode_one(clip, abs_index)
        gc.collect()
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32), []
    return np.concatenate(outputs, axis=0), kept_indices


class ClipEmbeddingCache:
    """In-process cache keyed by video path + clip span + sampling config."""

    def __init__(self) -> None:
        self._store: dict[tuple[Any, ...], np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key_for(
        clip: Any,
        *,
        frames_per_clip: int,
        embedding_backend: str,
        embedding_model: str,
        clip_encode_mode: str = "image_mean",
    ) -> tuple[Any, ...]:
        return (
            str(getattr(clip, "video_path", "")),
            round(float(getattr(clip, "start_s", 0.0)), 3),
            round(float(getattr(clip, "end_s", 0.0)), 3),
            int(frames_per_clip),
            str(embedding_backend),
            str(embedding_model),
            str(clip_encode_mode),
        )

    def get(self, key: tuple[Any, ...]) -> np.ndarray | None:
        value = self._store.get(key)
        if value is None:
            self.misses += 1
            return None
        self.hits += 1
        return value

    def put(self, key: tuple[Any, ...], vector: np.ndarray) -> None:
        self._store[key] = np.asarray(vector, dtype=np.float32)

    def stats(self) -> dict[str, int]:
        return {"cache_hits": self.hits, "cache_misses": self.misses, "cache_size": len(self._store)}


def embed_clips_with_cache(
    embedder: Any,
    clips: list[Any],
    *,
    cache: ClipEmbeddingCache,
    embedding_backend: str,
    skip_failed_clips: bool = True,
) -> tuple[np.ndarray, list[int], dict[str, int]]:
    """Embed clips with cross-example cache; only encode cache misses in batch."""

    if not clips:
        return np.zeros((0, 0), dtype=np.float32), [], cache.stats()

    model_name = str(getattr(embedder, "name", embedding_backend))
    frames_per_clip = int(getattr(embedder, "frames_per_clip", 1))
    clip_encode_mode = str(getattr(embedder, "clip_encode_mode", "image_mean"))
    vectors: list[np.ndarray | None] = [None] * len(clips)
    miss_clips: list[Any] = []
    miss_indices: list[int] = []

    for index, clip in enumerate(clips):
        key = cache.key_for(
            clip,
            frames_per_clip=frames_per_clip,
            embedding_backend=embedding_backend,
            embedding_model=model_name,
            clip_encode_mode=clip_encode_mode,
        )
        cached = cache.get(key)
        if cached is not None:
            vectors[index] = cached
        else:
            miss_clips.append(clip)
            miss_indices.append(index)

    if miss_clips:
        encoded, kept = embedder.encode_clip_records_with_keep(
            miss_clips,
            skip_failed_clips=skip_failed_clips,
        )
        for local_kept, vector in zip(kept, encoded):
            abs_index = miss_indices[local_kept]
            clip = clips[abs_index]
            key = cache.key_for(
                clip,
                frames_per_clip=frames_per_clip,
                embedding_backend=embedding_backend,
                embedding_model=model_name,
                clip_encode_mode=clip_encode_mode,
            )
            vec = np.asarray(vector, dtype=np.float32)
            cache.put(key, vec)
            vectors[abs_index] = vec

    kept_indices = [index for index, vector in enumerate(vectors) if vector is not None]
    if not kept_indices:
        return np.zeros((0, 0), dtype=np.float32), [], cache.stats()
    stacked = np.stack([vectors[index] for index in kept_indices], axis=0).astype(np.float32)
    return stacked, kept_indices, cache.stats()


def sample_clip_frames(video_path: str, start_s: float, end_s: float, *, frames_per_clip: int) -> list[Image.Image]:
    import cv2  # type: ignore

    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    if frames_per_clip == 1:
        timestamps = [(start_s + end_s) / 2.0]
    else:
        span = max(end_s - start_s, 0.001)
        timestamps = [start_s + span * i / (frames_per_clip - 1) for i in range(frames_per_clip)]
    images = []
    try:
        for timestamp_s in timestamps:
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame))
    finally:
        cap.release()
    if not images:
        raise RuntimeError(f"could not sample frames from {video_path} [{start_s}, {end_s}]")
    return images


def sample_image_groups_for_clips(
    clips: list[Any],
    *,
    frames_per_clip: int,
    decode_workers: int,
    decode_strategy: str,
) -> list[list[Image.Image]]:
    grouped = _group_clip_indices_by_video(clips)
    image_groups: list[list[Image.Image] | None] = [None] * len(clips)

    video_items = list(grouped.items())
    if decode_workers > 1 and len(video_items) > 1:
        # Bound in-flight decodes to decode_workers; do not submit the whole chunk at once.
        with ThreadPoolExecutor(max_workers=decode_workers) as pool:
            for batch_start in range(0, len(video_items), decode_workers):
                batch = video_items[batch_start : batch_start + decode_workers]
                futures = [
                    pool.submit(
                        sample_clip_frames_for_video,
                        video_path,
                        [
                            (clip_index, clips[clip_index].start_s, clips[clip_index].end_s)
                            for clip_index in clip_indices
                        ],
                        frames_per_clip=frames_per_clip,
                        strategy=decode_strategy,
                    )
                    for video_path, clip_indices in batch
                ]
                for future in as_completed(futures):
                    for clip_index, images in future.result().items():
                        image_groups[clip_index] = images
    else:
        for video_path, clip_indices in video_items:
            sampled = sample_clip_frames_for_video(
                video_path,
                [(clip_index, clips[clip_index].start_s, clips[clip_index].end_s) for clip_index in clip_indices],
                frames_per_clip=frames_per_clip,
                strategy=decode_strategy,
            )
            for clip_index, images in sampled.items():
                image_groups[clip_index] = images

    if any(images is None for images in image_groups):
        raise RuntimeError("internal error: missing sampled frames for a clip")
    return [images for images in image_groups if images is not None]


def sample_clip_frames_for_video(
    video_path: str,
    clips: list[tuple[int, float, float]],
    *,
    frames_per_clip: int,
    strategy: str = "seek",
) -> dict[int, list[Image.Image]]:
    """Sample frames for many clip spans while opening the video only once."""

    import cv2  # type: ignore

    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")

    try:
        requests = _frame_requests(clips, frames_per_clip=frames_per_clip)
        if strategy == "seek":
            sampled = _sample_sorted_requests_with_seek(cap, requests)
        elif strategy == "scan":
            sampled = _sample_sorted_requests_with_scan(cap, requests)
        else:
            raise ValueError("strategy must be 'seek' or 'scan'")
    finally:
        cap.release()

    for clip_index, start_s, end_s in clips:
        if not sampled.get(clip_index):
            raise RuntimeError(f"could not sample frames from {video_path} [{start_s}, {end_s}]")
    return sampled


def _frame_requests(
    clips: list[tuple[int, float, float]],
    *,
    frames_per_clip: int,
) -> list[tuple[int, float]]:
    requests = []
    for clip_index, start_s, end_s in clips:
        for timestamp_s in _clip_timestamps(start_s, end_s, frames_per_clip=frames_per_clip):
            requests.append((clip_index, timestamp_s))
    return sorted(requests, key=lambda item: item[1])


def _sample_sorted_requests_with_seek(cap: Any, requests: list[tuple[int, float]]) -> dict[int, list[Image.Image]]:
    import cv2  # type: ignore

    sampled: dict[int, list[Image.Image]] = {}
    for clip_index, timestamp_s in requests:
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, timestamp_s) * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sampled.setdefault(clip_index, []).append(Image.fromarray(frame))
    return sampled


def _sample_sorted_requests_with_scan(cap: Any, requests: list[tuple[int, float]]) -> dict[int, list[Image.Image]]:
    import cv2  # type: ignore

    sampled: dict[int, list[Image.Image]] = {}
    if not requests:
        return sampled

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_index = 0
    request_index = 0
    while request_index < len(requests):
        ok, frame = cap.read()
        if not ok:
            break
        timestamp_s = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
        if timestamp_s <= 0.0 and fps > 0.0:
            timestamp_s = frame_index / fps
        if timestamp_s >= max(0.0, requests[request_index][1]):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame)
            while request_index < len(requests) and timestamp_s >= max(0.0, requests[request_index][1]):
                clip_index, _ = requests[request_index]
                sampled.setdefault(clip_index, []).append(image.copy())
                request_index += 1
        frame_index += 1
    return sampled


def _clip_timestamps(start_s: float, end_s: float, *, frames_per_clip: int) -> list[float]:
    if frames_per_clip <= 0:
        raise ValueError("frames_per_clip must be positive")
    if frames_per_clip == 1:
        return [(start_s + end_s) / 2.0]
    span = max(end_s - start_s, 0.001)
    return [start_s + span * i / (frames_per_clip - 1) for i in range(frames_per_clip)]


def _group_clip_indices_by_video(clips: list[Any]) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, clip in enumerate(clips):
        grouped[str(clip.video_path)].append(index)
    return dict(grouped)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _feature_tensor(features: Any) -> Any:
    for attr in ("image_embeds", "text_embeds", "pooler_output"):
        value = getattr(features, attr, None)
        if value is not None:
            return value
    return features


def cosine_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
