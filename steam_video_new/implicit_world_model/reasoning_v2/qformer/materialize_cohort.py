"""Materialize independent caption/entity/visual/time slots from frozen L1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable, Sequence

import numpy as np

from .contracts import SLOT_NAMES, FeatureRow
from .feature_store import FourSlotFeatureStore, write_feature_store


MODEL_NAME = "Qwen/Qwen3-VL-Embedding-2B"
CAPTION_PROMPT = "Represent this bounded video-event caption for evidence retrieval."
ENTITY_PROMPT = "Represent these visible entities and states for evidence retrieval."
VISUAL_PROMPT = "Represent this video frame for event and state retrieval."


def materialize_cohort(
    overlay_paths: Sequence[Path],
    destination: Path,
    *,
    device: str,
    batch_size: int = 8,
    frames_per_node: int = 3,
    max_nodes: int | None = None,
    max_nodes_per_video: int | None = None,
) -> dict[str, Any]:
    """Build four independent slots; never consume hidden question/answer fields."""

    if frames_per_node <= 0:
        raise ValueError("frames per node must be positive")
    nodes = _load_safe_unique_nodes(overlay_paths)
    if max_nodes_per_video is not None:
        if max_nodes_per_video <= 0:
            raise ValueError("max nodes per video must be positive")
        kept: list[dict[str, Any]] = []
        per_video: dict[str, int] = {}
        for node in nodes:
            video_id = str(node.get("video_id") or "")
            count = per_video.get(video_id, 0)
            if count >= max_nodes_per_video:
                continue
            kept.append(node)
            per_video[video_id] = count + 1
        nodes = kept
    if max_nodes is not None:
        nodes = nodes[:max_nodes]
    if not nodes:
        raise ValueError("cohort contains no safe L1 nodes")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("cohort materialization requires sentence-transformers") from exc

    model = SentenceTransformer(MODEL_NAME, device=device)
    captions = [str(node.get("text") or "").strip() for node in nodes]
    caption_validity = np.asarray([bool(text) for text in captions], dtype=np.bool_)
    caption_inputs = [text or "Unavailable bounded caption." for text in captions]
    caption = _encode(model, caption_inputs, CAPTION_PROMPT, batch_size)

    entity_texts = [_entity_state_text(node) for node in nodes]
    entity_validity = np.asarray([bool(text) for text in entity_texts], dtype=np.bool_)
    entity_inputs = [text or "No bounded entity or state fields available." for text in entity_texts]
    entity_state = _encode(model, entity_inputs, ENTITY_PROMPT, batch_size)
    entity_state[~entity_validity] = 0.0

    visual, visual_validity, visual_failures = _encode_visual_nodes(
        model,
        nodes,
        batch_size=batch_size,
        frames_per_node=frames_per_node,
    )
    time = np.asarray([_time_values(node) for node in nodes], dtype=np.float32)
    time_validity = np.ones(len(nodes), dtype=np.bool_)
    validity = np.stack(
        (caption_validity, entity_validity, visual_validity, time_validity), axis=1
    )
    rows = tuple(
        FeatureRow(
            index,
            str(node["node_id"]),
            str(node["video_id"]),
            _lineage_hash(node),
        )
        for index, node in enumerate(nodes)
    )
    manifest = write_feature_store(
        destination,
        arrays={
            "caption": caption.astype(np.float32),
            "entity_state": entity_state.astype(np.float32),
            "visual": visual.astype(np.float32),
            "time": time,
        },
        validity=validity,
        rows=rows,
        encoders={
            "caption": f"{MODEL_NAME}:bounded-caption/v1",
            "entity_state": f"{MODEL_NAME}:bounded-entity-state/v1",
            "visual": f"{MODEL_NAME}:visual-{frames_per_node}-frame-mean/v1",
            "time": "deterministic-normalized-time/v1",
        },
        source_contract="cgbench-question-independent-l1-pre-read-four-slot/v1",
        boundary_audit_version="reasoning-v2-address-value-boundary/v1",
        normalized={
            "caption": True,
            "entity_state": True,
            "visual": True,
            "time": False,
        },
    )
    store = FourSlotFeatureStore(manifest)
    return {
        "schema_version": "steam-qformer-cohort-materialization/v0.1",
        "manifest": str(manifest),
        "node_count": len(store),
        "video_count": len({row.video_id for row in rows}),
        "frames_per_node": frames_per_node,
        "coverage": store.coverage(),
        "visual_failures": visual_failures,
        "complete_four_slot_count": int(validity.all(axis=1).sum()),
        "source_contract": store.manifest.source_contract,
    }


def _load_safe_unique_nodes(paths: Iterable[Path]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(paths):
        payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        overlay_metadata = payload.get("metadata") or {}
        raw_video = str(overlay_metadata.get("raw_video_path") or "")
        duration = float(
            ((overlay_metadata.get("video_l1_extraction") or {}).get("duration_s") or 0.0)
        )
        for node in payload.get("l1_observations") or ():
            if not isinstance(node, dict):
                continue
            provenance = node.get("provenance") or {}
            visibility = (node.get("metadata") or {}).get("visibility") or {}
            if (
                provenance.get("uses_hidden_supervision") is True
                or visibility.get("hidden_supervision") is True
            ):
                raise ValueError(f"hidden supervision found in node {node.get('node_id')}")
            copied = dict(node)
            copied["_raw_video_path"] = raw_video
            copied["_video_duration_s"] = duration
            key = (str(node.get("video_id") or ""), str(node.get("node_id") or ""))
            unique.setdefault(key, copied)
    return [unique[key] for key in sorted(unique)]


def _encode(model: Any, values: Sequence[Any], prompt: str, batch_size: int) -> np.ndarray:
    matrix = model.encode(
        list(values),
        prompt=prompt,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(values):
        raise ValueError(f"embedding model returned unexpected shape {matrix.shape}")
    return matrix


def _encode_visual_nodes(
    model: Any,
    nodes: Sequence[dict[str, Any]],
    *,
    batch_size: int,
    frames_per_node: int,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("visual materialization requires opencv-python") from exc

    frame_paths: list[str] = []
    owners: list[int] = []
    failures: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="steam-qformer-frames-") as directory:
        temp = Path(directory)
        # Bound decoder memory/file descriptors by processing one video at a
        # time. Large benchmark cohorts may contain thousands of videos; a
        # persistent VideoCapture per path grows to tens of GB.
        nodes_by_video: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for index, node in enumerate(nodes):
            nodes_by_video.setdefault(str(node.get("_raw_video_path") or ""), []).append(
                (index, node)
            )
        for video_path, indexed_nodes in nodes_by_video.items():
            if not video_path or not Path(video_path).is_file():
                for _, node in indexed_nodes:
                    failures.append({"node_id": node.get("node_id"), "reason": "video_missing"})
                continue
            capture = cv2.VideoCapture(video_path)
            try:
                for index, node in indexed_nodes:
                    start, end = _span(node)
                    times = np.linspace(start, end, frames_per_node + 2, dtype=np.float64)[1:-1]
                    node_frame_count = 0
                    for frame_number, time_s in enumerate(times):
                        capture.set(cv2.CAP_PROP_POS_MSEC, float(time_s) * 1000.0)
                        success, frame = capture.read()
                        if not success:
                            continue
                        output = temp / f"{index:07d}_{frame_number:02d}.jpg"
                        if not cv2.imwrite(str(output), frame):
                            continue
                        frame_paths.append(str(output))
                        owners.append(index)
                        node_frame_count += 1
                    if not node_frame_count:
                        failures.append(
                            {"node_id": node.get("node_id"), "reason": "frame_decode_failed"}
                        )
            finally:
                capture.release()
        if not frame_paths:
            raise ValueError("no visual frames could be decoded")
        frame_embeddings = _encode(model, frame_paths, VISUAL_PROMPT, batch_size)

    dimension = int(frame_embeddings.shape[1])
    visual = np.zeros((len(nodes), dimension), dtype=np.float32)
    counts = np.zeros(len(nodes), dtype=np.int32)
    for owner, embedding in zip(owners, frame_embeddings):
        visual[owner] += embedding
        counts[owner] += 1
    validity = counts > 0
    visual[validity] /= counts[validity, None]
    norms = np.linalg.norm(visual, axis=1, keepdims=True)
    visual[validity] /= np.clip(norms[validity], 1e-12, None)
    return visual, validity, failures


def _entity_state_text(node: dict[str, Any]) -> str:
    metadata = node.get("metadata") or {}
    parts: list[str] = []
    for value in metadata.get("participants") or ():
        if not isinstance(value, dict):
            continue
        fields = [
            str(value.get(name) or "").strip()
            for name in ("role", "entity_type", "surface", "visual_signature")
        ]
        fields = [field for field in fields if field]
        if fields:
            parts.append("entity=" + " | ".join(fields))
    for value in metadata.get("states") or ():
        if not isinstance(value, dict):
            continue
        attribute = str(value.get("attribute") or "").strip()
        state = str(value.get("value") or "").strip()
        if attribute and state:
            parts.append(f"state={attribute}:{state}")
    change = metadata.get("state_change")
    if isinstance(change, dict):
        before = str(change.get("before") or "").strip()
        after = str(change.get("after") or "").strip()
        attribute = str(change.get("attribute") or "").strip()
        if attribute and before and after:
            parts.append(f"state_change={attribute}:{before}->{after}")
    return "; ".join(parts)


def _span(node: dict[str, Any]) -> tuple[float, float]:
    payload = node.get("time_span") or {}
    start = float(payload.get("start_s") or 0.0)
    end = float(payload.get("end_s") or start)
    if start < 0 or end < start:
        raise ValueError(f"invalid node span: {node.get('node_id')}")
    return start, end


def _time_values(node: dict[str, Any]) -> list[float]:
    start, end = _span(node)
    metadata = node.get("_video_duration_s")
    duration = float(metadata) if metadata else 0.0
    if duration <= 0:
        video_path = Path(str(node.get("_raw_video_path") or ""))
        # The enclosing L1 overlay normally provides duration elsewhere; using
        # end as a lower-bound normalization is deterministic for pilot data.
        duration = max(end, 1.0)
        if not video_path.is_file():
            duration = max(end, 1.0)
    center = (start + end) / 2.0
    return [start / duration, end / duration, (end - start) / duration, center / duration]


def _lineage_hash(node: dict[str, Any]) -> str:
    payload = {
        "node_id": node.get("node_id"),
        "video_id": node.get("video_id"),
        "time_span": node.get("time_span"),
        "text": node.get("text"),
        "source_node_id": node.get("source_node_id"),
        "source_segments": node.get("source_segments"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--frames-per-node", type=int, default=3)
    parser.add_argument("--max-nodes", type=int)
    parser.add_argument("--max-nodes-per-video", type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)
    paths = sorted(args.root.expanduser().resolve().glob("**/causal_temporal_overlay.json"))
    report = materialize_cohort(
        paths,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        frames_per_node=args.frames_per_node,
        max_nodes=args.max_nodes,
        max_nodes_per_video=args.max_nodes_per_video,
    )
    output = args.report.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
