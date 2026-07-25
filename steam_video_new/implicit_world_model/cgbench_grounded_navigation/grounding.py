"""Ground CG-Bench read actions with Qwen-VL and persist Qwen3-VL embeddings."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Protocol, Sequence

from memory_graph.embedding import Qwen3VLEmbeddingProvider
from memory_graph.visual_verifier import VLMClient, _sample_frame_data_uris

from .builder import (
    EMBEDDING_MODEL,
    _checksum,
    _write_json,
    validate_cgbench_navigation_dataset,
)


GROUNDING_SCHEMA = "steam-cgbench-qwen-grounding/v0.1"
ALLOWED_RELEVANCE = {"relevant", "unrelated", "inconclusive"}


class TextEmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def encode(
        self, texts: Sequence[str], *, batch_size: int = 8
    ) -> Sequence[Sequence[float]]: ...


def ground_navigation_dataset(
    dataset: dict[str, Any],
    *,
    video_root: Path,
    client: VLMClient,
    frames_per_interval: int = 6,
    case_limit: int | None = None,
    selected_video_ids: set[str] | None = None,
    progress_callback: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read real frames without exposing clue labels or terminal answers."""
    if frames_per_interval < 2:
        raise ValueError("frames_per_interval must be at least two")
    result = deepcopy(dataset)
    completed = failed = repaired = 0
    cases = result.get("cases") or []
    if selected_video_ids is not None:
        cases = [case for case in cases if case.get("video_id") in selected_video_ids]
    if case_limit is not None:
        cases = cases[:case_limit]
    selected_ids = {case["case_id"] for case in cases}
    for case in result.get("cases") or []:
        if case["case_id"] not in selected_ids:
            continue
        video_path = (video_root / Path(case["video_ref"]).name).resolve()
        for transition in case.get("executed_transitions") or []:
            observation = transition["real_observation"]
            if observation.get("descriptor_status") == "grounded_qwen_vl_read":
                continue
            interval = transition["action"]["interval"]
            images, frame_records = _sample_frame_data_uris(
                video_path,
                windows=[{**interval, "purpose": "candidate_read"}],
                frames_per_window=frames_per_interval,
            )
            if not images:
                observation.update(
                    {"descriptor_status": "grounding_failed", "descriptor": None}
                )
                failed += 1
                continue
            payload = client.perceive(
                _descriptor_prompt(case, interval, frame_records),
                image_urls=images,
                system=(
                    "Describe only directly visible or readable evidence in the sampled video "
                    "frames. Do not infer identity, causality, intent, hidden events, or the "
                    "correct answer. Return strict JSON and no numeric confidence."
                ),
            )
            try:
                descriptor = _validated_descriptor(payload, frame_records)
            except ValueError as exc:
                repair = client.perceive(
                    _repair_prompt(case, interval, frame_records, str(exc)),
                    image_urls=images,
                    system=(
                        "Repair the prior JSON using only the attached sampled frames. Follow the "
                        "exact schema, cite frame indices, and do not add confidence, scores, or "
                        "facts not directly visible."
                    ),
                )
                try:
                    descriptor = _validated_descriptor(repair, frame_records)
                except ValueError as repair_exc:
                    observation.update(
                        {
                            "descriptor_status": "grounding_failed",
                            "descriptor": None,
                            "failure_reason": str(repair_exc),
                            "first_failure_reason": str(exc),
                            "rejected_model_payload": repair,
                        }
                    )
                    failed += 1
                    continue
                repaired += 1
            observation.update(
                {
                    "descriptor_status": "grounded_qwen_vl_read",
                    "descriptor": descriptor,
                    "observed_modalities": [
                        "sampled_video_frames",
                        "readable_text_in_frames",
                    ],
                    "unobserved_requested_modalities": ["audio"],
                    "producer": {"model": client.model, "protocol": GROUNDING_SCHEMA},
                }
            )
            transition["target"]["observation_descriptor"].update(
                {
                    "grounding_status": "grounded",
                    "descriptor_ref": transition["transition_id"],
                }
            )
            completed += 1
        if progress_callback is not None:
            progress_callback(
                result,
                {
                    "grounded_transition_count": completed,
                    "failed_transition_count": failed,
                    "repaired_transition_count": repaired,
                },
            )
    statuses = [
        transition.get("real_observation", {}).get("descriptor_status")
        for case in result.get("cases") or []
        for transition in case.get("executed_transitions") or []
    ]
    grounded_total = statuses.count("grounded_qwen_vl_read")
    result["annotation_status"] = (
        "qwen_grounded_over_cgbench_gt"
        if grounded_total == len(statuses)
        else "qwen_grounding_partial_over_cgbench_gt"
    )
    result["formal_eligible"] = False
    result["training_ready"] = False
    report = {
        "schema_version": GROUNDING_SCHEMA,
        "dataset_id": result.get("dataset_id"),
        "model": client.model,
        "grounded_transition_count": completed,
        "failed_transition_count": failed,
        "repaired_transition_count": repaired,
        "grounded_transition_total": grounded_total,
        "pending_transition_total": statuses.count("pending_qwen_vl_grounded_read"),
        "failed_transition_total": statuses.count("grounding_failed"),
        "selected_case_count": len(selected_ids),
        "formal_eligible": False,
        "training_ready": False,
        "label_source": "CG-Bench human answer and clue ground truth",
        "human_review_gate": "not_required; optional descriptor diagnostics only",
        "next_gate": "complete grounded reads, then run schema, coverage, and leakage checks",
    }
    return result, report


def attach_descriptor_embeddings(
    dataset: dict[str, Any],
    provider: TextEmbeddingProvider,
    *,
    output_path: Path,
    storage_uri: str | None = None,
    batch_size: int = 8,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Embed only grounded descriptor text; vectors stay in a .npy sidecar."""
    if provider.model_name != EMBEDDING_MODEL or provider.dimension != 2048:
        raise ValueError(
            "the embedding provider violates the Qwen3-VL-Embedding-2B contract"
        )
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required to persist embeddings") from exc
    result = deepcopy(dataset)
    rows: list[tuple[dict[str, Any], str, str]] = []
    for case in result.get("cases") or []:
        question = str((case.get("planner_input") or {}).get("question") or "")
        for transition in case.get("executed_transitions") or []:
            observation = transition.get("real_observation") or {}
            if observation.get("descriptor_status") != "grounded_qwen_vl_read":
                continue
            descriptor = observation.get("descriptor") or {}
            text = _embedding_text(question, descriptor)
            rows.append((transition, text, str(transition.get("transition_id"))))
    matrix = np.asarray(
        provider.encode([row[1] for row in rows], batch_size=batch_size),
        dtype=np.float32,
    )
    if matrix.shape != (len(rows), 2048):
        raise ValueError(
            f"embedding matrix has shape {matrix.shape}; expected {(len(rows), 2048)}"
        )
    if len(rows):
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, matrix)
    saved_path = (
        output_path if output_path.suffix == ".npy" else output_path.with_suffix(".npy")
    )
    checksum = hashlib.sha256(saved_path.read_bytes()).hexdigest()
    uri = storage_uri or saved_path.name
    for index, (transition, _, _) in enumerate(rows):
        transition["target"]["observation_descriptor"]["embedding_ref"] = {
            "model": EMBEDDING_MODEL,
            "status": "available",
            "dimension": 2048,
            "dtype": "float32",
            "normalized": True,
            "storage_uri": uri,
            "row_index": index,
            "checksum": checksum,
        }
    manifest = {
        "schema_version": "steam-cgbench-embedding-manifest/v0.1",
        "model": EMBEDDING_MODEL,
        "matrix_uri": uri,
        "row_count": len(rows),
        "dimension": 2048,
        "dtype": "float32",
        "normalized": True,
        "checksum": checksum,
        "rows": [
            {"row_index": i, "transition_id": row[2]} for i, row in enumerate(rows)
        ],
    }
    return result, manifest


def refresh_hidden_checksum(
    hidden: dict[str, Any], dataset: dict[str, Any]
) -> dict[str, Any]:
    result = deepcopy(hidden)
    result["dataset_sha256"] = _checksum(dataset)
    return result


def _descriptor_prompt(
    case: dict[str, Any], interval: dict[str, Any], frames: list[dict[str, Any]]
) -> str:
    planner = case.get("planner_input") or {}
    template = {
        "summary": "one factual sentence",
        "visible_entities": [],
        "visible_actions": [],
        "visible_states": [],
        "readable_text": [],
        "question_relevance": "inconclusive",
        "relevance_reason": "one sentence",
        "evidence_frames": [0],
    }
    return (
        "Inspect this candidate interval for the question. The frames are sparse samples. "
        "Question relevance means whether this interval contributes any partial fact needed "
        "to answer the question; it does not need to make the full answer sufficient. For an "
        "ordering question, one named item in the sequence is relevant. Use unrelated only "
        "when no visible fact contributes, and inconclusive when the frames cannot decide. "
        "Return: summary (one factual sentence); visible_entities (strings); visible_actions "
        "(strings); visible_states (strings); readable_text (strings); question_relevance "
        "(relevant|unrelated|inconclusive); relevance_reason (one sentence); evidence_frames "
        "(integer frame indices). Use empty lists when absent. Every list must contain at most six "
        "short unique items. Never repeat UI/subtitle text and never include the synthetic F0/F1 "
        "frame labels in readable_text. Return every template key; do not repeat the request.\n"
        + json.dumps(
            {
                "question": planner.get("question"),
                "choices": planner.get("choices"),
                "interval": interval,
                "frame_records": frames,
                "required_template": template,
            },
            ensure_ascii=False,
        )
    )


def _repair_prompt(
    case: dict[str, Any],
    interval: dict[str, Any],
    frames: list[dict[str, Any]],
    error: str,
) -> str:
    planner = case.get("planner_input") or {}
    template = {
        "summary": "one directly observed factual sentence",
        "visible_entities": [],
        "visible_actions": [],
        "visible_states": [],
        "readable_text": [],
        "question_relevance": "inconclusive",
        "relevance_reason": "one sentence",
        "evidence_frames": [0],
    }
    return (
        "The prior response failed validation. Re-inspect the attached frames and return exactly "
        "one JSON object with every key in "
        "the template. evidence_frames must be a JSON list of integer frame_index values from "
        "frame_records; use [] only if no frame supports any observation. Relevance means any "
        "partial contribution, not whether this interval alone fully answers the question. Do "
        "not repeat this request or the template. Every list must have at most six unique short "
        "items; do not repeat UI text.\n"
        + json.dumps(
            {
                "validation_error": error,
                "template": template,
                "question": planner.get("question"),
                "choices": planner.get("choices"),
                "interval": interval,
                "frame_records": frames,
            },
            ensure_ascii=False,
        )
    )


def _validated_descriptor(
    payload: dict[str, Any], frames: list[dict[str, Any]]
) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("parse_error"):
        raise ValueError("Qwen-VL response is not valid JSON")
    raw_relevance = str(payload.get("question_relevance") or "").strip().casefold()
    relevance = raw_relevance if raw_relevance in ALLOWED_RELEVANCE else "inconclusive"
    allowed_frames = {int(row["frame_index"]) for row in frames}
    cited = payload.get("evidence_frames")
    if not isinstance(cited, list):
        raise ValueError("evidence_frames is missing")
    try:
        cited_frames = list(dict.fromkeys(int(value) for value in cited))
    except (TypeError, ValueError) as exc:
        raise ValueError("evidence_frames contains a non-integer") from exc
    frame_index_status = "native_zero_based"
    if any(index not in allowed_frames for index in cited_frames):
        one_based = set(range(1, len(frames) + 1))
        if (
            cited_frames
            and 0 not in cited_frames
            and len(frames) in cited_frames
            and set(cited_frames) <= one_based
        ):
            cited_frames = [index - 1 for index in cited_frames]
            frame_index_status = "canonicalized_from_one_based"
        else:
            raise ValueError("descriptor cites an unsampled frame")
    result: dict[str, Any] = {
        "summary": str(payload.get("summary") or "").strip(),
        "question_relevance": relevance,
        "relevance_reason": str(payload.get("relevance_reason") or "").strip(),
        "evidence_frames": cited_frames,
        "frame_index_status": frame_index_status,
    }
    if raw_relevance not in ALLOWED_RELEVANCE:
        result["raw_question_relevance"] = raw_relevance or None
        result["relevance_schema_status"] = (
            "invalid_model_label_coerced_to_inconclusive"
        )
    for key in (
        "visible_entities",
        "visible_actions",
        "visible_states",
        "readable_text",
    ):
        values = payload.get(key)
        if not isinstance(values, list):
            raise ValueError(f"{key} is not a list")
        result[key] = list(
            dict.fromkeys(str(value).strip() for value in values if str(value).strip())
        )[:6]
    if not result["summary"] or not result["relevance_reason"]:
        raise ValueError("descriptor text is incomplete")
    return result


def _embedding_text(question: str, descriptor: dict[str, Any]) -> str:
    parts = [
        f"Question: {question}",
        f"Observed interval: {descriptor.get('summary', '')}",
    ]
    for key, label in (
        ("visible_entities", "Entities"),
        ("visible_actions", "Actions"),
        ("visible_states", "States"),
        ("readable_text", "Readable text"),
    ):
        values = descriptor.get(key) or []
        if values:
            parts.append(f"{label}: " + "; ".join(str(value) for value in values))
    return "\n".join(parts)


def _load_client(model: str, endpoint: str, timeout_s: int) -> VLMClient:
    import requests

    class _Client:
        def __init__(self) -> None:
            self.model = model

        def perceive(
            self, prompt: str, *, image_urls: list[str] | None = None, system: str = ""
        ) -> dict[str, Any]:
            content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
            content.extend(
                {"type": "image_url", "image_url": {"url": url}}
                for url in image_urls or []
            )
            response = requests.post(
                endpoint,
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": content},
                    ],
                    "temperature": 0,
                    "max_tokens": 512,
                },
                timeout=timeout_s,
            )
            response.raise_for_status()
            text = str(
                response.json()["choices"][0]["message"].get("content") or ""
            ).strip()
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL)
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            try:
                return json.loads(match.group() if match else text)
            except (json.JSONDecodeError, AttributeError):
                return {"parse_error": True, "raw_response": text[:1000]}

    return _Client()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ground = sub.add_parser("ground")
    ground.add_argument("--input", required=True, type=Path)
    ground.add_argument("--hidden-input", required=True, type=Path)
    ground.add_argument("--video-root", required=True, type=Path)
    ground.add_argument("--output", required=True, type=Path)
    ground.add_argument("--hidden-output", required=True, type=Path)
    ground.add_argument("--report", required=True, type=Path)
    ground.add_argument("--endpoint", required=True)
    ground.add_argument("--model", default="Qwen/Qwen3.5-9B")
    ground.add_argument("--frames-per-interval", type=int, default=6)
    ground.add_argument("--case-limit", type=int)
    ground.add_argument(
        "--review-packet",
        type=Path,
        help="Optional blinded audit packet; ground only videos represented in its public items.",
    )
    ground.add_argument("--timeout-s", type=int, default=120)
    embed = sub.add_parser("embed")
    embed.add_argument("--input", required=True, type=Path)
    embed.add_argument("--hidden-input", required=True, type=Path)
    embed.add_argument("--output", required=True, type=Path)
    embed.add_argument("--hidden-output", required=True, type=Path)
    embed.add_argument("--matrix", required=True, type=Path)
    embed.add_argument("--manifest", required=True, type=Path)
    embed.add_argument("--device")
    embed.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    dataset = json.loads(args.input.read_text(encoding="utf-8"))
    hidden = json.loads(args.hidden_input.read_text(encoding="utf-8"))
    if args.command == "ground":
        selected_video_ids = None
        if args.review_packet is not None:
            review = json.loads(args.review_packet.read_text(encoding="utf-8"))
            selected_video_ids = {
                str(item["video_id"]) for item in review.get("items") or []
            }

        def checkpoint(partial: dict[str, Any], counts: dict[str, Any]) -> None:
            _write_json(args.output, partial)
            _write_json(args.hidden_output, refresh_hidden_checksum(hidden, partial))
            _write_json(
                args.report,
                {
                    "schema_version": GROUNDING_SCHEMA,
                    "dataset_id": partial.get("dataset_id"),
                    **counts,
                    "status": "running_checkpoint",
                },
            )

        result, report = ground_navigation_dataset(
            dataset,
            video_root=args.video_root,
            client=_load_client(args.model, args.endpoint, args.timeout_s),
            frames_per_interval=args.frames_per_interval,
            case_limit=args.case_limit,
            selected_video_ids=selected_video_ids,
            progress_callback=checkpoint,
        )
        hidden = refresh_hidden_checksum(hidden, result)
        errors = validate_cgbench_navigation_dataset(result, hidden)
        if errors:
            raise ValueError("grounded dataset is invalid: " + "; ".join(errors[:8]))
        _write_json(args.output, result)
        _write_json(args.hidden_output, hidden)
        _write_json(args.report, report)
    else:
        provider = Qwen3VLEmbeddingProvider(device=args.device)
        result, manifest = attach_descriptor_embeddings(
            dataset, provider, output_path=args.matrix, batch_size=args.batch_size
        )
        hidden = refresh_hidden_checksum(hidden, result)
        errors = validate_cgbench_navigation_dataset(result, hidden)
        if errors:
            raise ValueError("embedded dataset is invalid: " + "; ".join(errors[:8]))
        _write_json(args.output, result)
        _write_json(args.hidden_output, hidden)
        _write_json(args.manifest, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
