"""Targeted raw-video reread for structured candidate-causal witnesses."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .types import CausalWitness, MemoryNode, RelationBelief, VisualVerification


class VLMClient(Protocol):
    model: str

    def perceive(
        self,
        prompt: str,
        *,
        image_urls: list[str] | None = None,
        system: str = ...,
    ) -> dict[str, Any]: ...


class VisualRereadProvider(Protocol):
    model: str

    def verify(
        self,
        *,
        video_path: str | Path,
        belief: RelationBelief,
        src: MemoryNode,
        dst: MemoryNode,
        witness: CausalWitness,
    ) -> VisualVerification: ...


@dataclass
class QwenVisualRereadProvider:
    """Use a local Qwen3.5-9B-compatible client for targeted frame rereads."""

    client: VLMClient
    frames_per_window: int = 5
    context_s: float = 0.5
    model: str = "Qwen/Qwen3.5-9B"

    def __post_init__(self) -> None:
        if self.frames_per_window < 2:
            raise ValueError("frames_per_window must be at least 2")
        client_model = getattr(self.client, "model", None)
        if client_model:
            self.model = str(client_model)

    def verify(
        self,
        *,
        video_path: str | Path,
        belief: RelationBelief,
        src: MemoryNode,
        dst: MemoryNode,
        witness: CausalWitness,
    ) -> VisualVerification:
        path = Path(video_path)
        if not path.is_file():
            return VisualVerification(
                status="inconclusive",
                model=self.model,
                reasons=(f"video path is unavailable: {path}",),
            )
        windows = _witness_windows(src, dst, context_s=self.context_s)
        frames, frame_records = _sample_frame_data_uris(
            path,
            windows=windows,
            frames_per_window=self.frames_per_window,
        )
        if not frames:
            return VisualVerification(
                status="inconclusive",
                model=self.model,
                video_windows=tuple(windows),
                reasons=("no video frames could be decoded",),
            )

        response = self.client.perceive(
            _visual_prompt(
                belief=belief,
                src=src,
                dst=dst,
                witness=witness,
                frame_records=frame_records,
            ),
            image_urls=frames,
            system=(
                "Inspect sampled video frames conservatively. Report only visible "
                "evidence in strict JSON; temporal order alone is not causality."
            ),
        )
        checks = {
            name: _optional_bool(response.get("checks", {}).get(name))
            for name in (
                "cause_visible",
                "effect_visible",
                "entity_continuity",
                "state_delta_visible",
                "mechanism_visible",
                "temporal_order_visible",
                "alternative_visible",
            )
        }
        evidence = _parse_evidence(response.get("evidence"))
        grounding_problems = _evidence_grounding_problems(
            frame_records,
            evidence=evidence,
            checks=checks,
        )
        required = (
            checks["cause_visible"],
            checks["effect_visible"],
            checks["entity_continuity"],
            checks["mechanism_visible"],
            checks["temporal_order_visible"],
        )
        if all(value is True for value in required) and not grounding_problems:
            status = "passed"
        elif any(value is False for value in required) or grounding_problems:
            status = "failed"
        else:
            status = "inconclusive"
        return VisualVerification(
            status=status,
            model=self.model,
            protocol_version="causal-visual-reread/v0.2",
            video_windows=tuple(windows),
            frame_records=tuple(frame_records),
            checks=checks,
            evidence=evidence,
            reasons=tuple(
                str(value) for value in response.get("reasons") or []
            )
            + tuple(grounding_problems),
        )


def _witness_windows(
    src: MemoryNode,
    dst: MemoryNode,
    *,
    context_s: float,
) -> list[dict[str, Any]]:
    return [
        {
            "start_s": max(0.0, src.time_span.start_s - context_s),
            "end_s": src.time_span.end_s + context_s,
            "purpose": "cause",
        },
        {
            "start_s": max(0.0, src.time_span.end_s - context_s),
            "end_s": dst.time_span.start_s + context_s,
            "purpose": "mechanism",
        },
        {
            "start_s": max(0.0, dst.time_span.start_s - context_s),
            "end_s": dst.time_span.end_s + context_s,
            "purpose": "effect",
        },
    ]


def _sample_frame_data_uris(
    video_path: Path,
    *,
    windows: list[dict[str, Any]],
    frames_per_window: int,
) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import cv2
    except ImportError:
        return [], []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return [], []
    frame_urls: list[str] = []
    frame_records: list[dict[str, Any]] = []
    try:
        for window in windows:
            start = float(window["start_s"])
            end = max(start, float(window["end_s"]))
            if frames_per_window == 1 or end == start:
                times = [start]
            else:
                step = (end - start) / (frames_per_window - 1)
                times = [start + step * index for index in range(frames_per_window)]
            for time_s in times:
                capture.set(cv2.CAP_PROP_POS_MSEC, time_s * 1000.0)
                ok, frame = capture.read()
                if not ok:
                    continue
                frame_index = len(frame_records)
                label = f"F{frame_index} {window['purpose']} {time_s:.2f}s"
                cv2.putText(
                    frame,
                    label,
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                encoded, buffer = cv2.imencode(".jpg", frame)
                if not encoded:
                    continue
                payload = base64.b64encode(buffer.tobytes()).decode("ascii")
                frame_urls.append(f"data:image/jpeg;base64,{payload}")
                frame_records.append(
                    {
                        "frame_index": frame_index,
                        "time_s": round(time_s, 3),
                        "purpose": str(window["purpose"]),
                    }
                )
    finally:
        capture.release()
    return frame_urls, frame_records


def _visual_prompt(
    *,
    belief: RelationBelief,
    src: MemoryNode,
    dst: MemoryNode,
    witness: CausalWitness,
    frame_records: list[dict[str, Any]],
) -> str:
    payload = {
        "relation": next(
            (
                name
                for name in ("explains", "enables")
                if name in belief.relation_probabilities
            ),
            witness.relation,
        ),
        "cause_event": {"id": src.node_id, "text": src.text},
        "effect_event": {"id": dst.node_id, "text": dst.text},
        "causal_witness": witness.to_dict(),
        "frame_records": frame_records,
    }
    return (
        "Verify this candidate-causal witness against the sampled frames. Every "
        "image is visibly labeled with its frame index, purpose, and timestamp. "
        "A check is true only when directly visible. Set unknown checks to null. "
        "Return JSON with `checks` containing cause_visible, effect_visible, "
        "entity_continuity, state_delta_visible, mechanism_visible, "
        "temporal_order_visible, alternative_visible. Also return `evidence` "
        "with integer frame-index lists named cause, effect, mechanism, and "
        "entity. Cause evidence must use cause frames, effect evidence must use "
        "effect frames, and entity evidence must cover both. Never infer a "
        "transition from endpoint descriptions alone. Include a `reasons` list.\n"
        + json.dumps(payload, indent=2)
    )


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _parse_evidence(value: Any) -> dict[str, tuple[int, ...]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, tuple[int, ...]] = {}
    for key in ("cause", "effect", "mechanism", "entity"):
        raw = value.get(key)
        if not isinstance(raw, list):
            continue
        parsed: list[int] = []
        for item in raw:
            try:
                parsed.append(int(item))
            except (TypeError, ValueError):
                continue
        result[key] = tuple(dict.fromkeys(parsed))
    return result


def _evidence_grounding_problems(
    frame_records: list[dict[str, Any]],
    *,
    evidence: dict[str, tuple[int, ...]],
    checks: dict[str, bool | None],
) -> list[str]:
    by_index = {
        int(record["frame_index"]): record for record in frame_records
    }
    problems: list[str] = []

    def records_for(name: str) -> list[dict[str, Any]]:
        indices = evidence.get(name, ())
        unknown = [index for index in indices if index not in by_index]
        if unknown:
            problems.append(f"{name} evidence cites unknown frames: {unknown}")
        return [by_index[index] for index in indices if index in by_index]

    cause = records_for("cause")
    effect = records_for("effect")
    mechanism = records_for("mechanism")
    entity = records_for("entity")
    if checks.get("cause_visible") is True and (
        not cause or any(record["purpose"] != "cause" for record in cause)
    ):
        problems.append("cause_visible lacks cause-window frame evidence")
    if checks.get("effect_visible") is True and (
        not effect or any(record["purpose"] != "effect" for record in effect)
    ):
        problems.append("effect_visible lacks effect-window frame evidence")
    if checks.get("mechanism_visible") is True and (
        not mechanism
        or any(
            record["purpose"] not in {"cause", "mechanism"}
            for record in mechanism
        )
    ):
        problems.append("mechanism_visible lacks cause/mechanism frame evidence")
    if checks.get("entity_continuity") is True:
        purposes = {record["purpose"] for record in entity}
        if not {"cause", "effect"}.issubset(purposes):
            problems.append("entity continuity lacks cause-and-effect frame evidence")
    if checks.get("temporal_order_visible") is True:
        if not cause or not effect:
            problems.append("temporal order lacks grounded endpoint frames")
        elif max(float(record["time_s"]) for record in cause) >= min(
            float(record["time_s"]) for record in effect
        ):
            problems.append("grounded cause frames do not precede effect frames")
    return problems
