"""Fine-grained video-only L1 extraction with coarse-to-fine localization."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from .adaptive_windowing import AdaptiveWindowProvider
from .atomic_events import atomicity_issues
from .contracts import AtomicEvent, EntityMention, StateAssertion
from .types import MemoryNode, TimeSpan
from .visual_verifier import VLMClient, _sample_frame_data_uris


@dataclass(frozen=True)
class VideoL1Config:
    coarse_window_s: float = 8.0
    coarse_stride_s: float = 6.0
    frames_per_coarse_window: int = 8
    fine_context_s: float = 1.0
    frames_per_fine_window: int = 12
    max_events_per_window: int = 4
    minimum_confidence: float = 0.5
    track_max_gap_s: float = 12.0
    localization_mode: str = "coarse_to_fine"
    request_concurrency: int = 1

    def __post_init__(self) -> None:
        if self.coarse_window_s <= 0 or self.coarse_stride_s <= 0:
            raise ValueError("coarse window and stride must be positive")
        if self.frames_per_coarse_window < 2 or self.frames_per_fine_window < 3:
            raise ValueError("coarse/fine frame counts are too small")
        if self.max_events_per_window < 1:
            raise ValueError("max_events_per_window must be positive")
        if self.request_concurrency < 1:
            raise ValueError("request_concurrency must be positive")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")
        if self.localization_mode not in {
            "coarse_to_fine",
            "grounded_single_pass",
        }:
            raise ValueError("unsupported L1 localization mode")


@dataclass(frozen=True)
class VideoL1ExtractionResult:
    nodes: tuple[MemoryNode, ...]
    duration_s: float
    coarse_window_count: int
    coarse_candidate_count: int
    fine_localized_count: int
    entity_track_count: int
    model_call_count: int | None = None
    rejected: tuple[dict[str, Any], ...] = ()
    model: str | None = None
    protocol_version: str = "video-only-l1/v0.1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_version": self.protocol_version,
            "model": self.model,
            "duration_s": self.duration_s,
            "coarse_window_count": self.coarse_window_count,
            "coarse_candidate_count": self.coarse_candidate_count,
            "fine_localized_count": self.fine_localized_count,
            "entity_track_count": self.entity_track_count,
            "model_call_count": self.model_call_count,
            "rejected": list(self.rejected),
            "nodes": [node.to_dict() for node in self.nodes],
        }


class VideoL1Provider(Protocol):
    model: str

    def extract(
        self,
        *,
        video_path: str | Path,
        video_id: str,
        observation_end_s: float | None = None,
    ) -> VideoL1ExtractionResult: ...


@dataclass(frozen=True)
class PayloadVideoL1Provider:
    """Replay a persisted visual-L1 result after the VLM server is unloaded."""

    payload: dict[str, Any]
    model: str = "precomputed-video-only-l1"

    def extract(
        self,
        *,
        video_path: str | Path,
        video_id: str,
        observation_end_s: float | None = None,
    ) -> VideoL1ExtractionResult:
        del video_path
        nodes = tuple(
            _memory_node_from_dict(value) for value in self.payload.get("nodes") or []
        )
        if not nodes or any(node.video_id != video_id for node in nodes):
            raise ValueError(
                "persisted video L1 nodes are empty or belong to another video"
            )
        if observation_end_s is not None and any(
            node.time_span.end_s > observation_end_s + 1e-6 for node in nodes
        ):
            raise ValueError("persisted video L1 exceeds the observation horizon")
        return VideoL1ExtractionResult(
            nodes=nodes,
            duration_s=float(
                self.payload.get("duration_s") or observation_end_s or 0.0
            ),
            coarse_window_count=int(self.payload.get("coarse_window_count") or 0),
            coarse_candidate_count=int(self.payload.get("coarse_candidate_count") or 0),
            fine_localized_count=len(nodes),
            entity_track_count=int(self.payload.get("entity_track_count") or 0),
            model_call_count=(
                int(self.payload["model_call_count"])
                if self.payload.get("model_call_count") is not None
                else None
            ),
            rejected=tuple(self.payload.get("rejected") or []),
            model=str(self.payload.get("model") or self.model),
            protocol_version=str(
                self.payload.get("protocol_version") or "video-only-l1/v0.1"
            ),
        )


@dataclass
class _LocalizedEvent:
    predicate: str
    grounded_caption: str
    time_span: TimeSpan
    confidence: float
    action_kind: str
    evidence_frames: tuple[int, ...]
    frame_records: tuple[dict[str, Any], ...]
    participants: list[dict[str, Any]]
    states: list[dict[str, Any]]
    state_change: dict[str, Any] | None
    visible_text: list[dict[str, Any]]
    coarse_window: dict[str, Any]
    localization_method: str


@dataclass
class QwenVideoL1Extractor:
    """Use a local Qwen VLM as a visual evidence producer, not a causal reasoner."""

    client: VLMClient
    config: VideoL1Config = field(default_factory=VideoL1Config)
    model: str = "Qwen/Qwen3.5-9B"
    window_provider: AdaptiveWindowProvider | None = None
    client_factory: Callable[[], VLMClient] | None = None

    def __post_init__(self) -> None:
        client_model = getattr(self.client, "model", None)
        if client_model:
            self.model = str(client_model)
        if self.config.request_concurrency > 1 and self.client_factory is None:
            raise ValueError(
                "client_factory is required when request_concurrency is greater than 1"
            )

    def extract(
        self,
        *,
        video_path: str | Path,
        video_id: str,
        observation_end_s: float | None = None,
    ) -> VideoL1ExtractionResult:
        path = Path(video_path)
        if not path.is_file():
            raise ValueError(f"raw video path is unavailable: {path}")
        duration_s = _video_duration_s(path)
        if observation_end_s is not None:
            duration_s = min(duration_s, float(observation_end_s))
        if duration_s <= 0:
            raise ValueError("raw video has no readable duration")

        windows = (
            self.window_provider.windows(path, duration_s=duration_s)
            if self.window_provider is not None
            else _coarse_windows(duration_s, self.config)
        )
        candidates: list[dict[str, Any]] = []
        single_pass_localized: list[_LocalizedEvent] = []
        rejected: list[dict[str, Any]] = []
        indexed_windows = list(enumerate(windows))
        if self.config.request_concurrency == 1:
            coarse_results = [
                self._scan_coarse_window(path, window_index, window, self.client)
                for window_index, window in indexed_windows
            ]
        else:
            assert self.client_factory is not None

            def scan(item: tuple[int, dict[str, Any]]):
                window_index, window = item
                return self._scan_coarse_window(
                    path,
                    window_index,
                    window,
                    self.client_factory(),
                )

            with ThreadPoolExecutor(
                max_workers=self.config.request_concurrency,
                thread_name_prefix="video-l1-window",
            ) as executor:
                # executor.map preserves input order, so node IDs and rejection
                # ordering remain deterministic despite concurrent inference.
                coarse_results = list(executor.map(scan, indexed_windows))

        for parsed, problems, window_localized in coarse_results:
            candidates.extend(parsed)
            rejected.extend(problems)
            single_pass_localized.extend(window_localized)

        localized: list[_LocalizedEvent] = single_pass_localized
        fine_candidate_count = 0
        if self.config.localization_mode == "coarse_to_fine":
            fine_candidates = _dedupe_candidates(candidates)
            fine_candidate_count = len(fine_candidates)
            for candidate_index, candidate in enumerate(fine_candidates):
                window = {
                    "start_s": max(
                        0.0,
                        float(candidate["coarse_start_s"]) - self.config.fine_context_s,
                    ),
                    "end_s": min(
                        duration_s,
                        float(candidate["coarse_end_s"]) + self.config.fine_context_s,
                    ),
                    "purpose": "fine_localization",
                }
                images, records = _sample_frame_data_uris(
                    path,
                    windows=[window],
                    frames_per_window=self.config.frames_per_fine_window,
                )
                if not images:
                    rejected.append(
                        {
                            "stage": "fine",
                            "candidate_index": candidate_index,
                            "reason": "no frames",
                        }
                    )
                    continue
                response = self.client.perceive(
                    _fine_prompt(candidate=candidate, frame_records=records),
                    image_urls=images,
                    system=(
                        "Localize only what is directly visible in the labeled frames. "
                        "Use frame indices as evidence and return strict JSON."
                    ),
                )
                try:
                    event = _parse_fine_response(
                        response,
                        candidate=candidate,
                        frame_records=records,
                        minimum_confidence=self.config.minimum_confidence,
                    )
                except ValueError as exc:
                    rejected.append(
                        {
                            "stage": "fine",
                            "candidate_index": candidate_index,
                            "reason": str(exc),
                        }
                    )
                    continue
                if event is not None:
                    localized.append(event)

        localized = _dedupe_localized(localized)
        track_count = _assign_track_ids(
            localized, max_gap_s=self.config.track_max_gap_s
        )
        nodes = tuple(
            _event_to_l1_node(event, video_id=video_id, index=index)
            for index, event in enumerate(localized, start=1)
        )
        return VideoL1ExtractionResult(
            nodes=nodes,
            duration_s=duration_s,
            coarse_window_count=len(windows),
            coarse_candidate_count=len(candidates),
            fine_localized_count=len(nodes),
            entity_track_count=track_count,
            model_call_count=len(windows) + fine_candidate_count,
            rejected=tuple(rejected),
            model=self.model,
            protocol_version=(
                "video-only-l1/v0.3-rich-grounded-single-pass"
                if self.config.localization_mode == "grounded_single_pass"
                else "video-only-l1/v0.1"
            ),
        )

    def _scan_coarse_window(
        self,
        path: Path,
        window_index: int,
        window: dict[str, Any],
        client: VLMClient,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[_LocalizedEvent]]:
        images, records = _sample_frame_data_uris(
            path,
            windows=[window],
            frames_per_window=self.config.frames_per_coarse_window,
        )
        if not images:
            return [], [
                {
                    "stage": "coarse",
                    "window_index": window_index,
                    "reason": "no frames",
                }
            ], []
        response = client.perceive(
            _coarse_prompt(
                window=window,
                frame_records=records,
                max_events=self.config.max_events_per_window,
            ),
            image_urls=images,
            system=(
                "Report only directly visible video events and states in strict JSON. "
                "Do not infer causes, goals, identity, or events between sampled frames."
            ),
        )
        parsed, problems = _parse_coarse_response(
            response,
            window=window,
            window_index=window_index,
            minimum_confidence=self.config.minimum_confidence,
            max_events=self.config.max_events_per_window,
        )
        localized: list[_LocalizedEvent] = []
        if self.config.localization_mode == "grounded_single_pass":
            for local_index, candidate in enumerate(parsed):
                try:
                    localized.append(_parse_single_pass_candidate(candidate, records))
                except ValueError as exc:
                    problems.append(
                        {
                            "stage": "single_pass",
                            "window_index": window_index,
                            "local_index": local_index,
                            "reason": str(exc),
                        }
                    )
        return parsed, problems, localized


@dataclass(frozen=True)
class VideoL1AtomicEventExtractor:
    """Promote already-atomic visual L1 nodes into revisable L1.5 hypotheses."""

    model: str = "video-only-l1-direct"

    def extract(self, nodes: list[MemoryNode]) -> list[AtomicEvent]:
        events: list[AtomicEvent] = []
        for index, node in enumerate(
            sorted(nodes, key=lambda value: (value.time_span.start_s, value.node_id)),
            start=1,
        ):
            participants = tuple(
                EntityMention(
                    mention_id=str(value["mention_id"]),
                    role=str(value["role"]),
                    entity_type=str(value["entity_type"]),
                    surface=str(value["surface"]),
                    confidence=float(value["confidence"]),
                    grounding_refs=(node.node_id,),
                )
                for value in node.metadata.get("participants") or []
            )
            mention_ids = {participant.mention_id for participant in participants}
            states = tuple(
                StateAssertion(
                    mention_id=str(value["mention_id"]),
                    attribute=str(value["attribute"]),
                    value=str(value["value"]),
                    confidence=float(value["confidence"]),
                    polarity=str(value.get("polarity") or "positive"),
                )
                for value in node.metadata.get("states") or []
                if str(value.get("mention_id") or "") in mention_ids
            )
            events.append(
                AtomicEvent(
                    event_id=f"atomic:visual:{index:04d}",
                    video_id=node.video_id,
                    time_span=node.time_span,
                    predicate=str(node.metadata.get("predicate") or node.text or ""),
                    evidence_refs=(node.node_id,),
                    confidence=float(node.metadata.get("confidence") or 0.0),
                    participants=participants,
                    states=states,
                    provenance={
                        "producer": "memory_graph.video_l1.VideoL1AtomicEventExtractor",
                        "model": self.model,
                        "layer": "L1.5",
                        "visual_l1_passthrough": True,
                    },
                )
            )
        return events


def _coarse_windows(duration_s: float, config: VideoL1Config) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + config.coarse_window_s)
        windows.append({"start_s": start, "end_s": end, "purpose": "coarse_scan"})
        if end >= duration_s:
            break
        start += config.coarse_stride_s
    return windows


def _video_duration_s(path: Path) -> float:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required for video-only L1 extraction"
        ) from exc
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0.0
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        return count / fps if fps > 0 and count > 0 else 0.0
    finally:
        capture.release()


def _coarse_prompt(
    *,
    window: dict[str, Any],
    frame_records: list[dict[str, Any]],
    max_events: int,
) -> str:
    return (
        "Scan this time window for directly visible atomic actions or state changes. "
        "The images are sparse samples, so do not claim an event between frames. "
        f"Return at most {max_events} events. Each event needs predicate, "
        "coarse_start_s, coarse_end_s, grounding_status (observed|inconclusive), action_kind "
        "(action|motion|contact|transfer|state_change|visible_response|other), "
        "visible_start_frame, visible_end_frame, evidence_frames (all integer "
        "indices from frame_records), participants "
        "[{role, entity_type, surface, visual_signature, evidence_frames}], and "
        "grounded_caption (one concise factual sentence describing only visible "
        "content), visible_text [{text, evidence_frames}] for legible packaging, "
        "sign, or burned-in subtitle text, visible states "
        "[{participant_index, attribute, value, polarity, "
        "evidence_frames}]. An optional state_change has participant_index, "
        "attribute, before, after, and evidence_frames. Every observed event and "
        "participant must cite directly visible frame indices. Do not output confidence, "
        "probability, score, reward, or utility. Return JSON as "
        '{"events": [...]}.\n'
        f"window={window}\nframe_records={frame_records}"
    )


def _fine_prompt(
    *,
    candidate: dict[str, Any],
    frame_records: list[dict[str, Any]],
) -> str:
    return (
        "Verify and localize this coarse event using the labeled frames. If it is "
        "not directly visible, set observed=false. Otherwise return observed=true, "
        "predicate, visible_start_frame, visible_end_frame, "
        "evidence_frames, action_kind, grounded_caption, visible_text with text/"
        "evidence_frames, participants with role/entity_type/surface/"
        "visual_signature/evidence_frames, visible states with participant_index/attribute/value/"
        "polarity/evidence_frames, and optional state_change with participant_index/"
        "attribute/before/after/evidence_frames. Do not infer causes or intent. "
        "All evidence fields contain integer frame indices. Do not output confidence, "
        "probability, score, reward, or utility. Return strict JSON.\n"
        f"candidate={candidate}\nframe_records={frame_records}"
    )


def _parse_coarse_response(
    payload: dict[str, Any],
    *,
    window: dict[str, Any],
    window_index: int,
    minimum_confidence: float,
    max_events: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_events = payload.get("events")
    if not isinstance(raw_events, list):
        return [], [
            {
                "stage": "coarse",
                "window_index": window_index,
                "reason": "missing events",
            }
        ]
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    start = float(window["start_s"])
    end = float(window["end_s"])
    for local_index, raw in enumerate(raw_events[:max_events]):
        reason: str | None = None
        if not isinstance(raw, dict):
            reason = "event is not an object"
        else:
            predicate = str(raw.get("predicate") or "").strip()
            confidence = _grounding_value(raw)
            event_start = _number(raw.get("coarse_start_s"))
            event_end = _number(raw.get("coarse_end_s"))
            if atomicity_issues(predicate):
                reason = f"non-atomic predicate: {atomicity_issues(predicate)}"
            elif confidence is None or confidence < minimum_confidence:
                reason = "confidence below threshold"
            elif event_start is None or event_end is None:
                reason = "missing coarse timestamps"
            elif event_start < start - 1e-6 or event_end > end + 1e-6:
                reason = "coarse timestamps fall outside sampled window"
            elif event_end <= event_start:
                reason = "coarse interval is empty or reversed"
        if reason is not None:
            rejected.append(
                {
                    "stage": "coarse",
                    "window_index": window_index,
                    "local_index": local_index,
                    "reason": reason,
                }
            )
            continue
        assert isinstance(raw, dict)
        accepted.append(
            {
                **raw,
                "predicate": str(raw["predicate"]).strip(),
                "confidence": float(confidence),
                "coarse_start_s": float(raw["coarse_start_s"]),
                "coarse_end_s": float(raw["coarse_end_s"]),
                "coarse_window": dict(window),
            }
        )
    return accepted, rejected


def _parse_fine_response(
    payload: dict[str, Any],
    *,
    candidate: dict[str, Any],
    frame_records: list[dict[str, Any]],
    minimum_confidence: float,
) -> _LocalizedEvent | None:
    if payload.get("observed") is not True:
        return None
    by_index = {int(record["frame_index"]): record for record in frame_records}
    start_index = _integer(payload.get("visible_start_frame"))
    end_index = _integer(payload.get("visible_end_frame"))
    if start_index not in by_index or end_index not in by_index:
        raise ValueError("localized endpoints cite unknown frames")
    start_s = float(by_index[start_index]["time_s"])
    end_s = float(by_index[end_index]["time_s"])
    if end_s <= start_s:
        raise ValueError("localized event endpoints are empty or temporally reversed")
    evidence = _frame_indices(payload.get("evidence_frames"), by_index)
    if not evidence or start_index not in evidence or end_index not in evidence:
        raise ValueError("localized event lacks endpoint frame evidence")
    predicate = str(
        payload.get("predicate") or candidate.get("predicate") or ""
    ).strip()
    issues = atomicity_issues(predicate)
    if issues:
        raise ValueError(f"localized predicate failed atomicity checks: {issues}")
    confidence = _grounding_value(payload, observed_default=True)
    if confidence is None or confidence < minimum_confidence:
        raise ValueError("localized event confidence is below threshold")
    participants = _parse_visual_participants(payload.get("participants"), by_index)
    states = _parse_visual_states(payload.get("states"), participants, by_index)
    state_change = _parse_state_change(
        payload.get("state_change"),
        participants,
        by_index,
    )
    visible_text = _parse_visible_text(payload.get("visible_text"), by_index)
    return _LocalizedEvent(
        predicate=predicate,
        grounded_caption=_grounded_caption(payload, predicate),
        time_span=TimeSpan(start_s, end_s),
        confidence=confidence,
        action_kind=str(
            payload.get("action_kind") or candidate.get("action_kind") or "other"
        ),
        evidence_frames=evidence,
        frame_records=tuple(frame_records),
        participants=participants,
        states=states,
        state_change=state_change,
        visible_text=visible_text,
        coarse_window=dict(candidate.get("coarse_window") or {}),
        localization_method="coarse_to_fine_frame_grounding",
    )


def _parse_single_pass_candidate(
    candidate: dict[str, Any],
    frame_records: list[dict[str, Any]],
) -> _LocalizedEvent:
    """Create a grounded L1 event without a second per-candidate VLM call."""

    by_index = {int(record["frame_index"]): record for record in frame_records}
    evidence = _frame_indices(candidate.get("evidence_frames"), by_index)
    start_index = _integer(candidate.get("visible_start_frame"))
    end_index = _integer(candidate.get("visible_end_frame"))
    endpoint_evidence = tuple(
        dict.fromkeys(index for index in (start_index, end_index) if index in by_index)
    )
    if not evidence and endpoint_evidence:
        evidence = endpoint_evidence
    if not evidence:
        raise ValueError("single-pass event lacks sampled-frame evidence")
    participants = _parse_visual_participants(candidate.get("participants"), by_index)
    raw_participants = candidate.get("participants")
    if isinstance(raw_participants, list) and raw_participants and not participants:
        raise ValueError("single-pass participants lack sampled-frame evidence")
    states = _parse_visual_states(candidate.get("states"), participants, by_index)
    state_change = _parse_state_change(
        candidate.get("state_change"),
        participants,
        by_index,
    )
    visible_text = _parse_visible_text(candidate.get("visible_text"), by_index)
    grounded_start_s = (
        float(by_index[start_index]["time_s"])
        if start_index in by_index
        and end_index in by_index
        and start_index != end_index
        else float(candidate["coarse_start_s"])
    )
    grounded_end_s = (
        float(by_index[end_index]["time_s"])
        if start_index in by_index
        and end_index in by_index
        and start_index != end_index
        else float(candidate["coarse_end_s"])
    )
    if grounded_end_s <= grounded_start_s:
        raise ValueError("single-pass grounded endpoints are empty or reversed")
    return _LocalizedEvent(
        predicate=str(candidate["predicate"]),
        grounded_caption=_grounded_caption(candidate, str(candidate["predicate"])),
        time_span=TimeSpan(grounded_start_s, grounded_end_s),
        confidence=1.0,
        action_kind=str(candidate.get("action_kind") or "other"),
        evidence_frames=evidence,
        frame_records=tuple(frame_records),
        participants=participants,
        states=states,
        state_change=state_change,
        visible_text=visible_text,
        coarse_window=dict(candidate.get("coarse_window") or {}),
        localization_method="grounded_single_pass_sampled_frames",
    )


def _grounding_value(
    payload: dict[str, Any], *, observed_default: bool = False
) -> float | None:
    """Adapt categorical grounding to the legacy numeric storage field.

    The returned value is a deterministic compatibility marker, never model
    confidence, reward, action utility, or a training target. Legacy persisted
    payloads containing numeric confidence remain readable.
    """

    status = str(payload.get("grounding_status") or "").strip().lower()
    if status == "observed":
        return 1.0
    if status in {"inconclusive", "not_observed", "rejected"}:
        return None
    legacy = _probability(payload.get("confidence"))
    if legacy is not None:
        return legacy
    return 1.0 if observed_default and payload.get("observed") is True else None


def _parse_visual_participants(
    value: Any,
    by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in value[:4]:
        if not isinstance(raw, dict):
            continue
        evidence = _frame_indices(raw.get("evidence_frames"), by_index)
        role = str(raw.get("role") or "other").strip()
        entity_type = str(raw.get("entity_type") or "other").strip()
        surface = str(raw.get("surface") or "").strip()
        if not evidence or not role or not entity_type or not surface:
            continue
        result.append(
            {
                "role": role,
                "entity_type": entity_type,
                "surface": surface,
                "visual_signature": str(raw.get("visual_signature") or "").strip(),
                "confidence": _probability(raw.get("confidence")) or 0.5,
                "evidence_frames": list(evidence),
            }
        )
    return result


def _parse_visual_states(
    value: Any,
    participants: list[dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, Any]] = []
    for raw in value[:6]:
        if not isinstance(raw, dict):
            continue
        participant_index = _integer(raw.get("participant_index"))
        evidence = _frame_indices(raw.get("evidence_frames"), by_index)
        attribute = str(raw.get("attribute") or "").strip()
        state_value = str(raw.get("value") or "").strip()
        if (
            participant_index is None
            or participant_index < 0
            or participant_index >= len(participants)
            or not evidence
            or not attribute
            or not state_value
        ):
            continue
        result.append(
            {
                "participant_index": participant_index,
                "attribute": attribute,
                "value": state_value,
                "polarity": (
                    str(raw.get("polarity"))
                    if raw.get("polarity") in {"positive", "negative"}
                    else "positive"
                ),
                "confidence": _probability(raw.get("confidence")) or 0.5,
                "evidence_frames": list(evidence),
            }
        )
    return result


def _parse_state_change(
    value: Any,
    participants: list[dict[str, Any]],
    by_index: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    participant_index = _integer(value.get("participant_index"))
    evidence = _frame_indices(value.get("evidence_frames"), by_index)
    attribute = str(value.get("attribute") or "").strip()
    before = str(value.get("before") or "").strip()
    after = str(value.get("after") or "").strip()
    if (
        participant_index is None
        or participant_index < 0
        or participant_index >= len(participants)
        or len(evidence) < 2
        or not attribute
        or not before
        or not after
        or before.lower() == after.lower()
    ):
        return None
    times = [float(by_index[index]["time_s"]) for index in evidence]
    if times != sorted(times):
        return None
    return {
        "participant_index": participant_index,
        "attribute": attribute,
        "before": before,
        "after": after,
        "confidence": _probability(value.get("confidence")) or 0.5,
        "evidence_frames": list(evidence),
    }


def _parse_visible_text(
    value: Any,
    by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result = []
    for raw in value[:8]:
        if not isinstance(raw, dict):
            continue
        text = " ".join(str(raw.get("text") or "").split())
        evidence = _frame_indices(raw.get("evidence_frames"), by_index)
        if not text or not evidence:
            continue
        result.append({"text": text[:240], "evidence_frames": list(evidence)})
    return result


def _grounded_caption(payload: dict[str, Any], predicate: str) -> str:
    caption = " ".join(str(payload.get("grounded_caption") or "").split())
    return caption[:600] if caption else predicate


def _assign_track_ids(events: list[_LocalizedEvent], *, max_gap_s: float) -> int:
    tracks: dict[tuple[str, str], list[tuple[float, str]]] = {}
    next_track = 1
    for event in sorted(events, key=lambda item: item.time_span.start_s):
        for participant_index, participant in enumerate(event.participants):
            signature = _norm(str(participant.get("visual_signature") or ""))
            key = (
                _norm(str(participant["entity_type"])),
                signature,
            )
            candidates = tracks.get(key, []) if signature else []
            reusable = next(
                (
                    track_id
                    for end_s, track_id in reversed(candidates)
                    if event.time_span.start_s - end_s <= max_gap_s
                ),
                None,
            )
            if reusable is None:
                reusable = f"track:{next_track:04d}"
                next_track += 1
            participant["mention_id"] = reusable
            participant["track_status"] = (
                "candidate_visual_signature" if signature else "event_local_only"
            )
            participant["local_index"] = participant_index
            tracks.setdefault(key, []).append((event.time_span.end_s, reusable))
        for state in event.states:
            index = int(state.pop("participant_index"))
            state["mention_id"] = event.participants[index]["mention_id"]
        if event.state_change is not None:
            index = int(event.state_change.pop("participant_index"))
            event.state_change["mention_id"] = event.participants[index]["mention_id"]
    return next_track - 1


def _event_to_l1_node(
    event: _LocalizedEvent,
    *,
    video_id: str,
    index: int,
) -> MemoryNode:
    node_id = f"visual_l1:{video_id}:{index:04d}"
    segment_ref = f"raw_video:{video_id}:{event.time_span.start_s:.3f}-{event.time_span.end_s:.3f}"
    return MemoryNode(
        node_id=node_id,
        video_id=video_id,
        time_span=event.time_span,
        provenance={
            "source_graph_id": f"video_l1:{video_id}",
            "source_node_type": "raw_video_frame_window",
            "adapter": "memory_graph.video_l1.QwenVideoL1Extractor",
            "producer": "memory_graph.video_l1.QwenVideoL1Extractor",
            "layer": "L1",
            "uses_hidden_supervision": False,
        },
        node_type="observation",
        text=event.grounded_caption,
        source_node_id=segment_ref,
        source_segments=[segment_ref],
        metadata={
            "predicate": event.predicate,
            "grounded_descriptor": event.grounded_caption,
            "confidence": event.confidence,
            "action_kind": event.action_kind,
            "participants": event.participants,
            "states": event.states,
            "state_change": event.state_change,
            "visible_text": event.visible_text,
            "observed_modalities": ["visual"],
            "entity_track_ids": sorted(
                {str(participant["mention_id"]) for participant in event.participants}
            ),
            "localization": {
                "method": event.localization_method,
                "evidence_frames": list(event.evidence_frames),
                "frame_records": list(event.frame_records),
                "coarse_window": event.coarse_window,
            },
            "visibility": {
                "mode": "video_only",
                "visible_to_agent": True,
                "hidden_supervision": False,
            },
            "source_type": "raw_video_visual_observation",
            "layer": "L1",
        },
    )


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        candidates,
        key=lambda item: (
            float(item["coarse_start_s"]),
            float(item["coarse_end_s"]),
            str(item["predicate"]),
        ),
    )
    result: list[dict[str, Any]] = []
    for candidate in ordered:
        duplicate = any(
            _norm(str(candidate["predicate"])) == _norm(str(existing["predicate"]))
            and _interval_iou(
                float(candidate["coarse_start_s"]),
                float(candidate["coarse_end_s"]),
                float(existing["coarse_start_s"]),
                float(existing["coarse_end_s"]),
            )
            >= 0.5
            for existing in result
        )
        if not duplicate:
            result.append(candidate)
    return result


def _dedupe_localized(events: list[_LocalizedEvent]) -> list[_LocalizedEvent]:
    result: list[_LocalizedEvent] = []
    for event in sorted(
        events, key=lambda item: (item.time_span.start_s, item.predicate)
    ):
        duplicate = any(
            _norm(event.predicate) == _norm(existing.predicate)
            and _interval_iou(
                event.time_span.start_s,
                event.time_span.end_s,
                existing.time_span.start_s,
                existing.time_span.end_s,
            )
            >= 0.5
            for existing in result
        )
        if not duplicate:
            result.append(event)
    return result


def _frame_indices(
    value: Any,
    by_index: dict[int, dict[str, Any]],
) -> tuple[int, ...]:
    if not isinstance(value, list):
        return ()
    result: list[int] = []
    for raw in value:
        index = _integer(raw)
        if index is not None and index in by_index and index not in result:
            result.append(index)
    return tuple(result)


def _interval_iou(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    return intersection / union if union > 0 else 0.0


def _probability(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and 0.0 <= number <= 1.0 else None


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _norm(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _memory_node_from_dict(payload: Any) -> MemoryNode:
    if not isinstance(payload, dict):
        raise ValueError("persisted video L1 node must be an object")
    return MemoryNode(
        node_id=str(payload.get("node_id") or ""),
        video_id=str(payload.get("video_id") or ""),
        time_span=TimeSpan.from_dict(payload.get("time_span") or {}),
        provenance=dict(payload.get("provenance") or {}),
        node_type=str(payload.get("node_type") or "observation"),
        text=str(payload["text"]) if payload.get("text") is not None else None,
        source_node_id=(
            str(payload["source_node_id"]) if payload.get("source_node_id") else None
        ),
        source_segments=[str(value) for value in payload.get("source_segments") or []],
        metadata=dict(payload.get("metadata") or {}),
    )
