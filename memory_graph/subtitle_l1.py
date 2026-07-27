"""Question-independent, time-aligned subtitle enrichment for visual L1 nodes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Mapping


SUBTITLE_ENRICHMENT_SCHEMA = "time-aligned-subtitle-l1/v0.1"
_TIMESTAMP = re.compile(
    r"(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})[,.](?P<ms>\d{3})"
)


@dataclass(frozen=True)
class SubtitleCue:
    start_s: float
    end_s: float
    text: str

    def __post_init__(self) -> None:
        if self.start_s < 0 or self.end_s <= self.start_s or not self.text.strip():
            raise ValueError("invalid subtitle cue")


def parse_srt(path: str | Path) -> tuple[SubtitleCue, ...]:
    source = Path(path).expanduser().resolve()
    blocks = re.split(r"\r?\n\s*\r?\n", source.read_text(encoding="utf-8-sig"))
    cues: list[SubtitleCue] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        timing_index = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing_index is None or timing_index + 1 >= len(lines):
            continue
        left, right = (part.strip() for part in lines[timing_index].split("-->", 1))
        start_s = _seconds(left)
        end_s = _seconds(right.split()[0])
        text = " ".join(lines[timing_index + 1 :]).strip()
        if end_s > start_s and text:
            cues.append(SubtitleCue(start_s, end_s, text))
    return tuple(sorted(cues, key=lambda cue: (cue.start_s, cue.end_s, cue.text)))


def enrich_video_l1_payload_with_subtitles(
    payload: Mapping[str, Any],
    subtitle_path: str | Path,
) -> dict[str, Any]:
    """Fuse overlapping cues while retaining modality and source provenance.

    The bounded semantic key is safe pre-read routing metadata.  The complete
    aligned transcript remains in the grounded value revealed by a real read.
    No question, answer, clue interval, or model-generated relevance is used.
    """

    source = Path(subtitle_path).expanduser().resolve()
    cues = parse_srt(source)
    source_checksum = hashlib.sha256(source.read_bytes()).hexdigest()
    enriched = deepcopy(dict(payload))
    enriched_nodes = []
    aligned_node_count = 0
    for raw in enriched.get("nodes") or ():
        node = deepcopy(raw)
        span = node.get("time_span") or {}
        start_s = float(span.get("start_s") or 0.0)
        end_s = float(span.get("end_s") or 0.0)
        aligned = tuple(
            cue
            for cue in cues
            if cue.start_s < end_s and start_s < cue.end_s
        )
        metadata = dict(node.get("metadata") or {})
        predicate = str(metadata.get("predicate") or node.get("text") or "event").strip()
        if aligned:
            aligned_node_count += 1
            transcript = " ".join(dict.fromkeys(cue.text for cue in aligned))
            metadata["transcript"] = transcript
            metadata["aligned_subtitle_cues"] = [
                {"start_s": cue.start_s, "end_s": cue.end_s, "text": cue.text}
                for cue in aligned
            ]
            metadata["semantic_key"] = _bounded(
                " | ".join(
                    value
                    for value in (predicate, f"aligned subtitle: {transcript}")
                    if value
                ),
                480,
            )
            metadata["observed_modalities"] = list(
                dict.fromkeys((*metadata.get("observed_modalities", ()), "visual", "subtitle"))
            )
            node["text"] = _bounded(
                f"{predicate}. Time-aligned subtitle: {transcript}", 2000
            )
            # Descriptor text changed; retaining the old row would make L1.5
            # correlations inconsistent with the persisted L1 evidence.
            node["embedding_ref"] = None
            provenance = dict(node.get("provenance") or {})
            provenance["subtitle_enrichment"] = {
                "schema_version": SUBTITLE_ENRICHMENT_SCHEMA,
                "source_path": str(source),
                "source_sha256": source_checksum,
                "alignment": "time_interval_overlap",
                "question_independent": True,
            }
            node["provenance"] = provenance
        node["metadata"] = metadata
        enriched_nodes.append(node)
    enriched["nodes"] = enriched_nodes
    enriched["descriptor_enrichment"] = {
        "schema_version": SUBTITLE_ENRICHMENT_SCHEMA,
        "source_path": str(source),
        "source_sha256": source_checksum,
        "cue_count": len(cues),
        "aligned_node_count": aligned_node_count,
        "question_independent": True,
        "uses_question_or_hidden_supervision": False,
    }
    windowing = dict(enriched.get("windowing") or {})
    windowing["descriptor_schema"] = SUBTITLE_ENRICHMENT_SCHEMA
    windowing["subtitle_enrichment"] = "time_aligned_when_available"
    enriched["windowing"] = windowing
    return enriched


def _seconds(value: str) -> float:
    match = _TIMESTAMP.fullmatch(value.strip())
    if match is None:
        raise ValueError(f"invalid SRT timestamp: {value}")
    return (
        int(match.group("h")) * 3600
        + int(match.group("m")) * 60
        + int(match.group("s"))
        + int(match.group("ms")) / 1000.0
    )


def _bounded(value: str, maximum: int) -> str:
    compact = " ".join(value.split())
    return compact if len(compact) <= maximum else compact[: maximum - 1].rstrip() + "…"
