"""Subtitle parsing helpers."""

from __future__ import annotations

import re
from pathlib import Path


def _timestamp_to_seconds(value: str) -> float:
    value = value.strip().replace(",", ".")
    pieces = [float(p) for p in value.split(":")]
    if len(pieces) == 3:
        return pieces[0] * 3600 + pieces[1] * 60 + pieces[2]
    if len(pieces) == 2:
        return pieces[0] * 60 + pieces[1]
    return pieces[0]


def parse_srt(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", text.strip())
    spans: list[dict] = []
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue
        timing_idx = 1 if re.match(r"^\d+$", lines[0]) else 0
        if timing_idx >= len(lines):
            continue
        timing = lines[timing_idx]
        match = re.match(r"(.+?)\s*-->\s*(.+)", timing)
        if not match:
            continue
        start_s = _timestamp_to_seconds(match.group(1))
        end_s = _timestamp_to_seconds(match.group(2))
        content = " ".join(lines[timing_idx + 1 :])
        spans.append(
            {
                "segment_id": f"srt_{len(spans):04d}",
                "source_type": "subtitle_span",
                "time_span": {"start_s": start_s, "end_s": end_s},
                "text": content,
                "provenance": {"source_file": str(path), "format": "srt"},
            }
        )
    return spans
