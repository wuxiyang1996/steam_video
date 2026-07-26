"""SIV-Bench adapter."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterator

from .base import DatasetAdapter, RawDatasetItem


def _parse_options(options_raw: str) -> list[dict[str, str]]:
    text = options_raw.strip()
    if not text:
        return []

    if "|" in text:
        parts = [part.strip() for part in text.split("|") if part.strip()]
    else:
        matches = re.findall(r"(?:^|,\s*)([A-Z])\.\s*(.*?)(?=,\s*[A-Z]\.\s*|$)", text)
        if matches:
            return [{"label": label.strip(), "text": option.strip()} for label, option in matches if option.strip()]
        parts = [text]

    options = []
    for part in parts:
        if "." in part[:3]:
            label, option_text = part.split(".", 1)
            options.append({"label": label.strip(), "text": option_text.strip()})
        else:
            options.append({"label": str(len(options)), "text": part})
    return options


class SIVBenchAdapter(DatasetAdapter):
    name = "siv_bench"

    def _resolve_video_path(self, row: dict[str, str]) -> Path | None:
        bench = self.dataset_root / "SIV-Bench"
        candidates = [
            bench / row.get("video_path", ""),
            bench / row.get("video", ""),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        video_name = Path(row.get("video_path") or row.get("video") or "").name
        if not video_name:
            return None
        for folder in ("wo_sub", "w_sub", "origin"):
            for path in (bench / folder).rglob(video_name):
                return path
        return None

    def _resolve_subtitle_path(self, video_path: Path | None) -> Path | None:
        if video_path is None:
            return None
        srt = video_path.with_suffix(".srt")
        if srt.exists():
            return srt
        alt = video_path.parent / "subtitles" / f"{video_path.stem}.srt"
        return alt if alt.exists() else None

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        bench = self.dataset_root / "SIV-Bench"
        qa_path = bench / "SIV-Bench-QA.tsv"
        count = 0
        with qa_path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                video_path = self._resolve_video_path(row)
                subtitle_path = self._resolve_subtitle_path(video_path)
                qid = row.get("question_id") or row.get("index")
                example_id = f"siv_bench:{row.get('index')}:{qid}"
                options_raw = row.get("options", "")
                options = _parse_options(options_raw)
                answer_idx = row.get("correct_answer_index")
                answer_label = None
                answer_text = row.get("answer")
                if answer_idx is not None and options:
                    answer_key = answer_idx.strip()
                    by_label = {option["label"]: option for option in options}
                    if answer_key in by_label:
                        answer_label = answer_key
                        answer_text = by_label[answer_key]["text"]
                    else:
                        try:
                            idx = int(answer_key)
                            if 0 <= idx < len(options):
                                answer_label = options[idx]["label"]
                                answer_text = options[idx]["text"]
                        except ValueError:
                            pass
                yield RawDatasetItem(
                    dataset=self.name,
                    example_id=example_id,
                    split=self.split,
                    task_family="short_video_social_interaction_qa",
                    video_id=Path(row.get("video_path") or row.get("video") or example_id).stem,
                    video_path=video_path,
                    duration_s=None,
                    question={
                        "question_id": str(qid),
                        "question_text": row.get("question", ""),
                        "options": options,
                        "answer": {"label": answer_label, "text": answer_text},
                        "answer_format": "multiple_choice",
                    },
                    subtitle_paths=[subtitle_path] if subtitle_path else [],
                    annotation_segments=[],
                    evidence_seeds=[],
                    hidden_supervision_sources=["official_answer"],
                    raw_source_refs=[
                        {
                            "source_name": "SIV-Bench-QA.tsv",
                            "source_item_id": example_id,
                            "fields_used": ["question", "answer", "options", "category"],
                        }
                    ],
                    metadata={"category": row.get("category")},
                )
                count += 1
                if limit is not None and count >= limit:
                    break
