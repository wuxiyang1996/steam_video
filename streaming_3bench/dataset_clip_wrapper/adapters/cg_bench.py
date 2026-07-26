"""CG-Bench adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from .base import DatasetAdapter, RawDatasetItem


class CGBenchAdapter(DatasetAdapter):
    name = "cg_bench"

    def __init__(self, dataset_root: Path, split: str = "train", use_mini: bool = True):
        super().__init__(dataset_root, split)
        self.use_mini = use_mini

    def _qa_path(self) -> Path:
        bench = self.dataset_root / "CG-Bench"
        return bench / ("cgbench_mini.json" if self.use_mini else "cgbench.json")

    def _resolve_video_path(self, bench: Path, video_uid: str, qid: str) -> Path | None:
        candidates = [
            bench / "cg_videos" / f"{video_uid}.mp4",
            bench / f"{video_uid}.mp4",
            bench / f"{qid}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        bench = self.dataset_root / "CG-Bench"
        records = json.loads(self._qa_path().read_text(encoding="utf-8"))
        count = 0
        for row in records:
            qid = row["qid"]
            video_uid = row["video_uid"]
            example_id = f"cg_bench:{qid}"
            video_path = self._resolve_video_path(bench, str(video_uid), str(qid))
            if video_path is None:
                continue
            clue_clip = bench / "cg_videos_clue" / f"{qid}.mp4"
            subtitle_dir = bench / "cg_subtitles" / "cg_subtitles"
            subtitle_path = subtitle_dir / f"{video_uid}.srt"
            subtitle_paths = [subtitle_path] if subtitle_path.exists() else []

            annotation_segments = []
            evidence_seeds = []
            for i, interval in enumerate(row.get("clue_intervals") or [], start=1):
                if not interval or len(interval) < 2:
                    continue
                span = {"start_s": float(interval[0]), "end_s": float(interval[1])}
                annotation_segments.append(
                    {
                        "segment_id": f"cg_clue_{i:03d}",
                        "source_type": "clue_interval",
                        "time_span": span,
                        "text": f"gold clue interval for qid={qid}",
                        "provenance": {"field": "clue_intervals"},
                    }
                )
                evidence_seeds.append(
                    {
                        "evidence_id": f"ev:cg:clue:{qid}:{i}",
                        "source_type": "clue_interval",
                        "time_span": span,
                        "text": f"clue interval [{interval[0]}, {interval[1]}]",
                        "trust_level": "gold",
                        "provenance": {"source_field": "clue_intervals"},
                    }
                )
                if clue_clip.exists():
                    evidence_seeds.append(
                        {
                            "evidence_id": f"ev:cg:clip:{qid}",
                            "source_type": "clue_clip",
                            "time_span": span,
                            "text": f"clue clip for qid={qid}",
                            "trust_level": "gold",
                            "media_ref": {"path": str(clue_clip)},
                            "provenance": {"source_field": "cg_videos_clue"},
                        }
                    )

            choices_raw = row.get("choices") or []
            if isinstance(choices_raw, dict):
                options = [{"label": k, "text": v} for k, v in choices_raw.items()]
                answer = row.get("answer")
                answer_text = choices_raw.get(answer)
            else:
                labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                options = [
                    {"label": labels[i], "text": text}
                    for i, text in enumerate(choices_raw)
                ]
                answer = row.get("right_answer") or row.get("answer")
                answer_text = row.get("answer")
                if answer and options:
                    for opt in options:
                        if opt["label"] == answer:
                            answer_text = opt["text"]
                            break
            yield RawDatasetItem(
                dataset=self.name,
                example_id=example_id,
                split=self.split,
                task_family="long_video_clue_grounded_qa",
                video_id=video_uid,
                video_path=video_path,
                duration_s=float(row.get("duration") or 0.0) or None,
                question={
                    "question_id": str(qid),
                    "question_text": row.get("question", ""),
                    "options": options,
                    "answer": {"label": answer, "text": answer_text},
                    "answer_format": "multiple_choice",
                },
                subtitle_paths=subtitle_paths,
                annotation_segments=annotation_segments,
                evidence_seeds=evidence_seeds,
                hidden_supervision_sources=["official_answer", "clue_intervals", "clue_clips"],
                raw_source_refs=[
                    {
                        "source_name": self._qa_path().name,
                        "source_item_id": str(qid),
                        "fields_used": ["question", "answer", "choices", "clue_intervals", "duration"],
                    }
                ],
                metadata={
                    "domain": row.get("domain"),
                    "sub_category": row.get("sub_category"),
                    "clue_clip_path": str(clue_clip) if clue_clip.exists() else None,
                },
            )
            count += 1
            if limit is not None and count >= limit:
                break
