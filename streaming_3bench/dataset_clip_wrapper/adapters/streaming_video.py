"""StreamBridge-style OVO-Bench, VideoMME, and StreamingBench adapters."""

from __future__ import annotations

import json
import re
import csv
import ast
from pathlib import Path
from typing import Any, Iterator

from .base import DatasetAdapter, RawDatasetItem


STREAMBRIDGE_ASSET_ROOT = Path("/mnt/is_data/xwu/video_skills/code/ml-streambridge/assets")
CLUSTER_OVO_BENCH_ROOT = Path("/net/mlfs01/export/users/dpatel/OVO-Bench")
CLUSTER_VIDEOMME_ROOT = Path("/net/nj-storage02/mnt/tank/datasets/WHB139426-Grounded-VideoLLM/videomme")


def _first_existing(candidates: list[Path]) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _label_from_index(index: int) -> str:
    return chr(ord("A") + index)


def _normalize_options(options: Any) -> list[dict[str, str]]:
    if hasattr(options, "tolist") and not isinstance(options, (str, dict, list, tuple)):
        options = options.tolist()
    if isinstance(options, tuple):
        options = list(options)
    if isinstance(options, str):
        text = options.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = [part.strip() for part in re.split(r"\s*[A-D]\.\s*", text) if part.strip()]
        return _normalize_options(parsed)
    if isinstance(options, dict):
        return [{"label": str(label), "text": str(text)} for label, text in options.items()]
    if not isinstance(options, list):
        return []

    normalized = []
    for idx, option in enumerate(options):
        label = _label_from_index(idx)
        text = str(option)
        if len(text) >= 3 and text[1:3] in {". ", ") "} and text[0].isalpha():
            label = text[0].upper()
            text = text[3:].strip()
        normalized.append({"label": label, "text": text})
    return normalized


def _answer_from_row(row: dict[str, Any], options: list[dict[str, str]]) -> dict[str, str | None]:
    answer = row.get("answer")
    if answer is None:
        answer = row.get("ground_truth_output") or row.get("ground_truth")
    gt = row.get("gt")
    if isinstance(gt, int) and 0 <= gt < len(options):
        return {"label": options[gt]["label"], "text": options[gt]["text"]}
    if isinstance(answer, str):
        stripped = answer.strip()
        if len(stripped) == 1 and stripped.isalpha():
            label = stripped.upper()
            return {"label": label, "text": next((opt["text"] for opt in options if opt["label"] == label), None)}
        for opt in options:
            if stripped == opt["text"] or stripped.lower() == opt["text"].lower():
                return {"label": opt["label"], "text": opt["text"]}
        return {"label": None, "text": stripped}
    return {"label": None, "text": None}


def _parse_timestamp_s(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return float(text)
    parts = text.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    except ValueError:
        return None
    return None


def _load_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("data", "questions", "records", "examples"):
                if isinstance(payload.get(key), list):
                    return [row for row in payload[key] if isinstance(row, dict)]
            return [payload]
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
    if suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    if suffix == ".parquet":
        import pandas as pd  # type: ignore

        return [dict(row) for row in pd.read_parquet(path).to_dict(orient="records")]
    raise ValueError(f"unsupported annotation file: {path}")


class OVOBenchAdapter(DatasetAdapter):
    """OVO-Bench format adapter.

    The local workspace currently provides StreamBridge tiny annotations. The
    adapter keeps the same fields as the official StreamBridge OVO path:
    ``video``, ``realtime``, ``question``, ``options``, ``gt`` and ``answer``.
    """

    name = "ovo_bench"

    def _root(self) -> Path:
        return _first_existing(
            [
                self.dataset_root / "OVO-Bench" / "data",
                self.dataset_root / "OVO-Bench",
                self.dataset_root / "streambridge_tiny",
                CLUSTER_OVO_BENCH_ROOT,
            ]
        )

    def _qa_path(self) -> Path:
        tiny_root = self.dataset_root / "streambridge_tiny"
        return _first_existing(
            [
                self.dataset_root / "OVO-Bench" / "data" / "ovo_bench_new.json",
                self.dataset_root / "OVO-Bench" / "data" / "ovo_bench.json",
                tiny_root / "tiny_ovo_bench_50videos.json",
                tiny_root / "tiny_ovo_bench.json",
                self.dataset_root / "ovo_bench.json",
                self.dataset_root / "OVO-Bench" / "ovo_bench.json",
                STREAMBRIDGE_ASSET_ROOT / "ovo_bench.json",
            ]
        )

    def _video_path(self, row: dict[str, Any], qid: str) -> Path | None:
        root = self._root()
        video_name = str(row.get("video") or "")
        candidates = [
            self.dataset_root / "streambridge_tiny" / "videos" / video_name,
            root / "videos" / video_name,
            root / "chunked_videos" / f"{qid}.mp4",
            root / "chunked_videos" / f"{Path(video_name).stem}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        chunk_matches = sorted(
            (root / "chunked_videos").glob(f"{qid}_*.mp4"),
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]) if path.stem.rsplit("_", 1)[-1].isdigit() else path.stem,
        )
        if chunk_matches:
            return chunk_matches[0]
        return None

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        qa_path = self._qa_path()
        records = json.loads(qa_path.read_text(encoding="utf-8"))
        count = 0
        for row in records:
            video_name = str(row.get("video") or "")
            realtime = row.get("realtime")
            realtime_s = float(realtime) if realtime is not None else None
            qid = str(row["id"]) if "id" in row else f"{Path(video_name).stem}:{count}"
            video_path = self._video_path(row, qid)
            video_id = video_path.stem if video_path else Path(video_name).stem
            options = _normalize_options(row.get("options"))
            answer = _answer_from_row(row, options)

            yield RawDatasetItem(
                dataset=self.name,
                example_id=f"ovo_bench:{qid}",
                split=self.split,
                task_family="streaming_video_realtime_qa",
                video_id=video_id,
                video_path=video_path,
                duration_s=None,
                question={
                    "question_id": qid,
                    "question_text": row.get("question", ""),
                    "question_type": row.get("task"),
                    "options": options,
                    "answer": answer,
                    "answer_format": "multiple_choice",
                    "time_anchor_s": realtime_s,
                },
                subtitle_paths=[],
                annotation_segments=[],
                evidence_seeds=[],
                hidden_supervision_sources=["official_answer"],
                raw_source_refs=[
                    {
                        "source_name": self._qa_path().name,
                        "source_item_id": qid,
                        "fields_used": ["video", "task", "realtime", "question", "options", "gt", "answer"],
                    }
                ],
                metadata={
                    "benchmark_family": "streaming_video_benchmarks",
                    "benchmark_format": "ovo_bench",
                    "streambridge_compatible": True,
                    "realtime_s": realtime_s,
                    "annotation_path": str(qa_path),
                },
            )
            count += 1
            if limit is not None and count >= limit:
                break


class VideoMMEAdapter(DatasetAdapter):
    """VideoMME format adapter for StreamBridge-style records."""

    name = "videomme"

    def _root(self) -> Path:
        return _first_existing(
            [
                self.dataset_root / "Video-MME",
                self.dataset_root / "videomme",
                self.dataset_root / "VideoMME",
                self.dataset_root / "streambridge_tiny",
                CLUSTER_VIDEOMME_ROOT,
            ]
        )

    def _qa_path(self) -> Path:
        tiny_root = self.dataset_root / "streambridge_tiny"
        return _first_existing(
            [
                self.dataset_root / "Video-MME" / "videomme" / "test-00000-of-00001.parquet",
                self.dataset_root / "Video-MME" / "videomme.json",
                tiny_root / "tiny_videomme.json",
                self.dataset_root / "videomme.json",
                self.dataset_root / "videomme" / "videomme.json",
                self.dataset_root / "VideoMME" / "videomme.json",
                STREAMBRIDGE_ASSET_ROOT / "videomme.json",
            ]
        )

    def _video_path(self, video_id: str) -> Path | None:
        root = self._root()
        candidates = [
            self.dataset_root / "streambridge_tiny" / "videos" / f"{video_id}.mp4",
            root / "videos" / f"{video_id}.mp4",
            root / "data" / f"{video_id}.mp4",
            root / f"{video_id}.mp4",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _subtitle_paths(self, video_id: str) -> list[Path]:
        root = self._root()
        candidates = [
            root / "subtitle" / f"{video_id}.srt",
            root / "subtitles" / f"{video_id}.srt",
            self.dataset_root / "Video-MME" / "subtitle" / f"{video_id}.srt",
        ]
        return list(dict.fromkeys(candidate for candidate in candidates if candidate.exists()))

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        qa_path = self._qa_path()
        records = _load_records(qa_path)
        count = 0
        for row in records:
            video_id = str(row.get("videoID") or "")
            qid = str(row.get("id") or row.get("question_id") or f"{video_id}:{count}")
            options = _normalize_options(row.get("options"))
            answer = _answer_from_row(row, options)
            video_path = self._video_path(video_id)

            yield RawDatasetItem(
                dataset=self.name,
                example_id=f"videomme:{qid}",
                split=self.split,
                task_family="streaming_video_whole_video_qa",
                video_id=video_id,
                video_path=video_path,
                duration_s=None,
                question={
                    "question_id": qid,
                    "question_text": row.get("question", ""),
                    "question_type": row.get("duration"),
                    "options": options,
                    "answer": answer,
                    "answer_format": "multiple_choice",
                },
                subtitle_paths=self._subtitle_paths(video_id),
                annotation_segments=[],
                evidence_seeds=[],
                hidden_supervision_sources=["official_answer"],
                raw_source_refs=[
                    {
                        "source_name": self._qa_path().name,
                        "source_item_id": qid,
                        "fields_used": ["videoID", "duration", "question", "options", "answer"],
                    }
                ],
                metadata={
                    "benchmark_family": "streaming_video_benchmarks",
                    "benchmark_format": "videomme",
                    "streambridge_compatible": True,
                    "duration_bucket": row.get("duration"),
                    "annotation_path": str(qa_path),
                },
            )
            count += 1
            if limit is not None and count >= limit:
                break


class StreamingBenchAdapter(DatasetAdapter):
    """StreamingBench adapter.

    Expected layouts include either the official preprocessed form:

    ``StreamingBench/questions_real.json`` and ``StreamingBench/videos/``

    or Hugging Face exports under ``StreamingBench`` with JSON/JSONL/Parquet
    annotation files. The adapter is deliberately schema-tolerant because HF
    previews expose fields such as ``video_path``, ``question``, ``options``,
    ``answer`` and ``time``/``timestamp`` while local preprocessing may choose
    slightly different names.
    """

    name = "streaming_bench"

    def __init__(self, dataset_root: Path, split: str = "train"):
        super().__init__(dataset_root, split)
        self._video_files_cache: list[Path] | None = None
        self._video_name_index: dict[str, list[Path]] | None = None
        self._sample_dir_index: dict[str, list[Path]] | None = None

    def _root(self) -> Path:
        return _first_existing(
            [
                self.dataset_root / "StreamingBench",
                self.dataset_root / "streamingbench",
                self.dataset_root / "streaming_bench",
            ]
        )

    def _qa_paths(self) -> list[Path]:
        root = self._root()
        candidates = [
            root / "questions_real.json",
            root / "questions_omni.json",
            root / "questions_sqa.json",
            root / "questions_proactive.json",
            root / "questions_proactive_50.json",
            root / "src" / "data" / "questions_real.json",
            root / "src" / "data" / "questions_omni.json",
            root / "src" / "data" / "questions_sqa.json",
            root / "src" / "data" / "questions_proactive.json",
            root / "src" / "data" / "questions_proactive_50.json",
            root / "questions.json",
            root / "annotations.json",
            root / "data.json",
            root / "test.json",
            root / "validation.json",
            root / "train.json",
            root / "questions_real.jsonl",
            root / "questions.jsonl",
        ]
        candidates.extend(sorted(root.glob("*.csv")))
        candidates.extend(sorted(root.glob("StreamingBench/*.csv")))
        candidates.extend(sorted(root.glob("src/data/*.csv")))
        candidates.extend(sorted(root.glob("*.parquet")))
        candidates.extend(sorted(root.glob("**/questions_*.json")))
        candidates.extend(sorted(root.glob("**/*.csv")))
        candidates.extend(sorted(root.glob("**/*.parquet")))
        seen = set()
        paths = []
        for candidate in candidates:
            try:
                rel_parts = candidate.relative_to(root).parts
            except ValueError:
                rel_parts = candidate.parts
            if any(part.startswith(".") for part in rel_parts) or any(
                part in {"archives", "extracted", "videos"} for part in rel_parts
            ):
                continue
            if candidate.exists() and candidate not in seen:
                paths.append(candidate)
                seen.add(candidate)
        return paths

    def _video_name(self, row: dict[str, Any]) -> str:
        for key in ("video_path", "video", "video_name", "video_file", "filename", "path"):
            value = row.get(key)
            if value:
                return str(value)
        video_id = row.get("video_id") or row.get("videoID") or row.get("id")
        if not video_id:
            qid = row.get("question_id") or row.get("sample_id") or row.get("qid") or row.get("uid")
            video_id = self._video_id_from_question_id(str(qid or ""))
        return str(video_id or "")

    def _video_id_from_question_id(self, qid: str) -> str:
        text = qid.strip()
        match = re.match(r"(.+?_sample_\d+)(?:_\d+)?$", text)
        if match:
            return match.group(1)
        return text

    def _video_stem_candidates(self, video_name: str, row: dict[str, Any]) -> list[str]:
        candidates: list[str] = []
        for raw in [
            video_name,
            Path(video_name).stem,
            self._video_id_from_question_id(str(row.get("question_id") or "")),
            str(row.get("question_id") or ""),
        ]:
            raw = raw.strip()
            if not raw:
                continue
            candidates.extend(
                [
                    raw,
                    raw.replace(" ", "_"),
                    raw.replace("_", " "),
                    re.sub(r"[^A-Za-z0-9_.-]+", "_", raw).strip("_"),
                ]
            )
        seen = set()
        unique = []
        for value in candidates:
            if value and value not in seen:
                unique.append(value)
                seen.add(value)
        return unique

    def _sample_dir_candidates(self, row: dict[str, Any]) -> list[str]:
        values = [
            str(row.get("question_id") or ""),
            str(row.get("sample_id") or ""),
            str(row.get("video_id") or ""),
            self._video_name(row),
        ]
        candidates = []
        for value in values:
            match = re.search(r"sample[_ ](\d+)", value, flags=re.IGNORECASE)
            if match:
                candidates.append(f"sample_{int(match.group(1))}")
            elif re.search(r"_\d+$", value):
                candidates.append(f"sample_{int(value.rsplit('_', 1)[1])}")
        seen = set()
        unique = []
        for value in candidates:
            if value not in seen:
                unique.append(value)
                seen.add(value)
        return unique

    def _rank_video_candidate(self, candidate: Path, row: dict[str, Any]) -> int:
        text = " ".join(
            [
                str(row.get("task_type") or ""),
                str(row.get("task") or ""),
                str(row.get("question_type") or ""),
                str(row.get("category") or ""),
                str(row.get("question_id") or ""),
            ]
        ).lower()
        words = {word for word in re.split(r"[^a-z0-9]+", text) if len(word) >= 4}
        path_text = str(candidate).lower()
        return sum(1 for word in words if word in path_text)

    def _video_indexes(self) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
        if self._video_name_index is not None and self._sample_dir_index is not None:
            return self._video_name_index, self._sample_dir_index

        root = self._root()
        search_roots = [candidate for candidate in (root / "extracted", root / "videos", root / "data") if candidate.exists()]
        if not search_roots:
            search_roots = [root]
        seen_paths: set[Path] = set()
        video_files: list[Path] = []
        for search_root in search_roots:
            if not search_root.exists():
                continue
            for suffix in ("*.mp4", "*.mkv", "*.webm", "*.mov"):
                for path in search_root.rglob(suffix):
                    if "__MACOSX" in path.parts or path.name.startswith("._"):
                        continue
                    if path not in seen_paths:
                        video_files.append(path)
                        seen_paths.add(path)

        name_index: dict[str, list[Path]] = {}
        sample_index: dict[str, list[Path]] = {}
        for path in video_files:
            for key in {path.name, path.stem, path.stem.replace(" ", "_"), path.stem.replace("_", " ")}:
                if key:
                    name_index.setdefault(key, []).append(path)
            for part in path.parts:
                match = re.fullmatch(r"sample[_ ](\d+)", part, flags=re.IGNORECASE)
                if match:
                    sample_index.setdefault(f"sample_{int(match.group(1))}", []).append(path)

        self._video_files_cache = video_files
        self._video_name_index = name_index
        self._sample_dir_index = sample_index
        return name_index, sample_index

    def _video_path(self, row: dict[str, Any], annotation_dir: Path | None = None) -> Path | None:
        root = self._root()
        video_name = self._video_name(row)
        if not video_name:
            return None
        path = Path(video_name)
        stems = [path.name]
        if path.suffix:
            stems.append(path.stem)
        else:
            stems.extend([f"{video_name}.mp4", f"{video_name}.mkv", f"{video_name}.webm"])
        for stem in self._video_stem_candidates(video_name, row):
            stems.extend([stem, f"{stem}.mp4", f"{stem}.mkv", f"{stem}.webm"])
        candidates = []
        if path.is_absolute():
            candidates.append(path)
        search_roots = [root]
        if annotation_dir is not None:
            search_roots.append(annotation_dir)
        search_roots.extend([root / "src" / "data", root / "data"])
        for search_root in search_roots:
            candidates.extend(
                [
                    search_root / video_name,
                    search_root / "videos" / video_name,
                    search_root / "data" / video_name,
                    search_root / "real" / video_name,
                    search_root / "omni" / video_name,
                    search_root / "sqa" / video_name,
                    search_root / "proactive" / video_name,
                ]
            )
        candidates.extend(
            [
                root / video_name,
                root / "videos" / video_name,
                root / "data" / video_name,
                root / "real" / video_name,
                root / "omni" / video_name,
                root / "sqa" / video_name,
                root / "proactive" / video_name,
            ]
        )
        for stem in stems:
            for search_root in search_roots:
                candidates.extend(
                    [
                        search_root / "videos" / stem,
                        search_root / "data" / "real" / stem,
                        search_root / "data" / "omni" / stem,
                        search_root / "data" / "sqa" / stem,
                        search_root / "data" / "proactive" / stem,
                        search_root / "real" / stem,
                        search_root / "omni" / stem,
                        search_root / "sqa" / stem,
                        search_root / "proactive" / stem,
                    ]
                )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        name_index, sample_index = self._video_indexes()
        indexed_matches: list[Path] = []
        for stem in stems:
            indexed_matches.extend(name_index.get(stem, []))
        for stem in self._video_stem_candidates(video_name, row):
            indexed_matches.extend(name_index.get(stem, []))
            for suffix in (".mp4", ".mkv", ".webm", ".mov"):
                indexed_matches.extend(name_index.get(f"{stem}{suffix}", []))
        if indexed_matches and not set(self._sample_dir_candidates(row)):
            indexed_matches.sort(key=lambda candidate: self._rank_video_candidate(candidate, row), reverse=True)
            return indexed_matches[0]
        sample_matches = []
        for sample_dir in self._sample_dir_candidates(row):
            sample_matches.extend(sample_index.get(sample_dir, []))
        if sample_matches:
            sample_matches.sort(key=lambda candidate: self._rank_video_candidate(candidate, row), reverse=True)
            return sample_matches[0]
        if indexed_matches:
            indexed_matches.sort(key=lambda candidate: self._rank_video_candidate(candidate, row), reverse=True)
            return indexed_matches[0]
        return None

    def _question_id(self, row: dict[str, Any], count: int) -> str:
        for key in ("question_id", "sample_id", "id", "qid", "uid"):
            value = row.get(key)
            if value is not None:
                return str(value)
        return f"streaming_bench:{count}"

    def iter_items(self, limit: int | None = None) -> Iterator[RawDatasetItem]:
        qa_paths = self._qa_paths()
        if not qa_paths:
            raise FileNotFoundError(
                "StreamingBench annotation not found. Expected one of "
                "StreamingBench/questions_*.json, questions.json, JSONL, or Parquet files "
                f"under dataset root {self.dataset_root}."
            )
        count = 0
        for qa_path in qa_paths:
            records = _load_records(qa_path)
            for row in records:
                qid = self._question_id(row, count)
                video_path = self._video_path(row, annotation_dir=qa_path.parent)
                video_name = self._video_name(row)
                video_id = str(
                    row.get("video_id") or row.get("videoID") or (video_path.stem if video_path else Path(video_name).stem)
                )
                options = _normalize_options(row.get("options") or row.get("choices"))
                answer = _answer_from_row(row, options)
                timestamp_s = _parse_timestamp_s(
                    row.get("time")
                    or row.get("timestamp")
                    or row.get("time_stamp")
                    or row.get("time_point")
                    or row.get("query_time")
                    or row.get("realtime")
                )
                duration_s = _parse_timestamp_s(row.get("duration") or row.get("duration_s"))
                task = row.get("task") or row.get("task_type") or row.get("question_type") or row.get("category")

                yield RawDatasetItem(
                    dataset=self.name,
                    example_id=f"streaming_bench:{qid}",
                    split=self.split,
                    task_family="streaming_video_realtime_qa",
                    video_id=video_id,
                    video_path=video_path,
                    duration_s=duration_s,
                    question={
                        "question_id": qid,
                        "question_text": row.get("question", ""),
                        "question_type": task,
                        "options": options,
                        "answer": answer,
                        "answer_format": "multiple_choice",
                        "time_anchor_s": timestamp_s,
                    },
                    subtitle_paths=[],
                    annotation_segments=[],
                    evidence_seeds=[],
                    hidden_supervision_sources=["official_answer"],
                    raw_source_refs=[
                        {
                        "source_name": qa_path.name,
                        "source_item_id": qid,
                        "source_path": str(qa_path),
                        "fields_used": sorted(row.keys()),
                    }
                ],
                    metadata={
                        "benchmark_family": "streaming_video_benchmarks",
                        "benchmark_format": "streaming_bench",
                        "streamingbench_compatible": True,
                        "query_time_s": timestamp_s,
                        "task": task,
                        "annotation_path": str(qa_path),
                    },
                )
                count += 1
                if limit is not None and count >= limit:
                    return
