#!/usr/bin/env python3
"""Stage StreamingBench under the shared video-skills dataset root.

By default this downloads only annotation CSVs. Pass ``--include-videos`` to
also download and extract the official zip archives, which are large.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shutil
import zipfile
from pathlib import Path


REPO_ID = "mjuicem/StreamingBench"
ANNOTATION_FILES = [
    "StreamingBench/Contextual_Understanding.csv",
    "StreamingBench/Omni_Source_Understanding.csv",
    "StreamingBench/Proactive_Output.csv",
    "StreamingBench/Proactive_Output_50.csv",
    "StreamingBench/Real_Time_Visual_Understanding.csv",
    "StreamingBench/Sequential_Question_Answering.csv",
]
VIDEO_ZIPS = [
    "Anomaly Context Understanding.zip",
    "Emotion Recognition.zip",
    "Misleading Context Understanding.zip",
    "Multimodal Alignment.zip",
    "Proactive Output_1-25.zip",
    "Proactive Output_26-50.zip",
    "Real-Time Visual Understanding_1-50.zip",
    "Real-Time Visual Understanding_51-100.zip",
    "Real-Time Visual Understanding_101-150.zip",
    "Real-Time Visual Understanding_151-200.zip",
    "Real-Time Visual Understanding_201-250.zip",
    "Real-Time Visual Understanding_251-300.zip",
    "Real-Time Visual Understanding_301-350.zip",
    "Real-Time Visual Understanding_351-400.zip",
    "Real-Time Visual Understanding_401-450.zip",
    "Real-Time Visual Understanding_451-500.zip",
    "Scene Understanding_1-25.zip",
    "Scene Understanding_26-50.zip",
    "Sequential Question Answering_1-25.zip",
    "Sequential Question Answering_26-50.zip",
    "Source Discrimination.zip",
]


def safe_extract(zip_path: Path, output_dir: Path) -> list[str]:
    extracted: list[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if not str(target).startswith(str(root)):
                raise RuntimeError(f"refusing unsafe zip member: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(str(target))
    return extracted


def archive_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")


def build_video_index(root: Path) -> dict[str, str]:
    videos_dir = root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    index: dict[str, str] = {}
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".mp4", ".mkv", ".webm", ".mov"}:
            continue
        index.setdefault(path.stem, str(path))
        normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_")
        index.setdefault(normalized, str(path))
        link = videos_dir / path.name
        if not link.exists() and path.resolve() != link.resolve():
            try:
                link.symlink_to(path)
            except OSError:
                pass
    return index


def stage_video_zip(
    *,
    repo_id: str,
    filename: str,
    archive_dir: Path,
    extract_dir: Path,
    force_extract: bool,
    cleanup_archives: bool,
) -> dict[str, object]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(archive_dir),
        )
    )
    zip_extract_dir = extract_dir / archive_stem(path)
    marker = extract_dir / f".extracted_{archive_stem(path)}.json"
    if marker.exists() and not force_extract:
        if cleanup_archives and path.exists():
            path.unlink()
            return {"zip": str(path), "status": "already_extracted_archive_removed", "files": 0}
        return {"zip": str(path), "status": "already_extracted", "files": 0}
    extracted = safe_extract(path, zip_extract_dir)
    marker.write_text(json.dumps({"zip": str(path), "files": len(extracted)}, indent=2) + "\n", encoding="utf-8")
    if cleanup_archives and path.exists():
        path.unlink()
        return {"zip": str(path), "status": "extracted_archive_removed", "files": len(extracted)}
    return {"zip": str(path), "status": "extracted", "files": len(extracted)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", default="/mnt/is_data/xwu/video_skills/data/datasets")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--include-videos", action="store_true")
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cleanup-archives", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    dataset_root = Path(args.dataset_root)
    root = dataset_root / "StreamingBench"
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", "/mnt/is_data/xwu/video_skills/data/models/hf_cache")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/mnt/is_data/xwu/video_skills/data/models/hf_cache/hub")

    manifest = {
        "repo_id": args.repo_id,
        "root": str(root),
        "annotations": [],
        "zips": [],
        "extracted_files": 0,
        "include_videos": args.include_videos,
        "workers": args.workers,
        "cleanup_archives": args.cleanup_archives,
    }

    for filename in ANNOTATION_FILES:
        path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(root),
        )
        manifest["annotations"].append(path)
        print(f"annotation: {path}", flush=True)

    if args.include_videos:
        archive_dir = root / "archives"
        archive_dir.mkdir(parents=True, exist_ok=True)
        extract_dir = root / "extracted"
        workers = max(1, min(args.workers, len(VIDEO_ZIPS)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    stage_video_zip,
                    repo_id=args.repo_id,
                    filename=filename,
                    archive_dir=archive_dir,
                    extract_dir=extract_dir,
                    force_extract=args.force_extract,
                    cleanup_archives=args.cleanup_archives,
                )
                for filename in VIDEO_ZIPS
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                manifest["zips"].append(str(result["zip"]))
                manifest["extracted_files"] += int(result["files"])
                print(
                    f"{result['status']}: {Path(str(result['zip'])).name} files={result['files']}",
                    flush=True,
                )
                (root / "stage_manifest.partial.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )

    video_index = build_video_index(root)
    manifest["video_index_size"] = len(video_index)
    (root / "video_index.json").write_text(json.dumps(video_index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (root / "stage_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote manifest: {root / 'stage_manifest.json'}", flush=True)
    print(f"video_index_size={len(video_index)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
