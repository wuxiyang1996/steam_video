#!/usr/bin/env python3
"""Build split-aware manifests for video-only expert-demo gathering."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ..adapters import get_adapter
from ..cluster_paths import DEFAULT_DATASET_ROOT
from ..dataset_graph_presets import regime_for_dataset, task_family_for
from ..schemas import BenchmarkProfile, DatasetName


DATASETS: tuple[DatasetName, ...] = ("video_holmes", "videomme", "ovo_bench", "cg_bench", "vrbench")
GOLD_KEYS = {
    "answer",
    "gold",
    "gold_answer",
    "gold_label",
    "gold_eval_only",
    "correct",
    "correct_answer",
    "correct_eval_only",
    "official_answer",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _drop_gold_keys(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: _drop_gold_keys(value) for key, value in payload.items() if str(key) not in GOLD_KEYS}
    if isinstance(payload, list):
        return [_drop_gold_keys(item) for item in payload]
    return payload


def _stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_groups(group_keys: list[str], *, train_ratio: float, dev_ratio: float, seed: str) -> dict[str, str]:
    if train_ratio <= 0 or dev_ratio < 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("ratios must satisfy train_ratio > 0, dev_ratio >= 0, train_ratio + dev_ratio < 1")
    ranked = sorted(group_keys, key=lambda key: _stable_hash(f"{seed}:{key}"))
    total = len(ranked)
    train_n = int(round(total * train_ratio))
    dev_n = int(round(total * dev_ratio))
    if total and train_n <= 0:
        train_n = 1
    if train_n + dev_n >= total and total > 1:
        dev_n = max(0, total - train_n - 1)
    out: dict[str, str] = {}
    for index, key in enumerate(ranked):
        if index < train_n:
            split = "train"
        elif index < train_n + dev_n:
            split = "dev"
        else:
            split = "test"
        out[key] = split
    return out


def _manifest_row(
    item: Any,
    *,
    split: str,
    source_split: str,
    benchmark_profile: BenchmarkProfile,
    seed: str,
) -> dict[str, Any]:
    regime = regime_for_dataset(item.dataset, benchmark_profile)
    question = _drop_gold_keys(item.question or {})
    question_id = str(question.get("question_id") or question.get("id") or item.example_id)
    group_key = f"{item.dataset}:{item.video_id}"
    return {
        "schema_version": "video-skills-relaunch/training-manifest-v0.1",
        "split": split,
        "source_split": source_split,
        "dataset": item.dataset,
        "example_id": item.example_id,
        "video_id": item.video_id,
        "question_id": question_id,
        "task_family": task_family_for(
            item.dataset,
            regime=regime,
            profile=benchmark_profile,
            adapter_task_family=item.task_family,
        ),
        "video_regime": regime.value,
        "benchmark_profile": benchmark_profile.value,
        "video": {
            "path": str(item.video_path) if item.video_path else "",
            "duration_s": item.duration_s,
        },
        "question": question,
        "split_group_key": group_key,
        "split_group_hash": _stable_hash(f"{seed}:{group_key}")[:16],
        "hidden_supervision": {
            "available_for_training": split == "train",
            "available_for_inference": False,
            "sources": list(item.hidden_supervision_sources or []),
        },
        "visible_runtime_mode": "video_only",
        "raw_source_refs": item.raw_source_refs or [],
    }


def build_manifests(
    *,
    dataset_root: Path,
    datasets: list[DatasetName],
    source_split: str,
    benchmark_profile: BenchmarkProfile,
    max_per_dataset: int | None,
    train_ratio: float,
    dev_ratio: float,
    seed: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        adapter = get_adapter(dataset, dataset_root, split=source_split)
        items = list(adapter.iter_items(limit=max_per_dataset))
        group_keys = sorted({f"{item.dataset}:{item.video_id}" for item in items})
        split_by_group = _split_groups(group_keys, train_ratio=train_ratio, dev_ratio=dev_ratio, seed=seed)
        for item in items:
            group_key = f"{item.dataset}:{item.video_id}"
            rows.append(
                _manifest_row(
                    item,
                    split=split_by_group[group_key],
                    source_split=source_split,
                    benchmark_profile=benchmark_profile,
                    seed=seed,
                )
            )
    manifests = {"train": [], "dev": [], "test": []}
    for row in rows:
        manifests[str(row["split"])].append(row)
    summary = {
        "schema_version": "video-skills-relaunch/training-manifest-summary-v0.1",
        "datasets": datasets,
        "source_split": source_split,
        "benchmark_profile": benchmark_profile.value,
        "seed": seed,
        "train_ratio": train_ratio,
        "dev_ratio": dev_ratio,
        "test_ratio": round(1.0 - train_ratio - dev_ratio, 6),
        "max_per_dataset": max_per_dataset,
        "counts": {split: len(rows) for split, rows in manifests.items()},
        "dataset_counts": {
            split: {
                dataset: sum(1 for row in split_rows if row.get("dataset") == dataset)
                for dataset in datasets
            }
            for split, split_rows in manifests.items()
        },
        "group_leakage_count": _group_leakage_count(manifests),
        "gold_fields_removed_from_questions": sorted(GOLD_KEYS),
    }
    return manifests, summary


def _group_leakage_count(manifests: dict[str, list[dict[str, Any]]]) -> int:
    seen: dict[str, str] = {}
    leaks = 0
    for split, rows in manifests.items():
        for row in rows:
            key = str(row.get("split_group_key") or "")
            if not key:
                continue
            prior = seen.setdefault(key, split)
            if prior != split:
                leaks += 1
    return leaks


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train/dev/test manifests for video-only expert-demo gathering.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--benchmark-profile", default="default", choices=[item.value for item in BenchmarkProfile])
    parser.add_argument("--max-per-dataset", type=int)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    parser.add_argument("--seed", default="video-skills-v0")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    manifests, summary = build_manifests(
        dataset_root=args.dataset_root,
        datasets=list(args.datasets),
        source_split=args.source_split,
        benchmark_profile=BenchmarkProfile(args.benchmark_profile),
        max_per_dataset=args.max_per_dataset,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )
    for split, rows in manifests.items():
        _write_jsonl(args.output_dir / f"video_only_{split}_manifest.jsonl", rows)
    _write_json(args.output_dir / "video_only_manifest_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
