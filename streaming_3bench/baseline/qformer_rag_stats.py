#!/usr/bin/env python3
"""Merge Q-Former RAG shards and compute paired uncertainty/significance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest


ARMS = ("uniform", "visual", "qformer")
DATASETS = ("ovo_bench", "videomme", "streaming_bench")


def load_arm(root: Path, arm: str) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(root.expanduser().resolve().rglob("records.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if row.get("arm") != arm:
                    continue
                key = (str(row["dataset"]), str(row["example_id"]))
                if key in records:
                    raise ValueError(f"duplicate {arm} result: {key}")
                records[key] = row
    if not records:
        raise ValueError(f"no {arm} records below {root}")
    return records


def paired_report(
    left: list[bool], right: list[bool], *, seed: int, samples: int
) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    delta = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(samples, len(delta)))
    bootstrap = delta[indices].mean(axis=1)
    discordant_left = int(((a == 1) & (b == 0)).sum())
    discordant_right = int(((a == 0) & (b == 1)).sum())
    discordant = discordant_left + discordant_right
    pvalue = (
        float(binomtest(discordant_left, discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "n": len(delta),
        "left_accuracy": float(a.mean()),
        "right_accuracy": float(b.mean()),
        "delta": float(delta.mean()),
        "bootstrap_95_ci": [
            float(np.quantile(bootstrap, 0.025)),
            float(np.quantile(bootstrap, 0.975)),
        ],
        "mcnemar_exact_p": pvalue,
        "left_only_correct": discordant_left,
        "right_only_correct": discordant_right,
    }


def holm_adjust(values: list[tuple[str, float]]) -> dict[str, float]:
    ordered = sorted(values, key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[name] = running
    return adjusted


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        parser.add_argument(f"--{arm}-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args(argv)
    arms = {
        arm: load_arm(getattr(args, f"{arm}_root"), arm)
        for arm in ARMS
    }
    common = set.intersection(*(set(rows) for rows in arms.values()))
    union = set.union(*(set(rows) for rows in arms.values()))
    if common != union:
        counts = {arm: len(rows) for arm, rows in arms.items()}
        raise ValueError(f"arms do not contain identical paired examples: {counts}")

    report: dict[str, Any] = {
        "schema_version": "streaming-3bench-qformer-paired-stats/v0.1",
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "benchmarks": {},
    }
    tests: list[tuple[str, float]] = []
    for dataset in DATASETS:
        keys = sorted(key for key in common if key[0] == dataset)
        if dataset == "streaming_bench":
            keys = [
                key
                for key in keys
                if all(rows[key].get("official_protocol_compatible") is True for rows in arms.values())
            ]
        benchmark: dict[str, Any] = {
            "n": len(keys),
            "failure_rate": {
                arm: sum(not rows[key].get("ok", False) for key in keys) / max(len(keys), 1)
                for arm, rows in arms.items()
            },
            "accuracy": {
                arm: sum(bool(rows[key].get("correct", False)) for key in keys) / max(len(keys), 1)
                for arm, rows in arms.items()
            },
            "comparisons": {},
        }
        for left, right in (("qformer", "visual"), ("visual", "uniform")):
            name = f"{dataset}:{left}-{right}"
            comparison = paired_report(
                [bool(arms[left][key].get("correct", False)) for key in keys],
                [bool(arms[right][key].get("correct", False)) for key in keys],
                seed=args.seed + len(tests),
                samples=args.bootstrap_samples,
            )
            benchmark["comparisons"][f"{left}_minus_{right}"] = comparison
            tests.append((name, comparison["mcnemar_exact_p"]))
        report["benchmarks"][dataset] = benchmark
    adjusted = holm_adjust(tests)
    for dataset, benchmark in report["benchmarks"].items():
        for key, comparison in benchmark["comparisons"].items():
            left, right = key.split("_minus_")
            comparison["holm_adjusted_p_across_six_tests"] = adjusted[
                f"{dataset}:{left}-{right}"
            ]
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
