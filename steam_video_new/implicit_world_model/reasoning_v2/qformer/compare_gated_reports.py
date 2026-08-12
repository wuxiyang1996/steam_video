"""Paired case-bootstrap comparison for held-out gated retrieval reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


METRICS = (
    ("mrr", "node_union_metrics", "mrr"),
    ("node_recall@8", "node_union_metrics", "recall@8"),
    ("clue_group_recall@4", "clue_group_metrics", "clue_group_recall@4"),
    ("clue_group_recall@8", "clue_group_metrics", "clue_group_recall@8"),
    (
        "all_clue_group_coverage@8",
        "clue_group_metrics",
        "all_clue_group_coverage@8",
    ),
)


def _case_values(report: Mapping[str, Any], section: str, metric: str) -> dict[str, float]:
    ranking = report["heldout_ranking"]
    return {
        str(row["case_id"]): float(row[section][metric])
        for row in ranking["per_case"]
    }


def paired_case_bootstrap(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    samples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Bootstrap paired question-level deltas; positive means candidate wins."""

    if samples < 100:
        raise ValueError("paired bootstrap requires at least 100 samples")
    rng = np.random.default_rng(seed)
    output: dict[str, Any] = {}
    for name, section, metric in METRICS:
        left = _case_values(baseline, section, metric)
        right = _case_values(candidate, section, metric)
        if set(left) != set(right) or not left:
            raise ValueError("reports must contain identical non-empty held-out case IDs")
        case_ids = sorted(left)
        deltas = np.asarray([right[key] - left[key] for key in case_ids])
        indices = rng.integers(0, len(deltas), size=(samples, len(deltas)))
        means = deltas[indices].mean(axis=1)
        output[name] = {
            "case_count": len(case_ids),
            "mean_delta": float(deltas.mean()),
            "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
            "bootstrap_probability_positive": float(np.mean(means > 0.0)),
            "improved_cases": int((deltas > 0.0).sum()),
            "worsened_cases": int((deltas < 0.0).sum()),
            "tied_cases": int((deltas == 0.0).sum()),
        }
    return output


def compare_reports(
    baseline_path: Path,
    candidate_paths: Sequence[Path],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    comparisons = {}
    for index, path in enumerate(candidate_paths):
        candidate = json.loads(path.read_text(encoding="utf-8"))
        key = (
            f"{candidate['variant']}:seed={candidate['config']['seed']}:"
            f"same_video_negatives={candidate['config'].get('same_video_negatives', 0)}"
        )
        comparisons[key] = paired_case_bootstrap(
            baseline, candidate, samples=samples, seed=seed + index
        )
    return {
        "schema_version": "steam-qformer-paired-bootstrap/v0.1",
        "baseline": str(baseline_path.resolve()),
        "bootstrap_samples": samples,
        "comparisons": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path, action="append")
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = compare_reports(
        args.baseline,
        args.candidate,
        samples=args.samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
