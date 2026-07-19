"""Calibrate relation-specific decision thresholds from graph audit labels."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


MIN_SAMPLES_PER_CLASS = 10
ECE_BINS = 10
POSITIVE_JUDGMENTS = {"supported"}
NEGATIVE_JUDGMENTS = {"unsupported", "contradicted"}
IGNORED_JUDGMENTS = {"plausible"}
TRUSTED_LABELS_SOURCE = "independent_human"


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON object from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _probabilities_by_key(graph: dict[str, Any], path: Path) -> dict[tuple[str, str, str], float]:
    result: dict[tuple[str, str, str], float] = {}
    relations = graph.get("relations")
    if not isinstance(relations, list):
        raise ValueError(f"{path}: 'relations' must be an array")

    for edge in relations:
        if not isinstance(edge, dict):
            continue
        src, dst = edge.get("src"), edge.get("dst")
        probabilities = edge.get("relation_probabilities")
        if not isinstance(src, str) or not isinstance(dst, str) or not isinstance(probabilities, dict):
            continue
        for relation, raw_probability in probabilities.items():
            if not isinstance(relation, str) or isinstance(raw_probability, bool):
                continue
            try:
                probability = float(raw_probability)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"{path}: probability for {(src, dst, relation)!r} must be finite and in [0, 1]"
                )
            key = (src, dst, relation)
            if key in result and not math.isclose(result[key], probability, abs_tol=1e-12):
                raise ValueError(f"{path}: conflicting probabilities for {key!r}")
            result[key] = probability
    return result


def collect_samples(
    audit_dir: Path,
    *,
    audit_filename: str = "audit.json",
) -> tuple[dict[str, list[tuple[float, int]]], dict[str, dict[str, int]]]:
    """Align audit judgments with sibling ``memory_graph.json`` probabilities."""
    samples: dict[str, list[tuple[float, int]]] = defaultdict(list)
    counts: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total_judgments": 0,
            "positive": 0,
            "negative": 0,
            "plausible_ignored": 0,
            "unknown_judgment_ignored": 0,
            "missing_probability": 0,
        }
    )
    audit_paths = sorted(audit_dir.rglob(audit_filename))
    if not audit_paths:
        raise ValueError(f"no {audit_filename} files found under {audit_dir}")

    for audit_path in audit_paths:
        graph_candidates = (
            audit_path.with_name("causal_temporal_overlay.json"),
            audit_path.with_name("memory_graph.json"),
        )
        graph_path = next((path for path in graph_candidates if path.is_file()), None)
        if graph_path is None:
            raise ValueError(
                f"missing graph for {audit_path}: expected causal_temporal_overlay.json "
                "or memory_graph.json"
            )
        audit = _load_object(audit_path)
        probabilities = _probabilities_by_key(_load_object(graph_path), graph_path)
        judgments = audit.get("candidate_edge_audit")
        summary = audit.get("computed_summary")
        incomplete = isinstance(summary, dict) and summary.get("audit_complete") is False
        if judgments is None and ("error" in audit or incomplete):
            continue
        if not isinstance(judgments, list):
            raise ValueError(f"{audit_path}: 'candidate_edge_audit' must be an array")

        for item in judgments:
            if not isinstance(item, dict):
                continue
            src, dst, relation, raw_judgment = (
                item.get("src"),
                item.get("dst"),
                item.get("relation"),
                item.get("judgment"),
            )
            if not all(isinstance(value, str) for value in (src, dst, relation, raw_judgment)):
                continue
            judgment = raw_judgment.strip().lower()
            relation_counts = counts[relation]
            relation_counts["total_judgments"] += 1
            if judgment in IGNORED_JUDGMENTS:
                relation_counts["plausible_ignored"] += 1
                continue
            if judgment not in POSITIVE_JUDGMENTS | NEGATIVE_JUDGMENTS:
                relation_counts["unknown_judgment_ignored"] += 1
                continue
            probability = probabilities.get((src, dst, relation))
            if probability is None and item.get("probability") is not None:
                try:
                    probability = float(item["probability"])
                except (TypeError, ValueError):
                    probability = None
            if probability is None:
                relation_counts["missing_probability"] += 1
                continue
            label = 1 if judgment in POSITIVE_JUDGMENTS else 0
            samples[relation].append((probability, label))
            relation_counts["positive" if label else "negative"] += 1

    return dict(samples), dict(counts)


def _classification_metrics(
    samples: Iterable[tuple[float, int]], threshold: float
) -> tuple[float, float, float]:
    true_positive = false_positive = false_negative = 0
    for probability, label in samples:
        predicted = probability >= threshold
        true_positive += int(predicted and label == 1)
        false_positive += int(predicted and label == 0)
        false_negative += int(not predicted and label == 1)
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def _best_threshold(samples: list[tuple[float, int]]) -> tuple[float, float, float, float]:
    candidates = sorted({0.0, 1.0, *(probability for probability, _ in samples)})
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        precision, recall, f1 = _classification_metrics(samples, threshold)
        candidate = (threshold, precision, recall, f1)
        if best is None or (f1, precision, threshold) > (best[3], best[1], best[0]):
            best = candidate
    assert best is not None
    return best


def _brier_score(samples: list[tuple[float, int]]) -> float:
    return sum((probability - label) ** 2 for probability, label in samples) / len(samples)


def _expected_calibration_error(samples: list[tuple[float, int]], bins: int = ECE_BINS) -> float:
    bucketed: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in samples:
        index = min(int(probability * bins), bins - 1)
        bucketed[index].append((probability, label))
    total = len(samples)
    error = 0.0
    for bucket in bucketed:
        if not bucket:
            continue
        mean_probability = sum(probability for probability, _ in bucket) / len(bucket)
        positive_rate = sum(label for _, label in bucket) / len(bucket)
        error += len(bucket) / total * abs(mean_probability - positive_rate)
    return error


def calibrate(
    audit_dir: Path,
    labels_source: str,
    *,
    min_samples_per_class: int = MIN_SAMPLES_PER_CLASS,
) -> dict[str, Any]:
    """Return relation-wise threshold and probability calibration metrics."""
    if min_samples_per_class <= 0:
        raise ValueError("min_samples_per_class must be positive")
    samples_by_relation, counts_by_relation = collect_samples(
        audit_dir,
        audit_filename=(
            "independent_audit.json"
            if labels_source == TRUSTED_LABELS_SOURCE
            else "audit.json"
        ),
    )
    trusted_labels = labels_source == TRUSTED_LABELS_SOURCE
    relation_names = sorted(set(samples_by_relation) | set(counts_by_relation))
    relations: dict[str, dict[str, Any]] = {}

    for relation in relation_names:
        samples = samples_by_relation.get(relation, [])
        counts = counts_by_relation[relation]
        sufficient = (
            counts["positive"] >= min_samples_per_class
            and counts["negative"] >= min_samples_per_class
        )
        relation_calibrated = trusted_labels and sufficient
        result: dict[str, Any] = {
            "threshold": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "brier": None,
            "ece": None,
            "ece_bins": ECE_BINS,
            "sample_counts": {
                **counts,
                "used": len(samples),
            },
            "trusted_labels": trusted_labels,
            "sufficient_class_samples": sufficient,
            "calibrated": relation_calibrated,
        }
        if samples:
            threshold, precision, recall, f1 = _best_threshold(samples)
            result.update(
                threshold=threshold,
                precision=precision,
                recall=recall,
                f1=f1,
                brier=_brier_score(samples),
                ece=_expected_calibration_error(samples),
            )
        relations[relation] = result

    return {
        "labels_source": labels_source,
        "trusted_labels": trusted_labels,
        "calibrated": bool(relations) and all(item["calibrated"] for item in relations.values()),
        "min_samples_per_class": min_samples_per_class,
        "relations": relations,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--labels-source", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.audit_dir.is_dir():
        raise SystemExit(f"--audit-dir is not a directory: {args.audit_dir}")
    try:
        report = calibrate(args.audit_dir, args.labels_source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
