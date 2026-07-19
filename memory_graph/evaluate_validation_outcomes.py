"""Evaluate pre/post-verifier relation quality from independent edge audits."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--labels-source",
        choices=["independent_human", "gpt_self_audit"],
        required=True,
    )
    return parser


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _accepted_probabilities(
    overlay: dict[str, Any],
) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    for edge in overlay.get("relations") or []:
        if not isinstance(edge, dict) or edge.get("status") == "deterministic":
            continue
        for relation, probability in (edge.get("relation_probabilities") or {}).items():
            values[(str(edge.get("src")), str(edge.get("dst")), str(relation))] = float(
                probability
            )
    return values


def _rejected_probabilities(
    overlay: dict[str, Any],
) -> dict[tuple[str, str, str], float]:
    values: dict[tuple[str, str, str], float] = {}
    report = overlay.get("build_report") or {}
    for row in report.get("rejected_relations") or []:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("src")), str(row.get("dst")), str(row.get("relation")))
        values[key] = float(row.get("probability") or 0.0)
    return values


def _audit_rows(audit: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for row in audit.get("candidate_edge_audit") or []:
        if isinstance(row, dict):
            yield row


def _classification(rows: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(predicted and label for predicted, label in rows)
    fp = sum(predicted and not label for predicted, label in rows)
    fn = sum(not predicted and label for predicted, label in rows)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def _brier(rows: list[tuple[float, int]]) -> float | None:
    return (
        sum((probability - label) ** 2 for probability, label in rows) / len(rows)
        if rows
        else None
    )


def _ece(rows: list[tuple[float, int]], bins: int = 10) -> float | None:
    if not rows:
        return None
    groups: list[list[tuple[float, int]]] = [[] for _ in range(bins)]
    for probability, label in rows:
        groups[min(int(probability * bins), bins - 1)].append((probability, label))
    return sum(
        len(group)
        / len(rows)
        * abs(
            sum(probability for probability, _ in group) / len(group)
            - sum(label for _, label in group) / len(group)
        )
        for group in groups
        if group
    )


def evaluate_summaries(
    samples: list[dict[str, Any]],
    *,
    labels_source: str,
) -> dict[str, Any]:
    per_relation_pre: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    per_relation_post: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    per_relation_prob: dict[str, list[tuple[float, int]]] = defaultdict(list)
    proposed = accepted = 0
    l1_statuses: dict[str, int] = defaultdict(int)
    events = l1_nodes = complete_audits = 0
    evaluated = 0

    for sample in samples:
        graph_path = Path(str(sample.get("graph_path") or ""))
        audit_path = Path(str(sample.get("audit_path") or ""))
        if not graph_path.is_file() or not audit_path.is_file():
            continue
        overlay = _load(graph_path)
        audit = _load(audit_path)
        post = _accepted_probabilities(overlay)
        rejected = _rejected_probabilities(overlay)
        pre = {**rejected, **post}
        proposed += len(pre)
        accepted += len(post)
        metadata = overlay.get("metadata") or {}
        l1_report = metadata.get("l1_reliability") or sample.get("l1_reliability") or {}
        l1_statuses[str(l1_report.get("status") or "missing")] += 1
        l1_nodes += len(overlay.get("l1_observations") or [])
        events += len(overlay.get("atomic_events") or [])
        summary = audit.get("computed_summary") or {}
        complete_audits += int(summary.get("audit_complete") is not False)
        evaluated += 1

        for row in _audit_rows(audit):
            judgment = str(row.get("judgment") or "").lower()
            if judgment == "plausible" or judgment not in {
                "supported",
                "unsupported",
                "contradicted",
            }:
                continue
            relation = str(row.get("relation"))
            key = (str(row.get("src")), str(row.get("dst")), relation)
            label = judgment == "supported"
            per_relation_pre[relation].append((key in pre, label))
            per_relation_post[relation].append((key in post, label))
            if key in pre:
                per_relation_prob[relation].append((pre[key], int(label)))

    relations: dict[str, Any] = {}
    for relation in sorted(set(per_relation_pre) | set(per_relation_post)):
        pre_metrics = _classification(per_relation_pre[relation])
        post_metrics = _classification(per_relation_post[relation])
        probabilities = per_relation_prob[relation]
        relations[relation] = {
            "pre_verifier": pre_metrics,
            "post_verifier": post_metrics,
            "supported_edges_filtered_by_verifier": post_metrics["fn"],
            "unsupported_edges_filtered_by_verifier": max(
                0,
                pre_metrics["fp"] - post_metrics["fp"],
            ),
            "brier_pre_verifier": _brier(probabilities),
            "ece_pre_verifier": _ece(probabilities),
            "audited_label_count": len(per_relation_pre[relation]),
        }

    strict_precisions = [
        metrics["post_verifier"]["precision"]
        for metrics in relations.values()
        if metrics["post_verifier"]["precision"] is not None
    ]
    return {
        "labels_source": labels_source,
        "trusted_evaluation": labels_source == "independent_human",
        "videos_evaluated": evaluated,
        "complete_audits": complete_audits,
        "l1_gate_status_counts": dict(l1_statuses),
        "atomic_events_per_l1_node": events / l1_nodes if l1_nodes else None,
        "candidate_relation_count_pre_verifier": proposed,
        "candidate_relation_count_post_verifier": accepted,
        "verifier_filter_rate": 1.0 - accepted / proposed if proposed else None,
        "relations": relations,
        "post_verifier_macro_precision": (
            sum(strict_precisions) / len(strict_precisions) if strict_precisions else None
        ),
        "candidate_causal_gate": (
            "pass"
            if labels_source == "independent_human"
            and strict_precisions
            and min(strict_precisions) >= 0.7
            else "fail"
        ),
        "recall_scope": "supported edges among independently audited candidates",
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = _load(args.summary)
    samples = summary.get("samples") or []
    if args.labels_source == "independent_human":
        samples = [
            {
                **sample,
                "audit_path": str(
                    Path(str(sample.get("graph_path") or "")).with_name(
                        "independent_audit.json"
                    )
                ),
            }
            for sample in samples
        ]
    evaluation = evaluate_summaries(samples, labels_source=args.labels_source)
    evaluation["source_summary"] = str(args.summary)
    evaluation["elapsed_s"] = summary.get("elapsed_s")
    evaluation["errors"] = summary.get("errors") or []
    output = args.output or args.summary.with_name("evaluation.json")
    output.write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in evaluation.items() if key != "relations"}, indent=2))
    return 0 if evaluation["candidate_causal_gate"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
