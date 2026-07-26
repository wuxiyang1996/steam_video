#!/usr/bin/env python3
"""Gate L1-only query-memory outputs before spending GPT-OSS L2 calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from ..verification.evaluate_l1_query_memory import evaluate_example
except ImportError:  # pragma: no cover - direct script execution
    from evaluate_l1_query_memory import evaluate_example


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def gate_report(
    example: dict[str, Any],
    *,
    topk: int,
    min_graph_nodes: int,
    min_memory_items: int,
    min_question_hits: int,
    min_top_score: float,
    min_option_margin: float,
) -> dict[str, Any]:
    report = evaluate_example(example, topk=topk)
    top_hits = report.get("top_question_hits") or []
    option_scores = report.get("option_scores") or []
    top_score = float(top_hits[0]["score"]) if top_hits else 0.0
    option_top = float(option_scores[0]["score"]) if option_scores else 0.0
    option_second = float(option_scores[1]["score"]) if len(option_scores) > 1 else 0.0
    option_margin = round(option_top - option_second, 4)

    checks = {
        "graph_nodes": report["graph_nodes"] >= min_graph_nodes,
        "memory_items": report["memory_items"] >= min_memory_items,
        "question_hits": len(top_hits) >= min_question_hits,
        "top_question_score": top_score >= min_top_score,
        "no_hidden_memory_items": report["hidden_memory_items"] == 0,
    }
    qa_answerability = report.get("qa_answerability") or {}
    checks["no_uniform_probe_fallback"] = qa_answerability.get("retrieval_fallback_reason") != "uniform_probe_no_lexical_match"
    missing_requirements = qa_answerability.get("missing_requirements") or []
    allowed_repair_l2 = qa_answerability.get("allowed_repair_l2") or []
    l2_route = "answer_then_verify_or_repair"
    if missing_requirements and allowed_repair_l2:
        l2_route = "repair_only"
    elif missing_requirements:
        l2_route = "abstain_only"
    checks["no_missing_required_modalities"] = not missing_requirements
    checks["repair_route_allowed"] = l2_route == "answer_then_verify_or_repair" or bool(allowed_repair_l2)
    checks["option_refs_discriminative"] = (
        float(qa_answerability.get("top2_shared_ref_ratio") or 0.0) < 0.75
        or option_margin >= 0.75
    )
    if top_hits:
        fine_or_l1_hits = [
            hit
            for hit in top_hits
            if hit.get("source") in {"clue_memory_graph", "evidence_index", "coarse_fine_graph.fine"}
            and hit.get("node_type") != "coarse_summary"
        ]
        checks["has_fine_or_l1_evidence"] = bool(fine_or_l1_hits)
    if option_scores:
        checks["option_margin"] = option_margin >= min_option_margin

    answer_then_repair_pass = all(value for key, value in checks.items() if key != "repair_route_allowed")
    repair_pass = (
        l2_route == "repair_only"
        and checks["graph_nodes"]
        and checks["memory_items"]
        and checks["no_hidden_memory_items"]
        and checks["repair_route_allowed"]
    )
    passed = answer_then_repair_pass or repair_pass
    report.update(
        {
            "gate_pass": passed,
            "l2_route": l2_route,
            "answer_then_repair_gate_pass": answer_then_repair_pass,
            "repair_gate_pass": repair_pass,
            "gate_checks": checks,
            "gate_scores": {
                "top_question_score": round(top_score, 4),
                "option_top_score": round(option_top, 4),
                "option_second_score": round(option_second, 4),
                "option_margin": option_margin,
            },
            "gate_thresholds": {
                "min_graph_nodes": min_graph_nodes,
                "min_memory_items": min_memory_items,
                "min_question_hits": min_question_hits,
                "min_top_score": min_top_score,
                "min_option_margin": min_option_margin,
            },
        }
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Select examples whose L1 query memory is worth L2 planning.")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--min-graph-nodes", type=int, default=20)
    parser.add_argument("--min-memory-items", type=int, default=20)
    parser.add_argument("--min-question-hits", type=int, default=1)
    parser.add_argument("--min-top-score", type=float, default=0.2)
    parser.add_argument("--min-option-margin", type=float, default=0.2)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--passed-output", type=Path)
    args = parser.parse_args()

    reports: list[dict[str, Any]] = []
    for path in args.paths:
        for example in _read_jsonl(path):
            report = gate_report(
                example,
                topk=args.topk,
                min_graph_nodes=args.min_graph_nodes,
                min_memory_items=args.min_memory_items,
                min_question_hits=args.min_question_hits,
                min_top_score=args.min_top_score,
                min_option_margin=args.min_option_margin,
            )
            report["source_path"] = str(path)
            reports.append(report)

    passed = [row for row in reports if row["gate_pass"]]
    payload = {
        "summary": {
            "examples": len(reports),
            "passed": len(passed),
            "pass_rate": round(len(passed) / len(reports), 4) if reports else 0.0,
            "datasets": sorted({str(row.get("dataset")) for row in reports}),
        },
        "passed_example_ids": [str(row.get("example_id")) for row in passed],
        "reports": reports,
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if args.passed_output:
        args.passed_output.parent.mkdir(parents=True, exist_ok=True)
        args.passed_output.write_text(
            "\n".join(str(row.get("example_id")) for row in passed) + ("\n" if passed else ""),
            encoding="utf-8",
        )
    print(text)


if __name__ == "__main__":
    main()
