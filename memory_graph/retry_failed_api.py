"""Retry OpenRouter label/audit for videos that already have embeddings."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from .graph_builder import build_memory_graph
from .openrouter_validation import GPTOSSGraphValidator
from .validate_video_holmes import _load_annotation, _load_qa_rows
from .validate_video_holmes_parallel import evaluate_summaries


def build_parser() -> argparse.ArgumentParser:
    workspace = Path("/fs/gamma-projects/vlm-robot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).parent / "outputs" / "video_holmes_50"),
    )
    parser.add_argument("--dataset-root", default=str(workspace / "datasets"))
    parser.add_argument("--video-skills-root", default=str(workspace / "Video_Skills"))
    parser.add_argument("--keys-py", default=str(workspace / "keys.py"))
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--api-workers", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout-s", type=int, default=180)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    raise SystemExit(
        "The legacy segment-level API retry path is disabled because it bypasses "
        "atomic extraction, L1 gating, and hard verifiers. Re-run "
        "memory_graph.validate_video_holmes_parallel; it now delegates to the "
        "unified trusted pipeline."
    )

    # Historical implementation below is intentionally unreachable.
    started = time.time()
    output_dir = Path(args.output_dir)
    ids_path = output_dir / "video_ids.json"
    video_ids = tuple(json.loads(ids_path.read_text(encoding="utf-8"))["video_ids"])
    missing = [
        video_id
        for video_id in video_ids
        if (output_dir / video_id / "node_embeddings.npy").exists()
        and not (output_dir / video_id / "audit.json").exists()
    ]
    print(f"[retry] {len(missing)} videos need API retry", flush=True)
    if not missing:
        return _rewrite_summary(output_dir, elapsed_s=0.0)

    jobs = [
        {
            "video_id": video_id,
            "sample_dir": str(output_dir / video_id),
            "dataset_root": args.dataset_root,
            "keys_py": args.keys_py,
            "video_skills_root": args.video_skills_root,
            "top_k": args.top_k,
            "timeout_s": args.timeout_s,
            "max_retries": args.max_retries,
        }
        for video_id in missing
    ]

    errors: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.api_workers) as pool:
        futures = {pool.submit(_retry_one, job): job["video_id"] for job in jobs}
        for future in as_completed(futures):
            video_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "video_id": video_id,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            if result.get("error"):
                errors.append(result)
                print(f"[retry] FAIL {video_id}: {result['error']}", flush=True)
            else:
                computed = result.get("computed_audit_summary") or {}
                print(
                    f"[retry] ok {video_id} "
                    f"strict={computed.get('strict_precision')} "
                    f"supported={computed.get('supported')}/"
                    f"{computed.get('audited_predictions')}",
                    flush=True,
                )

    return _rewrite_summary(
        output_dir,
        elapsed_s=round(time.time() - started, 2),
        extra_errors=errors,
    )


def _retry_one(job: dict[str, Any]) -> dict[str, Any]:
    video_id = job["video_id"]
    sample_dir = Path(job["sample_dir"])
    last_error = ""
    for attempt in range(1, int(job["max_retries"]) + 2):
        try:
            return _label_and_audit_from_disk(
                video_id=video_id,
                sample_dir=sample_dir,
                dataset_root=Path(job["dataset_root"]),
                keys_py=job["keys_py"],
                video_skills_root=job["video_skills_root"],
                top_k=int(job["top_k"]),
                timeout_s=int(job["timeout_s"]),
            )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= int(job["max_retries"]):
                time.sleep(2 * attempt)
                continue
            return {
                "video_id": video_id,
                "error": last_error,
                "traceback": traceback.format_exc(),
                "attempts": attempt,
            }
    return {"video_id": video_id, "error": last_error}


def _label_and_audit_from_disk(
    *,
    video_id: str,
    sample_dir: Path,
    dataset_root: Path,
    keys_py: str,
    video_skills_root: str,
    top_k: int,
    timeout_s: int,
) -> dict[str, Any]:
    canonical = json.loads((sample_dir / "canonical_example.json").read_text(encoding="utf-8"))
    from .adapter import canonical_to_memory_nodes

    source_graph, all_nodes = canonical_to_memory_nodes(canonical)
    nodes = [
        node
        for node in all_nodes
        if node.metadata.get("source_type") == "segment_description"
    ]
    embeddings = np.load(sample_dir / "node_embeddings.npy")
    if len(nodes) != len(embeddings):
        raise ValueError(
            f"{video_id}: node/embedding count mismatch {len(nodes)} vs {len(embeddings)}"
        )

    graph = build_memory_graph(
        graph_id=f"memory_graph:{canonical.get('example_id')}",
        example_id=str(canonical.get("example_id") or video_id),
        video_id=video_id,
        nodes=nodes,
        top_k_candidates=top_k,
    )
    validator = GPTOSSGraphValidator(
        keys_py_path=keys_py,
        video_skills_root=video_skills_root,
        timeout_s=timeout_s,
    )
    teacher_relations = validator.label_relations(nodes, embeddings.tolist(), top_k=top_k)
    graph.relations.extend(teacher_relations)
    graph.metadata.update(
        {
            "source_l1_graph_id": source_graph.get("graph_id"),
            "validation_split": "train",
            "relation_teacher": validator.model,
            "probabilistic_relation_status": "uncalibrated_prior",
            "relation_teacher_input": "segment_descriptions_only",
            "retry_from_disk": True,
        }
    )
    graph_path = sample_dir / "memory_graph.json"
    graph_path.write_text(json.dumps(graph.to_dict(), indent=2) + "\n", encoding="utf-8")

    qa_rows = _load_qa_rows(dataset_root, (video_id,))
    annotation = _load_annotation(dataset_root, video_id)
    audit = validator.audit_graph(
        graph,
        held_out_annotation=annotation,
        held_out_questions=qa_rows,
    )
    audit_path = sample_dir / "audit.json"
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    return {
        "video_id": video_id,
        "example_id": graph.example_id,
        "node_count": len(nodes),
        "deterministic_relation_count": len(graph.relations) - len(teacher_relations),
        "candidate_relation_count": len(teacher_relations),
        "graph_path": str(graph_path),
        "audit_path": str(audit_path),
        "audit_summary": audit.get("summary"),
        "computed_audit_summary": audit.get("computed_summary"),
        "temporal_consistency": audit.get("temporal_consistency"),
    }


def _rewrite_summary(
    output_dir: Path,
    *,
    elapsed_s: float,
    extra_errors: list[dict[str, Any]] | None = None,
) -> int:
    ids = json.loads((output_dir / "video_ids.json").read_text(encoding="utf-8"))["video_ids"]
    samples: list[dict[str, Any]] = []
    errors = list(extra_errors or [])
    for video_id in ids:
        audit_path = output_dir / video_id / "audit.json"
        graph_path = output_dir / video_id / "memory_graph.json"
        if not audit_path.exists() or not graph_path.exists():
            if not any(row.get("video_id") == video_id for row in errors):
                errors.append({"video_id": video_id, "error": "missing audit.json"})
            continue
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        deterministic = sum(
            1
            for relation in graph.get("relations") or []
            if relation.get("status") == "deterministic"
        )
        samples.append(
            {
                "video_id": video_id,
                "example_id": graph.get("example_id"),
                "node_count": len(graph.get("nodes") or []),
                "deterministic_relation_count": deterministic,
                "candidate_relation_count": len(graph.get("relations") or []) - deterministic,
                "graph_path": str(graph_path),
                "audit_path": str(audit_path),
                "audit_summary": audit.get("summary"),
                "computed_audit_summary": audit.get("computed_summary"),
                "temporal_consistency": audit.get("temporal_consistency"),
            }
        )

    evaluation = evaluate_summaries(samples)
    summary = {
        "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
        "embedding_dimension": 2048,
        "relation_teacher": "openai/gpt-oss-120b",
        "video_count_requested": len(ids),
        "video_count_succeeded": len(samples),
        "video_count_failed": len(errors),
        "retry_elapsed_s": elapsed_s,
        "evaluation": evaluation,
        "samples": samples,
        "errors": errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output_dir / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"evaluation": evaluation, "failed": len(errors)}, indent=2), flush=True)
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
